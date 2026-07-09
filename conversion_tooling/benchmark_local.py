#!/usr/bin/env python3
"""Local CPU latency benchmark with ONNX transcribers (FastConformer + Whisper).

Backends:
  fc_ctc   FastConformer CTC    — hybrid artifact dir (super_encoder + ctc_decoder)
  fc_rnnt  FastConformer RNN-T  — hybrid artifact dir (super_encoder + decoder_joint)
  whisper  Whisper              — asset dir (super_encoder + decoder(s) + configs + vocab)

Deps: onnxruntime, librosa, soundfile, numpy - any light ONNX env works.

  # measure one backend
  python benchmark_local.py --backend fc_rnnt --model <fastconformer hybrid_int8 dir> \\
      --audio wavs/ --out fc_rnnt_int8.json
  python benchmark_local.py --backend whisper --model <whisper asset dir> \\
      --audio wavs/ --out whisper_int8.json

  # merge result JSONs into a comparison table
  python benchmark_local.py --compare fc_rnnt_int8.json whisper_int8.json

``--model`` is the artifact DIRECTORY. ``--audio`` is a wav file, a directory of wavs, or a
jsonl manifest (uses each line's ``audio_filepath``).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # conversion_tooling/ (whisper/ + fastconformer/ live here)


# --------------------------------------------------------------------------- io
def gather_audio(spec: str, limit: int | None) -> list[str]:
    p = Path(spec)
    if p.is_dir():
        paths = sorted(str(q) for q in p.rglob("*.wav"))
    elif p.suffix == ".jsonl" or p.name.endswith(".json"):
        paths = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                paths.append(json.loads(line)["audio_filepath"])
    elif p.is_file():
        paths = [str(p)]
    else:
        raise FileNotFoundError(f"--audio {spec!r}: not a wav, dir, or manifest")
    if not paths:
        raise FileNotFoundError(f"--audio {spec!r}: no audio found")
    return paths[:limit] if limit else paths


def audio_duration(path: str) -> float:
    import soundfile as sf
    info = sf.info(path)
    return info.frames / float(info.samplerate)


# --------------------------------------------------------------- backends (lazy)
def make_transcriber(backend: str, model_dir: str, num_threads: int, language: str):
    """Return a ``path -> text`` callable for the chosen NeMo-free ONNX runtime.

    Each runtime lives in its own subfolder (fastconformer/ or whisper/); we add that to
    the path lazily so importing this file never pulls both stacks."""
    p = Path(model_dir)
    if p.is_file() or p.suffix == ".nemo":
        raise SystemExit(f"--backend {backend} expects an ONNX artifact DIRECTORY, got a file: {model_dir}")

    if backend in ("fc_ctc", "fc_rnnt"):
        sys.path.insert(0, str(_HERE / "fastconformer"))
        from onnx_fastconformer_transcriber import OnnxFastConformerCTC, OnnxFastConformerRNNT
        cls = OnnxFastConformerCTC if backend == "fc_ctc" else OnnxFastConformerRNNT
        t = cls(num_threads=num_threads, verbose=True)
        t.load(model_dir)
        return t.transcribe

    if backend == "whisper":
        sys.path.insert(0, str(_HERE / "whisper"))
        from onnx_whisper_transcriber import OnnxWhisperTranscriber
        # NB: OnnxWhisperTranscriber has no num_threads knob — it uses ORT defaults. We set
        # OMP_NUM_THREADS above (best-effort); for a strictly fair FC-vs-Whisper thread
        # comparison, add a num_threads arg to the Whisper transcriber's sessions.
        t = OnnxWhisperTranscriber(language=language, verbose=True)
        t.load(model_dir)
        return t.transcribe

    raise SystemExit(f"unknown --backend {backend!r}")


# ----------------------------------------------------------------------- timing
def benchmark(transcribe, paths: list[str], warmup: bool, repeats: int) -> dict:
    # Warm up on the first file, outside the timer (one-time JIT / graph opt).
    if warmup:
        transcribe(paths[0])

    per_file = []
    for i, path in enumerate(paths):
        dur = audio_duration(path)
        lat = []
        text = ""
        for _ in range(repeats):
            t0 = time.perf_counter()
            text = transcribe(path)
            lat.append(time.perf_counter() - t0)
        med = statistics.median(lat)
        per_file.append({"path": path, "duration_s": dur, "latency_s": med,
                         "rtf": med / dur if dur else None, "text": text})
        rtf = f"{med / dur:.3f}" if dur else "n/a"
        print(f"  [{i+1}/{len(paths)}] {Path(path).name}  {1000*med:.1f} ms  rtf={rtf}",
              flush=True)
        print(f"      {text!r}", flush=True)

    lats = [r["latency_s"] for r in per_file]
    durs = [r["duration_s"] for r in per_file]
    lats_sorted = sorted(lats)
    p90 = lats_sorted[min(len(lats_sorted) - 1, int(0.9 * len(lats_sorted)))]
    return {
        "num_files": len(per_file),
        "repeats": repeats,
        "total_audio_s": sum(durs),
        "total_proc_s": sum(lats),
        "mean_latency_ms": 1000 * statistics.mean(lats),
        "median_latency_ms": 1000 * statistics.median(lats),
        "p90_latency_ms": 1000 * p90,
        "mean_rtf": statistics.mean([r["rtf"] for r in per_file if r["rtf"]]),
        "aggregate_rtf": sum(lats) / sum(durs) if sum(durs) else None,
        "per_file": per_file,
    }


# ---------------------------------------------------------------------- compare
def compare(paths: list[str]) -> None:
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        s = d["summary"]
        label = d.get("variant") or d["backend"]  # variant distinguishes fp32/int8
        rows.append((label, s["median_latency_ms"], s["p90_latency_ms"],
                     s["aggregate_rtf"], s["num_files"], d.get("num_threads")))
    print(f"\n{'variant':22s} {'med ms':>9s} {'p90 ms':>9s} {'agg RTF':>9s} "
          f"{'files':>6s} {'thr':>4s}")
    print("-" * 66)
    for name, med, p90, rtf, n, thr in rows:
        print(f"{name:22s} {med:9.1f} {p90:9.1f} {rtf:9.4f} {n:6d} {str(thr):>4s}")

    # Generic int8-vs-fp32 speedup: for any "<base>_int8" with a matching "<base>" row.
    by = {r[0]: r for r in rows}
    for name, med, *_ in rows:
        if name.endswith("_int8") and name[:-5] in by:
            fp = by[name[:-5]]
            print(f"  {name} vs {name[:-5]}: {med:.1f}ms vs {fp[1]:.1f}ms "
                  f"({fp[1] / med:.2f}x {'faster' if med < fp[1] else 'slower'})")
    print()


# ------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["fc_ctc", "fc_rnnt", "whisper"])
    ap.add_argument("--model", help="ONNX artifact dir (FastConformer hybrid dir, or Whisper asset dir)")
    ap.add_argument("--audio", help="wav file, dir of wavs, or jsonl manifest")
    ap.add_argument("--out", help="write per-file + summary JSON here")
    ap.add_argument("--language", default="en", help="language code (Whisper only; default en)")
    ap.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                    help="untimed warmup run on the first file (default: on; --no-warmup to skip)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="timed runs per file; median is kept (default 1)")
    ap.add_argument("--num_threads", type=int, default=1,
                    help="CPU threads (default 1; applies to FastConformer — Whisper uses ORT defaults)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of files")
    ap.add_argument("--compare", nargs="+", metavar="JSON",
                    help="merge result JSONs into a comparison table and exit")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return 0

    if not (args.backend and args.model and args.audio):
        ap.error("--backend, --model, and --audio are required (unless --compare)")

    # Best-effort thread cap for backends that don't take num_threads (set before ORT import).
    os.environ.setdefault("OMP_NUM_THREADS", str(args.num_threads))

    paths = gather_audio(args.audio, args.limit)
    print(f"[BENCH] backend={args.backend} files={len(paths)} threads={args.num_threads} "
          f"warmup={args.warmup} repeats={args.repeats}")

    transcribe = make_transcriber(args.backend, args.model, args.num_threads, args.language)

    summary = benchmark(transcribe, paths, args.warmup, args.repeats)
    print(f"\n[BENCH] {args.backend}: median {summary['median_latency_ms']:.1f} ms/utt | "
          f"p90 {summary['p90_latency_ms']:.1f} ms | aggregate RTF "
          f"{summary['aggregate_rtf']:.4f} | {summary['num_files']} files")

    if args.out:
        # variant = backend + an _int8 tag when the model path is an int8 artifact, so the
        # --compare table can distinguish precisions (the backend name alone can't).
        variant = args.backend
        if "int8" in args.model.lower():
            variant = f"{args.backend}_int8"
        payload = {"backend": args.backend, "variant": variant, "model": args.model,
                   "num_threads": args.num_threads, "summary": summary}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"[BENCH] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
