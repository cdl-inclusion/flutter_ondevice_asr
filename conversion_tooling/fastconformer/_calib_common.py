"""Shared batch-calibration logic for the RNN-T + CTC streaming window sweeps.

Both reference transcribers expose the same interface — ``transcribe(path)`` (offline) and
``transcribe_streaming(path, chunk_s=, left_context_s=, lookahead_s=)`` — so one generic runner
serves both; the head-specific scripts (calibrate_streaming_{rnnt,ctc}.py) just build the model and
call :func:`run`.

Goal (see the calibration header in lib/models/fastconformer/fastconformer_transcriber.dart): find the
SMALLEST lookahead where STREAMED PARTIALS don't look visibly broken — NOT exact parity. Partials are
throwaway (the committed final is always a full re-decode), so we allow some streamed-vs-offline error
and optimise for snappiness (least lookahead = least latency). The metric is streamed-vs-OFFLINE word
error (the streaming-INDUCED artifact), not accuracy vs ground truth.
"""
from __future__ import annotations

import glob
import os
import re


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s.lower())).strip()


def floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _lev(a: list[str], b: list[str]) -> int:
    """Word-level Levenshtein distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return dp[n]


def wer(ref: str, hyp: str) -> float:
    """Word error rate of hyp vs ref (both raw text; normalised here)."""
    r, h = norm(ref).split(), norm(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    return _lev(r, h) / len(r)


def gather_wavs(audio: str | None, audio_dir: str | None) -> list[str]:
    if audio_dir:
        wavs = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
        if not wavs:
            raise SystemExit(f"no .wav files in {audio_dir}")
        return wavs
    if audio:
        return [audio]
    raise SystemExit("pass --audio <wav> or --audio-dir <dir>")


def run(model, head: str, wavs: list[str], *, chunks, lefts, looks, tol: float,
        default_chunk: float, default_left: float) -> None:
    """Sweep (chunk, left, lookahead); per setting report streamed-vs-offline WER aggregated over all
    clips (mean / max / #clean), then recommend the smallest lookahead whose MAX WER <= ``tol`` at the
    default chunk/left (snappiest that stays 'reasonable')."""
    print(f"== {head} streaming calibration ==  clips={len(wavs)}  tol(maxWER)<={tol}")
    offline = {}
    for p in wavs:
        offline[p] = model.transcribe(p)
        print(f"  offline[{os.path.basename(p)}]: {offline[p]}")
    print()
    print(f"{'chunk':>6}{'left':>6}{'look':>6}  {'meanWER':>8}{'maxWER':>8}{'clean':>7}  worst-clip")

    best_look = None
    for chunk in chunks:
        for left in lefts:
            for look in looks:
                wers = []
                worst = ("", 0.0)
                for p in wavs:
                    s = model.transcribe_streaming(
                        p, chunk_s=chunk, left_context_s=left, lookahead_s=look)
                    w = wer(offline[p], s)
                    wers.append(w)
                    if w >= worst[1]:
                        worst = (os.path.basename(p), w)
                mean_w = sum(wers) / len(wers)
                max_w = max(wers)
                clean = sum(1 for w in wers if w == 0.0)
                flag = ""
                if (chunk == default_chunk and left == default_left
                        and max_w <= tol and best_look is None):
                    best_look = look
                    flag = "  <- snappiest within tol"
                print(f"{chunk:6.2f}{left:6.2f}{look:6.2f}  {mean_w:8.3f}{max_w:8.3f}"
                      f"{clean:4d}/{len(wavs):<2d}  {worst[0]} ({worst[1]:.2f}){flag}")

    print()
    if best_look is not None:
        print(f"RECOMMEND lookahead={best_look:.2f}s at chunk={default_chunk} left={default_left} "
              f"(smallest with max streamed-vs-offline WER <= {tol}).")
    else:
        print(f"No swept lookahead kept max WER <= {tol} at chunk={default_chunk}/left={default_left}"
              " — widen --looks or raise --tol.")
