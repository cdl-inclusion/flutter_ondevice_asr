#!/usr/bin/env python3
"""Runner for FastConformer -> ONNX hybrid conversion.

Needs NeMo + torch + onnx/onnxruntime installed locally.

    # base HF checkpoint
    python convert_fastconformer_to_onnx_local.py \\
        --model nvidia/stt_en_fastconformer_hybrid_large_pc --out /tmp/fc_local

    # a LOCAL adapted checkpoint (.nemo file, or a dir containing one) + validate
    python convert_fastconformer_to_onnx_local.py \\
        --model /path/to/best_model/model.nemo --out /tmp/fc_local --validate

Writes <out>/hybrid/ (fp32) and <out>/hybrid_int8/ (int8).
"""

import argparse
import sys
from pathlib import Path


def _resolve_source(src: str):
    """Return (loader, arg, label): an HF id (from_pretrained) OR a local .nemo file/dir
    (restore_from). A local filesystem path -> adapted; anything else -> HF hub id."""
    p = Path(src)
    is_local = src.endswith(".nemo") or p.exists()
    if not is_local:
        return "hf", src, src
    if p.is_dir():
        nemo = p / "model.nemo"
        if not nemo.exists():
            cands = sorted(p.glob("*.nemo"))
            if not cands:
                raise FileNotFoundError(f"No .nemo found in {p}")
            nemo = cands[0]
    else:
        nemo = p
    return "nemo", str(nemo), str(nemo)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="HF repo id or a local checkpoint (.nemo file or a "
                         "dir containing one).")
    ap.add_argument("--language", default="en")
    ap.add_argument("--out", required=True,
                    help="Local output dir; writes <out>/hybrid (fp32) and <out>/hybrid_int8.")
    ap.add_argument("--validate", action="store_true",
                    help="After building, validate BOTH quant versions vs NeMo (both heads, "
                         "SYNTHETIC signals — a conservative worst case that often fails int8).")
    args = ap.parse_args()

    import logging
    logging.getLogger("nemo_logger").setLevel(logging.ERROR)
    from nemo.collections.asr.models import ASRModel
    from convert_fastconformer_to_onnx_lib import convert, quantize

    loader, arg, label = _resolve_source(args.model)
    print(f"[LOAD] ({'HF' if loader == 'hf' else 'local .nemo'}) {label}", flush=True)
    if loader == "hf":
        model = ASRModel.from_pretrained(model_name=arg, map_location="cpu")
    else:
        model = ASRModel.restore_from(arg, map_location="cpu")
    model.eval()

    out = Path(args.out)
    fp32, int8 = str(out / "hybrid"), str(out / "hybrid_int8")

    convert(model, fp32, language=args.language, base_model=label)

    validation = validation_int8 = None
    if args.validate:
        from convert_fastconformer_validation import validate_hybrid
        print("[VALIDATE] fp32 vs NeMo:", flush=True)
        validation = validate_hybrid(model, fp32)

    int8_ok = False
    try:
        quantize(fp32, int8)
        int8_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] int8 quantization failed ({e}); fp32 only.")

    if args.validate and int8_ok:
        from convert_fastconformer_validation import validate_hybrid
        print("[VALIDATE] int8 vs NeMo (lossy — larger deltas expected):", flush=True)
        validation_int8 = validate_hybrid(model, int8)

    print("\n" + "=" * 70)
    print("Built hybrid artifact:")
    print(f"  fp32 -> {fp32}")
    print(f"  int8 -> {int8 if int8_ok else '(SKIPPED — quant failed)'}")

    passed = True
    if validation is not None:
        from convert_fastconformer_validation import print_validation_cases
        passed = validation["passed"]
        print(f"\nValidation vs NeMo (fp32): {'PASS' if passed else 'FAIL'}")
        print_validation_cases(validation["cases"])
    if validation_int8 is not None:
        from convert_fastconformer_validation import print_validation_cases
        print(f"\nValidation vs NeMo (int8, lossy — larger deltas expected): "
              f"{'PASS' if validation_int8['passed'] else 'FAIL'}")
        print_validation_cases(validation_int8["cases"])
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
