# Building FastConformer Flutter assets

How to build the on-device FastConformer ASR assets and bundle them into the Flutter
app. For FastConformer, we build one super-encoder (feature extraction and encoder fused together) and two different heads (CTC and RNN-T), which can be combined with it.

## Architecture

```
NeMo hybrid FastConformer (e.g. nvidia/stt_en_fastconformer_hybrid_large_pc)
         ↓
   [build_fastconformer_assets.sh]  →  conversion_tooling/fastconformer/convert_fastconformer.py
         ↓
   fp32/   +   int8/ (int8)   — each dir contains:
     super_encoder.onnx   waveform -> encoder_out   (NeMo mel preprocessor merged into the encoder)
     ctc_decoder.onnx     encoder_out -> logprobs   (CTC head)
     decoder_joint.onnx   fused prednet + joint     (RNN-T with one greedy step)
     tokens.txt, meta.json
```


## Directory structure

```
flutter_ondevice_asr/
├── build_fastconformer_assets.sh            # ⭐ run this (from the repo root)
├── models/fastconformer/                    # temp build dir (gitignored) — fp32/ + int8/
├── conversion_tooling/                      # shared runtime modules (used by benchmark_local.py)
│   ├── onnx_transcriber.py                          # shared Transcriber interface + helpers
│   ├── onnx_fastconformer_transcriber.py           # NeMo-free ONNX runtime (Dart-port reference)
│   └── fastconformer/                        # the conversion tooling (self-contained, CPU)
│       ├── convert_fastconformer_to_onnx_lib.py    # convert() + quantize() — all the logic
│       ├── convert_fastconformer_validation.py     # validate_hybrid() vs NeMo (synthetic)
│       └── convert_fastconformer.py                # local runner (this is what the script calls)
└── assets/transcribers/fastconformer/
    └── int8/                          # bundled on-device asset (~149 MB)
        ├── super_encoder.onnx  ctc_decoder.onnx  decoder_joint.onnx  tokens.txt  meta.json
```

## Quick start

```bash
# 1) one-time: create the conversion venv (macOS: successfully tested Python 3.11, other python versions have been problematic with deps)
cd conversion_tooling
python3.11 -m venv fastconformer/venv && source fastconformer/venv/bin/activate
pip install -e ".[fastconformer]"
cd ..

# 2) build + bundle (base EN by default, for testing here)
./build_fastconformer_assets.sh
# ...or a specific model / a local adapted checkpoint:
./build_fastconformer_assets.sh nvidia/stt_en_fastconformer_hybrid_large_pc
./build_fastconformer_assets.sh /path/to/best_model      # a dir containing model.nemo
```

This writes `assets/transcribers/fastconformer/int8/` — the asset the Dart tests and
app use.

## What the build script does
Runs `python -m conversion_tooling.fastconformer.convert_fastconformer --model <M> --out assets/transcribers/fastconformer/` (from the repo root; add `--validate` in the script for parity checks):
* loads the checkpoint on CPU, exports the hybrid artifact (fp32 → `fp32/`)
* int8-quantizes it (→ `int8/`)
* validates both vs original NeMo checkpoint (parity check on synthetic data).

