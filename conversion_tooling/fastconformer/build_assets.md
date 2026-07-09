# Building FastConformer Flutter assets

How to build the on-device FastConformer ASR assets and bundle them into the Flutter
app. For FastConformer, we build one super-encoder (feature extraction and encoder fused together) and two different heads (CTC and RNN-T), which can be combined with it.

## Architecture

```
NeMo hybrid FastConformer (e.g. nvidia/stt_en_fastconformer_hybrid_large_pc)
         ↓
   [build_fastconformer_assets.sh]  →  conversion_tooling/fastconformer/convert_fastconformer.py
         ↓
   hybrid/ (fp32)   +   hybrid_int8/ (int8)   — each dir contains:
     super_encoder.onnx   waveform -> encoder_out   (NeMo mel preprocessor merged into the encoder)
     ctc_decoder.onnx     encoder_out -> logprobs   (CTC head)
     decoder_joint.onnx   fused prednet + joint     (RNN-T with one greedy step)
     tokens.txt, meta.json
```


## Directory structure

```
flutter_ondevice_asr/
├── build_fastconformer_assets.sh            # ⭐ run this (from the repo root)
├── models/fastconformer/                    # temp build dir (gitignored) — hybrid/ + hybrid_int8/
├── conversion_tooling/fastconformer/        # the conversion tooling (self-contained, CPU)
│   ├── convert_fastconformer_to_onnx_lib.py        # convert() + quantize() — all the logic
│   ├── convert_fastconformer_validation.py         # validate_hybrid() vs NeMo (synthetic)
│   ├── onnx_fastconformer_transcriber.py           # NeMo-free ONNX runtime (Dart-port reference)
│   ├── convert_fastconformer.py      # local runner (this is what the script calls)
│   └── requirements.txt           # local pip deps (NeMo stack)
└── assets/transcribers/fastconformer/
    └── hybrid_int8/                          # bundled on-device asset (~149 MB)
        ├── super_encoder.onnx  ctc_decoder.onnx  decoder_joint.onnx  tokens.txt  meta.json
```

## Quick start

```bash
# 1) one-time: create the conversion venv (macOS: successfully tested Python 3.11, other python versions have been problematic with deps)
cd conversion_tooling/fastconformer
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements_conversion_local.txt
cd ../..

# 2) build + bundle (base EN by default, for testing here)
./build_fastconformer_assets.sh
# ...or a specific model / a local adapted checkpoint:
./build_fastconformer_assets.sh nvidia/stt_en_fastconformer_hybrid_large_pc
./build_fastconformer_assets.sh /path/to/best_model      # a dir containing model.nemo
```

This writes `assets/transcribers/fastconformer/hybrid_int8/` — the asset the Dart tests and
app use.

## What the build script does
Runs `convert_fastconformer.py --model <M> --out assets/transcribers/fastconformer/ --validate`:
* loads the checkpoint on CPU, exports the hybrid artifact (fp32 → `hybrid/`)
* int8-quantizes it (→ `hybrid_int8/`)
* validates both vs original NeMo checkpoint (parity check on synthetic data).

## No local NeMo? Use Modal (identical output)
The heavy NeMo stack can be painful to install locally. The Modal runner does the same
conversion in a clean container (CPU-only, deps handled in-image):
```bash
cd conversion_tooling/fastconformer
python convert_fastconformer_to_onnx_modal.py --model nvidia/stt_en_fastconformer_hybrid_large_pc \
    --validate --out /tmp/fc
# then copy /tmp/fc/hybrid_int8/* into assets/transcribers/fastconformer/hybrid_int8/
```

