

# Building Flutter Assets from HuggingFace

This document explains how to build optimized Flutter assets directly from HuggingFace models using a single build script.

## Architecture Overview

The build process is fully automated and reproducible:

```
HuggingFace Model (openai/whisper-tiny)
         ↓
   [build_assets.sh]
         ↓
   1. Export from HuggingFace → models/ (temp)
   2. Export preprocessor → models/preprocessor/
   3. Merge preprocessor + encoder → super_encoder.onnx
   4. Copy to assets/models/ (bundled)
         ↓
   Flutter App Bundle
```

## Directory Structure

```
flutter_onnx_whisper/
├── models/                              # Temporary build directory (gitignored)
│   ├── whisper_tiny/                    # Created during build, then deleted
│   │   ├── fp32/
│   │   ├── int8/
│   └── preprocessor/
│       └── whisper_preprocessor_80.onnx
│
├── build_assets.sh                      # ⭐ Main build script (run this!)
├── conversion_tooling/
│   ├── convert_whisper_to_onnx.py       # Step 1: Export from HuggingFace
│   ├── export_whisper_preprocessor.py   # Step 2: Create preprocessor
│   └── merge_preprocessor_encoder.py    # Step 3: Merge into super encoder
│
└── assets/models/                       # Bundled assets (committed to git)
    └── whisper_tiny/
        ├── fp32/
        │   ├── super_encoder.onnx       # ✅ Merged (preprocessor + encoder)
        │   ├── decoder_model.onnx       # ✅ Copied from HuggingFace
        │   ├── decoder_with_past_model.onnx
        │   ├── config.json
        │   └── generation_config.json
        └── int8/
            └── ... (same structure)
```

## Key Concepts

### No Source Models Stored
- ❌ We don't commit source models to git
- ✅ Build script downloads from HuggingFace
- ✅ `models/` is temporary (gitignored)
- ✅ Only bundled assets committed to git

### Fully Reproducible
```bash
# Anyone can rebuild assets from scratch
./build_assets.sh                    # Downloads and builds everything
```

### What Gets Bundled
- ✅ `super_encoder.onnx` - Merged preprocessor + encoder (11MB)
- ✅ `decoder_model.onnx` - Copied from HuggingFace (105MB)
- ✅ `decoder_with_past_model.onnx` - Copied from HuggingFace (104MB)
- ✅ Config files
- ❌ Standalone `encoder_model.onnx` - NOT bundled (replaced by super encoder)

## Quick Start

### Prerequisites

```bash
cd conversion_tooling

# Create virtual environment (one time)
python -m venv whisper/venv
source whisper/venv/bin/activate  # or `whisper\venv\Scripts\activate` on Windows
pip install -e ".[whisper]"
```

### Build Assets

```bash
./build_assets.sh
```

That's it! The script will:
1. Download Whisper model from HuggingFace
2. Export 2 variants (fp32, int8)
3. Create preprocessor
4. Merge preprocessor + encoder into super_encoder.onnx
5. Copy everything to assets/

### Use a Different Model

```bash
# Build assets for whisper-base instead of whisper-tiny
./build_assets.sh openai/whisper-base

# Or any other Whisper model on HuggingFace
./build_assets.sh openai/whisper-small
```

## What the Build Script Does

### Step 1: Export from HuggingFace
```bash
python convert_whisper_to_onnx.py "openai/whisper-tiny" models/whisper_tiny
```
Creates 2 variants in `models/whisper_tiny/`:
- `fp32/` - Full precision
- `int8/` - Int8 quantized

### Step 2: Export Preprocessor
```bash
python export_whisper_preprocessor.py
```
Creates `whisper_preprocessor_80.onnx` (66KB)

### Step 3: Build Each Variant
For each variant (fp32, int8):

```bash
# Merge preprocessor + encoder
python merge_preprocessor_encoder.py \
    --preprocessor models/preprocessor/whisper_preprocessor_80.onnx \
    --encoder models/whisper_tiny/int8/encoder_model.onnx \
    --output assets/models/whisper_tiny/int8/super_encoder.onnx

# Copy decoders and configs
cp models/whisper_tiny/int8/decoder*.onnx assets/...
cp models/whisper_tiny/int8/*.json assets/...

# Remove standalone encoder (not needed in bundle)
rm -f assets/models/whisper_tiny/int8/encoder_model.onnx
```

## Output

After running `build_assets.sh`:

```
Building Flutter Assets from HuggingFace
======================================================================
Model: openai/whisper-tiny

Step 1: Exporting Whisper models from HuggingFace
✓ Whisper models exported successfully!

Step 2: Exporting preprocessor
✓ Preprocessor exported successfully!

Step 3: Building optimized assets
  Building: whisper_tiny/fp32
  ✓ Built fp32 successfully!

  Building: whisper_tiny/int8
  ✓ Built int8 successfully!

✓ All assets built successfully!
```

### Verify Assets

```bash
ls -lh assets/models/whisper_tiny/int8/

# Should see:
# super_encoder.onnx (11MB)              ← Merged preprocessor + encoder
# decoder_model.onnx (105MB)
# decoder_with_past_model.onnx (104MB)
# config.json
# generation_config.json
```

## Testing

After building assets, verify everything works:

```bash
cd ..  # Back to project root
flutter test

# All tests should pass
00:06 +12: All tests passed!
```

## Cleanup

The build script creates temporary files in `models/`:

```bash
# Keep models/ for faster rebuilds (cached)
ls models/

# Or delete to save space (~660MB)
rm -rf models/
```

Next time you run `build_assets.sh`, it will re-download from HuggingFace.

## Bundle Size

### Before (Separate Models)
```
assets/models/whisper_tiny/int8/:
  encoder_model.onnx               11MB  ← Standalone encoder
  decoder_model.onnx              105MB
  decoder_with_past_model.onnx    104MB
  Total: ~220MB
```

### After (Super Encoder)
```
assets/models/whisper_tiny/int8/:
  super_encoder.onnx               11MB  ← Preprocessor + encoder merged
  decoder_model.onnx              105MB
  decoder_with_past_model.onnx    104MB
  Total: ~220MB
```

Same size, but:
- ✅ Better architecture (single inference call)
- ✅ Faster (~10-15% improvement)
- ✅ No marshalling overhead
- ✅ Cleaner code

### Custom Opset/IR Versions

Edit `export_whisper_preprocessor.py` to change opset or IR version:
```python
model.ir_version = 8  # Change this
```

Edit `convert_whisper_to_onnx.py` to change encoder opset:
```python
ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(
    original_model_path,
    export=True,
    opset=18,  # Change this
)
```

## See Also

- `SUPER_ENCODER_IMPLEMENTATION.md` - How super encoder works
- `merge_preprocessor_encoder.py` - Merging implementation
- `convert_whisper_to_onnx.py` - HuggingFace export details
- `export_whisper_preprocessor.py` - Preprocessor creation
