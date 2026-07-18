#!/bin/bash
# Build FastConformer on-device assets (hybrid super-encoder) for the Flutter bundle.
#
# Runs the LOCAL, no-Modal conversion
# (conversion_tooling/fastconformer/convert_fastconformer_to_onnx_local.py) and writes the
# hybrid artifact DIRECTLY into assets/transcribers/fastconformer/ — both hybrid/ (fp32) and
# hybrid_int8/ (int8). No temp dir / copy step. (Add --validate to the converter call below
# to also parity-check the graphs vs NeMo.)
#
# Needs the FastConformer conversion venv (heavy NeMo stack; Python 3.11 on macOS — see
# conversion_tooling/fastconformer/requirements.txt). For a no-local-deps
# path, use conversion_tooling/fastconformer/convert_fastconformer_to_onnx_modal.py instead.
#
# Usage:
#   ./build_fastconformer_assets.sh [model_id_or_local_nemo]
# Examples:
#   ./build_fastconformer_assets.sh                                        # base EN (default)
#   ./build_fastconformer_assets.sh nvidia/stt_en_fastconformer_hybrid_large_pc
#   ./build_fastconformer_assets.sh /path/to/best_model                    # local adapted .nemo dir

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
CONVERSION_DIR="$PROJECT_ROOT/conversion_tooling/fastconformer"
ASSETS_DIR="$PROJECT_ROOT/assets/transcribers/fastconformer"

# Default model (base EN, generalizes — not a speaker-adapted fine-tune).
MODEL="${1:-nvidia/stt_en_fastconformer_hybrid_large_pc}"

echo "======================================================================="
echo "Building FastConformer hybrid assets"
echo "======================================================================="
echo "Model:  $MODEL"
echo "Output: $ASSETS_DIR   (hybrid/ = fp32, hybrid_int8/ = int8)"
echo ""

# Check the conversion venv (NeMo + torch + onnx; Python 3.11 recommended on macOS).
if [ ! -d "$CONVERSION_DIR/venv" ]; then
    echo "❌ Error: conversion venv not found at $CONVERSION_DIR/venv"
    echo "Create it (macOS: use Python 3.11 — see requirements.txt):"
    echo "  cd conversion_tooling/fastconformer"
    echo "  python3.11 -m venv venv && source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi
# Call the venv's interpreter DIRECTLY (robust — no `source activate` needed, so it works
# whether the script is run as `./build_...sh`, `bash build_...sh`, or `sh build_...sh`, and
# doesn't depend on a bare `python` existing on the system — macOS only has `python3`).
PY="$CONVERSION_DIR/venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "❌ Error: venv interpreter not found/executable at $PY"
    exit 1
fi
# Fail early (clear message, not a raw traceback) if the venv is bare / deps not installed.
# Use find_spec so we DON'T pay NeMo's heavy import cost just to check presence (NeMo is
# imported lazily, only when needed, inside the converter itself).
if ! "$PY" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('nemo') else 1)" >/dev/null 2>&1; then
    echo "❌ Error: the venv at $CONVERSION_DIR/venv is missing dependencies ('nemo' not found)."
    exit 1
fi

# Convert DIRECTLY into the assets dir — writes $ASSETS_DIR/hybrid (fp32) +
# $ASSETS_DIR/hybrid_int8 (int8). The converter clears ONLY the hybrid/ and hybrid_int8/
# subdirs (not the rest of the assets dir). To also parity-check the graphs vs NeMo, add
# --validate below (fp32 PASS gates; int8 FAIL on the synthetic gate is expected and
# non-gating — see build_assets.md).
echo "======================================================================="
echo "Converting (fp32 + int8) directly into assets"
echo "======================================================================="
"$PY" "$CONVERSION_DIR/convert_fastconformer.py" \
    --model "$MODEL" \
    --out "$ASSETS_DIR"

echo ""
echo "======================================================================="
echo "✓ All assets built successfully!"
echo "======================================================================="
