#!/bin/bash
# Build optimized assets for Flutter bundle from HuggingFace models
#
# This script:
# 1. Runs the Python conversion pipeline (convert_whisper_to_onnx.py)
# 2. Copies the output to Flutter assets directory
#
# Usage:
#   ./build_assets.sh [model_id]
#
# Examples:
#   ./build_assets.sh                      # Uses default: openai/whisper-tiny
#   ./build_assets.sh openai/whisper-base  # Use a different model

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
CONVERSION_DIR="$PROJECT_ROOT/conversion_tooling/whisper"
MODELS_DIR="$PROJECT_ROOT/models"  # Temporary build directory (gitignored)
ASSETS_DIR="$PROJECT_ROOT/assets/transcribers/whisper/models"

# Default model ID (can be overridden)
MODEL_ID="${1:-openai/whisper-tiny}"

echo "======================================================================="
echo "Building Flutter Assets from HuggingFace"
echo "======================================================================="
echo "Model: $MODEL_ID"
echo "Temp directory: $MODELS_DIR"
echo "Output: $ASSETS_DIR"
echo ""

# Check virtual environment
if [ ! -d "$CONVERSION_DIR/venv" ]; then
    echo "❌ Error: Virtual environment not found."
    echo "Run: cd conversion_tooling && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source "$CONVERSION_DIR/venv/bin/activate"

# Clean temp models directory
echo "1. Cleaning temporary models directory..."
rm -rf "$MODELS_DIR"

# Run the complete conversion pipeline
echo ""
echo "======================================================================="
echo "Running Whisper conversion pipeline"
echo "======================================================================="
echo ""

python "$CONVERSION_DIR/convert_whisper_to_onnx.py" \
    "$MODEL_ID" \
    "$MODELS_DIR/whisper_tiny"

echo ""
echo "✓ Conversion pipeline completed!"

# Copy assets to Flutter bundle
echo ""
echo "======================================================================="
echo "Copying models to Flutter assets"
echo "======================================================================="

# Function to copy assets for one model variant
copy_variant() {
    local variant=$1
    echo ""
    echo "-------------------------------------------------------------------"
    echo "Copying: whisper_tiny/$variant"
    echo "-------------------------------------------------------------------"

    local source_dir="$MODELS_DIR/whisper_tiny/$variant"
    local target_dir="$ASSETS_DIR/whisper_tiny/$variant"

    echo "  → Creating target directory..."
    mkdir -p "$target_dir"

    echo "  → Copying models and configs to assets..."
    cp "$source_dir/super_encoder.onnx" "$target_dir/"
    cp "$source_dir/decoder_model.onnx" "$target_dir/"
    cp "$source_dir/decoder_with_past_model.onnx" "$target_dir/"
    cp "$source_dir/config.json" "$target_dir/"
    cp "$source_dir/generation_config.json" "$target_dir/"
    cp "$source_dir/vocab.json" "$target_dir/"

    echo ""
    echo "  ✓ Copied $variant successfully!"
    ls -lh "$target_dir" | grep -E "(super_encoder|decoder|config|vocab)" | awk '{print "      " $9 " (" $5 ")"}'
}

# Only the int8 variant is fully built (super_encoder + vocab) and bundled in pubspec.yaml.
# The fp32 "default" variant is intentionally left as raw Optimum output by the converter
# (see convert_whisper_to_onnx.py run_conversion: super_encoder/vocab built for int8 only).
copy_variant "default_int8"

echo ""
echo "======================================================================="
echo "✓ All assets built successfully!"
echo "======================================================================="
