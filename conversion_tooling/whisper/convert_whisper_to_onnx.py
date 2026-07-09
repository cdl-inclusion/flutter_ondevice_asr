# Whisper model conversion pipeline for ONNX
#
# This script combines:
# - HuggingFace Whisper to ONNX conversion
# - Extraction of preprocessor
# - Merge of encoder with preprocessor to super_encoder
#
# Creates full precision and int8 quantized variants
#
# Model input can be:
# - HuggingFace model ID (e.g., "openai/whisper-tiny", "openai/whisper-base")
# - Local path to a fine-tuned Whisper model (after HF model has been saved locally)
#
# Example usage:
#   python convert_whisper_to_onnx.py "openai/whisper-tiny" /tmp/onnx_tiny
#   python convert_whisper_to_onnx.py "./my-finetuned-whisper" /tmp/onnx_tiny


from optimum.onnxruntime import ORTModelForSpeechSeq2Seq, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from onnxruntime import GraphOptimizationLevel
from onnxruntime.tools.optimize_onnx_model import optimize_model
from onnxruntime.quantization import QuantType, quantize_dynamic
import glob
from pathlib import Path

from transformers import WhisperProcessor, WhisperTokenizer
import os, shutil
import json
import time
import argparse
import shutil
import onnx
from onnx import compose
from whisper_preprocessor import WhisperPreprocessor80

# ONNX IR version to use for all models (for compatibility)
IR_VERSION = 8


def convert_to_onnx(original_model_path, default_onnx_folder):
    # base conversion to ONNX (uses default opset from Optimum library)
    # Note: opset parameter not supported in Optimum 2.1.0
    ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(
        original_model_path,
        export=True,
    )
    ort_model.save_pretrained(default_onnx_folder)

    for onnx_file in glob.glob(os.path.join(default_onnx_folder, "*.onnx")):
        model = onnx.load(onnx_file)
        model.ir_version = IR_VERSION
        onnx.save(model, onnx_file)

    print(f"Model saved to {default_onnx_folder}")

def copy_config_files(source_dir, target_dir):

    for json_file in glob.glob(os.path.join(source_dir, "*.json")):
        shutil.copy2(json_file, target_dir)


def quantize_onnx(onnx_input_dir, quantized_onnx_output_dir):
    # quantize to int8
    # one could quantize the optimized version as well but I didn't see much improvements here    
    print(f"\n>>> Quantizing models from {onnx_input_dir}...")

    onnx_model_paths = glob.glob(os.path.join(onnx_input_dir, "*.onnx"))
    for model_path in onnx_model_paths:
        base_name = os.path.basename(model_path)
        output_path_quantized = os.path.join(quantized_onnx_output_dir, base_name)
        print(f">>> Quantizing {base_name}...")
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path_quantized,
            # only do MatMul quantization (if doing conv, inference fails)
            op_types_to_quantize=['MatMul'],
            weight_type=QuantType.QInt8,
            extra_options={'DefaultTensorType': 1}  # 1 = FLOAT
        )
    copy_config_files(onnx_input_dir, quantized_onnx_output_dir)
    print(f"Quantized {onnx_input_dir}.")


def export_preprocessor_80(output_path="whisper_preprocessor_80.onnx"):
    """Export 80-band mel spectrogram preprocessor for Whisper.

    Args:
        output_path: Path where preprocessor ONNX model will be saved

    Returns:
        Path to the saved preprocessor model
    """
    print("Exporting WhisperPreprocessor80...")
    model = WhisperPreprocessor80.to_model_proto()

    # Set IR version to match encoder (onnxscript creates IR 10 by default otherwise)
    model.ir_version = IR_VERSION

    onnx.save(model, output_path)
    print(f"✓ Saved to {output_path} (IR version: {model.ir_version}, Opset: {model.opset_import[0].version})")
    return output_path


def make_super_encoder(preprocessor_path, encoder_path, output_path):
    """Merge preprocessor and encoder ONNX models into a single "super encoder".

    Uses onnx.compose.merge_models to combine the models by connecting:
    - Preprocessor output 'features' -> Encoder input 'input_features'

    The merged model will:
    - Input: raw audio waveform (Float32[batch, samples])
    - Output: encoder hidden states (Float32[batch, seq_len, hidden_dim])

    Args:
        preprocessor_path: Path to preprocessor ONNX model
        encoder_path: Path to encoder ONNX model
        output_path: Path where super encoder will be saved

    Returns:
        The merged ONNX model
    """
    print(f"Loading preprocessor from {preprocessor_path}...")
    preprocessor = onnx.load(preprocessor_path)

    print(f"Loading encoder from {encoder_path}...")
    encoder = onnx.load(encoder_path)

    print("\nPreprocessor info:")
    print(f"  Inputs: {[i.name for i in preprocessor.graph.input]}")
    print(f"  Outputs: {[o.name for o in preprocessor.graph.output]}")

    print("\nEncoder info:")
    print(f"  Inputs: {[i.name for i in encoder.graph.input]}")
    print(f"  Outputs: {[o.name for o in encoder.graph.output]}")

    # Check IR version compatibility
    if preprocessor.ir_version != encoder.ir_version:
        print(f"\n⚠ IR version mismatch: preprocessor={preprocessor.ir_version}, encoder={encoder.ir_version}")
        print(f"  Downgrading preprocessor to IR version {encoder.ir_version}...")
        preprocessor.ir_version = encoder.ir_version

    # Map preprocessor outputs to encoder inputs
    # Preprocessor outputs 'features' (mel spectrogram)
    # Encoder expects 'input_features'
    io_map = [("features", "input_features")]

    print(f"\nMerging with io_map: {io_map}")

    # Use ONNX compose to merge the models
    merged_model = compose.merge_models(
        preprocessor,
        encoder,
        io_map=io_map
    )

    # Validate the merged model
    print("\nValidating merged model...")
    try:
        onnx.checker.check_model(merged_model)
        print("✓ Model validation passed")
    except Exception as e:
        print(f"⚠ Validation warning: {e}")
        print("  (This may be OK - some validators are strict)")

    # Save merged model
    print(f"\nSaving merged model to {output_path}...")
    onnx.save(merged_model, output_path)

    # Print merged model info
    model_size_mb = len(merged_model.SerializeToString()) / (1024 * 1024)
    print(f"\n✓ Merged model saved successfully!")
    print(f"  Size: {model_size_mb:.2f} MB")
    print(f"  IR Version: {merged_model.ir_version}")
    print(f"  Opset: {merged_model.opset_import[0].version}")
    print(f"  Inputs: {[i.name for i in merged_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in merged_model.graph.output]}")

    return merged_model


def export_vocab(model_path: str, output_path: str) -> str:
    """Export tokenizer vocab to a JSON file.

    Args:
        model_path: HuggingFace model ID or local path
        output_path: Path where vocab.json will be saved

    Returns:
        Path to the saved vocab file
    """
    print(f"Exporting vocab from {model_path}...")
    tokenizer = WhisperTokenizer.from_pretrained(model_path)

    # The vocab is a dict mapping token strings to IDs
    vocab = tokenizer.get_vocab()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved vocab ({len(vocab)} tokens) to {output_path}")
    return output_path


def get_expected_files():
    """
    Get the list of expected files in each variant output folder.

    Returns:
        list: List of expected filenames after conversion
    """
    return [
        "super_encoder.onnx",
        "decoder_model.onnx",
        "decoder_with_past_model.onnx",
        "config.json",
        "generation_config.json",
        "vocab.json"
    ]


def run_conversion(original_model_path: str, onnx_output_folder: str) -> list[str]:
    """
    Convert a HuggingFace Whisper model to ONNX format with optimization and quantization.

    Args:
        original_model_path: Path to the Whisper model (HuggingFace Hub ID or local path)
        onnx_output_folder: Output folder for ONNX models

    Returns:
        list[str]: List of paths to the generated variant folders [default_folder, int8_folder]
    """
    default_onnx_folder = os.path.join(onnx_output_folder, 'default')
    default_int8_onnx_folder = os.path.join(onnx_output_folder, 'default_int8')
    preprocessor_folder = os.path.join(onnx_output_folder, 'preprocessor')

    os.makedirs(default_onnx_folder, exist_ok=True)
    os.makedirs(default_int8_onnx_folder, exist_ok=True)
    os.makedirs(preprocessor_folder, exist_ok=True)

    # Convert to ONNX
    convert_to_onnx(original_model_path, default_onnx_folder)

    # Quantize to int8
    quantize_onnx(default_onnx_folder, default_int8_onnx_folder)

    # Export preprocessor
    print("\n" + "=" * 70)
    print("Exporting preprocessor")
    print("=" * 70)
    preprocessor_path = os.path.join(preprocessor_folder, 'whisper_preprocessor_80.onnx')
    export_preprocessor_80(preprocessor_path)

    # Create super encoders for each variant we want to support later
    print("\n" + "=" * 70)
    print("Creating super encoders")
    print("=" * 70)

    variants = [
        # for now we only fully export the int8 variants and skip creating all the components like
        # super encoder for unsupported variants (e.g., default full precision)
        # skip:
        # ('default', default_onnx_folder),
        ('default_int8', default_int8_onnx_folder),
    ]

    for variant_name, variant_folder in variants:
        encoder_path = os.path.join(variant_folder, 'encoder_model.onnx')
        print(f"\n--- Creating super encoder for {variant_name} ---")
        super_encoder_path = os.path.join(variant_folder, 'super_encoder.onnx')
        make_super_encoder(preprocessor_path, encoder_path, super_encoder_path)

        # Remove standalone encoder since we now have super_encoder
        print(f"Removing standalone encoder_model.onnx from {variant_name}...")
        os.remove(encoder_path)

    # Export vocab to each variant folder (so each folder is self-contained)
    print("\n" + "=" * 70)
    print("Exporting vocab")
    print("=" * 70)

    for variant_name, variant_folder in variants:
        vocab_path = os.path.join(variant_folder, 'vocab.json')
        export_vocab(original_model_path, vocab_path)

    # Validate all expected files are present in each variant
    print("\n" + "=" * 70)
    print("Validating output files")
    print("=" * 70)

    expected_files = get_expected_files()

    for variant_name, variant_folder in variants:
        print(f"\nChecking {variant_name}...")
        missing_files = []
        for filename in expected_files:
            filepath = os.path.join(variant_folder, filename)
            if os.path.exists(filepath):
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename} - MISSING")
                missing_files.append(filename)

        if missing_files:
            raise FileNotFoundError(
                f"Conversion incomplete for {variant_name}. Missing files: {', '.join(missing_files)}"
            )

    print(f"\n✓ Conversion complete! Models saved to {onnx_output_folder}")

    return [variant_folder for _, variant_folder in variants]


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Convert Whisper model to ONNX format')
    parser.add_argument('original_model_path', type=str, default="openai/whisper-tiny", help='Path to the Whisper model (HuggingFace Hub ID or local path)')
    parser.add_argument('onnx_output_folder', type=str, help='Base folder where onnx model will be written to (each in subfolders)')
    args = parser.parse_args()

    run_conversion(args.original_model_path, args.onnx_output_folder)    