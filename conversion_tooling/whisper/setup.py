"""Setup for Whisper ONNX Conversion Tooling.

This package provides tools for converting Whisper models to ONNX format
with optimizations for on-device inference (mobile/embedded).

Includes:
- HuggingFace Whisper to ONNX conversion
- INT8 quantization
- ONNX-native preprocessor generation (based on onnx-asr)
- Preprocessor + encoder merging (super-encoder)

Model input can be:
- HuggingFace model ID (e.g., "openai/whisper-tiny")
- Local path to fine-tuned Whisper model

Install from Git:
    pip install git+https://github.com/your-org/flutter_ondevice_asr.git#subdirectory=conversion_tooling

Usage:
    from convert_whisper_to_onnx import run_conversion
    run_conversion("openai/whisper-tiny", "./output")
"""

from setuptools import setup

setup(
    name="whisper-conversion-tooling",
    version="0.1.0",
    description="Whisper model conversion to optimized ONNX for on-device inference",
    long_description=__doc__,
    long_description_content_type="text/plain",
    author="Katrin Tomanek",
    url="https://github.com/your-org/flutter_ondevice_asr",
    py_modules=[
        "whisper_preprocessor",
        "convert_whisper_to_onnx",
    ],
    install_requires=[
        # Pin versions for compatibility with transformers 4.46.3
        # Note: torch 2.5.1 is no longer on PyPI, using oldest available (2.6.0)
        # transformers 4.57+ has a regression affecting encoder-only training.
        "torch==2.6.0",
        "torchaudio==2.6.0",
        # HuggingFace ecosystem - pinned to known well working version aligned with training environment
        "transformers==4.46.3",
        "optimum[onnxruntime]>=1.19.0,<2.2.0",
        # ONNX tooling
        "onnx>=1.16.0,<1.21.0",
        "onnxscript>=0.5.0,<0.7.0",
        "onnxruntime>=1.19.0",
    ],
    python_requires=">=3.9",
)
