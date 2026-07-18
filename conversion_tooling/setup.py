"""Setup for on-device ONNX conversion tooling (Whisper + FastConformer).

Installable package with the converter selectable via extras:

    pip install "onnx-conversion-tooling[whisper] @ git+https://github.com/cdl-inclusion/flutter_ondevice_asr.git@<tag>#subdirectory=conversion_tooling"
    pip install "onnx-conversion-tooling[fastconformer] @ git+https://github.com/cdl-inclusion/flutter_ondevice_asr.git@<tag>#subdirectory=conversion_tooling"

NOTE: The two dependency stacks are mutually incompatible (NeMo 2.3.0
hard-pins numpy<2 and protobuf==4.24.4, while the Whisper pipeline needs
transformers==4.46.3 with the modern onnx stack). Don't install both extras
into the same environment — each converter should run in its own worker container.

Usage after install:

    from conversion_tooling.whisper.convert_whisper_to_onnx import run_conversion
    run_conversion("openai/whisper-tiny", "./output")
"""

from setuptools import setup

if __name__ == "__main__":
    setup(
        name="onnx-conversion-tooling",
        version="0.6.0",
        description="On-device ONNX conversion tooling for Whisper and FastConformer models",
        long_description=__doc__,
        long_description_content_type="text/plain",
        author="Katrin Tomanek",
        url="https://github.com/cdl-inclusion/flutter_ondevice_asr",
        packages=[
            "conversion_tooling",
            "conversion_tooling.whisper",
            "conversion_tooling.fastconformer",
        ],
        package_dir={"conversion_tooling": "."},
        # Shared code (onnx_conversion_constants) is stdlib-only; all real
        # dependencies live in the per-converter extras below.
        install_requires=[],
        extras_require={
            "whisper": [
                # torch 2.5.1 is no longer on PyPI, 2.6.0 is oldest available.
                "torch==2.6.0",
                "torchaudio==2.6.0",
                # transformers 4.57+ has a regression affecting Whisper encoder-only training.
                "transformers==4.46.3",
                "optimum[onnxruntime]>=1.19.0,<2.2.0",
                "onnx>=1.16.0,<1.21.0",
                "onnxscript>=0.5.0,<0.7.0",
                "onnxruntime>=1.19.0",
            ],
            # NeMo 2.3.0 HARD-pins numpy<2.0.0 and protobuf==4.24.4. Without them pip
            # drifts to the modern onnx/onnxscript stack (numpy 2.x + protobuf 7.x),
            # which breaks NeMo and numba. Keeping these pins forces the older,
            # mutually-compatible onnx-stack resolution.
            "fastconformer": [
                "nemo_toolkit[asr]==2.3.0",
                "numpy<2",
                "protobuf==4.24.4",
                "librosa",
                "soundfile",
                "onnx",
                "onnxruntime",
                "onnxscript",
            ],
        },
        python_requires=">=3.9",
    )
