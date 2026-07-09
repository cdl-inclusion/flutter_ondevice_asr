"""Shared ONNX conversion constants for the Whisper and FastConformer converters.

Both converters must emit artifacts with the SAME opset / IR version so the app's
single bundled ONNX Runtime loads all of them. Keep these in one place so the two
pipelines can never drift apart.

- ONNX_OPSET      : the ai.onnx opset to target. Whisper's super-encoder preprocessor
                    hardcodes ``onnxscript.opset18`` to match; keep them in sync.
- ONNX_IR_VERSION : the ONNX IR version to pin. The torch/onnxscript exporter
                    otherwise defaults to IR 10, which the bundled (mobile) ORT
                    won't load.
"""

ONNX_OPSET = 18
ONNX_IR_VERSION = 8
