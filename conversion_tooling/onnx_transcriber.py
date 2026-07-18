"""Shared interface + helpers for the ONNX reference transcribers.
"""

from __future__ import annotations

import abc

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000


def make_session(onnx_path, num_threads: int) -> ort.InferenceSession:
    """A CPU ORT session with full graph optimization (matches the mobile target)."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = num_threads
    return ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def load_audio(path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as mono float32 at ``sr`` (librosa imported lazily)."""
    import librosa
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio.astype(np.float32)


class Transcriber(abc.ABC):
    """Minimal interface shared by the ONNX reference transcribers."""

    SAMPLE_RATE = SAMPLE_RATE

    @abc.abstractmethod
    def load(self, model_dir: str) -> None:
        """Load ONNX session(s) + tokenizer from an artifact directory."""

    @abc.abstractmethod
    def transcribe(self, audio) -> str:
        """Transcribe an audio file path (or waveform) to text."""
