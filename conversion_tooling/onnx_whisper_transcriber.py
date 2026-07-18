"""Python implementation of ONNX Whisper transcriber.

This is a port of the Dart WhisperTranscriber implementation for inference
with ONNX Whisper models (super_encoder + decoder architecture).

Command line usage (from conversion_tooling directory):
    python onnx_whisper_transcriber.py ../assets/transcribers/whisper/models/whisper_tiny/default_int8 ../assets/audio/jfk_asknot.wav

Python usage:
    transcriber = OnnxWhisperTranscriber()
    transcriber.load("path/to/model_dir")
    )
    # Note: model_dir must contain: super_encoder.onnx, decoder_model.onnx,
    # decoder_with_past_model.onnx, generation_config.json, and vocab.json

    transcript = transcriber.transcribe("path/to/audio.wav")
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

# Shared interface + session/audio helpers (onnx_transcriber.py sits next to this file).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_transcriber import Transcriber, load_audio, make_session


class WhisperTokenizer:
    """Tokenizer for Whisper models (en-only and multilingual)."""

    def __init__(self):
        self._id_to_token: Optional[dict[int, str]] = None
        self._byte_decoder: Optional[dict[int, int]] = None

    def load_vocab(self, vocab_path: str) -> None:
        """Load vocabulary from JSON file."""
        if self._id_to_token is not None:
            return

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self._id_to_token = {v: k for k, v in vocab.items()}
        self._init_byte_decoder()

    def _init_byte_decoder(self) -> None:
        """Initialize byte decoder mapping (Unicode character -> raw byte)."""
        self._byte_decoder = {}
        b2u: dict[int, int] = {}

        # Add ranges exactly like HuggingFace's bytes_to_unicode
        for i in range(33, 127):  # ! to ~
            b2u[i] = i
        for i in range(161, 173):  # ¡ to ¬
            b2u[i] = i
        for i in range(174, 256):  # ® to ÿ
            b2u[i] = i

        # Any byte not added above is mapped to Unicode characters starting at 256
        n = 0
        for b in range(256):
            if b not in b2u:
                b2u[b] = 256 + n
                n += 1

        # For decoding: Unicode Character -> Raw Byte
        for byte_val, unicode_val in b2u.items():
            self._byte_decoder[unicode_val] = byte_val

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if self._id_to_token is None:
            raise RuntimeError("Tokenizer not loaded. Call load_vocab() first.")

        result = []
        all_bytes = []

        for token_id in token_ids:
            token = self._id_to_token.get(token_id)
            if token is None:
                continue

            # Handle special tokens separately
            if token.startswith("<|") and token.endswith("|>"):
                # Flush accumulated bytes first
                if all_bytes:
                    result.append(bytes(all_bytes).decode("utf-8", errors="replace"))
                    all_bytes = []

                # Add special token directly (or skip if requested)
                if not skip_special_tokens:
                    result.append(token)
                continue

            # Convert characters in token string back to raw bytes
            for char in token:
                char_code = ord(char)
                byte_val = self._byte_decoder.get(char_code)
                if byte_val is not None:
                    all_bytes.append(byte_val)
                elif char_code < 256:
                    all_bytes.append(char_code)

        # Flush remaining bytes
        if all_bytes:
            result.append(bytes(all_bytes).decode("utf-8", errors="replace"))

        return "".join(result)


class OnnxWhisperTranscriber(Transcriber):
    """ONNX Whisper ASR transcriber implementation."""

    DEFAULT_TOKENS_PER_SECOND = 6.0

    def __init__(
        self,
        language: str = "en",
        tokens_per_second: float = DEFAULT_TOKENS_PER_SECOND,
        num_threads: int = 8,
        verbose: bool = False,
    ):
        """Initialize the transcriber.

        Args:
            language: Language code (e.g., 'en', 'de', 'fr').
            tokens_per_second: Multiplier for dynamic max tokens calculation.
            num_threads: CPU threads for the ORT sessions (intra + inter op).
            verbose: Enable verbose logging.
        """
        self.language = language
        self.tokens_per_second = tokens_per_second
        self.num_threads = num_threads
        self.verbose = verbose

        self.super_encoder_session: Optional[ort.InferenceSession] = None
        self.decoder_session: Optional[ort.InferenceSession] = None
        self.decoder_with_past_session: Optional[ort.InferenceSession] = None
        self.tokenizer = WhisperTokenizer()

        # Token IDs (loaded from generation config)
        self.sot_token: int = 0
        self.eot_token: int = 0
        self.no_timestamps_token: int = 0
        self.transcribe_token: Optional[int] = None
        self.language_token: Optional[int] = None

    def load(
        self,
        model_path: str,
    ) -> None:
        """Load ONNX models and configuration.

        Args:
            encoder_path: Path to super_encoder.onnx.
            decoder_path: Path to decoder_model.onnx.
            decoder_with_past_path: Path to decoder_with_past_model.onnx.
                If not provided, derived from decoder_path directory.
            generation_config_path: Path to generation_config.json.
                If not provided, derived from encoder_path directory.
            vocab_path: Path to vocab.json file.
                If not provided, looks for vocab.json in the model directory.
        """
        # encoder_dir = Path(encoder_path).parent
        model_path = Path(model_path)
        encoder_path = str(model_path / "super_encoder.onnx")
        decoder_path = str(model_path / "decoder_model.onnx")

        # Derive paths if not provided
        decoder_with_past_path = str(model_path / "decoder_with_past_model.onnx")

        generation_config_path = str(model_path / "generation_config.json")

        # Load tokenizer vocab (expected in model directory)
        vocab_path = str(model_path / "vocab.json")

        if not Path(vocab_path).exists():
            raise FileNotFoundError(
                f"vocab.json not found at {vocab_path}. "
                f"Ensure the model was converted with the latest convert_whisper_to_onnx.py."
            )

        if self.verbose:
            print(f"Loading vocab from: {vocab_path}")
        self.tokenizer.load_vocab(vocab_path)

        # Load control tokens from generation config
        self._load_control_tokens(generation_config_path)

        # CPU-only ORT sessions (the ONNX model has ops optimized for mobile, not CUDA).
        if self.verbose:
            print(f"[ONNX] CPU execution provider with {self.num_threads} threads")
            print(f"Loading Super Encoder from: {encoder_path}")
        self.super_encoder_session = make_session(encoder_path, self.num_threads)

        if self.verbose:
            print(f"Loading Decoder from: {decoder_path}")
        self.decoder_session = make_session(decoder_path, self.num_threads)

        if self.verbose:
            print(f"Loading Decoder with Past from: {decoder_with_past_path}")
        self.decoder_with_past_session = make_session(decoder_with_past_path, self.num_threads)

        if self.verbose:
            print("Models loaded successfully!")

    def _load_control_tokens(self, config_path: str) -> None:
        """Load control token IDs from generation config."""
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        def get_token_id(value) -> int:
            if isinstance(value, int):
                return value
            elif isinstance(value, list) and len(value) > 0:
                return value[0]
            raise ValueError(f"Invalid token ID format: {value}")

        self.sot_token = get_token_id(config["decoder_start_token_id"])
        self.eot_token = get_token_id(config["eos_token_id"])
        self.no_timestamps_token = get_token_id(config["no_timestamps_token_id"])

        # Parse forced_decoder_ids for initial token sequence
        forced_decoder_ids = config.get("forced_decoder_ids")
        if forced_decoder_ids:
            for pair in forced_decoder_ids:
                position, token_id = pair[0], pair[1]
                if token_id is not None:
                    if position == 1 and token_id != self.no_timestamps_token:
                        self.language_token = token_id
                    elif position == 2:
                        self.transcribe_token = token_id

        # Use task_to_id for transcribe token if available
        task_to_id = config.get("task_to_id")
        if task_to_id and "transcribe" in task_to_id:
            self.transcribe_token = task_to_id["transcribe"]

        # If language specified but no language token, try lang_to_id
        if self.language and self.language_token is None:
            lang_to_id = config.get("lang_to_id")
            if lang_to_id:
                lang_key = f"<|{self.language}|>"
                self.language_token = lang_to_id.get(lang_key)

                if self.language_token is None:
                    available_langs = [k.replace("<|", "").replace("|>", "") for k in lang_to_id.keys()]
                    raise ValueError(
                        f'Language "{self.language}" not found. '
                        f"Available: {', '.join(available_langs)}"
                    )

        if self.verbose:
            print(
                f"Loaded token config: sot={self.sot_token}, eot={self.eot_token}, "
                f"lang={self.language_token}, transcribe={self.transcribe_token}, "
                f"notimestamps={self.no_timestamps_token}"
            )

    def _run_super_encoder(self, audio: np.ndarray) -> np.ndarray:
        """Run super encoder (preprocessor + encoder combined)."""
        audio_input = audio.reshape(1, -1)
        length_input = np.array([len(audio)], dtype=np.int64)

        outputs = self.super_encoder_session.run(
            None,
            {
                "waveforms": audio_input,
                "waveforms_lens": length_input,
            },
        )

        # Super encoder output: [0] = features_lens, [1] = last_hidden_state
        return outputs[1]

    def _run_decoder(self, audio_features: np.ndarray, max_tokens: int, verbose: bool = False) -> str:
        """Run decoder with KV-cache for autoregressive generation."""
        # Build initial tokens
        tokens = [
            self.sot_token,
            self.language_token,
            self.transcribe_token,
            self.no_timestamps_token,
        ]

        if verbose:
            print(
                f"Initial token IDs: sot={self.sot_token}, lang={self.language_token}, "
                f"transcribe={self.transcribe_token}, notimestamps={self.no_timestamps_token}"
            )

        past_key_values: Optional[dict[str, np.ndarray]] = None
        encoder_past_key_values: Optional[dict[str, np.ndarray]] = None

        decoder_output_names = [o.name for o in self.decoder_session.get_outputs()]
        decoder_with_past_output_names = [o.name for o in self.decoder_with_past_session.get_outputs()]

        max_new_tokens = max_tokens - len(tokens)

        for i in range(max_new_tokens):
            if past_key_values is None:
                # First iteration - use full decoder with all tokens
                tokens_input = np.array([tokens], dtype=np.int64)

                decoder_inputs = {
                    "input_ids": tokens_input,
                    "encoder_hidden_states": audio_features,
                }

                decoder_outputs = self.decoder_session.run(None, decoder_inputs)

                # Get logits for the last token position
                logits = decoder_outputs[0]  # shape: [1, seq_len, vocab_size]
                last_logits = logits[0, len(tokens) - 1, :]

                # Store past key values
                past_key_values = {}
                encoder_past_key_values = {}
                for j, output_name in enumerate(decoder_output_names[1:], start=1):
                    if output_name.startswith("present."):
                        past_name = output_name.replace("present.", "past_key_values.")
                        past_key_values[past_name] = decoder_outputs[j]

                        # Store encoder KV separately (stays constant)
                        if ".encoder." in output_name:
                            encoder_past_key_values[past_name] = decoder_outputs[j]

                if verbose:
                    print(
                        f"First iteration: stored {len(past_key_values)} past_key_values, "
                        f"{len(encoder_past_key_values)} from encoder"
                    )
            else:
                # Subsequent iterations - use decoder_with_past with only last token
                current_token = np.array([[tokens[-1]]], dtype=np.int64)

                decoder_with_past_inputs = {"input_ids": current_token}
                decoder_with_past_inputs.update(past_key_values)

                decoder_outputs = self.decoder_with_past_session.run(None, decoder_with_past_inputs)

                # Get logits
                logits = decoder_outputs[0]
                last_logits = logits[0, 0, :]

                # Update decoder past key values (encoder values stay constant)
                new_past = {}
                for j, output_name in enumerate(decoder_with_past_output_names[1:], start=1):
                    if output_name.startswith("present."):
                        past_name = output_name.replace("present.", "past_key_values.")
                        if ".decoder." in output_name:
                            new_past[past_name] = decoder_outputs[j]

                # Keep encoder KV, update decoder KV
                past_key_values = {**encoder_past_key_values, **new_past}

            # Greedy decoding: pick token with highest logit
            next_token = int(np.argmax(last_logits))

            if verbose:
                print(
                    f"Step {i} | ID: {next_token} | Logit: {last_logits[next_token]:.2f} | "
                    f"Text: {self.tokenizer.decode(tokens, skip_special_tokens=False)}"
                )

            # Stop at end of transcript token
            if next_token == self.eot_token:
                if verbose:
                    print("--- Reached eotToken")
                break

            tokens.append(next_token)

        transcript = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

        if verbose:
            print(f"Transcript: {transcript}")

        return transcript

    def transcribe(
        self,
        audio_path: str,
        max_output_tokens: Optional[int] = None,
        verbose: bool = False,
    ) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file (WAV, MP3, etc.).
            max_output_tokens: Maximum number of output tokens. If not provided,
                calculated dynamically based on audio duration.
            verbose: Enable verbose output during transcription.

        Returns:
            Transcribed text string.
        """
        if self.super_encoder_session is None:
            raise RuntimeError("Models not loaded. Call load() first.")

        # Load audio
        audio = load_audio(audio_path)

        # Run super encoder
        audio_features = self._run_super_encoder(audio)

        # Calculate max tokens
        audio_duration = len(audio) / self.SAMPLE_RATE
        if max_output_tokens is None:
            max_output_tokens = int(np.clip(audio_duration * self.tokens_per_second, 10, 224))

        if verbose:
            print(f"Audio: {audio_duration:.2f}s -> maxTokens: {max_output_tokens}")

        # Run decoder
        transcript = self._run_decoder(audio_features, max_output_tokens, verbose=verbose)

        # Filter out BLANK_AUDIO token
        if transcript == "[BLANK_AUDIO]":
            if verbose:
                print("BLANK_AUDIO detected - returning empty transcript")
            return ""

        return transcript


if __name__ == "__main__":
    # Example usage
    import sys

    # Parse --verbose flag
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("--verbose", "-v")]

    if len(args) < 2:
        print("Usage: python onnx_whisper_transcriber.py <model_dir> <audio_file> [--verbose]")
        print("  model_dir: Directory containing super_encoder.onnx, decoder_model.onnx, etc.")
        print("  audio_file: Path to audio file (WAV, MP3, etc.)")
        print("  --verbose, -v: Enable verbose output")
        sys.exit(1)

    model_dir = args[0]
    audio_file = args[1]

    transcriber = OnnxWhisperTranscriber(language="en", verbose=verbose)
    print("Loading model from:", model_dir)
    transcriber.load(model_dir)

    result = transcriber.transcribe(audio_file, verbose=verbose)
    print(f"\nTranscription: {result}")
