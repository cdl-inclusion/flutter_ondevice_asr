"""ONNX runtime for the FastConformer with CTC + RNN-T heads."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

# Shared interface + session/audio helpers (onnx_transcriber.py sits next to this file,
# both locally in conversion_tooling/ and flat-mounted in /root on Modal).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_transcriber import Transcriber, load_audio, make_session


def _load_tokens(tokens_path: Path) -> list[str]:
    """Read 'id\\tpiece' lines into an id-indexed list."""
    by_id: dict[int, str] = {}
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            idx, piece = line.split("\t", 1)
            by_id[int(idx)] = piece
    return [by_id.get(i, "") for i in range(max(by_id) + 1)]


def _detok(vocab: list[str], ids: list[int]) -> str:
    """Token ids -> text: id->piece lookup, join, U+2581 -> space. The on-device
    detok path (no SentencePiece library)."""
    return "".join(vocab[i] for i in ids).replace("▁", " ").strip()


def ctc_greedy_decode(log_probs: np.ndarray, blank_id: int) -> list[int]:
    """Greedy CTC: argmax per frame, collapse repeatitions, drop blank.

    ``log_probs``: [T, V+1]. Standard collapse keyed on the RAW previous id so
    that "a a" -> one 'a' but "a <blank> a" -> two 'a's.
    """
    ids = log_probs.argmax(axis=-1)
    out: list[int] = []
    prev = blank_id
    for t in ids:
        t = int(t)
        if t != prev and t != blank_id:
            out.append(t)
        prev = t
    return out


class OnnxFastConformerCTC(Transcriber):
    """CTC transcriber over the hybrid artifact: shared ``super_encoder`` (waveform ->
    encoder_out) then the standalone ``ctc_decoder`` (encoder_out -> logprobs), greedy
    CTC + tokens.txt detok. Raw waveform in — no featurizer."""

    def __init__(self, num_threads: int = 4, verbose: bool = False):
        self.verbose = verbose
        self._num_threads = num_threads
        self.enc: Optional[ort.InferenceSession] = None
        self.ctc: Optional[ort.InferenceSession] = None
        self.meta: dict = {}
        self.blank_id: int = 0
        self._in_wav = "waveforms"
        self._in_len = "waveforms_lens"
        self._ctc_in = "encoder_out"
        self._vocab: Optional[list[str]] = None

    def load(self, model_dir: str) -> None:
        model_dir = Path(model_dir)
        with open(model_dir / "meta.json") as f:
            self.meta = json.load(f)
        self.blank_id = int(self.meta["blank_id"])

        # Input names from the recorded IO (don't hardcode).
        se = self.meta.get("super_encoder_io", {})
        self._in_wav = se.get("input_waveform", "waveforms")
        self._in_len = se.get("input_length", "waveforms_lens")
        self._ctc_in = self.meta.get("ctc_decoder_io", {}).get("input_encoder", "encoder_out")

        se_onnx = self.meta.get("super_encoder_onnx", "super_encoder.onnx")
        ctc_onnx = self.meta.get("ctc_decoder_onnx", "ctc_decoder.onnx")
        self.enc = make_session(model_dir / se_onnx, self._num_threads)
        self.ctc = make_session(model_dir / ctc_onnx, self._num_threads)

        tokens_path = model_dir / "tokens.txt"
        if not tokens_path.exists():
            raise FileNotFoundError(f"tokens.txt missing in {model_dir} (needed for detok)")
        self._vocab = _load_tokens(tokens_path)

        if self.verbose:
            print(f"[ONNX-CTC] loaded {model_dir.name}; blank={self.blank_id}; "
                  f"vocab={len(self._vocab)}")

    def log_probs(self, signal: np.ndarray) -> np.ndarray:
        """Raw waveform -> CTC log-probs [T_enc, V+1] (single utterance)."""
        audio = np.asarray(signal, dtype=np.float32)[None]   # [1, T_samples]
        length = np.array([audio.shape[1]], dtype=np.int64)
        enc_out, _ = self.enc.run(None, {self._in_wav: audio, self._in_len: length})
        outs = self.ctc.run(None, {self._ctc_in: np.asarray(enc_out, dtype=np.float32)})
        return np.asarray(outs[0])[0]                         # [T_enc, V+1]

    def decode(self, log_probs: np.ndarray) -> str:
        return _detok(self._vocab, ctc_greedy_decode(log_probs, self.blank_id))

    def decode_ids(self, ids: list[int]) -> str:
        return _detok(self._vocab, ids)

    def transcribe(self, signal_or_path) -> str:
        if self.enc is None:
            raise RuntimeError("Call load() first.")
        signal = load_audio(signal_or_path) \
            if isinstance(signal_or_path, (str, Path)) \
            else np.asarray(signal_or_path, dtype=np.float32)
        return self.decode(self.log_probs(signal))


class OnnxFastConformerRNNT(Transcriber):
    """RNN-T transcriber over the hybrid artifact: shared ``super_encoder`` (waveform ->
    encoder_out) then a monotonic greedy loop over the fused ``decoder_joint``. Raw
    waveform in — no featurizer; detok via tokens.txt."""

    def __init__(self, num_threads: int = 4, verbose: bool = False,
                 max_symbols_per_step: int = 10):
        self.verbose = verbose
        self._num_threads = num_threads
        self.max_symbols = max_symbols_per_step
        self.enc: Optional[ort.InferenceSession] = None
        self.dj: Optional[ort.InferenceSession] = None
        self.meta: dict = {}
        self.blank_id = 0
        self.pred_hidden = 640
        self.pred_layers = 1
        self._vocab: Optional[list[str]] = None
        self._enc_in = ("waveforms", "waveforms_lens")
        self._dj_in = ("encoder_outputs", "targets", "prednet_lengths_orig",
                       "input_states_1", "input_states_2")

    def load(self, model_dir: str) -> None:
        model_dir = Path(model_dir)
        with open(model_dir / "meta.json") as f:
            self.meta = json.load(f)

        self.blank_id = int(self.meta["blank_id"])
        self.pred_hidden = int(self.meta.get("pred_hidden") or 640)
        self.pred_layers = int(self.meta.get("pred_rnn_layers") or 1)

        # Frame geometry (for streaming): samples of raw audio per encoder frame =
        # subsampling_factor × window_stride. 8 × 160 = 1280 samples = 80 ms @ 16 kHz.
        self.subsampling = int(self.meta.get("subsampling_factor") or 8)
        pp = self.meta.get("preprocessor") or {}
        self.win_stride = int(pp.get("n_window_stride") or 160)
        self.samples_per_frame = self.subsampling * self.win_stride

        se = self.meta.get("super_encoder_io", {})
        self._enc_in = (se.get("input_waveform", "waveforms"),
                        se.get("input_length", "waveforms_lens"))
        self._dj_in = tuple(i["name"] for i in self.meta["decoder_joint_io"]["inputs"])

        se_onnx = self.meta.get("super_encoder_onnx", "super_encoder.onnx")
        dj_onnx = self.meta.get("decoder_joint_onnx", "decoder_joint.onnx")
        self.enc = make_session(model_dir / se_onnx, self._num_threads)
        self.dj = make_session(model_dir / dj_onnx, self._num_threads)

        tokens_path = model_dir / "tokens.txt"
        if not tokens_path.exists():
            raise FileNotFoundError(f"tokens.txt missing in {model_dir}")
        self._vocab = _load_tokens(tokens_path)

        if self.verbose:
            print(f"[ONNX-RNNT] loaded {model_dir.name}; blank={self.blank_id}; "
                  f"pred_hidden={self.pred_hidden}; vocab={len(self._vocab)}")

    def _encode(self, signal: np.ndarray):
        """Raw waveform -> encoder embeddings via the shared super-encoder."""
        audio = np.asarray(signal, dtype=np.float32)[None]    # [1, T_samples]
        length = np.array([audio.shape[1]], dtype=np.int64)
        enc_out, enc_len = self.enc.run(
            None, {self._enc_in[0]: audio, self._enc_in[1]: length})
        return np.asarray(enc_out), int(np.asarray(enc_len).reshape(-1)[0])

    def _greedy_range(self, enc_out, start, end, h, c, label):
        """Monotonic greedy over the fused decoder_joint for encoder frames
        ``[start, end)``, **continuing from** the passed LSTM state ``(h, c)`` and last
        ``label``, and **returning** ``(emitted_ids, h, c, label)`` so a streaming caller
        can persist the state across chunks. enc_out: [1, D, T_enc]. blank advances time;
        a non-blank emits + advances label/state (capped by ``max_symbols``)."""
        plen = np.array([1], dtype=np.int32)
        hyp: list[int] = []
        for t in range(start, end):
            frame = enc_out[:, :, t:t + 1]                    # [1, D, 1]
            for _ in range(self.max_symbols):
                outs = self.dj.run(None, {
                    self._dj_in[0]: frame.astype(np.float32),
                    self._dj_in[1]: np.array([[label]], dtype=np.int32),
                    self._dj_in[2]: plen,
                    self._dj_in[3]: h,
                    self._dj_in[4]: c,
                })
                logits = np.asarray(outs[0]).reshape(-1)      # [V+1]
                k = int(logits.argmax())
                if k == self.blank_id:
                    break
                hyp.append(k)
                label = k
                h, c = np.asarray(outs[2]), np.asarray(outs[3])  # new LSTM state
        return hyp, h, c, label

    def _greedy(self, enc_out: np.ndarray, enc_len: int) -> list[int]:
        """One-shot greedy (zero initial state) over the whole encoder output. Thin wrapper
        over :meth:`_greedy_range`; SOS = blank (prednet embedding padding_idx)."""
        T = min(enc_out.shape[2], enc_len)
        h = np.zeros((self.pred_layers, 1, self.pred_hidden), np.float32)
        c = np.zeros((self.pred_layers, 1, self.pred_hidden), np.float32)
        hyp, _, _, _ = self._greedy_range(enc_out, 0, T, h, c, self.blank_id)
        return hyp

    def stream_ids(self, signal_or_path, *, chunk_s=0.8, left_context_s=1.6,
                   lookahead_s=0.0, drop_frames=None, verbose=False):
        """Streaming RNN-T decode based on chunked offline encoder + stateful decoder.

        Walk the waveform committing ``chunk_s`` of audio per step. For each step, re-run the
        (offline) super encoder over ``[left_context | chunk | lookahead]``, **drop the
        left-context frames**, decode only the committed (chunk) frames — continuing the LSTM
        ``(h, c)`` + ``label`` from the previous step — and **hold back** the lookahead frames
        (they are re-encoded/re-decoded next step, once they have real right context). This
        bounds per-step work (window, not whole buffer), keeps the decoder's internal LM
        continuous across chunks, and (via lookahead) recovers the right-context-dependent
        tokens — punctuation, some word boundaries — that a no-lookahead chunk truncates.

        ``lookahead_s`` = right context held back each step (0 = commit to the chunk edge).
        ``drop_frames`` overrides the leading-frame drop (calibration); default =
        ``round(left_present_samples / samples_per_frame)``. Returns the emitted token ids."""
        signal = load_audio(signal_or_path) \
            if isinstance(signal_or_path, (str, Path)) \
            else np.asarray(signal_or_path, dtype=np.float32)

        sr = self.SAMPLE_RATE
        chunk_n = int(round(chunk_s * sr))
        left_n = int(round(left_context_s * sr))
        look_n = int(round(lookahead_s * sr))
        spf = self.samples_per_frame

        h = np.zeros((self.pred_layers, 1, self.pred_hidden), np.float32)
        c = np.zeros((self.pred_layers, 1, self.pred_hidden), np.float32)
        label = self.blank_id
        ids: list[int] = []

        n = len(signal)
        pos = 0                                               # committed sample boundary
        while pos < n:
            commit_end = min(pos + chunk_n, n)
            win_start = max(0, pos - left_n)
            win_end = min(n, commit_end + look_n)             # include lookahead as right ctx
            window = signal[win_start:win_end]
            enc_out, enc_len = self._encode(window)
            t_win = min(enc_out.shape[2], enc_len)

            left_present = pos - win_start                    # samples of real left context
            drop = drop_frames if drop_frames is not None \
                else int(round(left_present / spf))
            drop = max(0, min(drop, t_win))

            # Commit only frames up to commit_end; the lookahead tail is re-decoded next
            # step. At the audio tail (no more lookahead available) commit everything.
            if win_end >= n:
                commit_frames = t_win
            else:
                commit_frames = min(t_win, int(round((commit_end - win_start) / spf)))
            commit_frames = max(drop, commit_frames)

            hyp, h, c, label = self._greedy_range(
                enc_out, drop, commit_frames, h, c, label)
            ids.extend(hyp)
            if verbose:
                print(f"[stream] pos={pos/sr:.2f}s win=[{win_start/sr:.2f},"
                      f"{win_end/sr:.2f}]s t_win={t_win} drop={drop} "
                      f"commit={commit_frames} emit={len(hyp)}")
            pos = commit_end
        return ids

    def transcribe_streaming(self, signal_or_path, **kw) -> str:
        """Streaming counterpart of :meth:`transcribe` — see :meth:`stream_ids`."""
        if self.enc is None:
            raise RuntimeError("Call load() first.")
        return self.decode_ids(self.stream_ids(signal_or_path, **kw))

    def decode_ids(self, ids: list[int]) -> str:
        return _detok(self._vocab, ids)

    def transcribe(self, signal_or_path) -> str:
        if self.enc is None:
            raise RuntimeError("Call load() first.")
        signal = load_audio(signal_or_path) \
            if isinstance(signal_or_path, (str, Path)) \
            else np.asarray(signal_or_path, dtype=np.float32)
        enc_out, enc_len = self._encode(signal)
        return self.decode_ids(self._greedy(enc_out, enc_len))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Transcribe an audio file with the FastConformer ONNX artifact.")
    ap.add_argument("onnx_dir",
                    help="onnx artifact dir (super_encoder + ctc_decoder + decoder_joint)")
    ap.add_argument("audio", help="path to a 16 kHz mono wav")
    ap.add_argument("--head", choices=["rnnt", "ctc"], default="rnnt",
                    help="decode head (default: rnnt)")
    args = ap.parse_args()

    cls = OnnxFastConformerRNNT if args.head == "rnnt" else OnnxFastConformerCTC
    t = cls(verbose=True)
    t.load(args.onnx_dir)
    print("Transcript:", t.transcribe(args.audio))
