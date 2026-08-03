"""Conversion logic for converting a NeMo hybrid FastConformer to ONNX (both CTC and RNN-T heads).

Preprocessor is converted to ONNX and fused into super encoder graph.
Also creates tokens.txt in replacement of the NeMo tokenizer.

Creates fp32 and int8 quantized versions.

We se NeMo's NATIVE export via ``set_export_config(
{'decoder_type': 'ctc'})`` rather than hand-wrapping the encoder. NeMo's
``forward_for_export`` handles the Conformer's masking / relative-position code
that does not trace cleanly otherwise. We introspect the produced graph for IO
names instead of hardcoding them.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

# Shared opset/IR constants.
from ..onnx_conversion_constants import ONNX_IR_VERSION, ONNX_OPSET

def _preprocessor_meta(model) -> dict:
    """Subset of cfg.preprocessor recorded in meta.json (provenance + the window stride
    the runtime uses to convert encoder frame indices to word timestamps)."""
    pp = dict(model.cfg.preprocessor)
    sr = int(pp.get("sample_rate", 16000))

    def _samples(key_size, key_n, default_s):
        if pp.get(key_size) is not None:
            return int(round(float(pp[key_size]) * sr))
        return int(pp.get(key_n, round(default_s * sr)))

    lzg = pp.get("log_zero_guard_value", 2.0 ** -24)
    if not isinstance(lzg, (int, float)):
        lzg = 2.0 ** -24
    return {
        "sample_rate": sr,
        "n_window_size": _samples("window_size", "n_window_size", 0.025),
        "n_window_stride": _samples("window_stride", "n_window_stride", 0.01),
        "n_fft": int(pp.get("n_fft") or 512),
        "features": int(pp.get("features", 80)),
        "preemph": pp.get("preemph", 0.97),
        "lowfreq": float(pp.get("lowfreq", 0.0) or 0.0),
        "highfreq": pp.get("highfreq", None),
        "mag_power": float(pp.get("mag_power", 2.0)),
        "normalize": pp.get("normalize", "per_feature"),
        "log_zero_guard_value": float(lzg),
    }


def _write_tokens(model, out_dir: Path) -> int:
    """Write the id->piece vocab to tokens.txt — the only tokenizer artifact the
    runtime needs (detok = lookup + join, no SentencePiece). Returns vocab_size
    (no blank). Per-language by construction: reads this model's own vocab."""
    vocab = list(model.tokenizer.vocab)  # id-ordered pieces (NeMo tokenizer wrapper)
    with open(out_dir / "tokens.txt", "w", encoding="utf-8") as f:
        for i, piece in enumerate(vocab):
            f.write(f"{i}\t{piece}\n")
    return len(vocab)


def _onnx_io_names(onnx_path: Path) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ins = [i.name for i in sess.get_inputs()]
    outs = [o.name for o in sess.get_outputs()]
    return {"inputs": ins, "outputs": outs}


def _onnx_io_detail(onnx_path: Path) -> dict:
    """Full IO: name + shape + type for each input/output (drives the decode loop)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def fmt(xs):
        return [{"name": x.name, "shape": list(x.shape), "type": x.type} for x in xs]

    return {"inputs": fmt(sess.get_inputs()), "outputs": fmt(sess.get_outputs())}


def _pin_opset_and_ir(onnx_path: Path, opset: int, ir_version: int) -> dict:
    """Pin IR version (and verify opset) for runtime compatibility."""
    import onnx
    m = onnx.load(str(onnx_path))
    actual_opset = max((i.version for i in m.opset_import if i.domain in ("", "ai.onnx")),
                       default=None)
    changed = False
    if m.ir_version != ir_version:
        m.ir_version = ir_version
        changed = True
    if changed:
        onnx.save(m, str(onnx_path))
    print(f"[CONVERT] opset={actual_opset} (target {opset}) ir_version={ir_version}")
    return {"opset": actual_opset, "ir_version": ir_version}


def _export_preprocessor_onnx(model, out_path: str, *,
                              opset: int = ONNX_OPSET,
                              ir_version: int = ONNX_IR_VERSION) -> dict:
    """Export NeMo's mel-spectrogram preprocessor (waveform -> log-mel features) to ONNX.

    This is the "super-encoder front end": baking NeMo's own
    ``AudioToMelSpectrogramPreprocessor`` into the graph so the on-device runtime can
    feed a RAW WAVEFORM (like Whisper's super_encoder) instead of needing a separate
    featurizer. Feature parity is exact by construction — the graph IS NeMo's
    preprocessor — which sidesteps the numpy-featurizer port + parity risk on Dart.

    IO: ``waveforms`` [B, T_samples] f32 + ``waveforms_lens`` [B] i64
        -> ``features`` [B, n_mels, T_frames] f32 + ``features_lens`` [B] i64
    """
    import torch
    import nemo.collections.asr.parts.preprocessing.features as nemo_features

    pp = model.preprocessor
    # Deterministic inference front end (match the numpy featurizer / eval): no dither,
    # no pad_to so T lines up exactly.
    pp.featurizer.dither = 0.0
    pp.featurizer.pad_to = 0

    # We must use the DYNAMO onnx exporter: only it traces NeMo's complex STFT
    # (`torch.stft(return_complex=True)`) — the legacy TorchScript exporter raises
    # "STFT does not currently support complex types". But dynamo can't trace NeMo's
    # `normalize_batch` (`torch.any(seq_len == 1).item()` is a data-dependent Python
    # bool on the dynamic length). Fix: temporarily replace normalize_batch with an
    # export-safe per-feature normalization — mathematically identical for our case
    # (batch=1, full length, no padding): per (batch, mel) mean/std over time, unbiased
    # std, +1e-5 eps, exactly NeMo's per_feature path minus the seq_len masking.
    _orig_normalize = nemo_features.normalize_batch

    def _export_safe_normalize_batch(x, seq_len, normalize_type, *args, **kwargs):
        if normalize_type == "per_feature":
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, unbiased=True, keepdim=True)
            return (x - mean) / (std + 1e-5), mean, std
        return _orig_normalize(x, seq_len, normalize_type, *args, **kwargs)

    class _PreprocessorWrapper(torch.nn.Module):
        def __init__(self, preprocessor):
            super().__init__()
            self.preprocessor = preprocessor

        def forward(self, waveforms, waveforms_lens):
            feats, feat_len = self.preprocessor(
                input_signal=waveforms, length=waveforms_lens)
            return feats, feat_len

    wrapper = _PreprocessorWrapper(pp).eval()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Export on CPU. In the eval the model is on GPU, but torch.stft requires the input
    # and the window buffer on the SAME device — and our dummy/runtime inputs are CPU.
    # Move the preprocessor to CPU for the export, then restore its original device.
    _buf = next(pp.buffers(), None)
    orig_device = _buf.device if _buf is not None else torch.device("cpu")
    pp.to("cpu")

    # Dummy = 1 s of audio; dynamic time axes so any-length input works at runtime.
    dummy_signal = torch.randn(1, 16000, dtype=torch.float32)
    dummy_length = torch.tensor([16000], dtype=torch.int64)

    print(f"[CONVERT] exporting preprocessor (waveform -> log-mel) -> {out}")
    nemo_features.normalize_batch = _export_safe_normalize_batch
    try:
        torch.onnx.export(
            wrapper,
            (dummy_signal, dummy_length),
            str(out),
            input_names=["waveforms", "waveforms_lens"],
            output_names=["features", "features_lens"],
            dynamic_axes={
                "waveforms": {0: "B", 1: "T_samples"},
                "waveforms_lens": {0: "B"},
                "features": {0: "B", 2: "T_frames"},
                "features_lens": {0: "B"},
            },
            opset_version=opset,
            dynamo=True,
        )
    finally:
        nemo_features.normalize_batch = _orig_normalize
        pp.to(orig_device)
    return _pin_opset_and_ir(out, opset, ir_version)


def _export_transducer_graphs(model, out_dir: str, *,
                              opset: int = ONNX_OPSET,
                              ir_version: int = ONNX_IR_VERSION) -> dict:
    """NeMo native transducer export -> a standalone ``encoder-*.onnx`` + a fused
    ``decoder_joint-*.onnx`` (prediction-net + joint; one greedy-loop step) in out_dir.

    Returns the two graph paths, their introspected IO (names + shapes, so the runtime
    binds without hardcoding), and the prednet/dims the hybrid meta needs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model.change_decoding_strategy(decoder_type="rnnt")
    try:
        model.set_export_config({"decoder_type": "rnnt"})
    except Exception as e:  # noqa: BLE001
        print(f"[CONVERT] set_export_config not available ({e}); relying on cur_decoder")

    print(f"[CONVERT] exporting RNN-T graphs -> {out}")
    model.export(str(out / "model.onnx"), onnx_opset_version=opset)

    # NeMo names the transducer subnets encoder-<base> / decoder_joint-<base>.
    enc_path = next((p for p in out.glob("encoder-*.onnx")), None)
    dj_path = next((p for p in out.glob("decoder_joint-*.onnx")), None)
    if enc_path is None or dj_path is None:
        raise RuntimeError(
            f"Expected encoder-*.onnx + decoder_joint-*.onnx; dir has "
            f"{[p.name for p in out.iterdir()]}")

    _pin_opset_and_ir(enc_path, opset, ir_version)
    _pin_opset_and_ir(dj_path, opset, ir_version)

    def _cfg(path, default=None):
        node = model.cfg
        for k in path.split("."):
            node = getattr(node, k, None) if node is not None else None
        return node if node is not None else default

    vocab_size = len(model.tokenizer.vocab)
    blank_id = getattr(model.joint, "num_classes_with_blank", vocab_size + 1) - 1
    return {
        "encoder_path": enc_path,
        "decoder_joint_path": dj_path,
        "encoder_io": _onnx_io_detail(enc_path),
        "decoder_joint_io": _onnx_io_detail(dj_path),
        "blank_id": int(blank_id),
        "vocab_size": int(vocab_size),
        "d_model": int(getattr(model.cfg.encoder, "d_model", 512)),
        "subsampling_factor": int(getattr(model.cfg.encoder, "subsampling_factor", 8)),
        "pred_hidden": _cfg("decoder.prednet.pred_hidden"),
        "pred_rnn_layers": _cfg("decoder.prednet.pred_rnn_layers"),
    }


def _export_ctc_decoder_onnx(model, out_path: str, *, d_model: int = None,
                             opset: int = ONNX_OPSET,
                             ir_version: int = ONNX_IR_VERSION) -> dict:
    """Export ONLY the CTC head (``encoder_out -> logprobs``) as a standalone graph.

    NeMo's ``model.export()`` only emits ENCODER-inclusive graphs; for the shared-encoder
    (sherpa-style) layout we want the CTC head by itself so it can sit on top of the same
    encoder the RNN-T head uses. ``model.ctc_decoder`` is a ``ConvASRDecoder`` — a
    ``Conv1d(d_model -> V+1, kernel_size=1)`` + ``log_softmax`` — with no masking /
    relative-position code, so it traces cleanly with the STANDARD exporter (unlike the
    encoder, which must go through ``model.export()``, and unlike the preprocessor's
    complex STFT, which needs dynamo).

    IO: ``encoder_out`` [B, d_model, T_enc] f32 -> ``logprobs`` [B, T_enc, V+1] f32.
    """
    import torch

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    d = int(d_model or getattr(model.cfg.encoder, "d_model", 512))

    class _CtcHead(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head

        def forward(self, encoder_out):                    # [B, D, T]
            return self.head(encoder_output=encoder_out)   # [B, T, V+1] log-probs

    wrapper = _CtcHead(model.ctc_decoder).eval()

    # Export on CPU (the dummy input is CPU); restore the head's device afterwards.
    _p = next(model.ctc_decoder.parameters(), None)
    orig_device = _p.device if _p is not None else torch.device("cpu")
    model.ctc_decoder.to("cpu")

    dummy = torch.randn(1, d, 137, dtype=torch.float32)  # [B, D, T]; dynamic B/T below
    print(f"[CONVERT] exporting standalone ctc_decoder (encoder_out -> logprobs) -> {out}")
    try:
        torch.onnx.export(
            wrapper, (dummy,), str(out),
            input_names=["encoder_out"], output_names=["logprobs"],
            dynamic_axes={"encoder_out": {0: "B", 2: "T"}, "logprobs": {0: "B", 1: "T"}},
            opset_version=opset,
        )
    finally:
        model.ctc_decoder.to(orig_device)
    return _pin_opset_and_ir(out, opset, ir_version)


def _export_graphs(model, out_dir: str, *, opset: int = ONNX_OPSET,
                   ir_version: int = ONNX_IR_VERSION) -> dict:
    """Export ALL the raw graphs:
      - the preprocessor (waveform -> features)               [via dynamo exporter]
      - the transducer encoder + fused rnnt decoder_joint     [via NeMo model.export()]
      - the standalone CTC head (encoder_out -> logprobs)

    """
    work = Path(out_dir)
    work.mkdir(parents=True, exist_ok=True)

    preproc_path = work / "preprocessor.onnx"
    _export_preprocessor_onnx(model, str(preproc_path), opset=opset, ir_version=ir_version)

    tg = _export_transducer_graphs(model, str(work), opset=opset, ir_version=ir_version)

    ctc_path = work / "ctc_decoder.onnx"
    _export_ctc_decoder_onnx(model, str(ctc_path), d_model=tg["d_model"],
                             opset=opset, ir_version=ir_version)

    return {
        "preproc_path": preproc_path,
        "encoder_path": tg["encoder_path"],
        "decoder_joint_path": tg["decoder_joint_path"],
        "ctc_path": ctc_path,
        "encoder_io": tg["encoder_io"],
        "decoder_joint_io": tg["decoder_joint_io"],
        "ctc_io": _onnx_io_detail(ctc_path),
        "blank_id": tg["blank_id"],
        "vocab_size": tg["vocab_size"],
        "d_model": tg["d_model"],
        "subsampling_factor": tg["subsampling_factor"],
        "pred_hidden": tg["pred_hidden"],
        "pred_rnn_layers": tg["pred_rnn_layers"],
        "preprocessor": _preprocessor_meta(model),
    }


# Encoder subgraph is given this name prefix during super-encoder assembly
# (see convert()), so every node in super_encoder.onnx wiithout this prefix belongs
# to the merged preprocessor. quantize() uses that to keep the mel front end fp32.
ENC_PREFIX = "enc/"


def convert(model, out_dir: str, *, language: str, base_model: str,
                                opset: int = ONNX_OPSET,
                                ir_version: int = ONNX_IR_VERSION) -> dict:
    """Create single onnx artifact directory for both heads.

    The sherpa-style split (encoder / ctc_decoder / decoder_joint): one
    ``waveform -> encoder_out`` graph feeds a standalone CTC head AND the RNN-T
    decoder_joint, so the ~108M-param encoder is exported/stored ONCE instead of being
    duplicated per head. Writes a self-contained dir:
      super_encoder.onnx  waveform -> encoder_out (preproc + encoder, merged)
      ctc_decoder.onnx    encoder_out -> logprobs (standalone CTC head)
      decoder_joint.onnx  fused prediction-net + joint (RNN-T, one greedy step)
      tokens.txt, meta.json

    """
    import onnx
    from onnx import compose

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Export every raw graph (preprocessor + transducer encoder/decoder_joint + CTC head)
    # into a scratch dir; assemble the final artifact from them below.
    work = out / "_graphs"
    g = _export_graphs(model, str(work), opset=opset, ir_version=ir_version)

    # Merge preprocessor + encoder -> super_encoder.onnx (waveform -> encoder_out). We
    # merge rather than re-trace because NeMo's encoder only exports cleanly via its own
    # model.export(); the encoder graph is prefixed `enc/` to avoid name collisions.
    pre = onnx.load(str(g["preproc_path"]))
    enc = compose.add_prefix(onnx.load(str(g["encoder_path"])), prefix=ENC_PREFIX)
    in_audio, in_len = (g["encoder_io"]["inputs"][0]["name"],
                        g["encoder_io"]["inputs"][1]["name"])
    merged = compose.merge_models(
        pre, enc,
        io_map=[("features", f"{ENC_PREFIX}{in_audio}"),
                ("features_lens", f"{ENC_PREFIX}{in_len}")])
    super_path = out / "super_encoder.onnx"
    onnx.save(merged, str(super_path))
    onnx_versions = _pin_opset_and_ir(super_path, opset, ir_version)

    # Copy the CTC head + fused decoder_joint through unchanged.
    shutil.copy2(g["ctc_path"], out / "ctc_decoder.onnx")
    shutil.copy2(g["decoder_joint_path"], out / "decoder_joint.onnx")

    # Self-contained artifact: tokens + one meta describing all three graphs.
    _write_tokens(model, out)
    se_io = _onnx_io_detail(super_path)
    se = {"inputs": [i["name"] for i in se_io["inputs"]],
          "outputs": [o["name"] for o in se_io["outputs"]]}
    ctc = {"inputs": [i["name"] for i in g["ctc_io"]["inputs"]],
           "outputs": [o["name"] for o in g["ctc_io"]["outputs"]]}
    # CTC head blank (aux head); should equal the RNN-T blank (shared vocab).
    ctc_blank = getattr(model.ctc_decoder, "num_classes_with_blank",
                        g["vocab_size"] + 1) - 1
    if int(ctc_blank) != int(g["blank_id"]):
        print(f"[CONVERT] (warn) CTC blank {ctc_blank} != RNN-T blank "
              f"{g['blank_id']} — heads have different vocabs?")

    meta = {
        "kind": "hybrid",
        "heads": ["ctc", "rnnt"],
        "language": language,
        "base_model": base_model,
        "onnx_opset": onnx_versions["opset"],
        "onnx_ir_version": onnx_versions["ir_version"],
        "blank_id": int(ctc_blank),
        "vocab_size": g["vocab_size"],
        "subsampling_factor": g["subsampling_factor"],
        "d_model": g["d_model"],
        "pred_hidden": g["pred_hidden"],
        "pred_rnn_layers": g["pred_rnn_layers"],
        "super_encoder_onnx": "super_encoder.onnx",
        "ctc_decoder_onnx": "ctc_decoder.onnx",
        "decoder_joint_onnx": "decoder_joint.onnx",
        # waveform -> encoder_out (read positionally at runtime).
        "super_encoder_io": {
            "input_waveform": se["inputs"][0],
            "input_length": se["inputs"][1],
            "output_encoder": se["outputs"][0],
            "output_length": se["outputs"][1],
            "all_inputs": se["inputs"],
            "all_outputs": se["outputs"],
        },
        # encoder_out -> logprobs.
        "ctc_decoder_io": {
            "input_encoder": ctc["inputs"][0],
            "output_logprobs": ctc["outputs"][0],
            "all_inputs": ctc["inputs"],
            "all_outputs": ctc["outputs"],
        },
        # Full IO detail drives the RNN-T greedy loop / state allocation.
        "decoder_joint_io": g["decoder_joint_io"],
        "preprocessor": g["preprocessor"],
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Drop the scratch graphs so the dir is a clean device artifact.
    shutil.rmtree(work, ignore_errors=True)

    print(f"[CONVERT] HYBRID super-encoder done: super_io={se} ctc_io={ctc} "
          f"blank={meta['blank_id']} vocab={meta['vocab_size']}")
    return meta


def quantize(fp32_dir: str, int8_dir: str, *,
             quantize_ctc_decoder: bool = False,
             quantize_decoder_joint: bool = False,
             per_channel: bool = False,
             ir_version: int = ONNX_IR_VERSION) -> dict:
    """INT8 (dynamic) quantize a hybrid artifact into a sibling dir. The big shared
    ``super_encoder`` needs ``sanitize_opset`` (duplicate ai.onnx imports from the dynamo
    preprocessor). Only the ~108M-param ``super_encoder`` is int8 by default; the two decode
    heads are kept fp32 as this has shown to lead to best results (no WER regression) with 
    minimal latency/size cost:
    The merged mel preprocessor is always kept fp32 (encoder still int8); see ``_quantize_graph``."""
    src, dst = Path(fp32_dir), Path(int8_dir)
    # Fresh output dir (clear stale files from a previous quantize).
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    with open(src / "meta.json") as f:
        meta = json.load(f)

    se_info = _quantize_graph(src / "super_encoder.onnx", dst / "super_encoder.onnx",
                              ir_version, per_channel=per_channel, sanitize_opset=True)

    if quantize_ctc_decoder:
        ctc_info = _quantize_graph(src / "ctc_decoder.onnx", dst / "ctc_decoder.onnx",
                                   ir_version, per_channel=per_channel)
    else:
        shutil.copy2(src / "ctc_decoder.onnx", dst / "ctc_decoder.onnx")
        ctc_info = {"int8_mb": (src / "ctc_decoder.onnx").stat().st_size / 1e6}
        print("[QUANT] ctc_decoder.onnx: kept fp32 (quantize_ctc_decoder=False)")

    if quantize_decoder_joint:
        dj_info = _quantize_graph(src / "decoder_joint.onnx", dst / "decoder_joint.onnx",
                                  ir_version, per_channel=per_channel)
    else:
        shutil.copy2(src / "decoder_joint.onnx", dst / "decoder_joint.onnx")
        dj_info = {"int8_mb": (src / "decoder_joint.onnx").stat().st_size / 1e6}
        print("[QUANT] decoder_joint.onnx: kept fp32 (quantize_decoder_joint=False)")

    shutil.copy2(src / "tokens.txt", dst / "tokens.txt")
    meta["quantization"] = "int8"
    meta["ctc_decoder_quantized"] = quantize_ctc_decoder
    meta["decoder_joint_quantized"] = quantize_decoder_joint
    meta["preproc_fp32"] = True  # mel front end always kept fp32 (see _quantize_graph)
    meta["per_channel"] = per_channel
    meta["fp32_dir"] = str(src)
    meta["int8_size_mb"] = round(
        se_info["int8_mb"] + ctc_info["int8_mb"] + dj_info["int8_mb"], 2)
    with open(dst / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[QUANT] hybrid super-encoder int8 done -> {dst} "
          f"({meta['int8_size_mb']:.1f} MB)")
    return meta



def _sanitize_default_opset(model, target_opset: int = ONNX_OPSET) -> int:
    """Collapse the model's default (``ai.onnx``) opset imports to EXACTLY ONE entry.

    onnxruntime's dynamic quantizer (``get_opset_version``) requires exactly one opset
    import with an empty / ``"ai.onnx"`` domain and raises "Failed to find proper
    ai.onnx domain" otherwise. The MERGED super-encoder (dynamo-exported preprocessor
    + ``onnx.compose.merge_models``) can end up with duplicate default-domain entries,
    which trips it. We rewrite the default domain to a single entry (highest version
    seen, or ``target_opset``) and preserve all non-default (function/onnxscript)
    domains unchanged. Returns the chosen default version."""
    defaults = [op for op in model.opset_import if op.domain in ("", "ai.onnx")]
    others = [op for op in model.opset_import if op.domain not in ("", "ai.onnx")]
    version = max((op.version for op in defaults), default=target_opset)
    del model.opset_import[:]
    keep = model.opset_import.add()
    keep.domain, keep.version = "", version
    for op in others:
        nxt = model.opset_import.add()
        nxt.domain, nxt.version = op.domain, op.version
    return version


def _quantize_graph(src: Path, dst: Path, ir_version: int,
                    per_channel: bool = False, sanitize_opset: bool = False) -> dict:
    """Dynamic-quantize one ONNX graph (QInt8 weights) and re-pin IR version.

    ``per_channel`` gives each weight output channel its own int8 scale (vs one scale
    per tensor). ``sanitize_opset`` normalizes the default opset import before
    quantizing (needed for the merged super-encoder — see ``_sanitize_default_opset``).

    The merged mel preprocessor is always kept fp32, even if other parts are quantized.
    In the super-encoder the encoder subgraph is prefixed ``ENC_PREFIX``, so if any 
    enc-prefixed nodes are present this graph is the super-encoder and every node without
     that prefix (the preprocessor) is excluded from quantization. Graphs with no
    enc-prefixed nodes (``ctc_decoder``, ``decoder_joint``) are quantized in full.
"""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import onnx

    dst.parent.mkdir(parents=True, exist_ok=True)

    m = onnx.load(str(src))

    quant_src = src
    if sanitize_opset:
        before = [(op.domain or "ai.onnx", op.version) for op in m.opset_import]
        _sanitize_default_opset(m)
        after = [(op.domain or "ai.onnx", op.version) for op in m.opset_import]
        quant_src = src.parent / f"_sanitized_{src.name}"
        onnx.save(m, str(quant_src))
        print(f"[QUANT] sanitized opset imports for {src.name}: {before} -> {after}")

    # Always keep the merged mel preprocessor fp32. 
    exclude_nodes = []
    if any(n.name.startswith(ENC_PREFIX) for n in m.graph.node if n.name):
        exclude_nodes = [n.name for n in m.graph.node
                         if n.name and not n.name.startswith(ENC_PREFIX)]
        print(f"[QUANT] {src.name}: super-encoder detected — keeping {len(exclude_nodes)} "
              f"preprocessor node(s) fp32 (non-'{ENC_PREFIX}')")

    # Quantize ONLY MatMul (-> MatMulInteger). We must NOT quantize Conv: it becomes
    # ConvInteger, which the mobile/on-device ONNX Runtime build has no kernel for
    # ("Could not find an implementation for ConvInteger"), so the model won't load on
    # device. This mirrors the Whisper converter (op_types_to_quantize=['MatMul']). The
    # encoder's Conv2d subsampling therefore stays fp32.
    quantize_dynamic(str(quant_src), str(dst), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"], per_channel=per_channel,
                     nodes_to_exclude=exclude_nodes)
    if quant_src != src:
        quant_src.unlink(missing_ok=True)

    # quantize_dynamic can bump ir_version, so re-pin to IR 8.
    m = onnx.load(str(dst))
    if m.ir_version != ir_version:
        m.ir_version = ir_version
        onnx.save(m, str(dst))

    info = {"src_mb": src.stat().st_size / 1e6, "int8_mb": dst.stat().st_size / 1e6}
    print(f"[QUANT] {src.name}: {info['src_mb']:.1f}MB -> {info['int8_mb']:.1f}MB "
          f"({info['src_mb'] / max(info['int8_mb'], 1e-9):.1f}x)")
    return info
