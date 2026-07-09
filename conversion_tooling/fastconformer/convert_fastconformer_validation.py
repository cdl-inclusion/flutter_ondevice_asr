"""Parity check of a built hybrid FastConformer ONNX artifact vs its original NeMo model.

Validates both heads on synthetic signals (random noise / swept sine). That's a fine
loss-free check for fp32, but it stresses int8 far harder than real speech (out-of-
distribution extremes blow up int8's activation-outlier error), so int8 can fail this test
even where its real-data WER is fine. 
"""

from __future__ import annotations

import os
import sys

_ENC_PARITY_THRESHOLD = 1e-2  # max|Δ| on encoder activations (mirrors the CTC logprob bar)


def _import_token_helpers():
    """Import the runtime detok/greedy helpers from onnx_fastconformer_transcriber, which is
    co-located with this file (same dir locally; copied to /root in the Modal image)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, "/root"):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    from onnx_fastconformer_transcriber import _load_tokens, _detok, ctc_greedy_decode
    return _load_tokens, _detok, ctc_greedy_decode


def greedy_decode(dj_sess, dj_in, enc_out, enc_len, blank_id, pred_layers, pred_hidden,
                  max_symbols: int = 10) -> list:
    """Monotonic greedy over the fused decoder_joint (standalone copy of the runtime
    loop, so validation needs no transcriber class)."""
    import numpy as np
    T = min(enc_out.shape[2], enc_len)
    h = np.zeros((pred_layers, 1, pred_hidden), np.float32)
    c = np.zeros((pred_layers, 1, pred_hidden), np.float32)
    label = blank_id
    plen = np.array([1], dtype=np.int32)
    hyp = []
    for t in range(T):
        frame = enc_out[:, :, t:t + 1]
        for _ in range(max_symbols):
            outs = dj_sess.run(None, {
                dj_in[0]: frame.astype(np.float32),
                dj_in[1]: np.array([[label]], dtype=np.int32),
                dj_in[2]: plen,
                dj_in[3]: h,
                dj_in[4]: c,
            })
            k = int(np.asarray(outs[0]).reshape(-1).argmax())
            if k == blank_id:
                break
            hyp.append(k)
            label = k
            h, c = np.asarray(outs[2]), np.asarray(outs[3])
    return hyp


def validate_hybrid(model, artifact_dir: str) -> dict:
    """Validate an ALREADY-BUILT hybrid artifact vs NeMo on synthetic signals (both heads):
      CTC   : super_encoder -> ctc_decoder (2 ONNX calls) vs NeMo end-to-end
              (preprocessor->encoder->ctc_decoder) — logprobs max|Δ| + greedy text.
      RNN-T : encoder parity (super_encoder out vs NeMo preprocessor->encoder) + our
              greedy(decoder_joint) vs NeMo greedy (token ids), off the SAME graphs.
    See the module docstring for why int8 can FAIL this synthetic gate. Returns {passed, cases}."""
    import json
    import numpy as np
    import onnxruntime as ort
    import torch

    _load_tokens, _detok, ctc_greedy_decode = _import_token_helpers()

    model.eval()
    with open(f"{artifact_dir}/meta.json") as f:
        meta = json.load(f)
    blank_id = int(meta["blank_id"])
    pred_layers = int(meta["pred_rnn_layers"])
    pred_hidden = int(meta["pred_hidden"])
    vocab = _load_tokens(f"{artifact_dir}/tokens.txt")

    se = ort.InferenceSession(f"{artifact_dir}/super_encoder.onnx", providers=["CPUExecutionProvider"])
    ctc = ort.InferenceSession(f"{artifact_dir}/ctc_decoder.onnx", providers=["CPUExecutionProvider"])
    dj = ort.InferenceSession(f"{artifact_dir}/decoder_joint.onnx", providers=["CPUExecutionProvider"])

    in_wav = meta["super_encoder_io"]["input_waveform"]
    in_len = meta["super_encoder_io"]["input_length"]
    ctc_in = meta["ctc_decoder_io"]["input_encoder"]
    dj_in = [i["name"] for i in meta["decoder_joint_io"]["inputs"]]

    # rnnt strategy for the NeMo greedy reference; ctc_decoder is a raw module (works regardless).
    model.change_decoding_strategy(decoder_type="rnnt")

    rng = np.random.RandomState(0)
    cases = {
        "noise_2s": (rng.randn(32000) * 0.1).astype(np.float32),
        "sine_1.5s": (0.3 * np.sin(
            2 * np.pi * np.cumsum(np.linspace(150, 3500, 24000)) / 16000)).astype(np.float32),
        "noise_1s": (rng.randn(16000) * 0.1).astype(np.float32),
    }

    results, passed = {}, True
    for name, sig in cases.items():
        # ONNX: shared super-encoder -> encoder_out.
        enc_out, enc_len = se.run(None, {
            in_wav: sig[None].astype(np.float32),
            in_len: np.array([len(sig)], dtype=np.int64),
        })
        enc_out = np.asarray(enc_out)
        enc_len = int(np.asarray(enc_len).reshape(-1)[0])

        # NeMo full path (preprocessor -> encoder) once; reused for both heads' refs.
        with torch.no_grad():
            feats, feat_len = model.preprocessor(
                input_signal=torch.from_numpy(sig).float().unsqueeze(0),
                length=torch.tensor([len(sig)]))
            enc_nemo_t, _ = model.encoder(audio_signal=feats, length=feat_len)
            lp_nemo = model.ctc_decoder(encoder_output=enc_nemo_t)[0].cpu().numpy()
        enc_nemo = enc_nemo_t.cpu().numpy()

        # (A) encoder parity (shared graph).
        Tn = min(enc_out.shape[2], enc_nemo.shape[2])
        enc_max_d = float(np.abs(enc_out[:, :, :Tn] - enc_nemo[:, :, :Tn]).max())

        # (B) CTC head: ONNX ctc_decoder on the ONNX encoder_out vs NeMo end-to-end.
        lp_onnx = np.asarray(ctc.run(None, {ctc_in: enc_out.astype(np.float32)})[0])[0]
        T = min(lp_onnx.shape[0], lp_nemo.shape[0])
        ctc_max_d = float(np.abs(lp_onnx[:T] - lp_nemo[:T]).max())
        ctc_txt_onnx = _detok(vocab, ctc_greedy_decode(lp_onnx, blank_id))
        ctc_txt_nemo = _detok(vocab, ctc_greedy_decode(lp_nemo, blank_id))
        ctc_text_match = ctc_txt_onnx == ctc_txt_nemo

        # (C) RNN-T head: our greedy vs NeMo greedy on the same ONNX encoder_out.
        our_ids = greedy_decode(dj, dj_in, enc_out, enc_len, blank_id, pred_layers, pred_hidden)
        with torch.no_grad():
            hyps = model.decoding.rnnt_decoder_predictions_tensor(
                torch.from_numpy(enc_out), torch.tensor([enc_len]), return_hypotheses=True)
        if isinstance(hyps, tuple):
            hyps = hyps[0]
        y = hyps[0].y_sequence
        nemo_ids = [int(t) for t in (y.tolist() if hasattr(y, "tolist") else y)]
        rnnt_ids_match = our_ids == nemo_ids

        ok = (enc_max_d < _ENC_PARITY_THRESHOLD and ctc_max_d < 1e-2
              and ctc_text_match and rnnt_ids_match)
        passed = passed and ok
        results[name] = {"ok": ok, "enc_max_abs_diff": enc_max_d,
                         "ctc_logprob_max_abs_diff": ctc_max_d, "ctc_text_match": ctc_text_match,
                         "rnnt_ids_match": rnnt_ids_match,
                         "ctc_onnx": ctc_txt_onnx[:80], "ctc_nemo": ctc_txt_nemo[:80]}
        print(f"[{'OK ' if ok else 'FAIL'}] {name:10s} enc max|Δ|={enc_max_d:.2e} "
              f"ctc lp max|Δ|={ctc_max_d:.2e} ctc_text={ctc_text_match} "
              f"rnnt_ids={rnnt_ids_match}", flush=True)
        if not ctc_text_match:
            print(f"        ctc onnx: {ctc_txt_onnx!r}\n        ctc nemo: {ctc_txt_nemo!r}")

    print(f"\n[{'PASS' if passed else 'FAIL'}] validation vs NeMo: passed={passed}")
    return {"passed": passed, "cases": results}


def print_validation_cases(cases: dict) -> None:
    for name, c in cases.items():
        print(f"  [{'OK ' if c['ok'] else 'FAIL'}] {name}  enc max|Δ|={c['enc_max_abs_diff']:.2e} "
              f"ctc lp max|Δ|={c['ctc_logprob_max_abs_diff']:.2e} "
              f"ctc_text={c['ctc_text_match']} rnnt_ids={c['rnnt_ids_match']}")
        if not c["ctc_text_match"]:
            print(f"        ctc onnx: {c['ctc_onnx']!r}\n        ctc nemo: {c['ctc_nemo']!r}")
