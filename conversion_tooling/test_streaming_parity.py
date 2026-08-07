"""Phase 1 parity test: streaming RNN-T reproduces the offline one-shot decode.

Guards the calibrated streaming parameters (chunk 0.8 s / left 1.6 s / lookahead 0.48 s,
drop = round(left_present / samples_per_frame)) against regression: on the reference clip the
streamed transcript must equal the offline decode after normalization (i.e. WER-identical;
punctuation may differ — see streaming_rnnt_plan.md §6a Finding 3). Also pins the two facts
the calibration established: (1) the round-to-nearest-frame drop is exact, and (2) lookahead
is what buys back the right-context-dependent tokens.

Run:  conversion_tooling/fastconformer/venv/bin/python -m pytest conversion_tooling/test_streaming_parity.py -s
  or:  conversion_tooling/fastconformer/venv/bin/python conversion_tooling/test_streaming_parity.py
"""
import os
import re
import sys

try:
    import pytest
except ImportError:  # the tooling venv may not have pytest — __main__ runs a plain fallback
    pytest = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_fastconformer_transcriber import OnnxFastConformerRNNT  # noqa: E402
from onnx_transcriber import load_audio  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
MODEL_DIR = os.path.join(_REPO, "assets/transcribers/fastconformer/int8")
AUDIO = os.path.join(_REPO, "assets/audio/jfk_asknot.wav")

# Calibrated streaming defaults (streaming_rnnt_plan.md §6a "Decision").
CHUNK_S, LEFT_S, LOOKAHEAD_S = 0.8, 1.6, 0.48

_assets_present = os.path.exists(os.path.join(MODEL_DIR, "super_encoder.onnx")) and \
    os.path.exists(os.path.join(MODEL_DIR, "decoder_joint.onnx"))
if pytest is not None:
    pytestmark = pytest.mark.skipif(
        not _assets_present, reason=f"FastConformer RNN-T assets not present at {MODEL_DIR}")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s.lower())).strip()


def _make_model():
    t = OnnxFastConformerRNNT(verbose=False)
    t.load(MODEL_DIR)
    return t


def _offline(model, signal):
    enc_out, enc_len = model._encode(signal)
    ids = model._greedy(enc_out, enc_len)
    return ids, model.decode_ids(ids)


if pytest is not None:
    @pytest.fixture(scope="module")
    def model():
        return _make_model()

    @pytest.fixture(scope="module")
    def signal():
        return load_audio(AUDIO)

    @pytest.fixture(scope="module")
    def offline(model, signal):
        return _offline(model, signal)


def test_streaming_matches_offline_text(model, signal, offline):
    """Streaming with calibrated params == offline decode (normalized / WER-identical)."""
    _, off_txt = offline
    ids = model.stream_ids(signal, chunk_s=CHUNK_S, left_context_s=LEFT_S,
                           lookahead_s=LOOKAHEAD_S)
    stream_txt = model.decode_ids(ids)
    print(f"\noffline: {off_txt}\nstream : {stream_txt}")
    assert _norm(stream_txt) == _norm(off_txt), \
        f"streamed text diverged from offline:\n  off={off_txt!r}\n  str={stream_txt!r}"


def test_lookahead_is_necessary(model, signal, offline):
    """Sanity check on the mechanism: with no lookahead the chunk edges break (text differs),
    which is exactly what the lookahead hold-back fixes (§6a Finding 2)."""
    _, off_txt = offline
    no_look = model.decode_ids(
        model.stream_ids(signal, chunk_s=CHUNK_S, left_context_s=LEFT_S, lookahead_s=0.0))
    assert _norm(no_look) != _norm(off_txt), \
        "expected no-lookahead streaming to differ from offline (else lookahead is untested)"


def test_frame_drop_rule_is_exact(model, signal, offline):
    """The round-to-nearest-frame drop is exact: forcing drop±1 is strictly worse (§6a
    Finding 1). With the calibrated params, auto-drop matches offline while ±1 does not."""
    _, off_txt = offline
    base_drop = round(LEFT_S * model.SAMPLE_RATE / model.samples_per_frame)
    auto = _norm(model.decode_ids(model.stream_ids(
        signal, chunk_s=CHUNK_S, left_context_s=LEFT_S, lookahead_s=LOOKAHEAD_S)))
    assert auto == _norm(off_txt)
    for dd in (-1, 1):
        forced = _norm(model.decode_ids(model.stream_ids(
            signal, chunk_s=CHUNK_S, left_context_s=LEFT_S, lookahead_s=LOOKAHEAD_S,
            drop_frames=base_drop + dd)))
        assert forced != _norm(off_txt), \
            f"drop{dd:+d} unexpectedly matched offline — drop rule not as calibrated"


if __name__ == "__main__":
    if pytest is not None:
        sys.exit(pytest.main([__file__, "-s", "-v"]))
    # Plain fallback (no pytest in the venv): run the three checks directly.
    if not _assets_present:
        print(f"SKIP: assets not present at {MODEL_DIR}")
        sys.exit(0)
    _m = _make_model()
    _s = load_audio(AUDIO)
    _off = _offline(_m, _s)
    failed = 0
    for fn in (test_streaming_matches_offline_text,
               test_lookahead_is_necessary,
               test_frame_drop_rule_is_exact):
        try:
            fn(_m, _s, _off)
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{'ALL PASSED' if not failed else f'{failed} FAILED'}")
    sys.exit(1 if failed else 0)
