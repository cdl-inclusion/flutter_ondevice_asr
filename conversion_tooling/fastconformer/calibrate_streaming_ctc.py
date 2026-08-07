#!/usr/bin/env python3
"""Calibrate the CTC streaming window (route (a)) — batch sweep of (chunk, left, lookahead).

CTC sibling of calibrate_streaming_rnnt.py. Reports streamed-vs-offline word error per setting over
one or many clips and recommends the snappiest lookahead within tolerance. CTC needs MORE lookahead
than RNN-T: it's frame-synchronous (no decoder state, only a `prev` argmax seed), so a token's CTC
spike-run straddling a commit boundary double-emits (e.g. "your your") with too little right context.
Measured on the int8 hybrid + jfk/crisp: needs ~0.8 s. int8 is the on-device target; fp32 may tolerate
less — re-run with --model <fp32_dir>.

Usage:
  python calibrate_streaming_ctc.py --model <int8|fp32 dir> \
      (--audio <wav> | --audio-dir <dir_of_wavs>) \
      [--chunks 0.8] [--lefts 1.6] [--looks 0.48,0.8,1.28] [--tol 0.05]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnx_fastconformer_transcriber import OnnxFastConformerCTC  # noqa: E402

import _calib_common as cc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="artifact dir (super_encoder + ctc_decoder)")
    ap.add_argument("--audio", help="single calibration wav")
    ap.add_argument("--audio-dir", help="directory of .wav clips (batch)")
    ap.add_argument("--chunks", default="0.8")
    ap.add_argument("--lefts", default="1.6")
    ap.add_argument("--looks", default="0.48,0.8,0.96,1.28")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max allowed streamed-vs-offline WER for the recommendation")
    args = ap.parse_args()

    wavs = cc.gather_wavs(args.audio, args.audio_dir)
    m = OnnxFastConformerCTC()
    m.load(args.model)
    chunks, lefts = cc.floats(args.chunks), cc.floats(args.lefts)
    cc.run(m, "CTC", wavs, chunks=chunks, lefts=lefts, looks=cc.floats(args.looks),
           tol=args.tol, default_chunk=chunks[0], default_left=lefts[0])


if __name__ == "__main__":
    main()
