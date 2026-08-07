#!/usr/bin/env python3
"""Calibrate the RNN-T streaming window (route (a)) — batch sweep of (chunk, left, lookahead).

Reports streamed-vs-offline word error per setting over one or many clips and recommends the
snappiest lookahead within tolerance. See _calib_common.run for the metric/goal. The Dart port
mirrors this decode (test/fastconformer_rnnt_streaming_test.dart); trust the Dart sweeps for the
shipped `_kRnnt*` defaults, this tool for exploration.

Usage:
  python calibrate_streaming_rnnt.py --model <int8|fp32 dir> \
      (--audio <wav> | --audio-dir <dir_of_wavs>) \
      [--chunks 0.8] [--lefts 1.6] [--looks 0.32,0.48,0.64] [--tol 0.05]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnx_fastconformer_transcriber import OnnxFastConformerRNNT  # noqa: E402

import _calib_common as cc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="artifact dir (super_encoder + decoder_joint)")
    ap.add_argument("--audio", help="single calibration wav")
    ap.add_argument("--audio-dir", help="directory of .wav clips (batch)")
    ap.add_argument("--chunks", default="0.8")
    ap.add_argument("--lefts", default="1.6")
    ap.add_argument("--looks", default="0.0,0.32,0.48,0.64")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max allowed streamed-vs-offline WER for the recommendation")
    args = ap.parse_args()

    wavs = cc.gather_wavs(args.audio, args.audio_dir)
    m = OnnxFastConformerRNNT()
    m.load(args.model)
    chunks, lefts = cc.floats(args.chunks), cc.floats(args.lefts)
    cc.run(m, "RNN-T", wavs, chunks=chunks, lefts=lefts, looks=cc.floats(args.looks),
           tol=args.tol, default_chunk=chunks[0], default_left=lefts[0])


if __name__ == "__main__":
    main()
