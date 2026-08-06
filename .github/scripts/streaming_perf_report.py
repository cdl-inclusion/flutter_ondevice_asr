#!/usr/bin/env python3
"""Merge streaming-perf JSONs into a phone x mode report that answers: (1) is incremental worth it
vs naive, (2) which phones run each mode in real time.

Accepts either layout — pass the directory (not a .json file):
  * CI (workflow):   <root>/<device>/streaming_perf_<mode>.json   -> pass <root>
  * local runner:    <dir>/streaming_perf_<mode>.json (flat)      -> pass <dir> (device = its name)

Usage:  python streaming_perf_report.py <dir>
"""
import glob
import json
import os
import sys

MODES = ["whisper_naive", "fc_naive", "fc_incremental"]  # column order

# ---------------------------------------------------------------------------
# REALTIME VERDICT POLICY (edit here — re-run the report, no device re-run needed)
# ---------------------------------------------------------------------------
# A mode is "realtime" on a device iff BOTH hold:
#   (1) max single-partial latency < the partial cadence (minPartialMs, per-summary), and
#   (2) RTF <= MAX_RTF  — i.e. it spends at most (1 - RTF_HEADROOM) of real time computing,
#       leaving RTF_HEADROOM of the CPU for VAD, audio capture, UI, and thermal throttling.
RTF_HEADROOM = 0.30                       # reserve 30% of real-time CPU budget
MAX_RTF = round(1.0 - RTF_HEADROOM, 2)    # => require RTF <= 0.70
POLICY = (f"REALTIME = max-partial < partial-cadence AND RTF <= {MAX_RTF:.2f} "
          f"({int(RTF_HEADROOM * 100)}% CPU headroom)")


def is_realtime(s):
    return s["partialMaxMs"] < s["minPartialMs"] and s["rtf"] <= MAX_RTF


def load(root):
    root = os.path.normpath(root)
    # Prefer per-device subdirs (CI); fall back to a flat dir of JSONs (local runner).
    paths = glob.glob(os.path.join(root, "*", "streaming_perf_*.json"))
    flat = not paths
    if flat:
        paths = glob.glob(os.path.join(root, "streaming_perf_*.json"))
    devices = {}  # device -> {mode -> summary}
    for path in paths:
        device = os.path.basename(root) if flat \
            else os.path.basename(os.path.dirname(path))
        with open(path) as f:
            s = json.load(f)
        devices.setdefault(device, {})[s["mode"]] = s
    return devices


def crossover(naive, incr):
    """First buffer-second where incremental's per-partial <= naive's (aligned by index)."""
    a, b = naive.get("perPartial", []), incr.get("perPartial", [])
    rows, flip = [], None
    for (sn, mn), (si, mi) in zip(a, b):
        rows.append((sn, mn, mi))
        # A real committed incremental partial (>=1ms; the ~0ms entries are no-ops where no new
        # chunk committed yet — always cheaper) that is at or below naive's cost.
        if flip is None and mi >= 1.0 and mi <= mn:
            flip = sn
    return rows, flip


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    devices = load(sys.argv[1])
    if not devices:
        sys.exit(f"no streaming_perf_*.json found under {sys.argv[1]} "
                 "(pass the DIRECTORY containing the JSONs, not a .json file)")

    print(f"Policy: {POLICY}")

    for device in sorted(devices):
        modes = devices[device]
        print(f"\n{'='*72}\nDEVICE: {device}\n{'='*72}")
        print(f"{'mode':<16}{'RTF':>6}{'pMean':>7}{'pP90':>7}{'pMax':>7}{'fMax':>7}"
              f"{'load':>7}  realtime")
        for m in MODES:
            s = modes.get(m)
            if not s:
                continue
            print(f"{m:<16}{s['rtf']:>6.2f}{s['partialMeanMs']:>7.0f}{s['partialP90Ms']:>7.0f}"
                  f"{s['partialMaxMs']:>7.0f}{s['finalMaxMs']:>7.0f}{s['loadMs']:>6}ms  "
                  f"{'YES' if is_realtime(s) else 'NO'}")

        # (1) incremental worth it?
        nv, inc = modes.get("fc_naive"), modes.get("fc_incremental")
        if nv and inc:
            rtf_x = nv["rtf"] / inc["rtf"] if inc["rtf"] else float("inf")
            pmax_x = nv["partialMaxMs"] / inc["partialMaxMs"] if inc["partialMaxMs"] else float("inf")
            print(f"\n  FC incremental vs naive: RTF {inc['rtf']:.2f} vs {nv['rtf']:.2f} "
                  f"({rtf_x:.1f}x); max-partial {inc['partialMaxMs']:.0f} vs "
                  f"{nv['partialMaxMs']:.0f}ms ({pmax_x:.1f}x). "
                  f"final (both full re-decode): {inc['finalMaxMs']:.0f}/{nv['finalMaxMs']:.0f}ms")
            rows, flip = crossover(nv, inc)
            print("  per-partial crossover (bufferSec: naive_ms / incr_ms):")
            print("   " + "  ".join(f"{s:.1f}:{mn:.0f}/{mi:.0f}" for s, mn, mi in rows))
            print(f"  -> incremental first matches/beats naive at ~{flip:.1f}s buffer"
                  if flip is not None else "  -> no crossover in range")

    # (2) realtime grid
    print(f"\n{'='*72}\nREALTIME GRID  ({POLICY})\n{'='*72}")
    print(f"{'device':<16}" + "".join(f"{m:>16}" for m in MODES))
    for device in sorted(devices):
        cells = []
        for m in MODES:
            s = devices[device].get(m)
            cells.append("—" if not s else ("YES" if is_realtime(s) else "NO"))
        print(f"{device:<16}" + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
