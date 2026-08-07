#!/usr/bin/env python3
"""Merge streaming-perf JSONs into a phone x mode report that answers: (1) is incremental worth it
vs naive, (2) which phones run each mode responsively.

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

# Responsiveness budgets (a UX call — tune to taste). We chose 1000/1000 ms as realistic for a user:
# ~1s is about the limit for live text to still feel snappy. RTF is CPU load, not a gate.
FINAL_BUDGET_MS = 1000      # last word -> committed text (adds on top of the ~1000ms eosMinSilence)
REFRESH_BUDGET_MS = 1000    # interval between live partial updates
POLICY = (f"RESPONSIVE = final < {FINAL_BUDGET_MS} ms AND partial-refresh < {REFRESH_BUDGET_MS} ms "
          f"(RTF = CPU load, not a gate)")
ASSUMPTIONS = [
    "final = last word -> committed text (adds on top of the ~1000ms eosMinSilence already waited).",
    "refresh = interval between live partial updates.",
    "RTF/CPUload is informational (>1 = busy CPU, not lagging — the guard self-throttles).",
]


def responsive(s):
    return s.get("finalMaxMs", 9e9) < FINAL_BUDGET_MS and \
        s.get("partialRefreshMs", 9e9) < REFRESH_BUDGET_MS


def load(root):
    root = os.path.normpath(root)
    paths = glob.glob(os.path.join(root, "*", "streaming_perf_*.json"))
    flat = not paths
    if flat:
        paths = glob.glob(os.path.join(root, "streaming_perf_*.json"))
    devices = {}
    for path in paths:
        device = os.path.basename(root) if flat else os.path.basename(os.path.dirname(path))
        with open(path) as f:
            s = json.load(f)
        devices.setdefault(device, {})[s["mode"]] = s
    return devices


def crossover(naive, incr):
    """First buffer-second where a real committed incremental partial (>=1ms) is <= naive's."""
    a, b = naive.get("perPartial", []), incr.get("perPartial", [])
    rows, flip = [], None
    for (sn, mn), (si, mi) in zip(a, b):
        rows.append((sn, mn, mi))
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
    print("Assumptions:")
    for a in ASSUMPTIONS:
        print(f"  - {a}")

    for device in sorted(devices):
        modes = devices[device]
        print(f"\n{'='*80}\nDEVICE: {device}\n{'='*80}")
        print(f"{'mode':<16}{'final':>7}{'refresh':>9}{'pMean':>7}{'pMax':>7}"
              f"{'nP':>4}{'CPUload':>9}  responsive")
        for m in MODES:
            s = modes.get(m)
            if not s:
                continue
            print(f"{m:<16}{s['finalMaxMs']:>6.0f}m{s['partialRefreshMs']:>8.0f}m"
                  f"{s['partialMeanMs']:>7.0f}{s['partialMaxMs']:>7.0f}{s['nPartials']:>4}"
                  f"{s['cpuLoadRtf']:>8.2f}x  {'YES' if responsive(s) else 'NO'}")

        nv, inc = modes.get("fc_naive"), modes.get("fc_incremental")
        if nv and inc:
            pmax_x = nv["partialMaxMs"] / inc["partialMaxMs"] if inc["partialMaxMs"] else float("inf")
            load_x = nv["cpuLoadRtf"] / inc["cpuLoadRtf"] if inc["cpuLoadRtf"] else float("inf")
            print(f"\n  FC incremental vs naive: max-partial {inc['partialMaxMs']:.0f} vs "
                  f"{nv['partialMaxMs']:.0f} ms ({pmax_x:.1f}x); CPU load {inc['cpuLoadRtf']:.2f} vs "
                  f"{nv['cpuLoadRtf']:.2f} ({load_x:.1f}x). final (both full re-decode): "
                  f"{inc['finalMaxMs']:.0f}/{nv['finalMaxMs']:.0f} ms")
            rows, flip = crossover(nv, inc)
            if rows:
                print("  per-partial crossover (bufferSec: naive_ms / incr_ms):")
                print("   " + "  ".join(f"{s:.1f}:{mn:.0f}/{mi:.0f}" for s, mn, mi in rows))
                print(f"  -> incremental first matches/beats naive at ~{flip:.1f}s buffer"
                      if flip is not None else "  -> no crossover in range")

    print(f"\n{'='*80}\nRESPONSIVE GRID  ({POLICY})\n{'='*80}")
    print(f"{'device':<16}" + "".join(f"{m:>16}" for m in MODES))
    for device in sorted(devices):
        cells = []
        for m in MODES:
            s = devices[device].get(m)
            cells.append("—" if not s else ("YES" if responsive(s) else "NO"))
        print(f"{device:<16}" + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
