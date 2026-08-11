#!/usr/bin/env python3
"""Summarize on-device ASR performance traces into a markdown and/or TSV table.

Reads every ``*.json`` Flutter timeline trace under a user-specified directory
(recursively), extracts the per-run inference timings, and reports a table with
one row (group) per PHONE and one column per MODEL: mean encoder / decoder /
total latencies (ms), optionally with standard deviations (--add-std). When
--audio is given, an RTF row (real-time factor = total / audio duration, read
from the WAV header) is included; the traces don't record the clip, so without
--audio no RTF is computed.

Each decoder head counts as its own model, with the head named before any
quantization suffix: a hybrid FastConformer artifact profiled with both heads
yields e.g. ``fc_ctc_fp32`` and ``fc_rnnt_fp32`` columns. The base model name
is the trace's subdirectory (``performance_`` prefix stripped); the phone comes
from the filename (``performance_trace_`` prefix and ``_ctc``/``_rnnt`` suffix
stripped).

Trace conventions (see lib/models/*/ and the integration tests that emit them):
  - ``run_super_encoder``            encoder pass (FastConformer and Whisper).
  - ``decode_from_encoder``          FastConformer CTC head decoder; the nested
                                     ``decode`` event is part of it (not added).
  - ``rnnt.decode_from_encoder`` +
    ``rnnt.decode``                  FastConformer RNN-T head; the two run
                                     sequentially, so their sum is the decoder.
  - ``run_decoder``                  Whisper autoregressive decoder loop.
  - ``*.transcribe`` / ``transcribe`` end-to-end wrapper event; the prefixed
                                     variant (ctc./rnnt./hybrid.*) is emitted by
                                     the integration-test helper and is preferred.
                                     NOTE: Whisper traces label it ``rnnt.transcribe``.

Usage:
  nonstreaming_perf_report.py TRACE_DIR [--md OUT.md] [--tsv OUT.tsv]
                              [--add-std] [--sort-by MODEL] [--audio WAV]

Phones are ordered fastest first by the --sort-by model's mean total (default
``whisper_tiny``; phones lacking that model go last). If the model isn't in the
data at all, ordering falls back to each phone's min total across models.

At least one of --md / --tsv must be given; only the requested files are
written. --add-std includes the standard deviation of every metric (encoder,
decoder, total); off by default.
"""

import argparse
import json
import statistics
import sys
import wave
from collections import defaultdict
from pathlib import Path

def audio_duration_seconds(path):
    """Duration of a PCM WAV file, from its header."""
    with wave.open(str(path), 'rb') as w:
        return w.getnframes() / w.getframerate()

ENCODER_EVENT = 'run_super_encoder'
# Recognized quantization suffixes in model directory names; a decoder-head
# label is inserted BEFORE these (fc_fp32 + rnnt -> fc_rnnt_fp32).
QUANT_SUFFIXES = ('fp32', 'fp16', 'int8', 'int4')
# Column order: quantized variants first, then float; models without a
# recognized quant suffix (e.g. whisper_tiny) go last. Alphabetical within.
QUANT_COLUMN_ORDER = ('int8', 'int4', 'fp16', 'fp32')


def model_column_key(model):
    quant = model.rpartition('_')[2]
    rank = (QUANT_COLUMN_ORDER.index(quant) if quant in QUANT_COLUMN_ORDER
            else len(QUANT_COLUMN_ORDER))
    return (rank, model)


def model_with_head(model, head):
    """Fold the decoder head into the model name, before any quant suffix."""
    if not head:
        return model
    base, _, last = model.rpartition('_')
    if base and last in QUANT_SUFFIXES:
        return f'{base}_{head}_{last}'
    return f'{model}_{head}'


def event_durations_ms(trace_path):
    """Duration lists (ms) per event name, pairing B/E events per thread."""
    with open(trace_path) as f:
        events = json.load(f).get('traceEvents', [])
    stacks = defaultdict(list)
    durs = defaultdict(list)
    for e in events:
        ph = e.get('ph')
        name = e.get('name')
        key = (e.get('pid'), e.get('tid'), name)
        if ph == 'X':
            durs[name].append(e['dur'] / 1000)
        elif ph == 'B':
            stacks[key].append(e['ts'])
        elif ph == 'E' and stacks[key]:
            durs[name].append((e['ts'] - stacks[key].pop()) / 1000)
    return durs


def classify_head(durs):
    """Head suffix ('' for single-path models) and per-run decoder durations."""
    if 'run_decoder' in durs:  # Whisper: single autoregressive decoder
        return '', durs['run_decoder']
    if 'rnnt.decode' in durs:  # RNN-T: greedy loop + encoder-output unpack
        loop = durs['rnnt.decode']
        unpack = durs.get('rnnt.decode_from_encoder', [])
        if len(unpack) == len(loop):
            return 'rnnt', [a + b for a, b in zip(loop, unpack)]
        return 'rnnt', loop
    if 'decode_from_encoder' in durs:  # CTC (its nested 'decode' is included)
        return 'ctc', durs['decode_from_encoder']
    return '', []


def total_durations(durs):
    """Per-run end-to-end durations; prefer the head-prefixed wrapper event."""
    prefixed = [n for n in durs if n.endswith('.transcribe')]
    if prefixed:
        return durs[max(prefixed, key=lambda n: statistics.mean(durs[n]))]
    return durs.get('transcribe', [])


def model_and_phone(trace_path, root):
    rel = trace_path.relative_to(root)
    model = str(rel.parent) if str(rel.parent) != '.' else '(root)'
    model = model.removeprefix('performance_')
    phone = trace_path.stem.removeprefix('performance_trace_')
    for suffix in ('_ctc', '_rnnt'):
        phone = phone.removesuffix(suffix)
    return model, phone


def _mean(vals):
    return statistics.mean(vals) if vals else float('nan')


def _sd(vals):
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def summarize(root, audio_seconds=None):
    """One measurement dict per (model, phone); the head is part of the model.

    [audio_seconds] is the duration of the profiled clip (not recorded in the
    traces themselves), used to derive RTF = total_time / audio_duration;
    without it no RTF fields are produced.
    """
    rows = []
    for trace_path in sorted(root.rglob('*.json')):
        try:
            durs = event_durations_ms(trace_path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f'skipping (not a JSON trace): {trace_path}', file=sys.stderr)
            continue
        totals = total_durations(durs)
        encoder = durs.get(ENCODER_EVENT, [])
        if not totals or not encoder:
            print(f'skipping (no transcribe/encoder events): {trace_path}',
                  file=sys.stderr)
            continue
        head, decoder = classify_head(durs)
        model, phone = model_and_phone(trace_path, root)
        model = model_with_head(model, head)
        rows.append({
            'model': model,
            'phone': phone,
            'runs': len(totals),
            'encoder_ms': _mean(encoder), 'encoder_sd_ms': _sd(encoder),
            'decoder_ms': _mean(decoder), 'decoder_sd_ms': _sd(decoder),
            'total_ms': _mean(totals), 'total_sd_ms': _sd(totals),
        })
        if audio_seconds:
            rows[-1]['rtf'] = _mean(totals) / 1000 / audio_seconds
            rows[-1]['rtf_sd'] = _sd(totals) / 1000 / audio_seconds
    return rows


def pivot(rows, sort_model=None):
    """(phones, models, cells): cells[phone][model] = measurement dict.

    Phones are ordered fastest first by [sort_model]'s mean total (phones
    without that model go last, ordered by their min total across models);
    without a sort_model, by min total across models. Model columns are
    grouped by quant (int8 before fp32, unsuffixed models last), then
    alphabetical within each group.
    """
    cells = defaultdict(dict)
    for r in rows:
        cells[r['phone']][r['model']] = r

    def phone_key(p):
        fallback = min(r['total_ms'] for r in cells[p].values())
        if sort_model is None:
            return (0, fallback)
        entry = cells[p].get(sort_model)
        return (0, entry['total_ms']) if entry else (1, fallback)

    phones = sorted(cells, key=phone_key)
    models = sorted({r['model'] for r in rows}, key=model_column_key)
    return phones, models, cells


# (label, mean key, sd key, decimals in md, decimals in tsv)
METRICS = (('Encoder', 'encoder_ms', 'encoder_sd_ms', 0, 1),
           ('Decoder', 'decoder_ms', 'decoder_sd_ms', 0, 1),
           ('Total', 'total_ms', 'total_sd_ms', 0, 1),
           ('RTF', 'rtf', 'rtf_sd', 3, 4))


def active_metrics(rows):
    """METRICS minus RTF when it wasn't computed (no --audio given)."""
    if any('rtf' in r for r in rows):
        return METRICS
    return METRICS[:-1]


def to_markdown(rows, add_std=False, sort_model=None):
    """Phone x metric rows, one column per model."""
    phones, models, cells = pivot(rows, sort_model)

    def fmt(r, mean_key, sd_key, decimals):
        value = f'{r[mean_key]:.{decimals}f}'
        if add_std:
            value += f' ± {r[sd_key]:.{decimals}f}'
        return value

    header = ['Phone', 'Metric'] + models
    lines = ['| ' + ' | '.join(header) + ' |',
             '|' + '|'.join('---' for _ in header) + '|']
    for phone in phones:
        for i, (label, mean_key, sd_key, decimals, _) in enumerate(active_metrics(rows)):
            name = phone if i == 0 else ''
            values = [fmt(cells[phone][m], mean_key, sd_key, decimals)
                      if m in cells[phone] else '—' for m in models]
            lines.append('| ' + ' | '.join([name, label] + values) + ' |')
    return '\n'.join(lines)


def to_tsv(rows, add_std=False, sort_model=None):
    """One row per phone; numeric column group per model for pivoting."""
    phones, models, cells = pivot(rows, sort_model)
    metrics = []  # (row key, decimals)
    for _, mean_key, sd_key, _, decimals in active_metrics(rows):
        metrics.append((mean_key, decimals))
        if add_std:
            metrics.append((sd_key, decimals))
    header = ['phone'] + [f'{m}_{key}' for m in models for key, _ in metrics]
    lines = ['\t'.join(header)]
    for phone in phones:
        values = [phone]
        for model in models:
            entry = cells[phone].get(model)
            values += ([f'{entry[key]:.{decimals}f}' for key, decimals in metrics]
                       if entry else [''] * len(metrics))
        lines.append('\t'.join(values))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(
        description='Summarize ASR performance traces (all *.json under a '
                    'directory) as a phones-by-models table (markdown and/or TSV).')
    parser.add_argument('directory', type=Path,
                        help='root directory to scan recursively for traces')
    parser.add_argument('--md', type=Path, metavar='OUT.md',
                        help='write the summary as a markdown table')
    parser.add_argument('--tsv', type=Path, metavar='OUT.tsv',
                        help='write the summary as a TSV file')
    parser.add_argument('--add-std', action='store_true',
                        help='include the standard deviation of every metric')
    parser.add_argument('--sort-by', metavar='MODEL', default='whisper_tiny_int8',
                        help='order phones by this model\'s mean total, fastest '
                             'first (default: %(default)s); phones without the '
                             'model go last')
    parser.add_argument('--audio', type=Path, metavar='WAV',
                        help='the audio clip the traces profiled; its duration is '
                             'read from the WAV header to compute RTF. Without '
                             'it, no RTF is reported.')
    args = parser.parse_args()

    if not args.md and not args.tsv:
        parser.error('at least one of --md or --tsv is required')
    if not args.directory.is_dir():
        parser.error(f'not a directory: {args.directory}')
    audio_seconds = None
    if args.audio:
        try:
            audio_seconds = audio_duration_seconds(args.audio)
        except (OSError, wave.Error) as e:
            parser.error(f'cannot read audio duration from {args.audio}: {e}')
        print(f'RTF audio reference: {args.audio.name} ({audio_seconds:.2f} s)',
              file=sys.stderr)
    else:
        print('warning: no --audio given; RTF not computed', file=sys.stderr)
    rows = summarize(args.directory, audio_seconds)
    if not rows:
        sys.exit(f'no usable trace files found under {args.directory}')
    phone_count = len({r['phone'] for r in rows})

    models = {r['model'] for r in rows}
    sort_model = args.sort_by
    if sort_model not in models:
        print(f"sort model '{sort_model}' not found (have: "
              f"{', '.join(sorted(models))}); ordering by min total instead",
              file=sys.stderr)
        sort_model = None

    if args.md:
        args.md.write_text(
            to_markdown(rows, add_std=args.add_std, sort_model=sort_model) + '\n')
        print(f'wrote {phone_count} phone rows to {args.md}', file=sys.stderr)
    if args.tsv:
        args.tsv.write_text(
            to_tsv(rows, add_std=args.add_std, sort_model=sort_model))
        print(f'wrote {phone_count} phone rows to {args.tsv}', file=sys.stderr)


if __name__ == '__main__':
    main()
