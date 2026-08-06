// Streaming performance harness (model-agnostic).
//
// Drives the StreamingTranscriber - feed the clip in chunks, let it segment. 
// Per-transcription latency comes from a `@visibleForTesting` timing seam on
// StreamingTranscriber (onDecodeTiming); feeding is serialized on `isTranscribing` so every partial
// is measured (deterministic — the same decode work on every device, so the only variable is speed).

// ignore_for_file: avoid_print — integration-test reporting; print is intentional.

import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter_ondevice_asr/flutter_ondevice_asr.dart';
import 'package:flutter_test/flutter_test.dart';

/// Realtime verdict: require RTF <= this, reserving the rest of the CPU budget (here 30%) for VAD,
/// audio capture, UI, and thermal throttling. Drives the in-test printed verdict + the `realtime`
/// JSON field. The report script (.github/scripts/streaming_perf_report.py) re-derives the verdict
/// from the raw numbers and owns the authoritative, tweakable policy (RTF_HEADROOM there).
const double kMaxRtf = 0.7; // 30% CPU headroom (RTF must stay at/under 0.70)

class _Ev {
  _Ev(this.audioSec, this.ms, this.isFinal);
  final double audioSec;
  final double ms;
  final bool isFinal;
}

double _median(List<double> xs) {
  if (xs.isEmpty) return 0;
  final s = [...xs]..sort();
  final n = s.length;
  return n.isOdd ? s[n ~/ 2] : (s[n ~/ 2 - 1] + s[n ~/ 2]) / 2;
}

double _p90(List<double> xs) {
  if (xs.isEmpty) return 0;
  final s = [...xs]..sort();
  return s[(0.9 * (s.length - 1)).round()];
}

/// Stream [audio] through StreamingTranscriber [runs] times (run 0 discarded as warmup) and report
/// per-partial / per-final decode latency, RTF, and a realtime verdict. Writes a compact JSON summary
/// to [outPath] and prints a human table.
///
/// [incremental] toggles `enableIncrementalStreaming` (only effective if the transcriber implements
/// IncrementalStreaming, e.g. FastConformer). The final is always a full re-decode
/// (fullRedecodeOnStreamingFinal default true) — the win, if any, is purely in the partials.
Future<void> runStreamingPerf({
  required TranscriberType type,
  required bool incremental,
  required String modelDirectory,
  required String language,
  required Float32List audio,
  required String mode,
  required String outPath,
  int eosMinSilenceMs = 1000,
  int minPartialMs = 500,
  int runs = 4,
  int sampleRate = 16000,
  int feedSamples = 1600, // 100 ms audio chunks, as an app would deliver
}) async {
  print('\n[${DateTime.now()}] START STREAMING PERF — $mode '
      '(eos=${eosMinSilenceMs}ms, partialEvery=${minPartialMs}ms, runs=$runs)');

  // 1. Load model (timed separately; not part of RTF).
  final transcriber = Transcriber.getInstance(type);
  final loadSw = Stopwatch()..start();
  final loadRes = await transcriber.loadModel(
      modelDirectory: modelDirectory, languageCode: language);
  loadSw.stop();
  if (loadRes is! Ok) {
    print('[$mode] loadModel ERROR: ${(loadRes as dynamic).error}');
  }
  expect(loadRes is Ok, true, reason: '[$mode] loadModel failed');
  final loadMs = loadSw.elapsedMilliseconds;
  final clipSec = audio.length / sampleRate;

  // 2. One StreamingTranscriber for the whole mode (loads Silero VAD once); reset between runs.
  final streaming = await StreamingTranscriber.create(
    transcriber: transcriber,
    eosMinSilence: eosMinSilenceMs,
    sampleRate: sampleRate,
    enablePartials: true,
    minPartialDuration: minPartialMs,
    enableIncrementalStreaming: incremental,
  );
  // Drain the result stream so the broadcast controller doesn't buffer.
  final sub = streaming.transcriptionStream.listen((_) {});

  final runEvents = <List<_Ev>>[];
  for (int r = 0; r < runs; r++) {
    final evs = <_Ev>[];
    streaming.onDecodeTiming =
        (isFinal, samples, ms) => evs.add(_Ev(samples / sampleRate, ms, isFinal));
    streaming.reset();

    for (int i = 0; i < audio.length; i += feedSamples) {
      final end = min(i + feedSamples, audio.length);
      await streaming.processAudioChunk(Float32List.sublistView(audio, i, end));
      // Serialize: let the fire-and-forget transcription finish so no partial is skipped, and the
      // decode timing is attributed cleanly. (Real apps feed in real time; here we want max cost.)
      while (streaming.isTranscribing) {
        await Future<void>.delayed(const Duration(milliseconds: 1));
      }
    }
    await streaming.flush();
    runEvents.add(evs);
  }
  streaming.onDecodeTiming = null;
  await sub.cancel();
  await streaming.dispose();

  // 3. Aggregate over measured runs (discard run 0 = warmup / JIT).
  final measured = runEvents.length > 1 ? runEvents.sublist(1) : runEvents;
  final rtfs = <double>[], pMean = <double>[], pP90 = <double>[],
      pMax = <double>[], fMax = <double>[];
  int nF = 0;
  final perRunPartials = <List<_Ev>>[]; // partials per measured run (segmentation is deterministic)
  for (final evs in measured) {
    final partials = [for (final e in evs) if (!e.isFinal) e.ms];
    final finals = [for (final e in evs) if (e.isFinal) e.ms];
    final total = evs.fold<double>(0, (a, e) => a + e.ms);
    rtfs.add(total / 1000.0 / clipSec);
    pMean.add(partials.isEmpty ? 0 : partials.reduce((a, b) => a + b) / partials.length);
    pP90.add(_p90(partials));
    pMax.add(partials.isEmpty ? 0 : partials.reduce(max));
    fMax.add(finals.isEmpty ? 0 : finals.reduce(max));
    nF = finals.length;
    perRunPartials.add([for (final e in evs) if (!e.isFinal) e]);
  }
  // Per-partial cost vs buffer length (seconds) — the naive-vs-incremental crossover. Segmentation
  // is deterministic across runs, so partial k has the same buffer length every run; median the ms.
  final nP = perRunPartials.isEmpty
      ? 0
      : perRunPartials.map((p) => p.length).reduce(min);
  final curve = <List<double>>[];
  for (int k = 0; k < nP; k++) {
    final bufSec = perRunPartials[0][k].audioSec;
    final msK = _median([for (final p in perRunPartials) p[k].ms]);
    curve.add([
      double.parse(bufSec.toStringAsFixed(2)),
      double.parse(msK.toStringAsFixed(1)),
    ]);
  }

  final rtf = _median(rtfs);
  final partialMax = _median(pMax);
  final realtime = partialMax < minPartialMs && rtf < kMaxRtf;

  final summary = {
    'mode': mode,
    'clipSeconds': double.parse(clipSec.toStringAsFixed(2)),
    'runs': measured.length,
    'loadMs': loadMs,
    'eosMinSilenceMs': eosMinSilenceMs,
    'minPartialMs': minPartialMs,
    'rtf': double.parse(rtf.toStringAsFixed(3)),
    'partialMeanMs': double.parse(_median(pMean).toStringAsFixed(1)),
    'partialP90Ms': double.parse(_median(pP90).toStringAsFixed(1)),
    'partialMaxMs': double.parse(partialMax.toStringAsFixed(1)),
    'finalMaxMs': double.parse(_median(fMax).toStringAsFixed(1)),
    'nPartials': nP,
    'nFinals': nF,
    'realtime': realtime,
    // [ [bufferSeconds, decodeMs], ... ] for one run — the per-partial cost-vs-length curve.
    'perPartial': curve,
  };

  print('\n=== STREAMING PERF: $mode ===');
  print('clip ${clipSec.toStringAsFixed(1)}s | load ${loadMs}ms | '
      'runs ${measured.length} (warmup discarded)');
  print('RTF ${rtf.toStringAsFixed(2)} | '
      'partials: mean ${summary['partialMeanMs']}ms p90 ${summary['partialP90Ms']}ms '
      'max ${summary['partialMaxMs']}ms (n=$nP) | '
      'finals: max ${summary['finalMaxMs']}ms (n=$nF)');
  print('per-partial cost vs buffer length (median across runs):');
  print('  ${curve.map((e) => '${e[0].toStringAsFixed(1)}s:${e[1].toStringAsFixed(0)}ms').join('  ')}');
  print('REALTIME: max-partial ${summary['partialMaxMs']}ms '
      '${partialMax < minPartialMs ? "<" : ">="} ${minPartialMs}ms cadence AND '
      'rtf ${rtf.toStringAsFixed(2)} ${rtf < kMaxRtf ? "<" : ">="} $kMaxRtf (30% headroom)  =>  '
      '${realtime ? "YES" : "NO"}');

  // 4. Write machine-readable summary + clean up.
  await File(outPath)
      .writeAsString(const JsonEncoder.withIndent('  ').convert(summary));
  print('[$mode] wrote $outPath');
  transcriber.dispose();
  print('[${DateTime.now()}] DONE — $mode');
}
