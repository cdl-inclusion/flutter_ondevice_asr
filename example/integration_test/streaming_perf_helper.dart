// Streaming performance harness (model-agnostic).
//
// Drives the real StreamingTranscriber and feeds the clip in chunks at real time (a 100 ms chunk
// then a 100 ms wait), so the fire-and-forget guard skips overlapping partials exactly like the live
// app: a slow model just emits fewer partial uodates (no audio is dropped — the buffer keeps growing
// and the next partial / the final covers it). Per-transcription latency comes from a
// `@visibleForTesting` timing seam on StreamingTranscriber (onDecodeTiming).
//
// It therefore judges responsiveness — final latency (time to committed text after a pause) and the
// realized partial-refresh rate — not raw CPU load (RTF): the guard self-throttles, so a busy CPU
// (RTF ~1) still stays responsive. See .github/workflows/streaming_performance_tests.md.

// ignore_for_file: avoid_print — integration-test reporting; print is intentional.

import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter_ondevice_asr/flutter_ondevice_asr.dart';
import 'package:flutter_test/flutter_test.dart';

class _Ev {
  _Ev(this.audioSec, this.ms, this.isFinal, this.atMs);
  final double audioSec; // segment-so-far length decoded
  final double ms; // decode latency
  final bool isFinal;
  final double atMs; // wall-clock (from run start) when this decode completed
}

double _median(List<double> xs) {
  if (xs.isEmpty) return 0;
  final s = [...xs]..sort();
  final n = s.length;
  return n.isOdd ? s[n ~/ 2] : (s[n ~/ 2 - 1] + s[n ~/ 2]) / 2;
}

/// Stream [audio] through StreamingTranscriber [runs] times (run 0 discarded as warmup) and report
/// responsiveness (final latency + partial-refresh rate), per-update decode latency, and CPU load.
/// Writes a compact JSON summary to [outPath] and prints a human table.
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
    final runSw = Stopwatch()..start();
    streaming.onDecodeTiming = (isFinal, samples, ms) =>
        evs.add(_Ev(samples / sampleRate, ms, isFinal, runSw.elapsedMilliseconds.toDouble()));
    streaming.reset();

    // Feed at REAL TIME — audio keeps arriving while a transcription runs. processAudioChunk always
    // buffers the audio; when a transcription is already in flight it just skips that partial-UPDATE
    // (no audio is dropped — the next partial / the final covers it). So a slow model emits fewer
    // partial updates, exactly like the live app; it never blocks or loses audio. Do NOT wait on the
    // transcription to finish — that would force every partial and misrepresent slow models.
    final chunkMs = (feedSamples * 1000 / sampleRate).round();
    for (int i = 0; i < audio.length; i += feedSamples) {
      final end = min(i + feedSamples, audio.length);
      await streaming.processAudioChunk(Float32List.sublistView(audio, i, end));
      await Future<void>.delayed(Duration(milliseconds: chunkMs));
    }
    await streaming.flush();
    runEvents.add(evs);
  }
  streaming.onDecodeTiming = null;
  await sub.cancel();
  await streaming.dispose();

  // 3. Aggregate over measured runs (discard run 0 = warmup / JIT). Real-time feed means a slow
  //    model emits fewer partials (the guard skips overlaps) exactly like the live app — so we judge
  //    responsiveness (final latency + partial refresh rate), not raw CPU load.
  final measured = runEvents.length > 1 ? runEvents.sublist(1) : runEvents;
  final rtfs = <double>[], pMean = <double>[], pMax = <double>[], fMax = <double>[],
      refreshGaps = <double>[], pCounts = <double>[];
  int nF = 0;
  final perRunPartials = <List<_Ev>>[];
  for (final evs in measured) {
    final partialMs = [for (final e in evs) if (!e.isFinal) e.ms];
    final finals = [for (final e in evs) if (e.isFinal) e.ms];
    final total = evs.fold<double>(0, (a, e) => a + e.ms);
    rtfs.add(total / 1000.0 / clipSec); // CPU load (sum of decode work ÷ audio), NOT keep-up
    pMean.add(partialMs.isEmpty ? 0 : partialMs.reduce((a, b) => a + b) / partialMs.length);
    pMax.add(partialMs.isEmpty ? 0 : partialMs.reduce(max));
    fMax.add(finals.isEmpty ? 0 : finals.reduce(max));
    pCounts.add(partialMs.length.toDouble());
    nF = finals.length;
    // UI refresh interval = wall-clock gaps between consecutive partials actually emitted.
    final pev = [for (final e in evs) if (!e.isFinal) e]
      ..sort((a, b) => a.atMs.compareTo(b.atMs));
    for (int k = 1; k < pev.length; k++) {
      refreshGaps.add(pev[k].atMs - pev[k - 1].atMs);
    }
    perRunPartials.add(pev);
  }
  // Per-partial cost vs buffer length — the naive-vs-incremental crossover (FastConformer, which
  // never skips, so its partials align across runs; less meaningful for a skipping model).
  final nP = perRunPartials.isEmpty ? 0 : perRunPartials.map((p) => p.length).reduce(min);
  final curve = <List<double>>[];
  for (int k = 0; k < nP; k++) {
    curve.add([
      double.parse(perRunPartials[0][k].audioSec.toStringAsFixed(2)),
      double.parse(_median([for (final p in perRunPartials) p[k].ms]).toStringAsFixed(1)),
    ]);
  }

  final summary = {
    'mode': mode,
    'clipSeconds': double.parse(clipSec.toStringAsFixed(2)),
    'runs': measured.length,
    'loadMs': loadMs,
    'eosMinSilenceMs': eosMinSilenceMs,
    'minPartialMs': minPartialMs,
    // Responsiveness (what "real-time" actually means for a fire-and-forget streamer):
    'finalMaxMs': double.parse(_median(fMax).toStringAsFixed(1)), // delay to committed text after a pause
    'partialRefreshMs': double.parse(_median(refreshGaps).toStringAsFixed(1)), // UI update interval
    'partialMeanMs': double.parse(_median(pMean).toStringAsFixed(1)), // per-update decode latency
    'partialMaxMs': double.parse(_median(pMax).toStringAsFixed(1)),
    'nPartials': _median(pCounts).round(), // realized partials (fewer when the model is slow)
    'nFinals': nF,
    'cpuLoadRtf': double.parse(_median(rtfs).toStringAsFixed(3)), // load, informational — NOT a gate
    'perPartial': curve,
  };

  print('\n=== STREAMING PERF: $mode ===  (real-time feed, guard skips like the live app)');
  print('clip ${clipSec.toStringAsFixed(1)}s | load ${loadMs}ms | runs ${measured.length} (warmup discarded)');
  print('RESPONSIVENESS: final ${summary['finalMaxMs']}ms after a pause · '
      'partials refresh ~every ${summary['partialRefreshMs']}ms · ${summary['nPartials']} partials');
  print('per-update decode: mean ${summary['partialMeanMs']}ms max ${summary['partialMaxMs']}ms · '
      'CPU load ${summary['cpuLoadRtf']}× real-time');
  print('per-partial cost vs buffer length (median across runs):');
  print('  ${curve.map((e) => '${e[0].toStringAsFixed(1)}s:${e[1].toStringAsFixed(0)}ms').join('  ')}');

  // 4. Write machine-readable summary + clean up.
  await File(outPath)
      .writeAsString(const JsonEncoder.withIndent('  ').convert(summary));
  print('[$mode] wrote $outPath');
  transcriber.dispose();
  print('[${DateTime.now()}] DONE — $mode');
}
