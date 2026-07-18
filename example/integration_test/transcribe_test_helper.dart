// Shared body for the non-streaming transcribe integration tests (model agnostic).

// ignore_for_file: avoid_print — integration-test progress logging; print is intentional.

import 'dart:developer' as dev;
import 'dart:io';
import 'dart:math';

import 'package:flutter/services.dart';
import 'package:flutter_ondevice_asr/model/transcription_result.dart';
import 'package:flutter_ondevice_asr/util/audio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_ondevice_asr/flutter_ondevice_asr.dart';
import 'package:path_provider/path_provider.dart';

/// lowercase, strip everything but letters/digits/spaces, collapse whitespace, trim.
/// Used for P&C-/whitespace-invariant comparison (FastConformer's spacing/casing differs).
String normalizeForCompare(String s) => s
    .toLowerCase()
    .replaceAll(RegExp(r'[^a-z0-9\s]'), ' ')
    .replaceAll(RegExp(r'\s+'), ' ')
    .trim();

/// Load [type] from [modelDirectory], transcribe the bundled [testAudioFile] 5x (for latency
/// stats), and assert the transcript equals [expectedTranscript].
///
/// [compareNormalized] = false → exact string match (Whisper); true → P&C/whitespace-invariant
/// (FastConformer, whose output differs only cosmetically: spaces before punctuation, and
/// CTC "ask" vs RNN-T "Ask").
Future<void> runTranscribeIntegrationTest({
  required TranscriberType type,
  required String modelDirectory,
  required String language,
  required String testAudioFile,
  required String expectedTranscript,
  bool compareNormalized = false,
  String label = '',
}) async {
  final totalSw = Stopwatch()..start();
  final stepSw = Stopwatch()..start();
  void logStep(String message) {
    print('[${DateTime.now()}] $message (${stepSw.elapsedMilliseconds}ms)');
    stepSw
      ..reset()
      ..start();
  }

  print('[${DateTime.now()}] START TEST — $label');

  final transcriber = Transcriber.getInstance(type);

  // 1. Load model
  final loadModelResult = await transcriber.loadModel(
      modelDirectory: modelDirectory, languageCode: language);
  if (loadModelResult is! Ok) {
    // Surface the REAL cause (Result.Error.error) rather than a bare "failed".
    print('[$label] loadModel ERROR: ${(loadModelResult as dynamic).error}');
  }
  expect(loadModelResult is Ok, true, reason: '[$label] loadModel failed');
  logStep('Model loaded');

  // 2. Bundle test audio to a temp file, then decode to Float32List (16 kHz mono).
  final audioAsset = await rootBundle.load(testAudioFile);
  final tempDir = await getTemporaryDirectory();
  if (!await tempDir.exists()) {
    await tempDir.create(recursive: true);
  }
  final tempAudioFile = File('${tempDir.path}/test_audio.wav');
  await tempAudioFile.writeAsBytes(audioAsset.buffer.asUint8List());
  final audio = await Audio.instance.loadAudio(tempAudioFile.path);
  logStep('Audio prepared');

  final audioDurationSec = audio.length / 16000.0;
  print('Audio duration: ${audioDurationSec.toStringAsFixed(2)} s '
      '(${audio.length} samples @ 16kHz)');

  // 3. Transcribe 5x for latency statistics.
  print('\n=== $label: 5 runs for performance statistics ===');
  final durations = <double>[];
  String? transcript;
  for (int run = 0; run < 5; run++) {
    stepSw
      ..reset()
      ..start();
    dev.Timeline.startSync('${type == .fastConformer ? 'ctc' : 'rnnt'}.transcribe');
    final result = await transcriber.transcribe(audio) as Ok<TranscriptionResult>;
    dev.Timeline.finishSync();
    durations.add(stepSw.elapsedMilliseconds.toDouble());
    if (run == 0) transcript = result.value.text;
  }
  final avg = durations.reduce((a, b) => a + b) / durations.length;
  final variance =
      durations.map((d) => pow(d - avg, 2)).reduce((a, b) => a + b) / durations.length;
  print('\n=== $label Performance ===');
  print('Average: ${avg.toStringAsFixed(1)} ms | Std: ${sqrt(variance).toStringAsFixed(1)} ms '
      '| Min: ${durations.reduce(min).toStringAsFixed(1)} ms '
      '| Max: ${durations.reduce(max).toStringAsFixed(1)} ms');
  print('Transcript: $transcript');

  // 4. Clean up + verify.
  await tempAudioFile.delete();
  transcriber.dispose();
  if (compareNormalized) {
    expect(normalizeForCompare(transcript!), normalizeForCompare(expectedTranscript),
        reason: '[$label] transcript mismatch (normalized)');
  } else {
    expect(transcript, expectedTranscript, reason: '[$label] transcript mismatch');
  }

  print('[${DateTime.now()}] TEST PASSED — $label — total ${totalSw.elapsed.inSeconds}s');
}
