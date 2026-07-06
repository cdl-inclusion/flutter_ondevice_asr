// Test for both FastConformer versions (CTC and RNN-T).

import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_ondevice_asr/common/result.dart';
import 'package:flutter_ondevice_asr/model/transcription_result.dart';
import 'package:flutter_ondevice_asr/transcriber.dart';
import 'package:flutter_ondevice_asr/models/fastconformer/fastconformer_transcriber.dart';
import 'package:flutter_ondevice_asr/util/audio.dart';
import 'package:flutter_test/flutter_test.dart';

import 'test_utils.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  // Reuse the audio bundled for the Whisper tests (16 kHz mono wav).
  final testAudioFile = toAbsolutePath('assets/audio/jfk_asknot.wav');

  const expectedTranscript =
      'And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.';

  // One HYBRID artifact dir feeds both heads (shared super_encoder.onnx + ctc_decoder.onnx
  // + decoder_joint.onnx).
  final hybridDir = toAbsolutePath('assets/transcribers/fastconformer/hybrid_int8');

  group('FastConformer super-encoder transcribe (jfk_asknot)', () {
    _headTest(
      label: 'CTC',
      modelDirectory: hybridDir,
      requiredFiles: const ['super_encoder.onnx', 'ctc_decoder.onnx'],
      create: () => FastConformerCtcTranscriber(),
      testAudioFile: testAudioFile,
      expectedTranscript: expectedTranscript,
    );

    _headTest(
      label: 'RNN-T',
      modelDirectory: hybridDir,
      requiredFiles: const ['super_encoder.onnx', 'decoder_joint.onnx'],
      create: () => FastConformerRnntTranscriber(),
      testAudioFile: testAudioFile,
      expectedTranscript: expectedTranscript,
    );
  });

  // Both heads expose per-word confidence + timestamps and a segment confidence when
  // getWordDetails/getSegmentDetails are set: CTC from its (log-softmax) log-probs, RNN-T
  // from a softmax over the joint logits at each emitted token.
  group('FastConformer word/segment details', () {
    _detailsTest(
      label: 'CTC',
      modelDirectory: hybridDir,
      requiredFiles: const ['super_encoder.onnx', 'ctc_decoder.onnx'],
      create: () => FastConformerCtcTranscriber(),
      testAudioFile: testAudioFile,
    );
    _detailsTest(
      label: 'RNN-T',
      modelDirectory: hybridDir,
      requiredFiles: const ['super_encoder.onnx', 'decoder_joint.onnx'],
      create: () => FastConformerRnntTranscriber(),
      testAudioFile: testAudioFile,
    );
  });
}

/// Register one head's word/segment-details test: transcribe with getWordDetails +
/// getSegmentDetails and assert confidences ∈ (0,1], non-empty roughly-monotonic word
/// spans within the clip, and that the words reconstruct the transcript.
void _detailsTest({
  required String label,
  required String modelDirectory,
  required List<String> requiredFiles,
  required Transcriber Function() create,
  required String testAudioFile,
}) {
  final assetPresent =
      requiredFiles.every((f) => File('$modelDirectory/$f').existsSync());

  test('$label word/segment details (confidence + timestamps)', () async {
    final fc = create();
    expect(
      await fc.loadModel(modelDirectory: modelDirectory, languageCode: 'en') is Ok,
      true,
    );
    final audio = await Audio.instance.loadAudio(testAudioFile);
    final durationSec = audio.length / 16000.0;

    final result = await fc.transcribe(
      audio,
      getWordDetails: true,
      getSegmentDetails: true,
    );
    expect(result is Ok, true);
    final tr = (result as Ok<TranscriptionResult>).value;

    // Word-level details.
    expect(tr.words, isNotNull);
    expect(tr.words!.isNotEmpty, true);
    double prevEnd = -1.0;
    for (final w in tr.words!) {
      expect(w.word.trim().isNotEmpty, true);
      expect(w.confidence, greaterThan(0.0));
      expect(w.confidence, lessThanOrEqualTo(1.0)); // exp(log-prob) is a probability
      expect(w.start, greaterThanOrEqualTo(0.0));
      expect(w.end, greaterThan(w.start)); // non-empty span
      expect(w.end, lessThanOrEqualTo(durationSec + 0.2)); // within the clip (+slack)
      expect(w.start, greaterThanOrEqualTo(prevEnd - 0.2)); // roughly monotonic
      prevEnd = w.end;
    }
    // Words reconstruct the transcript (P&C-/whitespace-invariant).
    final joined = tr.words!.map((w) => w.word).join(' ');
    expect(_normalizeForCompare(joined), _normalizeForCompare(tr.text));

    // Segment-level confidence.
    expect(tr.confidences, isNotNull);
    expect(tr.confidences!.single, greaterThan(0.0));
    expect(tr.confidences!.single, lessThanOrEqualTo(1.0));

    debugPrint('\n=== $label word details (${tr.words!.length} words) ===');
    for (final w in tr.words!.take(6)) {
      debugPrint('  ${w.word}  conf=${w.confidence.toStringAsFixed(3)} '
          '[${w.start.toStringAsFixed(2)}s..${w.end.toStringAsFixed(2)}s]');
    }
    debugPrint('segment confidence: ${tr.confidences!.single.toStringAsFixed(3)}');

    fc.dispose();
  }, skip: assetPresent ? false : '$label asset not present at $modelDirectory');
}

void _headTest({
  required String label,
  required String modelDirectory,
  required List<String> requiredFiles,
  required Transcriber Function() create,
  required String testAudioFile,
  required String expectedTranscript,
}) {
  final assetPresent =
      requiredFiles.every((f) => File('$modelDirectory/$f').existsSync());

  test('transcribe test audio ($label)', () async {
    debugPrint('  modelDirectory: $modelDirectory');

    final fc = create();
    final loadResult = await fc.loadModel(
      modelDirectory: modelDirectory,
      languageCode: 'en',
    );
    expect(loadResult is Ok, true);

    final audioData = await Audio.instance.loadAudio(testAudioFile);
    final audioDurationSec = audioData.length / 16000.0;
    debugPrint(
      '\nAudio duration: ${audioDurationSec.toStringAsFixed(2)} seconds '
      '(${audioData.length} samples @ 16kHz)',
    );

    // Run a few times for latency stats.
    String? transcript;
    final durations = <double>[];
    for (int run = 0; run < 3; run++) {
      final stopwatch = Stopwatch()..start();
      final result = await fc.transcribe(audioData);
      stopwatch.stop();
      durations.add(stopwatch.elapsedMilliseconds.toDouble());
      if (run == 0) {
        transcript = (result as Ok<TranscriptionResult>).value.text;
      }
    }

    final avg = durations.reduce((a, b) => a + b) / durations.length;
    debugPrint('\n=== Performance ($label) ===');
    debugPrint('Average: ${avg.toStringAsFixed(1)} ms');
    debugPrint('Min: ${durations.reduce(min).toStringAsFixed(1)} ms');
    debugPrint('Max: ${durations.reduce(max).toStringAsFixed(1)} ms');
    debugPrint('\nTranscript: $transcript');

    expect(transcript, isNotNull);
    expect(transcript!.isNotEmpty, true);
    expect(_normalizeForCompare(transcript), _normalizeForCompare(expectedTranscript));

    fc.dispose();
  }, skip: assetPresent ? false : '$label asset not present at $modelDirectory');
}

/// lowercase, strip everything but letters/digits/spaces, collapse runs of whitespace, trim.
String _normalizeForCompare(String s) => s
    .toLowerCase()
    .replaceAll(RegExp(r'[^a-z0-9\s]'), ' ')
    .replaceAll(RegExp(r'\s+'), ' ')
    .trim();
