import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_ondevice_asr/common/result.dart';
import 'package:flutter_ondevice_asr/model/transcription_result.dart';
import 'package:flutter_ondevice_asr/models/whisper/whisper_transcriber.dart';
import 'package:flutter_ondevice_asr/util/audio.dart';
import 'package:flutter_test/flutter_test.dart';

import 'test_utils.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  // Model configuration - convert to absolute paths for filesystem access
  final modelDirectory = toAbsolutePath(
    'assets/transcribers/whisper/models/whisper_tiny/default_int8',
  );
  const language = 'en';

  final testAudioFile = toAbsolutePath('assets/audio/jfk_asknot.wav');

  debugPrint('Unit test paths (absolute, filesystem-based):');
  debugPrint('  modelDirectory: $modelDirectory');
  debugPrint('  testAudioFile: $testAudioFile');
  const expectedTranscript =
      'And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.';

  // // Alternative test audio:
  // const testAudioFile = 'assets/audio/crisp_autumn.wav';
  // const expectedTranscript = 'Crisp, autumn leaves crunch, underfoot.';

  test('language validation', () async {
    // Test 2 supported languages - should not throw
    expect(
      () => WhisperTranscriber().loadModel(
        modelDirectory: modelDirectory,
        languageCode: 'en',
      ),
      returnsNormally,
    );

    expect(
      () => WhisperTranscriber().loadModel(
        modelDirectory: modelDirectory,
        languageCode: 'de',
      ),
      returnsNormally,
    );

    // Test 1 unsupported language - should throw ArgumentError
    final result = await WhisperTranscriber().loadModel(
      modelDirectory: modelDirectory,
      languageCode: 'xyz',
    );
    expect(result is Error, true);
    if (result is Error) {
      expect(
        result.error.toString(),
        contains('Language "xyz" not found in model configuration.'),
      );
    }
  });

  test('transcribe test audio', () async {
    final whisper = WhisperTranscriber();
    await whisper.loadModel(
      modelDirectory: modelDirectory,
      languageCode: language,
    );
    final audioData = await Audio.instance.loadAudio(testAudioFile);

    // Audio is sampled at 16kHz (16000 samples per second)
    final audioDurationSec = audioData.length / 16000.0;
    debugPrint(
      '\nAudio duration: ${audioDurationSec.toStringAsFixed(2)} seconds (${audioData.length} samples @ 16kHz)',
    );

    // Test 1: WITHOUT confidence (run 5 times for statistics)
    debugPrint('\n=== Test 1: transcribe() WITHOUT confidence (5 runs) ===');

    String? transcript1;
    final durations = <double>[];

    for (int run = 0; run < 5; run++) {
      debugPrint('\n--- Run ${run + 1}/5 ---');
      final stopwatch = Stopwatch()..start();
      final result = await whisper.transcribe(audioData, maxOutputTokens: 32);
      stopwatch.stop();
      final durationMs = stopwatch.elapsedMilliseconds.toDouble();
      durations.add(durationMs);
      debugPrint('Duration: ${durationMs.toStringAsFixed(1)} ms');
      if (run == 0)
        transcript1 = (result as Ok<TranscriptionResult>).value.text;
    }

    // Calculate average and standard deviation
    final avg = durations.reduce((a, b) => a + b) / durations.length;
    final variance =
        durations.map((d) => pow(d - avg, 2)).reduce((a, b) => a + b) /
        durations.length;
    final std = sqrt(variance);

    debugPrint('\n=== Performance Statistics ===');
    debugPrint('Average: ${avg.toStringAsFixed(1)} ms');
    debugPrint('Std Dev: ${std.toStringAsFixed(1)} ms');
    debugPrint('Min: ${durations.reduce(min).toStringAsFixed(1)} ms');
    debugPrint('Max: ${durations.reduce(max).toStringAsFixed(1)} ms');
    debugPrint('\nTranscript: $transcript1');

    // Test 2: WITH confidence (single run)
    debugPrint('\n=== Test 2: transcribe() WITH confidence (1 run) ===');
    final result = await whisper.transcribe(
      audioData,
      getWordDetails: true,
      maxOutputTokens: 32,
    );
    final transcript2 = (result as Ok<TranscriptionResult>).value.text;
    final words = result.value.words!;

    debugPrint('Transcript: $transcript2');
    debugPrint(
      'Word confidences: ${words.map((w) => '${w.word}(${w.confidence.toStringAsFixed(3)})').join(', ')}',
    );
    debugPrint('Number of words: ${words.length}');

    // Verify word confidences match expected values
    final expectedWordConfidences = [
      ('And', 0.690),
      ('so', 0.923),
      ('my', 0.792),
      ('fellow', 0.998),
      ('Americans', 0.952),
      ('ask', 0.602),
      ('not', 0.460),
      ('what', 0.806),
      ('your', 0.592),
      ('country', 0.994),
      ('can', 0.988),
      ('do', 0.993),
      ('for', 0.979),
      ('you,', 0.380),
      ('ask', 0.891),
      ('what', 0.954),
      ('you', 0.994),
      ('can', 0.992),
      ('do', 0.995),
      ('for', 0.993),
      ('your', 0.966),
      ('country.', 0.810),
    ];

    expect(words.length, expectedWordConfidences.length);
    for (int i = 0; i < words.length; i++) {
      expect(words[i].word, expectedWordConfidences[i].$1);
      // Allow 10% tolerance for confidence values
      final expectedConf = expectedWordConfidences[i].$2;
      expect(words[i].confidence, closeTo(expectedConf, expectedConf * 0.1));
    }

    // Test 3: WITH segment confidence (single run)
    debugPrint(
      '\n=== Test 3: transcribe() WITH segment confidence (1 run) ===',
    );
    final resultSegment = await whisper.transcribe(
      audioData,
      getSegmentDetails: true,
      maxOutputTokens: 32,
    );
    final transcript3 = (resultSegment as Ok<TranscriptionResult>).value.text;
    final segmentConfidences = resultSegment.value.confidences;

    debugPrint('Transcript: $transcript3');
    debugPrint(
      'Segment confidence: ${segmentConfidences?.first.toStringAsFixed(3) ?? "null"}',
    );

    expect(transcript1, expectedTranscript);
    expect(transcript2, expectedTranscript);
    expect(transcript3, expectedTranscript);
    expect(segmentConfidences, isNotNull);
    expect(segmentConfidences!.length, 1);
    // Allow 10% tolerance for segment confidence
    expect(segmentConfidences.first, closeTo(0.839, 0.839 * 0.1));
  });
}
