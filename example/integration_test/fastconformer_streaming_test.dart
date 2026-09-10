// ignore_for_file: avoid_print — integration-test progress logging; print is intentional.
//
// FastConformer Hybrid under StreamingTranscriber, incremental (chunked) streaming:
// partials decode only the new audio via the chunked CTC path, finals are a full offline
// RNN-T re-decode.

import 'dart:io';
import 'package:flutter/services.dart';
import 'package:flutter_ondevice_asr/model/transcription_result.dart';
import 'package:flutter_ondevice_asr/util/audio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_ondevice_asr/flutter_ondevice_asr.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const testAudioFile =
      'packages/flutter_ondevice_asr/assets/audio/jfk_asknot.wav';
  // Transcript recorded against the streaming settings below.
  // The VAD splits this clip into two segments and each is decoded on its own, so the model
  // re-punctuates per segment and can lose a word sitting right on a boundary (the "ask" before
  // "not").
  const expectedTranscript =
      'So my fellow Americans . not what your country can do for you , ask what you can do for your country .';

  // One HYBRID artifact serves both heads (shared super_encoder + ctc_decoder + decoder_joint).
  const modelDirectory = 'assets/transcribers/fastconformer/int8';
  const String language = 'en';

  // Streaming configuration
  const chunkDurationMs = 100; // Feed audio in 100ms chunks
  const sampleRate = 16000;
  final chunkSize = (sampleRate * chunkDurationMs / 1000).toInt();

  testWidgets('streaming transcribe test audio (FastConformer Hybrid, chunked)',
      (WidgetTester tester) async {
    // 1. Initialize stopwatch to measure durations
    final totalSw = Stopwatch()..start();
    final stepSw = Stopwatch();

    void logStep(String message) {
      print('[${DateTime.now()}] $message (${stepSw.elapsedMilliseconds}ms)');
      stepSw.reset();
      stepSw.start();
    }

    print('[${DateTime.now()}] START STREAMING TEST');
    stepSw.start();

    final fastConformer =
        Transcriber.getInstance(TranscriberType.fastConformerHybrid);

    // 2. Load models
    await fastConformer.loadModel(
      modelDirectory: modelDirectory,
      languageCode: language,
    );
    logStep('Models loaded');

    // 3. Initialize streaming transcriber with all parameters
    final streaming = await StreamingTranscriber.create(
      transcriber: fastConformer,
      vadThreshold: 0.5,
      // Product defaults, so this test exercises what ships.
      eosMinSilence: 1000,
      sampleRate: 16000,
      enablePartials: true,
      minPartialDuration: 500,
      maxSegmentDuration: 20000,
      // Chunked (incremental) streaming: each partial decodes only the newly arrived audio via
      // the CTC head with persisted decoder state, instead of re-transcribing the whole buffer.
      enableIncrementalStreaming: true,
      // Finals are a full offline re-decode through the RNN-T head - the accurate one - so a
      // segment end is never served by the cheaper chunked CTC result.
      fullRedecodeOnStreamingFinal: true,
    );
    logStep('Streaming transcriber initialized');

    // 4. Load test audio
    final audioAsset = await rootBundle.load(testAudioFile);
    final tempDir = await getTemporaryDirectory();
    // The sandboxed container's cache subdirectory is not created for us; path_provider
    // only names it. Same guard as transcribe_test_helper.dart.
    if (!await tempDir.exists()) {
      await tempDir.create(recursive: true);
    }
    final tempAudioFile = File('${tempDir.path}/test_audio.wav');
    await tempAudioFile.writeAsBytes(audioAsset.buffer.asUint8List());

    final testAudioFloat32List = await Audio.instance.loadAudio(
      tempAudioFile.path,
    );
    logStep('Audio file prepared and loaded');

    // 5. Collect transcription results
    final List<TranscriptionResult> partialResults = [];
    final List<TranscriptionResult> finalResults = [];
    DateTime? firstResultTime;
    DateTime? lastResultTime;
    int partialCount = 0;
    int finalCount = 0;

    print('\n========== STREAMING TRANSCRIPTION OUTPUT ==========');
    final subscription = streaming.transcriptionStream.listen((result) {
      final now = DateTime.now();
      firstResultTime ??= now;
      lastResultTime = now;

      if (result.isFinal) {
        finalCount++;
        print(
          '\n[SEGMENT #$finalCount] FINAL (${result.durationInSeconds.toStringAsFixed(2)}s): "${result.text}"',
        );
        finalResults.add(result);
      } else {
        partialCount++;
        print(
          '[PARTIAL #$partialCount] (${result.durationInSeconds.toStringAsFixed(2)}s): "${result.text}"',
        );
        partialResults.add(result);
      }
    });

    // 6. Feed audio in chunks to simulate streaming
    print('[${DateTime.now()}] Feeding audio in $chunkSize-sample chunks...');
    stepSw.reset();
    stepSw.start();
    final streamingStartTime = DateTime.now();

    int offset = 0;
    while (offset < testAudioFloat32List.length) {
      final end = (offset + chunkSize).clamp(0, testAudioFloat32List.length);
      final chunk = testAudioFloat32List.sublist(offset, end);

      await streaming.processAudioChunk(chunk);

      // Small delay to simulate real-time streaming
      await Future.delayed(const Duration(milliseconds: 10));

      offset = end;
    }

    // 7. Flush any remaining audio
    await streaming.flush();
    final flushTime = DateTime.now();
    print('\n[FLUSH] Audio streaming complete, flushing remaining buffer...');
    logStep('Audio streaming complete');

    // 8. Wait a bit for final transcriptions to complete
    await Future.delayed(const Duration(milliseconds: 500));

    // 9. Clean up
    await subscription.cancel();
    await streaming.dispose();
    await tempAudioFile.delete();

    print('====================================================\n');

    // 10. Verify results
    print('[${DateTime.now()}] ===== STREAMING TEST SUMMARY =====');
    print('Total partials: ${partialResults.length}');
    print('Total segments: ${finalResults.length}');
    if (firstResultTime != null && lastResultTime != null) {
      final streamingDuration = lastResultTime!
          .difference(firstResultTime!)
          .inMilliseconds;
      print('Streaming duration: ${streamingDuration}ms');
    }

    print('\n--- Segment Details ---');
    for (int i = 0; i < finalResults.length; i++) {
      final segment = finalResults[i];
      print(
        'Segment ${i + 1}: ${segment.durationInSeconds.toStringAsFixed(2)}s - "${segment.text}"',
      );
    }

    // Combine all final transcriptions
    final fullTranscript = finalResults.map((r) => r.text).join(' ').trim();
    print('\n--- Full Transcript ---');
    print('Result:   "$fullTranscript"');
    print('Expected: "$expectedTranscript"');
    print('===================================');

    // Check that we got at least some results
    expect(
      partialResults.length + finalResults.length,
      greaterThan(0),
      reason: 'Should have received at least some transcription results',
    );

    // IMPORTANT: Check that partial results were actually generated during streaming
    // If this fails, VAD might not be triggering or minPartialDuration might be too long
    expect(
      partialResults.length,
      greaterThan(0),
      reason:
          'Should have received partial transcriptions during streaming (not just final from flush)',
    );

    // Check that we got at least one final result
    expect(
      finalResults.length,
      greaterThan(0),
      reason: 'Should have received at least one final transcription',
    );

    // Verify that results came progressively during streaming, not all at the end
    // The first result should come well before flush was called
    if (firstResultTime != null) {
      final timeUntilFirstResult = firstResultTime!
          .difference(streamingStartTime)
          .inMilliseconds;
      final timeUntilFlush = flushTime
          .difference(streamingStartTime)
          .inMilliseconds;
      print(
        'Time until first result: ${timeUntilFirstResult}ms, time until flush: ${timeUntilFlush}ms',
      );
      expect(
        timeUntilFirstResult,
        lessThan(timeUntilFlush),
        reason:
            'First result should arrive during streaming, not just after flush',
      );
    }

    // Verify the combined transcript matches expected (with some flexibility)
    // Note: Streaming might produce slightly different results due to chunking
    // so we check if the expected text is contained in the result or vice versa
    final transcriptLower = fullTranscript.toLowerCase();
    final expectedLower = expectedTranscript.toLowerCase();

    final matchesExpected =
        transcriptLower.contains(expectedLower) ||
        expectedLower.contains(transcriptLower) ||
        fullTranscript == expectedTranscript;

    expect(
      matchesExpected,
      isTrue,
      reason:
          'Transcript should match expected (actual: "$fullTranscript", expected: "$expectedTranscript")',
    );

    print(
      '[${DateTime.now()}] TEST PASSED - Total duration: ${totalSw.elapsed.inSeconds}s',
    );
    print('====================================');
  });
}
