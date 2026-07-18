import 'package:flutter/foundation.dart';
import 'package:flutter_ondevice_asr/flutter_ondevice_asr.dart';
import 'package:flutter_ondevice_asr/model/transcription_result.dart';
import 'package:flutter_ondevice_asr/util/audio.dart';
import 'package:flutter_test/flutter_test.dart';

import 'test_utils.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  final testAudioFile = toAbsolutePath('assets/audio/jfk_asknot.wav');
  final modelDirectory = toAbsolutePath(
    'assets/transcribers/whisper/models/whisper_tiny/default_int8',
  );

  debugPrint('Unit test paths (absolute, filesystem-based):');
  debugPrint('  modelDirectory: $modelDirectory');
  debugPrint('  testAudioFile: $testAudioFile');
  const language = 'en';
  const chunkDurationMs = 100;
  const sampleRate = 16000;
  final chunkSize = (sampleRate * chunkDurationMs / 1000).toInt();

  test('streaming with partials enabled and disabled', () async {
    // Load model once
    final whisper = Transcriber.getInstance(TranscriberType.whisper);
    await whisper.loadModel(
      modelDirectory: modelDirectory,
      languageCode: language,
    );

    // Create streaming instance once with all parameters
    final streaming = await StreamingTranscriber.create(
      transcriber: whisper,
      vadThreshold: 0.5,
      eosMinSilence: 300,
      sampleRate: 16000,
      enablePartials: true,
      minPartialDuration: 500,
      maxSegmentDuration: 10000,
    );

    final audioData = await Audio.instance.loadAudio(testAudioFile);

    // Test scenario 1: Partials enabled
    debugPrint('\n=== Test 1: Partials Enabled ===');
    streaming.configure(
      enablePartials: true,
      minPartialDuration: 500,
      maxSegmentDuration: 10000,
    );
    streaming.reset();

    var partials = <TranscriptionResult>[];
    var finals = <TranscriptionResult>[];
    var sub = streaming.transcriptionStream.listen((r) {
      r.isFinal ? finals.add(r) : partials.add(r);
      debugPrint('${r.isFinal ? "FINAL" : "PARTIAL"}: ${r.text}');
    });

    for (int i = 0; i < audioData.length; i += chunkSize) {
      final end = (i + chunkSize).clamp(0, audioData.length);
      await streaming.processAudioChunk(audioData.sublist(i, end));
      await Future.delayed(const Duration(milliseconds: 10));
    }
    await streaming.flush();
    await Future.delayed(const Duration(milliseconds: 500));
    await sub.cancel();

    debugPrint('Partials: ${partials.length}, Finals: ${finals.length}');
    expect(partials.length, greaterThan(0));
    expect(finals.length, greaterThan(0));

    // Test scenario 2: Partials disabled
    debugPrint('\n=== Test 2: Finals Only ===');
    streaming.configure(enablePartials: false);
    streaming.reset();

    partials = <TranscriptionResult>[];
    finals = <TranscriptionResult>[];
    sub = streaming.transcriptionStream.listen((r) {
      r.isFinal ? finals.add(r) : partials.add(r);
      debugPrint('${r.isFinal ? "FINAL" : "PARTIAL"}: ${r.text}');
    });

    for (int i = 0; i < audioData.length; i += chunkSize) {
      final end = (i + chunkSize).clamp(0, audioData.length);
      await streaming.processAudioChunk(audioData.sublist(i, end));
      await Future.delayed(const Duration(milliseconds: 10));
    }
    await streaming.flush();
    await Future.delayed(const Duration(milliseconds: 500));
    await sub.cancel();

    debugPrint('Partials: ${partials.length}, Finals: ${finals.length}');
    expect(partials.length, equals(0));
    expect(finals.length, greaterThan(0));

    await streaming.dispose();
  });
}
