// FastConformer under StreamingTranscriber, both modes:
//
//  * NAIVE (enableIncrementalStreaming: false) — VAD segments, each partial/final
//    re-transcribes the whole speech buffer. The baseline; also what Whisper uses.
//  * INCREMENTAL — the stateful chunked (route-(a)) path: partials decode only the new audio via
//    the bounded-window encoder + persisted decoder state and accumulate. Advertised by RNN-T
//    (persisted (h,c,label)), CTC (persisted `prev` collapse seed), and Hybrid (partials via the
//    chunked CTC path). Final = full offline re-decode by default (RNN-T head for Hybrid).
//

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_ondevice_asr/flutter_ondevice_asr.dart';
import 'package:flutter_ondevice_asr/model/transcription_result.dart';
import 'package:flutter_ondevice_asr/util/audio.dart';
import 'package:flutter_test/flutter_test.dart';

import 'test_utils.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  final testAudioFile = toAbsolutePath('assets/audio/jfk_asknot.wav');
  final modelDir = toAbsolutePath('assets/transcribers/fastconformer/int8');
  const requiredFiles = [
    'super_encoder.onnx',
    'ctc_decoder.onnx',
    'decoder_joint.onnx',
  ];
  final assetPresent =
      requiredFiles.every((f) => File('$modelDir/$f').existsSync());
  const language = 'en';
  const sampleRate = 16000;
  const chunkSize = sampleRate * 100 ~/ 1000; // 100 ms feed
  const anchors = ['fellow', 'americans', 'country'];

  group('FastConformer streaming', () {
    // NAIVE baseline — force incremental off so this stays a true whole-buffer baseline.
    _streamingTest(
      label: 'naive Hybrid (partials CTC / finals RNN-T)',
      type: TranscriberType.fastConformerHybrid,
      incremental: false,
      modelDir: modelDir, testAudioFile: testAudioFile, language: language,
      chunkSize: chunkSize, anchors: anchors, assetPresent: assetPresent,
    );
    _streamingTest(
      label: 'naive RNN-T',
      type: TranscriberType.fastConformerRnnt,
      incremental: false,
      modelDir: modelDir, testAudioFile: testAudioFile, language: language,
      chunkSize: chunkSize, anchors: anchors, assetPresent: assetPresent,
    );
    // INCREMENTAL — RNN-T, CTC, and Hybrid all advertise IncrementalStreaming.
    _streamingTest(
      label: 'incremental RNN-T',
      type: TranscriberType.fastConformerRnnt,
      incremental: true,
      modelDir: modelDir, testAudioFile: testAudioFile, language: language,
      chunkSize: chunkSize, anchors: anchors, assetPresent: assetPresent,
    );
    // Incremental CTC — chunked frame-synchronous partials (`prev`-seeded collapse).
    _streamingTest(
      label: 'incremental CTC',
      type: TranscriberType.fastConformerCtc,
      incremental: true,
      modelDir: modelDir, testAudioFile: testAudioFile, language: language,
      chunkSize: chunkSize, anchors: anchors, assetPresent: assetPresent,
    );
    // Incremental Hybrid — partials via the chunked CTC path; final = full RNN-T re-decode.
    _streamingTest(
      label: 'incremental Hybrid (partials chunked CTC / finals RNN-T)',
      type: TranscriberType.fastConformerHybrid,
      incremental: true,
      modelDir: modelDir, testAudioFile: testAudioFile, language: language,
      chunkSize: chunkSize, anchors: anchors, assetPresent: assetPresent,
    );
  });
}

void _streamingTest({
  required String label,
  required TranscriberType type,
  required bool incremental,
  required String modelDir,
  required String testAudioFile,
  required String language,
  required int chunkSize,
  required List<String> anchors,
  required bool assetPresent,
}) {
  test('$label — partials enabled and disabled', () async {
    final transcriber = Transcriber.getInstance(type);
    await transcriber.loadModel(modelDirectory: modelDir, languageCode: language);

    final streaming = await StreamingTranscriber.create(
      transcriber: transcriber,
      vadThreshold: 0.5,
      eosMinSilence: 1000,
      sampleRate: 16000,
      enablePartials: true,
      minPartialDuration: 500,
      maxSegmentDuration: 10000,
      enableIncrementalStreaming: incremental,
      // default fullRedecodeOnStreamingFinal: true
    );

    final audio = await Audio.instance.loadAudio(testAudioFile);

    // Scenario 1: partials enabled.
    debugPrint('\n=== [$label] partials enabled ===');
    streaming.configure(enablePartials: true, minPartialDuration: 500);
    streaming.reset();
    var run = await _feed(streaming, audio, chunkSize);
    debugPrint('Partials: ${run.partials.length}, Finals: ${run.finals.length}'
        '  joined: ${run.joinedFinals}');

    expect(run.partials.length, greaterThan(0));
    expect(run.finals.length, greaterThan(0));
    for (final a in anchors) {
      expect(_norm(run.joinedFinals), contains(a),
          reason: 'final transcript should contain "$a"');
    }
    // NB: session-level accumulation/decode correctness of the incremental path is proven
    // deterministically in fastconformer_rnnt_streaming_test.dart. We don't assert per-partial
    // monotonicity here — the default full-re-decode final is async/fire-and-forget, so a
    // segment's final can be emitted after the next segment's first partial, which is fine.

    // Scenario 2: partials disabled.
    debugPrint('\n=== [$label] finals only ===');
    streaming.configure(enablePartials: false);
    streaming.reset();
    run = await _feed(streaming, audio, chunkSize);
    debugPrint('Partials: ${run.partials.length}, Finals: ${run.finals.length}');
    expect(run.partials.length, equals(0));
    expect(run.finals.length, greaterThan(0));

    await streaming.dispose();
    transcriber.dispose();
  }, skip: assetPresent ? false : '$label asset not present at $modelDir');
}

Future<_StreamRun> _feed(
    StreamingTranscriber streaming, Float32List audio, int chunkSize) async {
  final partials = <TranscriptionResult>[];
  final finals = <TranscriptionResult>[];
  final sub = streaming.transcriptionStream.listen((r) {
    (r.isFinal ? finals : partials).add(r);
    debugPrint('${r.isFinal ? "FINAL" : "PARTIAL"}: ${r.text}');
  });

  for (int i = 0; i < audio.length; i += chunkSize) {
    final end = (i + chunkSize).clamp(0, audio.length);
    await streaming.processAudioChunk(audio.sublist(i, end));
    await Future.delayed(const Duration(milliseconds: 10));
  }
  await streaming.flush();
  await Future.delayed(const Duration(milliseconds: 500));
  await sub.cancel();
  return _StreamRun(partials, finals);
}

class _StreamRun {
  _StreamRun(this.partials, this.finals);
  final List<TranscriptionResult> partials;
  final List<TranscriptionResult> finals;
  String get joinedFinals => finals.map((r) => r.text).join(' ');
}

String _norm(String s) => s
    .toLowerCase()
    .replaceAll(RegExp(r'[^a-z0-9\s]'), ' ')
    .replaceAll(RegExp(r'\s+'), ' ')
    .trim();
