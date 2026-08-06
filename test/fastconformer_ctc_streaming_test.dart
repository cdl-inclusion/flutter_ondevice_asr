// Streaming CTC tests — the frame-synchronous sibling of fastconformer_rnnt_streaming_test.dart.
//
// Phase 0 — collapse-seed hand-off is exact: decoding one fixed encoder output in N sub-ranges
// while threading the `prev` (last raw argmax) seed must equal the one-shot decode byte-for-byte.
// (CTC's analogue of the RNN-T (h, c) + label state hand-off — here the whole cross-chunk state
// is a single int.)
//
// Phase 2 — chunked-encoder streaming parity: driving the incremental CTC session (chunked offline
// encoder + `prev`-seeded collapse + lookahead hold-back) over jfk must reproduce the one-shot
// decode's TEXT (normalized / WER-identical). Mirrors the RNN-T streaming parity test.

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_ondevice_asr/common/result.dart';
import 'package:flutter_ondevice_asr/models/fastconformer/fastconformer_transcriber.dart';
import 'package:flutter_ondevice_asr/util/audio.dart';
import 'package:flutter_test/flutter_test.dart';

import 'test_utils.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  final testAudioFile = toAbsolutePath('assets/audio/jfk_asknot.wav');
  final modelDir = toAbsolutePath('assets/transcribers/fastconformer/int8');
  const requiredFiles = ['super_encoder.onnx', 'ctc_decoder.onnx'];
  final assetPresent =
      requiredFiles.every((f) => File('$modelDir/$f').existsSync());
  final skip = assetPresent ? false : 'CTC asset not present at $modelDir';

  Future<FastConformerCtcTranscriber> load() async {
    final fc = FastConformerCtcTranscriber();
    expect(await fc.loadModel(modelDirectory: modelDir, languageCode: 'en') is Ok,
        true);
    return fc;
  }

  group('FastConformer CTC streaming', () {
    // ---- Phase 0: collapse-seed plumbing ----
    test('chunked collapse-seeded decode == one-shot decode (same encoder output)',
        () async {
      final fc = await load();
      final audio = await Audio.instance.loadAudio(testAudioFile);

      final oneShot = await fc.debugDecodeIdsOneShotCtc(audio);
      expect(oneShot, isNotEmpty);
      debugPrint('one-shot emitted ${oneShot.length} tokens');

      const splitSets = <List<int>>[
        [70],
        [10, 25, 50],
        [1, 2, 3, 40, 41, 90, 120],
        [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128],
      ];
      for (final splits in splitSets) {
        final chunked = await fc.debugDecodeIdsChunkedCtc(audio, splits);
        expect(chunked, equals(oneShot),
            reason: 'chunked decode with splits=$splits must equal one-shot');
      }
      fc.dispose();
    }, skip: skip);

    // ---- Phase 2: chunked-encoder streaming parity ----
    test('incremental streaming reproduces offline text (calibrated params)',
        () async {
      final fc = await load();
      final audio = await Audio.instance.loadAudio(testAudioFile);

      final offline = fc.debugDetok(await fc.debugDecodeIdsOneShotCtc(audio));
      final streamed = fc.debugDetok(await fc.debugStreamDecodeIdsCtc(audio));
      debugPrint('offline : $offline');
      debugPrint('streamed: $streamed');

      expect(_norm(streamed), equals(_norm(offline)),
          reason: 'streamed text must match offline (normalized / WER-identical)');
      fc.dispose();
    }, skip: skip);

    test('lookahead is necessary (no-lookahead diverges from offline)', () async {
      final fc = await load();
      final audio = await Audio.instance.loadAudio(testAudioFile);

      final offline = _norm(fc.debugDetok(await fc.debugDecodeIdsOneShotCtc(audio)));
      final noLook = _norm(
          fc.debugDetok(await fc.debugStreamDecodeIdsCtc(audio, lookaheadS: 0.0)));

      // Like the RNN-T test, this asserts the offline full-context encoder's right-context
      // truncation at chunk edges shows up without lookahead. It is clip/model-specific — if a
      // future CTC model happens to be robust here, relax/remove rather than force it.
      expect(noLook, isNot(equals(offline)),
          reason: 'without lookahead the chunk edges should break (else lookahead untested)');
      fc.dispose();
    }, skip: skip);

    test('streaming is stable across feed granularity', () async {
      final fc = await load();
      final audio = await Audio.instance.loadAudio(testAudioFile);

      // Different feed step sizes must yield the same committed ids (the commit cursor makes the
      // result independent of how the audio is chopped on the way in).
      final a = await fc.debugStreamDecodeIdsCtc(audio, feedSamples: 800);
      final b = await fc.debugStreamDecodeIdsCtc(audio, feedSamples: 1600);
      final c = await fc.debugStreamDecodeIdsCtc(audio, feedSamples: 5000);
      expect(a, equals(b));
      expect(b, equals(c));
      fc.dispose();
    }, skip: skip);
  });
}

/// lowercase, strip everything but letters/digits/spaces, collapse whitespace, trim.
String _norm(String s) => s
    .toLowerCase()
    .replaceAll(RegExp(r'[^a-z0-9\s]'), ' ')
    .replaceAll(RegExp(r'\s+'), ' ')
    .trim();
