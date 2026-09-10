// Test for StreamingTranscriber when a VAD 'end' that arrives while a decode is still running must not be
// dropped. The segment is snapshotted at the boundary and its final runs once the in-flight
// decode completes, so every utterance gets its own final and the next utterance is not glued on.
//
// Uses the real Silero VAD (from assets) and a StubTranscriber. No actual transcription done, just the 
//timing of the calls to the stub is tested.  The 11 s JFK clip is fed at far
// above real time through (a) an instant stub and (b) a stub whose decode takes 1.5 s, to enforce that in
// (b) every VAD 'end' lands while a decode is running. Segmentation must be identical: same number
// of finals, same sample-exact slices, same order.

import 'package:flutter/foundation.dart';
import 'package:flutter_ondevice_asr/flutter_ondevice_asr.dart';
import 'package:flutter_ondevice_asr/util/audio.dart';
import 'package:flutter_test/flutter_test.dart';

import 'stub_transcriber.dart';
import 'test_utils.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const sampleRate = 16000;
  const chunkSize = sampleRate * 100 ~/ 1000; // 100 ms feed
  final clip = toAbsolutePath('assets/audio/jfk_asknot.wav');

  /// Feed the clip through a StreamingTranscriber backed by [stub]; return the stub.
  Future<StubTranscriber> run(StubTranscriber stub, Float32List audio,
      {required bool partials}) async {
    final streaming = await StreamingTranscriber.create(
      transcriber: stub,
      vadThreshold: 0.5,
      eosMinSilence: 300,
      sampleRate: sampleRate,
      enablePartials: partials,
      minPartialDuration: 500,
      maxSegmentDuration: 30000, // never force a split in this test
    );
    final feedStart = DateTime.now();
    for (var i = 0; i < audio.length; i += chunkSize) {
      final end = (i + chunkSize).clamp(0, audio.length);
      await streaming.processAudioChunk(audio.sublist(i, end));
      await Future.delayed(const Duration(milliseconds: 1));
    }
    final feedWall = DateTime.now().difference(feedStart);
    await streaming.flush();
    await streaming.dispose();
    if (stub.delay > Duration.zero) {
      // Precondition of the scenario: the whole clip went in before the first decode finished,
      // so every VAD 'end' landed while a decode was in flight.
      expect(feedWall, lessThan(stub.delay),
          reason: 'clip must be fed faster than one decode so ends land mid-decode');
    }
    return stub;
  }

  test('segmentation is identical for an instant and a slow decoder', () async {
    final audio = await Audio.instance.loadAudio(clip);

    // Reference: instant decoder, no partials. Finals are produced directly at each VAD end.
    final ref = await run(StubTranscriber(), audio, partials: false);
    // Slow decoder with partials: every VAD end arrives while a decode is running.
    final slow = await run(
        StubTranscriber(delay: const Duration(milliseconds: 1500)), audio,
        partials: true);

    final refFinals = ref.finals;
    final slowFinals = slow.finals;
    debugPrint('ref finals=${refFinals.length} '
        'lens=${refFinals.map((f) => (f.audio.length / sampleRate).toStringAsFixed(2)).toList()}');
    // (offset, samples) per final: compare with the cloud test's output
    // (akolispeech_backend/tests/asr/test_streaming_endpoint.py) for cloud/device parity.
    debugPrint('device finals (offset, samples): '
        '${refFinals.map((f) => '(${offsetOf(audio, f.audio)}, ${f.audio.length})').toList()}');
    debugPrint('slow partials=${slow.partials.length} finals=${slowFinals.length} '
        'lens=${slowFinals.map((f) => (f.audio.length / sampleRate).toStringAsFixed(2)).toList()}');

    expect(refFinals.length, greaterThanOrEqualTo(2),
        reason: 'clip must contain more than one VAD segment for the test to mean anything');
    expect(slow.partials, isNotEmpty, reason: 'a partial must have been in flight');
    // The loop must never run two decodes at once, however slow the decoder is.
    expect(ref.concurrentCalls, 0);
    expect(slow.concurrentCalls, 0);

    // Same number of finals, each the exact same slice of the input, in the same order.
    expect(slowFinals.length, equals(refFinals.length),
        reason: 'a slow decoder must not drop or merge segments');
    for (var i = 0; i < refFinals.length; i++) {
      final refOff = offsetOf(audio, refFinals[i].audio);
      final slowOff = offsetOf(audio, slowFinals[i].audio);
      expect(refOff, greaterThanOrEqualTo(0));
      expect(slowOff, equals(refOff), reason: 'final $i starts at a different sample');
      expect(slowFinals[i].audio.length, equals(refFinals[i].audio.length),
          reason: 'final $i has a different length');
    }

    // The queued finals really were queued: none started before the first partial completed.
    final firstPartialDone = slow.partials.first.completed!;
    for (final f in slowFinals) {
      expect(f.started.isBefore(firstPartialDone), isFalse);
    }
  });
}
