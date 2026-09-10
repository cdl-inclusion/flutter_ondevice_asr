// StreamingTranscriber: a VAD 'start' that arrives while an incremental decode of the previous
// segment is still running must not reset the streamer under that decode. The
// reset is deferred until the in-flight chain completes; the segmentation itself is unchanged.
//
// Real Silero VAD, JFK clip fed far above real time through a StubIncrementalTranscriber whose
// decode takes 1.5 s, so every VAD 'end' and the following 'start' land during a decode.

import 'package:flutter/foundation.dart';
import 'package:flutter_ondevice_asr/flutter_ondevice_asr.dart';
import 'package:flutter_ondevice_asr/util/audio.dart';
import 'package:flutter_test/flutter_test.dart';

import 'stub_transcriber.dart';
import 'test_utils.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const sampleRate = 16000;
  const chunkSize = sampleRate * 100 ~/ 1000;
  final clip = toAbsolutePath('assets/audio/jfk_asknot.wav');

  Future<StubIncrementalTranscriber> run(
      StubIncrementalTranscriber stub, Float32List audio) async {
    final streaming = await StreamingTranscriber.create(
      transcriber: stub,
      vadThreshold: 0.5,
      eosMinSilence: 300,
      sampleRate: sampleRate,
      enablePartials: true,
      minPartialDuration: 500,
      maxSegmentDuration: 30000,
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
      // so every VAD 'end' and the 'start' after it landed while a decode was in flight.
      expect(feedWall, lessThan(stub.delay),
          reason: 'clip must be fed faster than one decode so starts land mid-decode');
    }
    return stub;
  }

  test('streamer is never reset while one of its decodes is in flight', () async {
    final audio = await Audio.instance.loadAudio(clip);

    final instant = await run(StubIncrementalTranscriber(), audio);
    final slow = await run(
        StubIncrementalTranscriber(delay: const Duration(milliseconds: 1500)),
        audio);

    debugPrint('instant: finals=${instant.finals.length} partials=${instant.streamCalls.length} '
        'resets=${instant.resets}');
    debugPrint('slow: finals=${slow.finals.length} partials=${slow.streamCalls.length} '
        'resets=${slow.resets} resetsDuringDecode=${slow.resetsDuringDecode}');

    // Precondition: the slow run really had a decode in flight across segment boundaries.
    expect(slow.streamCalls, isNotEmpty);
    expect(slow.finals.length, greaterThanOrEqualTo(2));

    expect(slow.resetsDuringDecode, 0,
        reason: 'streamReset() ran under a running streamTranscribe()');
    // The loop must never run two decodes at once, however slow the decoder is.
    expect(slow.concurrentCalls, 0);
    // Deferred, not dropped: the resets still happen once the in-flight chain completes.
    // Without this a fix that simply removed every streamReset() would pass, and the finals
    // would not notice (each is a full offline re-decode that resets the streamer itself).
    expect(slow.resets, greaterThan(0),
        reason: 'the deferred streamReset() must still run after the decode completes');
    expect(instant.resetsDuringDecode, 0);

    // Segmentation unaffected by the deferral: same sample-exact finals as the instant run.
    List<(int, int)> slices(StubTranscriber s) =>
        [for (final f in s.finals) (offsetOf(audio, f.audio), f.audio.length)];
    expect(slices(slow), equals(slices(instant)));
  });
}
