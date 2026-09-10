// Test that StreamingTranscriber resets the Silero recurrent state after every VAD 'end' and after
// vadStateResetSilenceSeconds of audio without speech.
// Both reset points lie between segments, so a clip must be segmented identically whether it is
// the first thing in the stream or follows a long stretch of silence.

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
  final clipPath = toAbsolutePath('assets/audio/jfk_asknot.wav');

  test('VAD state is reset between segments and after long silence', () async {
    final clip = await Audio.instance.loadAudio(clipPath);
    const silenceSec = 65;
    final silence = Float32List(silenceSec * sampleRate);
    final stream = Float32List(clip.length * 2 + silence.length)
      ..setRange(0, clip.length, clip)
      ..setRange(clip.length, clip.length + silence.length, silence)
      ..setRange(clip.length + silence.length, clip.length * 2 + silence.length, clip);

    final stub = StubTranscriber();
    final streaming = await StreamingTranscriber.create(
      transcriber: stub,
      vadThreshold: 0.5,
      eosMinSilence: 300,
      sampleRate: sampleRate,
      enablePartials: false,
      maxSegmentDuration: 30000,
    );
    for (var i = 0; i < stream.length; i += chunkSize) {
      final end = (i + chunkSize).clamp(0, stream.length);
      await streaming.processAudioChunk(stream.sublist(i, end));
    }
    await streaming.flush();
    final resets = streaming.vadStateResets;
    await streaming.dispose();

    // flush() only decodes when a VAD segment is still open; the clip ends in silence, so
    // every final here comes from a VAD end.
    final vadFinals = stub.finals;
    expect(vadFinals.length, greaterThanOrEqualTo(4));
    final shift = clip.length + silence.length;

    // The clip appears twice, so a final's audio matches in either pass and its offset alone
    // cannot say which pass produced it. Partition by stream order instead: offsets grow
    // within a pass and drop back when pass 2 starts. Splitting on that (rather than on
    // vadFinals.length / 2) keeps the comparison aligned even if the two passes disagree on
    // how many finals they produce - which is exactly the regression this test looks for.
    final firstHalf = Float32List.sublistView(stream, 0, shift);
    final offsets = [for (final f in vadFinals) offsetOf(firstHalf, f.audio)];
    expect(offsets, everyElement(greaterThanOrEqualTo(0)),
        reason: 'every final must be a verbatim slice of the clip');
    var split = offsets.length;
    for (var i = 1; i < offsets.length; i++) {
      if (offsets[i] <= offsets[i - 1]) {
        split = i;
        break;
      }
    }
    expect(split, lessThan(offsets.length),
        reason: 'no second pass detected: the clip after the silence produced no finals');

    final first = [
      for (var i = 0; i < split; i++) (offsets[i], vadFinals[i].audio.length)
    ];
    final second = [
      for (var i = split; i < offsets.length; i++)
        (offsets[i], vadFinals[i].audio.length)
    ];
    debugPrint('device finals pass 1 (offset, samples): $first');
    debugPrint('device finals pass 2: $second');
    expect(second, equals(first),
        reason: 'second pass segmented differently after the resets');

    // One reset per VAD end, plus one per vadStateResetSilenceSeconds of silence between the
    // clips (trailing silence of clip 1 and leading silence of clip 2 count too, so allow one
    // more than the inserted silence alone gives).
    final silenceResets = resets - vadFinals.length;
    final expected =
        silenceSec ~/ StreamingTranscriber.vadStateResetSilenceSeconds.toInt();
    debugPrint('resets=$resets ends=${vadFinals.length} expected>=$expected');
    expect(silenceResets, inInclusiveRange(expected, expected + 1));
  });
}
