// Tests for the StreamingTranscriber whether continous speech exceeding maxSegmentDuration is force-ended
// and the last 100 ms are carried into the next segment as forced-split overlap, so
// the next segment does not start mid-phoneme.
//
// Uses the real Silero VAD (from assets) and an instant StubTranscriber. eosMinSilence is set very
// high so the pause in the JFK clip does not end the segment; the 2 s max-segment cutoff is then the
// only thing that splits the 11 s clip.

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
  const vadChunk = 512;
  const maxSegmentMs = 2000;
  const overlap = sampleRate ~/ 10; // 100 ms carried across a forced split
  final clip = toAbsolutePath('assets/audio/jfk_asknot.wav');

  test('forced splits carry 100 ms of overlap into the next segment', () async {
    final stub = StubTranscriber();
    final streaming = await StreamingTranscriber.create(
      transcriber: stub,
      vadThreshold: 0.5,
      eosMinSilence: 5000, // never end on the clip's pause
      sampleRate: sampleRate,
      enablePartials: false,
      maxSegmentDuration: maxSegmentMs,
    );

    final audio = await Audio.instance.loadAudio(clip);
    for (var i = 0; i < audio.length; i += chunkSize) {
      final end = (i + chunkSize).clamp(0, audio.length);
      await streaming.processAudioChunk(audio.sublist(i, end));
      await Future.delayed(const Duration(milliseconds: 1));
    }
    await streaming.flush();
    await streaming.dispose();

    final finals = stub.finals;
    debugPrint('finals=${finals.length} '
        'lens=${finals.map((f) => (f.audio.length / sampleRate).toStringAsFixed(2)).toList()}');
    debugPrint('device forced finals (offset, samples): '
        '${finals.map((f) => '(${offsetOf(audio, f.audio)}, ${f.audio.length})').toList()}');
    // 11 s clip / ~1.9 s of new audio per forced segment -> several forced finals plus the flush.
    expect(finals.length, greaterThanOrEqualTo(4));
    // No partials were requested, so every recorded call must be a final.
    expect(stub.partials, isEmpty);

    final maxSegmentSamples = sampleRate * maxSegmentMs ~/ 1000;
    for (var i = 0; i < finals.length; i++) {
      final f = finals[i].audio;
      expect(offsetOf(audio, f), greaterThanOrEqualTo(0),
          reason: 'final $i must be a verbatim slice of the input');
      if (i < finals.length - 1) {
        // Forced finals are cut at the max-segment boundary (to VAD-chunk granularity).
        expect(f.length, greaterThanOrEqualTo(maxSegmentSamples));
        expect(f.length, lessThan(maxSegmentSamples + vadChunk));
      }
      if (i > 0) {
        // The segment begins with the tail of the previous one.
        final prev = finals[i - 1].audio;
        final tail = prev.sublist(prev.length - overlap);
        expect(f.length, greaterThanOrEqualTo(overlap));
        expect(f.sublist(0, overlap), equals(tail),
            reason: 'final $i must start with the last 100 ms of final ${i - 1}');
      }
    }
  });
}
