// Test pre-roll.
// ChunkedSpeechBuffer.removeFromFront: the streaming loop uses it to keep a fixed pre-roll
// (100 ms = 1600 samples at 16 kHz) of silence in front of speech.
// Removal must therefore be sample-exact even though audio is stored in 512-sample
// VAD chunks.

import 'package:flutter/foundation.dart';
import 'package:flutter_ondevice_asr/streaming/chunked_speech_buffer.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const chunk = 512;
  const keep = 1600; // 100 ms @ 16 kHz

  Float32List filled(int n, double v) => Float32List(n)..fillRange(0, n, v);

  test('trimming to a keep-target keeps exactly the target', () {
    final buf = ChunkedSpeechBuffer();
    // Simulate the streaming loop: add one VAD chunk, trim to keep-target, repeat.
    for (var i = 0; i < 50; i++) {
      buf.addChunk(filled(chunk, i.toDouble()));
      if (buf.length > keep) buf.removeFromFront(buf.length - keep);
      final expected = (i + 1) * chunk < keep ? (i + 1) * chunk : keep;
      expect(buf.length, equals(expected), reason: 'iteration $i');
      expect(buf.toFloat32List().length, equals(expected), reason: 'iteration $i');
    }
    // The retained audio is the most recent 1600 samples: its tail is the last chunk added.
    final out = buf.toFloat32List();
    expect(out.last, equals(49.0));
    expect(out.first, equals(46.0), reason: '1600 samples span the last 3.125 chunks');
  });

  test('a partial removal keeps the tail of the straddling chunk', () {
    final buf = ChunkedSpeechBuffer();
    buf.addChunk(Float32List.fromList(List.generate(chunk, (i) => i.toDouble())));
    buf.addChunk(filled(chunk, -1));
    buf.removeFromFront(100);
    expect(buf.length, equals(2 * chunk - 100));
    final out = buf.toFloat32List();
    expect(out.first, equals(100.0));
    expect(out[chunk - 100 - 1], equals((chunk - 1).toDouble()));
    expect(out[chunk - 100], equals(-1.0));
  });

  test('removes exactly the requested number of samples', () {
    final buf = ChunkedSpeechBuffer();
    for (var i = 0; i < 5; i++) {
      buf.addChunk(filled(chunk, i.toDouble()));
    }
    buf.removeFromFront(chunk * 2 + 10);
    expect(buf.length, equals(chunk * 3 - 10));
    final out = buf.toFloat32List();
    expect(out.first, equals(2.0), reason: 'chunks 0 and 1 removed, 10 samples of chunk 2');
    expect(out.last, equals(4.0));
  });

  test('requesting more than available empties the buffer', () {
    final buf = ChunkedSpeechBuffer();
    buf.addChunk(filled(chunk, 1));
    buf.addChunk(filled(chunk, 2));
    buf.removeFromFront(chunk * 10);
    expect(buf.length, equals(0));
    expect(buf.toFloat32List(), isEmpty);
  });

  test('requesting zero removes nothing', () {
    final buf = ChunkedSpeechBuffer();
    buf.addChunk(filled(chunk, 1));
    buf.removeFromFront(0);
    expect(buf.length, equals(chunk));
  });
}
