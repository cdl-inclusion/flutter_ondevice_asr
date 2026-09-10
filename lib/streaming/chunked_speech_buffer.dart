import 'dart:collection';

import 'package:flutter/foundation.dart';

class ChunkedSpeechBuffer {
  final Queue<Float32List> _chunks = Queue();
  int _totalSamples = 0;

  int get length => _totalSamples;

  void addChunk(Float32List chunk) {
    _chunks.addLast(chunk);
    _totalSamples += chunk.length;
  }

  void clear() {
    _chunks.clear();
    _totalSamples = 0;
  }

  /// Remove exactly [samples] samples from the front (or everything, if fewer available).
  /// Whole chunks are dropped first; a chunk straddling the cut is replaced by its tail, so the
  /// result is sample-exact. This leaves a 100ms pre-roll.
  void removeFromFront(int samples) {
    var remaining = samples;
    while (_chunks.isNotEmpty && _chunks.first.length <= remaining) {
      final removed = _chunks.removeFirst().length;
      remaining -= removed;
      _totalSamples -= removed;
    }
    if (remaining > 0 && _chunks.isNotEmpty) {
      final first = _chunks.removeFirst();
      _chunks.addFirst(first.sublist(remaining));
      _totalSamples -= remaining;
    }
  }

  /// Flatten to Float32List for transcription
  Float32List toFloat32List() {
    final out = Float32List(_totalSamples);
    int offset = 0;
    for (final chunk in _chunks) {
      out.setAll(offset, chunk);
      offset += chunk.length;
    }
    return out;
  }
}
