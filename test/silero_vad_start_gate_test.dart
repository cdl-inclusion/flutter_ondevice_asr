// test Silero VAD's start gate. 
// Uses a fake model with a scripted probability sequence, so the test is exact and needs no audio.

import 'dart:typed_data';

import 'package:flutter_ondevice_asr/vad/silero_vad.dart';
import 'package:flutter_test/flutter_test.dart';

/// A [SileroVAD] whose probabilities come from a script instead of the ONNX model.
class ScriptedVAD extends SileroVAD {
  ScriptedVAD(this.script);
  final List<double> script;
  int _i = 0;

  @override
  double call(Float32List x, int sr) => _i < script.length ? script[_i++] : 0.0;

  @override
  void resetStates({int batchSize = 1}) {}
}

List<String> events(List<double> script, {required int minSpeechChunks}) {
  final it = SileroVADIterator(
    model: ScriptedVAD(script),
    threshold: 0.5,
    samplingRate: 16000,
    minSilenceDurationMs: 96, // 3 chunks
    minSpeechChunks: minSpeechChunks,
  );
  final chunk = Float32List(512);
  final out = <String>[];
  for (var i = 0; i < script.length; i++) {
    final e = it.call(chunk);
    if (e != null) out.add('$e@$i');
  }
  return out;
}

void main() {
  final silence = List.filled(10, 0.05);
  final burst = [0.9, 0.9, 0.9]; // 3 chunks = 96 ms, the typical post-reset blip
  final speech = List.filled(12, 0.95); // 384 ms, simulating a real (short) word

  test('a 3-chunk burst is swallowed by a 6-chunk gate but not without a gate', () {
    final script = [...silence, ...burst, ...silence];
    expect(events(script, minSpeechChunks: 6), isEmpty);
    expect(events(script, minSpeechChunks: 0), equals(['start@10', 'end@16']));
  });

  test('a sustained onset passes the gate; start is reported on the 6th chunk', () {
    final script = [...silence, ...speech, ...silence];
    // Without a gate the start is at the first speech chunk (10); with the gate it is reported
    // when the 6th consecutive chunk passes (15). The streaming loop's extra pre-roll of
    // (gate - 1) chunks makes the segment audio identical in both cases.
    expect(events(script, minSpeechChunks: 0), equals(['start@10', 'end@25']));
    expect(events(script, minSpeechChunks: 6), equals(['start@15', 'end@25']));
  });

  test('a dip below the end threshold inside the gate discards the onset, a dip above it does not', () {
    final dipLow = [...silence, 0.9, 0.9, 0.1, ...silence];
    expect(events(dipLow, minSpeechChunks: 6), isEmpty);
    final dipMid = [...silence, 0.9, 0.9, 0.4, 0.9, 0.9, 0.9, 0.9, ...silence];
    expect(events(dipMid, minSpeechChunks: 6), equals(['start@15', 'end@20']));
  });
}
