
// A StubTranscriber used in tests. It does no actual ASR but it records every call 
// (a copy of the audio, whether it was a final, and timing) and returns a 
// fixed string after an optional [delay].

import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_ondevice_asr/common/result.dart';
import 'package:flutter_ondevice_asr/model/transcription_result.dart';
import 'package:flutter_ondevice_asr/transcriber.dart';

/// One recorded call to [StubTranscriber.transcribe] or [StubIncrementalTranscriber.streamTranscribe].
class StubCall {
  StubCall(this.audio, this.segmentEnd, this.started, this.seq);
  final Float32List audio;
  final bool segmentEnd;
  final DateTime started;

  /// Order in which the stub recorded this call. Used to merge the two record paths into one
  /// sequence; [started] cannot do that job because instant decodes share a timestamp.
  final int seq;
  DateTime? completed;
}


class StubTranscriber implements Transcriber {
  StubTranscriber({this.delay = Duration.zero, this.text = 'stub'});

  final Duration delay;
  final String text;
  final List<StubCall> calls = [];

  /// Number of decodes that started while another was still running. `StreamingTranscriber`
  /// guards against this (`_transcriptionInProgress`), so any value above 0 is a bug in that
  /// guard, not in the test.
  int concurrentCalls = 0;
  int _transcribeInFlight = 0;
  int _seq = 0;

  /// Every recorded call, in the order the stub saw them. [StubIncrementalTranscriber] widens
  /// this to include its `streamTranscribe` calls, so [finals] and [partials] stay correct on
  /// the incremental path too (a final arrives as `streamTranscribe(flush: true)` whenever
  /// `fullRedecodeOnStreamingFinal` is false).
  List<StubCall> get allCalls => calls;

  List<StubCall> get finals => allCalls.where((c) => c.segmentEnd).toList();
  List<StubCall> get partials => allCalls.where((c) => !c.segmentEnd).toList();

  @override
  String? get modelPath => null;

  @override
  Future<Result<void>> loadModel({
    required String modelDirectory,
    required String languageCode,
    double? tokensPerSecond,
  }) async =>
      Result.ok(null);

  @override
  Future<Result<TranscriptionResult>> transcribe(
    Float32List audio, {
    bool segmentEnd = true,
    bool getWordDetails = false,
    bool getSegmentDetails = false,
    int? maxOutputTokens,
    bool fastDecode = false,
  }) async {
    final call =
        StubCall(Float32List.fromList(audio), segmentEnd, DateTime.now(), _seq++);
    calls.add(call);
    if (_transcribeInFlight > 0) concurrentCalls++;
    _transcribeInFlight++;
    try {
      if (delay > Duration.zero) await Future.delayed(delay);
    } finally {
      _transcribeInFlight--;
    }
    call.completed = DateTime.now();
    return Result.ok(TranscriptionResult(
      text: '$text ${calls.length}',
      isFinal: segmentEnd,
      durationInSeconds: audio.length / 16000,
      timestamp: DateTime.now(),
    ));
  }

  @override
  Future<Result<TranscriptionResult>> transcribeFile(
    String path, {
    bool segmentEnd = true,
    bool getWordDetails = false,
    bool getSegmentDetails = false,
    int? maxOutputTokens,
    bool fastDecode = false,
  }) =>
      throw UnimplementedError();

  @override
  void dispose() {}
}

/// Index in [source] at which [slice] starts, or -1. Exact float comparison on the first
/// [probe] samples, then verifies the whole slice. Works because the streaming loop hands the
/// transcriber verbatim copies of the input samples.
int offsetOf(Float32List source, Float32List slice, {int probe = 64}) {
  if (slice.isEmpty || slice.length > source.length) return -1;
  final n = slice.length < probe ? slice.length : probe;
  outer:
  for (var i = 0; i + slice.length <= source.length; i++) {
    for (var j = 0; j < n; j++) {
      if (source[i + j] != slice[j]) continue outer;
    }
    for (var j = n; j < slice.length; j++) {
      if (source[i + j] != slice[j]) continue outer;
    }
    return i;
  }
  return -1;
}

/// A [StubTranscriber] that also implements [IncrementalStreaming], so `StreamingTranscriber`
/// routes partials through `streamTranscribe`. Records every `streamReset()` and flags a reset
/// that arrives while a `streamTranscribe` is still running (the state race in status_quo 3.6).
class StubIncrementalTranscriber extends StubTranscriber
    implements IncrementalStreaming {
  StubIncrementalTranscriber({super.delay, super.text});

  int resets = 0;
  int resetsDuringDecode = 0;
  int _streamInFlight = 0;
  final List<StubCall> streamCalls = [];

  @override
  List<StubCall> get allCalls =>
      [...calls, ...streamCalls]..sort((a, b) => a.seq.compareTo(b.seq));

  @override
  void streamReset() {
    resets++;
    if (_streamInFlight > 0) resetsDuringDecode++;
  }

  @override
  Future<Result<TranscriptionResult>> streamTranscribe(
    Float32List segmentAudio, {
    required bool flush,
  }) async {
    final call = StubCall(
        Float32List.fromList(segmentAudio), flush, DateTime.now(), _seq++);
    streamCalls.add(call);
    if (_streamInFlight > 0) concurrentCalls++;
    _streamInFlight++;
    try {
      if (delay > Duration.zero) await Future.delayed(delay);
    } finally {
      _streamInFlight--;
    }
    call.completed = DateTime.now();
    return Result.ok(TranscriptionResult(
      text: '$text stream ${streamCalls.length}',
      isFinal: flush,
      durationInSeconds: segmentAudio.length / 16000,
      timestamp: DateTime.now(),
    ));
  }
}
