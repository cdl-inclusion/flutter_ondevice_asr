import 'package:flutter/foundation.dart';

import '../common/result.dart';
import 'model/transcription_result.dart';
import 'models/fastconformer/fastconformer_transcriber.dart';
import 'models/whisper/whisper_transcriber.dart';
import 'transcriber_type.dart';

abstract class Transcriber {
  static Transcriber getInstance(TranscriberType type) {
    switch (type) {
      case TranscriberType.whisper:
        return WhisperTranscriber();
      case TranscriberType.fastConformerCtc:
        return FastConformerCtcTranscriber();
      case TranscriberType.fastConformerRnnt:
        return FastConformerRnntTranscriber();
      case TranscriberType.fastConformerHybrid:
        return FastConformerHybridTranscriber();
    }
  }

  String? get modelPath;

  Future<Result<void>> loadModel({
    required String modelDirectory,
    required String languageCode,
    // Optional decoding-budget hint (tokens per second of audio). Only used by
    // free-running autoregressive decoders (Whisper) to cap generation. Non-
    // autoregressive / frame-synchronous backends (FastConformer CTC + RNN-T) ignore it.
    double? tokensPerSecond,
  });

  Future<Result<TranscriptionResult>> transcribe(
    Float32List audio, {
    bool segmentEnd = true,
    bool getWordDetails = false,
    bool getSegmentDetails = false,
    int? maxOutputTokens,
    // Prefer the backend's fastest decoding path over more accurate one (might not be available for all backends).
    // Eg, for hybrid FastConformer, this forces the CTC head even for final segments (otherwise partials -> CTC, finals -> RNN-T).
    bool fastDecode = false,
  });

  Future<Result<TranscriptionResult>> transcribeFile(
    String path, {
    bool segmentEnd = true,
    bool getWordDetails = false,
    bool getSegmentDetails = false,
    int? maxOutputTokens,
    bool fastDecode = false,
  });

  void dispose();
}

/// Opt-in capability for **incremental (stateful) streaming**. Supporting Transcribers can implement it.
/// Functionality: decode only the newly-arrived
/// audio of a segment per call while persisting decoder state across calls, instead of
/// re-transcribing the whole growing buffer each partial. 
/// 
/// A transcriber advertises support by
/// implementing this interface; [StreamingTranscriber] then routes partials through it. Kept
/// separate from [Transcriber] on purpose, so that transcribers without it (e.g. Whisper) are wholly
/// unaffected and keep the naive whole-buffer path. Currently, only FastConformer RNN-T implements it.
abstract interface class IncrementalStreaming {
  /// Start a new streaming segment: reset the persisted decoder state + commit cursor. Call
  /// once at the start of each VAD segment (and after each finalized segment).
  void streamReset();

  /// Incrementally decode the current segment from its audio-so-far ([segmentAudio], sample
  /// 0 = segment start). Commits any whole new chunks now available (partial); [flush] commits
  /// the remaining tail at segment end. Returns the running transcript. Idempotent as
  /// [segmentAudio] grows across calls (already-committed audio is skipped via the cursor).
  Future<Result<TranscriptionResult>> streamTranscribe(
    Float32List segmentAudio, {
    required bool flush,
  });
}
