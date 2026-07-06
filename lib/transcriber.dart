import 'package:flutter/foundation.dart';

import '../common/result.dart';
import 'model/transcription_result.dart';
import 'models/fastconformer/fastconformer_transcriber.dart';
import 'models/fastconformer/fastconformer_rnnt_transcriber.dart';
import 'models/whisper/whisper_transcriber.dart';
import 'transcriber_type.dart';

abstract class Transcriber {
  static Transcriber getInstance(TranscriberType type) {
    switch (type) {
      case TranscriberType.whisper:
        return WhisperTranscriber();
      case TranscriberType.fastConformer:
        return FastConformerTranscriber();
      case TranscriberType.fastConformerRnnt:
        return FastConformerRnntTranscriber();
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
      });

  Future<Result<TranscriptionResult>> transcribeFile(
      String path, {
        bool segmentEnd = true,
        bool getWordDetails = false,
        bool getSegmentDetails = false,
        int? maxOutputTokens,
      });

  void dispose();
}
