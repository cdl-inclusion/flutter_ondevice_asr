// TODO combine CTC and RNNT transcriber to reduce scaffolding
// TODO potemtially create a shared class as joint entry point

import 'dart:convert';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter_ondevice_asr/util/utils.dart';
import 'package:logging/logging.dart';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';

import '../../common/result.dart';
import '../../transcriber.dart';
import '../../model/onnx_config.dart';
import '../../model/transcription_result.dart';
import '../../model/word.dart';
import '../../util/audio.dart';
import 'fastconformer_tokenizer.dart';

/// FastConformer CTC transcriber.
///
/// Uses the **super-encoder** ONNX graph (mel preprocessor baked into graph). 
/// Input is raw waveform. CTC produces log-probs. 
/// Greedy CTC + `tokens.txt` detok.
///
/// Artifacts (modelDirectory):
///   super_encoder.onnx  (waveforms[B,T] f32 + waveforms_lens[B] i64 -> logprobs[B,T_enc,V+1])
///   tokens.txt          (`id\tpiece`)
///   meta.json           (head=ctc_superencoder, blank_id, superencoder_io, ...)
class FastConformerTranscriber implements Transcriber {
  static const int sampleRate = 16000;

  final _logger = Logger('FastConformerTranscriber');
  final _tokenizer = FastConformerTokenizer();

  @override
  String? get modelPath => _modelPath;
  String? _modelPath;

  OrtSession? _session;
  int _blankId = 0;
  String _inWaveform = 'waveforms';
  String _inLength = 'waveforms_lens';

  // Seconds of audio per encoder frame = subsampling_factor × window_stride, used to
  // turn a frame index into a word timestamp. Default 8 × 10 ms = 80 ms; overridden
  // from meta.json in loadModel.
  double _secondsPerFrame = 0.08;

  @override
  Future<Result<void>> loadModel({
    required String modelDirectory,
    required String languageCode, // language not used for decoding, but required for interface compatibility
    double? tokensPerSecond, // ignored: CTC is non-autoregressive
  }) async {
    _modelPath = modelDirectory;

    // meta.json -> blank id + IO names.
    try {
      final metaStr = await Utils.loadString('$modelDirectory/meta.json');
      final meta = jsonDecode(metaStr) as Map<String, dynamic>;
      _blankId = (meta['blank_id'] as num).toInt();
      final io = meta['superencoder_io'] as Map<String, dynamic>?;
      if (io != null) {
        _inWaveform = (io['input_waveform'] as String?) ?? _inWaveform;
        _inLength = (io['input_length'] as String?) ?? _inLength;
      }

      // Encoder frame stride (for word timestamps): subsampling × window_stride.
      final subsampling = (meta['subsampling_factor'] as num?)?.toInt() ?? 8;
      final pp = meta['preprocessor'] as Map<String, dynamic>?;
      final strideSamples = (pp?['n_window_stride'] as num?)?.toInt();
      final sr = (pp?['sample_rate'] as num?)?.toInt() ?? sampleRate;
      if (strideSamples != null && sr > 0) {
        _secondsPerFrame = subsampling * strideSamples / sr;
      }
    } catch (e) {
      return Result.error(
        Exception('Failed to load meta.json from <$modelDirectory>: $e'),
      );
    }

    // tokens.txt -> detok vocab.
    final vocabResult = await _tokenizer.loadVocab(
      path: '$modelDirectory/tokens.txt',
    );
    if (vocabResult is Error) {
      return vocabResult;
    }

    // ORT session for super_encoder.onnx
    try {
      final bytes = await Utils.loadBytes('$modelDirectory/super_encoder.onnx');
      final onnxConfig = TranscriberOnnxConfig();
      _session = onnxConfig.createSession(bytes);
    } catch (e) {
      return Result.error(
        Exception('Failed to load ONNX graphs from <$modelDirectory>: $e'),
      );
    }

    _logger.finer(
      'Loaded FastConformer super-encoder: languageCode=$languageCode, '
      'blank=$_blankId, vocab=${_tokenizer.vocabSize}, in=($_inWaveform, $_inLength)',
    );
    return Result.ok(null);
  }

  @override
  Future<Result<TranscriptionResult>> transcribe(
    Float32List audio, {
    bool segmentEnd = true,
    bool getWordDetails = false, // supported: per-word confidence + timestamps from log-probs
    bool getSegmentDetails = false, // supported: segment confidence from log-probs
    int? maxOutputTokens,
  }) async {
    if (_session == null) {
      return Result.error(Exception('Call loadModel() first.'));
    }

    final runOptions = OrtRunOptions();
    final audioTensor = OrtValueTensor.createTensorWithDataList(audio, [
      1,
      audio.length,
    ]);
    final lengthTensor = OrtValueTensor.createTensorWithDataList(
      Int64List.fromList([audio.length]),
      [1],
    );

    List<OrtValue?>? outputs;
    try {
      outputs = await _session!.runAsync(runOptions, {
        _inWaveform: audioTensor,
        _inLength: lengthTensor,
      });
    } finally {
      audioTensor.release();
      lengthTensor.release();
      runOptions.release();
    }

    String transcript = '';
    List<Word>? words;
    List<double>? segmentConfidences;
    if (outputs != null && outputs.isNotEmpty) {
      // logprobs: [1, T_enc, vocab+1] — already log-softmax, so exp(logProb) is a
      // probability in [0, 1] (no softmax needed, unlike Whisper's raw logits).
      final logits = outputs[0]?.value as List;
      final frames = logits[0] as List; // [T_enc][vocab+1]
      final tokens = _ctcGreedyDecode(frames, _blankId);
      transcript = _tokenizer.decodeIds(
        tokens.map((t) => t.id).toList(growable: false),
      );

      if (getWordDetails) {
        words = _buildWords(tokens);
      }
      if (getSegmentDetails && tokens.isNotEmpty) {
        // Geometric mean of token probabilities = exp(mean log-prob).
        final meanLogProb =
            tokens.map((t) => t.logProb).reduce((a, b) => a + b) / tokens.length;
        segmentConfidences = [exp(meanLogProb)];
      }

      for (final out in outputs) {
        if (out is OrtValueTensor) {
          out.release();
        }
      }
    }

    final duration = audio.length / sampleRate;
    return Result.ok(
      TranscriptionResult(
        text: transcript,
        isFinal: segmentEnd,
        durationInSeconds: duration,
        timestamp: DateTime.now(),
        words: words,
        segments: (segmentEnd && transcript.isNotEmpty) ? [transcript] : null,
        confidences: segmentConfidences,
      ),
    );
  }

  @override
  Future<Result<TranscriptionResult>> transcribeFile(
    String path, {
    bool segmentEnd = true,
    bool getWordDetails = false,
    bool getSegmentDetails = false,
    int? maxOutputTokens,
  }) async {
    final audio = await compute(Audio.instance.loadAudio, path);
    return transcribe(
      audio,
      segmentEnd: segmentEnd,
      getWordDetails: getWordDetails,
      getSegmentDetails: getSegmentDetails,
      maxOutputTokens: maxOutputTokens,
    );
  }

  @override
  void dispose() {
    _session?.release();
  }

  /// Greedy CTC decode over the log-prob frames:
  /// (1) argmax per frame;
  /// (2) collapse repeats (keyed on the raw previous id so "a a" -> 'a' but
  ///     "a blank a" -> 'a a'), drop blank.
  ///
  /// Returns each emitted token with its log-prob at the emitting frame (the CTC output
  /// is already log-softmax, so `exp(logProb)` is the token probability) and that frame
  /// index (for timestamps). [frames] is [T_enc][vocab+1] of log-probs.
  List<({int id, double logProb, int frame})> _ctcGreedyDecode(
      List frames, int blankId) {
    final out = <({int id, double logProb, int frame})>[];
    int prev = blankId;
    for (int t = 0; t < frames.length; t++) {
      final row = frames[t] as List;
      int best = 0;
      double bestVal = (row[0] as num).toDouble();
      for (int j = 1; j < row.length; j++) {
        final v = (row[j] as num).toDouble();
        if (v > bestVal) {
          bestVal = v;
          best = j;
        }
      }
      // Standard CTC collapse: emit only on a *change* of argmax. A repeated token is
      // kept only when the model puts a blank between the two frames (so a real "the
      // the" survives), while frame-repeats of one token collapse to a single emission
      // — i.e. no spurious stutter.
      if (best != prev && best != blankId) {
        out.add((id: best, logProb: bestVal, frame: t));
      }
      prev = best;
    }
    return out;
  }

  /// Group emitted tokens into words (split on the `▁` word-boundary marker), with
  /// per-word confidence = the min token probability and start/end timestamps from the emitting frame indices.
  List<Word> _buildWords(List<({int id, double logProb, int frame})> tokens) {
    final words = <Word>[];
    var current = '';
    final confidences = <double>[];
    int? startFrame;
    int endFrame = 0;

    void flush() {
      if (current.trim().isEmpty) return;
      words.add(Word(
        word: current.trim(),
        confidence: confidences.isNotEmpty ? confidences.reduce(min) : -1.0,
        start: (startFrame ?? 0) * _secondsPerFrame,
        end: (endFrame + 1) * _secondsPerFrame,
      ));
    }

    for (final t in tokens) {
      final text = _tokenizer.decodeSingleToken(t.id);
      if (text.isEmpty) continue;
      // A leading `▁` marks a new word: close the current one first.
      if (_tokenizer.tokenStartsNewWord(t.id) && current.trim().isNotEmpty) {
        flush();
        current = '';
        confidences.clear();
        startFrame = null;
      }
      current += text;
      startFrame ??= t.frame;
      endFrame = t.frame;
      confidences.add(exp(t.logProb));
    }
    flush();
    return words;
  }
}
