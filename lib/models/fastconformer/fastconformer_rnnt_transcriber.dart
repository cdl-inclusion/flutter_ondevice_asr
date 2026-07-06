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

/// FastConformer RNN-T transcriber.
///
/// Two ONNX graphs
///   * `encoder.onnx` — **super-encoder** ONNX graph (mel preprocessor baked into graph)
///   * `decoder_joint.onnx` — **fused** prediction-net (LSTM) + joint. 
///     * One call is one greedy step: 
///     * `(encoder frame, prev label, LSTM (h, c) state)` -> `(joint logits, new h, new c)`.
///
/// Decode is a monotonic greedy loop (blank advances time; a non-blank emits a token
/// and advances the label + LSTM state, capped by [maxSymbolsPerStep]) + `tokens.txt` detok.
///
/// Artifact dir (modelDirectory):
///   encoder.onnx        (waveforms[B,T] f32 + waveforms_lens[B] i64 -> encoder_out[B,D,T_enc], encoded_lengths[B])
///   decoder_joint.onnx  (encoder frame + prev label + LSTM state -> logits[1,1,1,V+1], new state)
///   tokens.txt          (`id\tpiece`)
///   meta.json           (head=rnnt_superencoder, blank_id, pred_hidden, pred_rnn_layers,
///                        superencoder_io, decoder_joint_io)
class FastConformerRnntTranscriber implements Transcriber {
  static const int sampleRate = 16000;

  /// Cap on tokens emitted per encoder frame (prevents an infinite loop on a
  /// pathological joint). Mirrors NeMo's `max_symbols_per_step`.
  final int maxSymbolsPerStep;

  FastConformerRnntTranscriber({this.maxSymbolsPerStep = 10});

  final _logger = Logger('FastConformerRnntTranscriber');
  final _tokenizer = FastConformerTokenizer();

  @override
  String? get modelPath => _modelPath;
  String? _modelPath;

  OrtSession? _encoderSession;
  OrtSession? _decoderJointSession;

  int _blankId = 0;
  int _predHidden = 640;
  int _predLayers = 1;

  // Encoder (super-encoder) input names.
  String _inWaveform = 'waveforms';
  String _inLength = 'waveforms_lens';

  // decoder_joint input names, bound from meta (don't hardcode NeMo's).
  // Order: encoder_outputs, targets, target_length, input_state_1, input_state_2.
  late List<String> _djIn;

  // Seconds of audio per encoder frame = subsampling_factor × window_stride, used to
  // turn a frame index into a word timestamp. Default 8 × 10 ms = 80 ms; overridden
  // from meta.json in loadModel.
  double _secondsPerFrame = 0.08;

  @override
  Future<Result<void>> loadModel({
    required String modelDirectory,
    required String languageCode, // language not used for decoding, but required for interface compatibility
    // ignored: RNN-T decoding is frame-synchronous (bounded by encoder frames ×
    // maxSymbolsPerStep), so it needs no per-second token budget.
    double? tokensPerSecond,
  }) async {
    _modelPath = modelDirectory;

    String encoderOnnx = 'encoder.onnx'; // this is a super-encoder as well, as it includes feature extraction
    String decoderJointOnnx = 'decoder_joint.onnx';

    // meta.json -> blank id, prednet dims, IO names.
    try {
      final metaStr = await Utils.loadString('$modelDirectory/meta.json');
      final meta = jsonDecode(metaStr) as Map<String, dynamic>;
      _blankId = (meta['blank_id'] as num).toInt();
      _predHidden = (meta['pred_hidden'] as num?)?.toInt() ?? _predHidden;
      _predLayers = (meta['pred_rnn_layers'] as num?)?.toInt() ?? _predLayers;
      encoderOnnx = (meta['encoder_onnx'] as String?) ?? encoderOnnx;
      decoderJointOnnx = (meta['decoder_joint_onnx'] as String?) ?? decoderJointOnnx;

      final encIo = meta['superencoder_io'] as Map<String, dynamic>?;
      if (encIo != null) {
        _inWaveform = (encIo['input_waveform'] as String?) ?? _inWaveform;
        _inLength = (encIo['input_length'] as String?) ?? _inLength;
      }

      // Encoder frame stride (for word timestamps): subsampling × window_stride.
      final subsampling = (meta['subsampling_factor'] as num?)?.toInt() ?? 8;
      final pp = meta['preprocessor'] as Map<String, dynamic>?;
      final strideSamples = (pp?['n_window_stride'] as num?)?.toInt();
      final sr = (pp?['sample_rate'] as num?)?.toInt() ?? sampleRate;
      if (strideSamples != null && sr > 0) {
        _secondsPerFrame = subsampling * strideSamples / sr;
      }

      _djIn = _decoderJointInputNames(meta);
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

    // ORT sessions (encoder/decoder)
    try {
      final onnxConfig = TranscriberOnnxConfig();
      final encBytes = await Utils.loadBytes('$modelDirectory/$encoderOnnx');
      _encoderSession = onnxConfig.createSession(encBytes);
      final djBytes = await Utils.loadBytes('$modelDirectory/$decoderJointOnnx');
      _decoderJointSession = onnxConfig.createSession(djBytes);
    } catch (e) {
      return Result.error(
        Exception('Failed to load ONNX graphs from <$modelDirectory>: $e'),
      );
    }

    _logger.finer(
      'Loaded FastConformer RNN-T super-encoder: languageCode=$languageCode, '
      'blank=$_blankId, pred_hidden=$_predHidden, layers=$_predLayers, '
      'vocab=${_tokenizer.vocabSize}, enc_in=($_inWaveform, $_inLength), dj_in=$_djIn',
    );
    return Result.ok(null);
  }

  /// Pull the 5 decoder_joint input names from meta.json (in graph order).
  /// Our converter always writes `decoder_joint_io`, so a missing/short entry means a malformed or
  /// mismatched artifact.
  List<String> _decoderJointInputNames(Map<String, dynamic> meta) {
    final djIo = meta['decoder_joint_io'] as Map<String, dynamic>?;
    final inputs = djIo?['inputs'] as List?;
    if (inputs == null || inputs.length < 5) {
      throw StateError(
        'meta.json decoder_joint_io.inputs is missing or has fewer than 5 entries '
        '(got ${inputs?.length}); cannot bind decoder_joint input names',
      );
    }
    return inputs
        .map((i) => (i as Map<String, dynamic>)['name'] as String)
        .toList(growable: false);
  }

  @override
  Future<Result<TranscriptionResult>> transcribe(
    Float32List audio, {
    bool segmentEnd = true,
    bool getWordDetails = false,
    bool getSegmentDetails = false,
    int? maxOutputTokens,
  }) async {
    if (_encoderSession == null || _decoderJointSession == null) {
      return Result.error(Exception('Call loadModel() first.'));
    }

    // (1) Super-encoder: raw waveform -> encoder embeddings
    final runOptions = OrtRunOptions();
    final audioTensor = OrtValueTensor.createTensorWithDataList(audio, [
      1,
      audio.length,
    ]);
    final lengthTensor = OrtValueTensor.createTensorWithDataList(
      Int64List.fromList([audio.length]),
      [1],
    );

    // encChannels[d][t] = encoder_out[0, d, t]; encLen = valid time steps.
    List encChannels;
    int encLen;
    List<OrtValue?>? encOutputs;
    try {
      encOutputs = await _encoderSession!.runAsync(runOptions, {
        _inWaveform: audioTensor,
        _inLength: lengthTensor,
      });
      if (encOutputs == null || encOutputs.isEmpty) {
        return Result.error(Exception('Encoder produced no output.'));
      }
      // encoder_out: [1, D, T_enc]. Batch is always 1 for on-device inference:
      // batch=1 ([1, audio.length]) and output's leading dim is 1 (encVal[0] drops it to [D][T_enc]).
      final encVal = encOutputs[0]?.value as List;
      encChannels = encVal[0] as List; // [D][T_enc]
      // encoded_lengths: [1]
      final tEnc = (encChannels.isEmpty) ? 0 : (encChannels[0] as List).length;
      if (encOutputs.length > 1 && encOutputs[1]?.value != null) {
        final lenVal = encOutputs[1]!.value as List;
        encLen = (lenVal[0] as num).toInt();
      } else {
        encLen = tEnc;
      }
      if (encLen > tEnc) encLen = tEnc;
    } finally {
      audioTensor.release();
      lengthTensor.release();
      // Release encoder outputs now that values are copied into Dart lists.
      if (encOutputs != null) {
        for (final out in encOutputs) {
          if (out is OrtValueTensor) out.release();
        }
      }
      runOptions.release();
    }

    final dModel = encChannels.length;

    // (2) Monotonic greedy decode over the fused decoder_joint. Per-token confidence
    // (a softmax over the joint logits) is only computed when details are requested.
    final withConfidence = getWordDetails || getSegmentDetails;
    final tokens =
        _greedyDecode(encChannels, dModel, encLen, withConfidence: withConfidence);
    final transcript = _tokenizer.decodeIds(
      tokens.map((t) => t.id).toList(growable: false),
    );

    List<Word>? words;
    List<double>? segmentConfidences;
    if (getWordDetails) {
      words = _buildWords(tokens);
    }
    if (getSegmentDetails && tokens.isNotEmpty) {
      // Geometric mean of token probabilities = exp(mean log-prob).
      final meanLogProb =
          tokens.map((t) => t.logProb).reduce((a, b) => a + b) / tokens.length;
      segmentConfidences = [exp(meanLogProb)];
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

  /// Greedy RNN-T transducer decode:
  /// * blank advances time
  /// * a non-blank emits a token and advances the label + LSTM (h, c) state
  /// * capped at [maxSymbolsPerStep] symbols per frame.
  ///
  /// [encChannels] is `[D][T_enc]` (encoder_out[0]). SOS = blank (prednet padding_idx).
  /// Returns each emitted token with its frame index (for timestamps) and, when
  /// [withConfidence] is set, its log-prob = log_softmax(joint logits)[k] (so
  /// exp(logProb) is a probability); otherwise logProb is 0 and no softmax is run.
  List<({int id, double logProb, int frame})> _greedyDecode(
    List encChannels,
    int dModel,
    int encLen, {
    bool withConfidence = false,
  }) {
    final hyp = <({int id, double logProb, int frame})>[];
    final stateLen = _predLayers * _predHidden; // [layers, 1, hidden]
    final stateShape = [_predLayers, 1, _predHidden];

    // LSTM state, fed back across greedy steps (zero-initialized). 
    // For future streaming mode, (h, c) can be persisted across transcribe() calls instead of
    // reset here, to carry decoder context between consecutive audio chunks.
    OrtValueTensor h = OrtValueTensor.createTensorWithDataList(
      Float32List(stateLen),
      stateShape,
    );
    OrtValueTensor c = OrtValueTensor.createTensorWithDataList(
      Float32List(stateLen),
      stateShape,
    );

    int label = _blankId;
    final runOptions = OrtRunOptions();
    // target_length is constant [1].
    final targetLenTensor = OrtValueTensor.createTensorWithDataList(
      Int32List.fromList([1]),
      [1],
    );

    try {
      for (int t = 0; t < encLen; t++) {
        // encoder frame [1, D, 1].
        final frame = Float32List(dModel);
        for (int d = 0; d < dModel; d++) {
          frame[d] = ((encChannels[d] as List)[t] as num).toDouble();
        }
        final frameTensor = OrtValueTensor.createTensorWithDataList(frame, [
          1,
          dModel,
          1,
        ]);

        try {
          for (int s = 0; s < maxSymbolsPerStep; s++) {
            final targetTensor = OrtValueTensor.createTensorWithDataList(
              Int32List.fromList([label]),
              [1, 1],
            );

            final outputs = _decoderJointSession!.run(runOptions, {
              _djIn[0]: frameTensor,
              _djIn[1]: targetTensor,
              _djIn[2]: targetLenTensor,
              _djIn[3]: h, // LSTM state (see the streaming note above)
              _djIn[4]: c,
            });
            targetTensor.release();

            final res = _argmax(outputs[0]?.value, withLogProb: withConfidence);
            final k = res.index;

            if (k == _blankId) {
              // Discard everything from this step; time advances.
              _releaseAll(outputs);
              break;
            }

            // Emit + advance label and state (NOT time).
            hyp.add((id: k, logProb: res.logProb, frame: t));
            label = k;
            // outputs: [logits, lengths, new_state_1, new_state_2] (positional,
            // matching the Python reference). Keep new states; release the rest.
            final newH = outputs.length > 2 ? outputs[2] : null;
            final newC = outputs.length > 3 ? outputs[3] : null;
            _release(outputs[0]);
            if (outputs.length > 1) _release(outputs[1]);
            if (newH is OrtValueTensor && newC is OrtValueTensor) {
              h.release();
              c.release();
              h = newH;
              c = newC;
            } else {
              // Unexpected output shape — release whatever came back and stop.
              _release(newH);
              _release(newC);
            }
          }
        } finally {
          frameTensor.release();
        }
      }
    } finally {
      targetLenTensor.release();
      runOptions.release();
      h.release();
      c.release();
    }

    return hyp;
  }

  /// Argmax over the (flattened) joint logits of one greedy step, optionally with the
  /// token log-prob. The joint output is `[1, 1, 1, V+1]` (a single non-trivial dim), so
  /// a full flatten recovers the distribution. Unlike CTC, the RNN-T joint emits raw
  /// logits, so [withLogProb] runs a softmax: logProb = log_softmax(logits)[argmax] =
  /// -log(Σ exp(logit_i - max)), i.e. exp(logProb) is the probability of the argmax.
  ({int index, double logProb}) _argmax(dynamic value, {bool withLogProb = false}) {
    final logits = <double>[];
    void walk(dynamic v) {
      if (v is List) {
        for (final e in v) {
          walk(e);
        }
      } else if (v is num) {
        logits.add(v.toDouble());
      }
    }

    walk(value);

    int best = 0;
    double maxVal = logits.isEmpty ? 0.0 : logits[0];
    for (int i = 1; i < logits.length; i++) {
      if (logits[i] > maxVal) {
        maxVal = logits[i];
        best = i;
      }
    }

    double logProb = 0.0;
    if (withLogProb && logits.isNotEmpty) {
      double sumExp = 0.0;
      for (final l in logits) {
        sumExp += exp(l - maxVal);
      }
      logProb = -log(sumExp); // = log_softmax(logits)[best]
    }
    return (index: best, logProb: logProb);
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

  void _release(OrtValue? v) {
    if (v is OrtValueTensor) v.release();
  }

  void _releaseAll(List<OrtValue?> outputs) {
    for (final o in outputs) {
      _release(o);
    }
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
    _encoderSession?.release();
    _decoderJointSession?.release();
  }
}
