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

/// One emitted token: its id, the log-prob at the emitting step (`exp(logProb)` is a
/// probability), and the encoder frame index it was emitted at (for timestamps).
typedef _Token = ({int id, double logProb, int frame});

/// Abstract base for the on-device FastConformer transcribers. **Not instantiable** —
/// use [FastConformerCtcTranscriber] or [FastConformerRnntTranscriber].
///
/// Holds everything the two heads share: the `tokens.txt` tokenizer, the raw-waveform
/// encoder input names, the frame stride (for timestamps), and the head-agnostic
/// post-processing — detok, word grouping (confidence + timestamps), and segment
/// confidence. `loadModel` parses the common `meta.json` + `tokens.txt`, then defers the
/// ONNX-graph loading to [_loadGraphs]; `transcribe` defers the actual decode to
/// [_decodeTokens]. Both heads feed a raw waveform (mel preprocessor baked into the graph
/// — no Dart featurizer).
abstract class FastConformerTranscriber implements Transcriber {
  static const int sampleRate = 16000;

  final _logger = Logger('FastConformerTranscriber');
  final _tokenizer = FastConformerTokenizer();

  @override
  String? get modelPath => _modelPath;
  String? _modelPath;

  bool _loaded = false;
  int _blankId = 0;

  // Seconds of audio per encoder frame = subsampling_factor × window_stride, used to
  // turn a frame index into a word timestamp. Default 8 × 10 ms = 80 ms; from meta.json.
  double _secondsPerFrame = 0.08;

  // Encoder input names (both heads take a raw waveform).
  String _inWaveform = 'waveforms';
  String _inLength = 'waveforms_lens';

  // ------------------------------------------------------- head-specific hooks

  /// Load the head's ONNX session(s) from [modelDirectory] (with the parsed [meta]).
  /// Called by [loadModel] after the common meta/tokens parsing.
  Future<Result<void>> _loadGraphs(String modelDirectory, Map<String, dynamic> meta);

  /// Run the head's encoder + decode over [audio], returning the emitted tokens.
  /// [withConfidence] requests per-token log-probs (used only when word/segment details
  /// are requested; for RNN-T this gates an extra softmax, for CTC it's free).
  Future<List<_Token>> _decodeTokens(Float32List audio, {required bool withConfidence});

  // ------------------------------------------------------------- shared: load

  @override
  Future<Result<void>> loadModel({
    required String modelDirectory,
    // Accepted for interface compatibility (clients pass it), but NOT used for decoding:
    // FastConformer is monolingual per checkpoint — the language is fixed by which model
    // + tokens.txt is loaded, with no language/forced-decoder token. Logged for reference.
    required String languageCode,
    double? tokensPerSecond, // ignored: CTC is non-autoregressive; RNN-T is frame-bounded
  }) async {
    _modelPath = modelDirectory;

    // meta.json -> blank id, frame stride, encoder IO names.
    final Map<String, dynamic> meta;
    try {
      final metaStr = await Utils.loadString('$modelDirectory/meta.json');
      meta = jsonDecode(metaStr) as Map<String, dynamic>;
      _blankId = (meta['blank_id'] as num).toInt();
      _secondsPerFrame = _frameStride(meta);
      final io = meta['superencoder_io'] as Map<String, dynamic>?;
      if (io != null) {
        _inWaveform = (io['input_waveform'] as String?) ?? _inWaveform;
        _inLength = (io['input_length'] as String?) ?? _inLength;
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

    // Head-specific ONNX graph(s).
    final graphsResult = await _loadGraphs(modelDirectory, meta);
    if (graphsResult is Error) {
      return graphsResult;
    }

    _loaded = true;
    _logger.finer(
      'Loaded $runtimeType: languageCode=$languageCode, blank=$_blankId, '
      'vocab=${_tokenizer.vocabSize}, in=($_inWaveform, $_inLength)',
    );
    return Result.ok(null);
  }

  @override
  Future<Result<TranscriptionResult>> transcribe(
    Float32List audio, {
    bool segmentEnd = true,
    bool getWordDetails = false, // supported: per-word confidence + timestamps
    bool getSegmentDetails = false, // supported: segment confidence
    int? maxOutputTokens,
  }) async {
    if (!_loaded) {
      return Result.error(Exception('Call loadModel() first.'));
    }

    final withConfidence = getWordDetails || getSegmentDetails;
    final tokens = await _decodeTokens(audio, withConfidence: withConfidence);

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

  /// Encoder frame stride in seconds = subsampling_factor × window_stride (from meta),
  /// used to turn a frame index into a word timestamp. Falls back to 80 ms.
  double _frameStride(Map<String, dynamic> meta) {
    final subsampling = (meta['subsampling_factor'] as num?)?.toInt() ?? 8;
    final pp = meta['preprocessor'] as Map<String, dynamic>?;
    final strideSamples = (pp?['n_window_stride'] as num?)?.toInt();
    final sr = (pp?['sample_rate'] as num?)?.toInt() ?? sampleRate;
    if (strideSamples != null && sr > 0) {
      return subsampling * strideSamples / sr;
    }
    return _secondsPerFrame;
  }

  /// Group emitted tokens into words (split on the `▁` word-boundary marker), with
  /// per-word confidence = the MIN token probability (most conservative, matching the
  /// Whisper transcriber) and start/end timestamps from the emitting frame indices.
  List<Word> _buildWords(List<_Token> tokens) {
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
}

/// FastConformer **CTC** transcriber
/// 
/// One `super_encoder.onnx` graph (mel preprocessor baked in): raw waveform -> CTC
/// log-probs. Greedy collapse + `tokens.txt` detok.
class FastConformerCtcTranscriber extends FastConformerTranscriber {
  OrtSession? _session;

  @override
  Future<Result<void>> _loadGraphs(
      String modelDirectory, Map<String, dynamic> meta) async {
    // The merged CTC output tensor is `ctc/logprobs`; read outputs[0] positionally.
    try {
      final bytes = await Utils.loadBytes('$modelDirectory/super_encoder.onnx');
      _session = TranscriberOnnxConfig().createSession(bytes);
      return Result.ok(null);
    } catch (e) {
      return Result.error(
        Exception('Failed to load super_encoder.onnx from <$modelDirectory>: $e'),
      );
    }
  }

  @override
  Future<List<_Token>> _decodeTokens(Float32List audio,
      {required bool withConfidence}) async {
    // withConfidence is irrelevant for CTC — the log-prob is captured for free below.
    final runOptions = OrtRunOptions();
    final audioTensor =
        OrtValueTensor.createTensorWithDataList(audio, [1, audio.length]);
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

    if (outputs == null || outputs.isEmpty) return const [];
    // logprobs: [1, T_enc, vocab+1] — already log-softmax, so exp(logProb) is a
    // probability in [0, 1] (no softmax needed, unlike Whisper's raw logits).
    final logits = outputs[0]?.value as List;
    final frames = logits[0] as List; // [T_enc][vocab+1]
    final tokens = _decode(frames);
    _releaseAll(outputs);
    return tokens;
  }

  /// Greedy CTC decode over the log-prob frames:
  /// (1) argmax per frame;
  /// (2) collapse repeats (keyed on the raw previous id so "a a" -> 'a' but
  ///     "a blank a" -> 'a a'), drop blank.
  /// Each emitted token carries its log-prob at the emitting frame and that frame index.
  List<_Token> _decode(List frames) {
    final out = <_Token>[];
    int prev = _blankId;
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
      if (best != prev && best != _blankId) {
        out.add((id: best, logProb: bestVal, frame: t));
      }
      prev = best;
    }
    return out;
  }

  @override
  void dispose() {
    _session?.release();
  }
}

/// FastConformer **RNN-T (transducer)** transcriber
///
/// `encoder.onnx` (super-encoder, waveform -> encoder_out) + fused `decoder_joint.onnx`
/// (prediction-net LSTM + joint). A monotonic stateful greedy loop: blank advances time;
/// a non-blank emits a token and advances the label + LSTM (h, c) state, capped by
/// [maxSymbolsPerStep].
class FastConformerRnntTranscriber extends FastConformerTranscriber {
  /// Cap on tokens emitted per encoder frame (prevents an infinite loop on a
  /// pathological joint). Mirrors NeMo's `max_symbols_per_step`.
  final int maxSymbolsPerStep;

  FastConformerRnntTranscriber({this.maxSymbolsPerStep = 10});

  OrtSession? _encoderSession;
  OrtSession? _decoderJointSession;
  int _predHidden = 640;
  int _predLayers = 1;
  // decoder_joint input names, bound from meta (in graph order:
  // encoder_outputs, targets, target_length, input_state_1, input_state_2).
  late List<String> _djIn;

  @override
  Future<Result<void>> _loadGraphs(
      String modelDirectory, Map<String, dynamic> meta) async {
    try {
      _predHidden = (meta['pred_hidden'] as num?)?.toInt() ?? _predHidden;
      _predLayers = (meta['pred_rnn_layers'] as num?)?.toInt() ?? _predLayers;
      final encoderOnnx = (meta['encoder_onnx'] as String?) ?? 'encoder.onnx';
      final decoderJointOnnx =
          (meta['decoder_joint_onnx'] as String?) ?? 'decoder_joint.onnx';
      _djIn = _decoderJointInputNames(meta); // throws on a malformed artifact

      final onnxConfig = TranscriberOnnxConfig();
      _encoderSession =
          onnxConfig.createSession(await Utils.loadBytes('$modelDirectory/$encoderOnnx'));
      _decoderJointSession = onnxConfig
          .createSession(await Utils.loadBytes('$modelDirectory/$decoderJointOnnx'));
      return Result.ok(null);
    } catch (e) {
      return Result.error(
        Exception('Failed to load RNN-T graphs from <$modelDirectory>: $e'),
      );
    }
  }

  @override
  Future<List<_Token>> _decodeTokens(Float32List audio,
      {required bool withConfidence}) async {
    // (1) Super-encoder: raw waveform -> encoder embeddings.
    final runOptions = OrtRunOptions();
    final audioTensor =
        OrtValueTensor.createTensorWithDataList(audio, [1, audio.length]);
    final lengthTensor = OrtValueTensor.createTensorWithDataList(
      Int64List.fromList([audio.length]),
      [1],
    );

    // encChannels[d][t] = encoder_out[0, d, t]; encLen = valid time steps.
    List encChannels = const [];
    int encLen = 0;
    List<OrtValue?>? encOutputs;
    try {
      encOutputs = await _encoderSession!.runAsync(runOptions, {
        _inWaveform: audioTensor,
        _inLength: lengthTensor,
      });
      if (encOutputs == null || encOutputs.isEmpty) return const [];
      // encoder_out: [1, D, T_enc]. Batch is always 1 on-device: we transcribe a single
      // utterance, so the input is built with batch=1 and encVal[0] drops it to [D][T_enc].
      final encVal = encOutputs[0]?.value as List;
      encChannels = encVal[0] as List; // [D][T_enc]
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
      if (encOutputs != null) _releaseAll(encOutputs);
      runOptions.release();
    }

    // (2) Monotonic greedy decode over the fused decoder_joint.
    return _decode(encChannels, encChannels.length, encLen,
        withConfidence: withConfidence);
  }

  /// Greedy RNN-T transducer decode. [encChannels] is `[D][T_enc]` (encoder_out[0]).
  /// SOS = blank (prednet padding_idx). Returns each emitted token with its frame index
  /// (for timestamps) and, when [withConfidence] is set, its log-prob =
  /// log_softmax(joint logits)[k] (so exp(logProb) is a probability); otherwise logProb
  /// is 0 and no softmax is run.
  List<_Token> _decode(
    List encChannels,
    int dModel,
    int encLen, {
    bool withConfidence = false,
  }) {
    final hyp = <_Token>[];
    final stateLen = _predLayers * _predHidden; // [layers, 1, hidden]
    final stateShape = [_predLayers, 1, _predHidden];

    // LSTM state, fed back across greedy steps (zero-initialized). For a future
    // STREAMING mode, (h, c) could be persisted across transcribe() calls instead of
    // reset here, to carry decoder context between consecutive audio chunks.
    OrtValueTensor h =
        OrtValueTensor.createTensorWithDataList(Float32List(stateLen), stateShape);
    OrtValueTensor c =
        OrtValueTensor.createTensorWithDataList(Float32List(stateLen), stateShape);

    int label = _blankId;
    final runOptions = OrtRunOptions();
    // target_length is constant [1].
    final targetLenTensor =
        OrtValueTensor.createTensorWithDataList(Int32List.fromList([1]), [1]);

    try {
      for (int t = 0; t < encLen; t++) {
        // encoder frame [1, D, 1].
        final frame = Float32List(dModel);
        for (int d = 0; d < dModel; d++) {
          frame[d] = ((encChannels[d] as List)[t] as num).toDouble();
        }
        final frameTensor =
            OrtValueTensor.createTensorWithDataList(frame, [1, dModel, 1]);

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

  /// Pull the 5 decoder_joint input names from meta.json (in graph order). The converter
  /// always writes `decoder_joint_io`, so a missing/short entry means a malformed or
  /// mismatched artifact — fail fast rather than binding to guessed names.
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

  /// Argmax over the (flattened) joint logits of one greedy step, optionally with the
  /// token log-prob. The joint output is `[1, 1, 1, V+1]` (a single non-trivial dim), so
  /// a full flatten recovers the distribution. Unlike CTC, the RNN-T joint emits RAW
  /// logits, so [withLogProb] runs a softmax: logProb = log_softmax(logits)[argmax] =
  /// -log(Σ exp(logit_i - max)), i.e. exp(logProb) is the probability of the argmax.
  ({int index, double logProb}) _argmax(dynamic value,
      {bool withLogProb = false}) {
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

  @override
  void dispose() {
    _encoderSession?.release();
    _decoderJointSession?.release();
  }
}
