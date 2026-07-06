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
/// Both heads share one artifact dir with a single, shared super-encoder
/// (`super_encoder.onnx`: waveform -> encoder_out, mel preprocessor baked in) and a
/// per-head graph (`ctc_decoder.onnx` / `decoder_joint.onnx`). The base owns the shared
/// encoder session and the head-agnostic pieces — the tokenizer, running the encoder,
/// detok, word grouping (confidence + timestamps), and segment confidence. `loadModel`
/// parses the common `meta.json` + `tokens.txt`, loads the shared encoder, then defers
/// the head graph to [_loadHead]; `transcribe` runs the encoder once and hands the
/// outputs to [_decodeFromEncoder]. Raw waveform in — no Dart featurizer.
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

  // The SHARED super-encoder (waveform -> encoder_out), owned by the base.
  OrtSession? _encoderSession;
  String _inWaveform = 'waveforms';
  String _inLength = 'waveforms_lens';

  // ------------------------------------------------------- head-specific hooks

  /// Load the head's ONNX graph from [modelDirectory] (with the parsed [meta]). Called by
  /// [loadModel] after the shared encoder + common meta/tokens parsing.
  Future<Result<void>> _loadHead(String modelDirectory, Map<String, dynamic> meta);

  /// Decode the shared encoder's outputs into emitted tokens. [encOutputs] are the raw
  /// `super_encoder.onnx` outputs ([0] = encoder_out, [1] = encoded_lengths); the base
  /// releases them after this returns. [withConfidence] requests per-token log-probs
  /// (RNN-T gates an extra softmax on it; for CTC the log-prob is free).
  List<_Token> _decodeFromEncoder(List<OrtValue?> encOutputs,
      {required bool withConfidence});

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

    // meta.json -> blank id, frame stride, super-encoder IO names + graph name.
    final Map<String, dynamic> meta;
    String superEncoderOnnx = 'super_encoder.onnx';
    try {
      final metaStr = await Utils.loadString('$modelDirectory/meta.json');
      meta = jsonDecode(metaStr) as Map<String, dynamic>;
      _blankId = (meta['blank_id'] as num).toInt();
      _secondsPerFrame = _frameStride(meta);
      superEncoderOnnx = (meta['super_encoder_onnx'] as String?) ?? superEncoderOnnx;
      final io = meta['super_encoder_io'] as Map<String, dynamic>?;
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

    // Shared super-encoder session.
    try {
      final bytes = await Utils.loadBytes('$modelDirectory/$superEncoderOnnx');
      _encoderSession = TranscriberOnnxConfig().createSession(bytes);
    } catch (e) {
      return Result.error(
        Exception('Failed to load $superEncoderOnnx from <$modelDirectory>: $e'),
      );
    }

    // Head-specific ONNX graph.
    final headResult = await _loadHead(modelDirectory, meta);
    if (headResult is Error) {
      return headResult;
    }

    _loaded = true;
    _logger.finer(
      'Loaded $runtimeType: languageCode=$languageCode, blank=$_blankId, '
      'vocab=${_tokenizer.vocabSize}, in=($_inWaveform, $_inLength)',
    );
    return Result.ok(null);
  }

  // -------------------------------------------------------- shared: transcribe

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
    final tokens = await _runEncoderAndDecode(audio, withConfidence: withConfidence);

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

  /// Run the shared super-encoder once, then hand its outputs to the head's
  /// [_decodeFromEncoder]. Encoder outputs are released here after decoding.
  Future<List<_Token>> _runEncoderAndDecode(Float32List audio,
      {required bool withConfidence}) async {
    final runOptions = OrtRunOptions();
    final audioTensor =
        OrtValueTensor.createTensorWithDataList(audio, [1, audio.length]);
    final lengthTensor = OrtValueTensor.createTensorWithDataList(
      Int64List.fromList([audio.length]),
      [1],
    );

    List<OrtValue?>? encOutputs;
    try {
      encOutputs = await _encoderSession!.runAsync(runOptions, {
        _inWaveform: audioTensor,
        _inLength: lengthTensor,
      });
    } finally {
      audioTensor.release();
      lengthTensor.release();
      runOptions.release();
    }

    if (encOutputs == null || encOutputs.isEmpty) return const [];
    try {
      return _decodeFromEncoder(encOutputs, withConfidence: withConfidence);
    } finally {
      _releaseAll(encOutputs);
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
  }

  // ---------------------------------------------------------- shared: helpers

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

/// FastConformer **CTC** transcriber — the fast, non-autoregressive head.
///
/// Loads the standalone `ctc_decoder.onnx` (encoder_out -> logprobs) on top of the shared
/// super-encoder. Because the CTC output is already log-softmax, per-token confidence is
/// free (`exp(logProb)`). The encoder_out OrtValue is fed straight into `ctc_decoder` —
/// no Dart round-trip for the intermediate.
class FastConformerCtcTranscriber extends FastConformerTranscriber {
  OrtSession? _ctcSession;
  String _ctcIn = 'encoder_out';

  @override
  Future<Result<void>> _loadHead(
      String modelDirectory, Map<String, dynamic> meta) async {
    try {
      final ctcIo = meta['ctc_decoder_io'] as Map<String, dynamic>?;
      _ctcIn = (ctcIo?['input_encoder'] as String?) ?? _ctcIn;
      final ctcOnnx = (meta['ctc_decoder_onnx'] as String?) ?? 'ctc_decoder.onnx';
      _ctcSession =
          TranscriberOnnxConfig().createSession(await Utils.loadBytes('$modelDirectory/$ctcOnnx'));
      return Result.ok(null);
    } catch (e) {
      return Result.error(
        Exception('Failed to load ctc_decoder.onnx from <$modelDirectory>: $e'),
      );
    }
  }

  @override
  List<_Token> _decodeFromEncoder(List<OrtValue?> encOutputs,
      {required bool withConfidence}) {
    final encOut = encOutputs[0];
    if (encOut == null) return const [];

    final runOptions = OrtRunOptions();
    List<OrtValue?> outs;
    try {
      // Feed the encoder_out OrtValue straight into ctc_decoder (native handoff — no
      // Dart materialization of the intermediate; only the final logprobs are read).
      outs = _ctcSession!.run(runOptions, {_ctcIn: encOut});
    } finally {
      runOptions.release();
    }

    // logprobs: [1, T_enc, vocab+1] — already log-softmax, so exp(logProb) is a
    // probability in [0, 1] (no softmax needed, unlike Whisper's raw logits).
    final logits = outs[0]?.value as List;
    final frames = logits[0] as List; // [T_enc][vocab+1]
    final tokens = _decode(frames);
    _releaseAll(outs);
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
    _ctcSession?.release();
    super.dispose();
  }
}

/// FastConformer **RNN-T (transducer)** transcriber — the accurate head.
///
/// Loads the fused `decoder_joint.onnx` (prediction-net LSTM + joint) on top of the
/// shared super-encoder. A monotonic stateful greedy loop: blank advances time; a
/// non-blank emits a token and advances the label + LSTM (h, c) state, capped by
/// [maxSymbolsPerStep]. The joint emits RAW logits, so per-token confidence needs a
/// softmax (only run when details are requested).
class FastConformerRnntTranscriber extends FastConformerTranscriber {
  /// Cap on tokens emitted per encoder frame (prevents an infinite loop on a
  /// pathological joint). Mirrors NeMo's `max_symbols_per_step`.
  final int maxSymbolsPerStep;

  FastConformerRnntTranscriber({this.maxSymbolsPerStep = 10});

  OrtSession? _decoderJointSession;
  int _predHidden = 640;
  int _predLayers = 1;
  // decoder_joint input names, bound from meta (in graph order:
  // encoder_outputs, targets, target_length, input_state_1, input_state_2).
  late List<String> _djIn;

  @override
  Future<Result<void>> _loadHead(
      String modelDirectory, Map<String, dynamic> meta) async {
    try {
      _predHidden = (meta['pred_hidden'] as num?)?.toInt() ?? _predHidden;
      _predLayers = (meta['pred_rnn_layers'] as num?)?.toInt() ?? _predLayers;
      _djIn = _decoderJointInputNames(meta); // throws on a malformed artifact
      final djOnnx = (meta['decoder_joint_onnx'] as String?) ?? 'decoder_joint.onnx';
      _decoderJointSession =
          TranscriberOnnxConfig().createSession(await Utils.loadBytes('$modelDirectory/$djOnnx'));
      return Result.ok(null);
    } catch (e) {
      return Result.error(
        Exception('Failed to load decoder_joint.onnx from <$modelDirectory>: $e'),
      );
    }
  }

  @override
  List<_Token> _decodeFromEncoder(List<OrtValue?> encOutputs,
      {required bool withConfidence}) {
    // RNN-T slices the encoder output per frame, so read it into a Dart list.
    // encoder_out: [1, D, T_enc]; batch is always 1 on-device -> encVal[0] is [D][T_enc].
    final encVal = encOutputs[0]?.value as List;
    final encChannels = encVal[0] as List;
    final tEnc = encChannels.isEmpty ? 0 : (encChannels[0] as List).length;
    int encLen = tEnc;
    if (encOutputs.length > 1 && encOutputs[1]?.value != null) {
      encLen = ((encOutputs[1]!.value as List)[0] as num).toInt();
    }
    if (encLen > tEnc) encLen = tEnc;

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
    _decoderJointSession?.release();
    super.dispose();
  }
}
