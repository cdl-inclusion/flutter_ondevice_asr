import 'dart:convert';
import 'dart:developer' as dev;
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter_ondevice_asr/util/utils.dart';
import 'package:logging/logging.dart';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';

import '../../common/audio_constants.dart';
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

/// The result of a (sub-range) RNN-T greedy decode: the emitted [tokens] plus the final
/// LSTM state (`h`, `c`, each `[layers·hidden]` flattened) and last emitted `label`, so a
/// streaming caller can persist them and resume decoding on the next audio chunk. One-shot
/// decoding keeps only [tokens] and discards the state.
typedef _RnntDecodeState = ({List<_Token> tokens, Float32List h, Float32List c, int label});

/// Abstract base for the on-device FastConformer transcribers (not instantiable).
/// Use [FastConformerCtcTranscriber], [FastConformerRnntTranscriber], or
/// [FastConformerHybridTranscriber].
///
/// Both heads share one artifact dir with a single, shared super-encoder
/// (`super_encoder.onnx`: waveform -> encoder_out, mel preprocessor baked in) and a
/// per-head graph (`ctc_decoder.onnx` / `decoder_joint.onnx`). The base owns the shared
/// encoder session and the head-agnostic pieces — the tokenizer, running the encoder,
/// detok, word grouping (confidence + timestamps), and segment confidence. `loadModel`
/// parses the common `meta.json` + `tokens.txt`, loads the shared encoder, then defers
/// the head graph to [_loadHead]; `transcribe` runs the encoder once and hands the
/// outputs to [_decodeFromEncoder].
abstract class FastConformerTranscriber implements Transcriber {
  static const int sampleRate = kSampleRate;

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
  ///
  /// [segmentEnd] and [fastDecode] are the caller's flags from [transcribe], forwarded
  /// so a head-picking implementation (the hybrid) can route on them; the single-head
  /// transcribers ignore both.
  List<_Token> _decodeFromEncoder(
    List<OrtValue?> encOutputs, {
    required bool withConfidence,
    required bool segmentEnd,
    required bool fastDecode,
  });

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
    final loadModelTask = dev.TimelineTask()..start('load_model');
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
    loadModelTask.finish();
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
    bool fastDecode = false, // single-head transcribers have one path; only hybrid uses it
  }) async {
    final transcribeTask = dev.TimelineTask()..start('transcribe');
    if (!_loaded) {
      return Result.error(Exception('Call loadModel() first.'));
    }

    final withConfidence = getWordDetails || getSegmentDetails;
    final tokens = await _runEncoderAndDecode(audio,
        withConfidence: withConfidence,
        segmentEnd: segmentEnd,
        fastDecode: fastDecode);

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
    final transcribeResult =  TranscriptionResult(
        text: transcript,
        isFinal: segmentEnd,
        durationInSeconds: duration,
        timestamp: DateTime.now(),
        words: words,
        segments: (segmentEnd && transcript.isNotEmpty) ? [transcript] : null,
        confidences: segmentConfidences,
    );
    transcribeTask.finish();
    return Result.ok(transcribeResult);
  }

  /// Run the shared super-encoder once, then hand its outputs to the head's
  /// [_decodeFromEncoder]. Encoder outputs are released here after decoding.
  Future<List<_Token>> _runEncoderAndDecode(
    Float32List audio, {
    required bool withConfidence,
    required bool segmentEnd,
    required bool fastDecode,
  }) async {
    final encOutputs = await _runEncoder(audio);
    if (encOutputs == null || encOutputs.isEmpty) return const [];
    try {
      return _decodeFromEncoder(encOutputs,
          withConfidence: withConfidence,
          segmentEnd: segmentEnd,
          fastDecode: fastDecode);
    } finally {
      _releaseAll(encOutputs);
    }
  }

  /// Run the shared super-encoder on [audio] and return its raw outputs
  /// ([0] = encoder_out `[1, D, T_enc]`, [1] = encoded_lengths). The caller owns the
  /// returned OrtValues and must [_releaseAll] them. Factored out of [_runEncoderAndDecode]
  /// so the RNN-T debug hooks can drive the decoder directly off a real encoder output.
  Future<List<OrtValue?>?> _runEncoder(Float32List audio) async {
    dev.Timeline.startSync('run_super_encoder');
    final runOptions = OrtRunOptions();
    final audioTensor =
        OrtValueTensor.createTensorWithDataList(audio, [1, audio.length]);
    final lengthTensor = OrtValueTensor.createTensorWithDataList(
      Int64List.fromList([audio.length]),
      [1],
    );
    try {
      return await _encoderSession!.runAsync(runOptions, {
        _inWaveform: audioTensor,
        _inLength: lengthTensor,
      });
    } finally {
      audioTensor.release();
      lengthTensor.release();
      runOptions.release();
      dev.Timeline.finishSync();
    }
  }

  @override
  Future<Result<TranscriptionResult>> transcribeFile(
    String path, {
    bool segmentEnd = true,
    bool getWordDetails = false,
    bool getSegmentDetails = false,
    int? maxOutputTokens,
    bool fastDecode = false,
  }) async {
    final audio = await compute(Audio.instance.loadAudio, path);
    return transcribe(
      audio,
      segmentEnd: segmentEnd,
      getWordDetails: getWordDetails,
      getSegmentDetails: getSegmentDetails,
      maxOutputTokens: maxOutputTokens,
      fastDecode: fastDecode,
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

  /// Test hook: detokenize token ids to text via the loaded tokenizer (mirrors the runtime
  /// detok used by [transcribe]). Lets streaming/parity tests compare on text, not ids.
  @visibleForTesting
  String debugDetok(List<int> ids) => _tokenizer.decodeIds(ids);

  void _release(OrtValue? v) {
    if (v is OrtValueTensor) v.release();
  }

  void _releaseAll(List<OrtValue?> outputs) {
    for (final o in outputs) {
      _release(o);
    }
  }
}

/// CTC head implementation (session + greedy decode), shared by
/// [FastConformerCtcTranscriber] and [FastConformerHybridTranscriber].
///
/// Loads the standalone `ctc_decoder.onnx` (encoder_out -> logprobs) on top of the shared
/// super-encoder. CTC output is already log-softmax, so that per-token confidence is
/// free (`exp(logProb)`). The encoder_out OrtValue is fed straight into `ctc_decoder` —
/// no Dart round-trip for the intermediate.
mixin _CtcHead on FastConformerTranscriber {
  OrtSession? _ctcSession;
  String _ctcIn = 'encoder_out';

  Future<Result<void>> _loadCtcHead(
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

  List<_Token> _ctcDecodeFromEncoder(List<OrtValue?> encOutputs,
      {required bool withConfidence}) {
    dev.Timeline.startSync('decode_from_encoder');
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
    final tokens = _ctcGreedyDecode(frames);
    _releaseAll(outs);
    dev.Timeline.finishSync();
    return tokens;
  }

  /// Greedy CTC decode over the log-prob frames:
  /// (1) argmax per frame;
  /// (2) collapse repeats (keyed on the raw previous id so "a a" -> 'a' but
  ///     "a blank a" -> 'a a'), drop blank.
  /// Each emitted token carries its log-prob at the emitting frame and that frame index.
  List<_Token> _ctcGreedyDecode(List frames) {
    dev.Timeline.startSync('decode');
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
    dev.Timeline.finishSync();
    return out;
  }

  void _releaseCtcHead() {
    _ctcSession?.release();
  }
}

/// FastConformer CTC transcriber.
class FastConformerCtcTranscriber extends FastConformerTranscriber with _CtcHead {
  @override
  Future<Result<void>> _loadHead(
          String modelDirectory, Map<String, dynamic> meta) =>
      _loadCtcHead(modelDirectory, meta);

  @override
  List<_Token> _decodeFromEncoder(
    List<OrtValue?> encOutputs, {
    required bool withConfidence,
    required bool segmentEnd, // ignored: CTC is the only head
    required bool fastDecode, // ignored: CTC is the only head
  }) =>
      _ctcDecodeFromEncoder(encOutputs, withConfidence: withConfidence);

  @override
  void dispose() {
    _releaseCtcHead();
    super.dispose();
  }
}

/// RNN-T (transducer) head implementation, shared by
/// [FastConformerRnntTranscriber] and [FastConformerHybridTranscriber].
///
/// Loads the fused `decoder_joint.onnx` (prediction-net LSTM + joint) on top of the
/// shared super-encoder. A monotonic stateful greedy loop: blank advances time; a
/// non-blank emits a token and advances the label + LSTM (h, c) state, capped by
/// [maxSymbolsPerStep]. The joint emits RAW logits, so per-token confidence needs a
/// softmax (only run when details are requested).
mixin _RnntHead on FastConformerTranscriber {
  /// Cap on tokens emitted per encoder frame (prevents an infinite loop on a
  /// pathological joint). Mirrors NeMo's `max_symbols_per_step`.
  int maxSymbolsPerStep = 10;

  OrtSession? _decoderJointSession;
  int _predHidden = 640;
  int _predLayers = 1;
  // decoder_joint input names, bound from meta (in graph order:
  // encoder_outputs, targets, target_length, input_state_1, input_state_2).
  late List<String> _djIn;

  // ---------------------------------------------------- incremental streaming state
  // Incremental streamign via hunked offline encoder + stateful RNN-T decoder with lookahead hold-back.
  // Calibrated defaults set: commit 0.8 s/step, 1.6 s left context,
  // 0.48 s lookahead, drop = round(left_present / samples_per_frame). 
  // Note: we might want to re-calibrate at some point. 
  //
  // The CALLER owns the segment audio buffer (e.g. StreamingTranscriber's speech buffer) and passes the whole
  // segment-so-far to each [streamDecode]; the session keeps only decoder state + a commit
  // cursor into that buffer. One session at a time per transcriber instance.
  Float32List? _sH; // persisted LSTM state (flattened [layers·hidden]); null = no session
  Float32List? _sC;
  int _sLabel = 0;
  int _sCommitted = 0; // sample index into the segment buffer committed so far
  int _sChunkN = 0, _sLeftN = 0, _sLookN = 0; // window sizes in samples
  final List<_Token> _sTokens = <_Token>[]; // committed tokens this session

  /// Samples of raw audio per encoder frame = subsampling × window_stride (= 8 × 160 = 1280
  /// = 80 ms @ 16 kHz). Derived from the base's per-frame stride so it tracks meta.json.
  int get _samplesPerFrame =>
      (_secondsPerFrame * FastConformerTranscriber.sampleRate).round();

  /// Begin a streaming segment: reset persisted `(h, c)`, label, the commit cursor, and the
  /// window geometry. Call once per VAD segment, before the first [streamDecode].
  void streamReset({
    double chunkS = 0.8,
    double leftContextS = 1.6,
    double lookaheadS = 0.48,
  }) {
    final stateLen = _predLayers * _predHidden;
    _sH = Float32List(stateLen);
    _sC = Float32List(stateLen);
    _sLabel = _blankId;
    _sCommitted = 0;
    _sTokens.clear();
    const sr = FastConformerTranscriber.sampleRate;
    _sChunkN = (chunkS * sr).round();
    _sLeftN = (leftContextS * sr).round();
    _sLookN = (lookaheadS * sr).round();
  }

  /// Decode the current streaming segment from its audio-so-far ([segmentAudio], where sample
  /// 0 = segment start). Commits as many whole chunks as the buffer now allows — each needs
  /// chunk + lookahead of audio past the commit cursor — re-encoding only the
  /// `[left | chunk | lookahead]` window per step and continuing the persisted `(h, c, label)`.
  /// Set [flush] at segment end to commit the remaining tail (which has no further right
  /// context). Returns the running committed token list (the partial). Idempotent as
  /// [segmentAudio] grows across calls — already-committed audio is skipped via the cursor.
  /// Mirrors the Python reference `stream_ids`. [streamReset] must have been called first.
  Future<List<_Token>> streamDecode(Float32List segmentAudio,
      {required bool flush}) async {
    final spf = _samplesPerFrame;
    final bufEnd = segmentAudio.length;
    while (true) {
      final commitEnd = flush ? bufEnd : (_sCommitted + _sChunkN);
      if (commitEnd <= _sCommitted) break; // nothing new to commit
      if (!flush && commitEnd + _sLookN > bufEnd) break; // not enough lookahead yet

      final winStart = max(0, _sCommitted - _sLeftN);
      final winEnd = flush ? bufEnd : min(bufEnd, commitEnd + _sLookN);
      final window = Float32List.sublistView(segmentAudio, winStart, winEnd);

      final enc = await _runEncoder(window);
      if (enc == null || enc.isEmpty) break;
      try {
        final parsed = _parseEncoderOut(enc);
        final tWin = parsed.encLen; // already clamped to available frames
        if (tWin <= 0) {
          _sCommitted = commitEnd;
          continue;
        }

        int drop = ((_sCommitted - winStart) / spf).round();
        if (drop < 0) drop = 0;
        if (drop > tWin) drop = tWin;

        // Commit frames up to commitEnd; hold back the lookahead tail for the next step.
        // On [flush] (segment end) commit all frames. Never commit fewer than we drop.
        int commitFrames = flush
            ? tWin
            : min(tWin, ((commitEnd - winStart) / spf).round());
        if (commitFrames < drop) commitFrames = drop;

        final res = _rnntGreedyDecodeRange(
          parsed.encChannels,
          parsed.encChannels.length,
          drop,
          commitFrames,
          hInit: _sH!,
          cInit: _sC!,
          labelInit: _sLabel,
        );
        _sTokens.addAll(res.tokens);
        _sH = res.h;
        _sC = res.c;
        _sLabel = res.label;
        _sCommitted = commitEnd;
      } finally {
        _releaseAll(enc);
      }
    }
    return List.unmodifiable(_sTokens);
  }

  /// Test hook: full-audio streaming decode → token ids. Resets a session, then feeds the
  /// GROWING segment prefix in [feedSamples]-sized steps (simulating how StreamingTranscriber
  /// passes its accumulating speech buffer, decoupled from the internal chunk size),
  /// finalizes, and returns the committed ids. Mirrors the Python reference `stream_ids`.
  @visibleForTesting
  Future<List<int>> debugStreamDecodeIds(
    Float32List audio, {
    double chunkS = 0.8,
    double leftContextS = 1.6,
    double lookaheadS = 0.48,
    int feedSamples = 1600,
  }) async {
    streamReset(
        chunkS: chunkS, leftContextS: leftContextS, lookaheadS: lookaheadS);
    for (int end = feedSamples; end < audio.length; end += feedSamples) {
      await streamDecode(Float32List.sublistView(audio, 0, end), flush: false);
    }
    await streamDecode(audio, flush: true);
    return _sTokens.map((t) => t.id).toList(growable: false);
  }

  Future<Result<void>> _loadRnntHead(
      String modelDirectory, Map<String, dynamic> meta) async {
    try {
      dev.Timeline.startSync('rnnt.load_head');
      _predHidden = (meta['pred_hidden'] as num?)?.toInt() ?? _predHidden;
      _predLayers = (meta['pred_rnn_layers'] as num?)?.toInt() ?? _predLayers;
      _djIn = _decoderJointInputNames(meta); // throws on a malformed artifact
      final djOnnx = (meta['decoder_joint_onnx'] as String?) ?? 'decoder_joint.onnx';
      _decoderJointSession =
          TranscriberOnnxConfig().createSession(await Utils.loadBytes('$modelDirectory/$djOnnx'));
      dev.Timeline.finishSync();
      return Result.ok(null);
    } catch (e) {
      return Result.error(
        Exception('Failed to load decoder_joint.onnx from <$modelDirectory>: $e'),
      );
    }
  }

  List<_Token> _rnntDecodeFromEncoder(List<OrtValue?> encOutputs,
      {required bool withConfidence}) {
    dev.Timeline.startSync('rnnt.decode_from_encoder');
    final parsed = _parseEncoderOut(encOutputs);
    dev.Timeline.finishSync();
    return _rnntGreedyDecode(
        parsed.encChannels, parsed.encChannels.length, parsed.encLen,
        withConfidence: withConfidence);
  }

  /// Read the shared encoder's outputs into a Dart `[D][T_enc]` view + the valid frame
  /// count. RNN-T slices the encoder output per frame, so it's materialized into a Dart
  /// list. encoder_out is `[1, D, T_enc]`; batch is always 1 on-device -> `encVal[0]` is
  /// `[D][T_enc]`. encoded_lengths (output [1]) caps T to the non-padded frames.
  ({List encChannels, int encLen}) _parseEncoderOut(List<OrtValue?> encOutputs) {
    final encVal = encOutputs[0]?.value as List;
    final encChannels = encVal[0] as List;
    final tEnc = encChannels.isEmpty ? 0 : (encChannels[0] as List).length;
    int encLen = tEnc;
    if (encOutputs.length > 1 && encOutputs[1]?.value != null) {
      encLen = ((encOutputs[1]!.value as List)[0] as num).toInt();
    }
    if (encLen > tEnc) encLen = tEnc;
    return (encChannels: encChannels, encLen: encLen);
  }

  /// Greedy RNN-T transducer decode over the full encoder output with zero initial state.
  /// Thin wrapper over [_rnntGreedyDecodeRange] (see it for the loop semantics); returns
  /// only the emitted tokens — the final LSTM state is discarded (one-shot / non-streaming).
  /// [encChannels] is `[D][T_enc]` (encoder_out[0]).
  List<_Token> _rnntGreedyDecode(
    List encChannels,
    int dModel,
    int encLen, {
    bool withConfidence = false,
  }) {
    final stateLen = _predLayers * _predHidden;
    return _rnntGreedyDecodeRange(
      encChannels,
      dModel,
      0,
      encLen,
      hInit: Float32List(stateLen),
      cInit: Float32List(stateLen),
      labelInit: _blankId,
      withConfidence: withConfidence,
    ).tokens;
  }

  /// Greedy RNN-T transducer decode over the frame sub-range `[startFrame, endFrame)` of
  /// [encChannels] (`[D][T_enc]`), **seeded** with an LSTM state (`hInit`/`cInit`, each a
  /// `[layers·hidden]` flattened Float32List) and last `labelInit`, and **returning** the
  /// final state + label so a streaming caller can persist them and resume on the next
  /// chunk. For one-shot decoding: zero state, `labelInit = blank`, range `[0, encLen)`.
  /// SOS = blank (prednet padding_idx).
  ///
  /// Loop semantics (unchanged from before): blank advances time; a non-blank emits a
  /// token and advances the label + LSTM (h, c) state, capped by [maxSymbolsPerStep]. When
  /// [withConfidence] is set each token carries its log-prob = log_softmax(joint logits)[k]
  /// (so exp(logProb) is a probability); otherwise logProb is 0 and no softmax is run.
  ///
  /// The ORT state tensors live entirely inside this call — seeded from the passed data,
  /// read back out at the end via [_readState] (an exact float32 round-trip). Nothing
  /// leaks to the caller, so it is safe to call repeatedly across chunks.
  _RnntDecodeState _rnntGreedyDecodeRange(
    List encChannels,
    int dModel,
    int startFrame,
    int endFrame, {
    required Float32List hInit,
    required Float32List cInit,
    required int labelInit,
    bool withConfidence = false,
  }) {
    dev.Timeline.startSync('rnnt.decode');
    final hyp = <_Token>[];
    final stateLen = _predLayers * _predHidden; // [layers, 1, hidden]
    final stateShape = [_predLayers, 1, _predHidden];

    // LSTM state seeded from the caller (zero for one-shot; the persisted (h, c) across
    // chunks in streaming mode). Copy the incoming data so the caller's Float32Lists are
    // not aliased by the ORT tensors we allocate and release here.
    OrtValueTensor h = OrtValueTensor.createTensorWithDataList(
        Float32List.fromList(hInit), stateShape);
    OrtValueTensor c = OrtValueTensor.createTensorWithDataList(
        Float32List.fromList(cInit), stateShape);

    int label = labelInit;
    final runOptions = OrtRunOptions();
    // target_length is constant [1].
    final targetLenTensor =
        OrtValueTensor.createTensorWithDataList(Int32List.fromList([1]), [1]);

    try {
      for (int t = startFrame; t < endFrame; t++) {
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
              _djIn[3]: h, // LSTM state (seeded / persisted for streaming)
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
      // Read the final LSTM state back out (exact float32 round-trip) for the caller to
      // persist across chunks; one-shot callers ignore it.
      final hOut = _readState(h, stateLen);
      final cOut = _readState(c, stateLen);
      dev.Timeline.finishSync();
      return (tokens: hyp, h: hOut, c: cOut, label: label);
    } finally {
      targetLenTensor.release();
      runOptions.release();
      h.release();
      c.release();
    }
  }

  /// Flatten an LSTM-state OrtValue (`[layers, 1, hidden]`) into a `[layers·hidden]`
  /// Float32List. The tensor is float32 internally, so this round-trip is exact.
  Float32List _readState(OrtValue? state, int len) {
    final out = Float32List(len);
    int i = 0;
    void walk(dynamic v) {
      if (v is List) {
        for (final e in v) {
          walk(e);
        }
      } else if (v is num && i < len) {
        out[i++] = v.toDouble();
      }
    }

    walk(state?.value);
    return out;
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

  void _releaseRnntHead() {
    _decoderJointSession?.release();
  }
}

/// FastConformer RNN-T transcriber. Supports [IncrementalStreaming] (chunked offline encoder +
/// stateful decoder with lookahead hold-back); see the `_RnntHead` streaming methods.
class FastConformerRnntTranscriber extends FastConformerTranscriber
    with _RnntHead
    implements IncrementalStreaming {
  FastConformerRnntTranscriber({int maxSymbolsPerStep = 10}) {
    this.maxSymbolsPerStep = maxSymbolsPerStep;
  }

  // `streamReset` is provided by `_RnntHead`; here we adapt `streamDecode` to the
  // Transcriber-shaped result the StreamingTranscriber consumes.
  @override
  Future<Result<TranscriptionResult>> streamTranscribe(
    Float32List segmentAudio, {
    required bool flush,
  }) async {
    if (!_loaded) return Result.error(Exception('Call loadModel() first.'));
    final tokens = await streamDecode(segmentAudio, flush: flush);
    final text =
        _tokenizer.decodeIds(tokens.map((t) => t.id).toList(growable: false));
    return Result.ok(TranscriptionResult(
      text: text,
      isFinal: flush,
      durationInSeconds: segmentAudio.length / FastConformerTranscriber.sampleRate,
      timestamp: DateTime.now(),
      segments: (flush && text.isNotEmpty) ? [text] : null,
    ));
  }

  @override
  Future<Result<void>> _loadHead(
          String modelDirectory, Map<String, dynamic> meta) =>
      _loadRnntHead(modelDirectory, meta);

  @override
  List<_Token> _decodeFromEncoder(
    List<OrtValue?> encOutputs, {
    required bool withConfidence,
    required bool segmentEnd, // ignored: RNN-T is the only head
    required bool fastDecode, // ignored: RNN-T is the only head
  }) =>
      _rnntDecodeFromEncoder(encOutputs, withConfidence: withConfidence);

  // ---- test hooks (Phase 0 streaming): drive the decoder off a real encoder output ----

  /// One-shot RNN-T decode of [audio] → token ids: runs the shared encoder, then a single
  /// full-range greedy decode with zero initial state. The reference for the streaming
  /// state-plumbing parity check ([debugDecodeIdsChunked]).
  @visibleForTesting
  Future<List<int>> debugDecodeIdsOneShot(Float32List audio) async {
    final encOutputs = await _runEncoder(audio);
    if (encOutputs == null || encOutputs.isEmpty) return const [];
    try {
      final parsed = _parseEncoderOut(encOutputs);
      final tokens = _rnntGreedyDecode(
          parsed.encChannels, parsed.encChannels.length, parsed.encLen);
      return tokens.map((t) => t.id).toList(growable: false);
    } finally {
      _releaseAll(encOutputs);
    }
  }

  /// Chunked RNN-T decode of [audio] → token ids: splits the encoder frame range at
  /// [splitFrames] and decodes each sub-range in turn, **threading the LSTM (h, c) state +
  /// last label** from one sub-range to the next — i.e. the streaming decode path, but run
  /// over a single pre-computed encoder output so the encoder is held constant. With
  /// correct state plumbing this MUST equal [debugDecodeIdsOneShot] exactly, for any split
  /// set. (Phase 0 proves the decoder-state hand-off before the encoder is chunked.)
  @visibleForTesting
  Future<List<int>> debugDecodeIdsChunked(
      Float32List audio, List<int> splitFrames) async {
    final encOutputs = await _runEncoder(audio);
    if (encOutputs == null || encOutputs.isEmpty) return const [];
    try {
      final parsed = _parseEncoderOut(encOutputs);
      final encChannels = parsed.encChannels;
      final dModel = encChannels.length;
      final encLen = parsed.encLen;
      final stateLen = _predLayers * _predHidden;

      // Persisted-across-chunks state (zero at the segment start), threaded below.
      Float32List h = Float32List(stateLen);
      Float32List c = Float32List(stateLen);
      int label = _blankId;
      final ids = <int>[];

      // Sub-range boundaries: the in-range splits (sorted, de-duped) then encLen.
      final bounds = <int>{
        for (final f in splitFrames)
          if (f > 0 && f < encLen) f,
        encLen,
      }.toList()
        ..sort();

      int start = 0;
      for (final end in bounds) {
        if (end <= start) continue;
        final res = _rnntGreedyDecodeRange(encChannels, dModel, start, end,
            hInit: h, cInit: c, labelInit: label);
        ids.addAll(res.tokens.map((t) => t.id));
        h = res.h;
        c = res.c;
        label = res.label;
        start = end;
      }
      return ids;
    } finally {
      _releaseAll(encOutputs);
    }
  }

  @override
  void dispose() {
    _releaseRnntHead();
    super.dispose();
  }
}

/// FastConformer hybrid transcriber — both heads on the the shared super-encoder.
///
/// This works for hybrid FastConformer checkpoints having one encoder, a CTC and an
/// RNN-T head). This loads the shared encoder once plus the two small head graphs
/// (`ctc_decoder.onnx` + `decoder_joint.onnx`) and picks the head per [transcribe] call:
/// partials (`segmentEnd: false`) use the fast non-autoregressive CTC head, finals the
/// accurate RNN-T head. 
/// This logic can be overridden by passing `fastDecode: true` to always decode with CTC.
class FastConformerHybridTranscriber extends FastConformerTranscriber
    with _CtcHead, _RnntHead {
  FastConformerHybridTranscriber({int maxSymbolsPerStep = 10}) {
    this.maxSymbolsPerStep = maxSymbolsPerStep;
  }

  @override
  Future<Result<void>> _loadHead(
      String modelDirectory, Map<String, dynamic> meta) async {
    final ctcResult = await _loadCtcHead(modelDirectory, meta);
    if (ctcResult is Error) {
      return ctcResult;
    }
    return _loadRnntHead(modelDirectory, meta);
  }

  /// The head routing: CTC when [fastDecode] pins it or the call is a partial
  /// (`segmentEnd: false`); the accurate RNN-T head for unpinned finals.
  @override
  List<_Token> _decodeFromEncoder(
    List<OrtValue?> encOutputs, {
    required bool withConfidence,
    required bool segmentEnd,
    required bool fastDecode,
  }) =>
      (fastDecode || !segmentEnd)
          ? _ctcDecodeFromEncoder(encOutputs, withConfidence: withConfidence)
          : _rnntDecodeFromEncoder(encOutputs, withConfidence: withConfidence);

  @override
  void dispose() {
    _releaseCtcHead();
    _releaseRnntHead();
    super.dispose();
  }
}
