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
import 'whisper_tokenizer.dart';

class WhisperTranscriber implements Transcriber {
  // Default tokens per second for dynamic calculation (conservative estimate for fast speech)
  static const double defaultTokensPerSecond = 6.0;

  final _logger = Logger('WhisperTranscriber');

  final _tokenizer = WhisperTokenizer();

  @override
  String? get modelPath => _modelPath;
  String? _modelPath;

  OrtSession? superEncoderSession; // Combined preprocessor + encoder
  OrtSession? decoderSession;
  OrtSession? decoderWithPastSession;

  late int sotToken;
  late int eotToken;
  late int noTimestampsToken;
  int? transcribeToken;
  int? languageToken;

  double _tokensPerSecond = defaultTokensPerSecond;

  @override
  Future<Result<void>> loadModel({
    required String modelDirectory,
    required String languageCode,
    double? tokensPerSecond,
  }) async {
    final loadModelTimelineTask = dev.TimelineTask()..start('load_whisper_model');
    _tokensPerSecond = tokensPerSecond ?? defaultTokensPerSecond;
    _modelPath = modelDirectory;

    final vocabFutureResult = _loadVocab(modelPath: modelDirectory);

    final controlTokenFutureResult = _loadControlTokens(
      modelPath: modelDirectory,
      languageCode: languageCode,
    );

    final onnxConfig = TranscriberOnnxConfig();

    final superEncoderFutureResult = _loadSuperEncoder(
      modelPath: modelDirectory,
      onnxConfig: onnxConfig,
    );
    final decoderFutureResult = _loadDecoder(
      modelPath: modelDirectory,
      onnxConfig: onnxConfig,
    );
    final decoderWithPastFutureResult = _loadDecoderWithPast(
      modelPath: modelDirectory,
      onnxConfig: onnxConfig,
    );

    final results = await Future.wait([
      vocabFutureResult,
      controlTokenFutureResult,
      superEncoderFutureResult,
      decoderFutureResult,
      decoderWithPastFutureResult,
    ]);
    String errorMessage = '';
    for (final result in results) {
      if (result is Error) {
        errorMessage = '$errorMessage\n${result.error.toString()}';
      }
    }
    loadModelTimelineTask.finish();
    if (errorMessage.isNotEmpty) {
      _logger.warning(errorMessage);
      return Result.error(Exception(errorMessage));
    }
    return Result.ok(null);
  }

  @override
  Future<Result<TranscriptionResult>> transcribe(
    Float32List audio, {
    bool segmentEnd = true,
    bool getWordDetails = false,
    bool getSegmentDetails = false,
    int? maxOutputTokens,
  }) async {
    dev.Timeline.startSync('run_super_encoder');
    final encoded = await _runSuperEncoder(audio);
    dev.Timeline.finishSync();
    final audioFeaturesTensor = encoded?[1] as OrtValueTensor;

    final audioDuration = audio.length / kSampleRate;
    final effectiveMaxTokens =
        maxOutputTokens ??
        (audioDuration * _tokensPerSecond).ceil().clamp(10, 224);

    dev.Timeline.startSync('run_decoder');
    final decoderResultResult = await _runDecoder(
      audioFeaturesTensor,
      effectiveMaxTokens,
      computeTokenConfidence: getWordDetails,
      computeSegmentConfidence: getSegmentDetails,
    );
    dev.Timeline.finishSync();
    audioFeaturesTensor.release();

    Map<String, dynamic> decoderResult = {};
    if (decoderResultResult is Error<Map<String, dynamic>>) {
      return Result.error(decoderResultResult.error);
    } else if (decoderResultResult is Ok<Map<String, dynamic>>) {
      decoderResult = decoderResultResult.value;
    }

    String transcript = decoderResult['transcript'] as String;
    List<Word>? words;

    if (transcript == '[BLANK_AUDIO]') {
      transcript = '';
    }

    if (getWordDetails && transcript.isNotEmpty) {
      final confidences = decoderResult['confidences'] as List<double>?;
      final tokens = decoderResult['tokens'] as List<int>?;
      if (confidences != null && tokens != null) {
        words = _mapTokensToWordsWithConfidences(tokens, confidences);
      }
    }

    // Extract segment-level confidence if requested
    List<double>? segmentConfidences;
    if (getSegmentDetails && transcript.isNotEmpty) {
      final segmentConfidence = decoderResult['segmentConfidence'] as double?;
      if (segmentConfidence != null) {
        segmentConfidences = [segmentConfidence];
      }
    }

    final duration = audio.length / kSampleRate;

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
    superEncoderSession?.release();
    decoderSession?.release();
    decoderWithPastSession?.release();
  }

  /// Map tokens to words with proper confidence aggregation.
  /// Uses minimum confidence across all tokens that form a word.
  List<Word> _mapTokensToWordsWithConfidences(
    List<int> tokens,
    List<double> confidences,
  ) {
    final words = <Word>[];
    String currentWordText = '';
    final currentWordConfidences = <double>[];

    for (int i = 0; i < tokens.length; i++) {
      final tokenText = _tokenizer.decodeSingleToken(tokens[i]);

      // Skip empty tokens (special tokens that were filtered)
      if (tokenText.isEmpty) continue;

      // Check if this token starts a new word (has leading space)
      final startsNewWord = _tokenizer.tokenStartsNewWord(tokens[i]);

      if (startsNewWord && currentWordText.trim().isNotEmpty) {
        // Finish previous word - use minimum confidence
        words.add(
          Word(
            word: currentWordText.trim(),
            confidence: currentWordConfidences.isNotEmpty
                ? currentWordConfidences.reduce(min)
                : -1.0,
            start: -1.0,
            end: -1.0,
          ),
        );
        currentWordText = '';
        currentWordConfidences.clear();
      }

      currentWordText += tokenText;
      if (i < confidences.length) {
        currentWordConfidences.add(confidences[i]);
      }
    }

    // Don't forget the last word
    if (currentWordText.trim().isNotEmpty) {
      words.add(
        Word(
          word: currentWordText.trim(),
          confidence: currentWordConfidences.isNotEmpty
              ? currentWordConfidences.reduce(min)
              : -1.0,
          start: -1.0,
          end: -1.0,
        ),
      );
    }

    return words;
  }

  /// Transcribe methods.
  Future<List<OrtValue?>?> _runSuperEncoder(Float32List audio) async {
    // Run super encoder (preprocessor + encoder combined)
    // Takes raw audio, outputs hidden states
    final runOptions = OrtRunOptions();
    final audioTensor = OrtValueTensor.createTensorWithDataList(audio, [
      1,
      audio.length,
    ]);
    final lengthTensor = OrtValueTensor.createTensorWithDataList(
      Int64List.fromList([audio.length]),
      [1],
    );

    final outputs = await superEncoderSession!.runAsync(runOptions, {
      'waveforms': audioTensor,
      'waveforms_lens': lengthTensor,
    });

    audioTensor.release();
    lengthTensor.release();
    runOptions.release();

    return outputs;
  }

  Future<Result<Map<String, dynamic>>> _runDecoder(
    OrtValueTensor audioFeaturesTensor,
    int maxTokens, {
    bool computeTokenConfidence = false,
    bool computeSegmentConfidence = false,
  }) async {
    final List<int> tokens = [
      sotToken,
      languageToken!,
      transcribeToken!,
      noTimestampsToken,
    ];

    _logger.finer(
      'Initial token IDs: sot=$sotToken, lang=$languageToken, transcribe=$transcribeToken, notimestamps=$noTimestampsToken',
    );

    Map<String, OrtValue>? pastKeyValues;
    Map<String, OrtValue>?
    encoderPastKeyValues; // Stays constant after first iteration
    final decoderOutputNames = decoderSession!.outputNames;
    final decoderWithPastOutputNames = decoderWithPastSession!.outputNames;
    final List<double> confidences = [];
    double sumLogProbs = 0.0;
    int generatedTokenCount = 0;

    final runOptions = OrtRunOptions();
    final singleTokenBuffer = Int64List(1);
    final decoderWithPastInputs = <String, OrtValue>{};

    // Account for initial tokens (4 tokens) in max_length calculation
    // max_new_tokens = max_length - decoder_input_ids.shape[1]
    final maxNewTokens = maxTokens - tokens.length;

    for (int i = 0; i < maxNewTokens; ++i) {
      dev.Timeline.startSync('decoder_run_$i');
      List<OrtValue?>? decoderOutputs;
      List<double> lastLogits;

      if (pastKeyValues == null) {
        // First iteration - use full decoder with all tokens
        final tokensTensor = OrtValueTensor.createTensorWithDataList(
          Int64List.fromList(tokens),
          [1, tokens.length],
        );

        final decoderInputs = {
          'input_ids': tokensTensor,
          'encoder_hidden_states': audioFeaturesTensor,
        };

        decoderOutputs = await decoderSession!.runAsync(
          runOptions,
          decoderInputs,
        );

        tokensTensor.release();

        // Get logits - we compute for all 4 initial tokens but only use the
        // last one. This is necessary to establish the KV cache for subsequent
        // iterations.
        final logits = decoderOutputs?[0]?.value as List;
        lastLogits = (logits[0][tokens.length - 1] as List).cast<double>();

        // Release logits tensor after extracting values
        final logitsTensor = decoderOutputs?[0];
        if (logitsTensor is OrtValueTensor) {
          logitsTensor.release();
        }

        pastKeyValues = {};
        encoderPastKeyValues = {};
        for (int j = 1; j < (decoderOutputs?.length ?? 0); j++) {
          final outputName = decoderOutputNames[j];
          if (outputName.startsWith('present.')) {
            final pastName = outputName.replaceFirst(
              'present.',
              'past_key_values.',
            );
            final ortValue = decoderOutputs![j]!;
            pastKeyValues[pastName] = ortValue;

            if (outputName.contains('.encoder.')) {
              encoderPastKeyValues[pastName] = ortValue;
            }
          }
        }
      } else {
        singleTokenBuffer[0] = tokens.last;
        final currentTokenTensor = OrtValueTensor.createTensorWithDataList(
          singleTokenBuffer,
          [1, 1],
        );

        decoderWithPastInputs.clear();
        decoderWithPastInputs['input_ids'] = currentTokenTensor;
        decoderWithPastInputs.addAll(pastKeyValues);

        decoderOutputs = decoderWithPastSession!.run(
          runOptions,
          decoderWithPastInputs,
        );
        currentTokenTensor.release();

        final logits = decoderOutputs[0]?.value as List;
        lastLogits = (logits[0][0] as List).cast<double>();

        final logitsTensor = decoderOutputs[0];
        if (logitsTensor is OrtValueTensor) {
          logitsTensor.release();
        }

        final decoderKeys = <String>[];
        for (final key in pastKeyValues.keys) {
          if (key.contains('.decoder.')) {
            decoderKeys.add(key);
          }
        }

        for (final key in decoderKeys) {
          final value = pastKeyValues[key];
          if (value is OrtValueTensor) {
            value.release();
          }
          pastKeyValues.remove(key);
        }

        for (int j = 1; j < decoderOutputs.length; j++) {
          final outputName = decoderWithPastOutputNames[j];
          if (outputName.startsWith('present.') &&
              outputName.contains('.decoder.')) {
            final pastName = outputName.replaceFirst(
              'present.',
              'past_key_values.',
            );
            pastKeyValues[pastName] = decoderOutputs[j]!;
          }
        }
      }

      int nextToken = 0;
      double maxLogit = lastLogits[0];
      for (int j = 1; j < lastLogits.length; j++) {
        if (lastLogits[j] > maxLogit) {
          maxLogit = lastLogits[j];
          nextToken = j;
        }
      }

      _logger.finer(
        'Step $i | ID: $nextToken | Logit: ${maxLogit.toStringAsFixed(2)} | Text: ${_tokenizer.decode(tokens, skipSpecialTokens: false)}',
      );

      if (nextToken == eotToken) {
        dev.Timeline.finishSync();
        break;
      }
      tokens.add(nextToken);

      // Compute token confidence if needed (expensive since we iterate over all ~51k logits per token)
      // confidence(token) = exp(logit[token] - maxLogit) / expSum
      // since we do greedy decoding chosing token with max logit, this simplifies to:
      // confidence(nextToken) = exp(maxLogit - maxLogit) / expSum
      //                       = exp(0) / expSum
      //                       = 1 / expSum
      if (computeSegmentConfidence || computeTokenConfidence) {
        double expSum = 0.0;
        for (int j = 0; j < lastLogits.length; j++) {
          expSum += exp(lastLogits[j] - maxLogit);
        }
        final confidence = 1.0 / expSum;

        if (computeTokenConfidence) {
          confidences.add(confidence);
        }
        if (computeSegmentConfidence) {
          sumLogProbs += log(confidence);
          generatedTokenCount++;
        }
      }
      dev.Timeline.finishSync();
    }
    runOptions.release();
    if (pastKeyValues != null) {
      for (final value in pastKeyValues.values) {
        if (value is OrtValueTensor) {
          value.release();
        }
      }
    }

    final transcriptResult = _tokenizer.decode(tokens, skipSpecialTokens: true);
    if (transcriptResult is Error<String>) {
      return Result.error(transcriptResult.error);
    }
    String transcript = '';
    if (transcriptResult is Ok<String>) {
      _logger.finest(
        'Average confidence: ${confidences.isEmpty ? 0 : confidences.reduce((a, b) => a + b) / confidences.length}',
      );
      transcript = transcriptResult.value.trim();
    }
    // Skip initial prompt tokens (SOT, language, transcribe, notimestamps)
    // to align tokens with confidences (which are only for generated tokens)
    final generatedTokens = tokens.length > 4 ? tokens.sublist(4) : <int>[];

    // Compute segment-level confidence from average log probability
    double? segmentConfidence;
    if (computeSegmentConfidence && generatedTokenCount > 0) {
      final avgLogProb = sumLogProbs / generatedTokenCount;
      segmentConfidence = exp(avgLogProb);
    }
    return Result.ok({
      'transcript': transcript,
      'confidences': confidences,
      'tokens': generatedTokens,
      'segmentConfidence': segmentConfidence,
    });
  }

  /// Load model methods.
  Future<Result<void>> _loadVocab({required String modelPath}) async {
    dev.Timeline.startSync('load_vocab');
    final vocabPath = '$modelPath/vocab.json';
    final result = await _tokenizer.loadVocab(path: vocabPath);
    dev.Timeline.finishSync();
    return result;
  }

  Future<Result<void>> _loadSuperEncoder({
    required String modelPath,
    required OnnxConfig onnxConfig,
  }) async {
    dev.Timeline.startSync('load_super_encoder');
    _logger.finer('Load super encoder');
    final encoderPath = '$modelPath/super_encoder.onnx';
    try {
      final encoderBytes = await Utils.loadBytes(encoderPath);
      superEncoderSession = onnxConfig.createSession(encoderBytes);
      dev.Timeline.finishSync();
      return Result.ok(null);
    } catch (e) {
      return Result.error(
        Exception('Failed to load super_encoder.onnx from <$modelPath>: $e'),
      );
    }
  }

  Future<Result<void>> _loadDecoder({
    required String modelPath,
    required OnnxConfig onnxConfig,
  }) async {
    dev.Timeline.startSync('load_decoder');
    _logger.finer('Load decoder');
    final decoderPath = '$modelPath/decoder_model.onnx';
    try {
      final decoderBytes = await Utils.loadBytes(decoderPath);
      decoderSession = onnxConfig.createSession(decoderBytes);
      dev.Timeline.finishSync();
      return Result.ok(null);
    } catch (e) {
      return Result.error(
        Exception('Failed to load decoder_model.onnx from <$modelPath>: $e'),
      );
    }
  }

  Future<Result<void>> _loadDecoderWithPast({
    required String modelPath,
    required OnnxConfig onnxConfig,
  }) async {
    dev.Timeline.startSync('load_decoder_with_past');
    _logger.finer('Load decoder with past');
    final decoderWithPastPath = '$modelPath/decoder_with_past_model.onnx';
    try {
      final decoderWithPastBytes = await Utils.loadBytes(decoderWithPastPath);
      decoderWithPastSession = onnxConfig.createSession(decoderWithPastBytes);
      dev.Timeline.finishSync();
      return Result.ok(null);
    } catch (e) {
      return Result.error(
        Exception(
          'Failed to load decoder_with_past_model.onnx from <$modelPath>: $e',
        ),
      );
    }
  }

  Future<Result<void>> _loadControlTokens({
    required String modelPath,
    required String languageCode,
  }) async {
    final loadControlTokenTask = dev.TimelineTask()..start('load_control_token');
    final configPath = '$modelPath/generation_config.json';
    String configContent;
    try {
      configContent = await Utils.loadString(configPath);
    } catch (e) {
      return Result.error(
        Exception(
          'Failed to load generation_config.json from <$modelPath>: $e',
        ),
      );
    }
    final config = jsonDecode(configContent) as Map<String, dynamic>;

    int getTokenId(dynamic value) {
      if (value is int) {
        return value;
      } else if (value is List && value.isNotEmpty) {
        return value[0] as int;
      }
      throw FormatException('Invalid token ID format: $value');
    }

    sotToken = getTokenId(config['decoder_start_token_id']);
    eotToken = getTokenId(config['eos_token_id']);
    noTimestampsToken = getTokenId(config['no_timestamps_token_id']);

    final forcedDecoderIds = config['forced_decoder_ids'] as List<dynamic>?;
    if (forcedDecoderIds != null && forcedDecoderIds.isNotEmpty) {
      for (final pair in forcedDecoderIds) {
        final position = pair[0] as int;
        final tokenId = pair[1] as int?;

        if (tokenId != null) {
          if (position == 1 && tokenId != noTimestampsToken) {
            languageToken = tokenId;
          } else if (position == 2) {
            transcribeToken = tokenId;
          }
        }
      }
    }

    final taskToId = config['task_to_id'] as Map<String, dynamic>?;
    if (taskToId != null && taskToId.containsKey('transcribe')) {
      transcribeToken = taskToId['transcribe'] as int;
    }

    if (languageToken == null) {
      final langToId = config['lang_to_id'] as Map<String, dynamic>?;
      if (langToId != null) {
        final langKey = '<|$languageCode|>';
        languageToken = langToId[langKey] as int?;

        if (languageToken == null) {
          return Result.error(
            Exception(
              'Language "$languageCode" not found in model configuration. '
              'This model may not support this language. '
              'Available languages: ${langToId.keys.map((k) => k.replaceAll(RegExp(r'[<|>]'), '')).join(", ")}',
            ),
          );
        }
      }
    }

    _logger.finer(
      'Loaded token config: sot=$sotToken, eot=$eotToken, lang=$languageToken,'
      ' transcribe=$transcribeToken, notimestamps=$noTimestampsToken',
    );
    loadControlTokenTask.finish();
    return Result.ok(null);
  }
}
