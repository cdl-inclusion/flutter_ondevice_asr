import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:logging/logging.dart';

import '../common/audio_constants.dart';
import '../common/result.dart';
import '../model/onnx_config.dart';
import '../model/transcription_result.dart';
import '../transcriber.dart';
import '../vad/silero_vad.dart';
import 'chunked_speech_buffer.dart';
import 'float32_ring_buffer.dart';

class StreamingTranscriber {
  final _logger = Logger('StreamingTranscriber');
  final Transcriber _transcriber;

  final int _sampleRate;
  final double _vadThreshold;
  final int _eosMinSilence;
  final int _vadMinSpeech; // ms, start gate; 0 = off

  bool _enablePartials = true;
  int _minPartialDuration = 500;
  int _maxSegmentDuration = 20000;

  // Incremental streaming (on by default, togglable): non-null iff the wrapped transcriber advertises
  // [IncrementalStreaming] AND it's enabled. When null, the naive whole-buffer path runs
  // (unchanged — this is what Whisper always uses). See streaming_rnnt_plan.md.
  IncrementalStreaming? _streamer;
  // At VAD segment end, re-decode the whole segment once (best quality, restores punctuation)
  // vs. commit the streamed result as-is. Default true.
  bool _fullRedecodeOnFinal = true;

  late final SileroVAD _sileroModel;
  late final SileroVADIterator _sileroVad;
  late final StreamController<TranscriptionResult> _transcriptionController;

  final ChunkedSpeechBuffer _speechBuffer = ChunkedSpeechBuffer();
  final Float32RingBuffer _vadChunkBuffer = Float32RingBuffer();
  bool _isRecordingSpeech = false;
  int _chunkCounter = 0;
  double _bufferDurationSeconds =
      0.0; // Total duration in buffer since VAD start
  double _newAudioDurationSeconds =
      0.0; // Duration of new audio since last partial transcription
  late final double _vadChunkDurationSeconds;
  bool _transcriptionInProgress = false;
  // A VAD 'start' arrived while a decode was running; reset the incremental streamer once
  // that decode (and any queued final) has finished instead of under it.
  bool _streamResetDeferred = false;
  bool _bufferContainsSpeech = false;

  // Segments whose VAD 'end' (or forced end) arrived while a decode was still running.
  // Each is snapshotted at its boundary and its final runs, in order, as soon as the in-flight
  // decode completes, so no segment is dropped or glued onto the next utterance.
  final List<_PendingFinal> _pendingFinals = [];

  // Audio retained across a forced (max-duration) split so the next segment does not start
  // mid-phoneme. Same value as the cloud streaming loop.
  static const double _forcedSplitOverlapSeconds = 0.1;

  // Reset VAD state after every VAD 'end' and, while no speech is in
  // progress, after this much audio without any VAD event. 
  static const double vadStateResetSilenceSeconds = 10.0;
  // Keep track of audio since the last VAD event while not recording (actual audio time, not wall clock).
  double _silenceSinceVadEventSeconds = 0.0;

  /// Number of Silero VAD state resets performed since construction / [reset]. For tests.
  @visibleForTesting
  int vadStateResets = 0;

  bool _isDisposed = false;

  /// Test seam: invoked after every partial/final decode with (isFinal, audioSamples decoded,
  /// wall-clock decode ms). Null in production (zero overhead). Used by the streaming perf harness
  /// to measure per-transcription latency without re-implementing the VAD/segmentation loop.
  @visibleForTesting
  void Function(bool isFinal, int audioSamples, double decodeMs)? onDecodeTiming;

  StreamingTranscriber._({
    required Transcriber transcriber,
    required int sampleRate,
    required double vadThreshold,
    required int eosMinSilence,
    required int vadMinSpeech,
  }) : _transcriber = transcriber,
       _sampleRate = sampleRate,
       _vadThreshold = vadThreshold,
       _eosMinSilence = eosMinSilence,
       _vadMinSpeech = vadMinSpeech {
    _transcriptionController =
        StreamController<TranscriptionResult>.broadcast();
  }

  /// Create and initialize a new StreamingTranscriber instance
  ///
  /// Required:
  /// - [transcriber]: Any initialized Transcriber implementation (eg Whisper)
  ///
  /// Optional VAD parameters:
  /// - [vadThreshold]: VAD sensitivity, 0.0-1.0 (default: **0.5**)
  ///   - Higher = less sensitive (fewer false positives, may miss quiet speech)
  ///   - Lower = more sensitive (catches quiet speech, more false positives)
  /// - [eosMinSilence]: Silence duration in ms to end a segment (default: **1000**)
  /// - [vadMinSpeech]: start gate in ms (default: **200**). A speech start only counts once the
  ///   VAD probability has stayed above [vadThreshold] for this long; shorter bursts (eg breaths,
  ///   faint sounds) would otherwise easily classified as speech by a freshly reset VAD, with this start
  ///   gate they are effectively ignored without hurting actual speech typically.
  /// - [sampleRate]: Audio sample rate in Hz (default: **16000**)
  ///   - Must match your audio input
  ///
  /// Optional session parameters (can be changed later via `configure()`):
  /// - [enablePartials]: Emit partial transcriptions during speech (default: **true**)
  ///   This will trigger a transcriber call whenever enough data for a partial is collected
  ///   (len >= minPartialDuration) and especially for short minPartialDuration this will lead
  ///   to significant system use. For weaker devices, it will be important to set minPartialDuration
  ///   conservatively (ie, high). However, in order for transcriptions to feel real-time we would
  ///   ideally set minPartialDuration to 300ms.
  /// - [minPartialDuration]: Minimum ms between partial updates (default: **500**)
  /// - [maxSegmentDuration]: Maximum segment length in ms before forcing end (default: **20000**)
  ///   We limit this to the maximum segment length, Whisper can natively handle. We intentionally
  ///   skip any sort of sliding window approaches in the streaming-based transcription for efficiency.
  ///
  /// Incremental streaming (only if [transcriber] implements [IncrementalStreaming]:
  /// - [enableIncrementalStreaming]: use the stateful chunked path for partials (default: **true**).
  ///   When true (default, and supported), each partial decodes only the newly-arrived audio (persisting
  ///   decoder state) instead of re-transcribing the whole growing buffer. Set to **false** to force ALL
  ///   transcribers onto the naive whole-buffer path — byte-for-byte the pre-streaming behavior.
  ///   Caveat: route (a) is validated on one clean EN clip but its on-device latency/RTF win and
  ///   atypical-speech parity are still unverified (Phase 3); flip back to false if the incremental
  ///   path regresses on atypical speech. See streaming_rnnt_status.md / streaming_efficiency_model.md.
  /// - [fullRedecodeOnStreamingFinal]: at VAD segment end, re-decode the whole segment once for
  ///   the final (default: **true**). Set false to commit
  ///   the streamed result as-is (cheaper, but the final inherits streaming's chunk-edge/punct
  ///   differences).
  static Future<StreamingTranscriber> create({
    required Transcriber transcriber,
    double vadThreshold = 0.5,
    int eosMinSilence = 1000,
    int vadMinSpeech = 200,
    int sampleRate = kSampleRate,
    bool enablePartials = true,
    int minPartialDuration = 500,
    int maxSegmentDuration = 20000,
    bool enableIncrementalStreaming = true,
    bool fullRedecodeOnStreamingFinal = true,
  }) async {
    final instance = StreamingTranscriber._(
      transcriber: transcriber,
      sampleRate: sampleRate,
      vadThreshold: vadThreshold,
      eosMinSilence: eosMinSilence,
      vadMinSpeech: vadMinSpeech,
    );

    // Set initial mutable parameters
    instance._enablePartials = enablePartials;
    instance._minPartialDuration = minPartialDuration;
    instance._maxSegmentDuration = maxSegmentDuration;

    // Opt in to incremental streaming only when the transcriber advertises the capability.
    instance._streamer =
        (enableIncrementalStreaming && transcriber is IncrementalStreaming)
            ? transcriber as IncrementalStreaming
            : null;
    instance._fullRedecodeOnFinal = fullRedecodeOnStreamingFinal;

    await instance._initializeVAD();
    return instance;
  }

  Future<void> _initializeVAD() async {
    _sileroModel = SileroVAD();

    final config = VadOnnxConfig();
    await _sileroModel.loadModel(config);

    _sileroVad = SileroVADIterator(
      model: _sileroModel,
      threshold: _vadThreshold,
      samplingRate: _sampleRate,
      minSilenceDurationMs: _eosMinSilence,
      minSpeechChunks: _vadMinSpeechChunks,
    );

    _vadChunkDurationSeconds = _sileroModel.requiredChunkSize / _sampleRate;

    _logger.fine('[Streaming] Using Silero VAD (threshold: $_vadThreshold)');
  }

  void configure({
    bool? enablePartials,
    int? minPartialDuration,
    int? maxSegmentDuration,
  }) {
    if (enablePartials != null) _enablePartials = enablePartials;
    if (minPartialDuration != null) _minPartialDuration = minPartialDuration;
    if (maxSegmentDuration != null) _maxSegmentDuration = maxSegmentDuration;
  }

  Stream<TranscriptionResult> get transcriptionStream =>
      _transcriptionController.stream;

  /// Process an audio chunk and emit transcription results
  ///
  /// Audio must be Float32List in range [-1.0, 1.0] at the configured sample
  /// rate (default 16kHz).
  /// Note: For Silero VAD with 16kHz, audio will be buffered into 512-sample
  /// chunks automatically.
  Future<Result<void>> processAudioChunk(Float32List audioChunk) async {
    if (_isDisposed) {
      return Result.error(Exception('StreamingTranscriber has been disposed'));
    }

    _vadChunkBuffer.addAll(audioChunk);

    // Process VAD in 512-sample (sileroModel.requiredChunkSize) chunks
    while (_vadChunkBuffer.length >= _sileroModel.requiredChunkSize) {
      final vadChunk = _vadChunkBuffer.consume(_sileroModel.requiredChunkSize);
      _chunkCounter++;

      _speechBuffer.addChunk(vadChunk);
      _bufferDurationSeconds += _vadChunkDurationSeconds;
      _newAudioDurationSeconds += _vadChunkDurationSeconds;
      final bufferDuration = _bufferDurationSeconds;

      if (_chunkCounter % 50 == 0) {
        _logger.finest(
          '[Streaming] Chunk #$_chunkCounter: Buffer=${bufferDuration.toStringAsFixed(2)}s, RecordingSpeech=$_isRecordingSpeech',
        );
      }

      final vadEvent = _sileroVad.call(vadChunk);

      if (vadEvent != null) {
        if (vadEvent == 'start') {
          _isRecordingSpeech = true;
          _bufferContainsSpeech = true;

          // New segment: reset the incremental decoder state so its commit cursor aligns with
          // the current buffer origin. This line always runs, but only has an effect in
          // incremental mode: `_streamer` is non-null only when the wrapped transcriber
          // implements IncrementalStreaming AND it's enabled. On the naive path `_streamer`
          // is null, so `?.` short-circuits and this is a genuine no-op (nothing streaming
          // executes) — which is what keeps `enableIncrementalStreaming: false` byte-for-byte
          // identical to the pre-streaming behavior.
          //
          // If a decode is still running (a partial or final of the previous segment), do not
          // reset under it: the window loop reads the commit cursor between awaits and would
          // decode garbage. Defer; the reset runs when the in-flight chain completes (a
          // final resets the streamer itself, otherwise `_startDecode` does it).
          if (_transcriptionInProgress) {
            _streamResetDeferred = true;
          } else {
            _streamer?.streamReset();
          }

          if (!_transcriptionInProgress) {
            _newAudioDurationSeconds = 0.0;
          }
          _logger.finest(
            '[Streaming] Speech started (buffer: ${bufferDuration.toStringAsFixed(2)}s)',
          );
        } else if (vadEvent == 'end') {
          _isRecordingSpeech = false;
          _logger.finest(
            '[Streaming] Speech ended (buffer: ${bufferDuration.toStringAsFixed(2)}s',
          );
          _transcribeCurrentSpeechBuffer(bufferDuration, false);
          // reset vad internal state after segment end
          _resetVadState('segment end');
        }
        _silenceSinceVadEventSeconds = 0.0;
      } else {
        if (_isRecordingSpeech) {
          if (_isSegmentTooLong(bufferDuration)) {
            _logger.finest(
              '[Streaming] Max segment duration reached, forcing end',
            );
            _transcribeCurrentSpeechBuffer(bufferDuration, false,
                keepOverlap: true);
          } else if (_isReadyToTranscribePartials()) {
            _transcribeCurrentSpeechBuffer(bufferDuration, true);
          } else {
            _logger.finest(
              '[Streaming] Collection speech (buffer: ${bufferDuration.toStringAsFixed(2)})',
            );
          }
        } else {
          if (_isSegmentTooLong(bufferDuration)) {
            _logger.finest(
              '[Streaming] Stale buffered speech exceeded max segment duration, forcing transcription',
            );
            _transcribeCurrentSpeechBuffer(bufferDuration, false);
          }
          if (!_bufferContainsSpeech) {
            // 100 ms of pre-roll plus the chunks the start gate is still evaluating, so a
            // gated 'start' yields the same segment audio as an ungated one. The gate fires on
            // its last chunk, which is appended after this trim, hence gate - 1. Same as cloud.
            final emptyFramesToKeep = (0.1 * _sampleRate).toInt() +
                (_vadMinSpeechChunks > 0 ? _vadMinSpeechChunks - 1 : 0) *
                    _sileroModel.requiredChunkSize;
            if (_speechBuffer.length > emptyFramesToKeep) {
              _speechBuffer.removeFromFront(
                _speechBuffer.length - emptyFramesToKeep,
              );
              // Keep the duration counter equal to what the buffer actually holds. Otherwise
              // leading silence counts towards the max-segment cutoff of the next segment,
              // and long silence triggers pointless "stale buffer" decodes of pure pre-roll.
              _bufferDurationSeconds = _speechBuffer.length / _sampleRate;
            }
          }
          _silenceSinceVadEventSeconds += _vadChunkDurationSeconds;
          if (_silenceSinceVadEventSeconds >= vadStateResetSilenceSeconds) {
            _resetVadState(
                '${vadStateResetSilenceSeconds.toStringAsFixed(0)}s without speech');
            _silenceSinceVadEventSeconds = 0.0;
          }
        }
      }
    }
    return Result.ok(null);
  }

  void _resetVadState(String reason) {
    _sileroVad.resetStates();
    vadStateResets++;
    _logger.fine('[Streaming] VAD state reset after $reason');
  }

  /// Flush any remaining audio in the buffer (call this when stopping the stream)
  Future<void> flush() async {
    // Wait for any in-progress transcription (and any final queued behind it) to complete.
    while (_transcriptionInProgress) {
      await Future.delayed(const Duration(milliseconds: 10));
    }

    // Only decode if the VAD has started a segment that has not ended yet. While not
    // recording, the buffer holds just the pre-roll (100 ms plus the start-gate lookback of
    // room noise); decoding that bypasses the VAD and the start gate and can produce a
    // hallucinated fragment on every stop.
    if (_speechBuffer.length > 0 && _bufferContainsSpeech) {
      final audioData = _speechBuffer.toFloat32List();

      final streamer = _streamer;
      final timingCb = onDecodeTiming;
      final timingSw = timingCb != null ? (Stopwatch()..start()) : null;
      final result = (streamer != null && !_fullRedecodeOnFinal)
          ? await streamer.streamTranscribe(audioData, flush: true)
          : await _transcriber.transcribe(
              audioData,
              segmentEnd: true,
              getWordDetails: false,
            );
      streamer?.streamReset();
      if (timingCb != null) {
        timingCb(true, audioData.length, timingSw!.elapsedMicroseconds / 1000.0);
      }

      // Only emit if we got actual text
      if (result is Ok<TranscriptionResult> && result.value.text.isNotEmpty) {
        if (!_transcriptionController.isClosed) {
          _transcriptionController.add(result.value);
        }

        _logger.finest(
          '[Streaming] Flush final (${result.value.durationInSeconds.toStringAsFixed(2)}s): ${result.value.text}',
        );
      }
      _speechBuffer.clear();
      _bufferDurationSeconds = 0.0;
      _bufferContainsSpeech = false;
    }
  }

  /// Helper method to run transcription asynchronously without blocking audio processing
  Future<void> _transcribeAsync(
    Float32List audio,
    double duration, {
    required bool isFinal,
    bool forceFullDecode = false,
  }) async {
    try {
      final Result<TranscriptionResult> result;
      final streamer = _streamer;
      final timingCb = onDecodeTiming;
      final timingSw = timingCb != null ? (Stopwatch()..start()) : null;
      if (streamer != null && forceFullDecode) {
        // Queued final: decode offline and reset the streamer for the next segment.
        result = await _transcriber.transcribe(audio,
            segmentEnd: true, getWordDetails: false);
        streamer.streamReset();
      } else if (streamer != null) {
        // Incremental path. Partials (and the final when re-decode is off) decode only the
        // new audio via the persisted decoder state; the final uses a full re-decode by
        // default for best quality. Reset after any final so the next segment (incl. a
        // max-duration forced split mid-speech) starts clean.
        result = (isFinal && _fullRedecodeOnFinal)
            ? await _transcriber.transcribe(audio,
                segmentEnd: true, getWordDetails: false)
            : await streamer.streamTranscribe(audio, flush: isFinal);
        if (isFinal) streamer.streamReset();
      } else {
        result = await _transcriber.transcribe(
          audio,
          segmentEnd: isFinal,
          getWordDetails: false,
        );
      }
      if (timingCb != null) {
        timingCb(isFinal, audio.length, timingSw!.elapsedMicroseconds / 1000.0);
      }

      // Only emit if we got actual text
      if (result is Ok<TranscriptionResult> && result.value.text.isNotEmpty) {
        if (!_transcriptionController.isClosed) {
          _transcriptionController.add(result.value);
        }

        final label = result.value.isFinal ? 'Final' : 'Partial';
        _logger.finest(
          '[Streaming] $label (${result.value.durationInSeconds.toStringAsFixed(2)}s): ${result.value.text}',
        );
      }
    } catch (e) {
      _logger.warning('[Streaming] Transcription error: $e');
    }
  }

  /// Reset the streaming state (clears buffers and VAD state)
  void reset() {
    _speechBuffer.clear();
    _vadChunkBuffer.clear();
    _bufferDurationSeconds = 0.0;
    _newAudioDurationSeconds = 0.0;
    _isRecordingSpeech = false;
    _bufferContainsSpeech = false;
    _chunkCounter = 0;
    _pendingFinals.clear();
    _streamResetDeferred = false;
    _silenceSinceVadEventSeconds = 0.0;
    vadStateResets = 0;

    _sileroVad.resetStates();

    _logger.finest('[Streaming] State reset');
  }

  /// Dispose of resources
  Future<void> dispose() async {
    if (_isDisposed) return;

    _isDisposed = true;

    // Flush any remaining audio
    await flush();

    await _transcriptionController.close();

    // Dispose VAD resources
    _sileroVad.model.dispose();

    _logger.finest('[Streaming] Disposed');
  }

  void _transcribeCurrentSpeechBuffer(
    double bufferDuration,
    bool processingPartials, {
    bool keepOverlap = false,
  }) {
    if (_speechBuffer.length == 0) return;

    if (processingPartials) {
      // Partials are best-effort: skip if a decode is already running.
      if (_transcriptionInProgress) return;
      final audioToTranscribe = _speechBuffer.toFloat32List();
      _newAudioDurationSeconds = 0.0;
      _startDecode(audioToTranscribe, bufferDuration, isFinal: false);
      return;
    }

    // Final: the segment boundary is now, regardless of whether a decode is running.
    // Snapshot the segment and clear the buffer so the next segment starts clean.
    final audioToTranscribe = _speechBuffer.toFloat32List();
    _speechBuffer.clear();
    _bufferDurationSeconds = 0.0;
    _bufferContainsSpeech = false; // Buffer cleared, no more untranscribed speech
    _newAudioDurationSeconds = 0.0;

    if (keepOverlap) {
      // Forced split mid-speech: carry the tail of the old segment into the new one.
      final overlapSamples = (_forcedSplitOverlapSeconds * _sampleRate).toInt();
      final start = audioToTranscribe.length - overlapSamples;
      if (start > 0) {
        _speechBuffer.addChunk(audioToTranscribe.sublist(start));
        _bufferDurationSeconds = overlapSamples / _sampleRate;
        _bufferContainsSpeech = true;
      }
    }

    if (_transcriptionInProgress) {
      // Queue the final; it runs when the in-flight decode completes.
      _pendingFinals.add(_PendingFinal(audioToTranscribe, bufferDuration));
      _logger.finest(
        '[Streaming] Decode in progress, queued final (${bufferDuration.toStringAsFixed(2)}s)',
      );
      return;
    }

    _startDecode(audioToTranscribe, bufferDuration, isFinal: true);
  }

  /// Run one decode asynchronously without blocking audio processing. On completion, run a
  /// queued final if one accumulated meanwhile.
  void _startDecode(
    Float32List audio,
    double duration, {
    required bool isFinal,
    bool forceFullDecode = false,
  }) {
    _transcriptionInProgress = true;
    _transcribeAsync(audio, duration,
            isFinal: isFinal, forceFullDecode: forceFullDecode)
        .then((_) {
      _transcriptionInProgress = false;
      if (_pendingFinals.isNotEmpty) {
        final pending = _pendingFinals.removeAt(0);
        // Full re-decode: the incremental streamer state belongs to the segment that has
        // since started, so it cannot be trusted for this audio. The final resets the
        // streamer when it completes, which also covers any deferred 'start' reset.
        _streamResetDeferred = false;
        _startDecode(pending.audio, pending.duration,
            isFinal: true, forceFullDecode: true);
      } else if (_streamResetDeferred) {
        _streamResetDeferred = false;
        // Nothing queued: this decode did not end with a reset unless it was a final,
        // so apply the 'start' reset that was deferred while it ran.
        if (!isFinal) _streamer?.streamReset();
      }
    });
  }

  /// Start-gate length in VAD chunks (32 ms each at 16 kHz); 0 = gate off.
  int get _vadMinSpeechChunks => _vadMinSpeech <= 0
      ? 0
      : (_vadMinSpeech * _sampleRate / 1000 / _sileroModel.requiredChunkSize).round();

  bool _isSegmentTooLong(double bufferDuration) {
    return bufferDuration * 1000 >= _maxSegmentDuration;
  }

  bool _isReadyToTranscribePartials() {
    return _enablePartials &&
        _newAudioDurationSeconds * 1000 >= _minPartialDuration;
  }
}

class _PendingFinal {
  _PendingFinal(this.audio, this.duration);
  final Float32List audio;
  final double duration;
}
