// Streaming performance test — Whisper int8, naive streaming (Whisper has no incremental path).
// Serves BOTH tiny and small: the model dir is overridable at build time via
// --dart-define=MODEL_DIR=... (defaults to tiny), matching the non-streaming whisper perf test.
// The mode label (and hence the output filename) is derived from the model size, so the tiny and
// small workflows each pull their own streaming_perf_whisper[_small]_naive.json.

// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_ondevice_asr/transcriber_type.dart';
import 'package:flutter_ondevice_asr/util/audio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:logging/logging.dart';
import 'package:path_provider/path_provider.dart';

import 'streaming_perf_helper.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const testAudioFile =
      'packages/flutter_ondevice_asr/assets/audio/jfk_asknot.wav';
  const modelDirectory = String.fromEnvironment(
    'MODEL_DIR',
    defaultValue: 'assets/transcribers/whisper/models/whisper_tiny/int8',
  );
  const language = 'en';
  // Always size-qualified: whisper_tiny_naive / whisper_small_naive (drives the output filename).
  final mode = modelDirectory.contains('whisper_small')
      ? 'whisper_small_naive'
      : 'whisper_tiny_naive';

  setUp(() {
    Logger.root.level = Level.ALL;
    Logger.root.onRecord.listen((record) {
      print('${record.loggerName}: ${record.time}: ${record.message}');
    });
  });

  testWidgets('Whisper streaming perf — naive ($mode)', (tester) async {
    final asset = await rootBundle.load(testAudioFile);
    final tmp = await getTemporaryDirectory();
    final f = File('${tmp.path}/perf_audio.wav');
    await f.writeAsBytes(asset.buffer.asUint8List());
    final Float32List audio = await Audio.instance.loadAudio(f.path);

    await runStreamingPerf(
      type: TranscriberType.whisper,
      incremental: false, // Whisper has no incremental path
      modelDirectory: modelDirectory,
      language: language,
      audio: audio,
      mode: mode,
      outPath: '/sdcard/Documents/streaming_perf_$mode.json',
    );
  });
}
