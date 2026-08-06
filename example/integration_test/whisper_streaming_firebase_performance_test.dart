// Streaming performance test — Whisper tiny int8, naive streaming.
// Writes one JSON summary to /sdcard/Documents (pulled by the workflow).

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
  const modelDirectory =
      'assets/transcribers/whisper/models/whisper_tiny/int8';
  const language = 'en';

  setUp(() {
    Logger.root.level = Level.ALL;
    Logger.root.onRecord.listen((record) {
      print('${record.loggerName}: ${record.time}: ${record.message}');
    });
  });

  testWidgets('Whisper tiny streaming perf — naive', (tester) async {
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
      mode: 'whisper_naive',
      outPath: '/sdcard/Documents/streaming_perf_whisper_naive.json',
    );
  });
}
