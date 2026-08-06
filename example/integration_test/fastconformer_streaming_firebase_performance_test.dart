// Streaming performance test — FastConformer int8 hybrid, naive vs incremental.
// Writes one JSON summary per mode to /sdcard/Documents (pulled by the workflow).

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
  const modelDirectory = 'assets/transcribers/fastconformer/int8';
  const language = 'en';

  setUp(() {
    Logger.root.level = Level.ALL;
    Logger.root.onRecord.listen((record) {
      print('${record.loggerName}: ${record.time}: ${record.message}');
    });
  });

  Future<Float32List> loadAudio() async {
    final asset = await rootBundle.load(testAudioFile);
    final tmp = await getTemporaryDirectory();
    final f = File('${tmp.path}/perf_audio.wav');
    await f.writeAsBytes(asset.buffer.asUint8List());
    return Audio.instance.loadAudio(f.path);
  }

  testWidgets('FastConformer hybrid streaming perf — naive', (tester) async {
    await runStreamingPerf(
      type: TranscriberType.fastConformerHybrid,
      incremental: false,
      modelDirectory: modelDirectory,
      language: language,
      audio: await loadAudio(),
      mode: 'fc_naive',
      outPath: '/sdcard/Documents/streaming_perf_fc_naive.json',
    );
  });

  testWidgets('FastConformer hybrid streaming perf — incremental', (tester) async {
    await runStreamingPerf(
      type: TranscriberType.fastConformerHybrid,
      incremental: true,
      modelDirectory: modelDirectory,
      language: language,
      audio: await loadAudio(),
      mode: 'fc_incremental',
      outPath: '/sdcard/Documents/streaming_perf_fc_incremental.json',
    );
  });
}
