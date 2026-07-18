import 'dart:convert';
import 'dart:io';
import 'package:flutter_ondevice_asr/transcriber_type.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:logging/logging.dart';

import 'transcribe_test_helper.dart';

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const testAudioFile =
      'packages/flutter_ondevice_asr/assets/audio/jfk_asknot.wav';
  const expectedTranscript =
      'And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.';
  const modelDirectory =
      'assets/transcribers/whisper/models/whisper_tiny/default_int8';
  const String language = 'en';

  // Alternative test audio:
  // const testAudioFile = 'packages/flutter_ondevice_asr/assets/audio/crisp_autumn.wav';
  // const expectedTranscript = 'crisp autumn leaves crunch underfoot';
  // Alternative: use external model paths (not bundled):
  // const modelDirectory = '/tmp/onnx_tiny/default';  // multilingual
  // const modelDirectory = '/tmp/onnx_tiny_en/default';  // English-only

  setUp(() {
    Logger.root.level = Level.ALL; // defaults to Level.INFO
    Logger.root.onRecord.listen((record) {
      print('${record.loggerName}: ${record.time}: ${record.message}');
    });
  });

  testWidgets('transcribe test audio', (WidgetTester tester) async {
    final timeline = await binding.traceTimeline(() async {
      await runTranscribeIntegrationTest(
        type: TranscriberType.whisper,
        modelDirectory: modelDirectory,
        language: language,
        testAudioFile: testAudioFile,
        expectedTranscript: expectedTranscript,
        // Whisper matches the reference exactly (it emits standard punctuation/casing).
        compareNormalized: false,
        label: 'Whisper',
      );
    }, streams: ["Dart"]);

    final String traceData = const JsonEncoder.withIndent(
      '  ',
    ).convert(timeline.toJson());
    final file = File('/sdcard/Documents/performance_trace.json');
    await file.writeAsString(traceData);
  });
}
