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
  const modelDirectory = 'assets/transcribers/fastconformer/int8';
  const String language = 'en';

  setUp(() {
    Logger.root.level = Level.ALL; // defaults to Level.INFO
    Logger.root.onRecord.listen((record) {
      print('${record.loggerName}: ${record.time}: ${record.message}');
    });
  });

  testWidgets('transcribe test audio (FastConformer CTC)', (tester) async {
    final timeline = await binding.traceTimeline(() async {
      await runTranscribeIntegrationTest(
        type: TranscriberType.fastConformerCtc,
        modelDirectory: modelDirectory,
        language: language,
        testAudioFile: testAudioFile,
        expectedTranscript: expectedTranscript,
        compareNormalized: true,
        label: 'FastConformer CTC',
      );
    }, streams: ["Dart"]);

    final String traceData = const JsonEncoder.withIndent(
      '  ',
    ).convert(timeline.toJson());
    final file = File('/sdcard/Documents/performance_trace_ctc.json');
    await file.writeAsString(traceData);
  });

  testWidgets('transcribe test audio (FastConformer RNN-T)', (tester) async {
    final timeline = await binding.traceTimeline(() async {
      await runTranscribeIntegrationTest(
        type: TranscriberType.fastConformerRnnt,
        modelDirectory: modelDirectory,
        language: language,
        testAudioFile: testAudioFile,
        expectedTranscript: expectedTranscript,
        compareNormalized: true,
        label: 'FastConformer RNN-T',
      );
    }, streams: ["Dart"]);

    final String traceData = const JsonEncoder.withIndent(
      '  ',
    ).convert(timeline.toJson());
    final file = File('/sdcard/Documents/performance_trace_rnnt.json');
    await file.writeAsString(traceData);
  });
}
