// ignore_for_file: avoid_print — integration-test progress logging; print is intentional.

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
  // Overridable at build time via --dart-define=MODEL_DIR=... so the same test
  // can run against different Whisper model sizes (tiny, small, ...).
  const modelDirectory = String.fromEnvironment(
    'MODEL_DIR',
    defaultValue: 'assets/transcribers/whisper/models/whisper_tiny/int8',
  );
  const String language = 'en';

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
        // Normalized (P&C-/whitespace-invariant): the SAME test runs across Whisper sizes via
        // MODEL_DIR, and larger models punctuate differently (whisper-small adds a comma after
        // "Americans" that tiny omits). For a perf test, matching the words is the right check.
        compareNormalized: true,
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
