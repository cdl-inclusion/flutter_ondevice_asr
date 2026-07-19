// ignore_for_file: avoid_print — integration-test logging.

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_ondevice_asr/flutter_ondevice_asr.dart';
import 'package:logging/logging.dart';

import 'transcribe_test_helper.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const testAudioFile = 'packages/flutter_ondevice_asr/assets/audio/jfk_asknot.wav';
  const expectedTranscript =
      'And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.';
  // One HYBRID artifact serves BOTH heads (shared super_encoder + ctc_decoder + decoder_joint).
  const modelDirectory = 'assets/transcribers/fastconformer/int8';
  const language = 'en';

  setUp(() {
    Logger.root.level = Level.ALL;
    Logger.root.onRecord.listen((record) {
      print('${record.loggerName}: ${record.time}: ${record.message}');
    });
  });
  
  // pure CTC
  testWidgets('transcribe test audio (FastConformer CTC)', (tester) async {
    await runTranscribeIntegrationTest(
      type: TranscriberType.fastConformerCtc,
      modelDirectory: modelDirectory,
      language: language,
      testAudioFile: testAudioFile,
      expectedTranscript: expectedTranscript,
      compareNormalized: true,
      label: 'FastConformer CTC',
    );
  });

  // hybrid version effectively running only CTC because of fastDecode=True
  testWidgets('transcribe test audio (FastConformer Hybrid, fastDecode -> CTC)',
      (tester) async {
    await runTranscribeIntegrationTest(
      type: TranscriberType.fastConformerHybrid,
      modelDirectory: modelDirectory,
      language: language,
      testAudioFile: testAudioFile,
      expectedTranscript: expectedTranscript,
      compareNormalized: true,
      fastDecode: true,
      label: 'FastConformer Hybrid (fastDecode CTC)',
    );
  });

  // pure RNN-T
  testWidgets('transcribe test audio (FastConformer RNN-T)', (tester) async {
    await runTranscribeIntegrationTest(
      type: TranscriberType.fastConformerRnnt,
      modelDirectory: modelDirectory,
      language: language,
      testAudioFile: testAudioFile,
      expectedTranscript: expectedTranscript,
      compareNormalized: true,
      label: 'FastConformer RNN-T',
    );
  });

  // hybrid version effectively running only RNN-T because all segments are finals for non-streaming
  testWidgets('transcribe test audio (FastConformer Hybrid, finals -> RNN-T)',
      (tester) async {
    await runTranscribeIntegrationTest(
      type: TranscriberType.fastConformerHybrid,
      modelDirectory: modelDirectory,
      language: language,
      testAudioFile: testAudioFile,
      expectedTranscript: expectedTranscript,
      compareNormalized: true,
      label: 'FastConformer Hybrid (RNN-T)',
    );
  });

}
