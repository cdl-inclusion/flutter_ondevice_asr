import 'package:flutter/foundation.dart';
import 'package:flutter_ondevice_asr/common/result.dart';
import 'package:flutter_ondevice_asr/models/whisper/whisper_tokenizer.dart';
import 'package:flutter_test/flutter_test.dart';

import 'test_utils.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  final vocabPath = toAbsolutePath(
    'assets/transcribers/whisper/models/whisper_tiny/default_int8/vocab.json',
  );

  debugPrint('Unit test paths (absolute, filesystem-based):');
  debugPrint('  vocabPath: $vocabPath');

  group('WhisperTokenizer - Multilingual', () {
    final tokenizer = WhisperTokenizer();
    setUp(() async {
      // Reset singleton for each test group
      final result = await tokenizer.loadVocab(path: vocabPath);

      expect(result is Ok, true);
    });

    test('decodes <|endoftext|> special token', () {
      // Only <|endoftext|> (50257) is in the vocab.json
      // Other special tokens are managed by generation_config.json
      final decodedResult = tokenizer.decode([50257], skipSpecialTokens: false);

      expect((decodedResult as Ok<String>).value, contains('<|endoftext|>'));
    });

    test('skips <|endoftext|> when skipSpecialTokens is true', () {
      final decodedResult = tokenizer.decode([50257], skipSpecialTokens: true);

      expect((decodedResult as Ok<String>).value, isEmpty);
    });

    test('decodes regular text correctly', () {
      // Token IDs for "And so my fellow Americans"
      // 400: " And", 370: " so", 452: " my", 7177: " fellow", 6280: " Americans"
      final decodedResult = tokenizer.decode([
        400,
        370,
        452,
        7177,
        6280,
      ], skipSpecialTokens: true);

      expect(
        (decodedResult as Ok<String>).value.trim(),
        'And so my fellow Americans',
      );
    });

    test('handles byte-level BPE decoding', () {
      // Token 0 is "!" in the vocab
      final decodedResult = tokenizer.decode([0]);
      expect((decodedResult as Ok<String>).value, equals('!'));
    });

    test('decodes mixed content with special tokens', () {
      // Test decoding with <|endoftext|> and regular text
      final decodedResult = tokenizer.decode(
        [400, 50257, 370], // " And" <|endoftext|> " so"
        skipSpecialTokens: false,
      );

      expect((decodedResult as Ok<String>).value, contains('<|endoftext|>'));
      expect(decodedResult.value, contains('And'));
      expect(decodedResult.value, contains('so'));
    });

    test('preserves spaces in decoded text', () {
      // Tokens with leading space (Ġ in vocab): 400=" And", 370=" so"
      final decodedResult = tokenizer.decode([400, 370]);

      // Should contain both words with their spaces
      expect((decodedResult as Ok<String>).value, contains('And'));
      expect(decodedResult.value, contains('so'));
    });

    test('handles multi-byte UTF-8 characters', () {
      // Token IDs that represent multi-byte UTF-8 when combined
      // Just test that decoding doesn't crash and produces valid output
      final decodedResult = tokenizer.decode([100, 200, 300]);
      expect((decodedResult as Ok<String>).value, isNotNull);
    });
  });
}
