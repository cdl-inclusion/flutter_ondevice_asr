
import 'package:flutter/foundation.dart';
import 'package:flutter_ondevice_asr/common/result.dart';
import 'package:flutter_ondevice_asr/models/fastconformer/fastconformer_tokenizer.dart';
import 'package:flutter_test/flutter_test.dart';

import 'test_utils.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  final tokensPath = toAbsolutePath(
    'assets/transcribers/fastconformer/hybrid_int8/tokens.txt',
  );

  debugPrint('Unit test paths (absolute, filesystem-based):');
  debugPrint('  tokensPath: $tokensPath');

  group('FastConformerTokenizer', () {
    final tokenizer = FastConformerTokenizer();

    setUp(() async {
      final result = await tokenizer.loadVocab(path: tokensPath);
      expect(result is Ok, true);
    });

    test('loads the full vocab', () {
      expect(tokenizer.isLoaded, true);
      expect(tokenizer.vocabSize, 1024);
    });

    test('decodes word-piece ids with ▁ -> space', () {
      // 4=▁the, 8=▁and, 29=▁you, 660=▁country
      final text = tokenizer.decodeIds([4, 8, 29, 660]);
      expect(text, 'the and you country');
    });

    test('joins subword pieces without inserting spaces', () {
      // 4=▁the, 1=s  ->  "▁the" + "s" -> " thes" -> "thes"
      final text = tokenizer.decodeIds([4, 1]);
      expect(text, 'thes');
    });

    test('decodes a single word', () {
      expect(tokenizer.decodeIds([660]), 'country');
    });

    test('throws on out-of-range ids (fail fast on a model/tokens mismatch)', () {
      expect(() => tokenizer.decodeIds([4, 999999, 660]), throwsRangeError);
    });

    test('decodeSingleToken keeps ▁ as a leading space (for word grouping)', () {
      expect(tokenizer.decodeSingleToken(4), ' the'); // ▁the
      expect(tokenizer.decodeSingleToken(1), 's'); // s (mid-word piece)
      expect(tokenizer.decodeSingleToken(999999), ''); // out of range -> empty
    });

    test('tokenStartsNewWord reflects the ▁ marker', () {
      expect(tokenizer.tokenStartsNewWord(4), true); // ▁the
      expect(tokenizer.tokenStartsNewWord(1), false); // s
    });

    test('empty id list -> empty string', () {
      expect(tokenizer.decodeIds([]), '');
    });
  });
}
