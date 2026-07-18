import 'package:logging/logging.dart';

import '../../common/result.dart';
import '../../util/utils.dart';

/// Self-contained (de)-tokenizer for FastConformer (NeMo SentencePiece BPE).
/// Uses `tokens.txt` for detokenizing (`id<TAB>piece`, one per line).
///
/// Uses the SentencePiece word-boundary marker `▁` (U+2581) to denote a
/// leading space. 
/// Detokenization procedure:
/// (1) lookup mapping from id → piece
/// (2) join all pieces
/// (3) replace `▁` with space and trim
/// Mirrors the Python reference `_detok` in `onnx_fastconformer_transcriber.py`.
class FastConformerTokenizer {
  final _logger = Logger('FastConformerTokenizer');

  /// SentencePiece word-boundary marker (leading space).
  static const String _spaceMarker = '▁';

  // index = token id, value = piece (string rep)
  // Empty string for any unused/missing id.
  List<String> _idToPiece = [];

  Future<Result<void>> loadVocab({required String path}) async {
    _logger.fine('Loading tokens from path: <$path>');
    final String content;
    try {
      content = await Utils.loadString(path);
    } catch (e) {
      return Result.error(Exception('Error loading tokens from <$path>: $e'));
    }

    final byId = <int, String>{};
    var maxId = -1;
    for (final rawLine in content.split('\n')) {
      final line = rawLine.endsWith('\r')
          ? rawLine.substring(0, rawLine.length - 1)
          : rawLine;
      if (line.isEmpty) continue;
      final tab = line.indexOf('\t');
      if (tab < 0) continue; // skip malformed lines without tabs
      final id = int.tryParse(line.substring(0, tab));
      if (id == null) continue;
      byId[id] = line.substring(tab + 1);
      if (id > maxId) maxId = id;
    }

    if (maxId < 0) {
      return Result.error(Exception('Error parsing <$path>'));
    }

    final list = List<String>.filled(maxId + 1, '');
    byId.forEach((id, piece) => list[id] = piece);
    _idToPiece = list;
    _logger.fine('Loaded ${byId.length} items (maxId=$maxId)');
    return Result.ok(null);
  }

  bool get isLoaded => _idToPiece.isNotEmpty;

  int get vocabSize => _idToPiece.length;

  /// Token ids -> text: id→piece lookup, join, `▁`→space, trim.
  String decodeIds(List<int> ids) {
    final sb = StringBuffer();
    for (final id in ids) {
      if (id < 0 || id >= _idToPiece.length) {
        throw RangeError(
          'decodeIds: out-of-range token id $id (vocab=${_idToPiece.length})',
        );
      }
      sb.write(_idToPiece[id]);
    }
    return sb.toString().replaceAll(_spaceMarker, ' ').trim();
  }

  /// Decode a single token to its text (with `▁` -> leading space preserved, not
  /// trimmed). Used with [tokenStartsNewWord] to group tokens into words for
  /// word-level detail/confidence. Empty for out-of-range ids.
  String decodeSingleToken(int id) {
    if (id < 0 || id >= _idToPiece.length) return '';
    return _idToPiece[id].replaceAll(_spaceMarker, ' ');
  }

  /// Whether the piece for [id] starts a new word (leading `▁`).
  bool tokenStartsNewWord(int id) {
    if (id < 0 || id >= _idToPiece.length) return false;
    return _idToPiece[id].startsWith(_spaceMarker);
  }
}
