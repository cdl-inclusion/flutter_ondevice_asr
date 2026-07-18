import 'dart:math';
import 'dart:typed_data';

/// Ring buffer using pre-allocated Float32List for efficient audio streaming.
/// Uses setRange() for fast bulk copies instead of element-by-element loops.
class Float32RingBuffer {
  Float32List _data;
  int _start = 0;
  int _length = 0;

  Float32RingBuffer([int initialCapacity = 1024])
    : _data = Float32List(initialCapacity < 1 ? 1 : initialCapacity);

  int get length => _length;

  int get _capacity => _data.length;

  void addAll(Float32List data) {
    _ensureCapacity(_length + data.length);

    final writePos = (_start + _length) % _capacity;
    final firstPart = min(data.length, _capacity - writePos);

    // Use setRange for fast memcpy-style copy
    _data.setRange(writePos, writePos + firstPart, data);

    if (firstPart < data.length) {
      // Wrap around to beginning
      _data.setRange(0, data.length - firstPart, data, firstPart);
    }

    _length += data.length;
  }

  /// Drain exactly [count] samples into a new Float32List.
  /// Caller must check length >= count first.
  Float32List consume(int count) {
    if (count > _length) {
      throw StateError(
        'Cannot consume $count samples, only $_length available',
      );
    }

    final out = Float32List(count);
    final firstPart = min(count, _capacity - _start);

    // Use setRange for fast memcpy-style copy
    out.setRange(0, firstPart, _data, _start);

    if (firstPart < count) {
      // Wrap around - read from beginning
      out.setRange(firstPart, count, _data, 0);
    }

    _start = (_start + count) % _capacity;
    _length -= count;

    return out;
  }

  void removeFromFront(int count) {
    if (count > _length) {
      count = _length;
    }
    _start = (_start + count) % _capacity;
    _length -= count;
  }

  void keepTail(int count) {
    if (count >= _length) return;
    final toRemove = _length - count;
    _start = (_start + toRemove) % _capacity;
    _length = count;
  }

  void clear() {
    _start = 0;
    _length = 0;
  }

  void _ensureCapacity(int required) {
    if (required <= _capacity) return;

    // Grow by 2x or to required size, whichever is larger
    final newCapacity = max(_capacity * 2, required);
    final newData = Float32List(newCapacity);

    // Copy existing data to new buffer (linearized)
    if (_length > 0) {
      final firstPart = min(_length, _capacity - _start);
      newData.setRange(0, firstPart, _data, _start);

      if (firstPart < _length) {
        // Wrap around - copy from beginning of old buffer
        newData.setRange(firstPart, _length, _data, 0);
      }
    }

    _data = newData;
    _start = 0;
  }
}
