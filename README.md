# Flutter On-Device ASR

This is a Flutter library for on-device automatic speech recognition (ASR), used in the Project Euphonia Tool.
Please see [Project Euphonia](https://github.com/cdl-inclusion/ProjectEuphonia) for more information on the overall tool as well as other components.

Library overview:
- Model-agnostic architecture supporting arbitrary ASR models
- Streaming and non-streaming transcription
- ONNX Runtime-based inference (different quantization schemes)

The Whisper transcriber implementation has been tested on several different Android devices. When based on Whisper-tiny, it should run on mid-level phones in streaming mode. A fallback to non-streaming or semi-streaming (wait until segment end detected) should allow weaker devices to handle on-device transcription as well.

## Library Structure

This library uses a **model-agnostic architecture** that separates transcription models from streaming logic, making it easy to support multiple ASR models with a unified API.

**Core Abstractions:**
- `Transcriber`: Abstract base interface for all ASR models (defines `loadModels()`, `transcribe()`, `transcribeFile()`)
- `StreamingTranscriber`: Model-agnostic streaming implementation that works with any `Transcriber`
- `TranscriptionResult`: Unified result format for both streaming and non-streaming transcription, containing:
  - Transcribed text
  - Final/partial indicator (for streaming)
  - Audio duration and timestamp
  - Optional word-level data with timing and confidence scores
  - Optional segment-level data and confidence scores
- `Word`: Word-level transcription with confidence score and timing information (start/end times)
- `OnnxConfig`: Runtime configuration for ONNX models

**Model Implementations:**
- `WhisperTranscriber`: Whisper model implementation in `lib/models/whisper/` (implements `Transcriber`)
- Additional models can be added by implementing the `Transcriber` interface

**Shared Components (used by all models):**
- `Audio`: Audio loading and preprocessing utilities
- `SileroVAD`: Voice activity detection for streaming
- Streaming logic: VAD-based segmentation and partial transcription

**Assets:**
- `assets/transcribers/whisper/models/`: Whisper ONNX model files (super_encoder, decoders, configs, vocab)
- `assets/vad/silero_vad/`: Voice activity detection model

## Details on Whisper Implementation

The Whisper implementation uses a modified architecture for efficiency on phones:

### Tokenization

* Uses `WhisperTokenizer` (`lib/models/whisper/whisper_tokenizer.dart`) for text encoding/decoding.
* Works for en-only and multilingual whisper models

### Combined Preprocessing and Encoding

#### ONNX-Native Preprocessing

All audio preprocessing (mel spectrogram extraction) is done within ONNX Runtime using standard operators (HannWindow, STFT, MatMul, Log, Pad). The preprocessor is defined in Python using `onnxscript` (`conversion_tooling/whisper_preprocessor.py`) and exported to ONNX format, ensuring it matches the core Whisper implementation exactly. This approach offers several benefits over Dart-based FFT:
- **Performance**: ~15% faster (160ms → 135ms measured on macOS), with bigger gains expected on Android devices due to NEON/SIMD optimizations
- **Quality**: Better transcription accuracy by using the exact same preprocessing operations as during Whisper training
- **Efficiency**: Fewer memory allocations and reduced GC pressure by keeping data in native memory
- **Portability**: Same ONNX graph works across all platforms

This requires ONNX Runtime 1.22+ (opset 17 support), which is why we use `onnxruntime_v2`.

#### Encoder with Preprocessing Stage

The preprocessing stage (log-mel spectrogram conversion) is merged into the encoder, creating a "super-encoder" that processes raw audio directly in the ONNX graph. This is faster then running a seperate process to extract the log-mel spectrogram and then passing this into the encoder.

By merging the preprocessing and encoder into one onnx model/graph, we eliminate the need to transfer intermediate tensors between the host language (here: Dart) and the inference runtime (her: Onnx). This should be particularly valuable on mobile devices where memory bandwidth is limited and minimizing data transfers between different execution contexts significantly impacts performance and battery life.

It also makes the code much simpler and abstracts preprocessing logic from the app (better guarantee that preprocessing during inference is the same as during training, which is critical for performance as encoder is very sensitive to preprocessing changes).

### Decoder

* We both `decoder.onnx` and `decoder_with_past.onnx` for efficient autoregressive generation with KV-cache. 
* The FFI (Foreign Function Interface) communication in `onnxruntime_v2` enables direct Dart-to-native function calls with minimal overhead, avoiding the serialization cost (serializing/deserializing data) of MethodChannel alternatives (eg flutter_onnxruntime). This is essential for autoregressive decoding, which makes many decoder calls per transcription (depending on output length), where MethodChannel overhead would add overhead of wasted time.

### Asset Generation

All model specific asset files for the Whisper transcriber can be generated using the code in `conversion_tooling`.

This will first download the multilingual Whisper tiny model from HuggingFace, convert to Onnx using Optimum, then create the preprocessor as seperate Onnx model and eventually merge that with the encoder into a "super encoder" onnx model. All files are then moved to the assets folder.

We create both an unquantized as well as the int8 qunatized version. For usage on phones, unless quality impacts are too significant, it is highly recommended to use the int8 version.

## Configuration


### Streaming

Streaming-based system has the following parameters to set:

* **vadThreshold**: VAD sensitivity, 0.0-1.0
  - Higher = less sensitive (fewer false positives, may miss quiet speech)
  - Lower = more sensitive (catches quiet speech, more false positives)
* **eosMinSilence**: Silence duration in ms to end a segment
  - How long to wait after speech stops before finalizing. 
  - Defaults are good for standard speech, but may need adjustment for particularly slow or fast speech.
* **enablePartials**: Emit partial transcriptions during speech. This will trigger a transcriber call whenever enough data for a partial is collected (len >= minPartialDuration) and especially for short minPartialDuration this will lead to significant system use. For weaker devices, it will be important to set minPartialDuration conservatively (ie, high). However, in order for transcriptions to feel real-time we would ideally set minPartialDuration to 300ms.
* **minPartialDuration**: Minimum ms between partial updates. Only relevent of `enablePartials=true`.
* **maxSegmentDuration**: Maximum segment length in ms before forcing end of segment. We limit this to the maximum segment length, Whisper can natively handle (30seconds). Practically, we will often however have shorter max segment length to allow for smooth transcriptions, recommended is 15 secs.


How to set them will depend both on the speaker (wrt to the VAD setting) as well as on the device where transcription is being run.

**Speaker-specific settings:**

* eosMinSilence: if someone speaks very slowly, increase this, so that we don't cut segments too often and then end up transcribing individual words out of context.
* minPartialDuration: for a slow speaker, also increase this. Ideally we have 2 words per partial to transcribe for reasonable transcription quality. 
* enablePartials: if someone speaks really slowly, partial transcripts are probably not very helpful and will instead lead to very poor partial transcripts, likely confusing the speaker. In this case consider turning partials off completely, set the eosMinSilence aligned with the speaker's pausing structure, crank up maxSegmentDuration. Testing with the user will be very important!

**Hardware-specific settings:**

* no all devices will be powerful enough to allow streaming
* a simple inbetween solution is streaming without partials. That will allow the user to just open the microphone and start speaking, but we are limiting transcriptions to whenever full segments (based on VAD events) are captured.
* when we run streaming _with_ partials, the general rule of thumb is: `minPartialDuration >= transcription_time * 1.1`. Ie, if the device needs `500ms` to transcribe a chunk, then the partials should be at least `550ms`, to prevent bursty streaming behavior and a backlog of transcription data.


## Testing

**Unit Tests** (`test/`):
- `audio_test.dart`: Audio loading utilities
- `whisper_tokenizer_test.dart`: Tokenizer encoding/decoding
- `whisper_test.dart`: Whisper transcriber functionality
- `whisper_streaming_test.dart`: Streaming transcriber functionality

Run with: `flutter test`

**Integration Tests** (`integration_test/`):

Run with: `flutter test integration_test/`

## Performance Measurement

Measured on test audio (`assets/audio/jfk_asknot.wav`, 11 seconds) in non-streaming mode. Run `flutter run --release -d <DEVICE_ID> integration_test/whisper_test.dart` (from `example/` directory) to measure on your device.

| Device | Model | Inference time (avg ± std) |
| -- | -- | -- |
| Macbook Pro M2 | default | 576.4 ± 37.3 ms |
| Macbook Pro M2 | default_int8 | 477.8 ± 22.1 ms |
| Samsung Galaxy 11A+ Tablet (SM X230)| default | 7758.6 ± 121.4 ms |
| Samsung Galaxy 11A+ Tablet (SM X230) | default_int8 | 1396.6 ± 143.0 ms |
| Pixel 6a | default_int8 | 917.2 ± 53.2 ms |
| Huawei Y9 Prime 2019 (STK-L21) | default_int8 | 3370.0 +- 120.5 ms|
| Samsung Tablet SM X115 | default_int8 | 1509.8 +- 48.9 ms|



**Quantization Impact:**
- Mac M4: int8 provides ~1.2x speedup (minimal benefit)
- Samsung Tablet: int8 provides ~6.8x speedup (significant benefit)

The int8 quantization is much more effective on mobile devices with less optimized hardware for full-precision inference.

Future work: Extend measurements to corpus with varying audio lengths.


## Installation

### Setting Up This Project for Development

### Clone the repository and install dependencies
   ```bash
   git clone <repository-url>
   cd flutter_ondevice_asr
   flutter pub get
   ```

### Generate assets

The library requires Whisper model files and tokenizer vocabularies. You need to generate them with the `conversion_tooling/`.

Create a python environment and the required dependencies:
```
cd conversion_tooling
python -m venv venv
source venv/bin/activate
pip install -r requirements
```

Then run the asset generation script
```
cd ..
./build_assets.sh
```


This script runs through these steps:

1. Downloads Whisper models from HuggingFace using `convert_whisper_to_onnx.py`
2. Generates preprocessor with `export_whisper_preprocessor.py`
3. Merges preprocessor + encoder into super-encoder using `merge_preprocessor_encoder.py`
4. Outputs to `assets/transcribers/whisper/models/{default,default_int8,default_int8_optimum}/`


### Running Unit Tests

Unit tests (`flutter test`) run in a pure Dart VM without building native libraries. Since this library uses `onnxruntime_v2` (which uses FFI to call native ONNX Runtime), you need to install the native library locally for unit tests to work.

The unit tests are on purpose maximally verbose. For the transcription and especially the streaming transcription test, this will show information for every 512 frame sample being recorded and allows to track the streaming decisions for debugging purposes.

This level of verbosity needs to be avoided in a production setting!

Integration tests (`integration_test/`) don't require this setup as they build the full app with native libraries included automatically.

### Running performance tests

Connect an android device, and run the integration test whisper_non_streaming_performance_test.dart on it

```
flutter drive --driver=test_driver/perf_driver.dart --target=integration_test/whisper_test.dart --profile --no-dds --dart-define=PERFORMANCE=true
```

this generates a trace file in example/build named `performance_trace.json`. You can open 
this file in chrome://tracing.

### Running performance tests on Firebase Test lab

```
cd example/android
./gradlew app:assembleProfile -Ptarget="integration_test/whisper_non_streaming_firebase_performance_test.dart"
./gradlew app:assembleAndroidTest

gcloud firebase test android run --type instrumentation \
    --app ../build/app/outputs/apk/profile/app-profile.apk \
    --test ../build/app/outputs/apk/androidTest/debug/app-debug-androidTest.apk \
    --device model=gta9pwifi,version=34,locale=en,orientation=portrait \
    --device model=akita,version=34,locale=en,orientation=portrait \
    --timeout 15m \
    --directories-to-pull /sdcard/Documents
```


### macOS Setup

1. **Install ONNX Runtime via Homebrew:**
   ```bash
   brew install onnxruntime
   ```

   This installs ONNX Runtime 1.23.2+ (compatible with `onnxruntime_v2: ^1.23.2`).

2. **Create version compatibility symlink:**

   The package looks for `libonnxruntime.1.21.0.dylib`, but Homebrew installs version 1.23.2. Create a compatibility symlink:
   ```bash
   cd /opt/homebrew/Cellar/onnxruntime/1.23.2_2/lib
   ln -s libonnxruntime.1.23.2.dylib libonnxruntime.1.21.0.dylib
   ```

   Note: Adjust the version path (`1.23.2_2`) if Homebrew installed a different version. Check with: `ls /opt/homebrew/Cellar/onnxruntime/`

3. **Create symlink in project root:**
   ```bash
   cd /path/to/flutter_onnx_whisper
   ln -sf /opt/homebrew/Cellar/onnxruntime/1.23.2_2/lib/libonnxruntime.1.21.0.dylib .
   ```

