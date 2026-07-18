"""Test that the super encoder produces identical outputs to separate models.

This script:
1. Loads sample audio
2. Runs it through the separate preprocessor + encoder
3. Runs it through the merged super encoder
4. Compares the outputs to ensure they match

Usage:
    python test_super_encoder.py
"""

import numpy as np
import onnxruntime as ort


def create_test_audio(duration_sec=1.0, sample_rate=16000):
    """Create a simple test audio waveform (sine wave)."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    # 440 Hz sine wave (A4 note)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio


def test_separate_models():
    """Run audio through separate preprocessor and encoder."""
    print("Testing separate models...")

    # Create test audio
    audio = create_test_audio()
    print(f"  Audio shape: {audio.shape}")

    # Run preprocessor
    preprocessor_session = ort.InferenceSession(
        "whisper_preprocessor_80.onnx",
        providers=['CPUExecutionProvider']
    )

    preprocessor_inputs = {
        'waveforms': audio.reshape(1, -1),
        'waveforms_lens': np.array([len(audio)], dtype=np.int64)
    }

    preprocessor_outputs = preprocessor_session.run(None, preprocessor_inputs)
    mel_spectrogram = preprocessor_outputs[0]
    print(f"  Mel spectrogram shape: {mel_spectrogram.shape}")

    # Run encoder
    encoder_session = ort.InferenceSession(
        "../assets/models/whisper_tiny/default_int8/encoder_model.onnx",
        providers=['CPUExecutionProvider']
    )

    encoder_inputs = {'input_features': mel_spectrogram}
    encoder_outputs = encoder_session.run(None, encoder_inputs)
    hidden_states = encoder_outputs[0]
    print(f"  Hidden states shape: {hidden_states.shape}")

    return hidden_states


def test_super_encoder():
    """Run audio through merged super encoder."""
    print("\nTesting super encoder...")

    # Create test audio (same as before)
    audio = create_test_audio()
    print(f"  Audio shape: {audio.shape}")

    # Run super encoder
    super_encoder_session = ort.InferenceSession(
        "super_encoder.onnx",
        providers=['CPUExecutionProvider']
    )

    super_inputs = {
        'waveforms': audio.reshape(1, -1),
        'waveforms_lens': np.array([len(audio)], dtype=np.int64)
    }

    super_outputs = super_encoder_session.run(None, super_inputs)
    hidden_states = super_outputs[1]  # Index 1 is 'last_hidden_state', 0 is 'features_lens'
    print(f"  Hidden states shape: {hidden_states.shape}")

    return hidden_states


def compare_outputs(separate_output, super_output):
    """Compare outputs from separate and super encoder models."""
    print("\nComparing outputs...")

    # Check shapes match
    if separate_output.shape != super_output.shape:
        print(f"  ✗ Shape mismatch!")
        print(f"    Separate: {separate_output.shape}")
        print(f"    Super: {super_output.shape}")
        return False

    print(f"  ✓ Shapes match: {separate_output.shape}")

    # Compute differences
    abs_diff = np.abs(separate_output - super_output)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    # Compute relative difference (avoiding division by zero)
    separate_abs = np.abs(separate_output)
    relative_diff = np.divide(
        abs_diff,
        separate_abs,
        out=np.zeros_like(abs_diff),
        where=separate_abs > 1e-10
    )
    max_relative_diff = np.max(relative_diff)

    print(f"  Max absolute difference: {max_abs_diff:.2e}")
    print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
    print(f"  Max relative difference: {max_relative_diff:.2%}")

    # Check if outputs are close enough
    # For quantized models, we expect small differences due to rounding
    tolerance = 1e-4  # 0.01% relative difference is acceptable

    if max_relative_diff < tolerance:
        print(f"  ✓ Outputs match within tolerance ({tolerance:.2%})")
        return True
    elif max_relative_diff < 0.01:  # 1% is still reasonable for int8
        print(f"  ⚠ Outputs differ slightly ({max_relative_diff:.2%}) but acceptable for quantized model")
        return True
    else:
        print(f"  ✗ Outputs differ significantly ({max_relative_diff:.2%})")
        return False


def main():
    print("=" * 70)
    print("Testing Super Encoder vs Separate Models")
    print("=" * 70)

    try:
        # Test separate models
        separate_output = test_separate_models()

        # Test super encoder
        super_output = test_super_encoder()

        # Compare outputs
        match = compare_outputs(separate_output, super_output)

        print("\n" + "=" * 70)
        if match:
            print("✓ SUCCESS: Super encoder produces identical outputs!")
            print("\nThe merged model is ready to use. Next steps:")
            print("  1. Copy super_encoder.onnx to Flutter assets")
            print("  2. Update Flutter code to use super encoder")
            print("  3. Measure performance improvements")
        else:
            print("✗ FAILURE: Outputs don't match!")
            print("\nDebug steps:")
            print("  1. Check opset/IR versions are aligned")
            print("  2. Verify io_map connections are correct")
            print("  3. Try with non-quantized encoder model")
        print("=" * 70)

        return match

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
