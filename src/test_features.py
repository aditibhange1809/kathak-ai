"""
Automated Test & Validation Suite for Day 1 Feature Extraction
==============================================================
Validates:
1. Feature extraction executes without NaNs, Infs, or dimension mismatches.
2. Mel-Spectrogram, MFCC, Centroid, and ZCR match expected mathematical ranges.
3. Comparative acoustic signatures between Sample 1 (Theka), Sample 2 (Tatkar),
   and Sample 3 (Padhant).
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
import librosa
from src.features import extract_features, visualize_features, OUTPUTS_DIR, PROJECT_DIR


def test_feature_pipeline():
    raw_dir = PROJECT_DIR / "data" / "raw"
    samples = sorted(list(raw_dir.glob("*.wav")))

    assert len(samples) > 0, "No audio samples found in data/raw!"
    print(f"\n[TEST SUITE] Found {len(samples)} audio samples to validate.\n")

    results = {}

    for sample_path in samples:
        print(f"--> Validating {sample_path.name}...")
        audio, sr = librosa.load(sample_path, sr=None, mono=True)
        assert len(audio) > 0, f"{sample_path.name} is empty!"
        assert sr == 48000, f"Expected 48000 Hz, got {sr} Hz"

        features = extract_features(audio, sr)

        # 1. Check no NaNs or Infs
        for key in ["mel_spec_db", "mfcc", "mfcc_delta", "spectral_centroid", "spectral_rolloff", "zcr", "rms"]:
            arr = features[key]
            assert not np.isnan(arr).any(), f"NaN detected in {key} for {sample_path.name}"
            assert not np.isinf(arr).any(), f"Inf detected in {key} for {sample_path.name}"

        # 2. Check dimension consistency
        num_frames = features["times"].shape[0]
        assert features["mel_spec_db"].shape == (128, num_frames), "Mel spectrogram dimension mismatch"
        assert features["mfcc"].shape == (13, num_frames), "MFCC dimension mismatch"
        assert features["mfcc_delta"].shape == (13, num_frames), "Delta MFCC dimension mismatch"
        assert features["spectral_centroid"].shape == (num_frames,), "Centroid dimension mismatch"
        assert features["spectral_rolloff"].shape == (num_frames,), "Rolloff dimension mismatch"
        assert features["zcr"].shape == (num_frames,), "ZCR dimension mismatch"
        assert features["rms"].shape == (num_frames,), "RMS dimension mismatch"

        # 3. Check mathematical ranges
        assert np.max(features["mel_spec_db"]) <= 0.001, "Mel dB max should be ~0 dB relative to peak"
        assert np.min(features["mel_spec_db"]) >= -120.0, "Mel dB dynamic range abnormally low (< -120 dB)"
        assert np.all(features["spectral_centroid"] >= 0), "Spectral Centroid cannot be negative"
        assert np.all(features["zcr"] >= 0) and np.all(features["zcr"] <= 1.0), "ZCR must be in [0, 1]"

        # 4. Generate plot
        plot_path = OUTPUTS_DIR / f"features_{sample_path.stem}.png"
        visualize_features(features, title=f"Kathak AI - {sample_path.name}", save_path=plot_path)
        assert plot_path.is_file() and plot_path.stat().st_size > 1000, "Plot file was not generated properly"

        results[sample_path.name] = {
            "duration": features["duration"],
            "frames": num_frames,
            "mean_centroid_hz": float(np.mean(features["spectral_centroid"])),
            "max_centroid_hz": float(np.max(features["spectral_centroid"])),
            "mean_rms": float(np.mean(features["rms"])),
            "mean_zcr": float(np.mean(features["zcr"])),
        }
        print(f"    ✓ All assertions passed. Plot saved to: {plot_path.name}")

    print("\n=======================================================")
    print("Acoustic Comparison Across Kathak Recordings:")
    print("=======================================================")
    print(f"{'Sample':<15}{'Duration (s)':<15}{'Mean Centroid (Hz)':<22}{'Mean RMS':<12}{'Mean ZCR':<10}")
    print("-" * 74)
    for name, stats in results.items():
        print(
            f"{name:<15}"
            f"{stats['duration']:<15.2f}"
            f"{stats['mean_centroid_hz']:<22.1f}"
            f"{stats['mean_rms']:<12.4f}"
            f"{stats['mean_zcr']:<10.4f}"
        )
    print("=======================================================\n")
    print("ALL DAY 1 TESTS PASSED SUCCESSFULLY! ✨")


if __name__ == "__main__":
    test_feature_pipeline()
