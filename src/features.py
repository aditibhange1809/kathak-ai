"""
Audio Feature Extraction for Kathak AI
=======================================
Extracts time-frequency and timbral features essential for Indian Classical
Dance audio analysis:
  - Mel-Spectrogram (Frequency-band energy distribution)
  - MFCCs (Mel-Frequency Cepstral Coefficients + Delta velocities)
  - Spectral Centroid (Timbre brightness: Ghungroos vs. Bass Bols)
  - Spectral Rolloff (High-frequency cutoff)
  - Zero-Crossing Rate (Percussive/unvoiced detection)
"""

from pathlib import Path
import sys
from typing import Dict, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_features(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    n_mfcc: int = 13,
) -> Dict[str, np.ndarray]:
    """
    Extracts a comprehensive set of spectral and timbral features from audio.

    Args:
        audio: 1D NumPy array representing mono audio signal.
        sr: Sampling rate in Hz.
        n_fft: Window length for Short-Time Fourier Transform (STFT).
        hop_length: Step size between successive analysis frames.
        n_mels: Number of Mel filter bands.
        n_mfcc: Number of Mel-Frequency Cepstral Coefficients.

    Returns:
        Dict mapping feature names to their computed 1D or 2D NumPy arrays.
    """
    # Ensure mono float32
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    # 1. Short-Time Fourier Transform (STFT)
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
    magnitude_spectrogram = np.abs(stft)

    # 2. Mel-Spectrogram (Perceptually scaled frequency energy)
    mel_spec = librosa.feature.melspectrogram(
        S=magnitude_spectrogram**2,
        sr=sr,
        n_mels=n_mels,
        fmax=sr // 2,
    )
    # Convert to log scale (Decibels) to match human ear sensitivity
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 3. MFCCs (Mel-Frequency Cepstral Coefficients)
    # Encodes the spectral envelope (phonetic identity of Kathak bols)
    mfcc = librosa.feature.mfcc(S=mel_spec_db, n_mfcc=n_mfcc)
    
    # Delta MFCCs (First derivative: rate of change / bol transition velocity)
    mfcc_delta = librosa.feature.delta(mfcc)

    # 4. Spectral Centroid (Spectral 'Center of Mass' - indicates brightness)
    # Ghungroos produce high centroid (>4000 Hz); Tabla Bayan produces low (<500 Hz)
    spectral_centroid = librosa.feature.spectral_centroid(
        S=magnitude_spectrogram, sr=sr
    )[0]

    # 5. Spectral Rolloff (Frequency below which 85% of spectral energy lies)
    spectral_rolloff = librosa.feature.spectral_rolloff(
        S=magnitude_spectrogram, sr=sr, roll_percent=0.85
    )[0]

    # 6. Zero-Crossing Rate (ZCR)
    # Sharp percussive strikes or unvoiced consonants produce high ZCR
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length)[0]

    # 7. RMS Energy per frame
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]

    # Time axis for frame-level features
    frames = np.arange(mel_spec_db.shape[1])
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    return {
        "mel_spec_db": mel_spec_db,
        "mfcc": mfcc,
        "mfcc_delta": mfcc_delta,
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff": spectral_rolloff,
        "zcr": zcr,
        "rms": rms,
        "times": times,
        "sr": sr,
        "hop_length": hop_length,
        "duration": len(audio) / sr,
    }


def visualize_features(
    features: Dict[str, np.ndarray],
    title: str = "Kathak Audio Feature Analysis",
    save_path: Path = None,
) -> Path:
    """
    Renders an informative 4-panel dashboard of spectral and timbral features.
    """
    times = features["times"]
    sr = features["sr"]
    hop_length = features["hop_length"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Panel 1: Mel Spectrogram
    img1 = librosa.display.specshow(
        features["mel_spec_db"],
        x_axis="time",
        y_axis="mel",
        sr=sr,
        hop_length=hop_length,
        ax=axes[0],
        cmap="magma",
    )
    axes[0].set_title(f"{title} - Mel-Spectrogram (dB)")
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

    # Panel 2: MFCCs Heatmap (Coefficients 1 to 13)
    img2 = librosa.display.specshow(
        features["mfcc"],
        x_axis="time",
        sr=sr,
        hop_length=hop_length,
        ax=axes[1],
        cmap="coolwarm",
    )
    axes[1].set_ylabel("MFCC Index")
    axes[1].set_title("MFCCs (Timbral Envelope of Spoken Bols)")
    fig.colorbar(img2, ax=axes[1])

    # Panel 3: Spectral Centroid & Spectral Rolloff
    axes[2].plot(times, features["spectral_centroid"], color="cyan", label="Spectral Centroid (Hz)", linewidth=1.2)
    axes[2].plot(times, features["spectral_rolloff"], color="orange", linestyle="--", label="Spectral Rolloff 85% (Hz)", alpha=0.8)
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_title("Spectral Centroid & Rolloff (Brightness & Ghungroo Activity)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Energy (RMS) & Zero Crossing Rate
    ax4_twin = axes[3].twinx()
    line1 = axes[3].plot(times, features["rms"], color="limegreen", label="RMS Energy", linewidth=1.5)
    line2 = ax4_twin.plot(times, features["zcr"], color="crimson", alpha=0.6, label="Zero-Crossing Rate (ZCR)", linewidth=1)
    axes[3].set_ylabel("RMS Energy", color="limegreen")
    ax4_twin.set_ylabel("ZCR", color="crimson")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_title("Energy Dynamics & Percussive Transients")
    
    # Combined legend for twin axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[3].legend(lines, labels, loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = OUTPUTS_DIR / "spectral_features.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def main():
    # Allow passing file from command line or default to sample.wav
    audio_file = PROJECT_DIR / "data" / "raw" / "sample.wav"
    if len(sys.argv) > 1:
        audio_file = Path(sys.argv[1])
        if not audio_file.is_absolute():
            audio_file = PROJECT_DIR / audio_file

    if not audio_file.is_file():
        print(f"Error: Audio file not found at {audio_file}")
        sys.exit(1)

    print(f"Loading audio for feature extraction: {audio_file.name}")
    audio, sr = librosa.load(audio_file, sr=None, mono=True)

    print("Extracting spectral, timbral, and dynamic features...")
    features = extract_features(audio, sr)

    print("\nFeature Extraction Summary:")
    print("--------------------------------------------------")
    print(f"Sample Rate:          {features['sr']} Hz")
    print(f"Duration:             {features['duration']:.2f} seconds")
    print(f"Mel-Spectrogram Shape: {features['mel_spec_db'].shape} (Mels x Frames)")
    print(f"MFCC Matrix Shape:    {features['mfcc'].shape} (Coefficients x Frames)")
    print(f"Mean Centroid:        {np.mean(features['spectral_centroid']):.1f} Hz")
    print(f"Max Centroid:         {np.max(features['spectral_centroid']):.1f} Hz")
    print(f"Mean RMS Energy:      {np.mean(features['rms']):.4f}")
    print(f"Mean ZCR:             {np.mean(features['zcr']):.4f}")
    print("--------------------------------------------------")

    out_name = f"features_{audio_file.stem}.png"
    out_path = OUTPUTS_DIR / out_name
    visualize_features(features, title=f"Kathak AI: {audio_file.name}", save_path=out_path)
    print(f"Visualization saved successfully to: {out_path}")


if __name__ == "__main__":
    main()
