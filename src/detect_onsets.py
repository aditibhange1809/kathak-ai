from pathlib import Path
import sys
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Set project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def main():
    # Default to sample3.wav if no file specified
    audio_path = PROJECT_DIR / "data" / "raw" / "sample3.wav"
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])

    if not audio_path.is_file():
        print(f"Error: File '{audio_path}' not found.")
        sys.exit(1)

    print(f"Loading audio for onset detection: {audio_path.name}")
    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    # Detect onset frames using librosa
    # backtrack=True helps snap the marker back to the exact local energy start
    onset_frames = librosa.onset.onset_detect(
        y=audio, sr=sr, backtrack=True, units="frames"
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    print(f"\nDetected {len(onset_times)} individual bol onset markers:")
    print("-" * 40)
    print(f"{'Onset #':<10}{'Timestamp (s)':<15}")
    print("-" * 40)
    for idx, time_val in enumerate(onset_times, 1):
        print(f"{idx:<10}{time_val:<15.3f}")
    print("-" * 40)

    # Plot waveform with detected onset lines
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(audio, sr=sr, alpha=0.6, label="Waveform")
    plt.vlines(
        onset_times,
        ymin=-1,
        ymax=1,
        color="r",
        linestyle="--",
        label="Detected Onsets",
    )
    plt.title(f"Kathak Bol Onset Detection ({audio_path.name})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_image = OUTPUTS_DIR / "onsets.png"
    plt.savefig(out_image)
    print(f"\nOnset plot saved to {out_image}")


if __name__ == "__main__":
    main()