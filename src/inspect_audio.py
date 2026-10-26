
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs"

if len(sys.argv) >= 2:
    AUDIO_PATH = Path(sys.argv[1]).expanduser()

    # Resolve relative paths from the project folder
    if not AUDIO_PATH.is_absolute():
        AUDIO_PATH = PROJECT_DIR / AUDIO_PATH
else:
    # Display available recordings
    audio_files = sorted(
        list(RAW_DIR.glob("*.wav")) +
        list(RAW_DIR.glob("*.mp3")) +
        list(RAW_DIR.glob("*.flac"))
    )

    if not audio_files:
        raise FileNotFoundError(
            f"No audio files found in {RAW_DIR}"
        )

    print("Available audio files:")
    for i, file in enumerate(audio_files, start=1):
        print(f"{i}. {file.name}")

    choice = input("\nEnter the file number: ").strip()

    if not choice.isdigit() or not (1 <= int(choice) <= len(audio_files)):
        raise ValueError("Invalid selection. Run the script again.")

    AUDIO_PATH = audio_files[int(choice) - 1]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not AUDIO_PATH.is_file():
    raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stop with a useful message if the recording is missing.
if not AUDIO_PATH.exists():
    raise FileNotFoundError(
        f"Audio file not found: {AUDIO_PATH}\n"
        "Place your recording at data/raw/sample.wav"
    )

# Load the audio samples and their sample rate.
audio, sample_rate = sf.read(AUDIO_PATH)

# Convert the samples to a NumPy array.
audio = np.asarray(audio)

# Calculate basic properties.
num_samples = len(audio)
duration_seconds = num_samples / sample_rate

print("Audio analysis")
print("--------------------")
print(f"File: {AUDIO_PATH.name}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Samples: {num_samples}")
print(f"Duration: {duration_seconds:.2f} seconds")
print(f"Array shape: {audio.shape}")

if audio.ndim == 1:
    print("Channels: 1 (mono)")
else:
    print(f"Channels: {audio.shape[1]}")

# For a basic waveform plot, use one channel if stereo.
if audio.ndim == 2:
    signal = audio[:, 0]
else:
    signal = audio

# Create a time value for every sample.
time = np.arange(len(signal)) / sample_rate

# Plot and save the waveform.
plt.figure(figsize=(12, 4))
plt.plot(time, signal, linewidth=0.5)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Kathak Recording Waveform")
plt.tight_layout()

plot_path = OUTPUT_DIR / "waveform.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"Waveform saved to: {plot_path}")
