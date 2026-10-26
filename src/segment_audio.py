from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Set project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def get_audio_filepath():
    """Retrieves audio path from CLI argument or interactive prompt."""
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        if file_path.is_file():
            return file_path
        print(f"Error: Specified file '{file_path}' not found.")

    raw_dir = PROJECT_DIR / "data" / "raw"
    audio_files = sorted(
        list(raw_dir.glob("*.wav")) + list(raw_dir.glob("*.mp3"))
    )

    if not audio_files:
        print(f"No audio files found in {raw_dir}")
        sys.exit(1)

    print("\nAvailable audio files:")
    for idx, filepath in enumerate(audio_files, 1):
        print(f"  [{idx}] {filepath.name}")

    try:
        choice = int(input("\nEnter the file number: ")) - 1
        if 0 <= choice < len(audio_files):
            return audio_files[choice]
    except ValueError:
        pass

    print("Invalid selection.")
    sys.exit(1)


def compute_rms(audio, frame_length, hop_length):
    """Calculates Root Mean Square (RMS) energy for framing windows."""
    num_frames = 1 + (len(audio) - frame_length) // hop_length
    rms = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        rms[i] = np.sqrt(np.mean(frame**2))
    return rms


def detect_activity_segments(
    audio, sr, frame_ms=25, hop_ms=10, max_gap_ms=100
):
    """Detects active sound intervals based on RMS energy thresholding."""
    frame_length = int(sr * (frame_ms / 1000.0))
    hop_length = int(sr * (hop_ms / 1000.0))

    rms = compute_rms(audio, frame_length, hop_length)

    # Activity threshold
    threshold = max(0.01 * rms.max(), 0.002)
    active_frames = rms > threshold

    # Merge small inactive gaps
    gap_frames = int((max_gap_ms / 1000.0) / (hop_ms / 1000.0))
    padded_active = active_frames.copy()

    for i in range(len(active_frames)):
        if not active_frames[i]:
            # Check if surrounded by active frames within max_gap_ms
            prev_active = np.any(active_frames[max(0, i - gap_frames) : i])
            next_active = np.any(
                active_frames[i + 1 : min(len(active_frames), i + gap_frames)]
            )
            if prev_active and next_active:
                padded_active[i] = True

    # Convert active frame boolean array to timestamp intervals
    segments = []
    in_segment = False
    start_time = 0.0

    for i, active in enumerate(padded_active):
        frame_time = i * (hop_ms / 1000.0)
        if active and not in_segment:
            in_segment = True
            start_time = frame_time
        elif not active and in_segment:
            in_segment = False
            end_time = frame_time
            segments.append((start_time, end_time))

    if in_segment:
        segments.append((start_time, len(audio) / sr))

    return rms, threshold, segments, hop_length


def main():
    audio_path = get_audio_filepath()
    print(f"\nProcessing audio: {audio_path.name}")

    audio, sr = sf.read(audio_path)

    # Convert to mono if multi-channel
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    rms, threshold, segments, hop_length = detect_activity_segments(
        audio, sr
    )

    print(f"\nDetected {len(segments)} candidate activity segments:")
    print("-" * 50)
    print(f"{'Segment':<10}{'Start (s)':<12}{'End (s)':<12}{'Duration (s)':<12}")
    print("-" * 50)
    for idx, (start, end) in enumerate(segments, 1):
        duration = end - start
        print(f"{idx:<10}{start:<12.3f}{end:<12.3f}{duration:<12.3f}")
    print("-" * 50)

    # Plot RMS & Segments
    times = np.arange(len(rms)) * (10 / 1000.0)  # 10ms hop
    plt.figure(figsize=(12, 4))
    plt.plot(times, rms, label="RMS energy")
    plt.axhline(
        y=threshold,
        color="r",
        linestyle="--",
        label="Activity threshold",
    )

    for start, end in segments:
        plt.axvspan(start, end, color="g", alpha=0.2)

    plt.title(f"Kathak Audio: Detected Activity Segments ({audio_path.name})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("RMS energy")
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_image = OUTPUTS_DIR / "audio_segments.png"
    plt.savefig(out_image)
    print(f"\nPlot saved to {out_image}")


if __name__ == "__main__":
    main()