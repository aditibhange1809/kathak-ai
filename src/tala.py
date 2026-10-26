"""
Tala & Rhythm Tracking Engine for Kathak AI
============================================
Tracks rhythmic structure, tempo (Laya), and meter alignment in Indian
Classical Dance, specifically focused on Teental (16 Matras):
  - Autocorrelation / Librosa tempo and beat tracking
  - Canonical 16-Matra Teental Grid Generation (Sam, Taali, Khali)
  - Onset Quantization & Micro-timing Error (Jitter) Calculation
  - Laya Stability & Tempo Drift Analysis
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np

# Project root setup
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Standard Teental Theka bols for the 16 Matras
TEENTAL_THEKA = [
    "Dha", "Dhin", "Dhin", "Dha",  # Vibhag 1 (Sam - Beat 1)
    "Dha", "Dhin", "Dhin", "Dha",  # Vibhag 2 (Taali - Beat 5)
    "Dha", "Tin",  "Tin",  "Ta",   # Vibhag 3 (Khali - Beat 9)
    "Ta",  "Dhin", "Dhin", "Dha",  # Vibhag 4 (Taali - Beat 13)
]


@dataclass
class MatraInfo:
    cycle: int
    matra: int  # 1 to 16
    timestamp: float
    action: str  # 'SAM', 'TAALI_2', 'KHALI', 'TAALI_3', 'MATRA'
    canonical_bol: str


def get_matra_action(matra: int) -> str:
    """Returns the structural classical action for a given Teental matra."""
    if matra == 1:
        return "SAM"
    elif matra == 5:
        return "TAALI_2"
    elif matra == 9:
        return "KHALI"
    elif matra == 13:
        return "TAALI_3"
    return "MATRA"


def estimate_tempo_and_beats(
    audio: np.ndarray,
    sr: int,
    start_bpm: float = 60.0
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimates global tempo (BPM) and dynamic beat timestamps.

    Returns:
        tempo_bpm: Estimated tempo in beats per minute.
        beat_times: Timestamps of tracked beats.
        onset_times: Timestamps of spectral onset events.
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # 1. Spectral onset envelope
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    
    # 2. Onset peaks (physical strikes / foot taps)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        backtrack=True,
        units="frames"
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # 3. Tempo and beat tracking
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        start_bpm=start_bpm,
        units="frames"
    )
    tempo_bpm = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    return tempo_bpm, beat_times, onset_times


def generate_teental_grid(
    t0: float,
    tempo_bpm: float,
    total_duration: float,
) -> List[MatraInfo]:
    """
    Generates an idealized 16-Matra Teental theoretical grid from anchor time t0.
    """
    matra_duration = 60.0 / tempo_bpm
    grid: List[MatraInfo] = []

    current_time = t0
    global_beat_index = 0

    while current_time <= total_duration:
        cycle_num = (global_beat_index // 16) + 1
        matra_num = (global_beat_index % 16) + 1
        action = get_matra_action(matra_num)
        bol = TEENTAL_THEKA[matra_num - 1]

        grid.append(
            MatraInfo(
                cycle=cycle_num,
                matra=matra_num,
                timestamp=current_time,
                action=action,
                canonical_bol=bol,
            )
        )
        current_time += matra_duration
        global_beat_index += 1

    return grid


def quantize_onsets_to_grid(
    onsets: np.ndarray,
    grid: List[MatraInfo],
    max_tolerance_sec: float = 0.150,
) -> Dict:
    """
    Quantizes physical detected onsets against the theoretical Teental Matra grid.
    
    Calculates micro-timing deviations (jitter in milliseconds) and flags
    whether the strike is ON_BEAT, EARLY (rushing), or LATE (dragging).
    """
    grid_alignments = []
    matched_onsets = set()

    for m in grid:
        # Find nearest onset to this matra
        if len(onsets) > 0:
            diffs = onsets - m.timestamp
            closest_idx = int(np.argmin(np.abs(diffs)))
            min_diff = diffs[closest_idx]

            if abs(min_diff) <= max_tolerance_sec:
                matched_onsets.add(closest_idx)
                offset_ms = min_diff * 1000.0  # ms
                if abs(offset_ms) <= 35.0:
                    status = "ON_BEAT"
                elif offset_ms < -35.0:
                    status = "EARLY_RUSHING"
                else:
                    status = "LATE_DRAGGING"

                grid_alignments.append({
                    "matra": m.matra,
                    "cycle": m.cycle,
                    "action": m.action,
                    "bol": m.canonical_bol,
                    "target_time": m.timestamp,
                    "actual_time": float(onsets[closest_idx]),
                    "offset_ms": float(offset_ms),
                    "status": status,
                    "missed": False,
                })
                continue

        # Missed beat (no onset detected within tolerance window)
        grid_alignments.append({
            "matra": m.matra,
            "cycle": m.cycle,
            "action": m.action,
            "bol": m.canonical_bol,
            "target_time": m.timestamp,
            "actual_time": None,
            "offset_ms": None,
            "status": "MISSED",
            "missed": True,
        })

    # Find ghost/extra strikes
    extra_onsets = [
        float(onsets[i]) for i in range(len(onsets)) if i not in matched_onsets
    ]

    return {
        "alignments": grid_alignments,
        "extra_onsets": extra_onsets,
    }


def compute_laya_analytics(
    tempo_bpm: float,
    alignment_data: Dict,
) -> Dict:
    """
    Computes diagnostic Laya (tempo) and rhythmic precision metrics.
    """
    alignments = alignment_data["alignments"]
    valid_offsets = [
        a["offset_ms"] for a in alignments if a["offset_ms"] is not None
    ]

    # Categorize Laya speed
    if tempo_bpm < 75.0:
        laya_category = "Vilambit (Slow)"
    elif tempo_bpm <= 150.0:
        laya_category = "Madhya (Medium)"
    else:
        laya_category = "Drut (Fast)"

    if valid_offsets:
        mean_offset = float(np.mean(valid_offsets))
        abs_mean_error = float(np.mean(np.abs(valid_offsets)))
        timing_jitter_std = float(np.std(valid_offsets))
        precision_pct = max(0.0, 100.0 - (abs_mean_error * 1.2))
    else:
        mean_offset = 0.0
        abs_mean_error = 0.0
        timing_jitter_std = 0.0
        precision_pct = 0.0

    # Sam Precision (Specific offset on Beat 1)
    sam_alignments = [
        a for a in alignments if a["action"] == "SAM" and a["offset_ms"] is not None
    ]
    sam_offsets = [s["offset_ms"] for s in sam_alignments]
    mean_sam_offset = float(np.mean(sam_offsets)) if sam_offsets else None

    # Missed beats count
    missed_count = sum(1 for a in alignments if a["missed"])
    matched_count = len(valid_offsets)

    return {
        "tempo_bpm": tempo_bpm,
        "laya_category": laya_category,
        "matched_beats": matched_count,
        "missed_beats": missed_count,
        "extra_strikes": len(alignment_data["extra_onsets"]),
        "mean_offset_ms": mean_offset,
        "mean_absolute_error_ms": abs_mean_error,
        "timing_jitter_std_ms": timing_jitter_std,
        "sam_mean_offset_ms": mean_sam_offset,
        "rhythmic_precision_score": round(precision_pct, 1),
    }


def visualize_tala_grid(
    audio: np.ndarray,
    sr: int,
    onsets: np.ndarray,
    grid: List[MatraInfo],
    alignment_data: Dict,
    title: str = "Kathak Teental Alignment Analysis",
    save_path: Optional[Path] = None,
) -> Path:
    """
    Produces a dual-panel visualization:
      Panel 1: Waveform with overlay of physical onsets vs. theoretical Matra grid.
      Panel 2: Rhythmic timing error per Matra (Early vs. Late in ms).
    """
    times = np.arange(len(audio)) / sr
    alignments = alignment_data["alignments"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=False)

    # Panel 1: Waveform & Matra Grid
    ax1.plot(times, audio, color="#2b2d42", alpha=0.4, linewidth=0.5, label="Waveform")

    # Plot detected onsets
    if len(onsets) > 0:
        ax1.vlines(
            onsets,
            ymin=-0.9,
            ymax=0.9,
            color="#e63946",
            linestyle=":",
            alpha=0.6,
            linewidth=1.2,
            label="Detected Strikes (Onsets)",
        )

    # Plot Matras by action type
    for m in grid:
        if m.action == "SAM":
            ax1.axvline(m.timestamp, color="#2a9d8f", linewidth=2.2, linestyle="-")
            ax1.text(m.timestamp, 0.95, f"SAM (C{m.cycle})", color="#2a9d8f", rotation=90, fontsize=9, fontweight="bold")
        elif m.action in ("TAALI_2", "TAALI_3"):
            ax1.axvline(m.timestamp, color="#e76f51", linewidth=1.5, linestyle="--")
            ax1.text(m.timestamp, 0.75, f"M{m.matra}", color="#e76f51", rotation=90, fontsize=8)
        elif m.action == "KHALI":
            ax1.axvline(m.timestamp, color="#457b9d", linewidth=1.8, linestyle="-.")
            ax1.text(m.timestamp, 0.75, f"KHALI (M9)", color="#457b9d", rotation=90, fontsize=8, fontweight="bold")
        else:
            ax1.axvline(m.timestamp, color="#adb5bd", linewidth=0.8, linestyle=":")

    ax1.set_title(f"{title} - 16-Matra Teental Grid Overlay")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time (seconds)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.2)

    # Panel 2: Micro-timing Error (Early vs Late)
    matra_labels = []
    offsets = []
    colors = []

    for a in alignments:
        lbl = f"C{a['cycle']} M{a['matra']}\n({a['bol']})"
        matra_labels.append(lbl)
        if a["offset_ms"] is not None:
            off = a["offset_ms"]
            offsets.append(off)
            if a["status"] == "ON_BEAT":
                colors.append("#2a9d8f")  # Green
            elif a["status"] == "EARLY_RUSHING":
                colors.append("#f4a261")  # Orange
            else:
                colors.append("#e76f51")  # Red
        else:
            offsets.append(0.0)
            colors.append("#6c757d")  # Gray (missed)

    indices = np.arange(len(matra_labels))
    bars = ax2.bar(indices, offsets, color=colors, edgecolor="black", width=0.6)

    ax2.axhline(0, color="black", linewidth=1)
    ax2.axhline(35, color="gray", linestyle="--", alpha=0.5, label="Tolerance (±35ms)")
    ax2.axhline(-35, color="gray", linestyle="--", alpha=0.5)

    ax2.set_xticks(indices)
    ax2.set_xticklabels(matra_labels, fontsize=7, rotation=45)
    ax2.set_ylabel("Timing Offset (ms)\n[− Early / + Late]")
    ax2.set_title("Rhythmic Accuracy per Matra (Green: On Beat | Orange: Rushing | Red: Dragging)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = OUTPUTS_DIR / "tala_alignment.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def main():
    audio_file = PROJECT_DIR / "data" / "raw" / "sample.wav"
    if len(sys.argv) > 1:
        audio_file = Path(sys.argv[1])
        if not audio_file.is_absolute():
            audio_file = PROJECT_DIR / audio_file

    if not audio_file.is_file():
        print(f"Error: File not found at {audio_file}")
        sys.exit(1)

    print(f"Loading audio for Tala analysis: {audio_file.name}")
    audio, sr = librosa.load(audio_file, sr=None, mono=True)
    duration = len(audio) / sr

    tempo_bpm, beat_times, onset_times = estimate_tempo_and_beats(audio, sr)
    print(f"Estimated Tempo: {tempo_bpm:.1f} BPM")
    print(f"Detected Onsets: {len(onset_times)}")

    # Anchor to the first prominent beat / onset as Sam
    t0 = beat_times[0] if len(beat_times) > 0 else onset_times[0]
    print(f"Anchoring Sam (Beat 1) at: {t0:.3f}s")

    # Generate theoretical 16-Matra Teental grid
    grid = generate_teental_grid(t0=t0, tempo_bpm=tempo_bpm, total_duration=duration)
    print(f"Generated {len(grid)} theoretical Teental matras across {(len(grid) // 16) + 1} cycles")

    # Quantize onsets to grid
    alignment_data = quantize_onsets_to_grid(onset_times, grid)

    # Compute diagnostics
    analytics = compute_laya_analytics(tempo_bpm, alignment_data)

    print("\n================ TALA & LAYA ANALYSIS ================")
    print(f"File:                      {audio_file.name}")
    print(f"Tempo:                     {analytics['tempo_bpm']:.1f} BPM ({analytics['laya_category']})")
    print(f"Matched Matras:            {analytics['matched_beats']} / {len(grid)}")
    print(f"Missed Matras:             {analytics['missed_beats']}")
    print(f"Extra / Ghost Strikes:     {analytics['extra_strikes']}")
    print(f"Mean Abs Timing Error:     {analytics['mean_absolute_error_ms']:.1f} ms")
    print(f"Timing Jitter (Std Dev):   {analytics['timing_jitter_std_ms']:.1f} ms")
    if analytics['sam_mean_offset_ms'] is not None:
        print(f"Sam (Beat 1) Offset:       {analytics['sam_mean_offset_ms']:+.1f} ms")
    print(f"Rhythmic Precision Score:  {analytics['rhythmic_precision_score']} / 100")
    print("======================================================\n")

    out_plot = OUTPUTS_DIR / f"tala_{audio_file.stem}.png"
    visualize_tala_grid(audio, sr, onset_times, grid, alignment_data, title=f"Kathak Tala: {audio_file.name}", save_path=out_plot)
    print(f"Tala visualization saved to: {out_plot}")


if __name__ == "__main__":
    main()
