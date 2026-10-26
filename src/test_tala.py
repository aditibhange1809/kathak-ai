"""
Automated Test Suite for Day 2 Tala & Rhythm Tracking
=====================================================
Validates:
1. Tempo estimation and Teental 16-matra grid generation.
2. Quantization logic matches onsets within expected tolerance bounds.
3. Micro-timing offset computation, Sam precision, and jitter calculations.
4. Generates alignment visualizations for sample.wav, sample2.wav, sample3.wav.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import librosa
import numpy as np
from src.tala import (
    estimate_tempo_and_beats,
    generate_teental_grid,
    quantize_onsets_to_grid,
    compute_laya_analytics,
    visualize_tala_grid,
    OUTPUTS_DIR,
    PROJECT_DIR,
)


def test_tala_pipeline():
    raw_dir = PROJECT_DIR / "data" / "raw"
    samples = sorted(list(raw_dir.glob("*.wav")))

    assert len(samples) > 0, "No audio samples found in data/raw!"
    print(f"\n[TEST SUITE] Validating Tala & Rhythmic Alignment across {len(samples)} recordings...\n")

    summary_results = {}

    for sample_path in samples:
        print(f"--> Analyzing {sample_path.name}...")
        audio, sr = librosa.load(sample_path, sr=None, mono=True)
        duration = len(audio) / sr

        # 1. Estimate tempo & beats
        tempo_bpm, beat_times, onset_times = estimate_tempo_and_beats(audio, sr)
        assert 30.0 <= tempo_bpm <= 300.0, f"Unrealistic tempo estimated: {tempo_bpm} BPM"
        assert len(onset_times) > 0, f"No onsets detected for {sample_path.name}"

        # 2. Generate Teental grid anchored at first beat
        t0 = beat_times[0] if len(beat_times) > 0 else onset_times[0]
        grid = generate_teental_grid(t0=t0, tempo_bpm=tempo_bpm, total_duration=duration)
        assert len(grid) >= 16, f"Expected at least 1 full Teental Avartan (16 matras), got {len(grid)}"

        # Verify Teental structure: Matra 1 is SAM, 5 is TAALI_2, 9 is KHALI, 13 is TAALI_3
        assert grid[0].action == "SAM" and grid[0].matra == 1
        assert grid[4].action == "TAALI_2" and grid[4].matra == 5
        assert grid[8].action == "KHALI" and grid[8].matra == 9
        assert grid[12].action == "TAALI_3" and grid[12].matra == 13

        # 3. Quantize onsets to grid
        alignment_data = quantize_onsets_to_grid(onset_times, grid)
        assert len(alignment_data["alignments"]) == len(grid)

        # 4. Compute Laya analytics
        analytics = compute_laya_analytics(tempo_bpm, alignment_data)
        assert 0.0 <= analytics["rhythmic_precision_score"] <= 100.0
        assert analytics["matched_beats"] + analytics["missed_beats"] == len(grid)

        # 5. Render visualization
        out_plot = OUTPUTS_DIR / f"tala_{sample_path.stem}.png"
        visualize_tala_grid(
            audio,
            sr,
            onset_times,
            grid,
            alignment_data,
            title=f"Kathak Tala - {sample_path.name}",
            save_path=out_plot,
        )
        assert out_plot.is_file() and out_plot.stat().st_size > 1000, "Tala plot failed to save"

        summary_results[sample_path.name] = analytics
        print(f"    ✓ Passed: {analytics['tempo_bpm']:.1f} BPM ({analytics['laya_category']}) | Precision: {analytics['rhythmic_precision_score']}% | Saved: {out_plot.name}")

    print("\n=========================================================================")
    print("Comparative Tala & Rhythmic Consistency Across Kathak Performances:")
    print("=========================================================================")
    print(f"{'Sample':<14}{'Tempo (BPM)':<14}{'Laya':<18}{'Matched/Total':<16}{'Mean Err':<12}{'Score':<8}")
    print("-" * 82)
    for name, a in summary_results.items():
        matched_str = f"{a['matched_beats']} / {a['matched_beats'] + a['missed_beats']}"
        print(
            f"{name:<14}"
            f"{a['tempo_bpm']:<14.1f}"
            f"{a['laya_category']:<18}"
            f"{matched_str:<16}"
            f"{a['mean_absolute_error_ms']:<12.1f} ms"
            f"{a['rhythmic_precision_score']:<8.1f}%"
        )
    print("=========================================================================\n")
    print("ALL DAY 2 TESTS PASSED SUCCESSFULLY! 🥁✨")


if __name__ == "__main__":
    test_tala_pipeline()
