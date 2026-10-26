"""
Master Kathak AI Pipeline Runner
=================================
Runs all three pipeline stages on any Kathak audio recording:
  1. Feature Extraction (Spectrogram, MFCCs, Centroid, ZCR)
  2. Tala & Rhythm Tracking (BPM, 16-Matra Teental Grid, Jitter, Sam Precision)
  3. Bol Recognition & Repertoire Alignment (Whisper ASR, Lexicon Mapping, Composition Matching)
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import librosa
from src.features import extract_features, visualize_features
from src.tala import (
    estimate_tempo_and_beats,
    generate_teental_grid,
    quantize_onsets_to_grid,
    compute_laya_analytics,
    visualize_tala_grid,
)
from src.bol_aligner import (
    transcribe_kathak_audio,
    identify_closest_composition,
    visualize_bol_timeline,
)
from src.evaluator import (
    KathakEvaluator,
    visualize_evaluation_scorecard,
)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def run_full_pipeline(audio_file_path: Path):
    if not audio_file_path.is_file():
        print(f"Error: Audio file not found at {audio_file_path}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print(f"        KATHAK AI: FULL PIPELINE ANALYSIS")
    print(f"        Processing: {audio_file_path.name}")
    print("=" * 65)

    audio, sr = librosa.load(audio_file_path, sr=None, mono=True)
    duration = len(audio) / sr

    # -------------------------------------------------------------
    # STAGE 1: Feature Extraction
    # -------------------------------------------------------------
    print("\n[STAGE 1/3] Extracting Spectral & Timbral Features...")
    features = extract_features(audio, sr)
    feat_plot = OUTPUTS_DIR / f"features_{audio_file_path.stem}.png"
    visualize_features(features, title=f"Kathak AI - {audio_file_path.name}", save_path=feat_plot)
    print(f"  ✓ Sample Rate:       {sr} Hz")
    print(f"  ✓ Duration:          {duration:.2f} seconds")
    print(f"  ✓ Mean Centroid:     {features['spectral_centroid'].mean():.1f} Hz")
    print(f"  ✓ Feature Plot:      {feat_plot.name}")

    # -------------------------------------------------------------
    # STAGE 2: Tala & Rhythmic Meter Alignment
    # -------------------------------------------------------------
    print("\n[STAGE 2/3] Tracking Rhythm, Tempo & 16-Matra Teental Grid...")
    tempo_bpm, beat_times, onset_times = estimate_tempo_and_beats(audio, sr)
    t0 = beat_times[0] if len(beat_times) > 0 else onset_times[0]
    grid = generate_teental_grid(t0=t0, tempo_bpm=tempo_bpm, total_duration=duration)
    alignment_data = quantize_onsets_to_grid(onset_times, grid)
    tala_metrics = compute_laya_analytics(tempo_bpm, alignment_data)

    tala_plot = OUTPUTS_DIR / f"tala_{audio_file_path.stem}.png"
    visualize_tala_grid(
        audio, sr, onset_times, grid, alignment_data,
        title=f"Kathak Tala - {audio_file_path.name}",
        save_path=tala_plot,
    )
    print(f"  ✓ Estimated Tempo:   {tala_metrics['tempo_bpm']:.1f} BPM ({tala_metrics['laya_category']})")
    print(f"  ✓ Matched Matras:    {tala_metrics['matched_beats']} / {len(grid)}")
    print(f"  ✓ Mean Abs Error:    {tala_metrics['mean_absolute_error_ms']:.1f} ms")
    print(f"  ✓ Timing Jitter:     {tala_metrics['timing_jitter_std_ms']:.1f} ms")
    if tala_metrics['sam_mean_offset_ms'] is not None:
        print(f"  ✓ Sam Offset:        {tala_metrics['sam_mean_offset_ms']:+.1f} ms")
    print(f"  ✓ Rhythmic Score:    {tala_metrics['rhythmic_precision_score']}%")
    print(f"  ✓ Tala Plot:         {tala_plot.name}")

    # -------------------------------------------------------------
    # STAGE 3: Bol Recognition & Composition Matching
    # -------------------------------------------------------------
    print("\n[STAGE 3/3] Transcribing Kathak Bols & Matching Repertoire...")
    bol_events, raw_transcript = transcribe_kathak_audio(audio_file_path)
    tokens = [b.token for b in bol_events]
    best_comp, bol_alignment = identify_closest_composition(tokens)

    bol_plot = OUTPUTS_DIR / f"bols_{audio_file_path.stem}.png"
    visualize_bol_timeline(
        audio, sr, bol_events, best_comp, bol_alignment["similarity_score"],
        title=f"Kathak Bols - {audio_file_path.name}",
        save_path=bol_plot,
    )
    print(f"  ✓ Detected Bols:     {len(tokens)}")
    print(f"  ✓ Matched Work:      {best_comp}")
    print(f"  ✓ Bol Accuracy:      {bol_alignment['similarity_score']}%")
    print(f"  ✓ Correct Bols:      {bol_alignment['correct_count']} | Subs: {bol_alignment['substitutions']} | Omissions: {bol_alignment['omissions']}")
    print(f"  ✓ Bol Plot:          {bol_plot.name}")

    # -------------------------------------------------------------
    # STAGE 4: Pedagogical Diagnostic Evaluation
    # -------------------------------------------------------------
    evaluator = KathakEvaluator()
    scorecard, _ = evaluator.evaluate_audio(audio_file_path, target_composition_name=best_comp)
    report_plot = OUTPUTS_DIR / f"report_{audio_file_path.stem}.png"
    visualize_evaluation_scorecard(scorecard, save_path=report_plot)

    # -------------------------------------------------------------
    # Consolidated Report Card
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"             FINAL KATHAK EVALUATION REPORT")
    print("=" * 65)
    print(f"Recording:             {audio_file_path.name}")
    print(f"Matched Repertoire:    {scorecard.matched_repertoire}")
    print(f"Tempo / Laya:          {scorecard.tempo_bpm:.1f} BPM ({scorecard.laya_category})")
    print("-" * 65)
    print("FOUR PILLARS OF PERFORMANCE:")
    print(f"  1. Laya Sthirata (Tempo Stability):     {scorecard.laya_stability_score:.1f}%")
    print(f"  2. Sam Precision (Beat 1 Accuracy):     {scorecard.sam_precision_score:.1f}%")
    print(f"  3. Matra Synchrony (Rhythmic Jitter):   {scorecard.rhythmic_synchrony_score:.1f}%")
    print(f"  4. Bol Shuddhata (Syllable Accuracy):   {scorecard.bol_accuracy_score:.1f}%")
    print("-" * 65)
    print(f"COMPOSITE SCORE:       {scorecard.composite_score} / 100")
    print(f"PROFICIENCY GRADE:     {scorecard.letter_grade} ({scorecard.title_award})")
    print("-" * 65)
    print("EXECUTIVE GURU CRITIQUES:")
    for f in scorecard.feedback_items:
        print(f"  [{f.severity}] {f.category}: {f.message}")
    print("-" * 65)
    print(f"Visual artifacts generated in outputs/:")
    print(f"  1. Spectrogram & MFCCs:      outputs/{feat_plot.name}")
    print(f"  2. Teental Grid Alignment:   outputs/{tala_plot.name}")
    print(f"  3. Bol Recognition Timeline: outputs/{bol_plot.name}")
    print(f"  4. Executive Scorecard:      outputs/{report_plot.name}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    target = PROJECT_ROOT / "data" / "raw" / "sample.wav"
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if not target.is_absolute():
            target = PROJECT_ROOT / target
    run_full_pipeline(target)
