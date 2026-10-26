"""
Kathak Performance Evaluation & Automated Feedback Engine
==========================================================
Synthesizes Tala rhythm tracking, micro-timing quantization, and bol
recognition into an automated pedagogical dance coach.

Calculates composite scores across 4 classical pillars:
  1. Laya Sthirata (Tempo Stability & Drift)
  2. Sam Precision (Landing accuracy on Beat 1)
  3. Matra Synchrony (Micro-timing jitter across 16 beats)
  4. Bol Shuddhata (Pronunciation & Syllable sequence accuracy)

Generates actionable diagnostic critique reports and visual scorecards.
"""

from dataclasses import dataclass, field
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

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.features import extract_features
from src.tala import (
    estimate_tempo_and_beats,
    generate_teental_grid,
    quantize_onsets_to_grid,
    compute_laya_analytics,
)
from src.bol_aligner import (
    transcribe_kathak_audio,
    identify_closest_composition,
    align_bol_sequences,
    CANONICAL_COMPOSITIONS,
)


@dataclass
class DiagnosticFeedback:
    category: str  # 'SAM', 'LAYA', 'RHYTHM', 'BOLS'
    severity: str  # 'EXCELLENT', 'INFO', 'WARNING', 'CRITICAL'
    message: str
    timestamp: Optional[float] = None


@dataclass
class PerformanceScorecard:
    filename: str
    duration: float
    tempo_bpm: float
    laya_category: str
    matched_repertoire: str
    # Pillar Scores (0 to 100)
    laya_stability_score: float
    sam_precision_score: float
    rhythmic_synchrony_score: float
    bol_accuracy_score: float
    composite_score: float
    letter_grade: str
    title_award: str
    feedback_items: List[DiagnosticFeedback] = field(default_factory=list)


def determine_letter_grade(score: float) -> Tuple[str, str]:
    """Maps composite percentage score to classical dance proficiency rank."""
    if score >= 90.0:
        return "A+", "Pandit / Vidushi (Mastery Grade)"
    elif score >= 80.0:
        return "A", "Visharad (Advanced Grade)"
    elif score >= 70.0:
        return "B+", "Madhyama Purna (Intermediate-High Grade)"
    elif score >= 58.0:
        return "B", "Madhyama (Intermediate Grade)"
    elif score >= 45.0:
        return "C", "Praveshika (Foundational Grade)"
    else:
        return "D", "Prathama (Novice Grade - Needs Guidance)"


class KathakEvaluator:
    """
    Automated Guru / Pedagogical Evaluator for Kathak Performances.
    """

    def __init__(self, tolerance_ms: float = 35.0):
        self.tolerance_ms = tolerance_ms

    def evaluate_audio(
        self,
        audio_path: Path,
        target_composition_name: Optional[str] = None,
    ) -> Tuple[PerformanceScorecard, Dict]:
        """
        Runs comprehensive analysis and grades performance across the 4 pillars.
        """
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found at: {audio_path}")

        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = len(audio) / sr

        # -------------------------------------------------------------
        # 1. Tala & Rhythm Tracking
        # -------------------------------------------------------------
        tempo_bpm, beat_times, onset_times = estimate_tempo_and_beats(audio, sr)
        t0 = beat_times[0] if len(beat_times) > 0 else (onset_times[0] if len(onset_times) > 0 else 0.0)
        grid = generate_teental_grid(t0=t0, tempo_bpm=tempo_bpm, total_duration=duration)
        alignment_data = quantize_onsets_to_grid(onset_times, grid)
        tala_metrics = compute_laya_analytics(tempo_bpm, alignment_data)

        # -------------------------------------------------------------
        # 2. Bol Transcription & Alignment
        # -------------------------------------------------------------
        bol_events, raw_transcript = transcribe_kathak_audio(audio_path)
        detected_tokens = [b.token for b in bol_events]

        if target_composition_name and target_composition_name in CANONICAL_COMPOSITIONS:
            best_repertoire = target_composition_name
            bol_alignment = align_bol_sequences(detected_tokens, CANONICAL_COMPOSITIONS[best_repertoire])
        else:
            best_repertoire, bol_alignment = identify_closest_composition(detected_tokens)

        # -------------------------------------------------------------
        # 3. Pillar 1: Laya Sthirata (Tempo Stability Score)
        # -------------------------------------------------------------
        # Measure variance in inter-beat intervals (IBI)
        if len(beat_times) >= 3:
            ibis = np.diff(beat_times)
            mean_ibi = np.mean(ibis)
            ibi_std = np.std(ibis)
            # Coefficient of Variation (CV = std / mean)
            cv_pct = (ibi_std / mean_ibi) * 100.0 if mean_ibi > 0 else 50.0
            laya_stability_score = max(0.0, min(100.0, 100.0 - (cv_pct * 3.0)))
        else:
            laya_stability_score = 50.0

        # -------------------------------------------------------------
        # 4. Pillar 2: Sam Precision (Beat 1 Mastery)
        # -------------------------------------------------------------
        sam_alignments = [
            a for a in alignment_data["alignments"]
            if a["action"] == "SAM" and a["offset_ms"] is not None
        ]
        if sam_alignments:
            mean_sam_error = float(np.mean([abs(s["offset_ms"]) for s in sam_alignments]))
            # 0ms error = 100%, 70ms error = 0%
            sam_precision_score = max(0.0, min(100.0, 100.0 - (mean_sam_error * 1.4)))
        else:
            sam_precision_score = 20.0  # Penalized for missing Sam entirely
            mean_sam_error = None

        # -------------------------------------------------------------
        # 5. Pillar 3: Matra Synchrony (Micro-timing Jitter)
        # -------------------------------------------------------------
        mean_abs_error = tala_metrics["mean_absolute_error_ms"]
        # 0ms error = 100%, 80ms error = 0%
        rhythmic_synchrony_score = max(0.0, min(100.0, 100.0 - (mean_abs_error * 1.25)))

        # -------------------------------------------------------------
        # 6. Pillar 4: Bol Shuddhata (Syllable Accuracy)
        # -------------------------------------------------------------
        bol_accuracy_score = float(bol_alignment["similarity_score"])

        # -------------------------------------------------------------
        # 7. Composite Weighted Score
        # -------------------------------------------------------------
        # Weights: Rhythmic Synchrony (35%), Laya Stability (25%), Sam Precision (20%), Bol Accuracy (20%)
        composite_score = round(
            (0.35 * rhythmic_synchrony_score) +
            (0.25 * laya_stability_score) +
            (0.20 * sam_precision_score) +
            (0.20 * bol_accuracy_score),
            1
        )
        letter_grade, title_award = determine_letter_grade(composite_score)

        # -------------------------------------------------------------
        # 8. Pedagogical Diagnostic Critiques
        # -------------------------------------------------------------
        feedback: List[DiagnosticFeedback] = []

        # Sam Feedback
        if sam_alignments:
            first_sam_offset = sam_alignments[0]["offset_ms"]
            if abs(first_sam_offset) <= 20.0:
                feedback.append(DiagnosticFeedback(
                    category="SAM",
                    severity="EXCELLENT",
                    message=f"Outstanding Sam landing! Initial Beat 1 landed within {first_sam_offset:+.1f}ms.",
                    timestamp=sam_alignments[0]["target_time"],
                ))
            elif first_sam_offset < -20.0:
                feedback.append(DiagnosticFeedback(
                    category="SAM",
                    severity="WARNING",
                    message=f"Rushed the Sam by {abs(first_sam_offset):.1f}ms. Wait for the full cycle before landing.",
                    timestamp=sam_alignments[0]["target_time"],
                ))
            else:
                feedback.append(DiagnosticFeedback(
                    category="SAM",
                    severity="WARNING",
                    message=f"Delayed Sam strike (+{first_sam_offset:.1f}ms late). Focus on crisp foot impact on Beat 1.",
                    timestamp=sam_alignments[0]["target_time"],
                ))
        else:
            feedback.append(DiagnosticFeedback(
                category="SAM",
                severity="CRITICAL",
                message="Sam (Beat 1) was completely missed or indistinct in this cycle.",
            ))

        # Laya Feedback
        if laya_stability_score >= 80.0:
            feedback.append(DiagnosticFeedback(
                category="LAYA",
                severity="EXCELLENT",
                message=f"Very stable {tala_metrics['laya_category']} pulse ({tempo_bpm:.1f} BPM). No noticeable tempo drift.",
            ))
        elif laya_stability_score >= 60.0:
            feedback.append(DiagnosticFeedback(
                category="LAYA",
                severity="INFO",
                message=f"Moderate Laya consistency ({tempo_bpm:.1f} BPM). Minor micro-timing fluctuations observed.",
            ))
        else:
            feedback.append(DiagnosticFeedback(
                category="LAYA",
                severity="CRITICAL",
                message="High tempo drift detected. The internal clock is speeding up or dragging across Vibhags.",
            ))

        # Khali (Beat 9) Check
        khali_alignments = [
            a for a in alignment_data["alignments"]
            if a["action"] == "KHALI" and a["offset_ms"] is not None
        ]
        for k in khali_alignments:
            if k["status"] == "EARLY_RUSHING":
                feedback.append(DiagnosticFeedback(
                    category="RHYTHM",
                    severity="WARNING",
                    message=f"Rushed Khali (Beat 9) in Cycle {k['cycle']} by {abs(k['offset_ms']):.1f}ms. Maintain open pause.",
                    timestamp=k["target_time"],
                ))

        # Bol Feedback
        if bol_accuracy_score >= 70.0:
            feedback.append(DiagnosticFeedback(
                category="BOLS",
                severity="EXCELLENT",
                message=f"Strong recitation fidelity to {best_repertoire} ({bol_alignment['correct_count']} clean bols).",
            ))
        else:
            feedback.append(DiagnosticFeedback(
                category="BOLS",
                severity="INFO",
                message=f"Matched repertoire: {best_repertoire}. Detected {bol_alignment['substitutions']} substitutions and {bol_alignment['omissions']} omitted bols.",
            ))

        scorecard = PerformanceScorecard(
            filename=audio_path.name,
            duration=duration,
            tempo_bpm=tempo_bpm,
            laya_category=tala_metrics["laya_category"],
            matched_repertoire=best_repertoire,
            laya_stability_score=round(laya_stability_score, 1),
            sam_precision_score=round(sam_precision_score, 1),
            rhythmic_synchrony_score=round(rhythmic_synchrony_score, 1),
            bol_accuracy_score=round(bol_accuracy_score, 1),
            composite_score=composite_score,
            letter_grade=letter_grade,
            title_award=title_award,
            feedback_items=feedback,
        )

        raw_data = {
            "tala_metrics": tala_metrics,
            "bol_alignment": bol_alignment,
            "raw_transcript": raw_transcript,
            "grid": grid,
            "onset_times": onset_times,
            "beat_times": beat_times,
        }

        return scorecard, raw_data


def visualize_evaluation_scorecard(
    scorecard: PerformanceScorecard,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Renders an executive visual report card with pillar bar chart,
    composite grade badge, and diagnostic feedback list.
    """
    fig = plt.figure(figsize=(14, 8), facecolor="#f8f9fa")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1.1])

    # 1. Pillar Bar Chart (Top Left)
    ax_bars = fig.add_subplot(gs[0, 0])
    pillars = [
        "Laya Stability\n(Tempo)",
        "Sam Precision\n(Beat 1)",
        "Matra Synchrony\n(Rhythm)",
        "Bol Shuddhata\n(Pronunciation)",
    ]
    scores = [
        scorecard.laya_stability_score,
        scorecard.sam_precision_score,
        scorecard.rhythmic_synchrony_score,
        scorecard.bol_accuracy_score,
    ]
    bar_colors = ["#2a9d8f", "#e76f51", "#457b9d", "#e9c46a"]

    bars = ax_bars.barh(pillars, scores, color=bar_colors, edgecolor="#264653", height=0.55)
    ax_bars.set_xlim(0, 105)
    ax_bars.set_xlabel("Pillar Score (0 - 100)")
    ax_bars.set_title("Performance Across 4 Pillars of Kathak", fontweight="bold", fontsize=11)
    ax_bars.grid(True, axis="x", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, scores):
        ax_bars.text(val + 2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontweight="bold", fontsize=9)

    # 2. Executive Badge & Details (Top Right)
    ax_badge = fig.add_subplot(gs[0, 1])
    ax_badge.axis("off")

    badge_box = dict(boxstyle="round,pad=1.0", facecolor="#2b2d42", edgecolor="#d90429", linewidth=2.5)
    badge_text = (
        f"KATHAK AI PERFORMANCE GRADE\n\n"
        f"OVERALL SCORE: {scorecard.composite_score} / 100\n"
        f"GRADE: {scorecard.letter_grade}\n\n"
        f"{scorecard.title_award}\n"
        f"Repertoire: {scorecard.matched_repertoire}\n"
        f"Tempo: {scorecard.tempo_bpm:.1f} BPM ({scorecard.laya_category})"
    )
    ax_badge.text(0.5, 0.5, badge_text, ha="center", va="center", fontsize=11, color="white", fontweight="bold", bbox=badge_box)

    # 3. Diagnostic Critiques List (Bottom Span)
    ax_feed = fig.add_subplot(gs[1, :])
    ax_feed.axis("off")

    feedback_lines = ["\n  EXECUTIVE GURU CRITIQUE & ACTIONABLE DIAGNOSTICS:"]
    for idx, item in enumerate(scorecard.feedback_items, 1):
        icon = "✓" if item.severity == "EXCELLENT" else ("ℹ" if item.severity == "INFO" else "⚠")
        t_str = f" [at {item.timestamp:.2f}s]" if item.timestamp else ""
        feedback_lines.append(f"  {idx}. [{icon} {item.category}] {item.message}{t_str}")

    feedback_text = "\n".join(feedback_lines)
    feed_box = dict(boxstyle="round,pad=0.8", facecolor="#edf2f4", edgecolor="#8d99ae", linewidth=1.5)
    ax_feed.text(0.02, 0.9, feedback_text, ha="left", va="top", fontsize=10, family="sans-serif", bbox=feed_box)

    plt.suptitle(f"Kathak Dance Evaluation Report: {scorecard.filename}", fontsize=14, fontweight="bold", color="#1d3557")
    plt.tight_layout()

    if save_path is None:
        save_path = OUTPUTS_DIR / f"report_{Path(scorecard.filename).stem}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def main():
    audio_file = PROJECT_DIR / "data" / "raw" / "sample.wav"
    if len(sys.argv) > 1:
        audio_file = Path(sys.argv[1])
        if not audio_file.is_absolute():
            audio_file = PROJECT_DIR / audio_file

    evaluator = KathakEvaluator()
    print(f"\n[EVALUATOR] Running Complete Diagnostic Evaluation on: {audio_file.name}...")
    scorecard, raw_data = evaluator.evaluate_audio(audio_file)

    print("\n" + "=" * 65)
    print(f"             KATHAK PERFORMANCE REPORT CARD")
    print("=" * 65)
    print(f"Recording:                 {scorecard.filename}")
    print(f"Duration:                  {scorecard.duration:.2f} seconds")
    print(f"Matched Composition:       {scorecard.matched_repertoire}")
    print(f"Tempo & Laya:              {scorecard.tempo_bpm:.1f} BPM ({scorecard.laya_category})")
    print("-" * 65)
    print("FOUR PILLARS OF PERFORMANCE:")
    print(f"  1. Laya Sthirata (Tempo Stability):     {scorecard.laya_stability_score:.1f}%")
    print(f"  2. Sam Precision (Beat 1 Accuracy):     {scorecard.sam_precision_score:.1f}%")
    print(f"  3. Matra Synchrony (Rhythmic Jitter):   {scorecard.rhythmic_synchrony_score:.1f}%")
    print(f"  4. Bol Shuddhata (Syllable Accuracy):   {scorecard.bol_accuracy_score:.1f}%")
    print("-" * 65)
    print(f"COMPOSITE SCORE:           {scorecard.composite_score} / 100")
    print(f"PROFICIENCY GRADE:         {scorecard.letter_grade} ({scorecard.title_award})")
    print("=" * 65)
    print("\nPEDAGOGICAL ACTIONABLE CRITIQUES:")
    for idx, f in enumerate(scorecard.feedback_items, 1):
        print(f"  [{f.severity}] {f.category}: {f.message}")
    print("=" * 65 + "\n")

    out_report = OUTPUTS_DIR / f"report_{audio_file.stem}.png"
    visualize_evaluation_scorecard(scorecard, save_path=out_report)
    print(f"Executive Report Card image saved to: {out_report}")


if __name__ == "__main__":
    main()
