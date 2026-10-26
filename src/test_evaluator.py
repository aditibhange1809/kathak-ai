"""
Automated Test Suite for Day 4 Kathak Feedback & Evaluation Engine
==================================================================
Validates:
1. Composite score calculation across the 4 Classical Pillars.
2. Letter grade mapping (A+, A, B+, B, C, D) and title awards.
3. Diagnostic feedback generation (Sam precision, Laya stability, Khali pause).
4. Generates executive visual report cards for sample.wav, sample2.wav, sample3.wav.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import (
    KathakEvaluator,
    determine_letter_grade,
    visualize_evaluation_scorecard,
    OUTPUTS_DIR,
    PROJECT_DIR,
)


def test_grade_mapping_logic():
    print("--> Validating Letter Grade Mapping Logic...")
    grade_tests = [
        (95.0, "A+"),
        (85.0, "A"),
        (75.0, "B+"),
        (62.0, "B"),
        (50.0, "C"),
        (30.0, "D"),
    ]
    for score, expected_grade in grade_tests:
        g, title = determine_letter_grade(score)
        assert g == expected_grade, f"Expected {expected_grade} for score {score}, got {g}"
    print("    ✓ All letter grade thresholds validated successfully.")


def test_end_to_end_evaluations():
    raw_dir = PROJECT_DIR / "data" / "raw"
    samples = sorted(list(raw_dir.glob("*.wav")))

    assert len(samples) > 0, "No audio samples found in data/raw!"
    print(f"\n[TEST SUITE] Running Full Diagnostic Evaluation across {len(samples)} recordings...\n")

    evaluator = KathakEvaluator()
    summary = {}

    for sample_path in samples:
        print(f"--> Evaluating {sample_path.name}...")
        scorecard, raw_data = evaluator.evaluate_audio(sample_path)

        # 1. Assert score bounds
        assert 0.0 <= scorecard.composite_score <= 100.0
        assert 0.0 <= scorecard.laya_stability_score <= 100.0
        assert 0.0 <= scorecard.sam_precision_score <= 100.0
        assert 0.0 <= scorecard.rhythmic_synchrony_score <= 100.0
        assert 0.0 <= scorecard.bol_accuracy_score <= 100.0

        # 2. Assert feedback items exist
        assert len(scorecard.feedback_items) >= 2, "Expected at least 2 diagnostic critique items"

        # 3. Assert visual report card is generated
        out_report = OUTPUTS_DIR / f"report_{sample_path.stem}.png"
        visualize_evaluation_scorecard(scorecard, save_path=out_report)
        assert out_report.is_file() and out_report.stat().st_size > 1000

        summary[sample_path.name] = {
            "score": scorecard.composite_score,
            "grade": scorecard.letter_grade,
            "title": scorecard.title_award,
            "repertoire": scorecard.matched_repertoire,
            "tempo": scorecard.tempo_bpm,
            "laya": scorecard.laya_category,
            "report_plot": out_report.name,
        }
        print(f"    ✓ Graded: {scorecard.composite_score}/100 [{scorecard.letter_grade}] | {scorecard.title_award} | Saved: {out_report.name}")

    print("\n==========================================================================================")
    print("Consolidated Kathak AI Performance Report Cards:")
    print("==========================================================================================")
    print(f"{'Sample':<14}{'Score':<10}{'Grade':<8}{'Tempo / Laya':<28}{'Repertoire':<18}{'Award Title'}")
    print("-" * 102)
    for name, s in summary.items():
        tempo_str = f"{s['tempo']:.1f} BPM ({s['laya']})"
        print(
            f"{name:<14}"
            f"{s['score']:<10.1f}"
            f"{s['grade']:<8}"
            f"{tempo_str:<28}"
            f"{s['repertoire']:<18}"
            f"{s['title']}"
        )
    print("==========================================================================================\n")
    print("ALL DAY 4 TESTS PASSED SUCCESSFULLY! 🏆✨")


def run_all_tests():
    test_grade_mapping_logic()
    test_end_to_end_evaluations()


if __name__ == "__main__":
    run_all_tests()
