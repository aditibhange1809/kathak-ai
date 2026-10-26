"""
Automated Test Suite for Day 3 Kathak Bol Recognition & Alignment
=================================================================
Validates:
1. Phonetic bol lexicon normalization (Devanagari & Romanized variations).
2. Levenshtein / SequenceMatcher alignment scoring & diff categorization.
3. Automatic identification of canonical compositions (Teental Theka, Tatkar).
4. Generates bol recognition timelines across sample recordings.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import librosa
from src.bol_aligner import (
    normalize_token,
    align_bol_sequences,
    identify_closest_composition,
    transcribe_kathak_audio,
    visualize_bol_timeline,
    CANONICAL_COMPOSITIONS,
    OUTPUTS_DIR,
    PROJECT_DIR,
)


def test_lexicon_normalization():
    print("--> Validating Phonetic Bol Normalization...")
    test_cases = {
        "धा": "Dha",
        "दा": "Dha",
        "डा": "Dha",
        "धिं": "Dhin",
        "दिन": "Dhin",
        "ता": "Ta",
        "थेई": "Thei",
        "तेई": "Thei",
        "तत": "Tat",
        "तिगदा": "Tigda",
        "दिगदिग": "Digdig",
        "क्राम": "Kram",
        "तेटे": "Tete",
        "dha": "Dha",
        "thei": "Thei",
        "tat": "Tat",
    }
    for raw, expected in test_cases.items():
        res = normalize_token(raw)
        assert res == expected, f"Expected {expected} for '{raw}', got {res}"
    print("    ✓ Lexicon normalization passed for all phonetic variations.")


def test_sequence_alignment_logic():
    print("--> Validating Sequence Alignment Engine...")
    target = ["Dha", "Dhin", "Dhin", "Dha"]
    
    # 1. Exact Match
    res_exact = align_bol_sequences(["Dha", "Dhin", "Dhin", "Dha"], target)
    assert res_exact["similarity_score"] == 100.0
    assert res_exact["correct_count"] == 4

    # 2. Single Substitution
    res_sub = align_bol_sequences(["Dha", "Ta", "Dhin", "Dha"], target)
    assert res_sub["similarity_score"] < 100.0
    assert res_sub["substitutions"] == 1

    # 3. Insertion & Omission
    res_drift = align_bol_sequences(["Dha", "Dhin", "Thei", "Dhin"], target)
    assert res_drift["similarity_score"] > 0.0

    print("    ✓ Alignment math and diff categorization passed.")


def test_end_to_end_recognition():
    raw_dir = PROJECT_DIR / "data" / "raw"
    samples = sorted(list(raw_dir.glob("*.wav")))

    assert len(samples) > 0, "No audio samples found in data/raw!"
    print(f"\n[TEST SUITE] Transcribing and Aligning {len(samples)} Audio Recordings...\n")

    summary_results = {}

    for sample_path in samples:
        print(f"--> Processing {sample_path.name} with Kathak Bol Recognizer...")
        bol_events, raw_text = transcribe_kathak_audio(sample_path)
        tokens = [b.token for b in bol_events]

        assert len(tokens) > 0, f"No tokens detected in {sample_path.name}"

        # Match to canonical library
        best_comp, alignment = identify_closest_composition(tokens)
        assert alignment["similarity_score"] > 0.0

        # Render timeline plot
        audio, sr = librosa.load(sample_path, sr=None, mono=True)
        out_plot = OUTPUTS_DIR / f"bols_{sample_path.stem}.png"
        visualize_bol_timeline(
            audio,
            sr,
            bol_events,
            best_comp,
            alignment["similarity_score"],
            title=f"Kathak Bol Recognition - {sample_path.name}",
            save_path=out_plot,
        )
        assert out_plot.is_file() and out_plot.stat().st_size > 1000

        summary_results[sample_path.name] = {
            "num_bols": len(tokens),
            "matched_repertoire": best_comp,
            "accuracy": alignment["similarity_score"],
            "raw_snippet": (raw_text[:40] + "...") if len(raw_text) > 40 else raw_text,
            "plot": out_plot.name,
        }
        print(f"    ✓ Matched: {best_comp} ({alignment['similarity_score']}%) | {len(tokens)} bols | Saved: {out_plot.name}")

    print("\n=========================================================================")
    print("Kathak Bol Recognition & Repertoire Matching Summary:")
    print("=========================================================================")
    print(f"{'Sample':<14}{'Bols':<8}{'Matched Composition':<22}{'Accuracy':<12}{'Raw Transcript'}")
    print("-" * 84)
    for name, r in summary_results.items():
        print(
            f"{name:<14}"
            f"{r['num_bols']:<8}"
            f"{r['matched_repertoire']:<22}"
            f"{r['accuracy']:<12.1f}%"
            f"{r['raw_snippet']}"
        )
    print("=========================================================================\n")
    print("ALL DAY 3 TESTS PASSED SUCCESSFULLY! 🗣️✨")


def run_all_tests():
    test_lexicon_normalization()
    test_sequence_alignment_logic()
    test_end_to_end_recognition()


if __name__ == "__main__":
    run_all_tests()
