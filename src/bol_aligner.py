"""
Kathak Bol Recognition & Canonical Sequence Alignment Engine
=============================================================
Transcribes Kathak padhant (spoken dance syllables) and aligns detected
bol sequences to canonical classical compositions (Theka, Tatkar, Tukda, Tihai):
  - Domain-biased Faster-Whisper ASR with Devanagari Kathak prompts
  - Phonetic Bol Lexicon normalization (Devanagari & Romanized mapping)
  - Sequence alignment & Levenshtein-based similarity scoring
  - Automatic composition identification from classical repertoire
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Tuple
import unicodedata

from faster_whisper import WhisperModel
import librosa
import matplotlib.pyplot as plt
import numpy as np

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Classical Kathak Repertoire & Phonetic Normalization Lexicon
# ---------------------------------------------------------------------------

# Canonical bol mappings from common ASR transcription slips to standardized tokens
PHONETIC_BOL_MAP = {
    # Dha variations
    "धा": "Dha", "दा": "Dha", "डा": "Dha", "ध": "Dha", "दध": "Dha", "dha": "Dha", "da": "Dha",
    # Dhin variations
    "धिं": "Dhin", "दिं": "Dhin", "दिन": "Dhin", "धिन": "Dhin", "थिं": "Dhin", "dhin": "Dhin", "din": "Dhin",
    # Ta variations
    "ता": "Ta", "तां": "Ta", "त": "Ta", "टा": "Ta", "ta": "Ta", "taa": "Ta",
    # Tin variations
    "तिं": "Tin", "तिन": "Tin", "तीं": "Tin", "tin": "Tin",
    # Thei variations
    "थेई": "Thei", "तेई": "Thei", "थे": "Thei", "ते": "Thei", "thei": "Thei", "thay": "Thei", "tei": "Thei",
    # Tat variations
    "तत": "Tat", "तद": "Tat", "तट्": "Tat", "tat": "Tat", "tatt": "Tat",
    # Aa variations
    "आ": "Aa", "aa": "Aa", "a": "Aa",
    # Tigda variations
    "तिगदा": "Tigda", "तिगधा": "Tigda", "तिट": "Tigda", "tigda": "Tigda",
    # Digdig variations
    "दिगदिग": "Digdig", "दिग": "Digdig", "digdig": "Digdig",
    # Kram / Tran variations
    "क्राम": "Kram", "क्रम": "Kram", "क्रे": "Kram", "kram": "Kram",
    # Tete variations
    "तेटे": "Tete", "टेटे": "Tete", "तिति": "Tete", "tete": "Tete",
    # Kade variations
    "खेडे": "Kade", "खेटे": "Kade", "कड़े": "Kade", "kade": "Kade",
    # Ga variations
    "गाँ": "Ga", "गद": "Ga", "गा": "Ga", "ga": "Ga",
}

# Canonical Kathak Compositions
CANONICAL_COMPOSITIONS = {
    "Teental_Theka": [
        "Dha", "Dhin", "Dhin", "Dha",
        "Dha", "Dhin", "Dhin", "Dha",
        "Dha", "Tin",  "Tin",  "Ta",
        "Ta",  "Dhin", "Dhin", "Dha",
    ],
    "Tatkar_Basic": [
        "Ta", "Thei", "Thei", "Tat",
        "Aa", "Thei", "Thei", "Tat",
        "Ta", "Thei", "Thei", "Tat",
        "Dha", "Thei", "Thei", "Tat",
    ],
    "Tatkar_Double": [
        "Ta", "Thei", "Thei", "Tat",
        "Aa", "Thei", "Thei", "Tat",
    ],
    "Natwari_Tukda": [
        "Tigda", "Digdig", "Thei",
        "Tigda", "Digdig", "Thei",
        "Ta", "Thei", "Tat", "Thei", "Dha",
    ],
    "Tihai_Standard": [
        "Ta", "Thei", "Thei", "Tat", "Dha",
        "Ta", "Thei", "Thei", "Tat", "Dha",
        "Ta", "Thei", "Thei", "Tat", "Dha",
    ],
}

# Domain prompt in Hindi with classical syllables to steer Whisper's decoder
KATHAK_ASR_PROMPT = (
    "धा धिं धिं धा धा धिं धिं धा धा तिं तिं ता ता धिं धिं धा "
    "ता थेई थेई तत आ थेई थेई तत तिगदा दिगदिग थेई क्राम तेटे कड़े"
)


@dataclass
class DetectedBol:
    token: str
    original_text: str
    start: float
    end: float


def normalize_token(raw_word: str) -> Optional[str]:
    """Cleans punctuation and maps raw token to canonical bol lexicon without breaking Devanagari matras."""
    cleaned = "".join(c for c in raw_word if not unicodedata.category(c).startswith("P")).strip()
    if not cleaned:
        return None
    if cleaned in PHONETIC_BOL_MAP:
        return PHONETIC_BOL_MAP[cleaned]
    lower = cleaned.lower()
    if lower in PHONETIC_BOL_MAP:
        return PHONETIC_BOL_MAP[lower]
    return None


def transcribe_kathak_audio(
    audio_path: Path,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
) -> Tuple[List[DetectedBol], str]:
    """
    Transcribes Kathak padhant audio using faster-whisper with domain biasing.

    Returns:
        bol_events: List of DetectedBol with normalized tokens and timestamps.
        raw_transcript: Full raw text output from the model.
    """
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        str(audio_path),
        language="hi",
        initial_prompt=KATHAK_ASR_PROMPT,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        temperature=0.0,
    )

    bol_events: List[DetectedBol] = []
    raw_texts = []

    for seg in segments:
        raw_texts.append(seg.text)
        if seg.words:
            for w in seg.words:
                norm = normalize_token(w.word)
                if norm:
                    bol_events.append(
                        DetectedBol(
                            token=norm,
                            original_text=w.word,
                            start=w.start,
                            end=w.end,
                        )
                    )
        else:
            for word in seg.text.strip().split():
                norm = normalize_token(word)
                if norm:
                    bol_events.append(
                        DetectedBol(
                            token=norm,
                            original_text=word,
                            start=seg.start,
                            end=seg.end,
                        )
                    )

    raw_transcript = " ".join(raw_texts).strip()
    return bol_events, raw_transcript


def align_bol_sequences(
    detected_tokens: List[str],
    target_composition: List[str],
) -> Dict:
    """
    Aligns detected tokens with a target canonical composition using SequenceMatcher.
    """
    matcher = SequenceMatcher(None, detected_tokens, target_composition)
    ratio = matcher.ratio()

    diffs = []
    correct_count = 0
    substitutions = 0
    insertions = 0
    omissions = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                diffs.append({
                    "type": "CORRECT",
                    "detected": detected_tokens[i1 + k],
                    "target": target_composition[j1 + k],
                })
                correct_count += 1
        elif tag == "replace":
            det_slice = detected_tokens[i1:i2]
            tgt_slice = target_composition[j1:j2]
            max_len = max(len(det_slice), len(tgt_slice))
            for k in range(max_len):
                d_tok = det_slice[k] if k < len(det_slice) else None
                t_tok = tgt_slice[k] if k < len(tgt_slice) else None
                diffs.append({
                    "type": "SUBSTITUTION",
                    "detected": d_tok,
                    "target": t_tok,
                })
                substitutions += 1
        elif tag == "delete":
            for k in range(i1, i2):
                diffs.append({
                    "type": "INSERTION_EXTRA",
                    "detected": detected_tokens[k],
                    "target": None,
                })
                insertions += 1
        elif tag == "insert":
            for k in range(j1, j2):
                diffs.append({
                    "type": "OMISSION_MISSED",
                    "detected": None,
                    "target": target_composition[k],
                })
                omissions += 1

    accuracy_pct = round(ratio * 100.0, 1)

    return {
        "similarity_score": accuracy_pct,
        "correct_count": correct_count,
        "substitutions": substitutions,
        "insertions": insertions,
        "omissions": omissions,
        "detected_length": len(detected_tokens),
        "target_length": len(target_composition),
        "alignment_diff": diffs,
    }


def identify_closest_composition(
    detected_tokens: List[str],
) -> Tuple[str, Dict]:
    """
    Searches the classical Kathak library to find the closest matching composition.
    """
    best_name = "Unknown"
    best_alignment = {"similarity_score": 0.0}

    for name, comp_tokens in CANONICAL_COMPOSITIONS.items():
        alignment = align_bol_sequences(detected_tokens, comp_tokens)
        if alignment["similarity_score"] > best_alignment["similarity_score"]:
            best_alignment = alignment
            best_name = name

    return best_name, best_alignment


def visualize_bol_timeline(
    audio: np.ndarray,
    sr: int,
    bol_events: List[DetectedBol],
    target_composition_name: str,
    alignment_score: float,
    title: str = "Kathak Bol Recognition Timeline",
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plots the audio waveform with time-aligned recognized bol labels.
    """
    times = np.arange(len(audio)) / sr
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(times, audio, color="#2b2d42", alpha=0.35, linewidth=0.5, label="Waveform")

    for idx, b in enumerate(bol_events):
        mid_time = (b.start + b.end) / 2.0
        y_pos = 0.55 if idx % 2 == 0 else 0.78

        ax.axvspan(b.start, b.end, color="#457b9d", alpha=0.2)
        ax.text(
            mid_time,
            y_pos,
            b.token,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#e9c46a", edgecolor="#264653", alpha=0.9),
        )

    ax.set_title(f"{title} | Matched: {target_composition_name} ({alignment_score}% Accuracy)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.0, 1.1)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path is None:
        save_path = OUTPUTS_DIR / "bol_alignment.png"
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

    print(f"\n[BOL RECOGNIZER] Transcribing Kathak Padhant: {audio_file.name}")
    bol_events, raw_transcript = transcribe_kathak_audio(audio_file)
    detected_tokens = [b.token for b in bol_events]

    print(f"Raw ASR Text:       {raw_transcript}")
    print(f"Normalized Tokens:  {detected_tokens}")
    print(f"Detected Bols:      {len(detected_tokens)}")

    best_name, alignment = identify_closest_composition(detected_tokens)

    print("\n================ BOL ALIGNMENT REPORT ================")
    print(f"Identified Repertoire:   {best_name}")
    print(f"Similarity Score:        {alignment['similarity_score']}%")
    print(f"Correct Bols:            {alignment['correct_count']}")
    print(f"Substituted Bols:        {alignment['substitutions']}")
    print(f"Extra / Inserted Bols:   {alignment['insertions']}")
    print(f"Missed / Omitted Bols:   {alignment['omissions']}")
    print("------------------------------------------------------")
    print(f"{'Status':<18}{'Detected Token':<18}{'Target Token':<18}")
    print("-" * 54)
    for diff in alignment["alignment_diff"]:
        det = diff["detected"] or "-"
        tgt = diff["target"] or "-"
        print(f"{diff['type']:<18}{det:<18}{tgt:<18}")
    print("======================================================\n")

    audio, sr = librosa.load(audio_file, sr=None, mono=True)
    out_plot = OUTPUTS_DIR / f"bols_{audio_file.stem}.png"
    visualize_bol_timeline(
        audio,
        sr,
        bol_events,
        best_name,
        alignment["similarity_score"],
        title=f"Kathak Bol Recognition: {audio_file.name}",
        save_path=out_plot,
    )
    print(f"Bol timeline visualization saved to: {out_plot}")


if __name__ == "__main__":
    main()
