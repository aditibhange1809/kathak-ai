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

# Use the tiny model with int8 compute type to stay safely under Render's 512MB RAM limit
model_size = "tiny"
model = WhisperModel(model_size, device="cpu", compute_type="int8")
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
    # Dhi variations (Jhaptaal, Dadra, Kehrava)
    "धि": "Dhi", "धी": "Dhi", "दी": "Dhi", "दि": "Dhi", "dhi": "Dhi", "dhee": "Dhi",
    # Na variations (Jhaptaal, Dadra, Kehrava)
    "ना": "Na", "न": "Na", "नां": "Na", "na": "Na", "naa": "Na",
    # Ta variations
    "ता": "Ta", "तां": "Ta", "त": "Ta", "टा": "Ta", "ta": "Ta", "taa": "Ta",
    # Tin variations
    "तिं": "Tin", "तिन": "Tin", "तीं": "Tin", "tin": "Tin",
    # Ti variations (Jhaptaal, Kehrava, Dadra)
    "ती": "Ti", "ति": "Ti", "ti": "Ti", "tee": "Ti",
    # Tu variations (Dadra hallmark)
    "तू": "Tu", "तु": "Tu", "तुं": "Tu", "tu": "Tu", "too": "Tu",
    # Ge variations (Kehrava, Dadra)
    "गे": "Ge", "ग": "Ge", "गेह": "Ge", "ge": "Ge", "gay": "Ge",
    # Ka variations (Kehrava hallmark)
    "क": "Ka", "का": "Ka", "ka": "Ka", "kaa": "Ka",
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

# Canonical Kathak Compositions & Tala Thekas
CANONICAL_COMPOSITIONS = {
    # ── Teental (16 Matras) ──
    "Teental_Theka": [
        "Dha", "Dhin", "Dhin", "Dha",
        "Dha", "Dhin", "Dhin", "Dha",
        "Dha", "Tin",  "Tin",  "Ta",
        "Ta",  "Dhin", "Dhin", "Dha",
    ],
    # ── Jhaptaal (10 Matras) ──
    "Jhaptaal_Theka": [
        "Dhi", "Na",
        "Dhi", "Dhi", "Na",
        "Ti",  "Na",
        "Dhi", "Dhi", "Na",
    ],
    "Jhaptaal_Theka_Var": [
        "Dha", "Dhi", "Na",
        "Dha", "Ti",  "Na",
        "Dhi", "Dhi", "Na",
    ],
    "Jhaptaal_Theka_DhaStart": [
        "Dha", "Na",
        "Dhi", "Dhi", "Na",
        "Ti",  "Na",
        "Dhi", "Dhi", "Na",
    ],
    # ── Dadra (6 Matras) ──
    "Dadra_Theka": [
        "Dha", "Dhi", "Na",
        "Dha", "Tu",  "Na",
    ],
    "Dadra_Theka_Var": [
        "Dha", "Ge",  "Na",
        "Dha", "Ti",  "Na",
    ],
    # ── Kehrava (8 Matras) ──
    "Kehrava_Theka": [
        "Dha", "Ge", "Na", "Ti",
        "Na",  "Ka", "Dhi", "Na",
    ],
    # ── Classical Repertoire ──
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

# Domain prompt in Hindi with classical syllables across all 4 Taals to steer Whisper
KATHAK_ASR_PROMPT = (
    "धा धिं धिं धा धा धिं धिं धा धा तिं तिं ता ता धिं धिं धा "
    "धि ना धि धि ना ती ना धि धि ना "
    "धा धि ना धा तू ना धा गे ना धा ती ना "
    "धा गे ना ती ना क धि ना "
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


def identify_taal(
    detected_tokens: List[str],
) -> Tuple[str, float]:
    """
    Identifies the Indian Classical Tala (Teental, Jhaptaal, Dadra, Kehrava)
    from detected Kathak recitation syllables using sequence alignment and
    distinctive phonetic signatures.

    Returns:
        taal_name: "Teental" | "Jhaptaal" | "Dadra" | "Kehrava"
        confidence: Percentage match (50.0 to 99.0)
    """
    if not detected_tokens:
        return "Teental", 50.0

    scores: Dict[str, float] = {
        "Teental": 0.0,
        "Jhaptaal": 0.0,
        "Dadra": 0.0,
        "Kehrava": 0.0,
    }

    token_lower = [t.lower() for t in detected_tokens]
    bigrams = [f"{token_lower[i]} {token_lower[i+1]}" for i in range(len(token_lower) - 1)]

    # 1. Distinctive signature bols weighting
    # "Tu" is an exclusive hallmark of Dadra theka (Dha Dhi Na Dha Tu Na)
    if "tu" in token_lower:
        scores["Dadra"] += 55.0

    # "Ka" is a hallmark of Kehrava theka (Na Ka Dhi Na)
    if "ka" in token_lower:
        scores["Kehrava"] += 55.0

    # "Ge" appears in Kehrava ("Dha Ge Na Ti") and Dadra variant ("Dha Ge Na")
    if "ge" in token_lower:
        scores["Kehrava"] += 30.0
        scores["Dadra"] += 15.0

    # Jhaptaal signature bigrams: "ti na", "dhi na", "na dhi"
    has_ti_na = any("ti na" in bg or "tin na" in bg for bg in bigrams)
    has_dhi_na = any("dhi na" in bg or "dhin na" in bg for bg in bigrams)
    if has_ti_na:
        scores["Jhaptaal"] += 60.0
    if has_dhi_na and "ka" not in token_lower and "tu" not in token_lower:
        scores["Jhaptaal"] += 35.0

    # "Ti" / "Na" / "Dhi" combination in high density strongly indicates Jhaptaal
    dhi_count = token_lower.count("dhi") + token_lower.count("dhin")
    na_count = token_lower.count("na")
    ti_count = token_lower.count("ti") + token_lower.count("tin")

    if (dhi_count >= 1 or "dha" in token_lower) and na_count >= 1 and (ti_count >= 1 or has_ti_na):
        if "tu" not in token_lower and "ka" not in token_lower:
            scores["Jhaptaal"] += 45.0

    # If Na appears multiple times, it is not Teental Theka (Teental theka has zero Na)
    if na_count >= 2:
        scores["Teental"] -= 40.0

    # Teental hallmarks: classical dance tokens (Thei, Tat, Aa, Tigda, etc.)
    teental_indicators = ["thei", "tat", "aa", "tigda", "digdig", "kram", "tete", "kade"]
    if any(k in token_lower for k in teental_indicators):
        scores["Teental"] += 65.0

    # 2. Sequence alignment against the standard thekas
    theka_targets = {
        "Teental": CANONICAL_COMPOSITIONS["Teental_Theka"],
        "Jhaptaal": CANONICAL_COMPOSITIONS["Jhaptaal_Theka"],
        "Dadra": CANONICAL_COMPOSITIONS["Dadra_Theka"],
        "Kehrava": CANONICAL_COMPOSITIONS["Kehrava_Theka"],
    }

    for taal_name, base_theka in theka_targets.items():
        rep = max(1, (len(detected_tokens) + len(base_theka) - 1) // len(base_theka))
        extended = (base_theka * rep)[:max(len(detected_tokens), len(base_theka))]

        alignment = align_bol_sequences(detected_tokens, extended)
        sim = alignment["similarity_score"]
        scores[taal_name] += sim

        # Also check variants
        if taal_name == "Jhaptaal":
            var_ext = (CANONICAL_COMPOSITIONS["Jhaptaal_Theka_Var"] * rep)[:max(len(detected_tokens), 10)]
            var_align = align_bol_sequences(detected_tokens, var_ext)
            if var_align["similarity_score"] > sim:
                scores["Jhaptaal"] = max(scores["Jhaptaal"], var_align["similarity_score"] + 20.0)

        elif taal_name == "Dadra":
            var_ext = (CANONICAL_COMPOSITIONS["Dadra_Theka_Var"] * rep)[:max(len(detected_tokens), 6)]
            var_align = align_bol_sequences(detected_tokens, var_ext)
            if var_align["similarity_score"] > sim:
                scores["Dadra"] = max(scores["Dadra"], var_align["similarity_score"] + (15.0 if "ge" in token_lower else 0.0))

    best_taal = max(scores, key=scores.get)
    max_val = scores[best_taal]
    confidence = min(98.5, max(52.0, max_val * 0.82))

    return best_taal, round(confidence, 1)


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
