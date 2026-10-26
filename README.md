# Kathak AI 🩰🥁

> **An AI-powered audio analysis, Tala rhythm tracking, and bol recognition engine for Indian Classical Dance (Kathak).**

Kathak AI bridges digital signal processing (DSP), Music Information Retrieval (MIR), and transformer-based speech recognition (ASR) to analyze spoken dance recitations (*padhant*), footwork strikes (*tatkar*), and rhythmic cycles (*Tala*).

---

## 🌟 Key Capabilities

1. **Spectral & Timbral Feature Engineering:**
   - 128-band Mel-Spectrograms (dB) capturing human pitch perception.
   - 13 Mel-Frequency Cepstral Coefficients (MFCCs) + $\Delta$ velocity envelopes encoding the phonetic signature of classical dance syllables (*Dha, Dhin, Thei, Tat*).
   - Spectral Centroid & Rolloff tracking high-frequency ankle-bell (*ghungroo*) activity ($>4000\text{ Hz}$).
   - Zero-Crossing Rate (ZCR) and Short-Time RMS energy dynamics.

2. **Tala & Rhythmic Meter Alignment (Teental Engine):**
   - Autocorrelation & spectral novelty envelope tempo (BPM) estimation.
   - Canonical 16-Matra **Teental** metrical grid generation (Beats 1–16).
   - Classical structural beat tagging: **Sam** (Beat 1), **Taali 2** (Beat 5), **Khali** (Beat 9), and **Taali 3** (Beat 13).
   - Micro-timing quantization calculating signed deviations ($\delta = o_j - m_k$) in milliseconds (*On-Beat*, *Rushing*, or *Dragging*).

3. **Domain-Biased Kathak Bol Recognition:**
   - CTranslate2-accelerated `faster-whisper` conditioned on Devanagari Kathak prompts.
   - Unicode-safe Indic phonetic normalizer mapping ASR slips down to standardized tokens (`"धा" / "दा" / "डा"` $\rightarrow$ `"Dha"`).
   - Classical repertoire dictionary (*Teental Theka*, *Tatkar*, *Natwari Tukda*, *Tihai*).
   - Dynamic sequence alignment with opcode diffing (Correct matches, Substitutions, Extra insertions, Omissions).

4. **Automated Dance Guru & Feedback Engine:**
   - Composite performance scoring across **The 4 Classical Pillars**:
     - *Laya Sthirata* (Tempo Stability & IBI variance)
     - *Sam Precision* (Beat 1 landing accuracy)
     - *Matra Synchrony* (Timing jitter across beats)
     - *Bol Shuddhata* (Phonetic recitation fidelity)
   - Classical rank assignment (*Pandit/Vidushi*, *Visharad*, *Madhyama*, *Praveshika*, *Prathama*).
   - Actionable pedagogical critique generator diagnosing micro-timing errors in plain language.

---

## 🏗️ Architecture

```
                      Kathak Audio Recording (.wav)
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
   [ Stage 1: DSP ]       [ Stage 2: Tala ]         [ Stage 3: ASR ]
  Mel-Spectrogram & MFCC    16-Matra Teental Grid    Whisper Bol Recognizer
  Spectral Centroid & ZCR   Micro-timing Jitter       Phonetic Normalization
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   ▼
                    [ Stage 4: Guru Evaluator ]
                     Composite 4-Pillar Scoring
                     Actionable Pedagogical Critique
                     Visual Scorecard & Diagnostics
```

---

## 📂 Project Structure

```
kathak-ai/
├── data/
│   └── raw/                    # Audio recordings (WAV/MP3)
├── outputs/                    # Generated visual plots & reports
├── src/
│   ├── __init__.py             # Python package marker
│   ├── features.py             # Spectral feature extraction (Mel, MFCC, Centroid)
│   ├── tala.py                 # 16-Matra Teental grid & beat quantization
│   ├── bol_aligner.py          # Faster-Whisper ASR & canonical sequence aligner
│   ├── evaluator.py            # 4-Pillar composite grader & critique generator
│   ├── run_pipeline.py         # Unified master runner (Stages 1 to 4)
│   ├── inspect_audio.py        # Raw waveform & metadata inspector
│   ├── segment_audio.py        # RMS energy activity segmentation
│   ├── detect_onsets.py        # Spectral onset detection
│   ├── test_features.py        # Day 1 test suite
│   ├── test_tala.py            # Day 2 test suite
│   ├── test_bol_aligner.py     # Day 3 test suite
│   └── test_evaluator.py       # Day 4 test suite
├── .gitignore                  # Git exclusions (venv, pycache, OS files)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quickstart

### 1. Prerequisites & Installation

Clone the repository and set up a Python 3.10+ virtual environment:

```bash
git clone https://github.com/<your-username>/kathak-ai.git
cd kathak-ai

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

Analyze any audio recording through all four stages with a single command:

```bash
python src/run_pipeline.py data/raw/sample.wav
```

### 3. Run Specific Modules

```bash
# Extract and visualize spectral features (Day 1)
python src/features.py data/raw/sample.wav

# Track Tala rhythm and Teental Matra grid (Day 2)
python src/tala.py data/raw/sample.wav

# Transcribe and align Kathak bols (Day 3)
python src/bol_aligner.py data/raw/sample3.wav

# Generate performance report card & diagnostic critique (Day 4)
python src/evaluator.py data/raw/sample.wav
```

### 4. Run Test Suites

```bash
python src/test_features.py
python src/test_tala.py
python src/test_bol_aligner.py
python src/test_evaluator.py
```

---

## 📊 Visual Diagnostics

The pipeline outputs high-resolution diagnostic plots to `outputs/`:
- `features_<name>.png`: 4-panel spectral dashboard (Mel-Spectrogram, MFCCs, Centroid, ZCR).
- `tala_<name>.png`: Dual-panel waveform grid with signed early/late millisecond deviation bars.
- `bols_<name>.png`: Time-aligned bol timeline overlaid on the audio wave.
- `report_<name>.png`: Executive performance scorecard with 4-pillar bar chart, grade badge, and diagnostic critique.

---

## 📜 License

MIT License. Developed for Indian Classical Dance AI research and pedagogy.
