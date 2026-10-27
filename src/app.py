"""
Kathak AI — FastAPI Backend (Day 5)
====================================
REST API exposing the full Kathak analysis pipeline:

  POST /analyze            — Upload audio, run all 4 stages, return JSON report
  GET  /audio/{filename}   — Stream raw audio files from data/raw/
  GET  /outputs/{filename} — Serve generated plot images from outputs/
  GET  /health             — Service liveness check
  GET  /                   — Serve the interactive web dashboard (index.html)
"""

import base64
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path
import sys

# Ensure project root is importable
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.evaluator import KathakEvaluator, visualize_evaluation_scorecard
from src.features import extract_features, visualize_features
from src.tala import (
    compute_laya_analytics,
    estimate_tempo_and_beats,
    generate_teental_grid,
    quantize_onsets_to_grid,
    visualize_tala_grid,
)
from src.bol_aligner import (
    identify_closest_composition,
    transcribe_kathak_audio,
    visualize_bol_timeline,
)
import librosa
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Kathak AI",
    description="AI-powered Tala rhythm tracking, bol recognition, and performance evaluation for Indian Classical Dance",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (frontend HTML/CSS/JS)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# Serve generated output images
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


# ── Helper: encode image to base64 ───────────────────────────────────────────
def img_to_b64(path: Path) -> str:
    if path.is_file():
        return base64.b64encode(path.read_bytes()).decode()
    return ""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Kathak AI", "version": "1.0.0"}


@app.get("/audio/{filename}")
def serve_audio(filename: str):
    """Stream a raw audio file from data/raw/."""
    path = RAW_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file '{filename}' not found.")
    return FileResponse(str(path), media_type="audio/wav")


@app.get("/library")
def list_library():
    """Returns all available audio recordings in data/raw/."""
    files = sorted(RAW_DIR.glob("*.wav")) + sorted(RAW_DIR.glob("*.mp3"))
    return {
        "recordings": [
            {
                "filename": f.name,
                "size_kb": round(f.stat().st_size / 1024, 1),
                "url": f"/audio/{f.name}",
            }
            for f in files
        ]
    }


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Accepts an uploaded WAV file and runs the full Kathak AI pipeline:
      Stage 1: Spectral & timbral feature extraction
      Stage 2: Tala (16-matra Teental) rhythm tracking
      Stage 3: Kathak bol recognition & canonical sequence alignment
      Stage 4: 4-Pillar composite evaluation & diagnostic critique

    Returns a rich JSON payload including base64-encoded plot images.
    """
    # Validate file type
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac")):
        raise HTTPException(status_code=400, detail="Only .wav, .mp3, or .flac files are accepted.")

    # Save uploaded file to a temp location
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(OUTPUTS_DIR)) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    stem = tmp_path.stem

    try:
        t0 = time.time()

        # ── Stage 1: Feature Extraction ───────────────────────────────────────
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        duration = len(audio) / sr
        features = extract_features(audio, sr)

        feat_plot = OUTPUTS_DIR / f"features_{stem}.png"
        visualize_features(features, title=f"Kathak AI — {file.filename}", save_path=feat_plot)

        # ── Stage 2: Tala & Rhythm ────────────────────────────────────────────
        tempo_bpm, beat_times, onset_times = estimate_tempo_and_beats(audio, sr)
        anchor = beat_times[0] if len(beat_times) > 0 else (onset_times[0] if len(onset_times) > 0 else 0.0)
        grid = generate_teental_grid(t0=anchor, tempo_bpm=tempo_bpm, total_duration=duration)
        alignment_data = quantize_onsets_to_grid(onset_times, grid)
        tala_metrics = compute_laya_analytics(tempo_bpm, alignment_data)

        tala_plot = OUTPUTS_DIR / f"tala_{stem}.png"
        visualize_tala_grid(
            audio, sr, onset_times, grid, alignment_data,
            title=f"Kathak Tala — {file.filename}", save_path=tala_plot,
        )

        # ── Stage 3: Bol Recognition ──────────────────────────────────────────
        bol_events, raw_transcript = transcribe_kathak_audio(tmp_path)
        tokens = [b.token for b in bol_events]
        best_comp, bol_alignment = identify_closest_composition(tokens)

        bol_plot = OUTPUTS_DIR / f"bols_{stem}.png"
        visualize_bol_timeline(
            audio, sr, bol_events, best_comp, bol_alignment["similarity_score"],
            title=f"Kathak Bols — {file.filename}", save_path=bol_plot,
        )

        # ── Stage 4: Evaluation ───────────────────────────────────────────────
        evaluator = KathakEvaluator()
        scorecard, _ = evaluator.evaluate_audio(tmp_path, target_composition_name=best_comp)

        report_plot = OUTPUTS_DIR / f"report_{stem}.png"
        visualize_evaluation_scorecard(scorecard, save_path=report_plot)

        elapsed = round(time.time() - t0, 1)

        # ── Build Response ─────────────────────────────────────────────────────
        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "processing_time_sec": elapsed,
            "audio": {
                "duration_sec": round(duration, 2),
                "sample_rate_hz": sr,
                "mean_centroid_hz": round(float(np.mean(features["spectral_centroid"])), 1),
                "mean_rms": round(float(np.mean(features["rms"])), 4),
            },
            "tala": {
                "tempo_bpm": round(tala_metrics["tempo_bpm"], 1),
                "laya_category": tala_metrics["laya_category"],
                "matched_matras": tala_metrics["matched_beats"],
                "total_matras": len(grid),
                "missed_matras": tala_metrics["missed_beats"],
                "mean_abs_error_ms": round(tala_metrics["mean_absolute_error_ms"], 1),
                "timing_jitter_ms": round(tala_metrics["timing_jitter_std_ms"], 1),
                "sam_offset_ms": round(tala_metrics["sam_mean_offset_ms"], 1) if tala_metrics["sam_mean_offset_ms"] else None,
            },
            "bols": {
                "detected_count": len(tokens),
                "tokens": tokens,
                "raw_transcript": raw_transcript,
                "matched_repertoire": best_comp,
                "similarity_score": bol_alignment["similarity_score"],
                "correct_bols": bol_alignment["correct_count"],
                "substitutions": bol_alignment["substitutions"],
                "omissions": bol_alignment["omissions"],
            },
            "scorecard": {
                "composite_score": scorecard.composite_score,
                "letter_grade": scorecard.letter_grade,
                "title_award": scorecard.title_award,
                "pillars": {
                    "laya_stability": scorecard.laya_stability_score,
                    "sam_precision": scorecard.sam_precision_score,
                    "matra_synchrony": scorecard.rhythmic_synchrony_score,
                    "bol_accuracy": scorecard.bol_accuracy_score,
                },
                "feedback": [
                    {"category": f.category, "severity": f.severity, "message": f.message}
                    for f in scorecard.feedback_items
                ],
            },
            "plots": {
                "features": img_to_b64(feat_plot),
                "tala": img_to_b64(tala_plot),
                "bols": img_to_b64(bol_plot),
                "report": img_to_b64(report_plot),
            },
        })

    finally:
        # Clean up temp file
        if tmp_path.is_file():
            tmp_path.unlink()


@app.get("/")
def serve_dashboard():
    """Serve the interactive web dashboard."""
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Frontend not found. Ensure src/static/index.html exists.")
    return FileResponse(str(index))
