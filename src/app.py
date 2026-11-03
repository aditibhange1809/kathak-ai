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
from typing import Optional, Dict, List

# Ensure project root is importable
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.evaluator import KathakEvaluator, visualize_evaluation_scorecard
from src.features import extract_features, visualize_features
from src.tala import (
    compute_laya_analytics,
    estimate_tempo_and_beats,
    generate_tala_grid,
    generate_teental_grid,
    quantize_onsets_to_grid,
    visualize_tala_grid,
    TAAL_DEFINITIONS,
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
async def analyze_audio(
    file: UploadFile = File(...),
    taal: Optional[str] = Form(None),
):
    """
    Accepts an uploaded audio file (WAV/MP3) and optional Taal override ('Auto', 'Teental', 'Jhaptaal', etc.).
    Runs the full 4-stage classical Kathak AI pipeline:
      Stage 1: Spectral & timbral feature extraction
      Stage 2: Automatic Taal detection or user-specified Tala grid tracking
      Stage 3: Kathak bol recognition & canonical repertoire sequence alignment
      Stage 4: 4-Pillar composite evaluation & classical Shastra diagnostic critique

    Returns an exhaustive JSON payload with Tala metadata and base64 plots.
    """
    # Validate file type
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
        raise HTTPException(status_code=400, detail="Only audio files (.wav, .mp3, .flac, .ogg, .m4a) are accepted.")

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

        # ── Stage 2, 3 & 4: Multi-Tala Evaluation Pipeline ──────────────────
        evaluator = KathakEvaluator()
        # Parse taal override (treat "Auto", empty string or None as automatic detection)
        selected_taal = taal if (taal and taal != "Auto" and taal.strip() != "") else None
        scorecard, eval_raw = evaluator.evaluate_audio(tmp_path, target_taal=selected_taal)

        active_taal = eval_raw["taal_name"]
        taal_spec = eval_raw["taal_spec"]
        tala_metrics = eval_raw["tala_metrics"]
        grid = eval_raw["grid"]
        onset_times = eval_raw["onset_times"]
        alignment_data = eval_raw["alignment_data"]
        bol_alignment = eval_raw["bol_alignment"]
        bol_events = eval_raw["bol_events"]
        raw_transcript = eval_raw["raw_transcript"]
        tokens = [b.token for b in bol_events]

        # ── Visualizations ───────────────────────────────────────────────────
        tala_plot = OUTPUTS_DIR / f"tala_{stem}.png"
        visualize_tala_grid(
            audio, sr, onset_times, grid, alignment_data,
            title=f"Kathak {taal_spec['display_name']} — {file.filename}",
            taal_name=active_taal,
            save_path=tala_plot,
        )

        bol_plot = OUTPUTS_DIR / f"bols_{stem}.png"
        visualize_bol_timeline(
            audio, sr, bol_events, scorecard.matched_repertoire, bol_alignment["similarity_score"],
            title=f"Kathak Bols ({taal_spec['name']}) — {file.filename}", save_path=bol_plot,
        )

        report_plot = OUTPUTS_DIR / f"report_{stem}.png"
        visualize_evaluation_scorecard(scorecard, save_path=report_plot)

        # Save a copy for web audio playback
        playback_suffix = Path(file.filename).suffix or ".wav"
        playback_filename = f"playback_{stem}{playback_suffix}"
        playback_path = OUTPUTS_DIR / playback_filename
        shutil.copyfile(tmp_path, playback_path)

        elapsed = round(time.time() - t0, 1)
        cycle_matras = taal_spec["matras"]

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
                "playback_url": f"/outputs/{playback_filename}",
            },
            "tala": {
                "name": active_taal,
                "display_name": taal_spec["display_name"],
                "total_matras": cycle_matras,
                "vibhags": taal_spec["vibhags"],
                "vibhag_names": taal_spec["vibhag_names"],
                "structure": taal_spec["structure"],
                "theka": taal_spec["theka"],
                "theka_devanagari": taal_spec["theka_devanagari"],
                "actions": taal_spec["actions"],
                "signs": taal_spec["signs"],
                "vibhag_signs": taal_spec["vibhag_signs"],
                "vibhag_cls": taal_spec["vibhag_cls"],
                "vibhag_kriyas": taal_spec["vibhag_kriyas"],
                "gestures": taal_spec["gestures"],
                "confidence": scorecard.taal_confidence,
                "anchor_t0": round(float(eval_raw.get("anchor_t0", 0.0)), 3),
                "tempo_bpm": round(tala_metrics["tempo_bpm"], 1),
                "laya_category": tala_metrics["laya_category"],
                "matched_matras": tala_metrics["matched_beats"],
                "total_grid_matras": len(grid),
                "total_cycles": max([m.cycle for m in grid]) if grid else 1,
                "missed_matras": tala_metrics["missed_beats"],
                "mean_abs_error_ms": round(tala_metrics["mean_absolute_error_ms"], 1),
                "timing_jitter_ms": round(tala_metrics["timing_jitter_std_ms"], 1),
                "sam_offset_ms": round(tala_metrics["sam_mean_offset_ms"], 1) if tala_metrics["sam_mean_offset_ms"] else None,
                "grid_alignments": [
                    {
                        "matra": a["matra"],
                        "cycle": a["cycle"],
                        "global_matra": (a["cycle"] - 1) * cycle_matras + a["matra"],
                        "action": a["action"],
                        "bol": a["bol"],
                        "target_time": round(float(a["target_time"]), 3),
                        "actual_time": round(float(a["actual_time"]), 3) if a["actual_time"] is not None else None,
                        "offset_ms": round(float(a["offset_ms"]), 1) if a["offset_ms"] is not None else None,
                        "status": a["status"],
                        "missed": a["missed"],
                    }
                    for a in alignment_data["alignments"]
                ],
                "expected_grid": [
                    {
                        "matra": m.matra,
                        "cycle": m.cycle,
                        "global_matra": (m.cycle - 1) * cycle_matras + m.matra,
                        "action": m.action,
                        "bol": m.canonical_bol,
                        "target_time": round(float(m.timestamp), 3),
                    }
                    for m in grid
                ],
            },
            "bols": {
                "detected_count": len(tokens),
                "tokens": tokens,
                "raw_transcript": raw_transcript,
                "matched_repertoire": scorecard.matched_repertoire,
                "similarity_score": bol_alignment["similarity_score"],
                "correct_bols": bol_alignment["correct_count"],
                "substitutions": bol_alignment["substitutions"],
                "omissions": bol_alignment["omissions"],
                "bol_events": [
                    {
                        "token": b.token,
                        "original_text": b.original_text,
                        "start": round(float(b.start), 3),
                        "end": round(float(b.end), 3),
                    }
                    for b in bol_events
                ],
            },
            "scorecard": {
                "composite_score": scorecard.composite_score,
                "letter_grade": scorecard.letter_grade,
                "title_award": scorecard.title_award,
                "taal_name": scorecard.taal_name,
                "taal_display": scorecard.taal_display,
                "taal_structure": scorecard.taal_structure,
                "taal_confidence": scorecard.taal_confidence,
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


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Kathak AI Server at http://localhost:8000\n")
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)

