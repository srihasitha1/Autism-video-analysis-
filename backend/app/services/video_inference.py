"""
app/services/video_inference.py
===============================
Thin adapter between the AutiSense backend and the existing ML pipeline.

Design decisions:
  - Adds the project-root model/ directory to sys.path so existing imports
    (video_loader, tta, config) resolve without modifying the ML code.
  - Loads the TF model + label encoder ONCE on first call (singleton).
  - Calls predict_video_score() from video_score.py directly, bypassing
    analyze_video() which has a hardcoded model path.
  - Maps the raw ML output to the schema expected by our DB.
"""

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any


from app.config import settings

logger = logging.getLogger("autisense.video_inference")

# ── Module-level singleton cache ────────────────────────────────
_model = None
_encoder = None
_pipeline_ready = False


def _resolve_model_paths() -> tuple[str, str]:
    """Resolve model and encoder paths from settings (relative to cwd)."""
    model_path = str(Path(settings.MODEL_VIDEO_PATH).resolve())
    encoder_path = str(Path(settings.MODEL_ENCODER_PATH).resolve())
    return model_path, encoder_path


def _ensure_model_dir_on_path():
    """Add the project-root model/ directory to sys.path for ML imports."""
    # The model/ directory is a sibling of backend/ in the project root
    project_root = Path(__file__).resolve().parents[3]  # backend/app/services -> project root
    model_dir = project_root / "model"

    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
        logger.info("Added model directory to sys.path: %s", model_dir)

    # Also add the project root for video_score.py imports
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _load_model_and_encoder():
    """Load TF model and label encoder from configured paths. Cached as singleton."""
    global _model, _encoder, _pipeline_ready

    if _pipeline_ready:
        return _model, _encoder

    model_path, encoder_path = _resolve_model_paths()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Video model not found: {model_path}")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder not found: {encoder_path}")

    logger.info("Loading video model from: %s", Path(model_path).name)

    # Try tf_keras first (for models saved with older tf_keras builds).
    # Fall back to standard Keras with compile=False for Keras 3.x compatibility.
    try:
        import tf_keras
        _model = tf_keras.models.load_model(model_path)
        logger.info("Model loaded via tf_keras compatibility layer")
    except ImportError:
        import tensorflow as tf
        _model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("Model loaded via standard Keras (compile=False)")

    logger.info("Loading label encoder from: %s", Path(encoder_path).name)
    with open(encoder_path, "rb") as f:
        _encoder = pickle.load(f)

    _pipeline_ready = True
    logger.info("Video inference pipeline ready (model + encoder loaded)")
    return _model, _encoder


def _classify_risk(score: float) -> str:
    """Classify autism score into risk levels matching video_score.py logic."""
    if score < 0.3:
        return "Low"
    elif score < 0.6:
        return "Moderate"
    else:
        return "High"


def _score_to_confidence(score: float) -> str:
    """
    Map autism score to a confidence qualifier.

    Score near 0 or 1 → high confidence (clear signal)
    Score near 0.5 → low confidence (ambiguous)
    """
    distance_from_center = abs(score - 0.5)
    if distance_from_center > 0.3:
        return "high"
    elif distance_from_center > 0.15:
        return "medium"
    else:
        return "low"


def run_inference(video_path: str) -> dict[str, Any]:
    """
    Run the full video inference pipeline on a single video file.

    This is the main entry point called by the Celery task.

    Args:
        video_path: Absolute path to the uploaded video file.

    Returns:
        dict with keys:
          - video_class_probabilities: {class_name: float}
          - video_score: float (0–1, composite autism score)
          - video_confidence: str ("high"/"medium"/"low")
          - risk_level: str ("Low"/"Moderate"/"High")
          - clips_evaluated: int

    Raises:
        FileNotFoundError: If model files are missing.
        RuntimeError: If frame extraction fails.
    """
    import numpy as np

    _ensure_model_dir_on_path()
    model, encoder = _load_model_and_encoder()

    # Import from the existing ML pipeline (now on sys.path)
    from video_loader import extract_sliding_window_clips
    from tta import tta_predict

    # Use the same config as video_score.py
    from config import CONFIG as ML_CONFIG

    logger.info("Starting inference on video: %s", Path(video_path).name)

    # ── Extract clips via sliding window ────────────────────────
    clips = extract_sliding_window_clips(
        video_path,
        sequence_length=ML_CONFIG["sequence_length"],
        img_height=ML_CONFIG["img_height"],
        img_width=ML_CONFIG["img_width"],
        overlap=0.5,
    )

    if not clips:
        raise RuntimeError(
            "Could not extract any clips from video. "
            "The video may be too short, corrupted, or in an unsupported format."
        )

    # Limit clips for reasonable processing time.
    # A 13-second video produces 99+ clips with sliding window — cap at 10.
    MAX_CLIPS = 10
    if len(clips) > MAX_CLIPS:
        import numpy as np
        original_count = len(clips)
        indices = np.linspace(0, len(clips) - 1, MAX_CLIPS, dtype=int)
        clips = [clips[i] for i in indices]
        logger.warning(
            "Clip count limited from %d to %d for performance (evenly sampled)",
            original_count, MAX_CLIPS,
        )

    logger.info("Running TTA inference on %d clips...", len(clips))

    # ── Predict each clip with TTA ──────────────────────────────
    # n_augments reduced from 8 to 2 for speed (800 → 20 forward passes)
    all_probs = []
    for i, clip in enumerate(clips):
        probs = tta_predict(clip, model, n_augments=2)
        all_probs.append(probs)
        logger.debug("Clip %d/%d done", i + 1, len(clips))

    # ── Average predictions across clips ────────────────────────
    avg_probs = np.mean(all_probs, axis=0)

    # ── Build probability dict ──────────────────────────────────
    labels = encoder.classes_
    prob_dict = {label: float(round(p, 4)) for label, p in zip(labels, avg_probs)}

    # ── Compute autism score (same formula as video_score.py) ───
    autism_score = (
        0.4 * prob_dict.get("arm_flapping", 0.0)
        + 0.3 * prob_dict.get("spinning", 0.0)
        + 0.4 * prob_dict.get("head_banging", 0.0)
    )
    autism_score = float(round(autism_score, 4))

    risk_level = _classify_risk(autism_score)
    confidence = _score_to_confidence(autism_score)

    logger.info(
        "Inference complete: score=%.4f risk=%s confidence=%s clips=%d",
        autism_score, risk_level, confidence, len(clips),
    )

    return {
        "video_class_probabilities": prob_dict,
        "video_score": autism_score,
        "video_confidence": confidence,
        "risk_level": risk_level,
        "clips_evaluated": len(clips),
    }
