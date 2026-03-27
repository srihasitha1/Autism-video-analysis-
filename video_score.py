import numpy as np
import tensorflow as tf
import pickle

from video_loader import extract_clips
from tta import tta_predict
from config import CONFIG


def load_model_and_encoder():
    model = tf.keras.models.load_model("autism_mobilenet_gru_phase1.h5")
    with open(CONFIG["encoder_save_path"], "rb") as f:
        encoder = pickle.load(f)

    return model, encoder


def predict_video_score(video_path, model, encoder):
    """
    Takes full video → returns autism score + probabilities
    """

    # 🔹 Extract multiple clips
    clips = extract_clips(
        video_path,
        sequence_length=CONFIG["sequence_length"],
        img_height=CONFIG["img_height"],
        img_width=CONFIG["img_width"],
        clips_per_video=10,   # increase for better accuracy
        jitter=False
    )

    if not clips:
        return {"error": "Could not extract clips"}

    all_probs = []

    # 🔹 Predict each clip
    for clip in clips:
        probs = tta_predict(clip, model, n_augments=8)
        all_probs.append(probs)

    # 🔹 Average predictions across clips
    avg_probs = np.mean(all_probs, axis=0)

    # 🔹 Convert to dict
    labels = encoder.classes_
    prob_dict = dict(zip(labels, avg_probs))

    # 🔥 AUTISM SCORE (core logic)
    autism_score = (
        0.4 * prob_dict["arm_flapping"] +
        0.3 * prob_dict["spinning"] +
        0.3 * prob_dict["head_banging"]
    )

    return {
        "autism_score": float(round(autism_score, 4)),
        "probabilities": {k: float(round(v, 4)) for k, v in prob_dict.items()}
    }

def classify_risk(score):
    if score < 0.3:
        return "Low"
    elif score < 0.6:
        return "Moderate"
    else:
        return "High"
    
def analyze_video(video_path):
    model, encoder = load_model_and_encoder()

    result = predict_video_score(video_path, model, encoder)

    if "error" in result:
        return result

    score = result["autism_score"]
    risk = classify_risk(score)

    return {
        "risk_level": risk,
        "autism_score": score,
        "behavior_scores": result["probabilities"]
    }