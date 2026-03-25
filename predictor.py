"""
predictor.py
============
Single-video inference using the trained MobileNetV2 + GRU model.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from video_loader import extract_single_clip


def predict_video(
    video_path:    str,
    model:         tf.keras.Model,
    label_encoder: LabelEncoder,
    config:        dict,
) -> dict:
    """
    Run inference on a single video file.

    Args:
        video_path    : Path to the video for inference.
        model         : Trained Keras model.
        label_encoder : Fitted LabelEncoder (int ↔ class name).
        config        : Global CONFIG dict.

    Returns:
        dict:
            "predicted_class"    — string
            "confidence"         — float  (0–1)
            "all_probabilities"  — {class_name: float}
            "error"              — string (only if extraction fails)
    """
    print(f"\n  Predicting: {video_path}")

    clip = extract_single_clip(
        video_path,
        sequence_length=config["sequence_length"],
        img_height=config["img_height"],
        img_width=config["img_width"],
    )

    if clip is None:
        return {"error": f"Frame extraction failed: {video_path}"}

    # (T, H, W, 3) → (1, T, H, W, 3)
    tensor = np.expand_dims(clip, axis=0).astype(np.float32)
    probs  = model.predict(tensor, verbose=0)[0]

    pred_idx   = int(np.argmax(probs))
    pred_class = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    all_probs = {
        label_encoder.inverse_transform([i])[0]: round(float(p), 4)
        for i, p in enumerate(probs)
    }

    print(f"  ┌─ Predicted : {pred_class}")
    print(f"  ├─ Confidence: {confidence:.2%}")
    print(f"  └─ All probabilities:")
    for cls_name, p in sorted(all_probs.items(), key=lambda kv: -kv[1]):
        bar = "█" * int(p * 30)
        print(f"       {cls_name:<15} {p:.4f}  {bar}")

    return {
        "predicted_class":   pred_class,
        "confidence":        round(confidence, 4),
        "all_probabilities": all_probs,
    }


def load_and_predict(
    video_path:    str,
    model_path:    str,
    encoder_path:  str,
    config:        dict,
) -> dict:
    """Load model + encoder from disk and predict a video."""
    import pickle
    model         = tf.keras.models.load_model(model_path)
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    return predict_video(video_path, model, label_encoder, config)
