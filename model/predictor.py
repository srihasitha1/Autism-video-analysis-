"""
predictor.py
============
Single-video inference using the trained MobileNetV2 + GRU model.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from video_loader import extract_sliding_window_clips
from tta import tta_predict


def predict_video(
    video_path:    str,
    model:         tf.keras.Model,
    label_encoder: LabelEncoder,
    config:        dict,
) -> dict:
    """
    Run inference on a single video file using sliding window.

    WHY sliding window:
        The model was trained on short fixed-length clips from trimmed videos.
        Real-world videos are longer and may contain mixed content. A single
        clip drawn from the full video may miss the actual behavior window.
        Sliding window extracts clips at 50% overlap across the entire video,
        averages all softmax outputs, and picks the dominant class — much more
        robust to variable-length real-world footage.

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
            "clips_evaluated"    — int (number of sliding window clips used)
            "error"              — string (only if extraction fails)
    """
    print(f"\n  Predicting: {video_path}")

    clips = extract_sliding_window_clips(
        video_path,
        sequence_length=config["sequence_length"],
        img_height=config["img_height"],
        img_width=config["img_width"],
        overlap=0.5,
    )

    if not clips:
        return {"error": f"Frame extraction failed: {video_path}"}

    print(f"  [Sliding Window] Evaluating {len(clips)} clips...")

    all_probs = []
    use_tta = config.get("use_tta", False)
    n_aug = config.get("tta_augments", 8)

    for clip in clips:
        if use_tta:
            probs = tta_predict(clip.astype(np.float32), model, n_augments=n_aug)
        else:
            tensor = np.expand_dims(clip, axis=0).astype(np.float32)
            probs  = model.predict(tensor, verbose=0)[0]
        all_probs.append(probs)

    # Average across all clips — reduces noise, improves robustness
    avg_probs  = np.mean(all_probs, axis=0)
    pred_idx   = int(np.argmax(avg_probs))
    pred_class = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(avg_probs[pred_idx])

    all_probs_dict = {
        label_encoder.inverse_transform([i])[0]: round(float(p), 4)
        for i, p in enumerate(avg_probs)
    }

    print(f"  ┌─ Predicted : {pred_class}")
    print(f"  ├─ Confidence: {confidence:.2%}  (avg of {len(clips)} clips)")
    print(f"  └─ All probabilities:")
    for cls_name, p in sorted(all_probs_dict.items(), key=lambda kv: -kv[1]):
        bar = "█" * int(p * 30)
        print(f"       {cls_name:<15} {p:.4f}  {bar}")

    return {
        "predicted_class":   pred_class,
        "confidence":        round(confidence, 4),
        "all_probabilities": all_probs_dict,
        "clips_evaluated":   len(clips),
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
