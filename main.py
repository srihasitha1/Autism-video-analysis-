"""
main.py
=======
Entry point for the Autism Behavior Detection System (Transfer Learning version).

Training pipeline:
    python main.py

Inference on a new video:
    python main.py --predict path/to/video.mp4

What happens during training
─────────────────────────────
1. Videos loaded → multiple clips extracted in memory (no disk writes)
2. Augmentation applied (in-memory)
3. Optical flow computed (if enabled — Improvement 7)
4. Class weights computed (handles the 54 normal vs ~19 minority imbalance)
5. Model built:
   - Single-stream: MobileNetV2 + GRU (default)
   - Dual-stream: MobileNetV2 + GRU + Flow CNN + GRU (if optical flow enabled)
6. Phase 1: Train GRU head only (CNN frozen, 20 epochs)
7. Phase 2: Fine-tune with improvements:
   - Mixup on 30% of batches   (Improvement 2)
   - Label smoothing = 0.1     (Improvement 3)
   - Discriminative LRs         (Improvement 4)
   - Cosine annealing schedule  (Improvement 5)
   - SWA weight averaging       (Improvement 6)
8. Best checkpoint saved, confusion matrix + training plot generated
9. SWA model saved separately   (Improvement 6)
"""

import argparse
import os
import pickle
import sys

from config          import CONFIG
from dataset_builder import build_dataset
from model           import build_transfer_model, build_dual_stream_model
from trainer         import train
from predictor       import load_and_predict


def run_training() -> None:
    # ── 1. Build dataset ─────────────────────────────────────────────────
    X, X_flow, y, label_encoder, class_weights = build_dataset(CONFIG)

    # Save encoder for later inference
    enc_path = CONFIG["encoder_save_path"]
    with open(enc_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"\n  LabelEncoder saved → {enc_path}")

    # ── 2. Build model ────────────────────────────────────────────────────
    use_flow = CONFIG.get("use_optical_flow", False) and X_flow is not None

    if use_flow:
        print("\n  Building DUAL-STREAM model (RGB + Optical Flow)")
        model = build_dual_stream_model(
            sequence_length=CONFIG["sequence_length"],
            img_height=CONFIG["img_height"],
            img_width=CONFIG["img_width"],
            num_classes=len(CONFIG["classes"]),
            rnn_units=CONFIG["rnn_units"],
            dropout_rate=CONFIG["dropout_rate"],
            l2_reg=CONFIG["l2_reg"],
            rnn_type=CONFIG["rnn_type"],
            flow_height=CONFIG.get("flow_height", 64),
            flow_width=CONFIG.get("flow_width", 64),
        )
    else:
        model = build_transfer_model(
            sequence_length=CONFIG["sequence_length"],
            img_height=CONFIG["img_height"],
            img_width=CONFIG["img_width"],
            num_classes=len(CONFIG["classes"]),
            rnn_units=CONFIG["rnn_units"],
            dropout_rate=CONFIG["dropout_rate"],
            l2_reg=CONFIG["l2_reg"],
            rnn_type=CONFIG["rnn_type"],
        )

    # ── 3. Two-phase training ─────────────────────────────────────────────
    train(model, X, y, CONFIG,
          class_weights=class_weights,
          X_flow=X_flow if use_flow else None)

    # ── 4. Save final model ───────────────────────────────────────────────
    # Save as the "final" model with all improvements applied
    final_path = "autism_final.keras"
    model.save(final_path)
    print(f"\n  Final model saved → {final_path}")

    # Also save as SWA model if SWA was used
    if CONFIG.get("use_swa", False):
        swa_path = "autism_mobilenet_swa.keras"
        model.save(swa_path)
        print(f"  SWA model saved  → {swa_path}")

    print("\n" + "═" * 65)
    print("  Training complete.")
    print(f"  Final model  → {final_path}")
    print(f"  Encoder      → {CONFIG['encoder_save_path']}")
    print(f"  Confusion    → confusion_matrix_final.png")
    print("═" * 65)


def run_inference(video_path: str) -> None:
    # Use final model if available, fall back to Phase 1 checkpoint
    model_candidates = [
        "autism_final.keras",
        "autism_mobilenet_swa.keras",
        "autism_mobilenet_gru_phase1.h5",
    ]
    enc_path = CONFIG["encoder_save_path"]

    model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            break

    if model_path is None:
        print("[ERROR] No trained model found. Run training first: python main.py")
        sys.exit(1)

    print(f"  Using model: {model_path}")

    if not os.path.exists(enc_path):
        print(f"[ERROR] Encoder not found: {enc_path}")
        sys.exit(1)

    result = load_and_predict(video_path, model_path, enc_path, CONFIG)

    if "error" not in result:
        print(f"\n  RESULT : {result['predicted_class']}"
              f"  ({result['confidence']:.2%} confidence)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autism Behavior Detection — MobileNetV2 + GRU Transfer Learning"
    )
    parser.add_argument(
        "--predict", metavar="VIDEO_PATH",
        help="Path to a video file for inference (skips training).",
        default=None,
    )
    args = parser.parse_args()

    if args.predict:
        run_inference(args.predict)
    else:
        run_training()
