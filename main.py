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
3. Class weights computed (handles the 54 normal vs ~19 minority imbalance)
4. MobileNetV2 + GRU model built (pretrained ImageNet weights)
5. Phase 1: Train GRU head only (CNN frozen, 20 epochs)
6. Phase 2: Fine-tune top 30 MobileNetV2 layers (40 epochs, LR=1e-4)
7. Best checkpoint saved, confusion matrix + training plot generated
"""

import argparse
import os
import pickle
import sys

from config          import CONFIG
from dataset_builder import build_dataset
from model           import build_transfer_model
from trainer         import train
from predictor       import load_and_predict


def run_training() -> None:
    # ── 1. Build dataset ─────────────────────────────────────────────────
    X, y, label_encoder, class_weights = build_dataset(CONFIG)

    # Save encoder for later inference
    enc_path = CONFIG["encoder_save_path"]
    with open(enc_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"\n  LabelEncoder saved → {enc_path}")

    # ── 2. Build model ────────────────────────────────────────────────────
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
    train(model, X, y, CONFIG, class_weights=class_weights)

    print("\n" + "═" * 65)
    print("  Training complete.")
    print(f"  Phase 1 best → {CONFIG['model_save_path'].replace('.keras','_phase1.keras')}")
    print(f"  Phase 2 best → {CONFIG['model_save_path'].replace('.keras','_phase2.keras')}")
    print(f"  Encoder      → {CONFIG['encoder_save_path']}")
    print("═" * 65)


def run_inference(video_path: str) -> None:
    # Use Phase 2 model if available, fall back to Phase 1
    p2_path = CONFIG["model_save_path"].replace(".keras", "_phase2.keras")
    p1_path = CONFIG["model_save_path"].replace(".keras", "_phase1.keras")
    enc_path = CONFIG["encoder_save_path"]

    if os.path.exists(p2_path):
        model_path = p2_path
    elif os.path.exists(p1_path):
        model_path = p1_path
        print("  [NOTE] Phase 2 model not found; using Phase 1.")
    else:
        print("[ERROR] No trained model found. Run training first: python main.py")
        sys.exit(1)

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
