"""
config.py
=========
Central configuration for the Transfer Learning-based Autism Behavior Detection System.

WHY these values were chosen is documented inline.
"""

CONFIG = {
    # ── Dataset ────────────────────────────────────────────────────────────────
    "dataset_path":         "dataset",
    "classes":              ["arm_flapping", "head_banging", "spinning", "normal"],
    "supported_extensions": (".mp4", ".avi", ".mov", ".mkv"),

    # ── Frame Sampling ─────────────────────────────────────────────────────────
    # WHY 224: MobileNetV2 was pretrained on ImageNet at 224×224.
    #          Using native resolution avoids a resolution mismatch penalty.
    # WHY 20 frames: captures ~0.7s of motion at 30fps — enough for one full
    #                gesture cycle without blowing up memory on CPU.
    "sequence_length":      8,
    "img_height":           96,
    "img_width":            96,

    # WHY 4 clips: 111 videos × 4 clips × 2 (augmentation) = ~888 samples.
    #              More clips = more temporal diversity from the same video.
    "clips_per_video":      4,

    # ── Transfer Learning Phases ───────────────────────────────────────────────
    # Phase 1 — Frozen base, train only LSTM + head
    # WHY: The pretrained CNN weights are fragile initially.
    #      Training the recurrent head first lets it stabilise before
    #      we start nudging the ImageNet features.
    "phase1_epochs":        20,
    "phase1_lr":            1e-3,   # High LR is fine — only new layers train

    # Phase 2 — Unfreeze top N layers of MobileNetV2, fine-tune end-to-end
    # WHY 30 layers: MobileNetV2 has 154 layers total.
    #                Unfreezing the last ~30 (last 2 inverted residual blocks)
    #                adapts high-level features to video/motion while preserving
    #                low-level edge/texture features that transfer universally.
    "phase2_epochs":        40,
    "phase2_lr":            1e-5,   # 10× lower — prevent destroying pretrained weights
    "unfreeze_top_layers":  30,

    # ── Augmentation ───────────────────────────────────────────────────────────
    "use_augmentation":     True,
    # Augmentations used (all applied consistently across all frames in a clip):
    #   horizontal_flip, brightness_jitter, contrast_jitter, temporal_reverse,
    #   gaussian_noise. See augmentation.py for details.

    # ── Training (shared) ──────────────────────────────────────────────────────
    # WHY batch_size=4: Each sample is (20, 224, 224, 3) = ~36MB float32.
    #                   batch=4 → ~144MB per batch — safe for 8GB RAM on CPU.
    "batch_size":           4,
    "test_split":           0.20,
    "val_split":            0.10,
    "early_stop_patience":  12,
    "lr_reduce_patience":   5,
    "lr_reduce_factor":     0.4,
    "min_lr":               1e-7,
    "use_class_weights":    True,

    # ── Model Architecture ─────────────────────────────────────────────────────
    # WHY GRU over LSTM: GRU has fewer parameters (no output gate) — better
    #                    for small datasets. Comparable accuracy in practice.
    "rnn_type":             "GRU",   # "GRU" or "LSTM"
    "rnn_units":            [128, 64],
    "dropout_rate":         0.5,
    "l2_reg":               1e-4,    # L2 regularisation on Dense layers

    # ── Paths ──────────────────────────────────────────────────────────────────
    "model_save_path":      "autism_mobilenet_gru.keras",
    "encoder_save_path":    "label_encoder.pkl",
    "plot_confusion":       "confusion_matrix.png",
    "plot_history":         "training_history.png",
}
