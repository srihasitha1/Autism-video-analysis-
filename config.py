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

    # ══════════════════════════════════════════════════════════════════════════
    # IMPROVEMENT 1 — Test-Time Augmentation (TTA)
    # ══════════════════════════════════════════════════════════════════════════
    # WHY: The model trains on augmented clips but predicts on a single clean
    #      clip — a train/test distribution mismatch. TTA averages predictions
    #      from the same augmentation distribution the model trained on.
    "use_tta":              True,
    "tta_augments":         8,       # 8 augmented + 1 original = 9 forward passes

    # ══════════════════════════════════════════════════════════════════════════
    # IMPROVEMENT 2 — Mixup Augmentation
    # ══════════════════════════════════════════════════════════════════════════
    # WHY: Blends random pairs of clips with soft labels. Forces the model to
    #      learn linear interpolations between classes → smoother decision
    #      boundary → better generalisation on small datasets.
    "use_mixup":            True,
    "mixup_alpha":          0.4,     # Beta(0.4, 0.4) distribution for λ
    "mixup_lr":             5e-5,    # Fine-tune LR for Mixup (lower than Phase 2)
    "mixup_finetune_epochs": 20,     # Max epochs for Mixup fine-tuning

    # ══════════════════════════════════════════════════════════════════════════
    # IMPROVEMENT 3 — Label Smoothing
    # ══════════════════════════════════════════════════════════════════════════
    # WHY: Prevents overconfident softmax outputs (1.0 → 0.9 for true class).
    #      Produces better-calibrated probabilities that generalise more reliably.
    "label_smoothing":          0.1,
    "smoothing_lr":             5e-5,    # Fine-tune LR for label smoothing
    "smoothing_finetune_epochs": 20,     # Max epochs

    # ══════════════════════════════════════════════════════════════════════════
    # IMPROVEMENT 4 — Discriminative Learning Rates
    # ══════════════════════════════════════════════════════════════════════════
    # WHY: A single LR is a compromise — too high for pretrained CNN weights,
    #      too low for the new GRU head. Discriminative LRs let each group
    #      update at the rate appropriate for how much it needs to change.
    "discriminative_lr":        True,
    "unfreeze_top_layers_p2":   60,      # Unfreeze 60 layers (up from 30)
    "unfreeze_layers_p3":       60,      # Used by improve.py for improvement 4
    "lr_head":                  1e-4,    # GRU + Dense head — highest LR
    "lr_top30":                 2e-5,    # Top 30 MobileNetV2 layers — medium
    "lr_next30":                5e-6,    # Layers 31–60 from top — lowest LR
    "disc_lr_head":             1e-4,    # Discriminative LR for head group
    "disc_lr_top30":            2e-5,    # Discriminative LR for top 30 layers
    "disc_lr_mid30":            5e-6,    # Discriminative LR for mid 30 layers
    "disc_finetune_epochs":     20,      # Max epochs for discriminative LR fine-tuning

    # ══════════════════════════════════════════════════════════════════════════
    # IMPROVEMENT 5 — Cosine Annealing LR Schedule
    # ══════════════════════════════════════════════════════════════════════════
    # WHY: ReduceLROnPlateau is reactive — it only reduces when val_loss stops
    #      improving. Cosine annealing proactively cycles the LR, allowing the
    #      optimizer to escape shallow local minima and find deeper ones.
    "use_cosine_annealing":     True,
    "cosine_initial_lr":        1e-4,    # Starting LR for cosine schedule
    "cosine_first_decay_epochs": 10,     # First cycle length in epochs
    "cosine_t_mul":             2.0,     # Each restart cycle is 2× longer
    "cosine_m_mul":             0.9,     # Peak LR drops by 10% at each restart
    "cosine_finetune_epochs":   20,      # Max epochs for cosine fine-tuning

    # ══════════════════════════════════════════════════════════════════════════
    # IMPROVEMENT 6 — Stochastic Weight Averaging (SWA)
    # ══════════════════════════════════════════════════════════════════════════
    # WHY: SGD/Adam often land in sharp, narrow minima that generalise poorly.
    #      SWA averages weights from the last N epochs → flatter minimum →
    #      better generalisation. Typically adds 0.5–1.5% on small datasets.
    "use_swa":              True,
    "swa_epochs":           15,          # Fine-tune epochs for SWA snapshot collection
    "swa_lr":               1e-5,        # Low constant LR for SWA collection

    # ══════════════════════════════════════════════════════════════════════════
    # IMPROVEMENT 7 — Optical Flow Dual-Stream
    # ══════════════════════════════════════════════════════════════════════════
    # WHY: RGB tells the model what the body looks like. Optical flow explicitly
    #      encodes how fast and in what direction things move — the kinematic
    #      signature of each behaviour. Arm flapping has a distinctive oscillating
    #      flow pattern that is unambiguous even when RGB frames are ambiguous.
    "use_optical_flow":     False,   # improve.py enables this for improvement 7
    "flow_height":          64,          # Flow maps are coarse — 64×64 is enough
    "flow_width":           64,
    "flow_finetune_epochs": 20,          # Max epochs for dual-stream fine-tuning
}
