"""
dataset_builder.py
==================
Builds X / y arrays from video folders.

Flow per class folder
─────────────────────
  for each video:
    extract N clips  (clips_per_video segments)
    for each clip:
      store original
      if augmentation: store augment_clip(clip)   ← in-memory, no disk

When optical flow is enabled (Improvement 7), also computes dense flow fields
between consecutive frames for each clip, producing a parallel X_flow array.

Returns
───────
  X             : (N, T, H, W, 3)   float32  — MobileNetV2-normalised
  X_flow        : (N, T-1, fH, fW, 2) float32 — optical flow (if enabled, else None)
  y             : (N, num_classes)   float32  — one-hot
  label_encoder : fitted LabelEncoder
  class_weights : {int: float}  for model.fit(class_weight=...)
"""

import os
from collections import Counter

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

from video_loader import extract_clips
from augmentation import augment_clip


def build_dataset(config: dict) -> tuple:
    """
    Build the full dataset from video folders.

    Args:
        config: Global CONFIG dict.

    Returns:
        (X, X_flow, y, label_encoder, class_weights)
        X_flow is None if use_optical_flow is False.
    """
    X_list:    list[np.ndarray] = []
    flow_list: list[np.ndarray] = []
    y_raw:     list[str]        = []

    dataset_path    = config["dataset_path"]
    classes         = config["classes"]
    seq_len         = config["sequence_length"]
    h               = config["img_height"]
    w               = config["img_width"]
    exts            = config["supported_extensions"]
    clips_per_video = config.get("clips_per_video", 3)
    use_aug         = config.get("use_augmentation", True)
    use_flow        = config.get("use_optical_flow", False)
    flow_h          = config.get("flow_height", 64)
    flow_w          = config.get("flow_width", 64)

    # Lazy import to avoid overhead when flow is disabled
    if use_flow:
        from optical_flow import extract_flow_clip

    print("=" * 65)
    print("  Building dataset")
    print(f"  clips_per_video : {clips_per_video}")
    print(f"  augmentation    : {use_aug}")
    print(f"  img resolution  : {h}×{w}")
    print(f"  optical flow    : {use_flow}  (flow res: {flow_h}×{flow_w})")
    print("=" * 65)

    for class_name in classes:
        folder = os.path.join(dataset_path, class_name)

        if not os.path.isdir(folder):
            print(f"\n  [SKIP] Folder not found: {folder}")
            continue

        video_files = [f for f in os.listdir(folder)
                       if f.lower().endswith(exts)]

        if not video_files:
            print(f"\n  [SKIP] No videos in: {folder}")
            continue

        print(f"\n  Class '{class_name}':  {len(video_files)} video(s)")
        total_samples = 0

        for vf in video_files:
            path  = os.path.join(folder, vf)
            clips = extract_clips(
                path, seq_len, h, w,
                clips_per_video=clips_per_video,
                jitter=use_aug,   # temporal jitter during augmentation pass
            )

            for clip in clips:
                # Always store the original clip
                X_list.append(clip)
                y_raw.append(class_name)
                total_samples += 1

                # Compute optical flow for the original clip
                if use_flow:
                    flow_list.append(extract_flow_clip(clip, flow_h, flow_w))

                # Store one augmented copy per clip
                if use_aug:
                    aug_clip = augment_clip(clip)
                    X_list.append(aug_clip)
                    y_raw.append(class_name)
                    total_samples += 1

                    # Compute optical flow for the augmented clip too
                    if use_flow:
                        flow_list.append(extract_flow_clip(aug_clip, flow_h, flow_w))

        samples_per_video = clips_per_video * (2 if use_aug else 1)
        print(f"    → {total_samples} samples  "
              f"({len(video_files)} videos × {samples_per_video})")

    if not X_list:
        raise ValueError(
            f"No samples loaded from '{dataset_path}'. "
            "Check that class subfolders exist and contain videos."
        )

    X = np.array(X_list, dtype=np.float16)   # (N, T, H, W, 3)
    del X_list  # Free the list — X array is the only copy now

    # Build optical flow array (or None)
    X_flow = None
    if use_flow and flow_list:
        X_flow = np.array(flow_list, dtype=np.float16)  # (N, T-1, fH, fW, 2)
        print(f"\n  X_flow shape    : {X_flow.shape}  ({X_flow.dtype})")
    del flow_list  # Free even if empty

    # Reclaim memory from the intermediate lists
    import gc
    gc.collect()

    # ── Label encoding ────────────────────────────────────────────────────
    le    = LabelEncoder()
    le.fit(classes)
    y_int = le.transform(y_raw)                              # (N,)
    y     = to_categorical(y_int, num_classes=len(classes))  # (N, C)

    # ── Class weights ─────────────────────────────────────────────────────
    # WHY: With 54 "normal" vs ~19 "arm_flapping" videos, the model will see
    #      the majority class far more often. Class weights make each class
    #      contribute equally to the loss regardless of sample count.
    unique = np.unique(y_int)
    cw     = compute_class_weight("balanced", classes=unique, y=y_int)
    class_weights = {int(c): float(w) for c, w in zip(unique, cw)}

    # ── Summary ───────────────────────────────────────────────────────────
    counts = Counter(y_raw)
    print("\n" + "=" * 65)
    print(f"  Total samples   : {len(X_list)}")
    print(f"  X shape         : {X.shape}  ({X.dtype})")
    print(f"  Pixel range     : [{X.min():.2f}, {X.max():.2f}]  (MobileNetV2: [-1,1])")
    print(f"  Per-class counts: {dict(counts)}")
    cw_display = {le.inverse_transform([k])[0]: round(v, 3)
                  for k, v in class_weights.items()}
    print(f"  Class weights   : {cw_display}")
    print("=" * 65)

    return X, X_flow, y, le, class_weights
