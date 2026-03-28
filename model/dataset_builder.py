"""
dataset_builder.py
==================
Builds X / y arrays from video folders.

CRITICAL FIX: Split is now done at VIDEO level (not clip level).
─────────────────────────────────────────────────────────────────
Previously, all clips from all videos were pooled first and then
train_test_split was applied to the clip array. This caused DATA LEAKAGE:
clips from the same source video appeared in both train AND test, so the
model was effectively evaluated on variants of its own training videos.
This inflated test accuracy to ~97% but the model failed on external videos.

The fix: partition the list of video FILES into train/val/test first.
Clips are then extracted ONLY from the video files within each partition.
No source video can contribute clips to more than one split.

Flow per class folder
─────────────────────
  split video files → train_files | val_files | test_files
  for each split:
    for each video file:
      extract N clips (clips_per_video segments)
      if augmentation and this is the train split: store augment_clip(clip)

When optical flow is enabled (Improvement 7), also computes dense flow fields
between consecutive frames for each clip, producing parallel X_flow arrays.

Returns
───────
  X_train, X_val, X_test              : (N, T, H, W, 3) float32
  X_flow_train, X_flow_val, X_flow_test: (N, T-1, fH, fW, 2) or None
  y_train, y_val, y_test               : (N, num_classes) float32 one-hot
  label_encoder : fitted LabelEncoder
  class_weights : {int: float}  for model.fit(class_weight=...)
"""

import os
import random
from collections import Counter

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

from video_loader import extract_clips
from augmentation import augment_clip


def _collect_clips_from_files(
    video_files:     list,
    folder:          str,
    class_name:      str,
    seq_len:         int,
    h:               int,
    w:               int,
    exts:            tuple,
    clips_per_video: int,
    use_aug:         bool,
    use_flow:        bool,
    flow_h:          int,
    flow_w:          int,
    is_train:        bool,
) -> tuple:
    """
    Internal helper: extract clips from a list of video files.

    Augmentation is only applied to the TRAIN split (is_train=True).
    Returns (X_list, flow_list, y_list).
    """
    if use_flow:
        from optical_flow import extract_flow_clip

    X_list    = []
    flow_list = []
    y_list    = []

    for vf in video_files:
        path  = os.path.join(folder, vf)
        clips = extract_clips(
            path, seq_len, h, w,
            clips_per_video=clips_per_video,
            jitter=(use_aug and is_train),
        )

        for clip in clips:
            X_list.append(clip)
            y_list.append(class_name)

            if use_flow:
                flow_list.append(extract_flow_clip(clip, flow_h, flow_w))

            # Only augment training clips — NEVER val or test
            if use_aug and is_train:
                aug_clip = augment_clip(clip)
                X_list.append(aug_clip)
                y_list.append(class_name)

                if use_flow:
                    flow_list.append(extract_flow_clip(aug_clip, flow_h, flow_w))

    return X_list, flow_list, y_list


def build_dataset(config: dict) -> tuple:
    """
    Build train / val / test datasets from video folders.

    The split is performed at the VIDEO FILE level to prevent data leakage.
    See module docstring for a full explanation.

    Args:
        config: Global CONFIG dict.

    Returns:
        (X_train, X_val, X_test,
         X_flow_train, X_flow_val, X_flow_test,   ← None if use_optical_flow=False
         y_train, y_val, y_test,
         label_encoder, class_weights)
    """
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
    test_split      = config.get("test_split", 0.20)
    val_split       = config.get("val_split", 0.10)
    random_seed     = 42

    print("=" * 65)
    print("  Building dataset  [VIDEO-LEVEL SPLIT — no data leakage]")
    print(f"  clips_per_video : {clips_per_video}")
    print(f"  augmentation    : {use_aug}  (train split only)")
    print(f"  img resolution  : {h}×{w}")
    print(f"  optical flow    : {use_flow}  (flow res: {flow_h}×{flow_w})")
    print(f"  splits          : train={1-test_split-val_split:.0%} | "
          f"val={val_split:.0%} | test={test_split:.0%}  (at video level)")
    print("=" * 65)

    # Accumulate clips per split
    X_tr, flow_tr, y_tr = [], [], []
    X_va, flow_va, y_va = [], [], []
    X_te, flow_te, y_te = [], [], []

    for class_name in classes:
        folder = os.path.join(dataset_path, class_name)

        if not os.path.isdir(folder):
            print(f"\n  [SKIP] Folder not found: {folder}")
            continue

        video_files = sorted([
            f for f in os.listdir(folder) if f.lower().endswith(exts)
        ])

        if not video_files:
            print(f"\n  [SKIP] No videos in: {folder}")
            continue

        # ── VIDEO-LEVEL SPLIT ──────────────────────────────────────────
        rng = random.Random(random_seed)
        rng.shuffle(video_files)

        n = len(video_files)
        n_test = max(1, round(n * test_split))
        n_val  = max(1, round(n * val_split))
        n_test = min(n_test, n - 2)   # leave at least 1 for train + val
        n_val  = min(n_val,  n - n_test - 1)

        test_files  = video_files[:n_test]
        val_files   = video_files[n_test: n_test + n_val]
        train_files = video_files[n_test + n_val:]

        print(f"\n  Class '{class_name}':  {n} videos  "
              f"→ train={len(train_files)}  val={len(val_files)}  test={len(test_files)}")

        # ── Extract clips for each split ───────────────────────────────
        kwargs = dict(
            folder=folder, class_name=class_name, seq_len=seq_len,
            h=h, w=w, exts=exts, clips_per_video=clips_per_video,
            use_aug=use_aug, use_flow=use_flow, flow_h=flow_h, flow_w=flow_w,
        )

        _x, _f, _y = _collect_clips_from_files(train_files, is_train=True,  **kwargs)
        X_tr.extend(_x); flow_tr.extend(_f); y_tr.extend(_y)

        _x, _f, _y = _collect_clips_from_files(val_files,   is_train=False, **kwargs)
        X_va.extend(_x); flow_va.extend(_f); y_va.extend(_y)

        _x, _f, _y = _collect_clips_from_files(test_files,  is_train=False, **kwargs)
        X_te.extend(_x); flow_te.extend(_f); y_te.extend(_y)

    # ── Validation ─────────────────────────────────────────────────────────
    for split_name, X_list in [("Train", X_tr), ("Val", X_va), ("Test", X_te)]:
        if not X_list:
            raise ValueError(
                f"No samples in {split_name} split from '{dataset_path}'. "
                "Dataset may be too small or folders missing."
            )

    # ── Convert to arrays ──────────────────────────────────────────────────
    import gc

    X_train = np.array(X_tr, dtype=np.float16); del X_tr
    X_val   = np.array(X_va, dtype=np.float16); del X_va
    X_test  = np.array(X_te, dtype=np.float16); del X_te
    gc.collect()

    X_flow_train = X_flow_val = X_flow_test = None
    if use_flow:
        X_flow_train = np.array(flow_tr, dtype=np.float16)
        X_flow_val   = np.array(flow_va, dtype=np.float16)
        X_flow_test  = np.array(flow_te, dtype=np.float16)
        print(f"\n  X_flow shapes: train={X_flow_train.shape} "
              f"val={X_flow_val.shape} test={X_flow_test.shape}")
    del flow_tr, flow_va, flow_te
    gc.collect()

    # ── Label encoding ─────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(classes)

    def _encode(y_raw):
        y_int = le.transform(y_raw)
        return to_categorical(y_int, num_classes=len(classes)), y_int

    y_train, y_tr_int = _encode(y_tr)
    y_val,   _        = _encode(y_va)
    y_test,  _        = _encode(y_te)

    # ── Class weights (from training split only) ───────────────────────────
    unique = np.unique(y_tr_int)
    cw     = compute_class_weight("balanced", classes=unique, y=y_tr_int)
    class_weights = {int(c): float(w) for c, w in zip(unique, cw)}

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  Train samples   : {len(X_train)}  "
          f"(from {len(y_tr)} clips, {len(y_tr)//clips_per_video//max(1,(2 if use_aug else 1))} source videos)")
    print(f"  Val   samples   : {len(X_val)}")
    print(f"  Test  samples   : {len(X_test)}")
    print(f"  X shape (train) : {X_train.shape}  ({X_train.dtype})")
    cw_display = {le.inverse_transform([k])[0]: round(v, 3)
                  for k, v in class_weights.items()}
    print(f"  Class weights   : {cw_display}")
    print(f"  Per-class (train): {dict(Counter(y_tr))}")
    print("=" * 65)

    return (X_train, X_val, X_test,
            X_flow_train, X_flow_val, X_flow_test,
            y_train, y_val, y_test,
            le, class_weights)

