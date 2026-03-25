"""
video_loader.py
===============
Frame extraction directly from video files — no disk writes.

Changes from v1
───────────────
• Frames preprocessed with mobilenet_v2.preprocess_input() instead of /255.
  WHY: MobileNetV2 was trained with pixel values in [-1, 1] via this function.
       Using raw [0,1] or [0,255] inputs breaks the BatchNorm statistics baked
       into the pretrained weights, degrading feature quality significantly.
• Clips sampled with slight random jitter when augmenting — adds temporal
  diversity even within a single video segment.
"""

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: read one clip from an open VideoCapture
# ─────────────────────────────────────────────────────────────────────────────

def _read_clip_segment(
    cap:        cv2.VideoCapture,
    seg_start:  int,
    seg_end:    int,
    seq_len:    int,
    img_h:      int,
    img_w:      int,
    jitter:     bool = False,
) -> np.ndarray | None:
    """
    Sample `seq_len` equally-spaced frames from [seg_start, seg_end].

    If jitter=True, add a small random offset to each index (temporal jitter),
    which creates variety between clips from the same segment.
    """
    indices = np.linspace(seg_start, seg_end - 1, seq_len, dtype=int)

    if jitter:
        # Shift indices by ±2 frames — subtle but adds real diversity
        noise   = np.random.randint(-2, 3, size=len(indices))
        indices = np.clip(indices + noise, seg_start, seg_end - 1)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        # WHY preprocess_input: maps uint8 [0,255] → float32 [-1, 1]
        # This matches the normalisation MobileNetV2 was trained with.
        frame = preprocess_input(frame.astype(np.float32))
        frames.append(frame)

    if len(frames) != seq_len:
        return None

    return np.stack(frames, axis=0)   # (T, H, W, 3)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: extract N clips from one video
# ─────────────────────────────────────────────────────────────────────────────

def extract_clips(
    video_path:      str,
    sequence_length: int,
    img_height:      int,
    img_width:       int,
    clips_per_video: int = 1,
    jitter:          bool = False,
) -> list[np.ndarray]:
    """
    Divide a video into `clips_per_video` equal segments and extract one clip
    per segment. Each clip is a tensor of shape (T, H, W, 3) with pixel values
    normalised for MobileNetV2 (range [-1, 1]).

    Args:
        video_path      : Path to the video file.
        sequence_length : Number of frames per clip (T).
        img_height      : Target frame height.
        img_width       : Target frame width.
        clips_per_video : Number of clips to extract.
        jitter          : If True, add small random temporal offsets.

    Returns:
        List of np.ndarray, each (T, H, W, 3). Empty list on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < sequence_length:
        print(f"  [WARN] Too few frames ({total}): {video_path}")
        cap.release()
        return []

    # Clamp clips_per_video so segments have at least seq_len frames each
    max_clips       = max(1, total // sequence_length)
    clips_per_video = min(clips_per_video, max_clips)
    seg_size        = total // clips_per_video

    clips = []
    for i in range(clips_per_video):
        seg_start = i * seg_size
        seg_end   = seg_start + seg_size
        clip      = _read_clip_segment(
            cap, seg_start, seg_end, sequence_length, img_height, img_width, jitter
        )
        if clip is not None:
            clips.append(clip)

    cap.release()
    return clips


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: inference — single clip from full video span
# ─────────────────────────────────────────────────────────────────────────────

def extract_single_clip(
    video_path:      str,
    sequence_length: int,
    img_height:      int,
    img_width:       int,
) -> np.ndarray | None:
    """Extract exactly one clip spanning the full video. Used for inference."""
    clips = extract_clips(video_path, sequence_length, img_height, img_width,
                          clips_per_video=1, jitter=False)
    return clips[0] if clips else None
