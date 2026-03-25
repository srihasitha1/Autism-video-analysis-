"""
optical_flow.py
===============
Dense optical flow extraction using Farneback method.

WHY optical flow is a powerful second stream
─────────────────────────────────────────────
The RGB stream (MobileNetV2) learns *what the body looks like* — shape, colour,
spatial arrangement. Optical flow explicitly encodes *how things move* — the
direction and speed of every pixel between consecutive frames.

Key behaviours have distinctive flow signatures:
  • Arm flapping → rapid, oscillating vertical flow in the arm region
  • Head banging → directional flow concentrated at the head
  • Spinning     → large-magnitude rotational flow across the whole body
  • Normal       → low-magnitude, inconsistent flow

These motion patterns are unambiguous in flow space even when the RGB frames
are ambiguous (e.g. a still frame of raised arms could be flapping or waving).

Implementation
──────────────
Uses OpenCV's `calcOpticalFlowFarneback()` — a dense method that computes flow
at every pixel (unlike sparse Lucas-Kanade). Chosen because:
  1. No GPU required (runs on CPU)
  2. Dense flow captures whole-body motion patterns, not just keypoints
  3. Well-tested, stable OpenCV implementation
"""

import cv2
import numpy as np


def extract_flow_clip(
    frames:     np.ndarray,
    flow_height: int = 64,
    flow_width:  int = 64,
) -> np.ndarray:
    """
    Compute dense Farneback optical flow between consecutive frames.

    For T RGB frames, produces T-1 flow fields. Each flow field has 2 channels
    (dx, dy) representing horizontal and vertical pixel displacement.

    Args:
        frames      : np.ndarray, shape (T, H, W, 3), values in [-1, 1] or [0, 255].
        flow_height : Target height for flow maps (smaller = faster, coarse is fine).
        flow_width  : Target width for flow maps.

    Returns:
        np.ndarray, shape (T-1, flow_height, flow_width, 2), normalised to [-1, 1].
    """
    T = frames.shape[0]
    flows = []

    for i in range(T - 1):
        # Convert to uint8 grayscale for Farneback
        # Handle both [-1,1] and [0,255] input ranges
        frame_a = frames[i]
        frame_b = frames[i + 1]

        if frame_a.max() <= 1.0:
            # MobileNetV2 range [-1, 1] → [0, 255]
            frame_a = ((frame_a + 1.0) * 127.5).astype(np.uint8)
            frame_b = ((frame_b + 1.0) * 127.5).astype(np.uint8)
        else:
            frame_a = frame_a.astype(np.uint8)
            frame_b = frame_b.astype(np.uint8)

        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)

        # Resize to flow dimensions (smaller = faster computation)
        gray_a = cv2.resize(gray_a, (flow_width, flow_height))
        gray_b = cv2.resize(gray_b, (flow_width, flow_height))

        # Farneback dense optical flow
        # pyr_scale=0.5, levels=3, winsize=15 — good balance of speed vs accuracy
        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flows.append(flow)  # shape (flow_height, flow_width, 2)

    flow_stack = np.stack(flows, axis=0)  # (T-1, H, W, 2)

    # Normalise to [-1, 1] — prevents large flow magnitudes from dominating
    max_mag = np.abs(flow_stack).max()
    if max_mag > 0:
        flow_stack = flow_stack / max_mag

    return flow_stack.astype(np.float32)
