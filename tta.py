"""
tta.py
======
Test-Time Augmentation (TTA) for inference.

WHY TTA improves accuracy
─────────────────────────
During training, the model sees augmented clips (flipped, brightness-shifted,
reversed, etc.). At test time, however, the default pipeline feeds a single
clean clip — creating a distribution mismatch between train and test.

TTA closes this gap by generating N augmented copies of the input clip,
running each through the model, and averaging the softmax probability vectors.
This reduces prediction variance and consistently adds 0.5–1.5% accuracy on
small datasets.

Augmentation set (9 total = 1 original + 8 transforms)
───────────────────────────────────────────────────────
  0. original            — baseline prediction
  1. horizontal flip     — behaviour is left-right symmetric
  2. brightness +0.1     — simulates brighter lighting
  3. brightness −0.1     — simulates dimmer lighting
  4. contrast ×1.1       — slightly sharper image
  5. contrast ×0.9       — slightly flatter image
  6. temporal reverse    — cyclic behaviours are time-symmetric
  7. flip + brightness   — compound transform
  8. flip + reverse      — compound transform
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Individual transform functions (clip-consistent)
# ─────────────────────────────────────────────────────────────────────────────

def _flip(clip: np.ndarray) -> np.ndarray:
    """Horizontal flip — mirrors all frames along the width axis."""
    return clip[:, :, ::-1, :].copy()


def _brightness(clip: np.ndarray, delta: float) -> np.ndarray:
    """Shift brightness by a constant offset, clipped to [-1, 1]."""
    return np.clip(clip + delta, -1.0, 1.0)


def _contrast(clip: np.ndarray, factor: float) -> np.ndarray:
    """Scale contrast around zero, clipped to [-1, 1]."""
    return np.clip(clip * factor, -1.0, 1.0)


def _temporal_reverse(clip: np.ndarray) -> np.ndarray:
    """Reverse the frame order — time flows backwards."""
    return clip[::-1, :, :, :].copy()


# ─────────────────────────────────────────────────────────────────────────────
# TTA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

# The fixed set of 9 TTA transforms (original + 8 augmented)
TTA_TRANSFORMS = [
    ("original",            lambda c: c),
    ("flip",                _flip),
    ("bright_+0.1",         lambda c: _brightness(c, 0.1)),
    ("bright_-0.1",         lambda c: _brightness(c, -0.1)),
    ("contrast_1.1",        lambda c: _contrast(c, 1.1)),
    ("contrast_0.9",        lambda c: _contrast(c, 0.9)),
    ("temporal_reverse",    _temporal_reverse),
    ("flip+bright",         lambda c: _brightness(_flip(c), 0.1)),
    ("flip+reverse",        lambda c: _temporal_reverse(_flip(c))),
]


def tta_predict(
    clip:   np.ndarray,
    model,
    n_augments: int = 8,
) -> np.ndarray:
    """
    Run TTA inference on a single clip.

    Generates `n_augments` augmented copies + 1 original = n_augments+1
    forward passes. Returns the averaged softmax probability vector.

    Args:
        clip        : np.ndarray, shape (T, H, W, 3), MobileNetV2-normalised.
        model       : Trained Keras model accepting (1, T, H, W, 3) input.
        n_augments  : Number of augmented copies to use (max 8).

    Returns:
        np.ndarray, shape (num_classes,) — averaged softmax probabilities.
    """
    transforms = TTA_TRANSFORMS[: n_augments + 1]  # original + n_augments

    all_probs = []
    for name, fn in transforms:
        aug_clip = fn(clip).astype(np.float32)
        tensor   = np.expand_dims(aug_clip, axis=0)   # (1, T, H, W, 3)
        probs    = model.predict(tensor, verbose=0)[0]
        all_probs.append(probs)

    # Average the softmax outputs — the core TTA mechanism
    return np.mean(all_probs, axis=0)
