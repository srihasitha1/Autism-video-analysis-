"""
augmentation.py
===============
In-memory clip-level augmentation — no disk writes.

Design principle — CLIP CONSISTENCY
─────────────────────────────────────
Every augmentation is applied identically to ALL frames in a clip.
Inconsistent augmentation (e.g. flipping frame 3 but not frame 7) would
create impossible motion signals and hurt temporal learning.

WHY each augmentation is included
───────────────────────────────────
• horizontal_flip   — Behavior is symmetric; a left-arm flap = right-arm flap.
                      Doubles effective dataset with zero label noise.
• brightness_jitter — Accounts for different lighting conditions across recording
                      environments (indoor/outdoor, camera quality differences).
• contrast_jitter   — Same reasoning as brightness; prevents learning lighting
                      artefacts as class-specific features.
• temporal_reverse  — Many behaviors (spinning, flapping) look similar forwards
                      and backwards. Teaches the model motion-direction invariance.
• gaussian_noise    — Simulates camera sensor noise and compression artefacts.
                      Acts as a regulariser, preventing pixel-level memorisation.

What is deliberately NOT included
───────────────────────────────────
• Rotation / heavy crop — Would destroy body-part spatial relationships that
                          are diagnostic (e.g. arm position relative to torso).
• Colour jitter (hue)   — Skin/clothing colour can be a legitimate feature;
                          hue shifts introduce label noise.
• Cutout / random erase — Too destructive for small 64-frame clips; risks
                          erasing the diagnostic gesture entirely.

Mixup (Improvement 2)
──────────────────────
• mixup_batch() blends random pairs of clips with λ ~ Beta(α, α).
  Labels become soft (e.g. 0.6 × arm_flapping + 0.4 × spinning).
  This smooths the decision boundary and acts as a strong regulariser.
"""

import numpy as np


def augment_clip(clip: np.ndarray) -> np.ndarray:
    """
    Apply randomly-selected augmentations to an entire clip.

    Args:
        clip : np.ndarray  shape (T, H, W, 3)
               Values expected in MobileNetV2 range [-1, 1].

    Returns:
        Augmented clip, same shape and dtype.
    """
    clip = clip.copy()

    # 1. Horizontal flip (50% probability)
    if np.random.rand() < 0.5:
        clip = clip[:, :, ::-1, :]          # flip W axis consistently

    # 2. Brightness jitter (60% probability)
    # Shift all pixels by a constant offset in [-0.2, +0.2]
    if np.random.rand() < 0.6:
        delta = np.random.uniform(-0.2, 0.2)
        clip  = np.clip(clip + delta, -1.0, 1.0)

    # 3. Contrast jitter (50% probability)
    # Scale pixel values around zero — preserves MobileNetV2 normalisation range
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.8, 1.2)
        clip   = np.clip(clip * factor, -1.0, 1.0)

    # 4. Temporal reverse (40% probability)
    # Only for behaviors that are symmetric in time
    if np.random.rand() < 0.4:
        clip = clip[::-1, :, :, :]

    # 5. Gaussian noise (40% probability)
    # Small sigma — barely perceptible, acts as regulariser
    if np.random.rand() < 0.4:
        noise = np.random.normal(0, 0.03, clip.shape).astype(np.float32)
        clip  = np.clip(clip + noise, -1.0, 1.0)

    return clip.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MIXUP (Improvement 2)
# ─────────────────────────────────────────────────────────────────────────────

def mixup_batch(
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    alpha:   float = 0.4,
) -> tuple:
    """
    Apply Mixup augmentation to a batch of clips.

    For each sample i, blend it with a randomly chosen sample j:
        x_mixed = λ * x_i + (1 - λ) * x_j
        y_mixed = λ * y_i + (1 - λ) * y_j
    where λ ~ Beta(alpha, alpha).

    WHY this works:
        Mixup forces the model to learn linear interpolations between classes,
        which smooths the decision boundary. On small datasets this acts as a
        strong regulariser that prevents the model from memorising individual
        clips, improving generalisation to unseen clips.

    Args:
        X_batch : np.ndarray, shape (B, T, H, W, 3) — clip batch.
        y_batch : np.ndarray, shape (B, num_classes) — one-hot (or soft) labels.
        alpha   : Beta distribution parameter. Smaller α → more extreme λ values
                  (more aggressive mixing). 0.4 is a good default for small datasets.

    Returns:
        (X_mixed, y_mixed) — same shapes as inputs.
    """
    batch_size = X_batch.shape[0]

    # Sample mixing coefficient from symmetric Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Random permutation to pick mixing partners
    indices = np.random.permutation(batch_size)

    X_mixed = lam * X_batch + (1 - lam) * X_batch[indices]
    y_mixed = lam * y_batch + (1 - lam) * y_batch[indices]

    return X_mixed.astype(np.float32), y_mixed.astype(np.float32)
