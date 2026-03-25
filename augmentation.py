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
