# Running the Improvement Pipeline on Google Colab

Step-by-step guide to run the 7-improvement pipeline on Google Colab with a T4 GPU.

---

## Prerequisites

Before opening Colab, upload the following files to your **Google Drive**:

```
MyDrive/
└── autism_project/
    ├── dataset/
    │   ├── arm_flapping/    ← video files (.mp4, .avi, .mov, .mkv)
    │   ├── head_banging/
    │   ├── spinning/
    │   └── normal/
    ├── autism_mobilenet_gru_phase1.h5   ← trained model checkpoint
    └── label_encoder.pkl                ← fitted LabelEncoder
```

> **NOTE**: If you have a `phase2.h5` file, upload that too. The pipeline
> will prefer it automatically. If only `phase1.h5` exists, it will be
> used as the baseline (it likely contains Phase 2 weights due to a save
> bug that has now been fixed).

---

## Colab Setup (Cell-by-Cell)

### Cell 1 — Mount Drive & Install Dependencies

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install pinned dependencies (Colab may ship a newer TF by default)
!pip install -q tensorflow==2.13.0 opencv-python>=4.8.0 scikit-learn>=1.3.0 \
    matplotlib>=3.7.0 seaborn>=0.12.0 "numpy>=1.24.0,<2.0.0"
```

### Cell 2 — Copy Data to Local SSD (5-10x faster I/O)

```python
import shutil, os

# === CONFIGURE THESE PATHS ===
DRIVE_PROJECT = "/content/drive/MyDrive/autism_project"
LOCAL_DIR      = "/content/autism_run"
# ==============================

os.makedirs(LOCAL_DIR, exist_ok=True)

# Copy dataset (or unzip if you uploaded a .zip)
dataset_src = os.path.join(DRIVE_PROJECT, "dataset")
dataset_dst = os.path.join(LOCAL_DIR, "dataset")

if os.path.exists(dataset_src) and not os.path.exists(dataset_dst):
    print("Copying dataset to local SSD...")
    shutil.copytree(dataset_src, dataset_dst)
    print("Done.")

# If you uploaded a zip instead:
# !unzip -q "{DRIVE_PROJECT}/dataset.zip" -d "{LOCAL_DIR}/"

# Copy model files
for f in ["autism_mobilenet_gru_phase1.h5", "autism_mobilenet_gru_phase2.h5", "label_encoder.pkl"]:
    src = os.path.join(DRIVE_PROJECT, f)
    dst = os.path.join(LOCAL_DIR, f)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"Copied: {f}")

print("\nLocal files:")
!ls -la {LOCAL_DIR}
```

### Cell 3 — Clone Repo & Configure Paths

```python
# Clone the repository (or upload the model/ folder manually)
%cd /content
!git clone https://github.com/YOUR_USERNAME/Autism-video-analysis-.git repo
%cd /content/repo/model

import sys
sys.path.insert(0, "/content/repo/model")

# Override CONFIG paths to point to Colab locations
from config import CONFIG

CONFIG["dataset_path"] = "/content/autism_run/dataset"
CONFIG["output_dir"]   = "/content/autism_run"  # All outputs saved here

print("Config updated:")
print(f"  dataset_path = {CONFIG['dataset_path']}")
print(f"  output_dir   = {CONFIG['output_dir']}")
```

### Cell 4 — Enable T4 GPU + Mixed Precision

```python
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Enable GPU memory growth (prevents TF from grabbing all VRAM at once)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Enable FP16 mixed precision — ~2x speedup on T4
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    print(f"GPU: {gpus[0].name}")
    print(f"Mixed Precision: {mixed_precision.global_policy().name}")
else:
    print("WARNING: No GPU detected! Go to Runtime → Change runtime type → T4 GPU")
```

### Cell 5 — Run the Improvement Pipeline

```python
from improve import main
main()
```

This will:
1. Load videos from the dataset
2. Verify baseline accuracy (≥90%)
3. Run all 7 improvements sequentially
4. Print a results table
5. Save the best model as `autism_final.h5`

### Cell 6 — Save Results Back to Drive

```python
import shutil, glob

DRIVE_OUTPUT = "/content/drive/MyDrive/autism_project/results"
os.makedirs(DRIVE_OUTPUT, exist_ok=True)

# Copy all generated files back to Drive
for pattern in ["*.h5", "*.png", "*.pkl"]:
    for f in glob.glob(os.path.join("/content/autism_run", pattern)):
        dst = os.path.join(DRIVE_OUTPUT, os.path.basename(f))
        shutil.copy2(f, dst)
        print(f"Saved → {dst}")

print(f"\nAll results saved to: {DRIVE_OUTPUT}")
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No GPU detected` | Runtime → Change runtime type → T4 GPU |
| `OOM` during optical flow | Reduce `clips_per_video` in CONFIG to 2 |
| `No saved model found` | Check that `.h5` file was copied to `/content/autism_run/` |
| `Frame extraction failed` | Some videos may be corrupt — check dataset |
| Slow data loading | Make sure you copied to local SSD (Cell 2), not reading from Drive |

---

## Expected Output

After running all cells, you should see:
```
═══════════════════════════════════════════════════════════════════════════
  RESULTS TABLE
═══════════════════════════════════════════════════════════════════════════
  Step | Improvement                    |  Val Acc | Test Acc |    Delta
  ─────────────────────────────────────────────────────────────────────────
     0 | Baseline (loaded .h5)          |        — |   93.80% |        —
     1 | TTA (9 passes)                 |        — |   94.XX% |   +0.XX%
     2 | Mixup fine-tune                |   XX.XX% |   XX.XX% |   +X.XX%
    ...
═══════════════════════════════════════════════════════════════════════════
```
