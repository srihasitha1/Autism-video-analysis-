"""
trainer.py
==========
Two-phase training strategy for transfer learning.

Phase 1 — HEAD TRAINING  (frozen CNN base)
───────────────────────────────────────────
• Only the GRU layers and classification head are trained.
• High learning rate (1e-3) is safe here — pretrained weights are frozen.
• Runs for phase1_epochs or until early stopping.
• Goal: converge the temporal head before touching CNN weights.

WHY this ordering matters:
  If we immediately fine-tune with a high LR, the random GRU weights will
  produce large gradients that propagate back into and destroy the pretrained
  CNN features. Stabilising the head first prevents this.

Phase 2 — FINE-TUNING  (top CNN layers unfrozen)
──────────────────────────────────────────────────
• Top `unfreeze_top_layers` layers of MobileNetV2 are unfrozen.
• Learning rate drops to 1e-4 (10× lower than Phase 1).
• WHY low LR: pretrained weights are close to a good solution already.
              Large updates would overwrite them, losing ImageNet knowledge.
• BatchNormalization layers in the base stay frozen throughout.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from model import unfreeze_top_layers, get_model_summary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model:         tf.keras.Model,
    X:             np.ndarray,
    y:             np.ndarray,
    config:        dict,
    class_weights: dict | None = None,
) -> tuple[tf.keras.callbacks.History, tf.keras.callbacks.History]:
    """
    Run Phase 1 (frozen) then Phase 2 (fine-tuned) training.

    Returns:
        (history_phase1, history_phase2)
    """
    # ── Stratified split ──────────────────────────────────────────────────
    y_int = np.argmax(y, axis=1)

    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=config["test_split"],
        random_state=42, stratify=y_int,
    )
    val_ratio = config["val_split"] / (1.0 - config["test_split"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full,
        test_size=val_ratio,
        random_state=42,
        stratify=np.argmax(y_tr_full, axis=1),
    )
    print(f"\n  Split → Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

    cw = class_weights if config.get("use_class_weights") else None

    # ── PHASE 1: Train head only ──────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  PHASE 1 — Training GRU head  (MobileNetV2 frozen)")
    print("═" * 65)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=config["phase1_lr"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    get_model_summary(model)

    callbacks_p1 = _make_callbacks(config, suffix="_phase1")
    history_p1   = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["phase1_epochs"],
        batch_size=config["batch_size"],
        class_weight=cw,
        callbacks=callbacks_p1,
    )

    # ── PHASE 2: Fine-tune top CNN layers ─────────────────────────────────
    print("\n" + "═" * 65)
    print(f"  PHASE 2 — Fine-tuning top {config['unfreeze_top_layers']} "
          f"MobileNetV2 layers")
    print("═" * 65)

    model = unfreeze_top_layers(model, config["unfreeze_top_layers"])

    # CRITICAL: recompile after changing trainability
    # WHY very low LR: preserves pretrained weights; we only nudge them.
    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=config["phase2_lr"],
            clipnorm=1.0,       # gradient clipping — extra safety for fine-tuning
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_p2 = _make_callbacks(config, suffix="_phase2")
    history_p2   = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["phase2_epochs"],
        batch_size=config["batch_size"],
        class_weight=cw,
        callbacks=callbacks_p2,
    )

    # ── Evaluation ────────────────────────────────────────────────────────
    _evaluate(model, X_test, y_test, config)
    _plot_combined_history(history_p1, history_p2, config["plot_history"])

    return history_p1, history_p2


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_callbacks(config: dict, suffix: str = "") -> list:
    """Build the standard callback set for one training phase."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["early_stop_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config["lr_reduce_factor"],
            patience=config["lr_reduce_patience"],
            min_lr=config["min_lr"],
            verbose=1,
        ),
        # Save best checkpoint per phase
        tf.keras.callbacks.ModelCheckpoint(
            "autism_mobilenet_gru_phase1.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]


def _evaluate(
    model:   tf.keras.Model,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    config:  dict,
) -> None:
    """Full evaluation: loss, accuracy, classification report, confusion matrix."""
    print("\n" + "═" * 65)
    print("  Final Test-set Evaluation")
    print("═" * 65)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n  Classification Report:\n")
    print(classification_report(
        y_true, y_pred,
        target_names=config["classes"],
        zero_division=0,
    ))

    _plot_confusion_matrix(y_true, y_pred, config["classes"], config["plot_confusion"])


def _plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Saved: {save_path}")


def _plot_combined_history(h1, h2, save_path):
    """Plot Phase 1 and Phase 2 histories on the same axes with a phase divider."""
    p1_len = len(h1.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in [
        (ax1, "accuracy", "Accuracy"),
        (ax2, "loss",     "Loss"),
    ]:
        p1_train = h1.history[metric]
        p1_val   = h1.history[f"val_{metric}"]
        p2_train = h2.history[metric]
        p2_val   = h2.history[f"val_{metric}"]

        xs1 = range(1, p1_len + 1)
        xs2 = range(p1_len + 1, p1_len + len(p2_train) + 1)

        ax.plot(xs1, p1_train, "b-",  lw=2, label="Train (P1)")
        ax.plot(xs1, p1_val,   "b--", lw=2, label="Val   (P1)")
        ax.plot(xs2, p2_train, "r-",  lw=2, label="Train (P2)")
        ax.plot(xs2, p2_val,   "r--", lw=2, label="Val   (P2)")

        ax.axvline(x=p1_len + 0.5, color="gray", linestyle=":", lw=1.5,
                   label="Fine-tune starts")
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("MobileNetV2 + GRU — Training History\n"
                 "Blue = Phase 1 (frozen), Red = Phase 2 (fine-tuned)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
