"""
trainer.py
==========
Two-phase training strategy for transfer learning, enhanced with:
  - Mixup augmentation         (Improvement 2)
  - Label smoothing            (Improvement 3)
  - Discriminative LRs         (Improvement 4)
  - Cosine annealing schedule  (Improvement 5)
  - Stochastic Weight Averaging (Improvement 6)

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
• Top layers of MobileNetV2 are unfrozen (30 or 60 depending on config).
• Learning rate set low (or discriminative per group).
• Mixup applied to 30% of batches as regularisation.
• Label smoothing prevents overconfident predictions.
• Cosine annealing cycles the LR to escape local minima.
• SWA averages weights from final epochs for flatter minima.
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

from model import unfreeze_top_layers, get_model_summary, get_variable_groups
from augmentation import mixup_batch
from config import output_path


# ─────────────────────────────────────────────────────────────────────────────
# SWA CALLBACK (Improvement 6)
# ─────────────────────────────────────────────────────────────────────────────

class SWACallback(tf.keras.callbacks.Callback):
    """
    Stochastic Weight Averaging callback.

    WHY SWA works:
        Adam often converges to sharp, narrow minima that generalise poorly.
        SWA averages weights from multiple points along the training trajectory,
        landing in a wider, flatter minimum that generalises better.
        Typically adds 0.5–1.5% accuracy on small datasets.

    Collects weight snapshots from the last `swa_epochs` epochs, then
    averages them after training completes.
    """

    def __init__(self, swa_epochs: int = 10):
        """
        Args:
            swa_epochs: Number of final epochs to collect weight snapshots from.
        """
        super().__init__()
        self.swa_epochs = swa_epochs
        self.weight_snapshots = []

    def on_epoch_end(self, epoch, logs=None):
        """Store a copy of the current weights after each epoch."""
        self.weight_snapshots.append(
            [w.numpy().copy() for w in self.model.trainable_weights]
        )
        # Keep only the last swa_epochs snapshots to save memory
        if len(self.weight_snapshots) > self.swa_epochs:
            self.weight_snapshots.pop(0)

    def get_averaged_weights(self):
        """
        Compute element-wise average across all stored snapshots.

        Returns:
            List of np.ndarray — averaged trainable weights.
        """
        if not self.weight_snapshots:
            return None

        n = len(self.weight_snapshots)
        print(f"\n  [SWA] Averaging weights from {n} snapshots...")

        averaged = []
        for i in range(len(self.weight_snapshots[0])):
            # Stack the i-th weight across all snapshots and take the mean
            stacked = np.stack([snap[i] for snap in self.weight_snapshots], axis=0)
            averaged.append(np.mean(stacked, axis=0))

        return averaged


def apply_swa(model, swa_callback, X_train):
    """
    Apply SWA: set averaged weights and re-run BatchNorm statistics.

    WHY re-run BN stats:
        After weight averaging, the BatchNormalization running statistics
        (mean, variance) no longer match the averaged weights. A forward
        pass over the training set recalculates them correctly.

    Args:
        model        : Trained model after Phase 2.
        swa_callback : SWACallback instance with collected snapshots.
        X_train      : Training data for BN stat recalculation.
    """
    averaged = swa_callback.get_averaged_weights()
    if averaged is None:
        print("  [SWA] No snapshots collected — skipping.")
        return

    # Set the averaged trainable weights
    for var, avg_val in zip(model.trainable_weights, averaged):
        var.assign(avg_val)
    print("  [SWA] Averaged weights applied.")

    # Re-run BN statistics — forward pass on training data
    print("  [SWA] Recalculating BatchNorm statistics...")
    # Set BN layers to training mode for stat update
    _ = model(X_train[:min(len(X_train), 64)].astype(np.float32), training=True)
    print("  [SWA] BatchNorm stats updated.")


def apply_swa_dual(model, swa_callback, X_train_rgb, X_train_flow):
    """
    Apply SWA for dual-stream model: set averaged weights and re-run BN stats.

    Same logic as apply_swa but handles the dual-input model format.

    Args:
        model          : Trained dual-stream model after Phase 2.
        swa_callback   : SWACallback instance with collected snapshots.
        X_train_rgb    : RGB training data.
        X_train_flow   : Flow training data.
    """
    averaged = swa_callback.get_averaged_weights()
    if averaged is None:
        print("  [SWA] No snapshots collected — skipping.")
        return

    for var, avg_val in zip(model.trainable_weights, averaged):
        var.assign(avg_val)
    print("  [SWA] Averaged weights applied.")

    print("  [SWA] Recalculating BatchNorm statistics...")
    n = min(len(X_train_rgb), 64)
    _ = model(
        [X_train_rgb[:n].astype(np.float32),
         X_train_flow[:n].astype(np.float32)],
        training=True,
    )
    print("  [SWA] BatchNorm stats updated.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model:         tf.keras.Model,
    X:             np.ndarray,
    y:             np.ndarray,
    config:        dict,
    class_weights: dict | None = None,
    X_flow:        np.ndarray | None = None,
) -> tuple:
    """
    Run Phase 1 (frozen) then Phase 2 (fine-tuned) training.

    Handles both single-stream (RGB only) and dual-stream (RGB + flow) models.

    Args:
        model         : Keras model (single or dual-stream).
        X             : RGB clip data, shape (N, T, H, W, 3).
        y             : One-hot labels, shape (N, num_classes).
        config        : Global CONFIG dict.
        class_weights : Optional per-class weight dict for balanced training.
        X_flow        : Optional optical flow data, shape (N, T-1, fH, fW, 2).

    Returns:
        (history_phase1, history_phase2)
    """
    use_flow = X_flow is not None
    use_mixup = config.get("use_mixup", False)
    use_disc_lr = config.get("discriminative_lr", False)
    use_cosine = config.get("use_cosine_annealing", False)
    use_swa = config.get("use_swa", False)
    label_smooth = config.get("label_smoothing", 0.0)

    # ── Stratified split ──────────────────────────────────────────────────
    y_int = np.argmax(y, axis=1)
    n_samples = len(X)
    indices = np.arange(n_samples)

    idx_tr_full, idx_test = train_test_split(
        indices, test_size=config["test_split"],
        random_state=42, stratify=y_int,
    )
    val_ratio = config["val_split"] / (1.0 - config["test_split"])
    idx_train, idx_val = train_test_split(
        idx_tr_full,
        test_size=val_ratio,
        random_state=42,
        stratify=y_int[idx_tr_full],
    )

    # Split RGB data
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    # Split flow data (if dual-stream)
    X_flow_train = X_flow_val = X_flow_test = None
    if use_flow:
        X_flow_train = X_flow[idx_train]
        X_flow_val   = X_flow[idx_val]
        X_flow_test  = X_flow[idx_test]

    print(f"\n  Split → Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

    cw = class_weights if config.get("use_class_weights") else None

    # Prepare inputs based on model type
    def _make_input(x_rgb, x_flow=None):
        """Create model input — single array or [rgb, flow] list."""
        if use_flow and x_flow is not None:
            return [x_rgb, x_flow]
        return x_rgb

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
    history_p1 = model.fit(
        _make_input(X_train, X_flow_train),
        y_train,
        validation_data=(_make_input(X_val, X_flow_val), y_val),
        epochs=config["phase1_epochs"],
        batch_size=config["batch_size"],
        class_weight=cw,
        callbacks=callbacks_p1,
    )

    # ── PHASE 2: Fine-tune top CNN layers ─────────────────────────────────
    # Determine how many layers to unfreeze
    n_unfreeze = config.get("unfreeze_top_layers_p2", config["unfreeze_top_layers"])
    if use_disc_lr:
        n_unfreeze = config.get("unfreeze_top_layers_p2", 60)

    print("\n" + "═" * 65)
    print(f"  PHASE 2 — Fine-tuning top {n_unfreeze} MobileNetV2 layers")
    if use_mixup:
        print(f"            Mixup α={config.get('mixup_alpha', 0.4)} on 30% of batches")
    if label_smooth > 0:
        print(f"            Label smoothing = {label_smooth}")
    if use_disc_lr:
        print(f"            Discriminative learning rates (3 groups)")
    if use_cosine:
        print(f"            Cosine annealing with warm restarts")
    if use_swa:
        print(f"            SWA over last {config.get('swa_epochs', 10)} epochs")
    print("═" * 65)

    model = unfreeze_top_layers(model, n_unfreeze)

    # ── Build Phase 2 loss ────────────────────────────────────────────────
    # Improvement 3: Label smoothing in Phase 2
    if label_smooth > 0:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smooth
        )
        print(f"  Loss: CategoricalCrossentropy(label_smoothing={label_smooth})")
    else:
        loss_fn = "categorical_crossentropy"

    # ── Build Phase 2 optimizer ───────────────────────────────────────────
    if use_disc_lr:
        # Improvement 4: Discriminative learning rates
        # We use a single optimizer with the head LR as the base.
        # The discriminative rates are applied via custom gradient scaling.
        var_groups = get_variable_groups(model, config)
        base_lr = config.get("lr_head", 1e-4)
    else:
        base_lr = config["phase2_lr"]

    # Improvement 5: Cosine annealing schedule
    if use_cosine:
        steps_per_epoch = max(1, len(X_train) // config["batch_size"])
        first_decay_steps = steps_per_epoch * config.get("cosine_first_decay_epochs", 10)

        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=base_lr,
            first_decay_steps=first_decay_steps,
            t_mul=config.get("cosine_t_mul", 2.0),
            alpha=float(config["min_lr"]) / base_lr,   # min LR as fraction
        )
        optimizer_p2 = optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0,
        )
        print(f"  Optimizer: Adam + CosineDecayRestarts "
              f"(first_cycle={config.get('cosine_first_decay_epochs', 10)} epochs)")
    else:
        optimizer_p2 = optimizers.Adam(
            learning_rate=base_lr,
            clipnorm=1.0,
        )

    # ── Phase 2 training ──────────────────────────────────────────────────
    if use_mixup or use_disc_lr:
        # Custom training loop for Mixup + Discriminative LR
        history_p2 = _custom_train_phase2(
            model, optimizer_p2, loss_fn,
            X_train, y_train, X_val, y_val,
            config, cw, var_groups if use_disc_lr else None,
            X_flow_train, X_flow_val, use_flow,
            use_mixup, use_swa,
        )
    else:
        # Standard model.fit for Phase 2
        model.compile(
            optimizer=optimizer_p2,
            loss=loss_fn,
            metrics=["accuracy"],
        )

        callbacks_p2 = _make_callbacks_p2(config, use_swa)
        history_p2 = model.fit(
            _make_input(X_train, X_flow_train),
            y_train,
            validation_data=(_make_input(X_val, X_flow_val), y_val),
            epochs=config["phase2_epochs"],
            batch_size=config["batch_size"],
            class_weight=cw,
            callbacks=callbacks_p2,
        )

        # Apply SWA if enabled
        if use_swa:
            swa_cb = [cb for cb in callbacks_p2 if isinstance(cb, SWACallback)]
            if swa_cb:
                if use_flow:
                    apply_swa_dual(model, swa_cb[0], X_train, X_flow_train)
                else:
                    apply_swa(model, swa_cb[0], X_train)

    # ── Evaluation ────────────────────────────────────────────────────────
    _evaluate(model, X_test, y_test, config, X_flow_test)
    if hasattr(history_p2, 'history'):
        _plot_combined_history(history_p1, history_p2, config["plot_history"])

    return history_p1, history_p2


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM PHASE 2 LOOP (Mixup + Discriminative LR + SWA)
# ─────────────────────────────────────────────────────────────────────────────

class _SimpleHistory:
    """Minimal history object compatible with plotting."""
    def __init__(self):
        self.history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}


def _custom_train_phase2(
    model, optimizer, loss_fn,
    X_train, y_train, X_val, y_val,
    config, class_weights, var_groups,
    X_flow_train, X_flow_val, use_flow,
    use_mixup, use_swa,
):
    """
    Custom Phase 2 training loop supporting Mixup + Discriminative LR + SWA.

    WHY a custom loop instead of model.fit():
        1. Mixup must be applied per-batch (not to the full dataset upfront)
           and only to 30% of batches.
        2. Discriminative LRs require scaling gradients differently per variable
           group, which is not natively supported by Keras optimizers in TF 2.13.

    Args:
        model          : Model after unfreeze_top_layers().
        optimizer      : Adam optimizer (with or without cosine schedule).
        loss_fn        : Loss function (with or without label smoothing).
        X_train/y_train: Training data.
        X_val/y_val    : Validation data.
        config         : CONFIG dict.
        class_weights  : Per-class weights dict or None.
        var_groups     : Discriminative LR variable groups or None.
        X_flow_train/val: Optical flow data or None.
        use_flow       : Whether dual-stream is active.
        use_mixup      : Whether to apply Mixup.
        use_swa        : Whether to collect SWA snapshots.

    Returns:
        _SimpleHistory object with training curves.
    """
    batch_size = config["batch_size"]
    epochs = config["phase2_epochs"]
    mixup_alpha = config.get("mixup_alpha", 0.4)
    patience = config["early_stop_patience"]

    # Compile model for metric tracking (needed for evaluate)
    if isinstance(loss_fn, str):
        compiled_loss = tf.keras.losses.CategoricalCrossentropy()
    else:
        compiled_loss = loss_fn

    model.compile(optimizer=optimizer, loss=compiled_loss, metrics=["accuracy"])

    history = _SimpleHistory()
    swa_callback = SWACallback(config.get("swa_epochs", 10)) if use_swa else None
    if swa_callback:
        swa_callback.model = model

    best_val_loss = float("inf")
    best_weights = None
    wait = 0

    # Build class weight tensor for loss weighting
    cw_tensor = None
    if class_weights:
        num_classes = y_train.shape[1]
        cw_array = np.ones(num_classes, dtype=np.float32)
        for cls_idx, weight in class_weights.items():
            cw_array[cls_idx] = weight
        cw_tensor = tf.constant(cw_array)

    # Compute LR scale factors for discriminative LR
    lr_scales = None
    if var_groups:
        base_lr = config.get("lr_head", 1e-4)
        lr_scales = {}
        for group_name, group_info in var_groups.items():
            scale = group_info["lr"] / base_lr
            for var in group_info["vars"]:
                lr_scales[var.ref()] = scale

    for epoch in range(epochs):
        # Shuffle training data
        perm = np.random.permutation(len(X_train))
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]
        X_flow_shuf = X_flow_train[perm] if use_flow else None

        n_batches = max(1, len(X_train) // batch_size)
        epoch_loss = 0.0
        epoch_acc = 0.0

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, len(X_train))

            xb = X_shuf[start:end].astype(np.float32)
            yb = y_shuf[start:end].astype(np.float32)
            fb = X_flow_shuf[start:end].astype(np.float32) if use_flow else None

            # Apply Mixup to 30% of batches (Improvement 2)
            if use_mixup and np.random.rand() < 0.3:
                xb, yb = mixup_batch(xb, yb, alpha=mixup_alpha)
                if use_flow and fb is not None:
                    # Also mixup the flow with the same permutation
                    fb_mixed, _ = mixup_batch(fb, yb, alpha=mixup_alpha)
                    fb = fb_mixed

            # Build input
            inp = [xb, fb] if use_flow else xb

            # Custom gradient step
            with tf.GradientTape() as tape:
                preds = model(inp, training=True)
                loss = compiled_loss(yb, preds)

                # Apply class weights manually
                if cw_tensor is not None:
                    sample_weights = tf.reduce_sum(yb * cw_tensor, axis=1)
                    loss = loss * tf.reduce_mean(sample_weights)

                # Add regularisation losses
                if model.losses:
                    loss = loss + tf.add_n(model.losses)

            grads = tape.gradient(loss, model.trainable_variables)

            # Apply discriminative LR scaling (Improvement 4)
            if lr_scales:
                scaled_grads = []
                for g, v in zip(grads, model.trainable_variables):
                    if g is not None:
                        scale = lr_scales.get(v.ref(), 1.0)
                        scaled_grads.append(g * scale)
                    else:
                        scaled_grads.append(g)
                grads = scaled_grads

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track metrics
            epoch_loss += float(loss)
            matches = tf.equal(tf.argmax(preds, 1), tf.argmax(yb, 1))
            epoch_acc += float(tf.reduce_mean(tf.cast(matches, tf.float32)))

        epoch_loss /= n_batches
        epoch_acc /= n_batches

        # Validation
        val_inp = [X_val.astype(np.float32), X_flow_val.astype(np.float32)] if use_flow else X_val.astype(np.float32)
        val_results = model.evaluate(val_inp, y_val, verbose=0)
        val_loss, val_acc = val_results[0], val_results[1]

        history.history["loss"].append(epoch_loss)
        history.history["accuracy"].append(epoch_acc)
        history.history["val_loss"].append(val_loss)
        history.history["val_accuracy"].append(val_acc)

        print(f"  Epoch {epoch+1:>3}/{epochs}  |  "
              f"loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}  |  "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")

        # SWA snapshot collection
        if swa_callback:
            swa_callback.on_epoch_end(epoch)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = [w.numpy().copy() for w in model.weights]
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} "
                      f"(patience={patience})")
                break

    # Restore best weights (before SWA)
    if best_weights:
        for w, bw in zip(model.weights, best_weights):
            w.assign(bw)
        print("  Restored best weights from training.")

    # Apply SWA (Improvement 6)
    if use_swa and swa_callback:
        if use_flow:
            apply_swa_dual(model, swa_callback, X_train, X_flow_train)
        else:
            apply_swa(model, swa_callback, X_train)

    # Save best Phase 2 model
    model.save(output_path(config, config["phase2_save_name"]))
    print(f"  Phase 2 model saved → {output_path(config, config['phase2_save_name'])}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_callbacks(config: dict, suffix: str = "") -> list:
    """Build the standard callback set for Phase 1."""
    ckpt_path = output_path(config, config["phase1_save_name"])
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
        # Save best Phase 1 checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]


def _make_callbacks_p2(config: dict, use_swa: bool = False) -> list:
    """
    Build callbacks for Phase 2 (standard model.fit path).

    Phase 2 uses cosine annealing instead of ReduceLROnPlateau when enabled.
    SWA callback is added if enabled.
    """
    ckpt_path = output_path(config, config["phase2_save_name"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["early_stop_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        # Save best Phase 2 checkpoint (separate from Phase 1)
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]

    # Only use ReduceLROnPlateau if NOT using cosine annealing
    if not config.get("use_cosine_annealing", False):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=config["lr_reduce_factor"],
                patience=config["lr_reduce_patience"],
                min_lr=config["min_lr"],
                verbose=1,
            )
        )

    # Add SWA callback
    if use_swa:
        callbacks.append(SWACallback(config.get("swa_epochs", 10)))

    return callbacks


def _evaluate(
    model:   tf.keras.Model,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    config:  dict,
    X_flow_test: np.ndarray | None = None,
) -> None:
    """Full evaluation: loss, accuracy, classification report, confusion matrix."""
    print("\n" + "═" * 65)
    print("  Final Test-set Evaluation")
    print("═" * 65)

    use_flow = X_flow_test is not None
    test_input = [X_test, X_flow_test] if use_flow else X_test

    loss, acc = model.evaluate(test_input, y_test, verbose=0)
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc * 100:.2f}%")

    pred_probs = model.predict(test_input, verbose=0)
    y_pred = np.argmax(pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n  Classification Report:\n")
    print(classification_report(
        y_true, y_pred,
        target_names=config["classes"],
        zero_division=0,
    ))

    # Save both standard and final confusion matrices
    _plot_confusion_matrix(y_true, y_pred, config["classes"],
                           output_path(config, config["plot_confusion"]))
    _plot_confusion_matrix(y_true, y_pred, config["classes"],
                           output_path(config, "confusion_matrix_final.png"))


def _plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Generate and save a heatmap confusion matrix."""
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
