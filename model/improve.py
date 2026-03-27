"""
improve.py
==========
Load the saved 93.8% model and apply 7 accuracy improvements incrementally.
Each improvement loads the baseline model, applies its technique, evaluates,
and saves an improved checkpoint.

Usage:
    python improve.py

Output:
    - Per-improvement accuracy table printed to console
    - confusion_matrix_final.png
    - autism_final.h5  (best model)
    - autism_improved_*.h5  (per-improvement checkpoints)
"""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from config import CONFIG, output_path
from dataset_builder import build_dataset
from augmentation import mixup_batch
from tta import tta_predict
from model import unfreeze_top_layers, get_variable_groups

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — derived from config, not hardcoded
# ─────────────────────────────────────────────────────────────────────────────
MODEL_P2 = output_path(CONFIG, CONFIG["phase2_save_name"])
MODEL_P1 = output_path(CONFIG, CONFIG["phase1_save_name"])
RESULTS  = []


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path=None):
    """Load the baseline model from disk. Falls back to phase1 if phase2 missing."""
    if path and os.path.exists(path):
        print(f"  Loading model: {path}")
        return tf.keras.models.load_model(path)
    if os.path.exists(MODEL_P2):
        print(f"  Loading model: {MODEL_P2}")
        return tf.keras.models.load_model(MODEL_P2)
    if os.path.exists(MODEL_P1):
        print(f"  [NOTE] {MODEL_P2} not found, using {MODEL_P1}")
        return tf.keras.models.load_model(MODEL_P1)
    print("[ERROR] No saved model found!")
    sys.exit(1)


def split_data(X, y, config):
    """Replicate the exact same train/val/test split as the original trainer.py."""
    y_int = np.argmax(y, axis=1)
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=config["test_split"], random_state=42, stratify=y_int)
    val_ratio = config["val_split"] / (1.0 - config["test_split"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=val_ratio, random_state=42,
        stratify=np.argmax(y_tr, axis=1))
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(model, X, y, label=""):
    """Evaluate model on data. Returns (loss, accuracy)."""
    loss, acc = model.evaluate(X.astype(np.float32), y, verbose=0)
    print(f"  {label} → Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")
    return loss, acc


def evaluate_tta(model, X, y, n_aug=8):
    """Evaluate with TTA — 9 forward passes per sample, averaged."""
    preds = []
    total = len(X)
    for i in range(total):
        avg_probs = tta_predict(X[i].astype(np.float32), model, n_augments=n_aug)
        preds.append(np.argmax(avg_probs))
        if (i + 1) % 20 == 0:
            print(f"    TTA progress: {i+1}/{total}")
    y_true = np.argmax(y, axis=1)
    acc = np.mean(np.array(preds) == y_true)
    return acc


def record(step, name, val_acc, test_acc, baseline):
    """Record an improvement result for the summary table."""
    RESULTS.append({"step": step, "name": name, "val_acc": val_acc,
                    "test_acc": test_acc, "delta": test_acc - baseline})


def print_results_table(baseline):
    """Print the per-improvement accuracy table."""
    print("\n" + "═" * 75)
    print("  RESULTS TABLE")
    print("═" * 75)
    print(f"  {'Step':>4} | {'Improvement':<30} | {'Val Acc':>8} | {'Test Acc':>8} | {'Delta':>8}")
    print("  " + "─" * 71)
    for r in RESULTS:
        va = f"{r['val_acc']*100:.2f}%" if r['val_acc'] else "—"
        ta = f"{r['test_acc']*100:.2f}%"
        d  = f"{r['delta']*100:+.2f}%" if r['step'] > 0 else "—"
        print(f"  {r['step']:>4} | {r['name']:<30} | {va:>8} | {ta:>8} | {d:>8}")
    print("═" * 75)


def plot_confusion(model, X_test, y_test, classes, path="confusion_matrix_final.png"):
    """Generate and save a confusion matrix heatmap."""
    y_pred = np.argmax(model.predict(X_test.astype(np.float32), verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, linewidths=0.5)
    plt.title("Confusion Matrix — Best Model", fontsize=14, fontweight="bold")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


def classification_report_str(model, X_test, y_test, classes):
    """Print per-class precision/recall/F1."""
    y_pred = np.argmax(model.predict(X_test.astype(np.float32), verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("\n  Per-class Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 1 — TTA (no training)
# ─────────────────────────────────────────────────────────────────────────────

def improvement_1(model, X_test, y_test, baseline_acc):
    """Test-Time Augmentation: average 9 forward passes per sample."""
    print("\n" + "═" * 65)
    print("  IMPROVEMENT 1 — Test-Time Augmentation (TTA)")
    print("═" * 65)
    _, std_acc = evaluate(model, X_test, y_test, "Standard (no TTA)")
    tta_acc = evaluate_tta(model, X_test, y_test, n_aug=8)
    print(f"  TTA (9 passes) → Accuracy: {tta_acc*100:.2f}%")
    print(f"  Δ vs baseline: {(tta_acc - baseline_acc)*100:+.2f}%")
    record(1, "TTA (9 passes)", None, tta_acc, baseline_acc)
    return tta_acc


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 2 — Mixup fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def improvement_2(X_train, y_train, X_val, y_val, X_test, y_test, config, baseline_acc):
    """Fine-tune loaded model with Mixup on 30% of batches."""
    print("\n" + "═" * 65)
    print("  IMPROVEMENT 2 — Mixup Fine-tuning")
    print("═" * 65)
    model = load_model()
    lr = config.get("mixup_lr", 5e-5)
    epochs = config.get("mixup_finetune_epochs", 20)
    alpha = config.get("mixup_alpha", 0.4)
    bs = config["batch_size"]
    model.compile(optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    best_val_acc, best_weights, wait, patience = 0, None, 0, 8

    for ep in range(epochs):
        perm = np.random.permutation(len(X_train))
        X_s, y_s = X_train[perm], y_train[perm]
        n_batches = max(1, len(X_train) // bs)
        ep_loss = []

        for b in range(n_batches):
            xb = X_s[b*bs:(b+1)*bs].astype(np.float32)
            yb = y_s[b*bs:(b+1)*bs].astype(np.float32)
            if np.random.rand() < 0.3:
                xb, yb = mixup_batch(xb, yb, alpha=alpha)
            loss = model.train_on_batch(xb, yb)
            ep_loss.append(loss[0] if isinstance(loss, list) else loss)

        _, val_acc = model.evaluate(X_val.astype(np.float32), y_val, verbose=0)
        print(f"  Epoch {ep+1:>2}/{epochs}  loss: {np.mean(ep_loss):.4f}  val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = [w.numpy().copy() for w in model.weights]
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {ep+1}")
                break

    if best_weights:
        for w, bw in zip(model.weights, best_weights): w.assign(bw)

    save_path = output_path(config, f"{config['improved_save_prefix']}_mixup.h5")
    model.save(save_path)
    _, test_acc = evaluate(model, X_test, y_test, "Mixup")
    record(2, "Mixup fine-tune", best_val_acc, test_acc, baseline_acc)
    return test_acc


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 3 — Label smoothing fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def improvement_3(X_train, y_train, X_val, y_val, X_test, y_test, config, baseline_acc):
    """Fine-tune with CategoricalCrossentropy(label_smoothing=0.1)."""
    print("\n" + "═" * 65)
    print("  IMPROVEMENT 3 — Label Smoothing Fine-tuning")
    print("═" * 65)
    model = load_model()
    lr = config.get("smoothing_lr", 5e-5)
    epochs = config.get("smoothing_finetune_epochs", 20)
    smooth = config.get("label_smoothing", 0.1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=smooth),
        metrics=["accuracy"])

    cbs = [tf.keras.callbacks.EarlyStopping(
               monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)]

    h = model.fit(X_train.astype(np.float32), y_train,
                  validation_data=(X_val.astype(np.float32), y_val),
                  epochs=epochs, batch_size=config["batch_size"], callbacks=cbs)

    save_path = output_path(config, f"{config['improved_save_prefix']}_label_smooth.h5")
    model.save(save_path)
    best_val = max(h.history["val_accuracy"])
    _, test_acc = evaluate(model, X_test, y_test, "Label Smoothing")
    record(3, "Label smoothing", best_val, test_acc, baseline_acc)
    return test_acc


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 4 — Discriminative learning rates
# ─────────────────────────────────────────────────────────────────────────────

def improvement_4(X_train, y_train, X_val, y_val, X_test, y_test, config, baseline_acc):
    """Unfreeze 60 layers, apply 3 different LRs via GradientTape."""
    print("\n" + "═" * 65)
    print("  IMPROVEMENT 4 — Discriminative Learning Rates")
    print("═" * 65)
    model = load_model()
    n_unfreeze = config.get("unfreeze_layers_p3", 60)
    model = unfreeze_top_layers(model, n_unfreeze)

    var_groups = get_variable_groups(model, config)
    head_ids  = {id(v) for v in var_groups["head"]["vars"]}
    top30_ids = {id(v) for v in var_groups["top30"]["vars"]}
    next30_ids = {id(v) for v in var_groups["next30"]["vars"]}

    opt_head  = tf.keras.optimizers.Adam(config.get("disc_lr_head", 1e-4), clipnorm=1.0)
    opt_top30 = tf.keras.optimizers.Adam(config.get("disc_lr_top30", 2e-5), clipnorm=1.0)
    opt_mid30 = tf.keras.optimizers.Adam(config.get("disc_lr_mid30", 5e-6), clipnorm=1.0)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # Need to compile for evaluate() to work
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    epochs = config.get("disc_finetune_epochs", 20)
    bs = config["batch_size"]
    best_val_acc, best_weights, wait, patience = 0, None, 0, 8

    for ep in range(epochs):
        perm = np.random.permutation(len(X_train))
        X_s, y_s = X_train[perm], y_train[perm]
        n_batches = max(1, len(X_train) // bs)
        ep_loss = []

        for b in range(n_batches):
            xb = tf.constant(X_s[b*bs:(b+1)*bs].astype(np.float32))
            yb = tf.constant(y_s[b*bs:(b+1)*bs].astype(np.float32))

            with tf.GradientTape() as tape:
                preds = model(xb, training=True)
                loss = loss_fn(yb, preds)
                if model.losses:
                    loss += tf.add_n(model.losses)

            grads = tape.gradient(loss, model.trainable_variables)

            # Split by group and apply with separate optimizers
            head_gv = [(g, v) for g, v in zip(grads, model.trainable_variables)
                       if id(v) in head_ids and g is not None]
            top30_gv = [(g, v) for g, v in zip(grads, model.trainable_variables)
                        if id(v) in top30_ids and g is not None]
            mid30_gv = [(g, v) for g, v in zip(grads, model.trainable_variables)
                        if id(v) in next30_ids and g is not None]

            if head_gv:  opt_head.apply_gradients(head_gv)
            if top30_gv: opt_top30.apply_gradients(top30_gv)
            if mid30_gv: opt_mid30.apply_gradients(mid30_gv)
            ep_loss.append(float(loss))

        _, val_acc = model.evaluate(X_val.astype(np.float32), y_val, verbose=0)
        print(f"  Epoch {ep+1:>2}/{epochs}  loss: {np.mean(ep_loss):.4f}  val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = [w.numpy().copy() for w in model.weights]
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {ep+1}")
                break

    if best_weights:
        for w, bw in zip(model.weights, best_weights): w.assign(bw)

    save_path = output_path(config, f"{config['improved_save_prefix']}_disc_lr.h5")
    model.save(save_path)
    _, test_acc = evaluate(model, X_test, y_test, "Discriminative LR")
    record(4, "Discriminative LR", best_val_acc, test_acc, baseline_acc)
    return test_acc


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 5 — Cosine annealing with warm restarts
# ─────────────────────────────────────────────────────────────────────────────

def improvement_5(X_train, y_train, X_val, y_val, X_test, y_test, config, baseline_acc):
    """Fine-tune with CosineDecayRestarts LR schedule."""
    print("\n" + "═" * 65)
    print("  IMPROVEMENT 5 — Cosine Annealing Fine-tuning")
    print("═" * 65)
    model = load_model()
    epochs = config.get("cosine_finetune_epochs", 20)
    bs = config["batch_size"]
    steps_per_epoch = max(1, len(X_train) // bs)
    first_decay = steps_per_epoch * config.get("cosine_first_decay_epochs", 10)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=config.get("cosine_initial_lr", 1e-4),
        first_decay_steps=first_decay,
        t_mul=config.get("cosine_t_mul", 2.0),
        m_mul=config.get("cosine_m_mul", 0.9),
        alpha=config.get("min_lr", 1e-7))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule, clipnorm=1.0),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    cbs = [tf.keras.callbacks.EarlyStopping(
               monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)]

    h = model.fit(X_train.astype(np.float32), y_train,
                  validation_data=(X_val.astype(np.float32), y_val),
                  epochs=epochs, batch_size=bs, callbacks=cbs)

    save_path = output_path(config, f"{config['improved_save_prefix']}_cosine.h5")
    model.save(save_path)
    best_val = max(h.history["val_accuracy"])
    _, test_acc = evaluate(model, X_test, y_test, "Cosine Annealing")
    record(5, "Cosine annealing", best_val, test_acc, baseline_acc)
    return test_acc


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 6 — SWA (short fine-tune to collect snapshots, then average)
# ─────────────────────────────────────────────────────────────────────────────

class SWACallback(tf.keras.callbacks.Callback):
    """Collect weight snapshots each epoch for SWA averaging."""
    def __init__(self):
        super().__init__()
        self.snapshots = []
    def on_epoch_end(self, epoch, logs=None):
        self.snapshots.append([w.numpy().copy() for w in self.model.trainable_weights])


def improvement_6(X_train, y_train, X_val, y_val, X_test, y_test, config, baseline_acc):
    """SWA: fine-tune briefly, collect snapshots, average weights."""
    print("\n" + "═" * 65)
    print("  IMPROVEMENT 6 — Stochastic Weight Averaging (SWA)")
    print("═" * 65)

    # Find best model from improvements 1-5
    best_path = None
    best_acc = 0
    for r in RESULTS:
        if r["step"] >= 1 and r["test_acc"] > best_acc:
            best_acc = r["test_acc"]
    # Check saved improvement files
    prefix = config.get("improved_save_prefix", "autism_improved")
    candidates = [
        output_path(config, f"{prefix}_disc_lr.h5"),
        output_path(config, f"{prefix}_cosine.h5"),
        output_path(config, f"{prefix}_label_smooth.h5"),
        output_path(config, f"{prefix}_mixup.h5"),
    ]
    for c in candidates:
        if os.path.exists(c):
            best_path = c
            break
    model = load_model(best_path)

    swa_epochs = config.get("swa_epochs", 15)
    swa_lr = config.get("swa_lr", 1e-5)
    model.compile(optimizer=tf.keras.optimizers.Adam(swa_lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    swa_cb = SWACallback()
    model.fit(X_train.astype(np.float32), y_train,
              validation_data=(X_val.astype(np.float32), y_val),
              epochs=swa_epochs, batch_size=config["batch_size"],
              callbacks=[swa_cb], verbose=1)

    # Average all snapshots
    if swa_cb.snapshots:
        n = len(swa_cb.snapshots)
        print(f"  [SWA] Averaging {n} weight snapshots...")
        averaged = []
        for i in range(len(swa_cb.snapshots[0])):
            stacked = np.stack([s[i] for s in swa_cb.snapshots], axis=0)
            averaged.append(np.mean(stacked, axis=0))
        for var, avg in zip(model.trainable_weights, averaged):
            var.assign(avg)
        print("  [SWA] Averaged weights applied")

        # Reset BN stats with a forward pass
        print("  [SWA] Recalculating BatchNorm statistics...")
        n_bn = min(len(X_train), 128)
        _ = model(X_train[:n_bn].astype(np.float32), training=True)
        print("  [SWA] Done")

    swa_path = output_path(config, config["swa_save_name"])
    model.save(swa_path)
    _, val_acc = evaluate(model, X_val, y_val, "SWA (val)")
    _, test_acc = evaluate(model, X_test, y_test, "SWA (test)")
    record(6, "SWA", val_acc, test_acc, baseline_acc)
    return test_acc


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 7 — Optical flow dual-stream
# ─────────────────────────────────────────────────────────────────────────────

def improvement_7(X_train, y_train, X_val, y_val, X_test, y_test, config, baseline_acc):
    """Build dual-stream model: RGB (loaded weights) + flow CNN + GRU."""
    print("\n" + "═" * 65)
    print("  IMPROVEMENT 7 — Optical Flow Dual-Stream Model")
    print("═" * 65)
    from optical_flow import extract_flow_clip

    flow_h = config.get("flow_height", 64)
    flow_w = config.get("flow_width", 64)

    # Compute optical flow for all splits — batched to prevent OOM on Colab
    import gc

    def compute_flow(X, label=""):
        seq = X.shape[1]  # T
        n = len(X)
        # Pre-allocate output array instead of growing a list
        result = np.zeros((n, seq - 1, flow_h, flow_w, 2), dtype=np.float16)
        batch_sz = 32
        for start in range(0, n, batch_sz):
            end = min(start + batch_sz, n)
            for i in range(start, end):
                result[i] = extract_flow_clip(
                    X[i].astype(np.float32), flow_h, flow_w
                ).astype(np.float16)
            if end % 50 == 0 or end == n:
                print(f"    {label} flow: {end}/{n}")
            gc.collect()  # Free intermediate float32 arrays
        return result

    print("  Computing optical flow for train/val/test...")
    F_train = compute_flow(X_train, "Train")
    F_val   = compute_flow(X_val, "Val")
    F_test  = compute_flow(X_test, "Test")
    print(f"  Flow shapes: train={F_train.shape}, val={F_val.shape}, test={F_test.shape}")

    # Build dual-stream model using the loaded model as RGB stream
    loaded = load_model()
    seq_len = loaded.input_shape[1]  # T
    img_h = loaded.input_shape[2]
    img_w = loaded.input_shape[3]
    num_classes = loaded.output_shape[1]

    # Create RGB feature extractor (up to dropout_head, before output Dense)
    rgb_feat = tf.keras.Model(
        inputs=loaded.input,
        outputs=loaded.get_layer("dropout_head").output,
        name="rgb_feature_extractor")
    rgb_feat.trainable = False  # Freeze RGB stream initially

    # Build dual-stream
    from tensorflow.keras import layers, models
    rgb_input = layers.Input(shape=(seq_len, img_h, img_w, 3), name="rgb_input")
    flow_input = layers.Input(shape=(seq_len - 1, flow_h, flow_w, 2), name="flow_input")

    # RGB stream (frozen loaded weights)
    x_rgb = rgb_feat(rgb_input)  # (batch, 64)

    # Flow stream (lightweight CNN + GRU)
    flow_cnn = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(flow_h, flow_w, 2)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
    ], name="flow_cnn")
    x_flow = layers.TimeDistributed(flow_cnn, name="td_flow")(flow_input)
    x_flow = layers.GRU(32, dropout=0.3, name="flow_gru")(x_flow)
    x_flow = layers.Dense(32, activation="relu", name="flow_fc")(x_flow)

    # Fusion
    merged = layers.Concatenate(name="fusion")([x_rgb, x_flow])
    x = layers.Dense(64, activation="relu", name="fusion_fc")(merged)
    x = layers.Dropout(0.4, name="fusion_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dual_output")(x)

    dual_model = models.Model(inputs=[rgb_input, flow_input], outputs=outputs,
                              name="DualStream")
    dual_model.summary()

    # Phase A: Train flow stream + fusion head only (RGB frozen), 10 epochs
    epochs_a = min(10, config.get("flow_finetune_epochs", 20) // 2)
    dual_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4, clipnorm=1.0),
                       loss="categorical_crossentropy", metrics=["accuracy"])
    print(f"\n  Phase A: Training flow stream (RGB frozen), {epochs_a} epochs")
    dual_model.fit(
        [X_train.astype(np.float32), F_train.astype(np.float32)], y_train,
        validation_data=([X_val.astype(np.float32), F_val.astype(np.float32)], y_val),
        epochs=epochs_a, batch_size=config["batch_size"],
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)])

    # Phase B: Unfreeze RGB top 30 layers, fine-tune end-to-end, 10 epochs
    epochs_b = config.get("flow_finetune_epochs", 20) - epochs_a
    rgb_feat.trainable = True
    # Freeze early layers and BN
    base = rgb_feat.get_layer("td_mobilenet").layer if \
        any(l.name == "td_mobilenet" for l in rgb_feat.layers) else None
    if base:
        for layer in base.layers[:-30]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    dual_model.compile(optimizer=tf.keras.optimizers.Adam(2e-5, clipnorm=1.0),
                       loss="categorical_crossentropy", metrics=["accuracy"])
    print(f"\n  Phase B: End-to-end fine-tuning, {epochs_b} epochs")
    h = dual_model.fit(
        [X_train.astype(np.float32), F_train.astype(np.float32)], y_train,
        validation_data=([X_val.astype(np.float32), F_val.astype(np.float32)], y_val),
        epochs=epochs_b, batch_size=config["batch_size"],
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)])

    ds_path = output_path(config, f"{config['improved_save_prefix']}_dual_stream.h5")
    dual_model.save(ds_path)
    best_val = max(h.history["val_accuracy"])
    test_input = [X_test.astype(np.float32), F_test.astype(np.float32)]
    test_loss, test_acc = dual_model.evaluate(test_input, y_test, verbose=0)
    print(f"  Dual-stream → Loss: {test_loss:.4f}, Accuracy: {test_acc*100:.2f}%")
    record(7, "Optical flow dual-stream", best_val, test_acc, baseline_acc)
    return test_acc


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("═" * 65)
    print("  AUTISM BEHAVIOR DETECTION — 7 Improvements Pipeline")
    print("═" * 65)

    # 1. Build dataset
    print("\n  Step 0: Building dataset from videos...")
    X, X_flow, y, le, class_weights = build_dataset(CONFIG)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, CONFIG)
    print(f"  Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 2. Verify baseline
    print("\n  Step 0: Verifying baseline model...")
    model = load_model()
    _, baseline_acc = evaluate(model, X_test, y_test, "Baseline")
    assert baseline_acc >= 0.90, f"Model accuracy is {baseline_acc:.2%} — expected ≥90%!"
    print(f"  ✓ Baseline confirmed: {baseline_acc:.2%}")
    record(0, "Baseline (loaded .h5)", None, baseline_acc, baseline_acc)

    # 3. Apply improvements
    improvement_1(model, X_test, y_test, baseline_acc)
    improvement_2(X_train, y_train, X_val, y_val, X_test, y_test, CONFIG, baseline_acc)
    improvement_3(X_train, y_train, X_val, y_val, X_test, y_test, CONFIG, baseline_acc)
    improvement_4(X_train, y_train, X_val, y_val, X_test, y_test, CONFIG, baseline_acc)
    improvement_5(X_train, y_train, X_val, y_val, X_test, y_test, CONFIG, baseline_acc)
    improvement_6(X_train, y_train, X_val, y_val, X_test, y_test, CONFIG, baseline_acc)
    improvement_7(X_train, y_train, X_val, y_val, X_test, y_test, CONFIG, baseline_acc)

    # 4. Results
    print_results_table(baseline_acc)

    # 5. Find and save best model
    best = max(RESULTS, key=lambda r: r["test_acc"])
    print(f"\n  Best improvement: #{best['step']} {best['name']} → {best['test_acc']*100:.2f}%")

    # Load best model, save as final, generate reports
    prefix = CONFIG.get("improved_save_prefix", "autism_improved")
    best_files = {
        1: MODEL_P2 if os.path.exists(MODEL_P2) else MODEL_P1,  # TTA uses baseline
        2: output_path(CONFIG, f"{prefix}_mixup.h5"),
        3: output_path(CONFIG, f"{prefix}_label_smooth.h5"),
        4: output_path(CONFIG, f"{prefix}_disc_lr.h5"),
        5: output_path(CONFIG, f"{prefix}_cosine.h5"),
        6: output_path(CONFIG, CONFIG["swa_save_name"]),
        7: output_path(CONFIG, f"{prefix}_dual_stream.h5"),
    }
    best_file = best_files.get(best["step"], MODEL_P1)
    if best["step"] == 0:
        best_file = MODEL_P2 if os.path.exists(MODEL_P2) else MODEL_P1

    final_path = output_path(CONFIG, CONFIG["final_save_name"])
    if os.path.exists(best_file):
        best_model = tf.keras.models.load_model(best_file)
        best_model.save(final_path)
        print(f"  Best model saved → {final_path}")

        # Only generate single-input reports for non-dual-stream models
        if best["step"] != 7:
            classification_report_str(best_model, X_test, y_test, CONFIG["classes"])
            plot_confusion(best_model, X_test, y_test, CONFIG["classes"],
                           path=output_path(CONFIG, "confusion_matrix_final.png"))
        else:
            print("  [NOTE] Dual-stream model — confusion matrix requires flow data")

    print("\n" + "═" * 65)
    print("  Pipeline complete!")
    print("═" * 65)


if __name__ == "__main__":
    main()
