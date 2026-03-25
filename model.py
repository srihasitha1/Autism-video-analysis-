"""
model.py
========
MobileNetV2 (pretrained ImageNet) + GRU — Transfer Learning architecture.

WHY this architecture beats a custom 3D CNN on small datasets
──────────────────────────────────────────────────────────────
A custom 3D CNN trained from scratch on ~600 samples must learn:
  • low-level features  (edges, textures)
  • mid-level features  (body parts, limbs)
  • high-level features (gestures)
  • temporal patterns
...all at once, with almost no data.

MobileNetV2 pretrained on ImageNet already encodes low-to-mid-level
visual features extremely well. We only need to teach it:
  • high-level autism behavior features  (a few Dense/GRU layers)
  • temporal patterns across frames      (GRU)

This reduces the learning problem from ~4.5M parameters → ~200K trainable
parameters in Phase 1, and ~800K in Phase 2 — much more tractable.

Architecture
────────────
  Input  : (batch, T=20, 224, 224, 3)
  ↓ TimeDistributed(MobileNetV2 base, include_top=False, pooling='avg')
    → (batch, T, 1280)          1280 = MobileNetV2 final feature dim
  ↓ TimeDistributed(Dense 256, relu, L2)   — project to smaller space
  ↓ TimeDistributed(BatchNorm + Dropout)
  ↓ Bidirectional GRU (128 units, return_sequences=True)
    WHY Bidirectional: sees motion in both temporal directions — useful for
    cyclic behaviors like spinning/flapping where the pattern is symmetric.
  ↓ GRU (64 units, return_sequences=False)
  ↓ Dense(64, relu, L2) → BatchNorm → Dropout(0.5)
  ↓ Dense(4, softmax)

WHY GRU over LSTM
──────────────────
GRU has 2 gates vs LSTM's 3 — ~25% fewer recurrent parameters.
On small datasets, fewer parameters = better generalisation.
Empirically matches LSTM accuracy on sequences < 50 steps.

WHY TimeDistributed
────────────────────
Applies the same MobileNetV2 weights to every frame independently,
sharing weights across time. This is parameter-efficient and means the
CNN only needs to understand one frame at a time; the GRU handles time.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_transfer_model(
    sequence_length: int,
    img_height:      int,
    img_width:       int,
    num_classes:     int,
    rnn_units:       list,
    dropout_rate:    float,
    l2_reg:          float,
    rnn_type:        str = "GRU",
) -> models.Model:
    """
    Build the MobileNetV2 + GRU/LSTM transfer learning model.

    The model is returned with the MobileNetV2 base FROZEN (Phase 1 ready).
    Call unfreeze_top_layers() before Phase 2.

    Args:
        sequence_length : Number of frames per clip (T).
        img_height      : Frame height (224 recommended for MobileNetV2).
        img_width       : Frame width  (224 recommended for MobileNetV2).
        num_classes     : Number of output classes.
        rnn_units       : List of hidden sizes for stacked GRU/LSTM layers.
        dropout_rate    : Dropout rate after the recurrent block.
        l2_reg          : L2 regularisation strength on Dense layers.
        rnn_type        : "GRU" (recommended) or "LSTM".

    Returns:
        Compiled-ready (uncompiled) Keras Model.
    """
    reg = regularizers.l2(l2_reg)

    # ── MobileNetV2 base (frozen initially) ───────────────────────────────
    # include_top=False  : drop the ImageNet 1000-class head
    # pooling='avg'      : GlobalAveragePooling2D applied automatically
    #                      → output shape (1280,) per frame
    # WHY NOT EfficientNet: EfficientNet has different preprocess_input
    #   expectations and is heavier on CPU. MobileNetV2 is the best CPU/
    #   accuracy trade-off for this use case.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        pooling="avg",          # → output (1280,)
        weights="imagenet",
    )
    base_model.trainable = False   # Freeze all layers — Phase 1

    # ── Input ─────────────────────────────────────────────────────────────
    inputs = layers.Input(
        shape=(sequence_length, img_height, img_width, 3),
        name="video_input",
    )

    # ── TimeDistributed CNN feature extraction ────────────────────────────
    # WHY TimeDistributed: applies base_model to each of the T frames
    # independently using shared weights. Output: (batch, T, 1280)
    x = layers.TimeDistributed(base_model, name="td_mobilenet")(inputs)

    # Project 1280 → 256 — reduce dimensionality before the recurrent block
    # WHY: feeding 1280-dim vectors into GRU would create a huge input weight
    #      matrix (1280×256 per GRU unit). This projection reduces it while
    #      keeping the most task-relevant features.
    x = layers.TimeDistributed(
        layers.Dense(128, activation="relu", kernel_regularizer=reg),
        name="td_projection",
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name="td_bn")(x)
    x = layers.TimeDistributed(layers.Dropout(0.3), name="td_dropout")(x)

    # ── Recurrent block ───────────────────────────────────────────────────
    RNNCell = layers.GRU if rnn_type.upper() == "GRU" else layers.LSTM

    # First layer: Bidirectional — captures both forward and backward motion
    # return_sequences=True so the next GRU can process the full sequence
    x = layers.Bidirectional(
        RNNCell(rnn_units[0], return_sequences=True, dropout=0.3,
                recurrent_dropout=0.2),
        name=f"bi_{rnn_type.lower()}_1",
    )(x)

    # Second layer: standard (not bidirectional) — compresses to single vector
    if len(rnn_units) > 1:
        x = RNNCell(
            rnn_units[1], return_sequences=False, dropout=0.3,
            recurrent_dropout=0.2, name=f"{rnn_type.lower()}_2",
        )(x)

    # ── Classification head ───────────────────────────────────────────────
    x = layers.Dense(64, activation="relu", kernel_regularizer=reg, name="fc1")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(dropout_rate, name="dropout_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs, outputs, name="MobileNetV2_GRU_AutismDetector")
    return model


def unfreeze_top_layers(model: models.Model, num_layers: int) -> models.Model:
    """
    Unfreeze the last `num_layers` layers of the MobileNetV2 base for Phase 2
    fine-tuning.

    WHY only the top layers:
      Bottom layers learn universal low-level features (Gabor filters, colour
      blobs) that transfer perfectly to any visual task. Fine-tuning them would
      risk destroying these features with limited medical video data.
      Top layers encode ImageNet-specific high-level semantics. Replacing those
      with task-specific features is exactly what fine-tuning should do.

    Args:
        model      : The trained Phase 1 model.
        num_layers : Number of MobileNetV2 layers to unfreeze from the top.

    Returns:
        Same model with layers unfrozen (in-place modification).
    """
    # Find the TimeDistributed MobileNetV2 layer
    td_layer   = model.get_layer("td_mobilenet")
    base_model = td_layer.layer     # the MobileNetV2 inside TimeDistributed

    base_model.trainable = True

    # Freeze everything except the last num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    # Always freeze BatchNormalization layers in the base
    # WHY: BN layers store running statistics from ImageNet. Updating them
    #      with our tiny medical video dataset would destabilise training.
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    print(f"  Unfroze top {num_layers} MobileNetV2 layers "
          f"(BN layers remain frozen).")
    print(f"  Trainable parameters: {trainable_count:,}")

    return model


def get_model_summary(model: models.Model) -> None:
    """Print a clean summary with trainable parameter counts."""
    total     = sum(tf.size(w).numpy() for w in model.weights)
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    frozen    = total - trainable

    model.summary()
    print(f"\n  Total params     : {total:>10,}")
    print(f"  Trainable params : {trainable:>10,}")
    print(f"  Frozen params    : {frozen:>10,}")
