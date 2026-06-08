"""Model builders: baseline CNN and (later) transfer-learning models."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config as C


def set_seeds(seed: int = C.SEED) -> None:
    """Seed Python, NumPy and TensorFlow for reproducible training."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_baseline_cnn(
    input_shape: tuple[int, int, int] = C.INPUT_SHAPE,
    num_classes: int = C.NUM_CLASSES,
    learning_rate: float = C.LEARNING_RATE,
) -> keras.Model:
    """A small CNN trained from scratch on Mel spectrograms.

    Three Conv-BN-MaxPool blocks followed by global average pooling and a dense
    head. Global pooling (instead of Flatten) keeps the parameter count low,
    which helps against overfitting on this small dataset.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),

            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),

            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),

            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="baseline_cnn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transfer_cnn(
    input_shape: tuple[int, int, int] = C.INPUT_SHAPE,
    num_classes: int = C.NUM_CLASSES,
    learning_rate: float = C.LEARNING_RATE,
    fine_tune: bool = False,
) -> keras.Model:
    """Transfer learning with a MobileNetV2 backbone pretrained on ImageNet.

    Single-channel Mel spectrograms are repeated to 3 channels and put through
    the backbone's preprocessing. The backbone is frozen by default (feature
    extraction); set ``fine_tune=True`` to unfreeze it for fine-tuning.
    """
    base = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet",
        input_shape=(input_shape[0], input_shape[1], 3),
    )
    base.trainable = fine_tune

    inp = keras.Input(shape=input_shape)
    x = layers.Concatenate()([inp, inp, inp]) if input_shape[-1] == 1 else inp
    x = layers.Rescaling(255.0)(x)                       # [0,1] -> [0,255]
    x = keras.applications.mobilenet_v2.preprocess_input(x)  # -> [-1,1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inp, out, name="transfer_mobilenetv2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenet_extractor(
    input_shape: tuple[int, int, int] = C.INPUT_SHAPE,
) -> keras.Model:
    """Frozen MobileNetV2 (ImageNet) that outputs a 1280-d pooled embedding.

    Single-channel Mel spectrograms are repeated to 3 channels and put through
    the backbone's preprocessing. Used to precompute features **once** so the
    classifier head can be trained cheaply on CPU (feature extraction).
    """
    base = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet",
        input_shape=(input_shape[0], input_shape[1], 3), pooling="avg",
    )
    base.trainable = False
    inp = keras.Input(shape=input_shape)
    x = layers.Concatenate()([inp, inp, inp]) if input_shape[-1] == 1 else inp
    x = layers.Rescaling(255.0)(x)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    out = base(x, training=False)
    return keras.Model(inp, out, name="mobilenet_extractor")


def build_transfer_head(
    input_dim: int = 1280,
    num_classes: int = C.NUM_CLASSES,
    learning_rate: float = C.LEARNING_RATE,
) -> keras.Model:
    """Small dense classifier trained on top of precomputed embeddings."""
    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="transfer_head",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def default_callbacks(
    monitor: str = "val_loss",
    patience: int = C.EARLY_STOPPING_PATIENCE,
) -> list[keras.callbacks.Callback]:
    """Early stopping (restoring best weights) + LR reduction on plateau."""
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=max(2, patience // 3), min_lr=1e-5
        ),
    ]
