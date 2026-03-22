"""
train.py
Train a face classifier using Transfer Learning on top of InceptionV3 or MobileNetV2.

Usage:
    python scripts/train.py \
        --data_dir dataset/train_processed/ \
        --val_dir  dataset/val_processed/ \
        --model    inceptionv3 \
        --epochs   30 \
        --batch    32 \
        --lr       1e-4 \
        --output   models/face_classifier.h5
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ──────────────────────────────────────────────
# Model builders
# ──────────────────────────────────────────────

def build_model(num_classes: int, backbone: str = "inceptionv3",
                img_size: int = 160, freeze_base: bool = True) -> keras.Model:
    """Build a transfer-learning model for face classification."""

    input_shape = (img_size, img_size, 3)

    if backbone == "inceptionv3":
        base = keras.applications.InceptionV3(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    elif backbone == "mobilenetv2":
        base = keras.applications.MobileNetV2(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    elif backbone == "facenet":
        # FaceNet-style: use InceptionV3 base + L2-normalised embeddings
        base = keras.applications.InceptionV3(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    if freeze_base:
        base.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name=f"face_classifier_{backbone}")
    return model


# ──────────────────────────────────────────────
# Data generators
# ──────────────────────────────────────────────

def make_generators(train_dir: str, val_dir: str | None,
                    img_size: int, batch_size: int):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        validation_split=0.2 if val_dir is None else 0.0,
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training" if val_dir is None else None,
    )

    if val_dir:
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)
        val_gen = val_datagen.flow_from_directory(
            val_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode="categorical",
        )
    else:
        val_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation",
        )

    return train_gen, val_gen


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train(args):
    print(f"\n🚀 Training face classifier")
    print(f"   Backbone : {args.model}")
    print(f"   Data dir : {args.data_dir}")
    print(f"   Epochs   : {args.epochs}")
    print(f"   Batch    : {args.batch}")
    print()

    train_gen, val_gen = make_generators(
        args.data_dir, args.val_dir, args.img_size, args.batch
    )

    num_classes = train_gen.num_classes
    class_names = {v: k for k, v in train_gen.class_indices.items()}
    print(f"   Classes  : {num_classes} → {list(train_gen.class_indices.keys())}")

    # Save class mapping
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "class_names.json", "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    # Build and compile
    model = build_model(num_classes, args.model, args.img_size)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            args.output, save_best_only=True, monitor="val_accuracy", verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir=str(output_dir / "logs")),
    ]

    # Phase 1: frozen base
    print("\n📌 Phase 1 — training top layers (frozen base)...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=min(args.epochs, 15),
        callbacks=callbacks,
    )

    # Phase 2: fine-tune last layers
    if args.epochs > 15:
        print("\n🔓 Phase 2 — fine-tuning last 30 layers...")
        base = model.layers[1]
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr / 10),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.epochs - 15,
            callbacks=callbacks,
        )

    print(f"\n✅ Model saved to: {args.output}")
    print(f"   Class map  : {output_dir}/class_names.json")


def main():
    parser = argparse.ArgumentParser(description="Train face recognition classifier")
    parser.add_argument("--data_dir",  required=True)
    parser.add_argument("--val_dir",   default=None)
    parser.add_argument("--model",     default="inceptionv3",
                        choices=["inceptionv3", "mobilenetv2", "facenet"])
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--img_size",  type=int,   default=160)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--output",    default="models/face_classifier.h5")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
