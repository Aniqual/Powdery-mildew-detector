"""
Train a MobileNetV2 classifier on the merged dataset and export a TFLite model for edge devices.

Usage:  python train.py --data ./data --epochs 12 --out ./model
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

IMG = 224


def load(split, data, batch, shuffle):
    return keras.utils.image_dataset_from_directory(
        Path(data) / split, image_size=(IMG, IMG), batch_size=batch,
        label_mode="categorical", shuffle=shuffle, seed=42)


def build(n_classes):
    aug = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ], name="augment")
    base = keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inp = keras.Input((IMG, IMG, 3))            # float32, raw 0..255 (same as on the edge device)
    x = aug(inp)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)  # MobileNetV2 expects -1..1
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./model")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--finetune-epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train, val, test = load("train", a.data, a.batch, True), load("val", a.data, a.batch, False), load("test", a.data, a.batch, False)
    classes = train.class_names
    (out / "labels.json").write_text(json.dumps(classes, indent=2))

    # class weights for imbalance
    counts = np.array([len(list((Path(a.data) / "train" / c).glob("*"))) for c in classes], dtype=float)
    weights = {i: counts.sum() / (len(classes) * c) for i, c in enumerate(counts)}

    model, base = build(len(classes))
    cb = [keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
          keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3)]

    model.compile(keras.optimizers.Adam(1e-3), "categorical_crossentropy", metrics=["accuracy"])
    model.fit(train.prefetch(2), validation_data=val, epochs=a.epochs, class_weight=weights, callbacks=cb)

    # fine-tune the top of the backbone
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(keras.optimizers.Adam(1e-5), "categorical_crossentropy", metrics=["accuracy"])
    model.fit(train.prefetch(2), validation_data=val, epochs=a.finetune_epochs, class_weight=weights, callbacks=cb)

    # evaluation
    y_true = np.concatenate([y.numpy().argmax(1) for _, y in test])
    y_pred = model.predict(test.map(lambda x, y: x)).argmax(1)
    print("\n" + classification_report(y_true, y_pred, target_names=classes, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    # mildew vs not-mildew view (what the AIoT alert actually uses)
    is_pm = np.array([c.endswith("powdery_mildew") for c in classes])
    t_bin, p_bin = is_pm[y_true], is_pm[y_pred]
    tp = (t_bin & p_bin).sum()
    print(f"\nPowdery-mildew alert  precision={tp / max(p_bin.sum(), 1):.3f}  recall={tp / max(t_bin.sum(), 1):.3f}")

    model.save(out / "model.keras")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range quantisation, ~4x smaller
    (out / "model.tflite").write_bytes(conv.convert())
    print(f"\nSaved {out/'model.tflite'} ({(out/'model.tflite').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
