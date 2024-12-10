import os
import cv2
import random
import json
from functools import partial
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import utils


def load_dataset(dataset_folder):
    images = []
    labels = []
    json_files = [el for el in dataset_folder.iterdir() if el.suffix == ".json"]

    for json_file in json_files:
        with open(json_file) as f:
            sample = json.load(f)

        images.append(cv2.imread(sample["image"]))
        labels.append([sample["angle"]])

    # Convert to np arrays & process images
    X = np.asarray(images)
    X = utils.preprocess_images(X)
    y = np.asarray(labels)

    return X, y


def create_tf_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = (
        dataset.shuffle(buffer_size=1000)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return dataset


if __name__ == "__main__":
    current_folder = Path(__file__).parent
    dataset_folder = current_folder / "dataset"
    models_folder = current_folder / "models"
    logs_folder = current_folder / "logs"

    print("Loading dataset ...")
    X, y = load_dataset(dataset_folder)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    batch_size = 32
    train_dataset = create_tf_dataset(X_train, y_train, batch_size)
    valid_dataset = create_tf_dataset(X_valid, y_valid, batch_size)
    test_dataset = create_tf_dataset(X_test, y_test, batch_size)

    # Enable multi-GPU strategy
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        DefaultConv2D = partial(
            keras.layers.Conv2D, kernel_size=3, activation="relu", padding="SAME"
        )

        model = keras.Sequential(
            [
                DefaultConv2D(filters=64, kernel_size=7, input_shape=[64, 128, 3]),
                keras.layers.MaxPooling2D(pool_size=2),
                DefaultConv2D(filters=128),
                DefaultConv2D(filters=128),
                keras.layers.MaxPooling2D(pool_size=2),
                DefaultConv2D(filters=256),
                DefaultConv2D(filters=256),
                keras.layers.MaxPooling2D(pool_size=2),
                keras.layers.Flatten(),
                keras.layers.Dense(units=128, activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(units=64, activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(units=1),
            ]
        )

        model.compile(loss="mean_squared_error", optimizer="SGD")

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=logs_folder,
        histogram_freq=1,
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    print("Starting training ...")
    history = model.fit(
        train_dataset,
        epochs=500,
        validation_data=valid_dataset,
        callbacks=[tensorboard_cb, early_stopping_cb],
    )

    print("Evaluating on test data ...")
    mse_test = model.evaluate(test_dataset)
    print(f"Test Data - MSE: {mse_test}")

    os.makedirs(models_folder, exist_ok=True)
    model.save(models_folder / "model2.h5")
