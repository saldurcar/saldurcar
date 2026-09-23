"""Public portfolio GRU example for battery thermal-stage classification."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


WINDOW = 30
TEST_SIZE = 0.25
RANDOM_STATE = 42


def make_windows(df, feature_columns, label_column="stage_id", window=WINDOW):
    data = df[feature_columns].to_numpy(dtype=float)
    labels = df[label_column].to_numpy(dtype=int)

    x, y = [], []
    for end in range(window - 1, len(df)):
        x.append(data[end - window + 1 : end + 1])
        y.append(labels[end])

    return np.asarray(x), np.asarray(y)


def build_gru(n_steps, n_features, n_classes):
    model = Sequential(
        [
            GRU(32, input_shape=(n_steps, n_features)),
            Dropout(0.30),
            Dense(16, activation="relu"),
            Dropout(0.20),
            Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_from_dataframe(df, feature_columns, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_raw, y_raw = make_windows(df, feature_columns)
    classes = np.unique(y_raw)
    class_to_index = {int(c): i for i, c in enumerate(classes)}
    y_idx = np.asarray([class_to_index[int(y)] for y in y_raw])

    x_train_raw, x_test_raw, y_train_idx, y_test_idx = train_test_split(
        x_raw,
        y_idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_idx if np.bincount(y_idx).min() >= 2 else None,
    )

    n_train, n_steps, n_features = x_train_raw.shape
    scaler = StandardScaler()
    scaler.fit(x_train_raw.reshape(-1, n_features))

    x_train = scaler.transform(x_train_raw.reshape(-1, n_features)).reshape(
        n_train, n_steps, n_features
    )
    x_test = scaler.transform(x_test_raw.reshape(-1, n_features)).reshape(
        len(x_test_raw), n_steps, n_features
    )

    y_train = to_categorical(y_train_idx, num_classes=len(classes))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_idx),
        y=y_train_idx,
    )
    class_weights = {
        int(c): float(w) for c, w in zip(np.unique(y_train_idx), weights)
    }

    model = build_gru(n_steps, n_features, len(classes))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6
        ),
    ]

    model.fit(
        x_train,
        y_train,
        validation_split=0.20,
        epochs=150,
        batch_size=16,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    probabilities = model.predict(x_test, verbose=0)
    y_pred_idx = np.argmax(probabilities, axis=1)

    print(classification_report(y_test_idx, y_pred_idx, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_idx, y_pred_idx))

    model.save(output_dir / "gru_stage_classifier.keras")
    joblib.dump(scaler, output_dir / "gru_scaler.pkl")
    return model, scaler
