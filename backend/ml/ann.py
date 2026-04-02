from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from ml.classification import train_and_evaluate_classification
from ml.data_utils import default_csv_path
from ml.data_utils import prepare_modeling_dataframe


def _to_base64_png() -> str:
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    plt.close()
    return image_b64


def _default_models_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return None


def build_ann_model(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))

    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def _training_curves_plot(history: tf.keras.callbacks.History) -> str:
    history_dict = history.history
    epochs = range(1, len(history_dict.get("loss", [])) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict.get("loss", []), label="Train Loss", linewidth=2)
    plt.plot(epochs, history_dict.get("val_loss", []), label="Val Loss", linewidth=2)
    plt.title("ANN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict.get("auc", []), label="Train AUC", linewidth=2)
    plt.plot(epochs, history_dict.get("val_auc", []), label="Val AUC", linewidth=2)
    plt.title("ANN Training AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()

    return _to_base64_png()


def _confusion_matrix_plot(cm: np.ndarray) -> str:
    plt.figure(figsize=(10,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    plt.title("ANN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    return _to_base64_png()


def _best_classical_metrics(
    csv_path: Path,
    models_dir: Path,
) -> tuple[str, dict[str, float | None]]:
    classification_results_path = models_dir / "classification" / "results.json"

    if classification_results_path.exists():
        with classification_results_path.open("r", encoding="utf-8") as fp:
            classification_results = json.load(fp)
    else:
        classification_results = train_and_evaluate_classification(
            csv_path=csv_path,
            models_dir=models_dir,
        )

    model_entries: dict[str, dict[str, Any]] = classification_results.get("models", {})
    if not model_entries:
        raise ValueError("No classical classification model results found.")

    best_model_name = ""
    best_model_payload: dict[str, Any] = {}
    best_score = float("-inf")

    for model_name, payload in model_entries.items():
        metrics = payload.get("metrics", {})
        score = float(metrics.get("f1_weighted", float("-inf")))
        if score > best_score:
            best_score = score
            best_model_name = model_name
            best_model_payload = payload

    if not best_model_name:
        raise ValueError("Unable to select best classical model.")

    metrics = best_model_payload.get("metrics", {})
    return best_model_name, {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "f1": float(metrics.get("f1_weighted", 0.0)),
        "auc_roc": (
            float(metrics["auc_roc"])
            if metrics.get("auc_roc") is not None
            else None
        ),
    }


def _comparison_plot(
    ann_metrics: dict[str, float | None],
    classical_name: str,
    classical_metrics: dict[str, float | None],
) -> str:
    labels = ["Accuracy", "F1", "AUC-ROC"]

    ann_values = [
        float(ann_metrics.get("accuracy") or 0.0),
        float(ann_metrics.get("f1") or 0.0),
        float(ann_metrics.get("auc_roc") or 0.0),
    ]
    classical_values = [
        float(classical_metrics.get("accuracy") or 0.0),
        float(classical_metrics.get("f1") or 0.0),
        float(classical_metrics.get("auc_roc") or 0.0),
    ]

    x = np.arange(len(labels))
    width = 0.34

    plt.figure(figsize=(10,6))
    plt.bar(x - width / 2, ann_values, width=width, label="ANN", color="#2563eb")
    plt.bar(
        x + width / 2,
        classical_values,
        width=width,
        label=classical_name,
        color="#0f766e",
    )

    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("ANN vs Best Classical Model")
    plt.legend()

    return _to_base64_png()


def train_and_evaluate_ann(
    csv_path: str | Path | None = None,
    models_dir: str | Path | None = None,
) -> dict[str, Any]:
    csv_path = Path(csv_path) if csv_path is not None else default_csv_path()
    models_dir = Path(models_dir) if models_dir is not None else _default_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    df = prepare_modeling_dataframe(csv_path)
    if "readmitted_30" not in df.columns:
        raise KeyError("Target column 'readmitted_30' not found after preprocessing.")

    drop_cols = [
        col
        for col in ("readmitted_30", "readmitted", "encounter_id", "patient_nbr")
        if col in df.columns
    ]

    X = df.drop(columns=drop_cols).astype(np.float32)
    y = df["readmitted_30"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, models_dir / "ann_scaler.pkl")

    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}

    ann_model = build_ann_model(input_dim=X_train_scaled.shape[1])

    checkpoint_path = models_dir / "ann_best.h5"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), save_best_only=True),
    ]

    history = ann_model.fit(
        X_train_scaled,
        y_train,
        epochs=100,
        batch_size=256,
        validation_split=0.15,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )

    ann_model_path = models_dir / "ann_model.h5"
    ann_model.save(ann_model_path)

    y_prob = ann_model.predict(X_test_scaled, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc_roc = _safe_auc(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    training_curves_b64 = _training_curves_plot(history)
    confusion_matrix_b64 = _confusion_matrix_plot(cm)

    best_classical_name, best_classical = _best_classical_metrics(
        csv_path=csv_path,
        models_dir=models_dir,
    )

    ann_metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "auc_roc": auc_roc,
    }
    comparison_plot_b64 = _comparison_plot(
        ann_metrics=ann_metrics,
        classical_name=best_classical_name,
        classical_metrics=best_classical,
    )

    results: dict[str, Any] = {
        "task": "ann_classification",
        "target": "readmitted_30",
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
        "class_weight": class_weight,
        "metrics": ann_metrics,
        "training_curves_plot": training_curves_b64,
        "confusion_matrix_plot": confusion_matrix_b64,
        "comparison_plot": comparison_plot_b64,
        "best_classical_model": {
            "model_name": best_classical_name,
            "metrics": best_classical,
        },
        "model_path": str(ann_model_path),
        "best_checkpoint_path": str(checkpoint_path),
    }

    with (models_dir / "ann_results.json").open("w", encoding="utf-8") as fp:
        json.dump(results, fp)

    return results
