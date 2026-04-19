from __future__ import annotations

import base64
import json
import os
import sys
from io import BytesIO
from typing import Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(CURRENT_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import joblib
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.data_utils import default_csv_path
from ml.data_utils import prepare_modeling_dataframe


def _to_base64_png() -> str:
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    plt.close()
    return encoded


def _default_models_dir() -> str:
    project_root = os.path.dirname(BACKEND_ROOT)
    return os.path.join(project_root, "models")


def _classification_artifact_paths(models_dir: str) -> tuple[str, str]:
    classification_dir = os.path.join(models_dir, "classification")
    selector_path = os.path.join(classification_dir, "variance_selector.pkl")
    scaler_path = os.path.join(classification_dir, "scaler.pkl")
    return selector_path, scaler_path


def _apply_shared_feature_pipeline(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    models_dir: str,
) -> tuple[np.ndarray, np.ndarray, Any, Any, str, str]:
    selector_path, classification_scaler_path = _classification_artifact_paths(models_dir)

    if os.path.exists(selector_path) and os.path.exists(classification_scaler_path):
        selector = joblib.load(selector_path)
        scaler = joblib.load(classification_scaler_path)
    else:
        classification_dir = os.path.dirname(selector_path)
        os.makedirs(classification_dir, exist_ok=True)

        selector = VarianceThreshold(threshold=0.009)
        X_train_reduced = selector.fit_transform(X_train_raw)
        X_test_reduced = selector.transform(X_test_raw)

        scaler = StandardScaler()
        scaler.fit(X_train_reduced)

        joblib.dump(selector, selector_path)
        joblib.dump(scaler, classification_scaler_path)

    X_train_reduced = selector.transform(X_train_raw)
    X_test_reduced = selector.transform(X_test_raw)

    X_train_scaled = scaler.transform(X_train_reduced).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_reduced).astype(np.float32)

    return (
        X_train_scaled,
        X_test_scaled,
        selector,
        scaler,
        selector_path,
        classification_scaler_path,
    )


def focal_loss(gamma: float = 2.5, alpha: float = 0.75):
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_cast = tf.cast(y_true, tf.float32)
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = (
            -y_true_cast * tf.math.log(y_pred_clipped)
            - (1.0 - y_true_cast) * tf.math.log(1.0 - y_pred_clipped)
        )
        p_t = y_true_cast * y_pred_clipped + (1.0 - y_true_cast) * (1.0 - y_pred_clipped)
        focal = alpha * tf.pow(1.0 - p_t, gamma) * bce
        return tf.reduce_mean(focal)

    return loss


def build_ann_model(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=l2(0.001))(inputs)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss(gamma=2.5, alpha=0.75),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return None


def _training_curves_plot(history: tf.keras.callbacks.History) -> str:
    history_data = history.history
    epochs = range(1, len(history_data.get("loss", [])) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_data.get("loss", []), label="Train Loss", linewidth=2)
    plt.plot(epochs, history_data.get("val_loss", []), label="Val Loss", linewidth=2)
    plt.title("ANN Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_data.get("auc", []), label="Train AUC", linewidth=2)
    plt.plot(epochs, history_data.get("val_auc", []), label="Val AUC", linewidth=2)
    plt.title("ANN AUC Curves")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()

    return _to_base64_png()


def _confusion_matrix_plot(cm: np.ndarray) -> str:
    plt.figure(figsize=(10, 6))
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


def _roc_curve_plot(y_true: np.ndarray, y_prob: np.ndarray, auc_value: float | None) -> str:
    plt.figure(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    label = "ROC Curve"
    if auc_value is not None:
        label = f"ROC Curve (AUC={auc_value:.3f})"
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random")
    plt.title("ANN ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    return _to_base64_png()


def _save_threshold(models_dir: str, best_threshold: float) -> str:
    threshold_path = os.path.join(models_dir, "ann_threshold.json")
    with open(threshold_path, "w", encoding="utf-8") as fp:
        json.dump({"best_threshold": float(best_threshold)}, fp, indent=2)
    return threshold_path


def load_ann_threshold(models_dir: str | None = None, default: float = 0.5) -> float:
    resolved_models_dir = models_dir if models_dir is not None else _default_models_dir()
    threshold_path = os.path.join(resolved_models_dir, "ann_threshold.json")
    if not os.path.exists(threshold_path):
        return float(default)

    try:
        with open(threshold_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return float(payload.get("best_threshold", default))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return float(default)


def train_and_evaluate_ann(
    csv_path: str | os.PathLike[str] | None = None,
    models_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    csv_path_resolved = str(csv_path) if csv_path is not None else str(default_csv_path())
    models_dir_resolved = str(models_dir) if models_dir is not None else _default_models_dir()

    os.makedirs(models_dir_resolved, exist_ok=True)

    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    df = prepare_modeling_dataframe(csv_path_resolved)
    if "readmitted_30" not in df.columns:
        raise KeyError("Target column 'readmitted_30' not found after preprocessing.")

    drop_cols = [
        col
        for col in ("readmitted_30", "readmitted", "encounter_id", "patient_nbr")
        if col in df.columns
    ]

    X = df.drop(columns=drop_cols)
    y = df["readmitted_30"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train_raw = X_train.to_numpy(dtype=np.float64)
    X_test_raw = X_test.to_numpy(dtype=np.float64)

    (
        X_train_scaled,
        X_test_scaled,
        _selector,
        classification_scaler,
        selector_path,
        classification_scaler_path,
    ) = _apply_shared_feature_pipeline(
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        models_dir=models_dir_resolved,
    )

    print(
        "ANN using shared classification feature pipeline "
        f"({X_train_scaled.shape[1]} selected features)."
    )

    sm = SMOTE(random_state=42, sampling_strategy=0.4)
    X_train_res_scaled, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    ann_scaler_path = os.path.join(models_dir_resolved, "ann_scaler.pkl")
    joblib.dump(classification_scaler, ann_scaler_path)

    model = build_ann_model(input_dim=X_train_res_scaled.shape[1])

    checkpoint_path = os.path.join(models_dir_resolved, "ann_best.h5")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            mode="min",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            mode="min",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        ),
    ]

    history = model.fit(
        X_train_res_scaled,
        y_train_res,
        epochs=50,
        batch_size=512,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0,
    )

    ann_model_path = os.path.join(models_dir_resolved, "ann_model.h5")
    model.save(ann_model_path)

    y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()

    proba_flat = y_pred_proba.flatten()
    print("Probability distribution:")
    print(f"  Min:    {proba_flat.min():.4f}")
    print(f"  Max:    {proba_flat.max():.4f}")
    print(f"  Mean:   {proba_flat.mean():.4f}")
    print(f"  Median: {np.median(proba_flat):.4f}")
    print(f"  % above 0.5: {(proba_flat > 0.5).mean() * 100:.1f}%")
    print(f"  % above 0.3: {(proba_flat > 0.3).mean() * 100:.1f}%")
    print(f"  % above 0.2: {(proba_flat > 0.2).mean() * 100:.1f}%")

    thresholds = np.arange(0.25, 0.75, 0.01)
    f1_scores = [
        f1_score(y_test, (y_pred_proba > t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_threshold = float(thresholds[int(np.argmax(f1_scores))]) if len(f1_scores) else 0.35
    if best_threshold < 0.30:
        print("WARNING: best_threshold below 0.30; model probabilities are compressed and may need retraining.")

    threshold_path = _save_threshold(models_dir_resolved, best_threshold)

    y_pred_final = (y_pred_proba > best_threshold).astype(int)

    recall_0 = float(recall_score(y_test, y_pred_final, pos_label=0, zero_division=0))
    recall_1 = float(recall_score(y_test, y_pred_final, pos_label=1, zero_division=0))
    if recall_0 < 0.50 or recall_1 < 0.40:
        print("WARNING: Model is still imbalanced after threshold tuning.")
        print(f"Recall class 0: {recall_0:.3f}, Recall class 1: {recall_1:.3f}")
        print("Consider retraining with different hyperparameters.")

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_final)),
        "f1": float(f1_score(y_test, y_pred_final, zero_division=0)),
        "auc_roc": _safe_auc(y_test, y_pred_proba),
        "recall": float(recall_score(y_test, y_pred_final, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred_final, zero_division=0)),
        "recall_class_0": recall_0,
        "recall_class_1": recall_1,
        "best_threshold": best_threshold,
    }

    report_text = classification_report(y_test, y_pred_final, zero_division=0)
    print("ANN Classification Report")
    print(report_text)
    print("ANN Metrics")
    print(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_test, y_pred_final, labels=[0, 1])
    training_curves_plot = _training_curves_plot(history)
    confusion_matrix_plot = _confusion_matrix_plot(cm)
    roc_curve_plot = _roc_curve_plot(y_test, y_pred_proba, metrics["auc_roc"])

    results: dict[str, Any] = {
        "task": "ann_classification",
        "target": "readmitted_30",
        "raw_train_shape": [int(X_train_raw.shape[0]), int(X_train_raw.shape[1])],
        "raw_test_shape": [int(X_test_raw.shape[0]), int(X_test_raw.shape[1])],
        "train_shape": [int(X_train_scaled.shape[0]), int(X_train_scaled.shape[1])],
        "train_resampled_shape": [int(X_train_res_scaled.shape[0]), int(X_train_res_scaled.shape[1])],
        "test_shape": [int(X_test_scaled.shape[0]), int(X_test_scaled.shape[1])],
        "shared_feature_space_with_rf": True,
        "class_weight": None,
        "metrics": metrics,
        "classification_report": report_text,
        "training_curves_plot": training_curves_plot,
        "confusion_matrix_plot": confusion_matrix_plot,
        "roc_curve_plot": roc_curve_plot,
        "model_path": ann_model_path,
        "best_checkpoint_path": checkpoint_path,
        "scaler_path": ann_scaler_path,
        "classification_selector_path": selector_path,
        "classification_scaler_path": classification_scaler_path,
        "threshold_path": threshold_path,
    }

    results_path = os.path.join(models_dir_resolved, "ann_results.json")
    with open(results_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp)

    return results


if __name__ == "__main__":
    output = train_and_evaluate_ann()
    print(json.dumps(output.get("metrics", {}), indent=2))
    print("model_path:", output.get("model_path"))
