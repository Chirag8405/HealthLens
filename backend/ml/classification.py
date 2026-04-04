from __future__ import annotations

import base64
import gc
import json
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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


def _confusion_matrix_plot(cm: np.ndarray, model_name: str) -> str:
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    return _to_base64_png()


def _single_roc_curve_plot(fpr: np.ndarray, tpr: np.ndarray, auc: float | None, model_name: str) -> str:
    plt.figure(figsize=(10, 6))
    label = f"{model_name}"
    if auc is not None:
        label = f"{model_name} (AUC={auc:.3f})"
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random")
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    return _to_base64_png()


def _roc_overlay_plot(roc_data: dict[str, dict[str, Any]]) -> str:
    plt.figure(figsize=(10, 6))
    for model_name, data in roc_data.items():
        fpr = data["fpr"]
        tpr = data["tpr"]
        auc = data["auc"]
        label = f"{model_name}"
        if auc is not None:
            label = f"{model_name} (AUC={auc:.3f})"
        plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    return _to_base64_png()


def _default_models_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray | None) -> float | None:
    if y_score is None:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    y_pred = model.predict(X_test)

    y_score: np.ndarray | None = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = np.asarray(model.decision_function(X_test), dtype=float)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    auc = _safe_auc(y_test, y_score)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    confusion_matrix_b64 = _confusion_matrix_plot(cm, model_name)

    roc_curve_b64 = ""
    roc_overlay_data: dict[str, Any] | None = None
    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_curve_b64 = _single_roc_curve_plot(fpr, tpr, auc, model_name)
        roc_overlay_data = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc,
        }

    result = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": [[int(x) for x in row] for row in cm.tolist()],
        "confusion_matrix_b64": confusion_matrix_b64,
        "roc_curve_b64": roc_curve_b64,
    }
    return result, roc_overlay_data


def train_and_evaluate_classification(
    csv_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    skip_svm: bool = False,
    profile: str | None = None,
    n_jobs: int | None = None,
    cv_folds: int | None = None,
    max_search_samples: int | None = None,
    max_svc_samples: int | None = None,
) -> dict[str, Any]:
    del profile, n_jobs, cv_folds, max_search_samples, max_svc_samples

    csv_path = Path(csv_path) if csv_path is not None else default_csv_path()
    models_dir = Path(models_dir) if models_dir is not None else _default_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    classification_dir = models_dir / "classification"
    classification_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_modeling_dataframe(csv_path)
    if "readmitted_30" not in df.columns:
        raise KeyError("Target column 'readmitted_30' not found after preprocessing.")

    drop_cols = [
        col
        for col in ("readmitted_30", "readmitted", "encounter_id", "patient_nbr")
        if col in df.columns
    ]
    X = df.drop(columns=drop_cols)
    y = df["readmitted_30"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train_raw = X_train.to_numpy(dtype=np.float64)
    X_test_raw = X_test.to_numpy(dtype=np.float64)
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    selector = VarianceThreshold(threshold=0.009)
    X_train_reduced = selector.fit_transform(X_train_raw)
    X_test_reduced = selector.transform(X_test_raw)
    joblib.dump(selector, classification_dir / "variance_selector.pkl")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reduced)
    X_test_scaled = scaler.transform(X_test_reduced)
    joblib.dump(scaler, classification_dir / "scaler.pkl")

    with (classification_dir / "feature_names.json").open("w", encoding="utf-8") as fp:
        json.dump(X.columns.tolist(), fp, indent=2)

    results_payload: dict[str, Any] = {}
    legacy_models: dict[str, dict[str, Any]] = {}
    roc_overlay_data: dict[str, dict[str, Any]] = {}

    ordered_models: list[tuple[str, str, Any]] = [
        (
            "logistic_regression",
            "LogisticRegression",
            LogisticRegression(max_iter=1000, solver="liblinear"),
        ),
        (
            "decision_tree",
            "DecisionTreeClassifier",
            DecisionTreeClassifier(random_state=42),
        ),
        (
            "knn",
            "KNeighborsClassifier",
            KNeighborsClassifier(n_neighbors=7),
        ),
        (
            "random_forest",
            "RandomForestClassifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_leaf=4,
                n_jobs=1,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]

    total_models = len(ordered_models) + (0 if skip_svm else 1)

    for i, (result_key, legacy_name, estimator) in enumerate(ordered_models, start=1):
        print(f"[{i}/{total_models}] Training {legacy_name}...", flush=True)
        started_at = time.perf_counter()

        model = estimator.fit(X_train_scaled, y_train_np)
        model_filename = f"{model.__class__.__name__.lower()}.joblib"
        joblib.dump(model, classification_dir / model_filename)

        metrics, roc_data = _evaluate_model(model, X_test_scaled, y_test_np, legacy_name)
        results_payload[result_key] = metrics
        legacy_models[legacy_name] = {
            "model_name": legacy_name,
            "best_params": {},
            "metrics": {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_weighted": metrics["f1"],
                "auc_roc": metrics["auc"],
            },
            "confusion_matrix_plot": metrics["confusion_matrix_b64"],
        }

        if roc_data is not None and metrics["auc"] is not None:
            roc_overlay_data[legacy_name] = roc_data

        elapsed = time.perf_counter() - started_at
        print(
            f"[{i}/{total_models}] {legacy_name} done - {elapsed:.1f}s | "
            f"accuracy={metrics['accuracy']:.3f} f1={metrics['f1']:.3f}",
            flush=True,
        )

        del model
        gc.collect()
        print(f"[ml/train] {legacy_name} done, memory freed")

    if not skip_svm:
        i = total_models
        print(f"[{i}/{total_models}] Training SVC...", flush=True)
        started_at = time.perf_counter()

        if X_train_scaled.shape[0] > 50_000:
            rng = np.random.default_rng(42)
            idx = rng.choice(X_train_scaled.shape[0], 50_000, replace=False)
            X_svm = X_train_scaled[idx]
            y_svm = y_train_np[idx]
        else:
            X_svm, y_svm = X_train_scaled, y_train_np

        svm_model = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            cache_size=500,
            max_iter=1000,
        ).fit(X_svm, y_svm)
        joblib.dump(svm_model, classification_dir / "svc.joblib")

        svm_metrics, svm_roc_data = _evaluate_model(svm_model, X_test_scaled, y_test_np, "SVC")
        results_payload["svm"] = svm_metrics
        legacy_models["SVC"] = {
            "model_name": "SVC",
            "best_params": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "cache_size": 500,
                "max_iter": 1000,
                "train_rows": int(X_svm.shape[0]),
            },
            "metrics": {
                "accuracy": svm_metrics["accuracy"],
                "precision": svm_metrics["precision"],
                "recall": svm_metrics["recall"],
                "f1_weighted": svm_metrics["f1"],
                "auc_roc": svm_metrics["auc"],
            },
            "confusion_matrix_plot": svm_metrics["confusion_matrix_b64"],
        }

        if svm_roc_data is not None and svm_metrics["auc"] is not None:
            roc_overlay_data["SVC"] = svm_roc_data

        elapsed = time.perf_counter() - started_at
        print(
            f"[{i}/{total_models}] SVC done - {elapsed:.1f}s | "
            f"accuracy={svm_metrics['accuracy']:.3f} f1={svm_metrics['f1']:.3f}",
            flush=True,
        )

        del svm_model
        gc.collect()
        print("[ml/train] SVC done, memory freed")
    else:
        print("[ml/train] SVC skipped (--skip-svm enabled)", flush=True)

    roc_overlay_b64 = _roc_overlay_plot(roc_overlay_data)

    results_payload["roc_overlay_b64"] = roc_overlay_b64
    results_payload["meta"] = {
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "n_features_after_variance": int(X_train_scaled.shape[1]),
    }
    results_payload["trained_at"] = datetime.now().isoformat()

    with (models_dir / "ml_results.json").open("w", encoding="utf-8") as fp:
        json.dump(results_payload, fp)

    legacy_summary: dict[str, Any] = {
        "task": "classification",
        "target": "readmitted_30",
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
        "models": legacy_models,
        "roc_curve_plot": roc_overlay_b64,
    }

    with (classification_dir / "results.json").open("w", encoding="utf-8") as fp:
        json.dump(legacy_summary, fp)

    return legacy_summary