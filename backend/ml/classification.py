from __future__ import annotations

import base64
import json
import os
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

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
    plt.figure(figsize=(10,6))
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


def _roc_overlay_plot(roc_data: dict[str, dict[str, Any]]) -> str:
    plt.figure(figsize=(10,6))
    for model_name, data in roc_data.items():
        fpr = data["fpr"]
        tpr = data["tpr"]
        auc = data["auc"]
        if auc is None:
            continue
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    return _to_base64_png()


def _default_models_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_proba))
    except ValueError:
        return None


def _subsample_train(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples <= 0 or y.shape[0] <= max_samples:
        return X, y

    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=max_samples,
        random_state=random_state,
        stratify=y,
    )
    return X_sub, y_sub


def train_and_evaluate_classification(
    csv_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    profile: str | None = None,
    n_jobs: int | None = None,
    cv_folds: int | None = None,
    max_search_samples: int | None = None,
    max_svc_samples: int | None = None,
) -> dict[str, Any]:
    csv_path = Path(csv_path) if csv_path is not None else default_csv_path()
    models_dir = Path(models_dir) if models_dir is not None else _default_models_dir()

    classification_dir = models_dir / "classification"
    classification_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_modeling_dataframe(csv_path)
    if "readmitted_30" not in df.columns:
        raise KeyError("Target column 'readmitted_30' not found after preprocessing.")

    drop_cols = [col for col in ("readmitted_30", "readmitted", "encounter_id", "patient_nbr") if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["readmitted_30"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    y_train_np = y_train.to_numpy()

    effective_profile = (profile or os.getenv("HEALTHLENS_CLASSIFICATION_PROFILE", "safe")).strip().lower()
    if effective_profile not in {"safe", "full"}:
        effective_profile = "safe"

    if n_jobs is None:
        n_jobs = 1 if effective_profile == "safe" else -1
    if cv_folds is None:
        cv_folds = 3 if effective_profile == "safe" else 5
    if max_search_samples is None and effective_profile == "safe":
        max_search_samples = 8_000
    if max_svc_samples is None and effective_profile == "safe":
        max_svc_samples = 6_000

    logistic_grid = {"C": [0.1, 1.0, 10.0]} if effective_profile == "safe" else {"C": [0.01, 0.1, 1.0, 10.0]}
    tree_depths = [4, 8, 12] if effective_profile == "safe" else list(range(3, 11))
    svc_grid = {"C": [0.5, 1.0], "gamma": ["scale"]} if effective_profile == "safe" else {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale"],
    }
    k_values = list(range(3, 10, 2)) if effective_profile == "safe" else list(range(3, 16))
    rf_estimators = 120 if effective_profile == "safe" else 200

    thread_limit = 1 if n_jobs == 1 else None

    # 1. Fit variance selector on raw full train and transform raw train/test.
    X_train_raw = X_train.to_numpy(dtype=np.float64)
    X_test_raw = X_test.to_numpy(dtype=np.float64)

    selector = VarianceThreshold(threshold=0.009)
    X_train_reduced = selector.fit_transform(X_train_raw)
    X_test_reduced = selector.transform(X_test_raw)
    joblib.dump(selector, classification_dir / "variance_selector.pkl")
    n_features_after = int(X_train_reduced.shape[1])
    print(f"Features after variance filter: {n_features_after}")

    # 2. Scale reduced features only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reduced)
    X_test_scaled = scaler.transform(X_test_reduced)

    # 3. Subsample from reduced feature arrays.
    X_search_scaled, y_search = _subsample_train(X_train_scaled, y_train_np, max_search_samples)
    X_svc_scaled, y_svc = _subsample_train(X_train_scaled, y_train_np, max_svc_samples)

    joblib.dump(scaler, classification_dir / "scaler.pkl")
    with (classification_dir / "feature_names.json").open("w", encoding="utf-8") as fp:
        json.dump(X.columns.tolist(), fp, indent=2)

    with threadpool_limits(limits=thread_limit):
        print("Starting LogisticRegression GridSearch...")
        logistic_search = GridSearchCV(
            estimator=LogisticRegression(max_iter=1000, solver="liblinear"),
            param_grid=logistic_grid,
            scoring="f1_weighted",
            cv=cv_folds,
            n_jobs=n_jobs,
        )
        logistic_search.fit(X_search_scaled, y_search)

        print("Starting DecisionTree GridSearch...")
        tree_search = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=42),
            param_grid={"max_depth": tree_depths},
            scoring="f1_weighted",
            cv=cv_folds,
            n_jobs=n_jobs,
        )
        tree_search.fit(X_search_scaled, y_search)

        print("Starting KNN search...")
        best_k = 3
        best_k_score = -np.inf
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            cv_scores = cross_val_score(
                knn,
                X_search_scaled,
                y_search,
                cv=cv_folds,
                scoring="f1_weighted",
                n_jobs=n_jobs,
            )
            score = float(np.mean(cv_scores))
            if score > best_k_score:
                best_k_score = score
                best_k = k

        knn_model = KNeighborsClassifier(n_neighbors=best_k)
        if effective_profile == "safe":
            knn_model.fit(X_search_scaled, y_search)
        else:
            knn_model.fit(X_train_scaled, y_train_np)

        print("Starting SVC GridSearch...")
        svc_search = GridSearchCV(
            estimator=SVC(kernel="rbf", probability=True, random_state=42),
            param_grid=svc_grid,
            scoring="f1_weighted",
            cv=cv_folds,
            n_jobs=n_jobs,
        )
        svc_search.fit(X_svc_scaled, y_svc)

        print("Starting RandomForest fit...")
        random_forest = RandomForestClassifier(
            n_estimators=rf_estimators,
            class_weight="balanced",
            random_state=42,
            n_jobs=n_jobs,
        )
        random_forest.fit(X_search_scaled, y_search)

    print("All models trained. Computing test metrics...")

    trained_models: dict[str, tuple[Any, dict[str, Any]]] = {
        "LogisticRegression": (logistic_search.best_estimator_, logistic_search.best_params_),
        "DecisionTreeClassifier": (tree_search.best_estimator_, tree_search.best_params_),
        "RandomForestClassifier": (
            random_forest,
            {"n_estimators": rf_estimators, "class_weight": "balanced"},
        ),
        "KNeighborsClassifier": (knn_model, {"n_neighbors": best_k}),
        "SVC": (svc_search.best_estimator_, svc_search.best_params_),
    }

    model_results: dict[str, dict[str, Any]] = {}
    roc_data: dict[str, dict[str, Any]] = {}

    for model_name, (model, params) in trained_models.items():
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        auc = _safe_auc(y_test.to_numpy(), y_proba)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_plot_b64 = _confusion_matrix_plot(cm, model_name)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc,
        }

        model_filename = model_name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, classification_dir / model_filename)

        model_results[model_name] = {
            "model_name": model_name,
            "best_params": params,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_weighted": f1,
                "auc_roc": auc,
            },
            "confusion_matrix_plot": cm_plot_b64,
        }

    roc_plot_b64 = _roc_overlay_plot(roc_data)

    summary: dict[str, Any] = {
        "task": "classification",
        "target": "readmitted_30",
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
        "training_profile": effective_profile,
        "search_train_samples": int(X_search_scaled.shape[0]),
        "svc_train_samples": int(X_svc_scaled.shape[0]),
        "cv_folds": int(cv_folds),
        "n_jobs": int(n_jobs),
        "n_features_after_variance": n_features_after,
        "models": model_results,
        "roc_curve_plot": roc_plot_b64,
    }

    with (classification_dir / "results.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp)

    return summary
