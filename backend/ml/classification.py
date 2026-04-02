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


def train_and_evaluate_classification(
    csv_path: str | Path | None = None,
    models_dir: str | Path | None = None,
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, classification_dir / "scaler.pkl")
    with (classification_dir / "feature_names.json").open("w", encoding="utf-8") as fp:
        json.dump(X.columns.tolist(), fp, indent=2)

    logistic_search = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000, solver="liblinear"),
        param_grid={"C": [0.01, 0.1, 1.0, 10.0]},
        scoring="f1_weighted",
        cv=5,
        n_jobs=-1,
    )
    logistic_search.fit(X_train_scaled, y_train)

    tree_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid={"max_depth": list(range(3, 11))},
        scoring="f1_weighted",
        cv=5,
        n_jobs=-1,
    )
    tree_search.fit(X_train_scaled, y_train)

    best_k = 3
    best_k_score = -np.inf
    for k in range(3, 16):
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(
            knn,
            X_train_scaled,
            y_train,
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        score = float(np.mean(cv_scores))
        if score > best_k_score:
            best_k_score = score
            best_k = k

    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_model.fit(X_train_scaled, y_train)

    svc_search = GridSearchCV(
        estimator=SVC(kernel="rbf", probability=True, random_state=42),
        param_grid={
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", 0.1, 0.01],
        },
        scoring="f1_weighted",
        cv=5,
        n_jobs=-1,
    )
    svc_search.fit(X_train_scaled, y_train)

    random_forest = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    random_forest.fit(X_train_scaled, y_train)

    trained_models: dict[str, tuple[Any, dict[str, Any]]] = {
        "LogisticRegression": (logistic_search.best_estimator_, logistic_search.best_params_),
        "DecisionTreeClassifier": (tree_search.best_estimator_, tree_search.best_params_),
        "RandomForestClassifier": (
            random_forest,
            {"n_estimators": 200, "class_weight": "balanced"},
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
        "models": model_results,
        "roc_curve_plot": roc_plot_b64,
    }

    with (classification_dir / "results.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp)

    return summary
