from __future__ import annotations

import base64
import gc
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def _actual_vs_predicted_plot(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> str:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.35, color="#2563eb", edgecolors="none")

    min_value = float(min(np.min(y_true), np.min(y_pred)))
    max_value = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([min_value, max_value], [min_value, max_value], "r--", linewidth=2)

    plt.title(f"Actual vs Predicted - {model_name}")
    plt.xlabel("Actual time_in_hospital")
    plt.ylabel("Predicted time_in_hospital")
    return _to_base64_png()


def _default_models_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


def train_and_evaluate_regression(
    csv_path: str | Path | None = None,
    models_dir: str | Path | None = None,
) -> dict[str, Any]:
    csv_path = Path(csv_path) if csv_path is not None else default_csv_path()
    models_dir = Path(models_dir) if models_dir is not None else _default_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    regression_dir = models_dir / "regression"
    regression_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_modeling_dataframe(csv_path)
    if "time_in_hospital" not in df.columns:
        raise KeyError("Target column 'time_in_hospital' not found after preprocessing.")

    id_cols = [col for col in ("encounter_id", "patient_nbr") if col in df.columns]
    drop_cols = ["time_in_hospital"] + id_cols

    X = df.drop(columns=drop_cols)
    y = df["time_in_hospital"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, regression_dir / "scaler.pkl")
    with (regression_dir / "feature_names.json").open("w", encoding="utf-8") as fp:
        json.dump(X.columns.tolist(), fp, indent=2)

    models: dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5),
        "Lasso": LassoCV(alphas=np.logspace(-4, 1, 40), cv=5, random_state=42, max_iter=20000),
    }

    model_results: dict[str, dict[str, Any]] = {}
    regression_payload: dict[str, Any] = {}

    api_key_map = {
        "LinearRegression": "linear_regression",
        "Ridge": "ridge",
        "Lasso": "lasso",
    }

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        metrics: dict[str, Any] = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

        if hasattr(model, "alpha_"):
            metrics["best_alpha"] = float(model.alpha_)

        scatter_plot_b64 = _actual_vs_predicted_plot(y_test.to_numpy(), y_pred, model_name)

        model_filename = model_name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, regression_dir / model_filename)

        model_results[model_name] = {
            "model_name": model_name,
            "metrics": metrics,
            "actual_vs_predicted_plot": scatter_plot_b64,
        }
        regression_payload[api_key_map[model_name]] = {
            **metrics,
            "actual_vs_predicted_b64": scatter_plot_b64,
        }

        del model
        gc.collect()
        print(f"[ml/train] {model_name} regression done, memory freed")

    regression_payload["meta"] = {
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
    }
    regression_payload["trained_at"] = datetime.now().isoformat()

    with (models_dir / "regression_results.json").open("w", encoding="utf-8") as fp:
        json.dump(regression_payload, fp)

    summary: dict[str, Any] = {
        "task": "regression",
        "target": "time_in_hospital",
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
        "models": model_results,
    }

    with (regression_dir / "results.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp)

    return summary