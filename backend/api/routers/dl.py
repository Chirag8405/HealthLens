from __future__ import annotations

import gc
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import APIRouter
from fastapi import File
from fastapi import HTTPException
from fastapi import Query
from fastapi import UploadFile

from ml.ann import train_and_evaluate_ann
from ml.autoencoder import default_autoencoder_dataset_root
from ml.autoencoder import train_and_evaluate_autoencoder
from ml.cnn import generate_gradcam_overlay
from ml.cnn import default_cnn_dataset_root
from ml.cnn import train_and_evaluate_cnn
from ml.data_utils import default_csv_path
from ml.lstm import FEATURE_COLUMNS
from ml.lstm import MAX_HOURS
from ml.lstm import STRIDE
from ml.lstm import WINDOW_SIZE
from ml.lstm import evaluate_lstm_task_b_risk_only
from ml.lstm import load_lstm_sepsis_tier_thresholds
from ml.lstm import train_and_evaluate_lstm
from ml.model_registry import get_model
from ml.model_registry import unload_heavy_models
from path_utils import project_root_from

router = APIRouter()

PROJECT_ROOT = project_root_from(__file__)
MODELS_DIR = PROJECT_ROOT / "models"
ANN_RESULTS_PATH = MODELS_DIR / "ann_results.json"
CNN_RESULTS_PATH = MODELS_DIR / "cnn_results.json"
AUTOENCODER_RESULTS_PATH = MODELS_DIR / "autoencoder_results.json"
LSTM_RESULTS_PATH = MODELS_DIR / "lstm_results.json"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _inverse_feature_value(value: float, scaler: Any, col_idx: int) -> float:
    if not hasattr(scaler, "data_min_") or not hasattr(scaler, "data_range_"):
        return float(value)

    min_value = float(scaler.data_min_[col_idx])
    range_value = float(scaler.data_range_[col_idx])
    if abs(range_value) < 1e-12:
        return min_value
    return float(value) * range_value + min_value


def generate_vitals_forecast(
    vitals_model: Any,
    X_sequence: np.ndarray,
    scaler: Any,
    hr_col_idx: int,
    spo2_col_idx: int,
) -> list[dict[str, float | int]]:
    """
    X_sequence shape: (timesteps, n_features)
    Returns per-step actual and predicted HR/O2Sat in clinical units.
    """
    forecasts: list[dict[str, float | int]] = []
    window_size = int(WINDOW_SIZE)
    n_steps = int(X_sequence.shape[0])

    for t in range(window_size, n_steps):
        window = X_sequence[t - window_size : t]
        window_input = np.expand_dims(window, 0)
        pred = vitals_model.predict(window_input, verbose=0)[0]

        actual_hr_scaled = float(X_sequence[t, hr_col_idx])
        actual_spo2_scaled = float(X_sequence[t, spo2_col_idx])
        pred_hr_scaled = float(pred[0])
        pred_spo2_scaled = float(pred[1])

        actual_hr = _inverse_feature_value(actual_hr_scaled, scaler, hr_col_idx)
        actual_spo2 = _inverse_feature_value(actual_spo2_scaled, scaler, spo2_col_idx)
        pred_hr = _inverse_feature_value(pred_hr_scaled, scaler, hr_col_idx)
        pred_spo2 = _inverse_feature_value(pred_spo2_scaled, scaler, spo2_col_idx)

        forecasts.append(
            {
                "hour": int(t - window_size + 1),
                "hr_actual": round(float(actual_hr), 1),
                "hr_predicted": round(float(pred_hr), 1),
                "spo2_actual": round(float(actual_spo2), 1),
                "spo2_predicted": round(float(pred_spo2), 1),
            }
        )

    return forecasts


def _compute_trend(values: list[float]) -> str:
    if len(values) < 3:
        return "stable"
    delta = values[-1] - values[0]
    if delta > 3:
        return "rising"
    if delta < -3:
        return "declining"
    return "stable"


def _risk_label(risk_tier: str) -> str:
    labels = {
        "LOW": "LOW Sepsis Risk - Continue monitoring",
        "MEDIUM": "MEDIUM Sepsis Risk - Monitor closely",
        "HIGH": "HIGH Sepsis Risk - Immediate clinical review advised",
    }
    return labels.get(risk_tier.upper(), "Risk assessment unavailable")


def _load_lstm_scaler(models_dir: Path) -> Any:
    # New canonical path, with fallback to historical artifact name.
    scaler_path = models_dir / "lstm_scaler.pkl"
    if scaler_path.exists():
        return joblib.load(scaler_path)

    legacy_path = models_dir / "lstm_minmax_scaler.pkl"
    if legacy_path.exists():
        return joblib.load(legacy_path)

    raise FileNotFoundError(f"LSTM scaler not found at {scaler_path} or {legacy_path}")


def _load_lstm_feature_cols(models_dir: Path) -> dict[str, int]:
    cols_path = models_dir / "lstm_feature_cols.json"
    if cols_path.exists():
        with cols_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return {
            "hr_col_idx": int(payload.get("hr_col_idx", 0)),
            "spo2_col_idx": int(payload.get("spo2_col_idx", 1)),
        }

    return {
        "hr_col_idx": int(FEATURE_COLUMNS.index("HR")),
        "spo2_col_idx": int(FEATURE_COLUMNS.index("O2Sat")),
    }


def _read_patient_psv(patient_psv_bytes: bytes) -> np.ndarray:
    try:
        df = pd.read_csv(BytesIO(patient_psv_bytes), sep="|")
    except Exception as exc:
        raise ValueError("Invalid PSV file. Could not parse patient time-series data.") from exc

    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in patient file: {missing_cols}")

    patient_df = df[FEATURE_COLUMNS].iloc[:MAX_HOURS].copy()
    if patient_df.empty:
        raise ValueError("Patient file has no rows.")

    patient_df = patient_df.replace(-1, np.nan).ffill().bfill().fillna(0.0)
    return patient_df.to_numpy(dtype=np.float32)


@router.get("/ann")
def get_ann_results(retrain: bool = Query(default=False)) -> dict[str, Any]:
    if not retrain:
        cached = _load_json(ANN_RESULTS_PATH)
        if cached is not None:
            return cached

    csv_path = str(default_csv_path())
    try:
        return train_and_evaluate_ann(csv_path=csv_path, models_dir=MODELS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ANN training failed: {exc}") from exc


@router.post("/cnn/train")
def train_cnn() -> dict[str, Any]:
    dataset_root = str(default_cnn_dataset_root())

    try:
        results = train_and_evaluate_cnn(dataset_root=dataset_root, models_dir=MODELS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CNN training failed: {exc}") from exc

    return {
        "status": "trained",
        "model_path": results.get("model_path"),
        "metrics": results.get("metrics", {}),
    }


@router.get("/cnn/results")
def get_cnn_results() -> dict[str, Any]:
    results = _load_json(CNN_RESULTS_PATH)
    if results is None:
        raise HTTPException(
            status_code=404,
            detail="CNN results not found. Train first via POST /dl/cnn/train.",
        )

    return results


@router.get("/autoencoder/results")
def get_autoencoder_results(retrain: bool = Query(default=False)) -> dict[str, Any]:
    if not retrain:
        cached = _load_json(AUTOENCODER_RESULTS_PATH)
        if cached is not None:
            return cached

    dataset_root = str(default_autoencoder_dataset_root())
    try:
        return train_and_evaluate_autoencoder(dataset_root=dataset_root, models_dir=MODELS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Autoencoder training failed: {exc}") from exc


@router.get("/lstm/results")
def get_lstm_results(
    retrain: bool = Query(default=False),
    max_patients: int | None = Query(default=1200, ge=100),
) -> dict[str, Any]:
    cached = _load_json(LSTM_RESULTS_PATH)

    if not retrain and cached is not None:
        task_b_cached = cached.get("task_b_sepsis", {})
        tier_keys: set[str] = set()
        if isinstance(task_b_cached, dict):
            tier_keys = set(task_b_cached.get("risk_tiers", {}).keys())

        if tier_keys == {"LOW", "MEDIUM", "HIGH"}:
            return cached

        inferred_max_patients = cached.get("data", {}).get("patients_used", max_patients)
        try:
            task_b_results = evaluate_lstm_task_b_risk_only(
                models_dir=MODELS_DIR,
                max_patients=inferred_max_patients,
            )
        except FileNotFoundError:
            return cached
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LSTM risk refresh failed: {exc}") from exc

        cached["task_b_sepsis"] = task_b_results
        cached.setdefault("artifacts", {})["sepsis_tiers_path"] = str(MODELS_DIR / "lstm_sepsis_tiers.json")
        with LSTM_RESULTS_PATH.open("w", encoding="utf-8") as fp:
            json.dump(cached, fp)
        return cached

    if not retrain and cached is None:
        try:
            task_b_results = evaluate_lstm_task_b_risk_only(
                models_dir=MODELS_DIR,
                max_patients=max_patients,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"{exc}. Train first via GET /dl/lstm/results?retrain=true.",
            ) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LSTM inference failed: {exc}") from exc

        return {
            "task": "physionet_lstm_multitask",
            "task_b_sepsis": task_b_results,
            "artifacts": {
                "sepsis_tiers_path": str(MODELS_DIR / "lstm_sepsis_tiers.json"),
            },
        }

    try:
        return train_and_evaluate_lstm(models_dir=MODELS_DIR, max_patients=max_patients)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LSTM training failed: {exc}") from exc


@router.post("/lstm/predict")
async def predict_lstm(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type is not None and file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected a PhysioNet PSV file, not an image.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        patient_features = _read_patient_psv(payload)

        vitals_model = get_model("lstm_vitals")
        sepsis_model = get_model("lstm_sepsis")
        scaler = _load_lstm_scaler(MODELS_DIR)
        feature_cols = _load_lstm_feature_cols(MODELS_DIR)

        hr_col_idx = int(feature_cols["hr_col_idx"])
        spo2_col_idx = int(feature_cols["spo2_col_idx"])

        scaled_features = scaler.transform(patient_features).astype(np.float32)

        forecast = generate_vitals_forecast(
            vitals_model=vitals_model,
            X_sequence=scaled_features,
            scaler=scaler,
            hr_col_idx=hr_col_idx,
            spo2_col_idx=spo2_col_idx,
        )

        risk_features = scaled_features
        if risk_features.shape[0] < WINDOW_SIZE:
            pad_rows = WINDOW_SIZE - risk_features.shape[0]
            pad = np.repeat(risk_features[[0]], pad_rows, axis=0)
            risk_features = np.vstack([pad, risk_features]).astype(np.float32)

        windows: list[np.ndarray] = []
        max_start = risk_features.shape[0] - WINDOW_SIZE
        for start in range(0, max_start + 1, STRIDE):
            end = start + WINDOW_SIZE
            windows.append(risk_features[start:end, :])

        X_patient = np.asarray(windows, dtype=np.float32)
        window_scores = sepsis_model.predict(X_patient, verbose=0).ravel()
        risk_score = float(window_scores.max()) if window_scores.size > 0 else 0.0

        tier_thresholds = load_lstm_sepsis_tier_thresholds(MODELS_DIR)
        medium_threshold = float(tier_thresholds.get("MEDIUM", 0.15))
        high_threshold = float(tier_thresholds.get("HIGH", 0.35))

        if risk_score >= high_threshold:
            risk_tier = "HIGH"
        elif risk_score >= medium_threshold:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "LOW"

        hr_pred_values = [float(item["hr_predicted"]) for item in forecast]
        spo2_pred_values = [float(item["spo2_predicted"]) for item in forecast]

        if hr_pred_values:
            hr_final = round(float(hr_pred_values[-1]), 1)
        else:
            hr_final = round(float(patient_features[-1, hr_col_idx]), 1)

        if spo2_pred_values:
            spo2_final = round(float(spo2_pred_values[-1]), 1)
        else:
            spo2_final = round(float(patient_features[-1, spo2_col_idx]), 1)

        result = {
            "sepsis_risk": round(risk_score, 4),
            "risk_tier": risk_tier,
            "risk_label": _risk_label(risk_tier),
            "forecast": forecast,
            "summary": {
                "hr_trend": _compute_trend(hr_pred_values),
                "spo2_trend": _compute_trend(spo2_pred_values),
                "hr_final_predicted": hr_final,
                "spo2_final_predicted": spo2_final,
            },
        }

        unload_heavy_models()
        gc.collect()
        tf.keras.backend.clear_session()
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LSTM prediction failed: {exc}") from exc


@router.post("/cnn/predict")
async def predict_cnn(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Invalid image file. Could not decode image bytes.")

        original_img = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)

        img_array = original_img.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, 0)

        # Keep model input aligned with MobileNetV2 preprocessing used in training.
        preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(
            (img_array * 255.0).astype(np.float32)
        )

        cnn_model = get_model("cnn")
        raw_prob = float(cnn_model.predict(preprocessed_img, verbose=0)[0][0])
        predicted_class_index = 1 if raw_prob >= 0.5 else 0

        label = "PNEUMONIA" if predicted_class_index == 1 else "NORMAL"
        confidence = raw_prob if predicted_class_index == 1 else (1.0 - raw_prob)

        gradcam_b64 = generate_gradcam_overlay(
            model=cnn_model,
            img_array=preprocessed_img,
            original_img=original_img,
            pred_class=predicted_class_index,
        )

        result = {
            "label": label,
            "confidence": round(float(confidence), 4),
            "gradcam_b64": gradcam_b64,
        }

        unload_heavy_models()
        gc.collect()
        tf.keras.backend.clear_session()
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CNN prediction failed: {exc}") from exc
