from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import tensorflow as tf
from fastapi import APIRouter
from fastapi import File
from fastapi import HTTPException
from fastapi import Query
from fastapi import UploadFile

from ml.ann import train_and_evaluate_ann
from ml.autoencoder import default_autoencoder_dataset_root
from ml.autoencoder import train_and_evaluate_autoencoder
from ml.cnn import default_cnn_dataset_root
from ml.cnn import predict_cnn_image
from ml.cnn import train_and_evaluate_cnn
from ml.data_utils import default_csv_path
from ml.lstm import evaluate_lstm_task_b_risk_only
from ml.lstm import predict_lstm_sepsis_risk
from ml.lstm import train_and_evaluate_lstm
from ml.model_registry import unload_heavy_models

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
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

        if tier_keys == {"ROUTINE", "ELEVATED", "HIGH"}:
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
        result = predict_lstm_sepsis_risk(patient_psv_bytes=payload, models_dir=MODELS_DIR)
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

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predict_cnn_image(image_bytes=image_bytes, models_dir=MODELS_DIR)
        unload_heavy_models()
        gc.collect()
        tf.keras.backend.clear_session()
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CNN prediction failed: {exc}") from exc
