from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi import File
from fastapi import HTTPException
from fastapi import Query
from fastapi import UploadFile

from ml.ann import train_and_evaluate_ann
from ml.cnn import default_cnn_dataset_root
from ml.cnn import predict_cnn_image
from ml.cnn import train_and_evaluate_cnn
from ml.data_utils import default_csv_path

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
ANN_RESULTS_PATH = MODELS_DIR / "ann_results.json"
CNN_RESULTS_PATH = MODELS_DIR / "cnn_results.json"


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


@router.post("/cnn/predict")
async def predict_cnn(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return predict_cnn_image(image_bytes=image_bytes, models_dir=MODELS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CNN prediction failed: {exc}") from exc
