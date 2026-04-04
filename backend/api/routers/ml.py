from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel

from ml.classification import train_and_evaluate_classification
from ml.clustering import run_clustering
from ml.data_utils import default_csv_path
from ml.regression import train_and_evaluate_regression

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"

CLASSIFICATION_RESULTS_PATH = MODELS_DIR / "ml_results.json"
REGRESSION_RESULTS_PATH = MODELS_DIR / "regression_results.json"
CLUSTERING_RESULTS_PATH = MODELS_DIR / "clustering_results.json"


class TrainRequest(BaseModel):
    csv_path: str | None = None
    skip_svm: bool = False


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_required_json(path: Path, detail: str) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=detail)
    return _load_json(path)


def _strip_base64_fields(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: _strip_base64_fields(value)
            for key, value in payload.items()
            if not key.endswith("_b64")
        }
    if isinstance(payload, list):
        return [_strip_base64_fields(item) for item in payload]
    return payload


@router.post("/train")
def train_models(request: TrainRequest | None = None) -> dict[str, Any]:
    csv_path = request.csv_path if request and request.csv_path else str(default_csv_path())
    skip_svm = request.skip_svm if request else False

    try:
        regression = train_and_evaluate_regression(csv_path=csv_path, models_dir=MODELS_DIR)
        classification = train_and_evaluate_classification(
            csv_path=csv_path,
            models_dir=MODELS_DIR,
            skip_svm=skip_svm,
        )
        clustering = run_clustering(csv_path=csv_path, models_dir=MODELS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ML training failed: {exc}") from exc

    summary = {
        "status": "trained",
        "csv_path": csv_path,
        "skip_svm": skip_svm,
        "artifacts_dir": str(MODELS_DIR),
        "regression_models": list(regression.get("models", {}).keys()),
        "classification_models": list(classification.get("models", {}).keys()),
        "clustering_algorithms": ["kmeans", "agglomerative"],
        "clustering_samples": clustering.get("n_samples"),
    }
    return {"summary": summary}


@router.get("/results")
def get_ml_results() -> dict[str, Any]:
    classification = _load_required_json(
        CLASSIFICATION_RESULTS_PATH,
        "ML results not found. Run POST /ml/train first.",
    )

    regression: dict[str, Any] = {}
    if REGRESSION_RESULTS_PATH.exists():
        regression = _load_json(REGRESSION_RESULTS_PATH)

    return {
        "classification": classification,
        "regression": regression,
    }


@router.get("/results/classification")
def get_classification_results() -> dict[str, Any]:
    return _load_required_json(
        CLASSIFICATION_RESULTS_PATH,
        "ML results not found. Run POST /ml/train first.",
    )


@router.get("/results/regression")
def get_regression_results() -> dict[str, Any]:
    return _load_required_json(
        REGRESSION_RESULTS_PATH,
        "Regression results not found. Run POST /ml/train first.",
    )


@router.get("/results/summary")
def get_ml_summary() -> dict[str, Any]:
    classification_exists = CLASSIFICATION_RESULTS_PATH.exists()
    regression_exists = REGRESSION_RESULTS_PATH.exists()

    if not classification_exists and not regression_exists:
        return {"status": "not_trained"}

    summary: dict[str, Any] = {"status": "trained"}

    if classification_exists:
        classification = _load_json(CLASSIFICATION_RESULTS_PATH)
        summary["classification"] = _strip_base64_fields(classification)

    if regression_exists:
        regression = _load_json(REGRESSION_RESULTS_PATH)
        summary["regression"] = _strip_base64_fields(regression)

    return summary


@router.get("/clusters")
def get_clusters() -> dict[str, Any]:
    return _load_required_json(
        CLUSTERING_RESULTS_PATH,
        "Clustering results not found. Run POST /ml/train first.",
    )