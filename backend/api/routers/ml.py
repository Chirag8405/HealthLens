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

CLASSIFICATION_RESULTS_PATH = MODELS_DIR / "classification" / "results.json"
REGRESSION_RESULTS_PATH = MODELS_DIR / "regression" / "results.json"
CLUSTERING_RESULTS_PATH = MODELS_DIR / "clustering" / "results.json"


class TrainRequest(BaseModel):
    csv_path: str | None = None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data


@router.post("/train")
def train_models(request: TrainRequest | None = None) -> dict[str, Any]:
    csv_path = request.csv_path if request and request.csv_path else str(default_csv_path())

    try:
        regression = train_and_evaluate_regression(csv_path=csv_path, models_dir=MODELS_DIR)
        classification = train_and_evaluate_classification(csv_path=csv_path, models_dir=MODELS_DIR)
        clustering = run_clustering(csv_path=csv_path, models_dir=MODELS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ML training failed: {exc}") from exc

    summary = {
        "status": "trained",
        "csv_path": csv_path,
        "artifacts_dir": str(MODELS_DIR),
        "regression_models": list(regression["models"].keys()),
        "classification_models": list(classification["models"].keys()),
        "clustering_algorithms": ["kmeans", "agglomerative"],
        "clustering_samples": clustering.get("n_samples"),
    }
    return {"summary": summary}


@router.get("/results")
def get_results() -> dict[str, Any]:
    classification = _load_json(CLASSIFICATION_RESULTS_PATH)
    regression = _load_json(REGRESSION_RESULTS_PATH)

    if classification is None or regression is None:
        csv_path = str(default_csv_path())
        try:
            regression = train_and_evaluate_regression(csv_path=csv_path, models_dir=MODELS_DIR)
            classification = train_and_evaluate_classification(csv_path=csv_path, models_dir=MODELS_DIR)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to generate ML results: {exc}") from exc

    return {
        "regression": regression,
        "classification": classification,
    }


@router.get("/clusters")
def get_clusters() -> dict[str, Any]:
    clustering = _load_json(CLUSTERING_RESULTS_PATH)

    if clustering is None:
        csv_path = str(default_csv_path())
        try:
            clustering = run_clustering(csv_path=csv_path, models_dir=MODELS_DIR)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to generate clustering results: {exc}") from exc

    return clustering
