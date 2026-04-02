from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Query

from ml.ann import train_and_evaluate_ann
from ml.data_utils import default_csv_path

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
ANN_RESULTS_PATH = MODELS_DIR / "ann_results.json"


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
