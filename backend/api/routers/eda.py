from typing import Any

from fastapi import APIRouter, HTTPException

from ml.eda import EDA

router = APIRouter(prefix="/eda")


@router.get("/plots")
def get_eda_plots() -> dict[str, dict[str, str]]:
    try:
        eda = EDA()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    plots = {
        "age": eda.age_distribution(),
        "readmission": eda.readmission_rates(),
        "correlation": eda.correlation_heatmap(),
        "los_vs_cost": eda.los_vs_cost(),
        "diagnosis": eda.diagnosis_frequency(),
        "imbalance": eda.class_imbalance(),
    }
    return {"plots": plots}


@router.get("/summary")
def get_eda_summary() -> dict[str, dict[str, Any]]:
    try:
        eda = EDA()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"summary": eda.summary()}
