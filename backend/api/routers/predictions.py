from __future__ import annotations

import uuid

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.db import get_async_session
from api.predictions_repo import get_recent_predictions
from api.predictions_repo import serialize_prediction
from api.predictions_repo import update_prediction_outcome

router = APIRouter(prefix="/predictions")


class OutcomeUpdateRequest(BaseModel):
    outcome_30d: bool


@router.get("/recent")
async def recent_predictions(
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, list[dict[str, object]]]:
    rows = await get_recent_predictions(session=session, limit=20)
    return {"predictions": [serialize_prediction(row) for row in rows]}


@router.post("/{id}/outcome")
async def record_prediction_outcome(
    id: uuid.UUID,
    payload: OutcomeUpdateRequest,
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, object]:
    updated = await update_prediction_outcome(
        session=session,
        prediction_id=id,
        outcome_30d=payload.outcome_30d,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return serialize_prediction(updated)
