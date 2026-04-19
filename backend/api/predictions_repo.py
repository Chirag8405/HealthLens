from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db import Prediction


def serialize_prediction(record: Prediction) -> dict[str, Any]:
    return {
        "id": str(record.id),
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "patient_ref": record.patient_ref,
        "risk_level": record.risk_level,
        "risk_score": record.risk_score,
        "rf_confidence": record.rf_confidence,
        "top_factors": record.top_factors,
        "outcome_30d": record.outcome_30d,
        "clinician_ack": record.clinician_ack,
        "ack_at": record.ack_at.isoformat() if record.ack_at else None,
    }


async def insert_prediction(
    session: AsyncSession,
    *,
    patient_ref: str,
    risk_level: str,
    risk_score: float,
    rf_confidence: float,
    top_factors: list[dict[str, Any]],
) -> Prediction:
    record = Prediction(
        patient_ref=patient_ref,
        risk_level=risk_level,
        risk_score=risk_score,
        rf_confidence=rf_confidence,
        top_factors=top_factors,
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return record


async def get_recent_predictions(session: AsyncSession, limit: int = 20) -> list[Prediction]:
    stmt = select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
    rows = await session.execute(stmt)
    return list(rows.scalars().all())


async def update_prediction_outcome(
    session: AsyncSession,
    prediction_id: uuid.UUID,
    outcome_30d: bool,
) -> Prediction | None:
    record = await session.get(Prediction, prediction_id)
    if record is None:
        return None

    record.outcome_30d = outcome_30d
    record.clinician_ack = True
    record.ack_at = datetime.utcnow()
    await session.commit()
    await session.refresh(record)
    return record
