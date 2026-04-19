from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import Text
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://healthlens:healthlens@localhost:5432/healthlens",
)

engine = create_async_engine(DATABASE_URL, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    patient_ref: Mapped[str] = mapped_column(Text, nullable=False)
    risk_level: Mapped[str] = mapped_column(Text, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    rf_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    top_factors: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    outcome_30d: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    clinician_ack: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false", default=False)
    ack_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


async def get_async_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def dispose_engine() -> None:
    await engine.dispose()
