from __future__ import annotations

import os
from typing import Any

import psutil
from fastapi import APIRouter

from ml.model_registry import loaded_models

router = APIRouter(prefix="/health")


@router.get("")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/memory")
def memory_status() -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    return {
        "rss_mb": round(mem.rss / 1024 / 1024, 1),
        "vms_mb": round(mem.vms / 1024 / 1024, 1),
        "system_available_gb": round(psutil.virtual_memory().available / 1024**3, 2),
        "loaded_models": loaded_models(),
    }
