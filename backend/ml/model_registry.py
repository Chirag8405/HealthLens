from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import joblib

_registry: dict[str, Any] = {}


def _models_base_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


def _first_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _get_path(name: str) -> str:
    base = _models_base_dir()
    paths: dict[str, list[Path]] = {
        "ann": [base / "ann_best.h5", base / "ann_model.h5"],
        "cnn": [base / "cnn_model.h5"],
        "autoencoder": [base / "autoencoder.h5"],
        "lstm_vitals": [base / "lstm_vitals.h5"],
        "lstm_sepsis": [base / "lstm_sepsis.h5"],
        "rf": [base / "rf_model.pkl", base / "classification" / "randomforestclassifier.joblib"],
        "svm": [base / "svm_model.pkl", base / "classification" / "svc.joblib"],
        "scaler": [base / "scaler.pkl", base / "classification" / "scaler.pkl"],
        "threshold": [base / "ann_threshold.json"],
        "ann_scaler": [base / "ann_scaler.pkl"],
        "classification_scaler": [base / "classification" / "scaler.pkl"],
        "feature_names": [base / "classification" / "feature_names.json"],
        "classification_selector": [base / "classification" / "variance_selector.pkl"],
        "lstm_scaler": [base / "lstm_minmax_scaler.pkl"],
    }

    if name not in paths:
        raise KeyError(f"Unknown model key: {name}")

    return str(_first_existing_path(paths[name]))


def get_model(name: str) -> Any:
    """Load model on first access, cache in memory after that."""
    if name not in _registry:
        path = _get_path(name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}. Run training first.")

        if path.endswith(".h5"):
            import tensorflow as tf

            _registry[name] = tf.keras.models.load_model(path, compile=False)
        elif path.endswith(".pkl") or path.endswith(".joblib"):
            _registry[name] = joblib.load(path)
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                _registry[name] = json.load(f)
        else:
            raise ValueError(f"Unsupported artifact extension for: {path}")

        print(f"[registry] Loaded {name} from {path}")

    return _registry[name]


def loaded_models() -> list[str]:
    return list(_registry.keys())


def unload_model(name: str) -> None:
    """Free memory by unloading a specific model."""
    if name in _registry:
        del _registry[name]
        gc.collect()
        print(f"[registry] Unloaded {name}")


def unload_heavy_models() -> None:
    """Call after inference to free Keras models from RAM."""
    for name in ["cnn", "autoencoder", "lstm_vitals", "lstm_sepsis"]:
        unload_model(name)
