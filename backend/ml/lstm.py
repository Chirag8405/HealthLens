from __future__ import annotations

import base64
import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
]

TARGET_COLUMN = "SepsisLabel"
MAX_HOURS = 48
WINDOW_SIZE = 12
STRIDE = 1
SEPSIS_HORIZON = 6
EPOCHS = 50
SEPSIS_EPOCHS = 80
BATCH_SIZE = 64
MAX_PATIENTS_DEFAULT = 1200
SEPSIS_TIER_THRESHOLDS = {
    "MODERATE": 0.15,
    "HIGH": 0.25,
    "CRITICAL": 0.35,
}
SEPSIS_TIER_NOTE = "Empirical thresholds based on training distribution"
SEPSIS_RISK_NOTE = (
    "Risk scores are relative rankings within this cohort. "
    "Thresholds are empirical and require clinical validation "
    "before operational use."
)


def _default_models_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


def default_lstm_dataset_dirs() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[2]
    candidate_pairs = [
        (project_root / "training" / "setA", project_root / "training" / "setB"),
        (
            project_root / "archive (2)" / "training_setA" / "training",
            project_root / "archive (2)" / "training_setB" / "training_setB",
        ),
    ]

    for set_a_dir, set_b_dir in candidate_pairs:
        if set_a_dir.exists() and set_b_dir.exists():
            return set_a_dir, set_b_dir

    raise FileNotFoundError(
        "PhysioNet dataset directories not found. Expected training/setA + training/setB "
        "or archive (2)/training_setA/training + archive (2)/training_setB/training_setB."
    )


def _to_base64_png() -> str:
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    plt.close()
    return encoded


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return None


def _lstm_sepsis_tiers_path(models_path: Path) -> Path:
    return models_path / "lstm_sepsis_tiers.json"


def save_lstm_sepsis_tier_thresholds(
    models_dir: str | Path | None = None,
    thresholds: dict[str, float] | None = None,
) -> Path:
    models_path = Path(models_dir) if models_dir is not None else _default_models_dir()
    models_path.mkdir(parents=True, exist_ok=True)

    resolved = dict(SEPSIS_TIER_THRESHOLDS)
    if thresholds is not None:
        resolved.update({k: float(v) for k, v in thresholds.items()})

    payload = {
        "thresholds": {
            "MODERATE": float(resolved["MODERATE"]),
            "HIGH": float(resolved["HIGH"]),
            "CRITICAL": float(resolved["CRITICAL"]),
        },
        "note": SEPSIS_TIER_NOTE,
    }

    path = _lstm_sepsis_tiers_path(models_path)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return path


def load_lstm_sepsis_tier_thresholds(models_dir: str | Path | None = None) -> dict[str, float]:
    models_path = Path(models_dir) if models_dir is not None else _default_models_dir()
    path = _lstm_sepsis_tiers_path(models_path)

    if not path.exists():
        save_lstm_sepsis_tier_thresholds(models_path)
        return dict(SEPSIS_TIER_THRESHOLDS)

    try:
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        raw = payload.get("thresholds", {})
        return {
            "MODERATE": float(raw.get("MODERATE", SEPSIS_TIER_THRESHOLDS["MODERATE"])),
            "HIGH": float(raw.get("HIGH", SEPSIS_TIER_THRESHOLDS["HIGH"])),
            "CRITICAL": float(raw.get("CRITICAL", SEPSIS_TIER_THRESHOLDS["CRITICAL"])),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        save_lstm_sepsis_tier_thresholds(models_path)
        return dict(SEPSIS_TIER_THRESHOLDS)


def assign_risk_tiers(
    proba: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> np.ndarray:
    resolved = dict(SEPSIS_TIER_THRESHOLDS) if thresholds is None else thresholds
    values = np.asarray(proba, dtype=np.float32).reshape(-1)

    tiers = np.full(len(values), "LOW", dtype=object)
    tiers[values >= float(resolved["MODERATE"])] = "MODERATE"
    tiers[values >= float(resolved["HIGH"])] = "HIGH"
    tiers[values >= float(resolved["CRITICAL"])] = "CRITICAL"
    return tiers


def recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)

    top_k_idx = np.argsort(y_prob_arr)[::-1][:k]
    captured = y_true_arr[top_k_idx].sum()
    total_pos = y_true_arr.sum()
    return float(captured / total_pos) if total_pos > 0 else 0.0


def ppv_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)

    top_k_idx = np.argsort(y_prob_arr)[::-1][:k]
    return float(y_true_arr[top_k_idx].mean()) if len(top_k_idx) > 0 else 0.0


def _build_task_b_risk_results(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tier_thresholds: dict[str, float],
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32).reshape(-1)
    sepsis_tiers = assign_risk_tiers(y_prob_arr, tier_thresholds)

    topk_metrics: dict[str, float] = {}
    for k in [50, 100, 200]:
        topk_metrics[f"recall_at_{k}"] = recall_at_k(y_true_arr, y_prob_arr, k)
        topk_metrics[f"ppv_at_{k}"] = ppv_at_k(y_true_arr, y_prob_arr, k)

    tier_stats: dict[str, dict[str, float | int]] = {}
    for tier in ["LOW", "MODERATE", "HIGH", "CRITICAL"]:
        mask = sepsis_tiers == tier
        count = int(mask.sum())
        pos_in_tier = int(y_true_arr[mask].sum()) if count > 0 else 0
        tier_stats[tier] = {
            "count": count,
            "positive_count": pos_in_tier,
            "positive_rate": round(float(pos_in_tier / count), 4) if count > 0 else 0.0,
            "pct_of_total": round(float(count / len(mask)), 4),
        }

    return {
        "model_type": "sepsis_risk_scorer",
        "auc_roc": _safe_auc(y_true_arr, y_prob_arr),
        "prob_distribution": {
            "min": float(y_prob_arr.min()),
            "max": float(y_prob_arr.max()),
            "mean": float(y_prob_arr.mean()),
            "median": float(np.median(y_prob_arr)),
            "pct_above_0_35": float((y_prob_arr >= 0.35).mean()),
            "pct_above_0_25": float((y_prob_arr >= 0.25).mean()),
            "pct_above_0_15": float((y_prob_arr >= 0.15).mean()),
        },
        "risk_tiers": tier_stats,
        "topk_metrics": topk_metrics,
        "note": SEPSIS_RISK_NOTE,
    }


@lru_cache(maxsize=1)
def _load_saved_sepsis_model(model_path_str: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path_str, compile=False)


def _resolve_dataset_dirs(
    set_a_dir: str | Path | None,
    set_b_dir: str | Path | None,
) -> tuple[Path, Path]:
    if set_a_dir is None or set_b_dir is None:
        default_a, default_b = default_lstm_dataset_dirs()
        resolved_a = Path(set_a_dir) if set_a_dir is not None else default_a
        resolved_b = Path(set_b_dir) if set_b_dir is not None else default_b
        return resolved_a, resolved_b

    return Path(set_a_dir), Path(set_b_dir)


def evaluate_lstm_task_b_risk_only(
    set_a_dir: str | Path | None = None,
    set_b_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    max_patients: int | None = MAX_PATIENTS_DEFAULT,
) -> dict[str, Any]:
    set_a, set_b = _resolve_dataset_dirs(set_a_dir, set_b_dir)
    models_path = Path(models_dir) if models_dir is not None else _default_models_dir()

    sepsis_model_path = models_path / "lstm_sepsis.h5"
    scaler_path = models_path / "lstm_minmax_scaler.pkl"
    if not sepsis_model_path.exists():
        raise FileNotFoundError(f"Sepsis model not found: {sepsis_model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"LSTM scaler not found: {scaler_path}")

    all_files, total_patients_available, patients_used = _list_patient_files(
        set_a,
        set_b,
        max_patients=max_patients,
    )
    train_files, temp_files = train_test_split(all_files, test_size=0.3, random_state=42)
    _val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    train_features_raw, _, _, _ = _load_patient_group(train_files)
    test_features_raw, test_labels, test_lengths, _ = _load_patient_group(test_files)

    train_ffill = _forward_fill_per_patient(train_features_raw)
    test_ffill = _forward_fill_per_patient(test_features_raw)

    train_medians = _compute_train_medians(train_ffill)
    test_filled = _fill_missing_with_medians(test_ffill, train_medians)

    scaler: MinMaxScaler = joblib.load(scaler_path)
    test_flat = test_filled.reshape(-1, test_filled.shape[-1])
    test_scaled = scaler.transform(test_flat).reshape(test_filled.shape).astype(np.float32)

    X_test_sepsis, y_test_sepsis = _build_sepsis_sequences(
        test_scaled,
        test_labels,
        test_lengths,
        WINDOW_SIZE,
        SEPSIS_HORIZON,
        STRIDE,
    )

    sepsis_model = _load_saved_sepsis_model(str(sepsis_model_path))
    sepsis_prob = sepsis_model.predict(X_test_sepsis, verbose=0).ravel()

    tier_thresholds = load_lstm_sepsis_tier_thresholds(models_path)
    sepsis_tiers_path = save_lstm_sepsis_tier_thresholds(models_path, tier_thresholds)
    task_b_results = _build_task_b_risk_results(y_test_sepsis, sepsis_prob, tier_thresholds)
    task_b_results.update(
        {
            "test_sequences": int(X_test_sepsis.shape[0]),
            "test_patients": int(len(test_files)),
            "patients_used": int(patients_used),
            "total_patients_available": int(total_patients_available),
            "tier_thresholds": tier_thresholds,
            "artifacts": {
                "sepsis_model_path": str(sepsis_model_path),
                "scaler_path": str(scaler_path),
                "sepsis_tiers_path": str(sepsis_tiers_path),
            },
        }
    )

    print("Task B inference-only risk results JSON:")
    print(json.dumps(task_b_results, indent=2))
    return task_b_results


def predict_lstm_sepsis_risk(
    patient_psv_bytes: bytes,
    models_dir: str | Path | None = None,
) -> dict[str, Any]:
    models_path = Path(models_dir) if models_dir is not None else _default_models_dir()
    sepsis_model_path = models_path / "lstm_sepsis.h5"
    scaler_path = models_path / "lstm_minmax_scaler.pkl"
    if not sepsis_model_path.exists():
        raise FileNotFoundError(f"Sepsis model not found: {sepsis_model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"LSTM scaler not found: {scaler_path}")

    try:
        df = pd.read_csv(BytesIO(patient_psv_bytes), sep="|")
    except Exception as exc:
        raise ValueError("Invalid PSV file. Could not parse patient time-series data.") from exc

    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in patient file: {missing_cols}")

    patient_df = df[FEATURE_COLUMNS].iloc[:MAX_HOURS].copy()
    if patient_df.empty:
        raise ValueError("Patient file has no rows.")

    patient_df = patient_df.replace(-1, np.nan).ffill().bfill().fillna(0.0)
    patient_features = patient_df.to_numpy(dtype=np.float32)

    scaler: MinMaxScaler = joblib.load(scaler_path)
    scaled_features = scaler.transform(patient_features).astype(np.float32)

    if scaled_features.shape[0] < WINDOW_SIZE:
        pad_rows = WINDOW_SIZE - scaled_features.shape[0]
        pad = np.repeat(scaled_features[[0]], pad_rows, axis=0)
        scaled_features = np.vstack([pad, scaled_features]).astype(np.float32)

    windows: list[np.ndarray] = []
    max_start = scaled_features.shape[0] - WINDOW_SIZE
    for start in range(0, max_start + 1, STRIDE):
        end = start + WINDOW_SIZE
        windows.append(scaled_features[start:end, :])

    X_patient = np.asarray(windows, dtype=np.float32)
    sepsis_model = _load_saved_sepsis_model(str(sepsis_model_path))
    window_scores = sepsis_model.predict(X_patient, verbose=0).ravel()

    peak_idx = int(np.argmax(window_scores)) if window_scores.size > 0 else 0
    risk_score = float(window_scores[peak_idx]) if window_scores.size > 0 else 0.0

    tier_thresholds = load_lstm_sepsis_tier_thresholds(models_path)
    risk_tier = str(assign_risk_tiers(np.asarray([risk_score], dtype=np.float32), tier_thresholds)[0])

    return {
        "sepsis_risk_score": risk_score,
        "sepsis_risk_tier": risk_tier,
        "sepsis_risk_note": SEPSIS_RISK_NOTE,
        "window_count": int(X_patient.shape[0]),
        "peak_window_index": peak_idx,
        "tier_thresholds": tier_thresholds,
    }


def _list_patient_files(
    set_a_dir: Path,
    set_b_dir: Path,
    max_patients: int | None,
) -> tuple[list[Path], int, int]:
    set_a_files = sorted(set_a_dir.glob("*.psv"))
    set_b_files = sorted(set_b_dir.glob("*.psv"))

    total_count = len(set_a_files) + len(set_b_files)
    if total_count == 0:
        raise FileNotFoundError(f"No .psv files found in {set_a_dir} and {set_b_dir}")

    if max_patients is None or max_patients >= total_count:
        return set_a_files + set_b_files, total_count, total_count

    rng = np.random.default_rng(42)
    set_a_sample = set_a_files.copy()
    set_b_sample = set_b_files.copy()
    rng.shuffle(set_a_sample)
    rng.shuffle(set_b_sample)

    target_a = min(len(set_a_sample), max_patients // 2)
    target_b = min(len(set_b_sample), max_patients - target_a)

    selected = set_a_sample[:target_a] + set_b_sample[:target_b]
    if len(selected) < max_patients:
        remaining = set_a_sample[target_a:] + set_b_sample[target_b:]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_patients - len(selected)])

    rng.shuffle(selected)
    return selected, total_count, len(selected)


def _load_single_patient(path: Path) -> tuple[np.ndarray, np.ndarray, int, str]:
    df = pd.read_csv(path, sep="|")

    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"File {path} is missing required columns: {missing_cols}")

    df = df[required_cols].iloc[:MAX_HOURS].copy()
    original_length = int(df.shape[0])

    features_df = df[FEATURE_COLUMNS].replace(-1, np.nan)
    features = features_df.to_numpy(dtype=np.float32)

    labels = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=np.float32)
    labels = np.where(np.isnan(labels), 0.0, labels)

    if original_length < MAX_HOURS:
        pad_rows = MAX_HOURS - original_length
        features_pad = np.full((pad_rows, len(FEATURE_COLUMNS)), -1.0, dtype=np.float32)
        labels_pad = np.full((pad_rows,), -1.0, dtype=np.float32)
        features = np.vstack([features, features_pad]).astype(np.float32)
        labels = np.concatenate([labels, labels_pad]).astype(np.float32)

    return features, labels, original_length, path.stem


def _load_patient_group(
    files: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    features_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    lengths_list: list[int] = []
    patient_ids: list[str] = []

    for path in files:
        features, labels, length, patient_id = _load_single_patient(path)
        features_list.append(features)
        labels_list.append(labels)
        lengths_list.append(length)
        patient_ids.append(patient_id)

    return (
        np.stack(features_list).astype(np.float32),
        np.stack(labels_list).astype(np.float32),
        np.asarray(lengths_list, dtype=np.int32),
        patient_ids,
    )


def _forward_fill_per_patient(features: np.ndarray) -> np.ndarray:
    out = np.empty_like(features, dtype=np.float32)
    for idx in range(features.shape[0]):
        frame = pd.DataFrame(features[idx], columns=FEATURE_COLUMNS)
        frame = frame.replace(-1.0, np.nan)
        frame = frame.ffill()
        out[idx] = frame.to_numpy(dtype=np.float32)
    return out


def _compute_train_medians(train_features: np.ndarray) -> np.ndarray:
    flattened = train_features.reshape(-1, train_features.shape[-1])
    medians = np.nanmedian(flattened, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    return medians.astype(np.float32)


def _fill_missing_with_medians(features: np.ndarray, medians: np.ndarray) -> np.ndarray:
    filled = features.copy()
    for col_idx in range(filled.shape[2]):
        col = filled[:, :, col_idx]
        nan_mask = np.isnan(col)
        if np.any(nan_mask):
            col[nan_mask] = medians[col_idx]
        filled[:, :, col_idx] = col
    return filled.astype(np.float32)


def _scale_features(
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    train_flat = train_features.reshape(-1, train_features.shape[-1])
    val_flat = val_features.reshape(-1, val_features.shape[-1])
    test_flat = test_features.reshape(-1, test_features.shape[-1])

    scaler.fit(train_flat)

    train_scaled = scaler.transform(train_flat).reshape(train_features.shape).astype(np.float32)
    val_scaled = scaler.transform(val_flat).reshape(val_features.shape).astype(np.float32)
    test_scaled = scaler.transform(test_flat).reshape(test_features.shape).astype(np.float32)

    return train_scaled, val_scaled, test_scaled, scaler


def _build_vitals_sequences(
    features: np.ndarray,
    lengths: np.ndarray,
    window: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    hr_idx = FEATURE_COLUMNS.index("HR")
    o2_idx = FEATURE_COLUMNS.index("O2Sat")

    X: list[np.ndarray] = []
    y: list[np.ndarray] = []

    for patient_idx, length in enumerate(lengths.tolist()):
        max_start = length - window - 1
        if max_start < 0:
            continue

        for start in range(0, max_start + 1, stride):
            end = start + window
            target_idx = end

            X.append(features[patient_idx, start:end, :])
            y.append(features[patient_idx, target_idx, [hr_idx, o2_idx]])

    if not X:
        raise ValueError("No Task A sequences created. Check window length and patient durations.")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def _build_sepsis_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    window: int,
    horizon: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    y: list[int] = []

    for patient_idx, length in enumerate(lengths.tolist()):
        max_start = length - window - horizon
        if max_start < 0:
            continue

        for start in range(0, max_start + 1, stride):
            end = start + window

            future_window = labels[patient_idx, end : end + horizon]
            valid_future = future_window[future_window >= 0]
            if len(valid_future) == 0:
                continue

            target = int(np.any(valid_future > 0.5))

            if np.any(features[patient_idx, start:end] < 0):
                continue

            X.append(features[patient_idx, start:end, :])
            y.append(target)

    if not X:
        raise ValueError("No Task B sequences created.")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)


def build_vitals_lstm(input_shape: tuple[int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(2),
        ],
        name="lstm_vitals_prediction",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def focal_loss_sepsis(gamma: float = 2.0, alpha: float = 0.25):
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_cast = tf.cast(y_true, tf.float32)
        y_true_cast = tf.reshape(y_true_cast, tf.shape(y_pred))
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        bce = (
            -y_true_cast * tf.math.log(y_pred_clipped)
            - (1 - y_true_cast) * tf.math.log(1 - y_pred_clipped)
        )
        p_t = y_true_cast * y_pred_clipped + (1 - y_true_cast) * (1 - y_pred_clipped)
        return tf.reduce_mean(alpha * tf.pow(1 - p_t, gamma) * bce)

    return loss


def build_sepsis_lstm(input_shape: tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_sepsis_prediction")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss_sepsis(gamma=2.0, alpha=0.25),
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def _loss_curve_plot(history: tf.keras.callbacks.History, title: str) -> str:
    history_data = history.history
    epochs = np.arange(1, len(history_data.get("loss", [])) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history_data.get("loss", []), label="Train Loss", linewidth=2)
    plt.plot(epochs, history_data.get("val_loss", []), label="Val Loss", linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    return _to_base64_png()


def _inverse_scale_feature(values: np.ndarray, scaler: MinMaxScaler, feature_index: int) -> np.ndarray:
    minimum = float(scaler.data_min_[feature_index])
    scale_range = float(scaler.data_range_[feature_index])
    if abs(scale_range) < 1e-12:
        return np.full_like(values, fill_value=minimum, dtype=np.float32)
    return (values * scale_range + minimum).astype(np.float32)


def _hr_actual_vs_predicted_plot(
    model: tf.keras.Model,
    test_features: np.ndarray,
    test_lengths: np.ndarray,
    test_patient_ids: list[str],
    scaler: MinMaxScaler,
) -> str:
    hr_idx = FEATURE_COLUMNS.index("HR")

    selected_indices = [
        idx for idx, length in enumerate(test_lengths.tolist()) if length >= WINDOW_SIZE + 1
    ][:3]
    if not selected_indices:
        raise ValueError("No test patients with enough timesteps for Task A plotting.")

    n_rows = len(selected_indices)
    plt.figure(figsize=(11, 3.6 * n_rows))

    for plot_idx, patient_idx in enumerate(selected_indices, start=1):
        length = int(test_lengths[patient_idx])
        max_start = length - WINDOW_SIZE - 1

        patient_windows: list[np.ndarray] = []
        target_hours: list[int] = []
        actual_hr_norm: list[float] = []

        for start in range(0, max_start + 1, STRIDE):
            end = start + WINDOW_SIZE
            patient_windows.append(test_features[patient_idx, start:end, :])
            target_hours.append(end + 1)
            actual_hr_norm.append(float(test_features[patient_idx, end, hr_idx]))

        X_patient = np.asarray(patient_windows, dtype=np.float32)
        pred_norm = model.predict(X_patient, verbose=0)[:, 0]

        actual_hr = _inverse_scale_feature(np.asarray(actual_hr_norm, dtype=np.float32), scaler, hr_idx)
        pred_hr = _inverse_scale_feature(pred_norm.astype(np.float32), scaler, hr_idx)

        plt.subplot(n_rows, 1, plot_idx)
        plt.plot(target_hours, actual_hr, label="Actual HR", linewidth=2)
        plt.plot(target_hours, pred_hr, label="Predicted HR", linewidth=1.8)
        plt.title(f"Patient {test_patient_ids[patient_idx]}: Actual vs Predicted HR")
        plt.xlabel("Hour")
        plt.ylabel("HR")
        plt.legend(loc="best")

    return _to_base64_png()


def _sepsis_roc_curve_plot(y_true: np.ndarray, y_prob: np.ndarray, auc_value: float | None) -> str:
    plt.figure(figsize=(8, 6))
    if auc_value is None:
        plt.text(0.5, 0.5, "ROC unavailable (single-class ground truth)", ha="center", va="center")
        plt.axis("off")
        return _to_base64_png()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={auc_value:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.2)
    plt.title("Sepsis Prediction ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    return _to_base64_png()


def train_and_evaluate_lstm(
    set_a_dir: str | Path | None = None,
    set_b_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    max_patients: int | None = MAX_PATIENTS_DEFAULT,
    epochs: int = EPOCHS,
    sepsis_epochs: int = SEPSIS_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    if set_a_dir is None or set_b_dir is None:
        default_a, default_b = default_lstm_dataset_dirs()
        set_a = Path(set_a_dir) if set_a_dir is not None else default_a
        set_b = Path(set_b_dir) if set_b_dir is not None else default_b
    else:
        set_a = Path(set_a_dir)
        set_b = Path(set_b_dir)

    models_path = Path(models_dir) if models_dir is not None else _default_models_dir()
    models_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    all_files, total_patients_available, patients_used = _list_patient_files(
        set_a,
        set_b,
        max_patients=max_patients,
    )

    train_files, temp_files = train_test_split(all_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    train_features_raw, train_labels, train_lengths, train_patient_ids = _load_patient_group(train_files)
    val_features_raw, val_labels, val_lengths, val_patient_ids = _load_patient_group(val_files)
    test_features_raw, test_labels, test_lengths, test_patient_ids = _load_patient_group(test_files)

    train_ffill = _forward_fill_per_patient(train_features_raw)
    val_ffill = _forward_fill_per_patient(val_features_raw)
    test_ffill = _forward_fill_per_patient(test_features_raw)

    train_medians = _compute_train_medians(train_ffill)
    train_filled = _fill_missing_with_medians(train_ffill, train_medians)
    val_filled = _fill_missing_with_medians(val_ffill, train_medians)
    test_filled = _fill_missing_with_medians(test_ffill, train_medians)

    train_scaled, val_scaled, test_scaled, scaler = _scale_features(train_filled, val_filled, test_filled)

    scaler_path = models_path / "lstm_minmax_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    X_train_vitals, y_train_vitals = _build_vitals_sequences(train_scaled, train_lengths, WINDOW_SIZE, STRIDE)
    X_val_vitals, y_val_vitals = _build_vitals_sequences(val_scaled, val_lengths, WINDOW_SIZE, STRIDE)
    X_test_vitals, y_test_vitals = _build_vitals_sequences(test_scaled, test_lengths, WINDOW_SIZE, STRIDE)

    X_train_sepsis, y_train_sepsis = _build_sepsis_sequences(
        train_scaled,
        train_labels,
        train_lengths,
        WINDOW_SIZE,
        SEPSIS_HORIZON,
        STRIDE,
    )
    X_val_sepsis, y_val_sepsis = _build_sepsis_sequences(
        val_scaled,
        val_labels,
        val_lengths,
        WINDOW_SIZE,
        SEPSIS_HORIZON,
        STRIDE,
    )
    X_test_sepsis, y_test_sepsis = _build_sepsis_sequences(
        test_scaled,
        test_labels,
        test_lengths,
        WINDOW_SIZE,
        SEPSIS_HORIZON,
        STRIDE,
    )

    vitals_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            mode="min",
        )
    ]

    vitals_model = build_vitals_lstm((WINDOW_SIZE, len(FEATURE_COLUMNS)))
    vitals_history = vitals_model.fit(
        X_train_vitals,
        y_train_vitals,
        validation_data=(X_val_vitals, y_val_vitals),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=vitals_callbacks,
        verbose=2,
    )

    vitals_model_path = models_path / "lstm_vitals.h5"
    vitals_model.save(vitals_model_path)

    vitals_pred_test = vitals_model.predict(X_test_vitals, verbose=0)
    vitals_test_mse = float(mean_squared_error(y_test_vitals, vitals_pred_test))
    vitals_test_mae = float(mean_absolute_error(y_test_vitals, vitals_pred_test))

    pos = int(y_train_sepsis.sum())
    neg = int(len(y_train_sepsis) - pos)
    if pos > 0:
        class_weight = {0: 1.0, 1: float(neg / pos)}
    else:
        class_weight = {0: 1.0, 1: 1.0}

    sepsis_model_path = models_path / "lstm_sepsis.h5"

    sepsis_model = build_sepsis_lstm((WINDOW_SIZE, len(FEATURE_COLUMNS)))
    sepsis_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=15,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.7,
            patience=8,
            min_lr=1e-6,
            mode="max",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(sepsis_model_path),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
        ),
    ]

    sepsis_history = sepsis_model.fit(
        X_train_sepsis,
        y_train_sepsis,
        validation_data=(X_val_sepsis, y_val_sepsis),
        epochs=sepsis_epochs,
        batch_size=batch_size,
        callbacks=sepsis_callbacks,
        class_weight=class_weight,
        verbose=2,
    )

    # sepsis_model.save(sepsis_model_path)  # removed - ModelCheckpoint handles this

    val_auc_values = sepsis_history.history.get("val_auc", [])
    print("Task B val_auc by epoch:")
    for idx, val_auc_value in enumerate(val_auc_values, start=1):
        print(f"  epoch={idx:02d}, val_auc={float(val_auc_value):.6f}")

    best_epoch = int(np.argmax(val_auc_values) + 1) if val_auc_values else 0
    best_val_auc = float(np.max(val_auc_values)) if val_auc_values else float("nan")
    print(f"ModelCheckpoint best epoch: {best_epoch} (val_auc={best_val_auc:.6f})")

    sepsis_prob = sepsis_model.predict(X_test_sepsis, verbose=0).ravel()

    tier_thresholds = load_lstm_sepsis_tier_thresholds(models_path)
    sepsis_tiers_path = save_lstm_sepsis_tier_thresholds(models_path, tier_thresholds)
    task_b_results = _build_task_b_risk_results(y_test_sepsis, sepsis_prob, tier_thresholds)
    print("Task B risk results JSON:")
    print(json.dumps(task_b_results, indent=2))

    hr_plot_b64 = _hr_actual_vs_predicted_plot(
        vitals_model,
        test_scaled,
        test_lengths,
        test_patient_ids,
        scaler,
    )
    vitals_loss_b64 = _loss_curve_plot(vitals_history, "Task A LSTM Loss Curve")
    sepsis_auc = task_b_results.get("auc_roc")
    sepsis_roc_b64 = _sepsis_roc_curve_plot(y_test_sepsis, sepsis_prob, sepsis_auc)

    results: dict[str, Any] = {
        "task": "physionet_lstm_multitask",
        "data": {
            "set_a_dir": str(set_a),
            "set_b_dir": str(set_b),
            "total_patients_available": int(total_patients_available),
            "patients_used": int(patients_used),
            "train_patients": int(len(train_files)),
            "val_patients": int(len(val_files)),
            "test_patients": int(len(test_files)),
            "window": WINDOW_SIZE,
            "stride": STRIDE,
            "sepsis_horizon": SEPSIS_HORIZON,
        },
        "task_a_vitals": {
            "target_features": ["HR", "O2Sat"],
            "train_sequences": int(X_train_vitals.shape[0]),
            "val_sequences": int(X_val_vitals.shape[0]),
            "test_sequences": int(X_test_vitals.shape[0]),
            "metrics": {
                "test_mse": vitals_test_mse,
                "test_mae": vitals_test_mae,
            },
            "actual_vs_predicted_hr_plot": hr_plot_b64,
            "loss_curve_plot": vitals_loss_b64,
        },
        "task_b_sepsis": {
            **task_b_results,
            "train_sequences": int(X_train_sepsis.shape[0]),
            "val_sequences": int(X_val_sepsis.shape[0]),
            "test_sequences": int(X_test_sepsis.shape[0]),
            "class_weight": class_weight,
            "training_curves": {
                "loss": [float(x) for x in sepsis_history.history.get("loss", [])],
                "val_loss": [float(x) for x in sepsis_history.history.get("val_loss", [])],
                "auc": [float(x) for x in sepsis_history.history.get("auc", [])],
                "val_auc": [float(x) for x in sepsis_history.history.get("val_auc", [])],
            },
            "roc_curve_plot": sepsis_roc_b64,
        },
        "artifacts": {
            "vitals_model_path": str(vitals_model_path),
            "sepsis_model_path": str(sepsis_model_path),
            "scaler_path": str(scaler_path),
            "sepsis_tiers_path": str(sepsis_tiers_path),
        },
    }

    print("Task B results JSON:")
    print(json.dumps(results["task_b_sepsis"], indent=2))

    results_path = models_path / "lstm_results.json"
    with results_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp)

    return results


if __name__ == "__main__":
    output = train_and_evaluate_lstm()
    summary = {
        "task_a_metrics": output.get("task_a_vitals", {}).get("metrics", {}),
        "task_b_auc_roc": output.get("task_b_sepsis", {}).get("auc_roc"),
        "task_b_topk_metrics": output.get("task_b_sepsis", {}).get("topk_metrics", {}),
    }
    print(json.dumps(summary, indent=2))
