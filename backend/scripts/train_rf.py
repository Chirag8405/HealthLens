from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.data_utils import default_csv_path
from ml.data_utils import prepare_modeling_dataframe
from path_utils import project_root_from


def _load_training_matrix(models_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    csv_path = default_csv_path()
    df = prepare_modeling_dataframe(csv_path)

    if "readmitted_30" not in df.columns:
        raise KeyError("Target column 'readmitted_30' not found after preprocessing.")

    drop_cols = [
        col
        for col in ("readmitted_30", "readmitted", "encounter_id", "patient_nbr")
        if col in df.columns
    ]
    X = df.drop(columns=drop_cols)
    y = df["readmitted_30"].astype(int)

    classification_dir = models_dir / "classification"
    feature_names_path = classification_dir / "feature_names.json"
    if feature_names_path.exists():
        with feature_names_path.open("r", encoding="utf-8") as fp:
            expected_feature_names = list(json.load(fp))
        X = X.reindex(columns=expected_feature_names, fill_value=0)

    X_train_df, X_test_df, y_train_series, y_test_series = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train_raw = X_train_df.to_numpy(dtype=np.float64)
    X_test_raw = X_test_df.to_numpy(dtype=np.float64)
    y_train_np = y_train_series.to_numpy(dtype=np.int64)
    y_test_np = y_test_series.to_numpy(dtype=np.int64)

    selector_path = classification_dir / "variance_selector.pkl"
    scaler_path = classification_dir / "scaler.pkl"
    if not selector_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Missing classification artifacts required for 109-feature RF training: "
            f"{selector_path} and/or {scaler_path}"
        )

    selector = joblib.load(selector_path)
    scaler = joblib.load(scaler_path)

    X_train_reduced = selector.transform(X_train_raw)
    X_test_reduced = selector.transform(X_test_raw)
    X_train_scaled = scaler.transform(X_train_reduced)
    X_test_scaled = scaler.transform(X_test_reduced)

    return X_train_scaled, X_test_scaled, y_train_np, y_test_np


def main() -> None:
    project_root = project_root_from(__file__)
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training matrix via classification feature pipeline...")
    X_train, X_test, y_train, y_test = _load_training_matrix(models_dir)

    print(f"X_train shape: {X_train.shape}")
    print(f"Class distribution: {np.bincount(y_train.astype(int))}")

    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        max_features="sqrt",
        min_samples_leaf=10,
        min_samples_split=20,
        max_leaf_nodes=500,
        n_jobs=1,
        class_weight="balanced",
        random_state=42,
    )

    idx = np.random.RandomState(42).choice(
        X_train.shape[0],
        min(50000, X_train.shape[0]),
        replace=False,
    )
    print(f"Training on {len(idx)} samples...")
    rf.fit(X_train[idx], y_train[idx])

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1:       {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"AUC-ROC:  {roc_auc_score(y_test, y_proba):.4f}")

    out_path = models_dir / "rf_model.pkl"
    joblib.dump(rf, out_path, compress=3)
    size_mb = out_path.stat().st_size / 1024**2
    print(f"Saved rf_model.pkl - {size_mb:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
