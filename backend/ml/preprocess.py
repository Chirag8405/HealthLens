from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

HIGH_MISSING_COLUMNS = ("weight", "payer_code", "medical_specialty")
DIAGNOSIS_COLUMNS = ("diag_1", "diag_2", "diag_3")
READMITTED_MAP = {"NO": 0, ">30": 1, "<30": 2}
ONE_HOT_COLUMN_MAP = {
    "admission_type": "admission_type_id",
    "discharge_disposition": "discharge_disposition_id",
    "admission_source": "admission_source_id",
}


class PreprocessingPipeline:
    def __init__(self, processed_dir: str | Path | None = None) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.processed_dir = (
            Path(processed_dir)
            if processed_dir is not None
            else self.project_root / "data" / "processed"
        )

        self.scaler_path = self.processed_dir / "scaler.pkl"
        self.feature_names_path = self.processed_dir / "feature_names.json"
        self.feature_contract_path = self.processed_dir / "feature_contract.json"
        self.label_encoders_path = self.processed_dir / "label_encoders.pkl"
        self.artifact_paths = {
            "X_train": self.processed_dir / "X_train.npy",
            "X_test": self.processed_dir / "X_test.npy",
            "y_train": self.processed_dir / "y_train.npy",
            "y_test": self.processed_dir / "y_test.npy",
        }

        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_names_: list[str] = []

    def _load_csv(self, csv_path: str | Path) -> pd.DataFrame:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        return pd.read_csv(csv_path).replace("?", np.nan)

    def _drop_high_missing_columns(self, df: pd.DataFrame, threshold: float = 0.40) -> pd.DataFrame:
        missing_ratio = df.isna().mean()
        dynamic_drop = [col for col, ratio in missing_ratio.items() if ratio > threshold]
        required_drop = [col for col in HIGH_MISSING_COLUMNS if col in df.columns]
        drop_cols = sorted(set(dynamic_drop + required_drop))
        return df.drop(columns=drop_cols)

    def _coerce_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        categorical_like_cols = (
            set(DIAGNOSIS_COLUMNS)
            | set(ONE_HOT_COLUMN_MAP.keys())
            | set(ONE_HOT_COLUMN_MAP.values())
            | {"readmitted", "race", "gender", "age"}
        )

        for col in df.columns:
            if col in categorical_like_cols or not pd.api.types.is_object_dtype(df[col]):
                continue

            non_null_values = df[col].dropna()
            if non_null_values.empty:
                continue

            convertible_ratio = pd.to_numeric(non_null_values, errors="coerce").notna().mean()
            if convertible_ratio >= 0.95:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    @staticmethod
    def _map_icd_to_bucket(code: object) -> str:
        if pd.isna(code):
            return "Z"

        value = str(code).strip().upper()
        if not value or value == "NAN":
            return "Z"

        if value[0].isalpha():
            return value[0] if "A" <= value[0] <= "Z" else "Z"

        numeric_part: list[str] = []
        for ch in value:
            if ch.isdigit() or ch == ".":
                numeric_part.append(ch)
            else:
                break

        if not numeric_part:
            return "Z"

        try:
            icd_num = float("".join(numeric_part))
        except ValueError:
            return "Z"

        if 1 <= icd_num <= 139:
            return "A"
        if 140 <= icd_num <= 239:
            return "B"
        if 240 <= icd_num <= 279:
            return "C"
        if 280 <= icd_num <= 289:
            return "D"
        if 290 <= icd_num <= 319:
            return "E"
        if 320 <= icd_num <= 389:
            return "F"
        if 390 <= icd_num <= 459:
            return "G"
        if 460 <= icd_num <= 519:
            return "H"
        if 520 <= icd_num <= 579:
            return "I"
        if 580 <= icd_num <= 629:
            return "J"
        if 630 <= icd_num <= 679:
            return "K"
        if 680 <= icd_num <= 709:
            return "L"
        if 710 <= icd_num <= 739:
            return "M"
        if 740 <= icd_num <= 759:
            return "N"
        if 760 <= icd_num <= 779:
            return "O"
        if 780 <= icd_num <= 799:
            return "P"
        if 800 <= icd_num <= 999:
            return "Q"
        return "Z"

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        target_like_cols = (
            set(ONE_HOT_COLUMN_MAP.keys())
            | set(ONE_HOT_COLUMN_MAP.values())
            | set(DIAGNOSIS_COLUMNS)
            | {"readmitted", "race", "gender", "age"}
        )

        numeric_cols = [
            col
            for col in df.select_dtypes(include=[np.number]).columns.tolist()
            if col not in target_like_cols
        ]
        categorical_cols = [col for col in df.columns if col not in numeric_cols]

        for col in numeric_cols:
            median_value = df[col].median()
            if pd.isna(median_value):
                median_value = 0.0
            df[col] = df[col].fillna(median_value)

        for col in categorical_cols:
            mode_series = df[col].mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)

        return df

    def _engineer_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        if "readmitted" not in df.columns:
            raise KeyError("Expected 'readmitted' column in dataset.")

        readmitted_clean = df["readmitted"].astype(str).str.strip().str.upper()
        unknown_values = sorted(set(readmitted_clean.unique()) - set(READMITTED_MAP.keys()))
        if unknown_values:
            raise ValueError(f"Unexpected values in readmitted: {unknown_values}")

        df["readmitted_30"] = (readmitted_clean == "<30").astype(int)
        df["readmitted"] = readmitted_clean.map(READMITTED_MAP).astype(int)
        return df

    def _engineer_diagnosis_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in DIAGNOSIS_COLUMNS:
            grouped_col = f"{col}_icd_group"
            if col in df.columns:
                df[grouped_col] = df[col].apply(self._map_icd_to_bucket)
            else:
                df[grouped_col] = "Z"

        df["diag_icd_group"] = (
            df["diag_1_icd_group"].astype(str)
            + "_"
            + df["diag_2_icd_group"].astype(str)
            + "_"
            + df["diag_3_icd_group"].astype(str)
        )

        drop_diag_cols = [col for col in DIAGNOSIS_COLUMNS if col in df.columns]
        if drop_diag_cols:
            df = df.drop(columns=drop_diag_cols)

        return df

    def _label_encode_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if "age" in df.columns:
            df["age_bracket"] = df["age"].astype(str)
            df = df.drop(columns=["age"])

        for col in ("gender", "race", "age_bracket"):
            if col not in df.columns:
                continue

            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            self.label_encoders[col] = encoder

        return df

    def _one_hot_encode_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        one_hot_columns: list[str] = []

        for normalized_col, source_col in ONE_HOT_COLUMN_MAP.items():
            if source_col in df.columns and normalized_col not in df.columns:
                df = df.rename(columns={source_col: normalized_col})

            if normalized_col in df.columns:
                one_hot_columns.append(normalized_col)

        if one_hot_columns:
            df[one_hot_columns] = df[one_hot_columns].astype(str)
            df = pd.get_dummies(
                df,
                columns=one_hot_columns,
                prefix=one_hot_columns,
                drop_first=True,
                dtype=int,
            )

        return df

    def _scale_numeric_columns(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        scaler = StandardScaler()

        if numeric_cols:
            X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, self.scaler_path)
        return X_train, X_test, numeric_cols

    def _print_report(
        self,
        raw_shape: tuple[int, int],
        processed_shape: tuple[int, int],
        X_train_shape: tuple[int, int],
        X_test_shape: tuple[int, int],
        y_binary: pd.Series,
        missing_before: int,
        missing_after: int,
    ) -> None:
        print("\nPreprocessing Report")
        print("-" * 60)
        print(f"Raw shape: {raw_shape}")
        print(f"Processed shape: {processed_shape}")
        print(f"Train shape: {X_train_shape}")
        print(f"Test shape: {X_test_shape}")
        print(f"Missing values before: {missing_before}")
        print(f"Missing values after: {missing_after}")

        distribution = y_binary.value_counts(dropna=False).sort_index()
        distribution_pct = y_binary.value_counts(normalize=True, dropna=False).sort_index() * 100

        print("readmitted_30 distribution:")
        for cls_value in distribution.index:
            count = int(distribution[cls_value])
            pct = float(distribution_pct[cls_value])
            print(f"  class {cls_value}: {count} ({pct:.2f}%)")
        print("-" * 60)

    def run(self, csv_path: str | Path) -> dict[str, Any]:
        df = self._load_csv(csv_path)
        raw_shape = (int(df.shape[0]), int(df.shape[1]))
        missing_before = int(df.isna().sum().sum())

        df = self._drop_high_missing_columns(df)
        df = self._coerce_numeric_columns(df)
        df = self._impute_missing_values(df)
        df = self._engineer_targets(df)
        df = self._engineer_diagnosis_groups(df)
        df = self._label_encode_columns(df)
        df = self._one_hot_encode_columns(df)

        missing_after = int(df.isna().sum().sum())

        y = df["readmitted_30"].astype(int)
        X = df.drop(columns=["readmitted", "readmitted_30"])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train, X_test, scaled_numeric_cols = self._scale_numeric_columns(X_train, X_test)

        self.feature_names_ = X_train.columns.tolist()
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        categorical_cols = [col for col in self.feature_names_ if col not in scaled_numeric_cols]
        feature_contract = {
            "feature_names": self.feature_names_,
            "n_features": len(self.feature_names_),
            "numerical_cols": scaled_numeric_cols,
            "categorical_cols": categorical_cols,
            "scaler_feature_names": scaled_numeric_cols,
        }

        with self.feature_contract_path.open("w", encoding="utf-8") as fp:
            json.dump(feature_contract, fp, indent=2)
        joblib.dump(self.label_encoders, self.label_encoders_path)

        print(f"Feature contract saved: {len(self.feature_names_)} features")
        print(f"Columns: {self.feature_names_[:10]} ...")

        with self.feature_names_path.open("w", encoding="utf-8") as fp:
            json.dump(self.feature_names_, fp, indent=2)

        np.save(self.artifact_paths["X_train"], X_train.to_numpy(), allow_pickle=True)
        np.save(self.artifact_paths["X_test"], X_test.to_numpy(), allow_pickle=True)
        np.save(self.artifact_paths["y_train"], y_train.to_numpy(), allow_pickle=True)
        np.save(self.artifact_paths["y_test"], y_test.to_numpy(), allow_pickle=True)

        self._print_report(
            raw_shape=raw_shape,
            processed_shape=(int(df.shape[0]), int(df.shape[1])),
            X_train_shape=(int(X_train.shape[0]), int(X_train.shape[1])),
            X_test_shape=(int(X_test.shape[0]), int(X_test.shape[1])),
            y_binary=y,
            missing_before=missing_before,
            missing_after=missing_after,
        )

        return {
            "raw_shape": raw_shape,
            "processed_shape": (int(df.shape[0]), int(df.shape[1])),
            "X_train_shape": (int(X_train.shape[0]), int(X_train.shape[1])),
            "X_test_shape": (int(X_test.shape[0]), int(X_test.shape[1])),
            "y_train_shape": (int(y_train.shape[0]),),
            "y_test_shape": (int(y_test.shape[0]),),
            "scaled_numeric_columns": scaled_numeric_cols,
            "artifacts_dir": str(self.processed_dir),
        }

    def load_processed(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train = np.load(self.artifact_paths["X_train"], allow_pickle=True)
        X_test = np.load(self.artifact_paths["X_test"], allow_pickle=True)
        y_train = np.load(self.artifact_paths["y_train"], allow_pickle=True)
        y_test = np.load(self.artifact_paths["y_test"], allow_pickle=True)
        return X_train, X_test, y_train, y_test

    def get_feature_names(self) -> list[str]:
        if self.feature_names_:
            return self.feature_names_

        if not self.feature_names_path.exists():
            raise FileNotFoundError(
                "feature_names.json not found. Run pipeline.run(csv_path) first."
            )

        with self.feature_names_path.open("r", encoding="utf-8") as fp:
            self.feature_names_ = list(json.load(fp))
        return self.feature_names_


def default_csv_path() -> Path:
    return Path(__file__).resolve().parents[2] / "archive" / "diabetic_data.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Diabetes 130-US Hospitals data")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=default_csv_path(),
        help="Path to diabetic_data.csv",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Directory where processed .npy files and scaler.pkl are saved",
    )
    args = parser.parse_args()

    pipeline = PreprocessingPipeline(processed_dir=args.processed_dir)
    shapes = pipeline.run(args.csv_path)
    print("Saved preprocessing artifacts to:", shapes["artifacts_dir"])


if __name__ == "__main__":
    main()
