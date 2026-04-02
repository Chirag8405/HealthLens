from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

HIGH_MISSING_COLUMNS = ("weight", "payer_code", "medical_specialty")
DIAGNOSIS_COLUMNS = ("diag_1", "diag_2", "diag_3")
READMITTED_MAP = {"NO": 0, ">30": 1, "<30": 2}
ONE_HOT_COLUMN_MAP = {
    "admission_type": "admission_type_id",
    "discharge_disposition": "discharge_disposition_id",
    "admission_source": "admission_source_id",
}


@dataclass
class PreprocessResult:
    data: pd.DataFrame
    X: pd.DataFrame
    y_multiclass: pd.Series
    y_binary: pd.Series
    label_encoders: dict[str, LabelEncoder]


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.replace("?", np.nan)
    return df


def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.40) -> pd.DataFrame:
    missing_ratio = df.isna().mean()
    auto_drop = [col for col, ratio in missing_ratio.items() if ratio > threshold]
    required_drop = [col for col in HIGH_MISSING_COLUMNS if col in df.columns]
    cols_to_drop = sorted(set(auto_drop + required_drop))
    return df.drop(columns=cols_to_drop)


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    skip_columns = (
        set(DIAGNOSIS_COLUMNS)
        | {"readmitted"}
        | set(ONE_HOT_COLUMN_MAP.keys())
        | set(ONE_HOT_COLUMN_MAP.values())
    )

    for col in df.columns:
        if col in skip_columns or not pd.api.types.is_object_dtype(df[col]):
            continue

        non_null_values = df[col].dropna()
        if non_null_values.empty:
            continue

        convertible_ratio = pd.to_numeric(non_null_values, errors="coerce").notna().mean()
        if convertible_ratio >= 0.95:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    categorical_override = (
        set(DIAGNOSIS_COLUMNS)
        | set(ONE_HOT_COLUMN_MAP.keys())
        | set(ONE_HOT_COLUMN_MAP.values())
        | {"readmitted", "race", "gender", "age"}
    )

    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns.tolist()
        if col not in categorical_override
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


def map_icd_to_bucket(code: object) -> str:
    if pd.isna(code):
        return "Z"

    value = str(code).strip().upper()
    if not value or value == "NAN":
        return "Z"

    if value[0].isalpha():
        return value[0] if "A" <= value[0] <= "Z" else "Z"

    numeric_part = []
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


def engineer_targets(df: pd.DataFrame) -> pd.DataFrame:
    if "readmitted" not in df.columns:
        raise KeyError("The dataset must contain the 'readmitted' column.")

    readmitted_clean = df["readmitted"].astype(str).str.strip().str.upper()
    unknown = sorted(set(readmitted_clean.unique()) - set(READMITTED_MAP.keys()))
    if unknown:
        raise ValueError(f"Unexpected values in readmitted column: {unknown}")

    df["readmitted_30"] = (readmitted_clean == "<30").astype(int)
    df["readmitted"] = readmitted_clean.map(READMITTED_MAP).astype(int)
    return df


def engineer_diagnosis_groups(df: pd.DataFrame) -> pd.DataFrame:
    for col in DIAGNOSIS_COLUMNS:
        chapter_col = f"{col}_icd_chapter"
        if col in df.columns:
            df[chapter_col] = df[col].apply(map_icd_to_bucket)
        else:
            df[chapter_col] = "Z"

    df["diag_icd_chapter_combo"] = (
        df["diag_1_icd_chapter"].astype(str)
        + "_"
        + df["diag_2_icd_chapter"].astype(str)
        + "_"
        + df["diag_3_icd_chapter"].astype(str)
    )
    return df


def label_encode_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    encoders: dict[str, LabelEncoder] = {}

    if "age" in df.columns:
        df["age_bracket"] = df["age"].astype(str)

    for col in ("gender", "race", "age_bracket"):
        if col not in df.columns:
            continue

        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder

    return df, encoders


def one_hot_encode_columns(df: pd.DataFrame) -> pd.DataFrame:
    one_hot_columns: list[str] = []

    for normalized_col, source_col in ONE_HOT_COLUMN_MAP.items():
        if normalized_col in df.columns:
            one_hot_columns.append(normalized_col)
            if source_col in df.columns and source_col != normalized_col:
                df = df.drop(columns=[source_col])
            continue

        if source_col in df.columns:
            df = df.rename(columns={source_col: normalized_col})
            one_hot_columns.append(normalized_col)

    if one_hot_columns:
        df[one_hot_columns] = df[one_hot_columns].astype(str)
        df = pd.get_dummies(df, columns=one_hot_columns, prefix=one_hot_columns, dtype=int)

    return df


def preprocess_diabetes_data(csv_path: str | Path) -> PreprocessResult:
    df = load_raw_data(csv_path)
    df = drop_high_missing_columns(df)
    df = coerce_numeric_columns(df)
    df = impute_missing_values(df)

    df = engineer_targets(df)
    df = engineer_diagnosis_groups(df)
    df, label_encoders = label_encode_columns(df)
    df = one_hot_encode_columns(df)

    y_multiclass = df["readmitted"].copy()
    y_binary = df["readmitted_30"].copy()
    X = df.drop(columns=["readmitted", "readmitted_30"])

    return PreprocessResult(
        data=df,
        X=X,
        y_multiclass=y_multiclass,
        y_binary=y_binary,
        label_encoders=label_encoders,
    )


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
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the preprocessed CSV",
    )
    args = parser.parse_args()

    result = preprocess_diabetes_data(args.csv_path)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.data.to_csv(args.output, index=False)
        print(f"Saved preprocessed data to: {args.output}")

    print(f"Processed rows: {result.data.shape[0]}")
    print(f"Processed columns: {result.data.shape[1]}")
    print(f"Feature matrix shape: {result.X.shape}")


if __name__ == "__main__":
    main()
