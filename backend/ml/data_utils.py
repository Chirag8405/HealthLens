from __future__ import annotations

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


def default_csv_path() -> Path:
    return Path(__file__).resolve().parents[2] / "archive" / "diabetic_data.csv"


def map_icd_to_bucket(code: object) -> str:
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


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
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


def prepare_modeling_dataframe(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.replace("?", np.nan)

    missing_ratio = df.isna().mean()
    dynamic_drop = [col for col, ratio in missing_ratio.items() if ratio > 0.40]
    required_drop = [col for col in HIGH_MISSING_COLUMNS if col in df.columns]
    drop_cols = sorted(set(dynamic_drop + required_drop))
    df = df.drop(columns=drop_cols)

    df = coerce_numeric_columns(df)
    df = impute_missing_values(df)

    readmitted_clean = df["readmitted"].astype(str).str.strip().str.upper()
    unknown_values = sorted(set(readmitted_clean.unique()) - set(READMITTED_MAP.keys()))
    if unknown_values:
        raise ValueError(f"Unexpected values in readmitted: {unknown_values}")

    df["readmitted_30"] = (readmitted_clean == "<30").astype(int)
    df["readmitted"] = readmitted_clean.map(READMITTED_MAP).astype(int)

    for col in DIAGNOSIS_COLUMNS:
        group_col = f"{col}_icd_group"
        if col in df.columns:
            df[group_col] = df[col].apply(map_icd_to_bucket)
        else:
            df[group_col] = "Z"

    df["diag_icd_group"] = (
        df["diag_1_icd_group"].astype(str)
        + "_"
        + df["diag_2_icd_group"].astype(str)
        + "_"
        + df["diag_3_icd_group"].astype(str)
    )

    for diag_col in DIAGNOSIS_COLUMNS:
        if diag_col in df.columns:
            df = df.drop(columns=[diag_col])

    if "age" in df.columns:
        df["age_bracket"] = df["age"].astype(str)
        df = df.drop(columns=["age"])

    for col in ("gender", "race", "age_bracket"):
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))

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

    remaining_obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if remaining_obj_cols:
        df[remaining_obj_cols] = df[remaining_obj_cols].astype(str)
        df = pd.get_dummies(df, columns=remaining_obj_cols, drop_first=True, dtype=int)

    return df
