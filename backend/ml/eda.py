from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HIGH_MISSING_COLUMNS = ("weight", "payer_code", "medical_specialty")
DIAGNOSIS_COLUMNS = ("diag_1", "diag_2", "diag_3")


class EDA:
    def __init__(self, csv_path: str | Path | None = None) -> None:
        self.csv_path = Path(csv_path) if csv_path is not None else self._default_csv_path()
        self.df = self._load_and_prepare()

    @staticmethod
    def _default_csv_path() -> Path:
        return Path(__file__).resolve().parents[2] / "archive" / "diabetic_data.csv"

    def _load_and_prepare(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        df = df.replace("?", np.nan)

        missing_ratio = df.isna().mean()
        dynamic_drop = [col for col, ratio in missing_ratio.items() if ratio > 0.40]
        required_drop = [col for col in HIGH_MISSING_COLUMNS if col in df.columns]
        drop_cols = sorted(set(dynamic_drop + required_drop))
        df = df.drop(columns=drop_cols)

        df = self._coerce_numeric_columns(df)
        df = self._impute_missing_values(df)

        readmitted_clean = df["readmitted"].astype(str).str.strip().str.upper()
        df["readmitted"] = readmitted_clean
        df["readmitted_30"] = (readmitted_clean == "<30").astype(int)

        for diag_col in DIAGNOSIS_COLUMNS:
            grouped_col = f"{diag_col}_icd_group"
            if diag_col in df.columns:
                df[grouped_col] = df[diag_col].apply(self._map_icd_to_bucket)
            else:
                df[grouped_col] = "Z"

        return df

    @staticmethod
    def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
        categorical_like = set(DIAGNOSIS_COLUMNS) | {
            "readmitted",
            "race",
            "gender",
            "age",
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
        }

        for col in df.columns:
            if col in categorical_like or not pd.api.types.is_object_dtype(df[col]):
                continue

            non_null = df[col].dropna()
            if non_null.empty:
                continue

            convertible_ratio = pd.to_numeric(non_null, errors="coerce").notna().mean()
            if convertible_ratio >= 0.95:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    @staticmethod
    def _impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
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

    @staticmethod
    def _figure_to_base64() -> str:
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        buffer.close()
        plt.close()
        return image_b64

    def age_distribution(self) -> str:
        plt.figure(figsize=(10,6))
        age_counts = self.df["age"].value_counts()

        def age_sort_key(label: str) -> int:
            digits = "".join(ch for ch in str(label) if ch.isdigit())
            return int(digits[:2]) if digits else 999

        ordered_labels = sorted(age_counts.index.tolist(), key=age_sort_key)
        ordered_counts = age_counts.reindex(ordered_labels)

        sns.barplot(x=ordered_counts.index, y=ordered_counts.values, color="#2563eb")
        plt.title("Age Distribution")
        plt.xlabel("Age Bracket")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        return self._figure_to_base64()

    def readmission_rates(self) -> str:
        plt.figure(figsize=(10,6))
        order = ["NO", ">30", "<30"]
        sns.countplot(data=self.df, x="readmitted", order=order, palette="Set2")
        plt.title("Readmission Classes")
        plt.xlabel("Readmitted")
        plt.ylabel("Count")
        return self._figure_to_base64()

    def correlation_heatmap(self) -> str:
        plt.figure(figsize=(10,6))
        numeric_df = self.df.select_dtypes(include=[np.number]).copy()
        if numeric_df.empty:
            plt.text(0.5, 0.5, "No numerical data available", ha="center", va="center")
            plt.axis("off")
            return self._figure_to_base64()

        variances = numeric_df.var(numeric_only=True).sort_values(ascending=False)
        top_cols = variances.head(20).index.tolist()
        corr_matrix = numeric_df[top_cols].corr()

        sns.heatmap(corr_matrix, cmap="coolwarm", center=0, linewidths=0.5)
        plt.title("Correlation Heatmap (Top 20 Numerical Features)")
        return self._figure_to_base64()

    def los_vs_cost(self) -> str:
        plt.figure(figsize=(10,6))
        sns.scatterplot(
            data=self.df,
            x="time_in_hospital",
            y="num_medications",
            hue="readmitted_30",
            palette="Set1",
            alpha=0.7,
        )
        plt.title("Length of Stay vs Medication Count")
        plt.xlabel("Time in Hospital (days)")
        plt.ylabel("Number of Medications")
        plt.legend(title="readmitted_30")
        return self._figure_to_base64()

    def diagnosis_frequency(self) -> str:
        plt.figure(figsize=(10,6))
        diag_group_cols = [f"{col}_icd_group" for col in DIAGNOSIS_COLUMNS]
        stacked = pd.concat([self.df[col] for col in diag_group_cols], axis=0)
        top_freq = stacked.value_counts().head(15)

        sns.barplot(x=top_freq.index, y=top_freq.values, color="#0f766e")
        plt.title("Top ICD Chapter Group Frequency")
        plt.xlabel("ICD Chapter Group")
        plt.ylabel("Frequency")
        return self._figure_to_base64()

    def class_imbalance(self) -> str:
        plt.figure(figsize=(10,6))
        counts = self.df["readmitted_30"].value_counts().sort_index()
        labels = [f"Class {int(idx)}" for idx in counts.index]

        plt.pie(counts.values, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title("Class Imbalance (readmitted_30)")
        return self._figure_to_base64()

    def summary(self) -> dict[str, Any]:
        numeric_df = self.df.select_dtypes(include=[np.number])
        numeric_summary = (
            numeric_df.describe().transpose()[["mean", "std", "min", "max"]].round(4)
            if not numeric_df.empty
            else pd.DataFrame()
        )

        summary_payload: dict[str, Any] = {
            "rows": int(self.df.shape[0]),
            "columns": int(self.df.shape[1]),
            "missing_values_total": int(self.df.isna().sum().sum()),
            "readmitted_distribution": {
                str(k): int(v) for k, v in self.df["readmitted"].value_counts().to_dict().items()
            },
            "readmitted_30_distribution": {
                str(k): int(v)
                for k, v in self.df["readmitted_30"].value_counts().sort_index().to_dict().items()
            },
            "numerical_summary": {
                col: {k: float(v) for k, v in stats.items()}
                for col, stats in numeric_summary.to_dict(orient="index").items()
            },
        }

        return summary_payload
