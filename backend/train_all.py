from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from typing import Callable

import pandas as pd

from ml.ann import train_and_evaluate_ann
from ml.autoencoder import train_and_evaluate_autoencoder
from ml.classification import train_and_evaluate_classification
from ml.cnn import train_and_evaluate_cnn
from ml.data_utils import default_csv_path
from ml.lstm import default_lstm_dataset_dirs
from ml.lstm import train_and_evaluate_lstm
from ml.preprocess import PreprocessingPipeline
from ml.regression import train_and_evaluate_regression
from path_utils import project_root_from


@dataclass
class TimerResult:
    name: str
    seconds: float


RESULT_CHECKPOINT_FILES: dict[str, str] = {
    "regression": "regression_results.json",
    "classification": "ml_results.json",
    "ann": "ann_results.json",
    "cnn": "cnn_results.json",
    "autoencoder": "autoencoder_results.json",
    "lstm": "lstm_results.json",
}

PROJECT_ROOT = project_root_from(__file__)


def _round_float(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _resolve_csv_path(data_dir: Path, csv_path_arg: str | None) -> Path:
    if csv_path_arg:
        return Path(csv_path_arg).resolve()

    candidate_paths = [
        data_dir / "diabetic_data.csv",
        data_dir / "archive" / "diabetic_data.csv",
        PROJECT_ROOT / "archive" / "diabetic_data.csv",
        default_csv_path(),
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    return candidate_paths[-1]


def _resolve_chest_xray_dir(data_dir: Path, chest_xray_arg: str | None) -> Path:
    if chest_xray_arg:
        return Path(chest_xray_arg).resolve()

    candidate_paths = [
        data_dir / "chest_xray",
        data_dir,
        PROJECT_ROOT / "chest_xray",
    ]

    for path in candidate_paths:
        if (path / "train").exists() and (path / "val").exists() and (path / "test").exists():
            return path

    return candidate_paths[-1]


def _resolve_lstm_dirs(
    data_dir: Path,
    set_a_arg: str | None,
    set_b_arg: str | None,
) -> tuple[Path, Path]:
    if set_a_arg and set_b_arg:
        return Path(set_a_arg).resolve(), Path(set_b_arg).resolve()

    candidate_pairs = [
        (data_dir / "training" / "setA", data_dir / "training" / "setB"),
        (
            data_dir / "archive (2)" / "training_setA" / "training",
            data_dir / "archive (2)" / "training_setB" / "training_setB",
        ),
    ]

    for set_a_dir, set_b_dir in candidate_pairs:
        if set_a_dir.exists() and set_b_dir.exists():
            return set_a_dir, set_b_dir

    return default_lstm_dataset_dirs()


def _time_call(name: str, fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, TimerResult]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, TimerResult(name=name, seconds=elapsed)


def _extract_model_metrics(summary: dict[str, Any], model_name: str) -> dict[str, Any]:
    model = summary.get("models", {}).get(model_name, {})
    return model.get("metrics", {})


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None

    return payload if isinstance(payload, dict) else None


def _normalize_regression_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("models"), dict):
        return payload

    model_map = {
        "linear_regression": "LinearRegression",
        "ridge": "Ridge",
        "lasso": "Lasso",
    }
    normalized_models: dict[str, Any] = {}

    for payload_key, model_name in model_map.items():
        model_payload = payload.get(payload_key)
        if not isinstance(model_payload, dict):
            continue

        metrics = {
            "mse": model_payload.get("mse"),
            "rmse": model_payload.get("rmse"),
            "mae": model_payload.get("mae"),
            "r2": model_payload.get("r2"),
        }
        if model_payload.get("best_alpha") is not None:
            metrics["best_alpha"] = model_payload.get("best_alpha")

        normalized_models[model_name] = {
            "model_name": model_name,
            "metrics": metrics,
            "actual_vs_predicted_plot": model_payload.get("actual_vs_predicted_b64", ""),
        }

    return {
        "task": "regression",
        "target": "time_in_hospital",
        "models": normalized_models,
    }


def _normalize_classification_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("models"), dict):
        return payload

    model_map = {
        "logistic_regression": "LogisticRegression",
        "decision_tree": "DecisionTreeClassifier",
        "random_forest": "RandomForestClassifier",
        "knn": "KNeighborsClassifier",
        "svm": "SVC",
    }
    normalized_models: dict[str, Any] = {}

    for payload_key, model_name in model_map.items():
        model_payload = payload.get(payload_key)
        if not isinstance(model_payload, dict):
            continue

        normalized_models[model_name] = {
            "model_name": model_name,
            "best_params": {},
            "metrics": {
                "accuracy": model_payload.get("accuracy"),
                "precision": model_payload.get("precision"),
                "recall": model_payload.get("recall"),
                "f1_weighted": model_payload.get("f1"),
                "auc_roc": model_payload.get("auc"),
            },
            "confusion_matrix_plot": model_payload.get("confusion_matrix_b64", ""),
        }

    return {
        "task": "classification",
        "target": "readmitted_30",
        "models": normalized_models,
        "roc_curve_plot": payload.get("roc_overlay_b64", ""),
    }


def _normalize_cached_model_summary(model_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    if model_key == "regression":
        return _normalize_regression_summary(payload)
    if model_key == "classification":
        return _normalize_classification_summary(payload)
    return payload


def _build_dataset_stats(csv_path: Path, preprocessing_summary: dict[str, Any] | None) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "csv_path": str(csv_path),
        "rows": None,
        "columns": None,
        "readmitted_30_positive": None,
        "readmitted_30_rate": None,
        "train_shape": None,
        "test_shape": None,
        "processed_features": None,
    }

    if preprocessing_summary:
        stats["train_shape"] = preprocessing_summary.get("X_train_shape")
        stats["test_shape"] = preprocessing_summary.get("X_test_shape")
        train_shape = preprocessing_summary.get("X_train_shape")
        if isinstance(train_shape, (list, tuple)) and len(train_shape) > 1:
            stats["processed_features"] = train_shape[1]

    try:
        df = pd.read_csv(csv_path)
        stats["rows"] = int(df.shape[0])
        stats["columns"] = int(df.shape[1])
        if "readmitted" in df.columns:
            positive = int((df["readmitted"].astype(str).str.upper() == "<30").sum())
            stats["readmitted_30_positive"] = positive
            stats["readmitted_30_rate"] = _round_float(positive / max(len(df), 1), 6)
    except Exception:
        # Keep stats partial if reading fails.
        pass

    return stats


def _build_results_summary(
    csv_path: Path,
    chest_xray_dir: Path,
    lstm_set_a: Path,
    lstm_set_b: Path,
    preprocessing_summary: dict[str, Any],
    regression_summary: dict[str, Any],
    classification_summary: dict[str, Any],
    ann_summary: dict[str, Any],
    cnn_summary: dict[str, Any],
    autoencoder_summary: dict[str, Any],
    lstm_summary: dict[str, Any],
    timers: list[TimerResult],
    cached_modules: list[str],
) -> dict[str, Any]:
    timing_map = {timer.name: round(timer.seconds, 3) for timer in timers}

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "csv_path": str(csv_path),
            "chest_xray_dir": str(chest_xray_dir),
            "lstm_set_a": str(lstm_set_a),
            "lstm_set_b": str(lstm_set_b),
        },
        "dataset_stats": _build_dataset_stats(csv_path, preprocessing_summary),
        "timings_seconds": timing_map,
        "cached_modules": sorted(cached_modules),
        "models": {
            "preprocessing": preprocessing_summary,
            "linear_regression": _extract_model_metrics(regression_summary, "LinearRegression"),
            "logistic_regression": _extract_model_metrics(classification_summary, "LogisticRegression"),
            "decision_tree": _extract_model_metrics(classification_summary, "DecisionTreeClassifier"),
            "random_forest": _extract_model_metrics(classification_summary, "RandomForestClassifier"),
            "knn": _extract_model_metrics(classification_summary, "KNeighborsClassifier"),
            "svm": _extract_model_metrics(classification_summary, "SVC"),
            "ann": ann_summary.get("metrics", {}),
            "cnn": cnn_summary.get("metrics", {}),
            "autoencoder": autoencoder_summary.get("metrics", {}),
            "lstm_vitals": lstm_summary.get("task_a_vitals", {}).get("metrics", {}),
            "lstm_sepsis": {
                "auc_roc": lstm_summary.get("task_b_sepsis", {}).get("auc_roc"),
                "base_rate": lstm_summary.get("task_b_sepsis", {}).get("base_rate"),
                "topk_metrics": lstm_summary.get("task_b_sepsis", {}).get("topk_metrics", {}),
            },
        },
    }

    return summary


def _print_summary_table(summary: dict[str, Any]) -> None:
    rows: list[tuple[str, str]] = []

    models = summary.get("models", {})
    timings = summary.get("timings_seconds", {})
    cached_modules = summary.get("cached_modules", [])

    rows.extend(
        [
            ("Linear Regression (R2)", str(models.get("linear_regression", {}).get("r2"))),
            ("Logistic Regression (AUC)", str(models.get("logistic_regression", {}).get("auc_roc"))),
            ("Decision Tree (AUC)", str(models.get("decision_tree", {}).get("auc_roc"))),
            ("Random Forest (AUC)", str(models.get("random_forest", {}).get("auc_roc"))),
            ("KNN (AUC)", str(models.get("knn", {}).get("auc_roc"))),
            ("SVM (AUC)", str(models.get("svm", {}).get("auc_roc"))),
            ("ANN (AUC)", str(models.get("ann", {}).get("auc_roc"))),
            ("CNN (AUC)", str(models.get("cnn", {}).get("auc"))),
            ("Autoencoder (Test MSE)", str(models.get("autoencoder", {}).get("test_mse"))),
            ("LSTM Vitals (Test MAE)", str(models.get("lstm_vitals", {}).get("test_mae"))),
            ("LSTM Sepsis (AUC)", str(models.get("lstm_sepsis", {}).get("auc_roc"))),
            ("Training Time - Total (s)", str(sum(float(v) for v in timings.values()))),
        ]
    )

    print("\n=== HEALTHLENS TRAINING SUMMARY ===")
    for label, value in rows:
        print(f"{label:<36} : {value}")

    if cached_modules:
        print("Cached modules:", ", ".join(sorted(str(name) for name in cached_modules)))
        print("Note: Modules showing 0.000s were loaded from cached artifacts. Use --force to retrain all modules.")


def _best_model(metrics: dict[str, dict[str, Any]], key: str, maximize: bool = True) -> tuple[str, float] | None:
    candidates: list[tuple[str, float]] = []
    for name, values in metrics.items():
        metric_value = values.get(key)
        try:
            if metric_value is not None:
                candidates.append((name, float(metric_value)))
        except (TypeError, ValueError):
            continue

    if not candidates:
        return None

    return max(candidates, key=lambda x: x[1]) if maximize else min(candidates, key=lambda x: x[1])


def _format_metric(value: Any, precision: int = 4) -> str:
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "N/A"


def _write_model_report(summary: dict[str, Any], report_path: Path) -> None:
    dataset = summary.get("dataset_stats", {})
    models = summary.get("models", {})
    timings = summary.get("timings_seconds", {})
    cached_modules = {str(name) for name in summary.get("cached_modules", [])}

    classification_metrics = {
        "Logistic Regression": models.get("logistic_regression", {}),
        "Decision Tree": models.get("decision_tree", {}),
        "Random Forest": models.get("random_forest", {}),
        "KNN": models.get("knn", {}),
        "SVM": models.get("svm", {}),
        "ANN": models.get("ann", {}),
        "CNN": models.get("cnn", {}),
        "LSTM Sepsis": {"auc_roc": models.get("lstm_sepsis", {}).get("auc_roc")},
    }

    regression_metrics = {
        "Linear Regression": models.get("linear_regression", {}),
        "LSTM Vitals": models.get("lstm_vitals", {}),
        "Autoencoder": models.get("autoencoder", {}),
    }

    best_auc = _best_model(classification_metrics, "auc_roc", maximize=True)
    best_accuracy = _best_model(classification_metrics, "accuracy", maximize=True)
    best_rmse = _best_model(
        {
            name: {"rmse": values.get("rmse")}
            for name, values in regression_metrics.items()
            if values.get("rmse") is not None
        },
        "rmse",
        maximize=False,
    )

    training_rows = "\n".join(
        f"| {name} | {_format_metric(seconds, 3)} | {'(cached — loaded from previous run)' if name in cached_modules else ''} |"
        for name, seconds in sorted(timings.items(), key=lambda x: x[0])
    )
    cache_note = (
        "Note: Modules showing 0.000s were loaded from cached artifacts. Use --force to retrain all modules."
        if cached_modules
        else "Note: All modules were freshly trained in this run."
    )

    report = f"""# HealthLens Model Report

## 1. Dataset Statistics

| Metric | Value |
|---|---:|
| Source CSV | {dataset.get("csv_path", "N/A")} |
| Rows | {dataset.get("rows", "N/A")} |
| Columns | {dataset.get("columns", "N/A")} |
| Positive 30-day readmissions | {dataset.get("readmitted_30_positive", "N/A")} |
| Positive rate | {_format_metric(dataset.get("readmitted_30_rate"), 4)} |
| Train shape | {dataset.get("train_shape", "N/A")} |
| Test shape | {dataset.get("test_shape", "N/A")} |
| Processed feature count | {dataset.get("processed_features", "N/A")} |

## 2. Model Metrics (Side-by-Side)

### Classification and Risk Models

| Model | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| Logistic Regression | {_format_metric(models.get("logistic_regression", {}).get("accuracy"))} | {_format_metric(models.get("logistic_regression", {}).get("f1_weighted"))} | {_format_metric(models.get("logistic_regression", {}).get("auc_roc"))} |
| Decision Tree | {_format_metric(models.get("decision_tree", {}).get("accuracy"))} | {_format_metric(models.get("decision_tree", {}).get("f1_weighted"))} | {_format_metric(models.get("decision_tree", {}).get("auc_roc"))} |
| Random Forest | {_format_metric(models.get("random_forest", {}).get("accuracy"))} | {_format_metric(models.get("random_forest", {}).get("f1_weighted"))} | {_format_metric(models.get("random_forest", {}).get("auc_roc"))} |
| KNN | {_format_metric(models.get("knn", {}).get("accuracy"))} | {_format_metric(models.get("knn", {}).get("f1_weighted"))} | {_format_metric(models.get("knn", {}).get("auc_roc"))} |
| SVM | {_format_metric(models.get("svm", {}).get("accuracy"))} | {_format_metric(models.get("svm", {}).get("f1_weighted"))} | {_format_metric(models.get("svm", {}).get("auc_roc"))} |
| ANN | {_format_metric(models.get("ann", {}).get("accuracy"))} | {_format_metric(models.get("ann", {}).get("f1"))} | {_format_metric(models.get("ann", {}).get("auc_roc"))} |
| CNN | {_format_metric(models.get("cnn", {}).get("accuracy"))} | {_format_metric(models.get("cnn", {}).get("f1"))} | {_format_metric(models.get("cnn", {}).get("auc"))} |
| LSTM Sepsis | N/A | N/A | {_format_metric(models.get("lstm_sepsis", {}).get("auc_roc"))} |

### Regression / Reconstruction / Forecast Models

| Model | MAE | RMSE | R2 | Test MSE |
|---|---:|---:|---:|---:|
| Linear Regression | {_format_metric(models.get("linear_regression", {}).get("mae"))} | {_format_metric(models.get("linear_regression", {}).get("rmse"))} | {_format_metric(models.get("linear_regression", {}).get("r2"))} | N/A |
| LSTM Vitals Forecast | {_format_metric(models.get("lstm_vitals", {}).get("test_mae"))} | N/A | N/A | {_format_metric(models.get("lstm_vitals", {}).get("test_mse"))} |
| Autoencoder Reconstruction | N/A | N/A | N/A | {_format_metric(models.get("autoencoder", {}).get("test_mse"))} |

## 3. Best Models

- Best AUC model: {best_auc[0] if best_auc else "N/A"} ({_format_metric(best_auc[1]) if best_auc else "N/A"})
- Best Accuracy model: {best_accuracy[0] if best_accuracy else "N/A"} ({_format_metric(best_accuracy[1]) if best_accuracy else "N/A"})
- Best RMSE model (lower is better): {best_rmse[0] if best_rmse else "N/A"} ({_format_metric(best_rmse[1]) if best_rmse else "N/A"})

## 4. Training Time Comparison

| Pipeline Step | Time (seconds) | Note |
|---|---:|---|
{training_rows if training_rows else "| N/A | N/A | N/A |"}

{cache_note}

## 5. Limitations and Next Steps

### Current limitations
- Inference uses a reduced structured feature set and default values for many sparse one-hot features.
- Model drift monitoring and calibration tracking are not yet automated in production.
- Deep learning training cost is high; repeated full retraining can be time-intensive.
- Explainability is currently tied to random forest SHAP at prediction time and may add latency.

### Recommended next steps
1. Add a reproducible feature-contract artifact (column schema + category encoding map) from training and consume it in inference.
2. Add test-time and production calibration checks (Brier score, reliability curves) for ANN and RF outputs.
3. Persist and compare historical training metrics to detect regressions automatically in CI.
4. Add model serving benchmarks and caching for SHAP explainers in long-running API processes.

---
Report generated at: {summary.get("generated_at_utc", "N/A")}
"""

    report_path.write_text(report, encoding="utf-8")


def _load_existing_results_summary(models_dir: Path) -> dict[str, Any] | None:
    summary_path = models_dir / "results_summary.json"
    return _safe_read_json(summary_path)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    project_root = PROJECT_ROOT
    data_dir = Path(args.data_dir).resolve()
    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    csv_path = _resolve_csv_path(data_dir, args.csv_path)
    chest_xray_dir = _resolve_chest_xray_dir(data_dir, args.chest_xray_dir)
    lstm_set_a, lstm_set_b = _resolve_lstm_dirs(data_dir, args.lstm_set_a_dir, args.lstm_set_b_dir)

    if args.report_only:
        existing = _load_existing_results_summary(models_dir)
        if existing is None:
            raise FileNotFoundError(
                f"No existing results summary found at {models_dir / 'results_summary.json'}. "
                "Run full training first or remove --report-only."
            )
        report_path = Path(args.report_path).resolve() if args.report_path else project_root / "model_report.md"
        _write_model_report(existing, report_path)
        print(f"Report written to: {report_path}")
        return existing

    result_paths = {
        key: models_dir / filename for key, filename in RESULT_CHECKPOINT_FILES.items()
    }

    def _load_or_train_model(
        model_key: str,
        model_name: str,
        train_callable: Callable[[], dict[str, Any]],
    ) -> tuple[dict[str, Any], TimerResult, bool]:
        result_path = result_paths[model_key]
        if not args.force and result_path.exists():
            print(f"[SKIP] {model_name} — cached results found")
            cached_payload = _safe_read_json(result_path)
            if cached_payload is not None:
                return (
                    _normalize_cached_model_summary(model_key, cached_payload),
                    TimerResult(name=model_key, seconds=0.0),
                    True,
                )
            print(f"[WARN] {model_name} cache unreadable, retraining")

        result, timer = _time_call(model_key, train_callable)
        return result, timer, False

    preprocessing_pipeline = PreprocessingPipeline(processed_dir=data_dir / "processed")

    preprocessing_summary, timer_pre = _time_call("preprocessing", preprocessing_pipeline.run, csv_path)
    regression_summary, timer_reg, regression_cached = _load_or_train_model(
        model_key="regression",
        model_name="Regression",
        train_callable=lambda: train_and_evaluate_regression(csv_path=csv_path, models_dir=models_dir),
    )
    classification_summary, timer_cls, classification_cached = _load_or_train_model(
        model_key="classification",
        model_name="Classification",
        train_callable=lambda: train_and_evaluate_classification(
            csv_path=csv_path,
            models_dir=models_dir,
            skip_svm=args.skip_svm,
        ),
    )
    ann_summary, timer_ann, ann_cached = _load_or_train_model(
        model_key="ann",
        model_name="ANN",
        train_callable=lambda: train_and_evaluate_ann(csv_path=csv_path, models_dir=models_dir),
    )
    cnn_summary, timer_cnn, cnn_cached = _load_or_train_model(
        model_key="cnn",
        model_name="CNN",
        train_callable=lambda: train_and_evaluate_cnn(dataset_root=chest_xray_dir, models_dir=models_dir),
    )
    autoencoder_summary, timer_auto, autoencoder_cached = _load_or_train_model(
        model_key="autoencoder",
        model_name="Autoencoder",
        train_callable=lambda: train_and_evaluate_autoencoder(
            dataset_root=chest_xray_dir,
            models_dir=models_dir,
        ),
    )
    lstm_summary, timer_lstm, lstm_cached = _load_or_train_model(
        model_key="lstm",
        model_name="LSTM",
        train_callable=lambda: train_and_evaluate_lstm(
            set_a_dir=lstm_set_a,
            set_b_dir=lstm_set_b,
            models_dir=models_dir,
            max_patients=args.max_patients,
        ),
    )

    cached_modules: list[str] = []
    if regression_cached:
        cached_modules.append("regression")
    if classification_cached:
        cached_modules.append("classification")
    if ann_cached:
        cached_modules.append("ann")
    if cnn_cached:
        cached_modules.append("cnn")
    if autoencoder_cached:
        cached_modules.append("autoencoder")
    if lstm_cached:
        cached_modules.append("lstm")

    timers = [timer_pre, timer_reg, timer_cls, timer_ann, timer_cnn, timer_auto, timer_lstm]

    summary = _build_results_summary(
        csv_path=csv_path,
        chest_xray_dir=chest_xray_dir,
        lstm_set_a=lstm_set_a,
        lstm_set_b=lstm_set_b,
        preprocessing_summary=preprocessing_summary,
        regression_summary=regression_summary,
        classification_summary=classification_summary,
        ann_summary=ann_summary,
        cnn_summary=cnn_summary,
        autoencoder_summary=autoencoder_summary,
        lstm_summary=lstm_summary,
        timers=timers,
        cached_modules=cached_modules,
    )

    summary_path = models_dir / "results_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    report_path = Path(args.report_path).resolve() if args.report_path else project_root / "model_report.md"
    _write_model_report(summary, report_path)

    _print_summary_table(summary)
    print(f"\nSaved summary JSON: {summary_path}")
    print(f"Saved markdown report: {report_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all HealthLens ML and DL models")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Data root directory (expects diabetic_data.csv and dataset subfolders).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(PROJECT_ROOT / "models"),
        help="Directory where model artifacts and summary files are written.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional explicit path to diabetic_data.csv.",
    )
    parser.add_argument(
        "--chest-xray-dir",
        type=str,
        default=None,
        help="Optional explicit path to chest_xray directory.",
    )
    parser.add_argument(
        "--lstm-set-a-dir",
        type=str,
        default=None,
        help="Optional explicit path to PhysioNet setA directory.",
    )
    parser.add_argument(
        "--lstm-set-b-dir",
        type=str,
        default=None,
        help="Optional explicit path to PhysioNet setB directory.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=1200,
        help="Maximum number of patients used for LSTM training.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Optional output path for model_report.md.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only regenerate model_report.md from an existing results_summary.json file.",
    )
    parser.add_argument(
        "--skip-svm",
        action="store_true",
        help="Skip SVM during classification training to reduce CPU runtime.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining all models even when cached result files are present.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
