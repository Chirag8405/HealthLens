# HealthLens Model Report

## 1) Dataset Statistics

| Metric | Value |
|---|---:|
| Dataset file | `archive/diabetic_data.csv` |
| Rows | 101,766 |
| Columns | 50 |
| 30-day readmission positives (`<30`) | 11,357 |
| 30-day readmission rate | 11.16% |

Notes:
- Stats above are from the currently available local diabetes tabular dataset snapshot.
- CNN/Autoencoder metrics are from `models/cnn_results.json` and `models/autoencoder_results.json`.
- LSTM metrics are from `models/lstm_results.json`.
- Classical ML summary artifacts (`models/classification/results.json`, `models/regression/results.json`) were not present at report generation time.

## 2) Side-by-Side Model Metrics

### Classification and Risk Models

| Model | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| Logistic Regression | N/A | N/A | N/A |
| Decision Tree | N/A | N/A | N/A |
| Random Forest | N/A | N/A | N/A |
| KNN | N/A | N/A | N/A |
| SVM | N/A | N/A | N/A |
| ANN (Readmission) | 0.7340 | 0.2660 | 0.6398 |
| CNN (Pneumonia) | 0.8846 | 0.9135 | 0.9611 |
| LSTM (Sepsis Risk) | N/A | N/A | 0.4889 |

### Regression / Forecast / Reconstruction Models

| Model | MAE | RMSE | R2 | Test MSE |
|---|---:|---:|---:|---:|
| Linear Regression | N/A | N/A | N/A | N/A |
| LSTM Vitals Forecast | 0.0321 | N/A | N/A | 0.0018 |
| Autoencoder Reconstruction | N/A | N/A | N/A | 0.0025 |

## 3) Best Models (Current Snapshot)

- Best AUC: CNN (0.9611)
- Best Accuracy: CNN (0.8846)
- Best tabular classification AUC (available): ANN (0.6398)
- Best regression model: Not available in current artifact snapshot

## 4) Training Time Comparison

| Pipeline Step | Time (seconds) |
|---|---:|
| Preprocessing | Pending (`train_all.py`) |
| Regression (Linear/Ridge/Lasso) | Pending (`train_all.py`) |
| Classification (LogReg/DT/RF/KNN/SVM) | Pending (`train_all.py`) |
| ANN | Pending (`train_all.py`) |
| CNN | Pending (`train_all.py`) |
| Autoencoder | Pending (`train_all.py`) |
| LSTM (Task A + Task B) | Pending (`train_all.py`) |
| Total | Pending (`train_all.py`) |

To generate a fully timed report with all model rows populated, run:

```bash
cd backend
python train_all.py --data-dir ./data --models-dir ./models
```

This command writes:
- `models/results_summary.json`
- `../model_report.md` (regenerated)

## 5) Limitations and Next Steps

### Current limitations
- Classical ML summary files are currently missing in this workspace snapshot, so those table rows are `N/A`.
- Inference for `/predict/full` uses a compact structured request and fills many sparse one-hot features with defaults.
- Sepsis and readmission models are trained on different data modalities and should not be interpreted as a single clinically calibrated score.
- SHAP explainability is tied to RandomForest and can add latency under high request throughput.

### Recommended next steps
1. Run `train_all.py` end-to-end to produce complete metrics and timing outputs for all required models.
2. Export and version a training feature-contract artifact (column ordering + category mapping) for stricter inference parity.
3. Add automated calibration and drift checks (e.g., reliability curves, Brier score, PSI/KS monitoring).
4. Cache SHAP explainers and add API-level performance benchmarks for `/predict/full`.
