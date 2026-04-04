# HealthLens Model Report

## 1. Dataset Statistics

| Metric | Value |
|---|---:|
| Source CSV | /home/chirag/Desktop/HealthLens/archive/diabetic_data.csv |
| Rows | 101766 |
| Columns | 50 |
| Positive 30-day readmissions | 11357 |
| Positive rate | 0.1116 |
| Train shape | (81412, 90) |
| Test shape | (20354, 90) |
| Processed feature count | 90 |

## 2. Model Metrics (Side-by-Side)

### Classification and Risk Models

| Model | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.8884 | 0.8379 | 0.6766 |
| Decision Tree | 0.8009 | 0.8068 | 0.5300 |
| Random Forest | 0.7532 | 0.7875 | 0.6789 |
| KNN | 0.8854 | 0.8413 | 0.5805 |
| SVM | N/A | N/A | N/A |
| ANN | 0.7387 | 0.2646 | 0.6397 |
| CNN | 0.8766 | 0.9082 | 0.9626 |
| LSTM Sepsis | N/A | N/A | 0.5258 |

### Regression / Reconstruction / Forecast Models

| Model | MAE | RMSE | R2 | Test MSE |
|---|---:|---:|---:|---:|
| Linear Regression | 107547595132.6749 | 2072168934001.9255 | -493318792562231981637632.0000 | N/A |
| LSTM Vitals Forecast | 0.0364 | N/A | N/A | 0.0023 |
| Autoencoder Reconstruction | N/A | N/A | N/A | 0.0026 |

## 3. Best Models

- Best AUC model: Random Forest (0.6789)
- Best Accuracy model: Logistic Regression (0.8884)
- Best RMSE model (lower is better): Linear Regression (2072168934001.9255)

## 4. Training Time Comparison

| Pipeline Step | Time (seconds) |
|---|---:|
| ann | 0.000 |
| autoencoder | 1613.932 |
| classification | 0.000 |
| cnn | 1270.018 |
| lstm | 500.541 |
| preprocessing | 2.593 |
| regression | 0.000 |

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
Report generated at: 2026-04-04T06:47:39.564884+00:00
