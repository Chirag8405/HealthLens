# HealthLens

HealthLens is a multi-model healthcare analytics and prediction platform with:

- Tabular ML for hospital readmission risk
- Deep learning for X-ray classification and denoising
- LSTM time-series modeling for vitals forecasting and sepsis risk stratification
- A Next.js frontend dashboard for EDA, model diagnostics, and live prediction flows
- A FastAPI backend that serves training and inference endpoints

## Repository Layout

```text
HealthLens/
  backend/
    api/
      main.py
      routers/
    ml/
      ann.py
      classification.py
      regression.py
      cnn.py
      autoencoder.py
      lstm.py
      preprocess.py
    train_all.py
    requirements.txt
  frontend/
    app/
    components/
    lib/
  models/
  archive/
  chest_xray/
  model_report.md
```

## 1) Dataset Download and Placement

HealthLens expects three data sources:

### A) Diabetes tabular dataset (readmission)
- File expected: `diabetic_data.csv`
- Place in one of:
  - `archive/diabetic_data.csv` (default)
  - `<data-dir>/diabetic_data.csv` when using `--data-dir`

### B) Chest X-ray dataset (CNN + Autoencoder)
- Folder expected with splits:

```text
chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

### C) PhysioNet-style time-series dataset (LSTM)
- Expected directories:
  - `training/setA`
  - `training/setB`
- Alternate supported structure is also handled internally via fallback paths.

## 2) Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run backend API:

```bash
uvicorn api.main:app --reload --port 8000
```

Health check:
- `GET http://localhost:8000/health`

## 3) Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:
- `http://localhost:3000`

Optional env config:
- `NEXT_PUBLIC_API_BASE_URL` (defaults to `http://localhost:8000`)

## 4) Train All Models (Unified Script)

From `backend/`:

```bash
python train_all.py --data-dir ./data --models-dir ./models
```

What it does:
- Runs preprocessing
- Trains tabular ML models (including Linear Regression, Logistic Regression, Decision Tree, Random Forest, KNN, SVM)
- Trains ANN, CNN, Autoencoder, and LSTM (Task A + Task B)
- Writes summary JSON to `models/results_summary.json`
- Regenerates model report at `../model_report.md`
- Prints a formatted console summary

Useful optional args:
- `--csv-path`
- `--chest-xray-dir`
- `--lstm-set-a-dir`
- `--lstm-set-b-dir`
- `--max-patients`
- `--report-path`
- `--report-only`

## 5) API Documentation

### Core

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health check |

### EDA

| Method | Endpoint | Description |
|---|---|---|
| GET | `/eda/summary` | Tabular dataset summary |
| GET | `/eda/plots` | Plot payloads as base64 |

### ML (Tabular)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ml/train` | Train regression, classification, clustering |
| GET | `/ml/results` | Return regression + classification outputs |
| GET | `/ml/clusters` | Return clustering outputs |

### DL

| Method | Endpoint | Description |
|---|---|---|
| GET | `/dl/ann` | ANN train/get cached results |
| POST | `/dl/cnn/train` | Train CNN |
| GET | `/dl/cnn/results` | CNN metrics and plots |
| POST | `/dl/cnn/predict` | X-ray inference (file upload) |
| GET | `/dl/autoencoder/results` | Autoencoder metrics and reconstructions |
| GET | `/dl/lstm/results` | LSTM Task A + Task B metrics |
| POST | `/dl/lstm/predict` | LSTM sepsis inference (PSV upload) |

### Predict

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict/full` | Unified readmission risk prediction with SHAP top factors |
| POST | `/predict/risk` | Backward-compatible lightweight risk score endpoint |

## 6) `/predict/full` Request/Response

### Request example

```json
{
  "age": 63,
  "gender": "Female",
  "race": "Caucasian",
  "time_in_hospital": 4,
  "num_lab_procedures": 48,
  "num_procedures": 1,
  "num_medications": 18,
  "number_outpatient": 0,
  "number_emergency": 1,
  "number_inpatient": 2,
  "number_diagnoses": 8,
  "admission_type_id": 1,
  "discharge_disposition_id": 3,
  "admission_source_id": 7,
  "a1c_result": ">8",
  "max_glu_serum": ">300",
  "insulin": "Up",
  "change": "Ch",
  "diabetes_med": "Yes"
}
```

### Response example

```json
{
  "readmission_risk_30day": 0.742381,
  "risk_level": "high",
  "top_risk_factors": [
    {"feature": "number_inpatient", "value": 2.0, "impact": 0.051237},
    {"feature": "discharge_high_risk", "value": 1.0, "impact": 0.041908}
  ],
  "ann_confidence": 0.688102,
  "rf_confidence": 0.742381,
  "recommendation": "High readmission risk: schedule follow-up within 72 hours, reconcile medications, and trigger case-management outreach."
}
```

## 7) Notes

- SHAP explainability for `/predict/full` requires `shap==0.45.0` (included in backend requirements).
- If model artifact files are missing, prediction endpoints return actionable error messages instructing to train first.
- `model_report.md` contains the latest generated model comparison report.
