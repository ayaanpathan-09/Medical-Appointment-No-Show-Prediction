# Medical Appointment No-Show Prediction System

A machine-learning system that scores medical appointments with a no-show
probability so clinic staff can prioritize outreach for patients most likely
to miss their appointment.

## Project status

This version is configured for the **real public Medical Appointment No Shows
dataset**, not only the 25-row test sample. The original Kaggle dataset contains
**110,527 appointments and 14 variables**.

The 25-row file in `tests/test_data/sample_appointments.csv` remains only for
fast automated tests. Do **not** use its metrics in your final report.

## Architecture

```text
Real Kaggle dataset
       ↓
Data ingestion + validation
       ↓
Preprocessing / cleaning
       ↓
Feature engineering
       ↓
Model comparison
(Logistic Regression / Decision Tree /
 Random Forest / XGBoost / SVM)
       ↓
Model registry
       ↓
Risk scoring
       ↓
FastAPI + Streamlit dashboard
```

## 1. Setup (Windows / VS Code)

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```

If `python` is not recognized, use `py` instead.

## 2. Download the real dataset

The source dataset is the public Kaggle **Medical Appointment No Shows**
dataset. It contains 110,527 appointments and 14 variables.

Run:

```powershell
python scripts\download_real_data.py
```

The file will be saved as:

```text
data/raw/KaggleV2-May-2016.csv
```

The dataset is licensed **CC BY-NC-SA 4.0**, so keep the dataset's attribution
and license conditions in mind when publishing or redistributing the project.

## 3. Run the complete real-data pipeline

After downloading the dataset:

```powershell
python scripts\run_real_pipeline.py
```

This performs:

1. Real-data ingestion and validation
2. Cleaning and date normalization
3. Feature engineering
4. Model comparison
5. Best-model registration and promotion
6. Dashboard-ready prediction generation

Generated files include:

```text
data/processed/ingested.csv
data/processed/ingestion_audit.json
data/processed/cleaned.csv
data/processed/features.csv
data/processed/model_summary.json
data/processed/predictions.csv
models_registry/<version>/model.pkl
models_registry/<version>/metadata.json
models_registry/production.json
```

## 4. Important validation fix for the Kaggle data

The public dataset stores `ScheduledDay` with a timestamp while
`AppointmentDay` is recorded at midnight. The ingestion validator therefore
compares their **calendar dates**, preventing valid same-day appointments from
being incorrectly rejected.

The ingestion layer also maps the public dataset's fields such as
`PatientId`, `AppointmentID`, `Gender`, `ScheduledDay`, `AppointmentDay`,
`SMS_received`, and `No-show` to the project's canonical schema.

## 5. Run the API

```powershell
uvicorn src.api.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

The API uses the trained production model from `models_registry/`.

## 6. Run the Streamlit dashboard

```powershell
streamlit run src/dashboard/app.py
```

The dashboard reads:

```text
data/processed/predictions.csv
```

and provides the risk-ranked appointment list and analytics.

## 7. Tests

The bundled 25-row sample remains available for automated tests:

```powershell
pytest tests/unit -v
pytest tests/integration -v
```

## 8. Dataset source

Original dataset:

- Kaggle: Medical Appointment No Shows
- Author/reference: Joni Hoppen / Aquarela Analytics
- Size: 110,527 appointments
- Variables: 14
- License: CC BY-NC-SA 4.0

For the final college report, cite the original Kaggle dataset rather than the
GitHub mirror used by the convenience download script.

## Project structure

```text
medical-noshow-prediction/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── scripts/
│   ├── download_real_data.py
│   └── run_real_pipeline.py
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── api/
│   ├── dashboard/
│   └── utils/
├── models_registry/
├── tests/
├── config/
├── deployment/
├── docs/
├── requirements.txt
├── .env.example
└── README.md
```

## Final-project note

The real dataset is intentionally **not committed to Git**. The project's
`.gitignore` excludes `data/raw/`, `data/processed/`, and trained model
artifacts so that large datasets and generated files are not accidentally
pushed to GitHub. Run the download and pipeline commands locally before your
final demo.
