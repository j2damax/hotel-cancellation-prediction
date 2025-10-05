# Hotel Cancellation Prediction

A Data-Driven Framework for Predicting Hotel Booking Cancellations using Machine Learning

## Overview

This project implements a complete machine learning pipeline for predicting hotel booking cancellations. It includes multiple models (Logistic Regression, Random Forest, XGBoost, and PyTorch MLP), MLflow experiment tracking, and a FastAPI-based prediction service containerized with Docker for deployment to Amazon ECR.

## Project Structure

```
hotel-cancellation-prediction/
├── data/
│   ├── raw/              # Raw hotel booking datasets
│   ├── processed/        # Cleaned and preprocessed data
│   └── features/         # Feature-engineered datasets ready for modeling
├── notebooks/            # Jupyter notebooks for interactive analysis
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02_preprocessing_analysis.ipynb # Preprocessing strategy comparison
│   ├── 03_feature_engineering.ipynb   # Feature engineering experiments
│   └── 04_model_evaluation.ipynb      # Model evaluation and SHAP analysis
├── scripts/              # Production-ready Python scripts
│   ├── train.py                # Training script for all models
│   ├── preprocessing.py        # Data preprocessing pipeline
│   ├── feature_engineering.py  # Feature engineering pipeline
│   ├── model_evaluation.py     # Model evaluation and comparison
│   └── test_api.py            # API testing client
├── src/                  # Core source code modules
├── models/              # Saved models and preprocessing artifacts
├── results/             # Evaluation results and reports
├── mlruns/              # MLflow experiment tracking data
├── .github/             # GitHub workflows and AI agent instructions
│   └── copilot-instructions.md # Comprehensive AI agent guidance
├── main.py              # FastAPI application
├── Dockerfile           # Docker container configuration
├── docker-compose.yml   # Docker Compose for local deployment
├── requirements.txt     # Python dependencies (enhanced for academic research)
├── DEPLOYMENT.md        # AWS ECR deployment guide
├── QUICKSTART.md        # Quick start guide
├── EDA.md              # Comprehensive EDA methodology (1,624 lines)
├── preprocessing.md     # Preprocessing strategies guide (1,445 lines)
├── features.md         # Feature engineering guide (1,653 lines)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Features

### Hybrid Architecture: Notebooks + Production Scripts

This project implements a **hybrid approach** combining interactive analysis with production-ready pipelines:

**📓 Jupyter Notebooks** (for research and analysis):
- `01_eda.ipynb` - Interactive exploratory data analysis with visualizations
- `02_preprocessing_analysis.ipynb` - Preprocessing strategy experimentation
- `03_feature_engineering.ipynb` - Feature engineering with effectiveness testing
- `04_model_evaluation.ipynb` - Model comparison with SHAP interpretability

**🐍 Python Scripts** (for production deployment):
- `preprocessing.py` - Automated data preprocessing pipeline
- `feature_engineering.py` - Production feature engineering with cross-validation
- `model_evaluation.py` - Comprehensive model evaluation with statistical testing
- `train.py` - Complete training pipeline for deployment

### Academic Research Framework

- **NIB 7072 Coursework Compliance**: Rigorous academic standards with statistical significance testing
- **Sri Lankan Tourism Focus**: Domain-specific features and business impact analysis  
- **5-Fold Cross-Validation**: Stratified sampling with performance confidence intervals
- **SHAP Interpretability**: Model explainability for actionable business insights
- **Comprehensive Documentation**: 4,700+ lines of methodology documentation

### Machine Learning Models

- **Logistic Regression**: Baseline linear model with L1/L2 regularization
- **Random Forest**: Ensemble tree-based model with optimized hyperparameters
- **XGBoost**: Gradient boosting model (historically strong performer in internal experiments)
- **PyTorch MLP**: Deep learning neural network with dropout and batch normalization

> Champion model is now selected dynamically per training run using cross-validation (F1 primary, ROC-AUC tie-break). Any previously hard-coded champion claims (e.g., a fixed XGBoost score) should be treated as historical examples only.

### MLflow Integration

- Experiment tracking for all models with Optuna hyperparameter optimization
- Automatic logging of parameters, metrics, and model artifacts
- Model comparison and versioning with statistical significance testing
- Easy model registry integration for production deployment

### FastAPI REST API

- `/predict` - Single prediction endpoint with Pydantic validation
- `/predict/batch` - Batch prediction endpoint for bulk processing
- `/health` - Health check endpoint with model availability verification
- `/model/interpretability` - Serve SHAP global + local explanation metadata (top features, exemplar cases)
- Interactive API documentation at `/docs` with schema validation

### Docker Containerization

- Optimized Docker image for production deployment
- Health checks included
- Ready for Amazon ECR deployment

## Installation

### Prerequisites

- Python 3.10+
- Docker (optional, for containerization)

### Local Setup

1. Clone the repository:

```bash
git clone https://github.com/j2damax/hotel-cancellation-prediction.git
cd hotel-cancellation-prediction
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Configure environment (optional):

```bash
# Copy environment template
cp .env.example .env
# Edit .env with your preferred settings (optional - defaults work fine)
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training Models

Run the training script to train all models with MLflow tracking:

```bash
python scripts/train.py
```

This will:

- Load the hotel booking dataset (place real data in `data/raw/` if replacing sample)
- Train 4 different models (LogReg, RF, XGBoost, PyTorch MLP)
- Perform optional stratified cross-validation (if `--cv-folds` provided)
- Select and persist a champion model (`models/champion_model.pkl`) with metadata
- Generate diagnostic + interpretability artifacts (see Artifacts section below)
- Log all experiments and runs to MLflow

View MLflow UI to compare models:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### Makefile Convenience

Common tasks (see `Makefile`):

```
make train                # Full CV training
make fast-train ROWS=800  # Smoke test limited rows
make api                  # Run API locally
make mlflow               # Launch MLflow UI
make export-latex         # (If added) run LaTeX export script
make artifacts-status     # List artifact files
make docker-build         # Build Docker image (IMAGE_TAG=git SHA)
make docker-push REGISTRY=<acct>.dkr.ecr.<region>.amazonaws.com
make docker-release REGISTRY=...  # Build + push (sha + latest)
```

### Fairness & LaTeX Utilities

- `scripts/fairness_analysis.py` – exploratory subgroup metrics
- `scripts/export_latex.py` – generate LaTeX tables into `report/latex/`

### Running the API

Start the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or run directly:

```bash
python main.py
```

Access the API:

- API Root: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Testing the API

Use the provided test client:

```bash
python scripts/test_api.py
```

Or make manual requests (see examples below).

### Making Predictions

Example using curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "lead_time": 120,
    "arrival_month": 7,
    "stays_weekend_nights": 2,
    "stays_week_nights": 3,
    "adults": 2,
    "children": 1,
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "booking_changes": 1,
    "adr": 95.50,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 2
  }'
```

Example using Python:

```python
import requests

data = {
    "lead_time": 120,
    "arrival_month": 7,
    "stays_weekend_nights": 2,
    "stays_week_nights": 3,
    "adults": 2,
    "children": 1,
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "booking_changes": 1,
    "adr": 95.50,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 2
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## Docker Deployment

### Using Docker Compose (Recommended for Local Testing)

The easiest way to run the application locally with Docker:

```bash
# Start both API and MLflow UI
docker-compose up

# Or run in detached mode
docker-compose up -d
```

This will start:

- API server on http://localhost:8000
- MLflow UI on http://localhost:5000

To stop:

```bash
docker-compose down
```

### Building the Docker Image

```bash
docker build -t hotel-cancellation-prediction .
```

### Running the Container Locally

```bash
docker run -p 8000:8000 hotel-cancellation-prediction
```

### Deploying to Amazon ECR

For detailed instructions on deploying to AWS ECR and running on ECS, EKS, or App Runner, see [DEPLOYMENT.md](DEPLOYMENT.md).

Quick start:

1. Authenticate Docker to ECR:

```bash
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
```

2. Create ECR repository (if not exists):

```bash
aws ecr create-repository --repository-name hotel-cancellation-prediction --region <region>
```

3. Tag the image:

```bash
docker tag hotel-cancellation-prediction:latest <account-id>.dkr.ecr.<region>.amazonaws.com/hotel-cancellation-prediction:latest
```

4. Push to ECR:

```bash
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/hotel-cancellation-prediction:latest
```

5. Deploy to ECS, EKS, or other AWS services using the ECR image.

## Input Features

The model expects the following features for prediction:

| Feature                     | Type  | Description                      | Range  |
| --------------------------- | ----- | -------------------------------- | ------ |
| lead_time                   | int   | Days between booking and arrival | ≥ 0    |
| arrival_month               | int   | Month of arrival                 | 1-12   |
| stays_weekend_nights        | int   | Number of weekend nights         | ≥ 0    |
| stays_week_nights           | int   | Number of week nights            | ≥ 0    |
| adults                      | int   | Number of adults                 | ≥ 1    |
| children                    | int   | Number of children               | ≥ 0    |
| is_repeated_guest           | int   | Repeated guest flag              | 0 or 1 |
| previous_cancellations      | int   | Previous cancellations count     | ≥ 0    |
| booking_changes             | int   | Number of booking changes        | ≥ 0    |
| adr                         | float | Average Daily Rate               | ≥ 0    |
| required_car_parking_spaces | int   | Parking spaces required          | ≥ 0    |
| total_of_special_requests   | int   | Number of special requests       | ≥ 0    |

## Model Performance

After training, you can compare model performance in the MLflow UI. Metrics tracked include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

### Current Champion (Most Recent Full Run)

The latest full-data training run (timestamp 2025-10-05) selected **XGBoost** as champion via 5-fold stratified CV (primary metric F1, tie-break ROC-AUC).

Cross-Validation (mean ± std):

- Accuracy: 0.8612 ± 0.0016
- Precision: 0.8386 ± 0.0024
- Recall: 0.7743 ± 0.0051
- F1: 0.8052 ± 0.0028
- ROC-AUC: 0.9376 ± 0.0010

Holdout Metrics:

- Accuracy: 0.8614
- Precision: 0.8417
- Recall: 0.7707
- F1: 0.8047
- ROC-AUC: 0.9384

Optimal Operating Threshold (F1-oriented): 0.35 → Precision 0.7664 / Recall 0.8620 / F1 0.8114

Artifacts: see `artifacts/champion_meta.json` and `artifacts/cv_metrics.json`.

> NOTE: Champion is re-evaluated each training run; values above are point-in-time and will update after subsequent executions.

## Cross-Validation & Champion Selection

The training pipeline performs optional stratified K-fold cross-validation to robustly compare candidate models and automatically select a champion.

### Running Cross-Validation

```bash
python scripts/train.py --cv-folds 5
```

### Champion Selection Criteria

1. Primary: Highest `f1_score_mean`
2. Tie-break: Highest `roc_auc_mean`
3. Reported with mean ± std across folds

### Key Artifacts

- `artifacts/cv_metrics.json` – Per-fold + aggregate metrics (F1, ROC-AUC, precision, recall)
- `artifacts/champion_meta.json` – Champion model name, metrics, selection rationale
- `models/champion_model.pkl` – Persisted champion model

### Example Metric Table (Illustrative Only)

| Model | F1 (mean ± std) | ROC-AUC (mean ± std) | Precision | Recall |
|-------|-----------------|----------------------|-----------|--------|
| LogisticRegression | 0.xxx ± 0.xxx | 0.xxx ± 0.xxx | 0.xxx | 0.xxx |
| RandomForest | 0.xxx ± 0.xxx | 0.xxx ± 0.xxx | 0.xxx | 0.xxx |
| XGBoost | 0.xxx ± 0.xxx | 0.xxx ± 0.xxx | 0.xxx | 0.xxx |
| PyTorch_MLP | 0.xxx ± 0.xxx | 0.xxx ± 0.xxx | 0.xxx | 0.xxx |

> Replace with values from your full-data run; above numbers are placeholders.

## Interpretability & SHAP

Model transparency is critical for both academic rigor and operational trust in the hospitality domain. We integrate SHAP (SHapley Additive exPlanations) to provide: (1) global feature importance, (2) per-booking local explanations, and (3) a human-readable feature name mapping.

### Generated Interpretability Artifacts

Produced automatically when a champion is finalized:

- `artifacts/shap_summary.png` – Beeswarm (global impact distribution)
- `artifacts/shap_importance_bar.png` – Top mean |SHAP| importance bar chart
- `artifacts/feature_importance.json` – Mean absolute SHAP values (machine names)
- `artifacts/shap_values_sample.json` – Sampled local explanations (true/false positive/negative exemplars)
- `artifacts/feature_name_map.json` – Mapping to human-readable labels
- `artifacts/threshold_sweep.csv` – Threshold vs. precision/recall/F1
- `artifacts/classification_report.json` – Precision/recall/F1 by class
- `artifacts/roc_curve.json`, `artifacts/pr_curve.json` – Curve coordinate data
- `artifacts/confusion_matrix.png` – Visual confusion matrix

### Top 10 Features (Sample Run)

Example (LogisticRegression champion on a limited 800-row run):

| Rank | Feature | Mean |SHAP| | Human Meaning |
|------|---------|-------------|----------------|
| 1 | country__te | 2.448 | Country (target encoded) |
| 2 | assigned_room_type | 1.229 | Assigned room type code |
| 3 | required_car_parking_spaces | 1.074 | Required car parking spaces |
| 4 | reserved_room_type | 0.943 | Reserved room type code |
| 5 | customer_type_target_encoded | 0.608 | Customer type (target encoded) |
| 6 | distribution_channel_target_encoded | 0.458 | Distribution channel (target encoded) |
| 7 | arrival_date_week_number | 0.416 | Week-of-year of arrival |
| 8 | booking_changes | 0.384 | Number of booking modifications |
| 9 | market_segment | 0.343 | Market segment raw category |
| 10 | lead_time | 0.253 | Days between booking & arrival |

> Values above are illustrative from a small sample run. For publication-quality reporting re-run on full dataset; SHAP magnitude ordering may shift slightly with more data and the final champion.

### Local Explanation Examples

From `shap_values_sample.json` (categories chosen: true_positive, false_positive, false_negative) – truncated illustration:

```json
{
  "category": "true_positive",
  "probability": 0.8560,
  "top_positive_contributors": [
    {"feature": "country__te", "shap": 2.1173},
    {"feature": "assigned_room_type", "shap": 1.5004},
    {"feature": "required_car_parking_spaces", "shap": 0.8191}
  ],
  "top_negative_contributors": [
    {"feature": "reserved_room_type", "shap": -0.7972},
    {"feature": "stays_in_weekend_nights", "shap": -0.4383},
    {"feature": "arrival_date_day_of_month", "shap": -0.1705}
  ]
}
```

Interpretation (business context): The model increased cancellation probability primarily due to (a) encoded country signal, (b) an assigned room type differing from reservation (upgrade/downgrade friction), and (c) required parking (proxy for certain traveler segments). Weekend stays and specific reserved room characteristics slightly mitigated predicted risk.

### Why SHAP?

- Additive, locally accurate decomposition of prediction log-odds (for linear / tree models)
- Consistent feature importance across heterogeneous model classes
- Actionability: Revenue management and overbooking policies can target high-impact drivers (e.g., long lead time + specific channel + encoded country cluster)

### Recomputing for Final Report

Run on the full dataset (omit `--limit-rows`) to produce stable global rankings:

```bash
python scripts/train.py --cv-folds 5 --categorical-strategy target
```

Use `feature_importance.json` + `champion_meta.json` for publication tables. Optionally convert to LaTeX.

### Notes & Future Enhancements

- Calibrated probabilities (Platt / isotonic) for better risk thresholds
- Drift monitoring: compare future SHAP distributions vs. baseline to detect market shifts
- Grouped importance (aggregate one-hot / target-encoded families) for cleaner reporting
The `/model/interpretability` endpoint provides: champion metadata, top global features, local explanation exemplars, and feature name mapping.

---

## Environment Configuration

The application supports environment-based configuration through `.env` files:

### Quick Setup

```bash
# Copy the template
cp .env.example .env

# Edit with your settings (optional - defaults work fine)
# Common customizations:
# API_PORT=8001               # Change API port
# MLFLOW_UI_PORT=5002         # Change MLflow UI port
# LOG_LEVEL=DEBUG             # Enable debug logging
# TRAINING_DATA_SIZE=20000    # Larger training dataset
```

### Key Environment Variables

- `API_PORT`: FastAPI server port (default: 8000)
- `MLFLOW_UI_PORT`: MLflow UI port (default: 5001)
- `MLFLOW_TRACKING_URI`: MLflow backend URI (default: file:./mlruns)
- `MODEL_PATH`: Directory for saved models (default: models/)
- `LOG_LEVEL`: Logging level (default: INFO)

For Hugging Face deployment specifics see `DEPLOYMENT.md`. For a minimal AWS (S3 + ECR/ECS) reference see `DEPLOYMENT_AWS.md` which documents bucket/repo naming, IAM actions, and runtime model fetch environment variables.

### Production Configuration

For production deployments, set secure values in `.env`:

```bash
# Security
API_KEY=your-secret-api-key
JWT_SECRET=your-jwt-secret

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012
ECR_REPOSITORY_NAME=hotel-cancellation-prediction

# Performance
MAX_WORKERS=8
BATCH_SIZE=200
```

## Dependencies

See `requirements.txt` for complete list. Key dependencies:

- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- torch >= 2.0.0
- mlflow >= 2.9.0
- fastapi >= 0.104.0
- uvicorn >= 0.24.0
- python-dotenv >= 1.0.0

## License

MIT License - see LICENSE file for details

## Author

Jayampathy Balasuriya

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Training CLI Options

The `scripts/train.py` script supports several optional arguments (inspect `--help` for the authoritative list):

- `--cv-folds INT` – Enable stratified K-fold cross-validation
- `--limit-rows INT` – Use only the first N rows (smoke tests / fast iteration)
- `--categorical-strategy {onehot,target,drop}` – Encoding strategy
- `--max-shap-samples INT` – (If implemented) Cap rows used for SHAP to control runtime

## Produced Artifacts (Summary)

| Artifact | Purpose |
|----------|---------|
| `artifacts/cv_metrics.json` | Cross-validation metrics per model + aggregates |
| `artifacts/champion_meta.json` | Champion model identity + selection rationale |
| `models/champion_model.pkl` | Persisted champion model for inference |
| `artifacts/confusion_matrix.png` | Visual performance diagnostic |
| `artifacts/roc_curve.json` / `pr_curve.json` | Curve data for reproducible plots |
| `artifacts/threshold_sweep.csv` | Threshold tuning metrics grid |
| `artifacts/classification_report.json` | Precision/Recall/F1 per class |
| `artifacts/shap_summary.png` | Global SHAP beeswarm plot |
| `artifacts/shap_importance_bar.png` | Ranked SHAP feature importances |
| `artifacts/feature_importance.json` | Structured global SHAP stats |
| `artifacts/shap_values_sample.json` | Local explanation exemplars |
| `artifacts/feature_name_map.json` | Human-readable labels for features |

## Readiness Checklist

Use this before producing final academic or deployment results:

1. Data present in `data/raw/` (expected rows & target distribution validated)
2. Run full training (no `--limit-rows`):
  ```bash
  python scripts/train.py --cv-folds 5 --categorical-strategy target
  ```
3. Confirm artifacts directory contains all files listed above
4. Inspect `champion_meta.json` – champion identity & metrics reasonable
5. Optional: Tune decision threshold using `threshold_sweep.csv`
6. Start API & verify endpoints:
  ```bash
  uvicorn main:app --port 8000
  curl localhost:8000/health
  curl localhost:8000/model/interpretability
  ```
7. Run tests:
  ```bash
  pytest -q
  ```
8. Launch MLflow UI and capture comparative metrics screenshot (for report)
9. Archive artifact visuals (confusion matrix, SHAP plots) for appendix
10. (Deployment) Build & push Docker image or run `docker-compose up` locally

## Future Enhancements

- Probability calibration (Platt / isotonic) for improved decision thresholds
- Drift monitoring via periodic SHAP distribution comparison
- Grouped feature attribution (aggregate encoded categories)
- Automated LaTeX export of metrics & importance tables
- Optional calibration & fairness diagnostics modules

---
Updated: 2025-10-05
