# Hotel Cancellation Prediction

A machine learning pipeline for predicting hotel booking cancellations with multiple models, MLflow tracking, and FastAPI REST API.

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
│   ├── push_to_hf.py           # Upload model to Hugging Face Hub
│   ├── deploy_to_hf_space.py   # Deploy app to Hugging Face Spaces
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
├── HUGGINGFACE_DEPLOYMENT.md # Hugging Face Space deployment guide  
├── DEPLOYMENT.md        # Legacy deployment documentation
├── QUICKSTART.md        # Quick start guide
├── EDA.md              # Comprehensive EDA methodology (1,624 lines)
├── preprocessing.md     # Preprocessing strategies guide (1,445 lines)
├── features.md         # Feature engineering guide (1,653 lines)
├── .gitignore          # Git ignore rules
└── README.md           # This file
├── data/                 # Raw, processed, and feature-engineered datasets
├── notebooks/            # Jupyter notebooks for analysis
├── scripts/              # Production Python scripts
│   ├── train.py          # Model training with cross-validation
│   ├── export_latex.py   # LaTeX table generation
│   └── test_api.py       # API testing
├── app/                  # FastAPI application modules
├── models/               # Saved models and artifacts
├── artifacts/            # Training artifacts and metrics
├── mlruns/               # MLflow experiment tracking
├── main.py               # FastAPI entry point
├── Dockerfile            # Container configuration
└── requirements.txt      # Python dependencies
```

## Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, PyTorch MLP
- **Cross-Validation**: Stratified K-fold with automatic champion selection (F1 score primary metric)
- **MLflow Tracking**: Experiment logging and model versioning
- **REST API**: FastAPI with `/predict`, `/predict/batch`, `/health`, and `/model/interpretability` endpoints
- **SHAP Interpretability**: Model explainability with global and local feature importance
- **Docker Support**: Containerized deployment ready
- **Hugging Face Integration**: Deploy as a Space with model artifact loading from Hub

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

```bash
# Clone and setup
git clone https://github.com/j2damax/hotel-cancellation-prediction.git
cd hotel-cancellation-prediction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train models
python scripts/train.py --cv-folds 5

# Start API
python main.py

# Test
curl http://localhost:8000/health
```

## Training

Train all models with cross-validation:

```bash
python scripts/train.py --cv-folds 5 --categorical-strategy target
```

Or use the Makefile:

```bash
make train              # Full CV training
make fast-train ROWS=800  # Quick smoke test
make mlflow             # Launch MLflow UI
```

The training pipeline will:
- Train multiple models (LogisticRegression, RandomForest, XGBoost, PyTorch MLP)
- Perform stratified K-fold cross-validation
- Select champion model based on F1 score
- Generate diagnostic artifacts (confusion matrix, ROC curves, SHAP plots)
- Save champion model to `models/champion_model.pkl`

View experiments in MLflow UI:

```bash
mlflow ui  # Open http://localhost:5000
```

## API Usage

Start the FastAPI server:

```bash
python main.py  # or: uvicorn main:app --host 0.0.0.0 --port 8000
```

Available endpoints:
- **Root**: http://localhost:8000
- **Docs**: http://localhost:8000/docs (interactive Swagger UI)
- **Health**: http://localhost:8000/health

### Make Predictions

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

See [DEPLOYMENT.md](DEPLOYMENT.md) for Hugging Face Space deployment.

**Using Docker Compose:**

```bash
docker-compose up  # Start API (port 8000) and MLflow UI (port 5000)
```

**Using Docker:**

```bash
docker build -t hotel-cancellation-prediction .
docker run -p 8000:8000 hotel-cancellation-prediction
```

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

Champion model is selected automatically via cross-validation (F1 score primary metric, ROC-AUC tie-break).

View metrics in:
- `artifacts/champion_meta.json` - Champion model metadata and performance
- `artifacts/cv_metrics.json` - Cross-validation results
- MLflow UI - Experiment tracking and comparison

## Artifacts

Training generates the following artifacts in `artifacts/`:

| File | Description |
|------|-------------|
| `cv_metrics.json` | Cross-validation metrics |
| `champion_meta.json` | Champion model metadata |
| `confusion_matrix.png` | Performance visualization |
| `roc_curve.json`, `pr_curve.json` | Curve data |
| `threshold_sweep.csv` | Threshold tuning metrics |
| `shap_summary.png` | Global SHAP plot |
| `feature_importance.json` | Feature importance scores |

## Interpretability

Access model interpretability via the `/model/interpretability` endpoint or generated artifacts:

```bash
curl http://localhost:8000/model/interpretability
```

Returns:
- Champion model information
- Top global features (SHAP importance)
- Local explanation examples
- Feature name mappings

## Makefile Commands

```bash
make help             # Show all available commands
make train            # Full CV training
make fast-train ROWS=800  # Quick smoke test
make api              # Start API server
make mlflow           # Launch MLflow UI
make test             # Run pytest
make docker-build     # Build Docker image
make docker-run       # Run container locally
```

## License

MIT License - see LICENSE file for details.

## Author

Jayampathy Balasuriya

## Contributing

Contributions are welcome! Please submit a Pull Request.
