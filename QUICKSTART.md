# Quick Start Guide

Get the Hotel Cancellation Prediction API running quickly.

## Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Docker

## Installation

```bash
# Clone repository
git clone https://github.com/j2damax/hotel-cancellation-prediction.git
cd hotel-cancellation-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Train Models

**Quick training (smoke test):**

```bash
python scripts/train.py --limit-rows 1000
```

**Full training with cross-validation:**

```bash
python scripts/train.py --cv-folds 5
```

This will:
- Train all models (LogisticRegression, RandomForest, XGBoost, PyTorch MLP)
- Select champion model based on F1 score
- Generate diagnostic artifacts and SHAP plots
- Save model to `models/champion_model.pkl`

## Start API

```bash
python main.py
```

Access:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Test API

```bash
# Using test script
python scripts/test_api.py

# Using curl
curl http://localhost:8000/health

# Make prediction
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

## Using Docker

**With Docker Compose:**

```bash
docker-compose up  # API on port 8000, MLflow on port 5000
```

**With Docker:**

```bash
docker build -t hotel-cancellation-prediction .
docker run -p 8000:8000 hotel-cancellation-prediction
```

## View MLflow Experiments

```bash
mlflow ui  # Open http://localhost:5000
```

## Makefile Commands

```bash
make train              # Full CV training
make fast-train ROWS=800  # Quick smoke test
make api                # Start API server
make mlflow             # Launch MLflow UI
make test               # Run tests
make docker-build       # Build Docker image
```

## Next Steps

- **Deploy**: See [DEPLOYMENT.md](DEPLOYMENT.md) for Hugging Face Space deployment
- **Documentation**: See [README.md](README.md) for detailed information
- **API Reference**: Visit http://localhost:8000/docs for interactive documentation

## Troubleshooting

**Model Not Loaded:**
- Run training first: `python scripts/train.py`

**Port Already in Use:**
- Use different port: `uvicorn main:app --port 8001`

**Import Errors:**
- Install dependencies: `pip install -r requirements.txt`
