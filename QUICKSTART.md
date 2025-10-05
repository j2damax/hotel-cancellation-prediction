# Quick Start Guide

Get the Hotel Cancellation Prediction API running in minutes.

## Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Docker for containerized deployment

## Quick Start (5 minutes)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/j2damax/hotel-cancellation-prediction.git
cd hotel-cancellation-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train (with Optional Cross-Validation)

Basic training (fast):

```bash
python scripts/train.py --limit-rows 1000
```

Full cross-validation + champion selection:

```bash
python scripts/train.py --cv-folds 5 --categorical-strategy target
```

What happens:
- Trains LogisticRegression, RandomForest, XGBoost, PyTorch MLP
- (If --cv-folds) Runs stratified K-fold CV & selects champion (F1 primary, ROC-AUC tie-break)
- Persists `models/champion_model.pkl` and `artifacts/champion_meta.json`
- Generates diagnostics (confusion matrix, ROC/PR curve data, threshold sweep)
- Generates SHAP interpretability artifacts
- Logs all runs to MLflow

### 3. Start the API

```bash
# Start the FastAPI server
python main.py
```

The API will be available at:
- **API Endpoint**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 4. Test the API

In a new terminal:

```bash
# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the test client
python scripts/test_api.py
```

Or use curl:

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
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

Sample response (structure only):
```json
{
   "prediction": 0,
   "probability": 0.31,
   "model_used": "<ChampionModel>"
}
```

### 5. Explore with Interactive Docs

Open your browser and go to http://localhost:8000/docs

You'll see the interactive Swagger UI where you can:
- View all available endpoints
- Test endpoints directly in the browser
- See request/response schemas
- Try out predictions with custom data

## Using Docker (Alternative)

### With Docker Compose (Easiest)

```bash
# Start API and MLflow UI
docker-compose up

# API will be at: http://localhost:8000
# MLflow UI at: http://localhost:5000
```

### With Docker Only

```bash
# Build the image
docker build -t hotel-cancellation-prediction .

# Run the container
docker run -p 8000:8000 hotel-cancellation-prediction
```

### 5. Interpretability Endpoint

After training with champion persistence, access global + local SHAP metadata:

```bash
curl http://localhost:8000/model/interpretability | jq
```

Returns champion info, top features, local exemplar explanations, and feature name map.

## View MLflow Experiments

```bash
# Start MLflow UI
mlflow ui

# Open browser to: http://localhost:5000
```

In the MLflow UI you can:
- Compare model performance
- View training metrics
- Track experiments
- Download trained models

## Artifacts Overview

Key files produced in `artifacts/` after a full run:

| File | Purpose |
|------|---------|
| cv_metrics.json | Cross-validation metrics summary |
| champion_meta.json | Champion model + selection rationale |
| threshold_sweep.csv | Threshold vs precision/recall/F1 |
| confusion_matrix.png | Visualization of performance |
| roc_curve.json / pr_curve.json | Curve coordinate data |
| classification_report.json | Per-class metrics |
| shap_summary.png | Global SHAP beeswarm plot |
| shap_importance_bar.png | Ranked SHAP importance |
| feature_importance.json | Structured SHAP stats |
| shap_values_sample.json | Local explanation exemplars |
| feature_name_map.json | Human-readable feature labels |

Model + preprocessing:
| models/preprocessor.pkl | Shared preprocessing pipeline |
| models/champion_model.pkl | Persisted best model |

## What's Next?

- **Customize Models**: Edit `scripts/train.py` to adjust model parameters
- **Add Real Data**: Place your data in `data/raw/` and modify the data loading
- **Deploy to Hugging Face Space**: See `[DEPLOYMENT.md](DEPLOYMENT.md)` for the Space-only guide
- **API Documentation**: Visit http://localhost:8000/docs for full API reference
- **Model Monitoring**: Implement monitoring with MLflow in production

## Tests & CI

Run tests locally:

```bash
pytest -q
```

Optional live API tests (disabled by default in CI):

```bash
RUN_LIVE_API=1 pytest -q
```

GitHub Actions workflow runs the test suite (skips live tests) and can upload artifacts.

## Troubleshooting

### Model Not Loaded Error

```
Error: Model not loaded. Please ensure the model is trained and available.
```

**Solution**: Run `python scripts/train.py` first to train the models.

### Port Already in Use

```
Error: Address already in use
```

**Solution**: 
- Stop other services on port 8000, or
- Use a different port: `uvicorn main:app --port 8001`

### Import Errors

```
ModuleNotFoundError: No module named 'xxx'
```

**Solution**: Install dependencies with `pip install -r requirements.txt`

### Memory Issues with PyTorch

**Solution**: If training fails due to memory:
- Reduce batch size in `scripts/train.py`
- Reduce model hidden dimensions
- Use CPU-only PyTorch

## Getting Help

- Check [README.md](README.md) for detailed documentation
- Review [DEPLOYMENT.md](DEPLOYMENT.md) for Hugging Face Space deployment
- Open an issue on GitHub for bugs or questions

## Example Use Cases

### 1. Real-time Booking Analysis

```python
import requests

# When a customer makes a booking
booking = {
    "lead_time": 45,
    "arrival_month": 8,
    "stays_weekend_nights": 2,
    "stays_week_nights": 5,
    "adults": 2,
    "children": 0,
    "is_repeated_guest": 1,
    "previous_cancellations": 0,
    "booking_changes": 0,
    "adr": 120.00,
    "required_car_parking_spaces": 1,
    "total_of_special_requests": 1
}

response = requests.post("http://localhost:8000/predict", json=booking)
result = response.json()

if result["probability"] > 0.7:
    print("High cancellation risk! Consider sending confirmation email.")
else:
    print("Low risk booking.")
```

### 2. Batch Processing

```python
import requests
import pandas as pd

# Load bookings
bookings_df = pd.read_csv("bookings.csv")
bookings_list = bookings_df.to_dict('records')

# Get predictions for all
response = requests.post(
    "http://localhost:8000/predict/batch",
    json=bookings_list
)

predictions = response.json()

# Add to dataframe
bookings_df['cancellation_prediction'] = [p['prediction'] for p in predictions]
bookings_df['cancellation_probability'] = [p['probability'] for p in predictions]

bookings_df.to_csv("bookings_with_predictions.csv", index=False)
```

## Performance Tips

1. **Use batch endpoint** for multiple predictions (more efficient)
2. **Cache scaler and model** if making many predictions
3. **Monitor MLflow** for model performance over time
4. **Use Docker** for consistent deployment environments
5. **Enable GPU** for faster PyTorch training (if available)

---

## Readiness Quick Checklist

1. Data present in `data/raw/`
2. Full training run completed (no `--limit-rows`)
3. `models/champion_model.pkl` exists
4. All core artifacts present (see table above)
5. `/health` and `/model/interpretability` endpoints return 200
6. Tests pass (`pytest -q`)
7. MLflow UI metrics reviewed

Happy predicting! 🏨📊
