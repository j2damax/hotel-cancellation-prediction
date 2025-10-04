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

### 2. Train Models

```bash
# Train all models (LogReg, RF, XGBoost, PyTorch MLP)
python scripts/train.py

# This will:
# - Generate sample hotel booking data
# - Train 4 different models
# - Save models with MLflow tracking
# - Create models/preprocessor.pkl for scaling & categorical encoding
```

Expected output:
```
================================================================================
Hotel Cancellation Prediction - Model Training
================================================================================

1. Loading data...
   Data shape: (10000, 13)
   Cancellation rate: 31.16%

2. Scaling features...
   Preprocessor saved to models/preprocessor.pkl

3. Training models...
--------------------------------------------------------------------------------

   Training Logistic Regression...
LogisticRegression - Accuracy: 0.6770, F1: 0.1387

   Training Random Forest...
RandomForest - Accuracy: 0.6815, F1: 0.1651

   Training XGBoost...
XGBoost - Accuracy: 0.6740, F1: 0.1850

   Training PyTorch MLP...
PyTorch_MLP - Accuracy: 0.6790, F1: 0.1662

================================================================================
Training completed! Check MLflow UI with: mlflow ui
================================================================================
```

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

Expected response:
```json
{
  "prediction": 0,
  "probability": 0.3195936381816864,
  "model_used": "XGBoost"
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

## What's Next?

- **Customize Models**: Edit `scripts/train.py` to adjust model parameters
- **Add Real Data**: Place your data in `data/raw/` and modify the data loading
- **Deploy to AWS**: Follow the [DEPLOYMENT.md](DEPLOYMENT.md) guide
- **API Documentation**: Visit http://localhost:8000/docs for full API reference
- **Model Monitoring**: Implement monitoring with MLflow in production

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
- Review [DEPLOYMENT.md](DEPLOYMENT.md) for AWS deployment
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

Happy predicting! 🏨📊
