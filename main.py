"""
FastAPI application for hotel cancellation prediction.
Provides REST API endpoint for making predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import mlflow
import joblib
import os
from typing import List, Optional


# Initialize FastAPI app
app = FastAPI(
    title="Hotel Cancellation Prediction API",
    description="API for predicting hotel booking cancellations using ML models",
    version="1.0.0"
)


# Pydantic models for request/response
class BookingFeatures(BaseModel):
    """Input features for a hotel booking."""
    lead_time: int = Field(..., description="Number of days between booking and arrival", ge=0)
    arrival_month: int = Field(..., description="Month of arrival (1-12)", ge=1, le=12)
    stays_weekend_nights: int = Field(..., description="Number of weekend nights", ge=0)
    stays_week_nights: int = Field(..., description="Number of week nights", ge=0)
    adults: int = Field(..., description="Number of adults", ge=1)
    children: int = Field(..., description="Number of children", ge=0)
    is_repeated_guest: int = Field(..., description="Whether guest is repeated (0 or 1)", ge=0, le=1)
    previous_cancellations: int = Field(..., description="Number of previous cancellations", ge=0)
    booking_changes: int = Field(..., description="Number of booking changes", ge=0)
    adr: float = Field(..., description="Average Daily Rate", ge=0)
    required_car_parking_spaces: int = Field(..., description="Number of parking spaces required", ge=0)
    total_of_special_requests: int = Field(..., description="Number of special requests", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int = Field(..., description="Predicted class (0: not canceled, 1: canceled)")
    probability: float = Field(..., description="Probability of cancellation")
    model_used: str = Field(..., description="Model used for prediction")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool


# Global variables for model and scaler
model = None
scaler = None
model_name = "XGBoost"  # Default model to use


def load_model_and_scaler():
    """Load the trained model and scaler."""
    global model, scaler
    
    try:
        # Load scaler
        scaler_path = "models/scaler.pkl"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"✓ Scaler loaded from {scaler_path}")
        else:
            print(f"⚠ Scaler not found at {scaler_path}")
            scaler = None
        
        # Try to load model from MLflow
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri("file:./mlruns")
            
            # Try to load the latest XGBoost model
            # In production, you would specify a specific run_id or use model registry
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name("hotel_cancellation_prediction")
            
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="tags.mlflow.runName = 'XGBoost'",
                    order_by=["start_time DESC"],
                    max_results=1
                )
                
                if runs:
                    run_id = runs[0].info.run_id
                    model_uri = f"runs:/{run_id}/model"
                    model = mlflow.xgboost.load_model(model_uri)
                    print(f"✓ XGBoost model loaded from MLflow (run_id: {run_id})")
                else:
                    print("⚠ No XGBoost runs found in MLflow")
            else:
                print("⚠ Experiment 'hotel_cancellation_prediction' not found")
        except Exception as e:
            print(f"⚠ Could not load model from MLflow: {e}")
            model = None
    
    except Exception as e:
        print(f"✗ Error loading model/scaler: {e}")
        model = None
        scaler = None


@app.on_event("startup")
async def startup_event():
    """Load model and scaler on startup."""
    print("=" * 80)
    print("Starting Hotel Cancellation Prediction API")
    print("=" * 80)
    load_model_and_scaler()
    print("=" * 80)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Hotel Cancellation Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(booking: BookingFeatures):
    """
    Predict whether a hotel booking will be canceled.
    
    Returns:
        PredictionResponse with prediction, probability, and model used
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is trained and available."
        )
    
    if scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Scaler not loaded. Please ensure the scaler is available."
        )
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([booking.dict()])
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0, 1]
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model_used=model_name
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(bookings: List[BookingFeatures]):
    """
    Predict cancellations for multiple bookings.
    
    Returns:
        List of PredictionResponse objects
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is trained and available."
        )
    
    if scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Scaler not loaded. Please ensure the scaler is available."
        )
    
    try:
        # Convert inputs to DataFrame
        input_data = pd.DataFrame([booking.dict() for booking in bookings])
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make predictions
        predictions = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)[:, 1]
        
        # Create response list
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append(
                PredictionResponse(
                    prediction=int(pred),
                    probability=float(prob),
                    model_used=model_name
                )
            )
        
        return results
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making batch prediction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
