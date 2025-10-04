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
import json
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "models/")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
MODEL_TYPE = os.getenv("MODEL_TYPE", "xgboost")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

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

# Inference artifacts
mte_mappings: Dict[str, Any] = {}
feature_contract: Dict[str, Any] = {}
feature_rules: Dict[str, Any] = {}
feature_schema: Dict[str, Any] = {}

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
TARGET_COL = os.getenv("TARGET_COL", "is_canceled")


def load_model_and_scaler():
    """Load the trained model and scaler."""
    global model, scaler
    
    try:
        # Load scaler
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"✓ Scaler loaded from {SCALER_PATH}")
        else:
            print(f"⚠ Scaler not found at {SCALER_PATH}")
            scaler = None
        
        # Try to load model from MLflow
        try:
            # Set MLflow tracking URI
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
            mlflow.set_tracking_uri(mlflow_uri)
            
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


def load_inference_artifacts():
    """Load persisted feature engineering artifacts required for inference."""
    global mte_mappings, feature_contract, feature_rules, feature_schema
    try:
        mte_path = os.path.join(ARTIFACT_DIR, 'mte_mappings.json')
        if os.path.exists(mte_path):
            with open(mte_path) as f:
                mte_mappings = json.load(f)['encodings']
            print(f"✓ Loaded MTE mappings ({len(mte_mappings)})")
        else:
            print("⚠ mte_mappings.json not found; proceeding without target encodings")

        contract_path = os.path.join(ARTIFACT_DIR, 'feature_contract.json')
        if os.path.exists(contract_path):
            with open(contract_path) as f:
                feature_contract = json.load(f)
            print("✓ Loaded feature contract")
        else:
            print("⚠ feature_contract.json missing")

        rules_path = os.path.join(ARTIFACT_DIR, 'feature_rules.json')
        if os.path.exists(rules_path):
            with open(rules_path) as f:
                feature_rules = json.load(f)['rules']
            print("✓ Loaded feature rules")
        else:
            print("⚠ feature_rules.json missing")

        schema_path = os.path.join(ARTIFACT_DIR, 'feature_schema.json')
        if os.path.exists(schema_path):
            with open(schema_path) as f:
                feature_schema = json.load(f)['schema']
            print("✓ Loaded feature schema")
        else:
            print("⚠ feature_schema.json missing")
    except Exception as e:
        print(f"⚠ Failed loading artifacts: {e}")


def _apply_deterministic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate deterministic feature engineering for inference.

    Expects raw columns consistent with API schema + any categoricals used for encodings.
    """
    df = df.copy()
    # total_stay_duration
    if {'stays_weekend_nights','stays_week_nights'}.issubset(df.columns):
        df['total_stay_duration'] = df['stays_weekend_nights'] + df['stays_week_nights']

    # is_family (children field available, babies not in API schema -> assume 0)
    if 'children' in df.columns:
        babies = df.get('babies', 0)
        df['is_family'] = ((df['children'] > 0) | (babies if np.isscalar(babies) else babies > 0)).astype(int)
    else:
        df['is_family'] = 0

    # guest_type
    def _guest_type(row):
        babies_v = row.get('babies', 0)
        if babies_v > 0:
            return 'family_with_babies'
        if row.get('children', 0) > 0:
            return 'family_with_children'
        if row['adults'] == 1:
            return 'solo_traveler'
        if row['adults'] == 2:
            return 'couple'
        return 'group'
    df['guest_type'] = df.apply(_guest_type, axis=1)

    # arrival_season / is_peak_season / arrival_quarter / is_summer_peak / is_holiday_season
    if 'arrival_month' in df.columns:
        m = df['arrival_month']
        season_map = {12:'winter',1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',6:'summer',7:'summer',8:'summer',9:'autumn',10:'autumn',11:'autumn'}
        df['arrival_season'] = m.map(season_map)
        df['is_peak_season'] = m.isin([5,6,7,8,9]).astype(int)
        df['arrival_quarter'] = m.apply(lambda x: f"Q{((x-1)//3)+1}")
        df['is_summer_peak'] = m.isin([7,8]).astype(int)
        df['is_holiday_season'] = m.isin([12,1]).astype(int)
    else:
        # Fallback blanks
        for col in ['arrival_season','is_peak_season','arrival_quarter','is_summer_peak','is_holiday_season']:
            df[col] = np.nan

    return df


def _apply_mean_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Apply stored mean target encoding mappings to incoming records."""
    if not mte_mappings:
        return df
    df = df.copy()
    for base_col, meta in mte_mappings.items():
        enc_col = meta['encoded_column']
        mapping = meta.get('categories', {})
        global_mean = meta.get('global_mean')
        if base_col not in df.columns:
            # Missing categorical; fill with global mean
            df[enc_col] = global_mean
        else:
            df[enc_col] = df[base_col].map(mapping).fillna(global_mean)
    return df


def _build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Produce final ordered feature matrix aligned with training feature contract."""
    df = _apply_deterministic_features(df)
    df = _apply_mean_target_encoding(df)
    if feature_contract.get('feature_order'):
        ordered_cols = feature_contract['feature_order']
        # Ensure presence; missing columns get NaN (or could fill with 0)
        for col in ordered_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[ordered_cols]
    return df


@app.on_event("startup")
async def startup_event():
    """Load model and scaler on startup."""
    print("=" * 80)
    print("Starting Hotel Cancellation Prediction API")
    print("=" * 80)
    load_model_and_scaler()
    load_inference_artifacts()
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
        raw_df = pd.DataFrame([booking.dict()])
        feature_df = _build_feature_matrix(raw_df)
        input_scaled = scaler.transform(feature_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = (model.predict_proba(input_scaled)[0, 1]
                       if hasattr(model, 'predict_proba') else float(model.predict(input_scaled)[0]))

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
        raw_df = pd.DataFrame([booking.dict() for booking in bookings])
        feature_df = _build_feature_matrix(raw_df)
        input_scaled = scaler.transform(feature_df)

        # Make predictions
        predictions = model.predict(input_scaled)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_scaled)[:, 1]
        else:
            probabilities = predictions.astype(float)

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
