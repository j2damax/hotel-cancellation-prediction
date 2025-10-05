"""
FastAPI application for hotel cancellation prediction.
Provides REST API endpoint for making predictions.
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import numpy as np
import mlflow
import joblib
import os
import time
import threading
import json
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "models/")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")
MODEL_TYPE = os.getenv("MODEL_TYPE", "xgboost")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# Remote artifact configuration (future deployment mode)
MODEL_S3_URI = os.getenv("MODEL_S3_URI")  # e.g. s3://hotel-cancel-models-prod-<id>/models
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
DECISION_THRESHOLD = os.getenv("DECISION_THRESHOLD")  # optional classification threshold (may be updated after load)
AWS_REGION = os.getenv("AWS_REGION")
ALLOW_START_WITHOUT_MODEL = os.getenv("ALLOW_START_WITHOUT_MODEL", "false").lower() == "true"

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 80)
    print("Starting Hotel Cancellation Prediction API (lifespan init)")
    print("=" * 80)
    load_model_and_scaler()
    load_inference_artifacts()
    if model is None and not ALLOW_START_WITHOUT_MODEL:
        print("✗ No model loaded during startup and ALLOW_START_WITHOUT_MODEL is false -> failing fast.")
        print("  To allow the API to start without a model (for debugging), set ALLOW_START_WITHOUT_MODEL=true.")
        raise RuntimeError("Model not loaded at startup. Ensure volumes are mounted or S3 env vars are set.")
    print("Initialization complete.")
    print("=" * 80)
    yield
    # Optional teardown logic here (e.g., close DB connections)
    print("Shutting down API - resources released.")

app = FastAPI(
    title="Hotel Cancellation Prediction API",
    description="API for predicting hotel booking cancellations using ML models",
    version="1.0.0",
    lifespan=lifespan
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
    
    model_config = ConfigDict(
        json_schema_extra={
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
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int = Field(..., description="Predicted class (0: not canceled, 1: canceled)")
    probability: float = Field(..., description="Probability of cancellation")
    model_used: str = Field(..., description="Model used for prediction")
    applied_threshold: float | None = Field(None, description="Threshold used to convert probability -> class")
    threshold_source: Optional[str] = Field(None, description="Origin of threshold: env|champion_meta|default")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    decision_threshold: Optional[float] = None


class LocalExplanation(BaseModel):
    category: str
    probability: Optional[float] = None
    top_positive_contributors: List[Dict[str, float]]
    top_negative_contributors: List[Dict[str, float]]


class InterpretabilityResponse(BaseModel):
    champion_model: Optional[str]
    shap_generated: bool
    shap_timestamp: Optional[str]
    decision_threshold: Optional[float]
    top_features: List[Dict[str, Any]]
    local_examples: List[LocalExplanation]
    feature_name_map: Dict[str, str]
    artifacts_available: List[str]


from src.preprocessing import PreprocessingPipeline

# Global variables for model and preprocessing pipeline
model = None
preprocessor: Optional[PreprocessingPipeline] = None
model_name = "XGBoost"  # Default model to use
model_version: Optional[str] = None  # populated when remote S3 fetch implemented
champion_meta_threshold: Optional[float] = None  # champion decision threshold (if provided)

# --- Runtime Metrics / Instrumentation ---
APP_START_TIME = time.time()
_metrics_lock = threading.Lock()
_prediction_request_count = 0
_prediction_total_latency_sec = 0.0
_last_model_reload_time: Optional[float] = None
_last_model_reload_version: Optional[str] = None

def _record_prediction_latency(latency_sec: float):
    global _prediction_request_count, _prediction_total_latency_sec
    with _metrics_lock:
        _prediction_request_count += 1
        _prediction_total_latency_sec += latency_sec

def _metrics_snapshot() -> dict:
    with _metrics_lock:
        avg_latency = (_prediction_total_latency_sec / _prediction_request_count) if _prediction_request_count else 0.0
        return {
            'uptime_seconds': round(time.time() - APP_START_TIME, 3),
            'model_loaded': model is not None,
            'model_version': model_version,
            'decision_threshold': _resolve_threshold()[0] if model is not None else None,
            'prediction_request_count': _prediction_request_count,
            'avg_prediction_latency_ms': round(avg_latency * 1000, 3),
            'last_model_reload_time': _last_model_reload_time,
            'last_model_reload_version': _last_model_reload_version
        }

# Inference artifacts
mte_mappings: Dict[str, Any] = {}
feature_contract: Dict[str, Any] = {}
feature_rules: Dict[str, Any] = {}
feature_schema: Dict[str, Any] = {}

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
TARGET_COL = os.getenv("TARGET_COL", "is_canceled")


def load_model_and_scaler():
    """Load the trained model and preprocessing pipeline (preferred) or legacy scaler.

    Future deployment flow (when MODEL_S3_URI is set):
      1. Resolve concrete version (read latest.txt if MODEL_VERSION == 'latest').
      2. Download champion_model.pkl, preprocessor.pkl, champion_meta.json.
      3. Populate model_version and DECISION_THRESHOLD for health reporting.
      4. Fallback to local MLflow load if remote fetch fails.
    """
    global model, preprocessor, model_version, DECISION_THRESHOLD, champion_meta_threshold, _last_model_reload_time, _last_model_reload_version

    try:
        # Load centralized preprocessor if present
        if os.path.exists(PREPROCESSOR_PATH):
            try:
                preprocessor = PreprocessingPipeline.load(PREPROCESSOR_PATH)
                print(f"✓ Preprocessor loaded from {PREPROCESSOR_PATH} (strategy={preprocessor.categorical_strategy})")
            except Exception as e:
                print(f"⚠ Failed to load preprocessor at {PREPROCESSOR_PATH}: {e}")
                preprocessor = None
        else:
            print(f"⚠ Preprocessor not found at {PREPROCESSOR_PATH}; attempting legacy scaler path")
            preprocessor = None
        # Remote S3 fetch (if configured)
        if MODEL_S3_URI:
            try:
                import boto3
                from botocore.exceptions import ClientError
                # Parse bucket + base prefix
                if not MODEL_S3_URI.startswith('s3://'):
                    raise ValueError('MODEL_S3_URI must start with s3://')
                remainder = MODEL_S3_URI[len('s3://'):]
                bucket, *rest = remainder.split('/', 1)
                base_prefix = rest[0].rstrip('/') if rest else ''

                s3 = boto3.client('s3', region_name=AWS_REGION) if AWS_REGION else boto3.client('s3')

                resolved_version = MODEL_VERSION
                if MODEL_VERSION == 'latest':
                    latest_key = f"{base_prefix}/latest.txt" if base_prefix else 'latest.txt'
                    obj = s3.get_object(Bucket=bucket, Key=latest_key)
                    resolved_version = obj['Body'].read().decode('utf-8').strip()
                    print(f"Resolved latest -> {resolved_version}")

                version_prefix = f"{base_prefix}/{resolved_version}" if base_prefix else resolved_version
                local_cache_dir = os.path.join('models', 'remote', resolved_version)
                os.makedirs(local_cache_dir, exist_ok=True)

                def _download(key_name: str, local_name: str):
                    target = os.path.join(local_cache_dir, local_name)
                    if not os.path.exists(target):
                        print(f"↓ S3 fetch s3://{bucket}/{version_prefix}/{key_name} -> {target}")
                        s3.download_file(bucket, f"{version_prefix}/{key_name}", target)
                    return target

                model_path = _download('champion_model.pkl', 'champion_model.pkl')
                preproc_path = _download('preprocessor.pkl', 'preprocessor.pkl')
                meta_path = _download('champion_meta.json', 'champion_meta.json')

                # Load model (assumed xgboost pickled or MLflow artifact)
                try:
                    model_candidate = joblib.load(model_path)
                    # Basic interface check
                    if hasattr(model_candidate, 'predict'):
                        globals()['model'] = model_candidate
                        print(f"✓ Loaded model from S3 cached path {model_path}")
                except Exception as e:
                    print(f"⚠ Failed loading S3 model pickle: {e}")

                # Override preprocessor if remote one available
                try:
                    remote_pre = PreprocessingPipeline.load(preproc_path)
                    preprocessor = remote_pre
                    print(f"✓ Loaded remote preprocessor from {preproc_path}")
                except Exception as e:
                    print(f"⚠ Failed loading remote preprocessor: {e}")

                # Champion meta for decision threshold
                try:
                    with open(meta_path) as f:
                        champ_meta = json.load(f)
                    # Store champion threshold if present
                    if 'decision_threshold' in champ_meta:
                        champion_meta_threshold = champ_meta['decision_threshold']
                    # If no explicit env override was provided pre-start, adopt champion threshold
                    if champion_meta_threshold is not None and not os.getenv('DECISION_THRESHOLD'):
                        os.environ['DECISION_THRESHOLD'] = str(champion_meta_threshold)
                    # Refresh module-level variable in all cases (env override may exist)
                    DECISION_THRESHOLD = os.getenv('DECISION_THRESHOLD')
                    model_version = resolved_version
                except Exception as e:
                    print(f"⚠ Could not read champion_meta.json: {e}")
            except ClientError as e:
                print(f"⚠ S3 fetch failed ({e.response['Error'].get('Code')}): falling back to local MLflow")
            except Exception as e:
                print(f"⚠ S3 fetch disabled due to error: {e}")

        else:
            print("ℹ MODEL_S3_URI not set; skipping remote fetch.")

        # Try to load model from MLflow (development / fallback)
        # Local champion pickle fallback (explicit)
        if model is None:
            local_champion = os.path.join('models','champion_model.pkl')
            if os.path.exists(local_champion):
                try:
                    model_candidate = joblib.load(local_champion)
                    if hasattr(model_candidate, 'predict'):
                        model = model_candidate
                        # Derive a pseudo version using file mtime for transparency
                        try:
                            mtime = int(os.path.getmtime(local_champion))
                            pseudo_version = f"local_{mtime}"
                        except Exception:
                            pseudo_version = 'local_champion'
                        globals()['model_version'] = pseudo_version
                        print(f"✓ Loaded local champion model from {local_champion} (model_version={pseudo_version})")
                except Exception as e:
                    print(f"⚠ Failed to load local champion model: {e}")

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

                if runs and model is None:  # only if no earlier model loaded
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

        if model is None:
            print("✗ No model successfully loaded (S3, local champion, MLflow all failed). API will report model_not_loaded.")
            print("  Troubleshooting suggestions:")
            print("   - If using S3: set MODEL_S3_URI (e.g. s3://<bucket>/models) and ensure AWS creds are available inside container")
            print("   - To mount local artifacts: docker run -v $(pwd)/models:/app/models:ro -v $(pwd)/artifacts:/app/artifacts:ro ...")
            print("   - To bake models for dev only: add 'COPY models/ ./models/' and 'COPY artifacts/ ./artifacts/' to Dockerfile (not for prod)")
        else:
            # Successful (re)load -> record reload timestamp & version
            _last_model_reload_time = time.time()
            _last_model_reload_version = model_version
            # Git SHA logging (best-effort)
            git_sha = os.getenv('GIT_SHA')
            if not git_sha:
                head_path = os.path.join('.git','HEAD')
                try:
                    if os.path.exists(head_path):
                        with open(head_path) as hf:
                            ref = hf.read().strip()
                        if ref.startswith('ref:'):
                            ref_file = ref.split(' ',1)[1]
                            ref_path = os.path.join('.git', ref_file)
                            if os.path.exists(ref_path):
                                with open(ref_path) as rf:
                                    git_sha = rf.read().strip()[:12]
                        else:
                            git_sha = ref[:12]
                except Exception:
                    git_sha = None
            if git_sha:
                print(f"ℹ Loaded model_version={model_version} (git_sha={git_sha})")

    except Exception as e:
        print(f"✗ Error loading model artifacts: {e}")
        model = None
        preprocessor = None


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

# --- Feature Alignment Layer -------------------------------------------------
API_TO_TRAINING_MAP = {
    # API schema -> training names (if different)
    'arrival_month': 'arrival_date_month',
    'stays_weekend_nights': 'stays_in_weekend_nights',
    'stays_week_nights': 'stays_in_week_nights'
}

TRAINING_REQUIRED_BASE = set([
    'hotel','lead_time','arrival_date_year','arrival_date_month','arrival_date_week_number',
    'arrival_date_day_of_month','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies',
    'meal','country','market_segment','distribution_channel','is_repeated_guest','previous_cancellations',
    'previous_bookings_not_canceled','reserved_room_type','assigned_room_type','booking_changes','deposit_type',
    'days_in_waiting_list','customer_type','adr','required_car_parking_spaces','total_of_special_requests'
])

ENGINEERED_COLUMNS = [
    'total_stay_duration','total_guests','is_family','guest_type','arrival_season','is_peak_season',
    'arrival_quarter','is_summer_peak','is_holiday_season'
]

TARGET_ENCODED_SUFFIX = '_target_encoded'

def _align_api_to_training(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Map the minimalist API payload into the richer training-time feature space.

    Steps:
      1. Rename API columns to training equivalents (month, stays names).
      2. Inject default placeholders for required categorical/numeric fields absent from API.
      3. Derive engineered features replicating training logic.
      4. Add target-encoded placeholder columns (they will be filled later by _apply_mean_target_encoding or left NaN).
    """
    df = raw_df.copy()
    # 1. Rename
    for api_col, train_col in API_TO_TRAINING_MAP.items():
        if api_col in df.columns and train_col not in df.columns:
            df[train_col] = df[api_col]
    # 2. Defaults for missing required base columns
    # Reasonable neutral defaults (0 or most common); could refine using distribution_baselines
    neutral_int_zero = ['hotel','arrival_date_year','arrival_date_week_number','arrival_date_day_of_month',
                        'babies','previous_bookings_not_canceled','days_in_waiting_list','reserved_room_type',
                        'assigned_room_type','deposit_type','meal','market_segment','distribution_channel','customer_type']
    for col in TRAINING_REQUIRED_BASE:
        if col not in df.columns:
            if col in neutral_int_zero:
                df[col] = 0
            elif col == 'country':
                df[col] = 'UNK'
            else:
                df[col] = 0
    # 3. Engineered features
    if 'stays_in_weekend_nights' in df.columns and 'stays_in_week_nights' in df.columns and 'total_stay_duration' not in df.columns:
        df['total_stay_duration'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    if 'adults' in df.columns and 'children' in df.columns and 'babies' in df.columns and 'total_guests' not in df.columns:
        df['total_guests'] = df['adults'] + df['children'].fillna(0) + df['babies']
    if 'children' in df.columns:
        babies_series = df['babies'] if 'babies' in df.columns else 0
        df['is_family'] = ((df['children'].fillna(0) > 0) | (babies_series > 0)).astype(int)
    else:
        df['is_family'] = 0
    # guest_type (similar logic to deterministic features)
    if 'guest_type' not in df.columns:
        def _guest(row):
            if row.get('babies',0) > 0: return 'family_with_babies'
            if row.get('children',0) > 0: return 'family_with_children'
            a = row.get('adults',0)
            if a == 1: return 'solo_traveler'
            if a == 2: return 'couple'
            return 'group'
        df['guest_type'] = df.apply(_guest, axis=1)
    # Seasonal features
    if 'arrival_date_month' in df.columns:
        m = df['arrival_date_month']
        season_map = {12:'winter',1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',6:'summer',7:'summer',8:'summer',9:'autumn',10:'autumn',11:'autumn'}
        df['arrival_season'] = m.map(season_map)
        df['is_peak_season'] = m.isin([5,6,7,8,9]).astype(int)
        df['arrival_quarter'] = m.apply(lambda x: f"Q{((x-1)//3)+1}")
        df['is_summer_peak'] = m.isin([7,8]).astype(int)
        df['is_holiday_season'] = m.isin([12,1]).astype(int)
    else:
        for c in ['arrival_season','is_peak_season','arrival_quarter','is_summer_peak','is_holiday_season']:
            if c not in df.columns:
                df[c] = np.nan
    # 4. Ensure target encoded columns present if contract expects them
    if feature_contract.get('feature_order'):
        for col in feature_contract['feature_order']:
            if col.endswith(TARGET_ENCODED_SUFFIX) and col not in df.columns:
                df[col] = np.nan
    return df

def _apply_scaler_if_available(df: pd.DataFrame) -> pd.DataFrame:
    """Apply stored scaler from preprocessing pipeline to overlapping numeric columns.

    This bypasses categorical handling differences because we already constructed the
    aligned feature frame using stored feature_contract ordering. Only numeric columns
    that were originally scaled are transformed.
    """
    global preprocessor
    if preprocessor is None or preprocessor.state is None or preprocessor._scaler is None:
        return df
    scaled_cols = [c for c in preprocessor.state.scaled_numeric if c in df.columns]
    if not scaled_cols:
        return df
    df_out = df.copy()
    # Ensure float dtype
    for c in scaled_cols:
        if not pd.api.types.is_float_dtype(df_out[c]):
            try:
                df_out[c] = df_out[c].astype('float64')
            except Exception:
                df_out[c] = pd.to_numeric(df_out[c], errors='coerce').astype('float64')
    df_out.loc[:, scaled_cols] = preprocessor._scaler.transform(df_out[scaled_cols])
    return df_out

def _prepare_for_target_strategy(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight preparation when using target encoding strategy.

    Applies basic renames and deterministic engineered features but does NOT attempt
    manual target encoding (delegated to preprocessor.transform). Avoids injecting
    placeholder *_target_encoded or __te columns to prevent clashes.
    """
    # Apply renames similar to alignment map (subset)
    for api_col, train_col in API_TO_TRAINING_MAP.items():
        if api_col in raw_df.columns and train_col not in raw_df.columns:
            raw_df[train_col] = raw_df[api_col]
    df = _apply_deterministic_features(raw_df)
    # Inject placeholder categorical columns that the target encoder saw during fit
    # We inspect the preprocessor.state.target_mappings keys to know required raw categorical bases.
    global preprocessor
    try:
        if preprocessor and getattr(preprocessor, 'state', None) and preprocessor.state.target_mappings:
            needed_cats = list(preprocessor.state.target_mappings.keys())
            for cat in needed_cats:
                if cat not in df.columns:
                    # Provide neutral / unknown placeholder. For geo-like fields use 'UNK'; else 'unknown'.
                    if cat in ('country',):
                        df[cat] = 'UNK'
                    else:
                        df[cat] = 'unknown'
        # Ensure engineered seasonality fields base column present (arrival_date_month) to derive consistent categories
        if 'arrival_date_month' in df.columns:
            # recompute seasonal groupings in case placeholders were added after deterministic step
            m = df['arrival_date_month']
            season_map = {12:'winter',1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',6:'summer',7:'summer',8:'summer',9:'autumn',10:'autumn',11:'autumn'}
            df['arrival_season'] = m.map(season_map)
            df['arrival_quarter'] = m.apply(lambda x: f"Q{((x-1)//3)+1}")
        # guest_type already created in deterministic features; if missing create again
        if 'guest_type' not in df.columns and {'adults','children'}.issubset(df.columns):
            def _guest(row):
                if row.get('children',0) > 0: return 'family_with_children'
                a = row.get('adults',0)
                if a == 1: return 'solo_traveler'
                if a == 2: return 'couple'
                return 'group'
            df['guest_type'] = df.apply(_guest, axis=1)
    except Exception as e:
        print(f"⚠ Target strategy placeholder injection failed: {e}")
    return df


def _resolve_threshold() -> tuple[float, str]:
    """Determine active classification threshold and its source.

    Order of precedence:
      1. DECISION_THRESHOLD env var (user override)
      2. champion_meta_threshold loaded from champion_meta.json
      3. default 0.5
    Returns: (threshold_value, source_label)
    """
    global DECISION_THRESHOLD, champion_meta_threshold
    if DECISION_THRESHOLD is not None:
        try:
            return float(DECISION_THRESHOLD), 'env'
        except ValueError:
            pass
    if champion_meta_threshold is not None:
        try:
            return float(champion_meta_threshold), 'champion_meta'
        except ValueError:
            pass
    return 0.5, 'default'


## Removed deprecated on_event startup in favor of lifespan context


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
    thr, _src = _resolve_threshold()
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_version=model_version,
        decision_threshold=thr if model is not None else None
    )


@app.get("/metrics", response_model=dict)
async def metrics():
    """Lightweight JSON metrics for operational insight."""
    return _metrics_snapshot()


class ReloadRequest(BaseModel):
    version: Optional[str] = Field(None, description="Specific model version to load (or 'latest')")
    force: bool = Field(False, description="Force reload even if version unchanged")
    threshold_override: Optional[float] = Field(None, description="Optional override for decision threshold after reload")


@app.post("/model/reload", response_model=dict)
async def reload_model(req: ReloadRequest):
    """Reload model / preprocessor from S3 (if configured) or local artifacts.

    Allows specifying a concrete version; falls back to environment MODEL_VERSION if not provided.
    """
    global MODEL_VERSION, DECISION_THRESHOLD
    if req.version:
        if req.version != MODEL_VERSION:
            MODEL_VERSION = req.version
            os.environ['MODEL_VERSION'] = req.version
        elif not req.force:
            return { 'reloaded': False, 'reason': 'version_unchanged', 'model_version': model_version }
    prev_version = model_version
    if req.threshold_override is not None:
        DECISION_THRESHOLD = str(req.threshold_override)
        os.environ['DECISION_THRESHOLD'] = str(req.threshold_override)
    load_model_and_scaler()
    return {
        'reloaded': model is not None,
        'previous_version': prev_version,
        'new_version': model_version,
        'active_threshold': _resolve_threshold()[0],
        'threshold_source': _resolve_threshold()[1]
    }


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
    
    if preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Preprocessor not loaded. Ensure training script produced models/preprocessor.pkl."
        )
    
    try:
        _t0 = time.perf_counter()
        # Convert input to DataFrame
        raw_df = pd.DataFrame([booking.dict()])
        if preprocessor and getattr(preprocessor, 'state', None) and preprocessor.state.categorical_strategy == 'target':
            # Use preprocessor.transform end-to-end (includes target encoding + scaling)
            prep_df = _prepare_for_target_strategy(raw_df.copy())
            try:
                input_processed = preprocessor.transform(prep_df)
                # Verify expected feature order columns present
                expected = set(preprocessor.state.feature_order)
                missing_after = [c for c in preprocessor.state.feature_order if c not in input_processed.columns]
                if missing_after:
                    raise ValueError(f"Post-transform missing encoded columns: {missing_after}. Raw provided columns: {list(prep_df.columns)}")
            except Exception as e:
                print(f"⚠ Target strategy transform failed, fallback to alignment path: {e}")
                aligned_df = _align_api_to_training(raw_df)
                feature_df = _build_feature_matrix(aligned_df)
                input_processed = _apply_scaler_if_available(feature_df)
        else:
            aligned_df = _align_api_to_training(raw_df)
            feature_df = _build_feature_matrix(aligned_df)
            input_processed = _apply_scaler_if_available(feature_df)

        # Defensive: coerce any remaining object / categorical dtypes
        if hasattr(input_processed, 'dtypes'):
            obj_cols = [c for c in input_processed.columns if input_processed[c].dtype == 'object']
            if obj_cols:
                coercion_map = {}
                for c in obj_cols:
                    # If column looks numeric, attempt float conversion first
                    try:
                        input_processed[c] = pd.to_numeric(input_processed[c])
                        coercion_map[c] = 'numeric_cast'
                    except Exception:
                        codes, uniques = pd.factorize(input_processed[c].astype(str))
                        input_processed[c] = codes.astype(np.int32)
                        coercion_map[c] = 'factorized'
                print(f"✓ Coerced object columns: {coercion_map}")

        # Probability
        probability = (model.predict_proba(input_processed)[0, 1]
                       if hasattr(model, 'predict_proba') else float(model.predict(input_processed)[0]))
        # Determine threshold & class
        thr, src = _resolve_threshold()
        prediction = int(probability >= thr)

        resp = PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model_used=model_name,
            applied_threshold=thr,
            threshold_source=src
        )
        _record_prediction_latency(time.perf_counter() - _t0)
        return resp
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
    
    if preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Preprocessor not loaded. Ensure training script produced models/preprocessor.pkl."
        )
    
    try:
        _t0 = time.perf_counter()
        # Convert inputs to DataFrame
        raw_df = pd.DataFrame([booking.dict() for booking in bookings])
        if preprocessor and getattr(preprocessor,'state',None) and preprocessor.state.categorical_strategy == 'target':
            prep_df = _prepare_for_target_strategy(raw_df.copy())
            try:
                input_processed = preprocessor.transform(prep_df)
                missing_after = [c for c in preprocessor.state.feature_order if c not in input_processed.columns]
                if missing_after:
                    raise ValueError(f"Post-transform missing encoded columns: {missing_after}. Raw provided columns: {list(prep_df.columns)}")
            except Exception as e:
                print(f"⚠ Target strategy batch transform failed, fallback alignment path: {e}")
                aligned_df = _align_api_to_training(raw_df)
                feature_df = _build_feature_matrix(aligned_df)
                input_processed = _apply_scaler_if_available(feature_df)
        else:
            aligned_df = _align_api_to_training(raw_df)
            feature_df = _build_feature_matrix(aligned_df)
            input_processed = _apply_scaler_if_available(feature_df)
        if hasattr(input_processed, 'dtypes'):
            obj_cols = [c for c in input_processed.columns if input_processed[c].dtype == 'object']
            if obj_cols:
                coercion_map = {}
                for c in obj_cols:
                    try:
                        input_processed[c] = pd.to_numeric(input_processed[c])
                        coercion_map[c] = 'numeric_cast'
                    except Exception:
                        codes, uniques = pd.factorize(input_processed[c].astype(str))
                        input_processed[c] = codes.astype(np.int32)
                        coercion_map[c] = 'factorized'
                print(f"✓ (batch) coerced object columns: {coercion_map}")

        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_processed)[:, 1]
        else:
            # Fallback: treat model.predict outputs as probabilities (rare for tree models)
            probabilities = model.predict(input_processed).astype(float)
        thr, src = _resolve_threshold()
        predictions = (probabilities >= thr).astype(int)

        # Create response list
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append(
                PredictionResponse(
                    prediction=int(pred),
                    probability=float(prob),
                    model_used=model_name,
                    applied_threshold=thr,
                    threshold_source=src
                )
            )

        _record_prediction_latency(time.perf_counter() - _t0)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making batch prediction: {str(e)}"
        )


@app.get("/model/interpretability", response_model=InterpretabilityResponse)
async def get_interpretability(top_k: int = 15):
    """Return global & sample local interpretability metadata.

    Reads artifacts produced by training script. If SHAP artifacts are missing, returns shap_generated=False
    with empty lists, allowing clients to degrade gracefully.
    """
    artifacts_present = []
    artifacts_dir = ARTIFACT_DIR
    champion_meta_path = os.path.join(artifacts_dir, 'champion_meta.json')
    feature_importance_path = os.path.join(artifacts_dir, 'feature_importance.json')
    shap_local_path = os.path.join(artifacts_dir, 'shap_values_sample.json')
    feature_name_map_path = os.path.join(artifacts_dir, 'feature_name_map.json')

    champion_model = None
    shap_generated = False
    shap_timestamp = None
    decision_threshold = None
    top_features: List[Dict[str, Any]] = []
    local_examples: List[LocalExplanation] = []
    feature_name_map: Dict[str, str] = {}

    # Champion metadata
    if os.path.exists(champion_meta_path):
        try:
            with open(champion_meta_path) as f:
                champ_meta = json.load(f)
            champion_model = champ_meta.get('model_name')
            decision_threshold = champ_meta.get('decision_threshold')
            shap_generated = bool(champ_meta.get('shap_generated'))
            shap_timestamp = champ_meta.get('shap_timestamp')
            artifacts_present.append('champion_meta.json')
        except Exception:
            pass

    # Feature name map
    if os.path.exists(feature_name_map_path):
        try:
            with open(feature_name_map_path) as f:
                feature_name_map = json.load(f)
            artifacts_present.append('feature_name_map.json')
        except Exception:
            feature_name_map = {}

    # Global feature importance
    if os.path.exists(feature_importance_path):
        try:
            with open(feature_importance_path) as f:
                importance = json.load(f)
            # importance is list of {feature, mean_abs_shap}
            for item in importance[:top_k]:
                feat = item.get('feature')
                human = feature_name_map.get(feat, feat)
                top_features.append({
                    'feature': feat,
                    'human_readable': human,
                    'mean_abs_shap': item.get('mean_abs_shap')
                })
            artifacts_present.append('feature_importance.json')
        except Exception:
            top_features = []

    # Local examples (convert raw shap values into top +/- lists)
    if os.path.exists(shap_local_path):
        try:
            with open(shap_local_path) as f:
                local_raw = json.load(f)
            for rec in local_raw:
                shap_dict = rec.get('shap_values', {})
                # Sort positive and negative contributions separately
                positives = sorted([ (k,v) for k,v in shap_dict.items() if v > 0 ], key=lambda x: x[1], reverse=True)[:5]
                negatives = sorted([ (k,v) for k,v in shap_dict.items() if v < 0 ], key=lambda x: x[1])[:5]
                local_examples.append(LocalExplanation(
                    category=rec.get('category','sample'),
                    probability=rec.get('probability'),
                    top_positive_contributors=[{'feature': f, 'shap': v, 'human_readable': feature_name_map.get(f, f)} for f,v in positives],
                    top_negative_contributors=[{'feature': f, 'shap': v, 'human_readable': feature_name_map.get(f, f)} for f,v in negatives]
                ))
            artifacts_present.append('shap_values_sample.json')
        except Exception:
            local_examples = []

    return InterpretabilityResponse(
        champion_model=champion_model,
        shap_generated=shap_generated and bool(top_features),
        shap_timestamp=shap_timestamp,
        decision_threshold=decision_threshold,
        top_features=top_features,
        local_examples=local_examples,
        feature_name_map=feature_name_map,
        artifacts_available=artifacts_present
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ------------------------- Optional Debug Endpoint ---------------------------
ENABLE_DEBUG = os.getenv('ENABLE_DEBUG_ENDPOINT', 'false').lower() == 'true'
if ENABLE_DEBUG:
    @app.get('/debug/model')
    async def debug_model():
        """Return internal diagnostic details about model loading (do not enable in production)."""
        def _file_info(path):
            if os.path.exists(path):
                try:
                    return { 'exists': True, 'size_bytes': os.path.getsize(path) }
                except Exception:
                    return { 'exists': True, 'size_bytes': None }
            return { 'exists': False, 'size_bytes': None }
        local_champion_path = os.path.join('models','champion_model.pkl')
        remote_cache_root = os.path.join('models','remote')
        remote_versions = []
        if os.path.exists(remote_cache_root):
            for root, dirs, files in os.walk(remote_cache_root):
                for f in files:
                    if f == 'champion_model.pkl':
                        remote_versions.append(os.path.relpath(os.path.join(root,f), remote_cache_root))
        return {
            'model_loaded': model is not None,
            'model_version': model_version,
            'env': {
                'MODEL_S3_URI': MODEL_S3_URI,
                'MODEL_VERSION': MODEL_VERSION,
                'AWS_REGION': AWS_REGION,
                'DECISION_THRESHOLD': DECISION_THRESHOLD
            },
            'paths': {
                'local_champion': _file_info(local_champion_path),
                'preprocessor': _file_info(PREPROCESSOR_PATH),
            },
            'remote_cached_versions': remote_versions,
            'feature_contract_loaded': bool(feature_contract),
            'preprocessor_loaded': preprocessor is not None and getattr(preprocessor,'state', None) is not None
        }
