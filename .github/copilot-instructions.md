# AI Agent Instructions - Hotel Cancellation Prediction

## Project Architecture

This is a production-ready ML service for predicting hotel booking cancellations using a multi-model ensemble approach with MLflow experiment tracking and FastAPI REST endpoints.

### Core Components & Data Flow

- `scripts/train.py`: Trains 4 models (LogReg, RandomForest, XGBoost, PyTorch MLP) with synthetic data generation
- `main.py`: FastAPI app that loads XGBoost as default production model with Pydantic validation
- `models/`: Stores trained models (XGBoost pkl) and preprocessing artifacts (scaler.pkl)
- `mlruns/`: MLflow experiment tracking database (local filesystem)

### Key Architecture Decisions

- **XGBoost as Production Default**: While 4 models are trained, only XGBoost is loaded for API predictions (see `load_model()` in main.py)
- **Shared Preprocessing**: Single StandardScaler trained during model training, reused for API inference
- **Synthetic Data Pipeline**: Training generates sample data internally rather than loading external datasets

## Development Workflows

### Training & Experimentation

```bash
python scripts/train.py  # Trains all models, saves artifacts to models/
mlflow ui                # View experiment comparisons at localhost:5000
```

### Local API Development

```bash
uvicorn main:app --reload --port 8000  # Development server with hot reload
python scripts/test_api.py             # Test client with sample requests
```

### Environment Configuration

```bash
cp .env.example .env  # Configure local environment variables
# Edit .env for custom ports, paths, or credentials
```

### Docker Deployment

```bash
docker-compose up  # Runs API + MLflow UI with volume mounts for persistence
```

## Project-Specific Conventions

### Model Loading Pattern

Models are loaded via MLflow artifacts but fall back to joblib for XGBoost if MLflow fails. The pattern in `main.py`:

```python
model = mlflow.xgboost.load_model("models/xgboost_model")  # Preferred
# Fallback: model = joblib.load("models/xgboost_model.pkl")
```

### Feature Schema Validation

All booking features use Pydantic with explicit validation ranges (see `BookingFeatures` class):

- Categorical constraints: `arrival_month` (1-12), `is_repeated_guest` (0-1)
- Business logic: `adults >= 1`, all counts `>= 0`
- Schema serves as both API contract and model input specification

### Environment Variable Strategy

- Configuration loaded from `.env` via python-dotenv with sensible defaults
- Key variables: `MODEL_PATH`, `SCALER_PATH`, `MLFLOW_TRACKING_URI`, `API_PORT`
- Docker compose uses env vars for port mapping and MLflow configuration
- Production secrets (API keys, database URLs) stored in `.env` (gitignored)

### Error Handling Strategy

- API returns structured error responses for validation failures
- Model loading errors cause startup failure (fail-fast principle)
- Health endpoint checks model availability before accepting predictions

## Critical Integration Points

### MLflow Integration

- Experiments auto-logged with metrics, parameters, and model artifacts
- Models registered to local filesystem backend (`file:./mlruns`)
- Production model loading expects MLflow run structure in `models/` directory

### Docker Volume Strategy

```yaml
- ./models:/app/models:ro # Read-only model artifacts
- ./mlruns:/app/mlruns:ro # Read-only experiment data
```

Changes to models require container restart since volumes are read-only.

## Testing & Debugging

### API Testing Pattern

`scripts/test_api.py` demonstrates the expected request/response cycle:

1. Health check validation
2. Single prediction with sample booking
3. Batch predictions with multiple bookings

### Model Validation

When modifying models, verify the complete pipeline:

1. Train new models: `python scripts/train.py`
2. Test API loading: `curl localhost:8000/health`
3. Validate predictions: `python scripts/test_api.py`

### Common Issues

- **Model not found**: Ensure `models/xgboost_model.pkl` and `models/scaler.pkl` exist after training
- **Prediction errors**: Check feature schema matches training data preprocessing
- **MLflow issues**: Verify `mlruns/` directory structure matches experiment logging
