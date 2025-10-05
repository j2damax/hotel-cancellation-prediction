# AI Agent Instructions - Hotel Cancellation Prediction

## Project Context & Academic Foundation

This is an **academic ML research project** (NIB 7072 coursework) developing a production-ready framework for predicting hotel booking cancellations, with specific focus on **Sri Lankan tourism market applications**. The project addresses the critical business challenge of perishable inventory in hospitality, where cancellation rates can cause 10-30% revenue losses.

**Research Objectives:**

- Comparative analysis of 4 ML paradigms (LogReg, RandomForest, XGBoost, PyTorch MLP)
- SHAP-based model interpretability for actionable business insights
- End-to-end MLOps pipeline from training to AWS deployment
- Translation of predictions into Sri Lankan hospitality strategies

**Future Vision:** Foundation for "Serendipity by Design" - a generative AI platform for narrative-driven cultural itineraries in Sri Lanka, using this model as a risk-assessment engine.

## Project Architecture

This is a production-ready ML service for predicting hotel booking cancellations using a multi-model ensemble approach with MLflow experiment tracking and FastAPI REST endpoints.

### Core Components & Data Flow

- `scripts/train.py`: Trains 4 models (LogReg, RandomForest, XGBoost, PyTorch MLP), optional stratified cross-validation, champion selection & persistence, diagnostics + SHAP artifact generation
- `main.py`: FastAPI app that loads the persisted champion model dynamically with Pydantic validation
- `models/`: Stores persisted champion model (`champion_model.pkl`) and preprocessing artifact (`preprocessor.pkl` consolidating scaling & categorical handling)
- `mlruns/`: MLflow experiment tracking database (local filesystem)

### Key Architecture Decisions

- **Dynamic Champion Selection**: Champion chosen per run via stratified cross-validation (primary metric F1; ROC-AUC tie-break). Historical experiments often favored XGBoost, but no model is hard-coded.
- **Academic Rigor**: Models evaluated using stratified cross-validation with Optuna hyperparameter optimization
- **Interpretability Focus**: SHAP (SHapley Additive exPlanations) used for model explainability and business insights
- **Shared Preprocessing**: Single preprocessing pipeline (scaling + categorical encoding) reused for API inference
- **Feature Engineering**: Novel features created (total_stay_duration, is_family, guest_type) for enhanced predictive power
- **Class Imbalance Handling**: 32.8% cancellation rate addressed using class weights rather than resampling

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

Inference loads `models/champion_model.pkl` (persisted after training). If future registry integration is added, replace with MLflow Model Registry URI loading.

### Academic Evaluation Framework

The project follows rigorous academic standards with specific evaluation methodology:

- **Primary Metric**: F1-Score chosen for imbalanced classification (32.8% cancellation rate)
- **Secondary Metrics**: ROC-AUC, Precision, Recall for comprehensive evaluation
- **Cross-Validation**: 5-fold stratified to preserve class distribution
- **Hyperparameter Optimization**: Optuna framework with Tree-structured Parzen Estimator (TPE)
- **Experiment Tracking**: All runs logged to MLflow with parameters, metrics, and model artifacts

### Feature Schema & Business Logic

All booking features use Pydantic with explicit validation ranges (see `BookingFeatures` class):

- **Key Predictors (SHAP-identified)**: `lead_time`, `avg_price_per_room`, `no_of_special_requests`, `market_segment_type`
- **Categorical constraints**: `arrival_month` (1-12), `is_repeated_guest` (0-1)
- **Business logic**: `adults >= 1`, all counts `>= 0`
- **Engineered Features**: `total_stay_duration`, `is_family`, `guest_type` for enhanced prediction
- Schema serves as both API contract and model input specification

### Environment Variable Strategy

- Configuration loaded from `.env` via python-dotenv with sensible defaults
- Key variables: `MODEL_PATH`, `PREPROCESSOR_PATH`, `MLFLOW_TRACKING_URI`, `API_PORT`
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

- **Model not found**: Ensure `models/xgboost_model.pkl` and `models/preprocessor.pkl` exist after training
- **Prediction errors**: Check feature schema matches training data preprocessing
- **MLflow issues**: Verify `mlruns/` directory structure matches experiment logging

### Training & Interpretability Flow (Current)

1. Run `python scripts/train.py --cv-folds 5` (optional folds) to train and evaluate models.
2. Cross-validation metrics written to `artifacts/cv_metrics.json`.
3. Champion selected and persisted to `models/champion_model.pkl`; metadata in `artifacts/champion_meta.json`.
4. Diagnostics generated: `confusion_matrix.png`, `roc_curve.json`, `pr_curve.json`, `threshold_sweep.csv`, `classification_report.json`.
5. SHAP artifacts produced: `shap_summary.png`, `shap_importance_bar.png`, `feature_importance.json`, `shap_values_sample.json`, `feature_name_map.json`.
6. API endpoint `/model/interpretability` serves champion + global feature importance + sampled local explanations.

Legacy exploration algorithms (Naive Bayes, Decision Tree, KNN) remain out-of-scope for the production pipeline to maintain academic rigor and manageable complexity.
