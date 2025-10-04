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

- `scripts/train.py`: Trains 4 models (LogReg, RandomForest, XGBoost, PyTorch MLP) with synthetic data generation
- `main.py`: FastAPI app that loads XGBoost as default production model with Pydantic validation
- `models/`: Stores trained models (XGBoost pkl) and preprocessing artifacts (scaler.pkl)
- `mlruns/`: MLflow experiment tracking database (local filesystem)

### Key Architecture Decisions

- **XGBoost as Champion Model**: Based on academic evaluation, XGBoost achieved highest F1-score (0.893) and ROC-AUC (0.958) in 5-fold cross-validation
- **Academic Rigor**: Models evaluated using stratified cross-validation with Optuna hyperparameter optimization
- **Interpretability Focus**: SHAP (SHapley Additive exPlanations) used for model explainability and business insights
- **Shared Preprocessing**: Single StandardScaler trained during model training, reused for API inference
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

Models are loaded via MLflow artifacts but fall back to joblib for XGBoost if MLflow fails. The pattern in `main.py`:

```python
model = mlflow.xgboost.load_model("models/xgboost_model")  # Preferred
# Fallback: model = joblib.load("models/xgboost_model.pkl")
```

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

### Predicting Hotel Bookings Cancelations Flow

1. Load the data set and plot the distribution of the target variable to understand class imbalance.
2. EDA (Exploratory Data Analysis)
    - plot number of missing values in each column. 
   - Check reservation_status and reservation_status_date has been updated after the booking is canceled.
   - Column company and agent have a high percentage of missing values. So, we can drop these two columns.
   - Drop columns by pipe line: agent, company, country, reservation_status, reservation_status_date, booking_changes
   - check lead time increases the chance of cancellation.
   - Spital analysis find where the customers are coming from.
   - corelation analysis to find the relationship between the features and the target variable, select the most important features.
   - perform mean encoding for categorical variables.
   handling outlier data
   - SMOTE method to handle the class imbalance problem.
2. Preprocess the data (e.g., handle missing values, encode categorical variables).
3. Split the data into training and testing sets.
4. Train the model using the training set.
5. Evaluate the model on the testing set and log metrics to MLflow.
6. Random Forrest and Logistic Regression are used for the model training.
7. Split the data into to train and test sets.
8. Build categorical and continuous pipelines.
9. Fit model
10. Initialize search space for hyperparameter tuning.
11. Use RandomizedSearchCV for hyperparameter tuning.
12. Find best hyperparameters and fit the model.
13. Regularization parameter C for Logistic Regression.
14. Number of trees in the forest for Random Forest.
15. Maximum depth of the tree for Random Forest.
16. Evaluate the model using accuracy, precision, recall, and F1-score.
17. cross validation is used for model evaluation.
18 Find which variable is most important for the prediction using SHAP (SHapley Additive exPlanations)




Logisytic Regression, naive bayes, random forest, Descision tree, KNN
Random Forrest confusion matrix