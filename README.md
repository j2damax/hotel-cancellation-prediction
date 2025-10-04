# Hotel Cancellation Prediction

A Data-Driven Framework for Predicting Hotel Booking Cancellations using Machine Learning

## Overview

This project implements a complete machine learning pipeline for predicting hotel booking cancellations. It includes multiple models (Logistic Regression, Random Forest, XGBoost, and PyTorch MLP), MLflow experiment tracking, and a FastAPI-based prediction service containerized with Docker for deployment to Amazon ECR.

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
├── DEPLOYMENT.md        # AWS ECR deployment guide
├── QUICKSTART.md        # Quick start guide
├── EDA.md              # Comprehensive EDA methodology (1,624 lines)
├── preprocessing.md     # Preprocessing strategies guide (1,445 lines)
├── features.md         # Feature engineering guide (1,653 lines)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Features

### Hybrid Architecture: Notebooks + Production Scripts

This project implements a **hybrid approach** combining interactive analysis with production-ready pipelines:

**📓 Jupyter Notebooks** (for research and analysis):
- `01_eda.ipynb` - Interactive exploratory data analysis with visualizations
- `02_preprocessing_analysis.ipynb` - Preprocessing strategy experimentation
- `03_feature_engineering.ipynb` - Feature engineering with effectiveness testing
- `04_model_evaluation.ipynb` - Model comparison with SHAP interpretability

**🐍 Python Scripts** (for production deployment):
- `preprocessing.py` - Automated data preprocessing pipeline
- `feature_engineering.py` - Production feature engineering with cross-validation
- `model_evaluation.py` - Comprehensive model evaluation with statistical testing
- `train.py` - Complete training pipeline for deployment

### Academic Research Framework

- **NIB 7072 Coursework Compliance**: Rigorous academic standards with statistical significance testing
- **Sri Lankan Tourism Focus**: Domain-specific features and business impact analysis  
- **5-Fold Cross-Validation**: Stratified sampling with performance confidence intervals
- **SHAP Interpretability**: Model explainability for actionable business insights
- **Comprehensive Documentation**: 4,700+ lines of methodology documentation

### Machine Learning Models

- **Logistic Regression**: Baseline linear model with L1/L2 regularization
- **Random Forest**: Ensemble tree-based model with optimized hyperparameters
- **XGBoost**: Gradient boosting model (champion model with F1=0.893, ROC-AUC=0.958)
- **PyTorch MLP**: Deep learning neural network with dropout and batch normalization

### MLflow Integration

- Experiment tracking for all models with Optuna hyperparameter optimization
- Automatic logging of parameters, metrics, and model artifacts
- Model comparison and versioning with statistical significance testing
- Easy model registry integration for production deployment

### FastAPI REST API

- `/predict` - Single prediction endpoint with Pydantic validation
- `/predict/batch` - Batch prediction endpoint for bulk processing
- `/health` - Health check endpoint with model availability verification
- Interactive API documentation at `/docs` with schema validation

### Docker Containerization

- Optimized Docker image for production deployment
- Health checks included
- Ready for Amazon ECR deployment

## Installation

### Prerequisites

- Python 3.10+
- Docker (optional, for containerization)

### Local Setup

1. Clone the repository:

```bash
git clone https://github.com/j2damax/hotel-cancellation-prediction.git
cd hotel-cancellation-prediction
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Configure environment (optional):

```bash
# Copy environment template
cp .env.example .env
# Edit .env with your preferred settings (optional - defaults work fine)
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training Models

Run the training script to train all models with MLflow tracking:

```bash
python scripts/train.py
```

This will:

- Generate sample hotel booking data
- Train 4 different models (LogReg, RF, XGBoost, PyTorch MLP)
- Log all experiments to MLflow
- Save the unified preprocessing artifact to `models/preprocessor.pkl`

View MLflow UI to compare models:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### Running the API

Start the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or run directly:

```bash
python main.py
```

Access the API:

- API Root: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Testing the API

Use the provided test client:

```bash
python scripts/test_api.py
```

Or make manual requests (see examples below).

### Making Predictions

Example using curl:

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

### Using Docker Compose (Recommended for Local Testing)

The easiest way to run the application locally with Docker:

```bash
# Start both API and MLflow UI
docker-compose up

# Or run in detached mode
docker-compose up -d
```

This will start:

- API server on http://localhost:8000
- MLflow UI on http://localhost:5000

To stop:

```bash
docker-compose down
```

### Building the Docker Image

```bash
docker build -t hotel-cancellation-prediction .
```

### Running the Container Locally

```bash
docker run -p 8000:8000 hotel-cancellation-prediction
```

### Deploying to Amazon ECR

For detailed instructions on deploying to AWS ECR and running on ECS, EKS, or App Runner, see [DEPLOYMENT.md](DEPLOYMENT.md).

Quick start:

1. Authenticate Docker to ECR:

```bash
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
```

2. Create ECR repository (if not exists):

```bash
aws ecr create-repository --repository-name hotel-cancellation-prediction --region <region>
```

3. Tag the image:

```bash
docker tag hotel-cancellation-prediction:latest <account-id>.dkr.ecr.<region>.amazonaws.com/hotel-cancellation-prediction:latest
```

4. Push to ECR:

```bash
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/hotel-cancellation-prediction:latest
```

5. Deploy to ECS, EKS, or other AWS services using the ECR image.

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

After training, you can compare model performance in the MLflow UI. Metrics tracked include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

## Environment Configuration

The application supports environment-based configuration through `.env` files:

### Quick Setup

```bash
# Copy the template
cp .env.example .env

# Edit with your settings (optional - defaults work fine)
# Common customizations:
# API_PORT=8001               # Change API port
# MLFLOW_UI_PORT=5002         # Change MLflow UI port
# LOG_LEVEL=DEBUG             # Enable debug logging
# TRAINING_DATA_SIZE=20000    # Larger training dataset
```

### Key Environment Variables

- `API_PORT`: FastAPI server port (default: 8000)
- `MLFLOW_UI_PORT`: MLflow UI port (default: 5001)
- `MLFLOW_TRACKING_URI`: MLflow backend URI (default: file:./mlruns)
- `MODEL_PATH`: Directory for saved models (default: models/)
- `LOG_LEVEL`: Logging level (default: INFO)

### Production Configuration

For production deployments, set secure values in `.env`:

```bash
# Security
API_KEY=your-secret-api-key
JWT_SECRET=your-jwt-secret

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012
ECR_REPOSITORY_NAME=hotel-cancellation-prediction

# Performance
MAX_WORKERS=8
BATCH_SIZE=200
```

## Dependencies

See `requirements.txt` for complete list. Key dependencies:

- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- torch >= 2.0.0
- mlflow >= 2.9.0
- fastapi >= 0.104.0
- uvicorn >= 0.24.0
- python-dotenv >= 1.0.0

## License

MIT License - see LICENSE file for details

## Author

Jayampathy Balasuriya

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
