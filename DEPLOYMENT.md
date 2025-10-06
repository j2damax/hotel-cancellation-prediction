# Deployment Guide

Deploy the Hotel Cancellation Prediction API as a Hugging Face Space.

## Overview

The API can be deployed as a Hugging Face Space (FastAPI or Docker Space) that dynamically loads model artifacts from a Hugging Face model repository.

## Prerequisites

- Hugging Face account
- Model repository with artifacts: `champion_model.pkl`, `preprocessor.pkl`, `champion_meta.json`
- (Optional) Additional SHAP and interpretability artifacts

## Deployment Steps

### 1. Create Space

```bash
git clone https://huggingface.co/spaces/<username>/<space-name>
cd <space-name>
```

Or create a new Space in the Hugging Face UI, then clone it locally.

### 2. Copy Runtime Files

```bash
cp ../hotel-cancellation-prediction/main.py .
cp -R ../hotel-cancellation-prediction/app ./app
cp ../hotel-cancellation-prediction/requirements.txt .
```

**Minimal requirements.txt for inference:**

```
fastapi
uvicorn[standard]
pydantic
scikit-learn==1.7.2
xgboost
pandas
numpy
joblib
huggingface_hub
python-dotenv
```

Add `torch` only if using PyTorch MLP model.

### 3. Configure Environment Variables

In Space settings (Variables & secrets):

```
HF_MODEL_REPO=<username>/hotel-cancel-champion  # Required
DECISION_THRESHOLD=0.42                         # Optional
```

### 4. (Optional) Dockerfile for Docker Space

```dockerfile
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY main.py ./
COPY app ./app
COPY src ./src
# Create directories for HuggingFace model downloads
RUN mkdir -p artifacts models
# Create non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

Note: Hugging Face Spaces use port 7860. The `models` directory is required for downloading artifacts from HuggingFace Hub at runtime.

### 5. Commit and Deploy

```bash
git add .
git commit -m "Deploy FastAPI hotel cancellation API"
git push
```

Watch build logs in the Space UI.

### 6. Test Endpoints

```bash
# Health check
curl https://huggingface.co/spaces/<username>/<space-name>/health

# Prediction
curl -X POST https://huggingface.co/spaces/<username>/<space-name>/predict \
  -H 'Content-Type: application/json' \
  -d '{"lead_time":30,"arrival_month":7,"adults":2,"children":0,"adr":120.0}'
```

For Docker Spaces, you may need `/proxy/` in the path.

## Updating Model

1. Retrain locally: `python scripts/train.py`
2. Push artifacts: `python scripts/push_to_hf.py`
3. Restart Space or push a commit to trigger refresh

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| 503 model_not_loaded | Missing `HF_MODEL_REPO` | Add environment variable |
| Permission denied: 'models' | Missing models directory in Docker | Ensure Dockerfile creates `models/` before switching to non-root user |
| InconsistentVersionWarning | Version mismatch | Pin `scikit-learn==1.7.2` |
| 500 preprocessor error | Code drift | Align preprocessing logic |
| Slow first request | Cold start | Normal; subsequent requests faster |

## Security

- Never commit secrets
- Use Space secrets UI for sensitive variables
- Make model repo private if needed

## Local Testing

Test locally before deploying:

```bash
docker build -t hotel-cancel .
docker run -p 8000:7860 -e HF_MODEL_REPO=<username>/hotel-cancel-champion hotel-cancel
curl localhost:8000/health
```

## Alternative Deployment

The containerized application can be deployed to any platform supporting Docker:

- Kubernetes
- Cloud Run (GCP)
- App Service (Azure)
- Fargate (AWS)
- Fly.io, Render, etc.

Key considerations:
1. Artifacts: Bake into image or load from Hugging Face Hub at startup
2. Expose `/health` endpoint for health checks
3. Use environment variables for configuration
