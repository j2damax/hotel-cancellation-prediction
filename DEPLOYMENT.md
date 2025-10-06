# Deployment Guide

Deploy the Hotel Cancellation Prediction API as a Hugging Face Space (recommended) using baked local artifacts or dynamic loading from a Hugging Face model repository.

## Overview

The API can be deployed as a Hugging Face Space (FastAPI or Docker Space) that dynamically loads model artifacts from a Hugging Face model repository.

## Prerequisites

* Hugging Face account
* (Option A) Baked artifacts: copy `models/champion_model.pkl`, `models/preprocessor.pkl`, `artifacts/champion_meta.json`
* (Option B) Remote artifacts: Publish a model repo containing at least `champion_model.pkl`, `preprocessor.pkl`, `champion_meta.json` and set `HF_MODEL_REPO`
* (Optional) SHAP / interpretability artifacts for richer `/model/interpretability`

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

Set in Space settings (Variables & secrets). Only `HF_MODEL_REPO` is needed for remote model loading; otherwise artifacts must be baked into `models/`.

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_MODEL_REPO` | Optional | Remote model repo id (`namespace/repo`) for dynamic snapshot download |
| `FORCE_HF_LOAD` | Optional | Force re-download even if baked artifacts exist (e.g. `true`) |
| `HF_HUB_CACHE` | Optional | Override hub cache path (e.g. `/home/user/.cache/hf`) |
| `DECISION_THRESHOLD` | Optional | Probability threshold override |
| `ALLOW_START_WITHOUT_MODEL` | Optional | Allow API to start while model unavailable |

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

Notes:
1. Hugging Face Spaces expose port 7860.
2. When using remote artifacts ensure the container user has write permission to the HF cache directory.
3. For faster cold starts bake artifacts directly in the image instead of remote loading.

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
2. Optionally push artifacts to HF: `python scripts/push_to_hf.py --repo <username>/<repo>`
3. Trigger Space rebuild (commit any change or restart in UI)
4. If using `FORCE_HF_LOAD=true`, revert to `false` after confirming new snapshot

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| 503 model_not_loaded | Missing `HF_MODEL_REPO` | Add environment variable |
| Permission denied: 'models' | Missing models directory in Docker | Ensure Dockerfile creates `models/` before switching to non-root user |
| InconsistentVersionWarning | Version mismatch | Pin `scikit-learn==1.7.2` |
| 500 preprocessor error | Code drift | Align preprocessing logic |
| Slow first request | Cold start | Normal; subsequent requests faster |

## Security

* Never commit secrets. This project has removed prior AWS / S3 variables; do not reintroduce them.
* Use Space Secrets for sensitive values if future auth is added.
* Rotate any historically exposed credentials externally—removal from repository history does not invalidate them.

## Local Testing

Test locally before deploying:

```bash
docker build -t hotel-cancel .
docker run -p 8000:7860 -e HF_MODEL_REPO=<username>/hotel-cancel-champion hotel-cancel
curl localhost:8000/health
```

## Alternative Deployment

Any Docker-capable platform (Kubernetes, Cloud Run, Azure App Service, Fly.io, Render, etc.) works. Key points:
1. Artifacts strategy: bake vs remote HF snapshot
2. Health probes: use `/health` (returns `model_not_loaded` until artifacts are ready)
3. Config via environment variables (see table above)
4. Scale: ensure sufficient memory for model + preprocessing pipeline (typically < 512MB for current baseline)

For more Space-specific guidance see `HUGGINGFACE_DEPLOYMENT.md`.
