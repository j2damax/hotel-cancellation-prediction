# Hugging Face Space Deployment Guide

This guide explains how to deploy the Hotel Cancellation Prediction API to Hugging Face Spaces.

## Prerequisites

1. A Hugging Face account
2. A Hugging Face token with write access
3. The `huggingface_hub` Python package installed

## Setup

### 1. Install Dependencies

```bash
pip install huggingface_hub
```

### 2. Login to Hugging Face

```bash
huggingface-cli login
```

Or set your token as an environment variable:

```bash
export HF_TOKEN=your_token_here
```

## Deployment

### Quick Deploy

Deploy to the default space (`j2damax/boking-cancelation-api`):

```bash
python scripts/deploy_to_hf_space.py
```

### Custom Space

Deploy to a different space:

```bash
python scripts/deploy_to_hf_space.py --space-id username/space-name
```

### With Token

Provide token directly:

```bash
python scripts/deploy_to_hf_space.py --token hf_your_token_here
```

### Skip Cache Clear

By default, the script clears all existing files in the space before deploying (force update). To skip this:

```bash
python scripts/deploy_to_hf_space.py --no-clear
```

## What Gets Deployed

The deployment script automatically packages and uploads:

1. **Application Code**
   - `app/` - FastAPI application modules
   - `src/` - Core preprocessing utilities
   - `main.py` - Application entry point

2. **Model Artifacts**
   - `models/champion_model.pkl` - Trained ML model
   - `models/preprocessor.pkl` - Data preprocessing pipeline
   - `models/champion_meta.json` - Model metadata

3. **Additional Artifacts**
   - `artifacts/` - Feature importance, SHAP values, metrics, etc.

4. **Configuration Files**
   - `requirements.txt` - Python dependencies
   - `Dockerfile` - Container configuration
   - `.env` - Environment variables
   - `README.md` - Space documentation

## Space Configuration

The deployment creates a Docker-based Hugging Face Space with:

- **SDK**: Docker
- **Port**: 7860 (Hugging Face Spaces default)
- **Python**: 3.10
- **Framework**: FastAPI + Uvicorn

## After Deployment

1. Navigate to your space: `https://huggingface.co/spaces/j2damax/boking-cancelation-api`
2. Wait for the build to complete (usually 2-5 minutes)
3. Access the API at the space URL
4. Interactive docs available at: `/docs`

## API Endpoints

Once deployed, the following endpoints are available:

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Make predictions
- `GET /docs` - Interactive API documentation

## Example Usage

```python
import requests

# Your space URL
SPACE_URL = "https://huggingface.co/spaces/j2damax/boking-cancelation-api"

# Make a prediction
payload = {
    "lead_time": 30,
    "arrival_month": 7,
    "adults": 2,
    "children": 0,
    "adr": 120.0
}

response = requests.post(f"{SPACE_URL}/predict", json=payload)
print(response.json())
```

## Troubleshooting

### Authentication Error
- Ensure you've logged in with `huggingface-cli login`
- Or provide token via `--token` flag or `HF_TOKEN` environment variable

### Build Failure
- Check the Space logs in the Hugging Face web interface
- Verify all model files exist in `models/remote/`
- Ensure requirements.txt dependencies are compatible

### Model Not Loading
- Check that model files were uploaded successfully
- Verify the model path configuration in `.env`
- Review startup logs in the Space logs

## Updating the Deployment

To update the space with new code or models:

```bash
# This will clear the space and deploy fresh
python scripts/deploy_to_hf_space.py
```

## Notes

- The script automatically clears the space cache before deploying (force update)
- Model files are automatically detected from `models/remote/` (latest timestamped directory)
- The deployment is fully self-contained - no external dependencies required
- Build time depends on Hugging Face infrastructure (typically 2-5 minutes)
