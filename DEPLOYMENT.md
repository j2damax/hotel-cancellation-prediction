# Hugging Face Space Deployment Guide

This document is the canonical deployment reference. The project now targets only a Hugging Face Space deployment (FastAPI or Docker Space). All previous cloud-specific (e.g., AWS) deployment paths have been fully removed from the repository.

## Overview

You can expose the FastAPI inference service as a public (or private) Hugging Face Space using either a plain FastAPI Space or a Docker Space. The service dynamically loads model artifacts (preprocessor + champion model + metadata) from a Hugging Face model repository you control (specified by `HF_MODEL_REPO`).

## When to Use a Space

| Use Case | Recommendation |
|----------|----------------|
| Public academic demo | FastAPI Space (quick build) |
| Needs custom system packages | Docker Space |
| Strict dependency pinning | Docker Space with explicit `requirements.txt` |
| Prototyping new features | FastAPI Space (faster iteration) |

## Prerequisites

- Hugging Face account
- A model repo containing: `champion_model.pkl`, `preprocessor.pkl`, `champion_meta.json` (pushed via your training + publish scripts)
- (Optional) Additional interpretability artifacts (feature importance, SHAP samples)

## 1. Create (or Clone) the Space

```bash
git clone https://huggingface.co/spaces/<org-or-user>/<space-name>
cd <space-name>
```

If starting fresh, create the Space in the UI (FastAPI or Docker type) then clone it locally.

## 2. Add Runtime Files

From your project root (sibling to the Space clone):

```bash
cp ../hotel-cancellation-prediction/main.py .
cp -R ../hotel-cancellation-prediction/app ./app
cp ../hotel-cancellation-prediction/requirements.txt .
```

Trim `requirements.txt` to inference essentials if desired:
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
Add `torch` only if your active champion uses the PyTorch MLP.

## 3. (Optional) Dockerfile (for Docker Space)

```dockerfile
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY main.py ./
COPY app ./app
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

Plain FastAPI Space: only `main.py` and `requirements.txt` are required.

## 4. Configure Environment Variables (Space Settings)

```
HF_MODEL_REPO=<org-or-user>/hotel-cancel-champion   # Required
# Optional
DECISION_THRESHOLD=0.42
ALLOW_START_WITHOUT_MODEL=true   # Development only
```

Resolution order for decision threshold:
1. ENV `DECISION_THRESHOLD`
2. `champion_meta.json` value
3. Default `0.5`

Artifact load order: local committed files (if any) → Hugging Face Hub (`HF_MODEL_REPO`).

## 5. Commit & Push

```bash
git add .
git commit -m "Add FastAPI inference service"
git push
```

The build log will show dependency installation and first artifact download (if needed).

## 6. Test Endpoints

```bash
curl -s https://huggingface.co/spaces/<org-or-user>/<space-name>/health
curl -s -X POST https://huggingface.co/spaces/<org-or-user>/<space-name>/predict \
  -H 'Content-Type: application/json' \
  -d '{"lead_time":30,"arrival_month":7,"adults":2,"children":0,"adr":120.0}'
```

Docker Spaces sometimes proxy through `/proxy/`; if health returns 404, try:
```
https://huggingface.co/spaces/<org-or-user>/<space-name>/proxy/health
```

## 7. Updating the Model

1. Retrain locally: `python scripts/train.py`
2. Publish artifacts: `python scripts/push_to_hf.py`
3. (If threshold changed) optionally set `DECISION_THRESHOLD` in Space settings
4. Trigger Space rebuild (restart from UI or push a no-op commit)

## 8. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| 503 model_not_loaded | `HF_MODEL_REPO` missing | Add env var in Space settings |
| InconsistentVersionWarning | Artifact sklearn > runtime | Pin `scikit-learn==1.7.2` |
| 500 during preprocess | Code drift vs. serialized pipeline | Align `app/` preprocessing code |
| Slow first response | Cold start & artifact download | Subsequent requests faster |
| PermissionError writing model | Read-only FS in Space | Loader falls back to `/tmp` automatically |

## 9. Security

- Never commit secrets. Use Space secret variables.
- Rotate any historical cloud credentials that may have been exposed (legacy AWS phase).
- Make the model repo private if artifacts should not be public.

## 10. Minimizing Image / Build Time

Remove: notebooks, `mlruns/`, large optional dependencies (e.g., `torch` if unused). Keep only inference stack. This reduces cold start time and image size.

## 11. Local Docker Run (Parity Test)

```bash
docker build -t hotel-cancel .
docker run -p 8000:7860 -e HF_MODEL_REPO=<org-or-user>/hotel-cancel-champion hotel-cancel
curl localhost:8000/health
```

## 12. Operational Notes

- Threshold source is reported via the health/metrics endpoints (if implemented).
- To test a new model version before making it public, publish to a staging repo and point a private Space at it.
- Consider adding a lightweight smoke test script that hits `/health` + `/predict` post-build.

---
Updated: 2025-10-05 (Simplified to Hugging Face Space only; removed residual cloud-specific deployment and monitoring content)

## Appendix: Operational Intents (Non-Cloud Specific)

While this repository no longer documents cloud-provider deployment steps, you can adapt the application to other platforms by:

- Building a minimal container image (see section above) and pushing to your chosen registry
- Using any container orchestrator (Kubernetes, Nomad, Fly.io, Render, etc.) to run `uvicorn main:app`
- Exposing `/health` and `/predict` HTTP endpoints via your platform's routing / ingress
- Optionally scraping a JSON `/metrics` endpoint if you add one (current example shown in README)

Key considerations if you self-host elsewhere:
1. Artifact Synchronization: either bake artifacts into the image (slower updates) or mount / pull from HF Hub at startup (current design).
2. Cold Start: first request may incur artifact download; consider a warm-up probe script post-deploy.
3. Observability: add structured logging (JSON) and ship logs to your aggregator of choice (e.g., OpenTelemetry collector).
4. Security: treat `HF_MODEL_REPO` as public info unless the repo is private; never embed secrets in the image.
5. Scaling: the FastAPI app is stateless; scale horizontally behind a load balancer; model reload endpoint (if added) should propagate via rolling restart or shared volume.

If future multi-cloud or provider-specific automation is reintroduced, it should live in a separate `deploy/` directory to avoid coupling the core ML workflow to any single platform.

---
End of Hugging Face–only deployment guide.
