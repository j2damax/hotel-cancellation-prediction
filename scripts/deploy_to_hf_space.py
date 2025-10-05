#!/usr/bin/env python
"""
Deploy the app directory to Hugging Face Spaces.

This script deploys the FastAPI application from the app/ directory to a Hugging Face Space.
It handles:
- Clearing the existing space (force update)
- Uploading all necessary files (app code, models, artifacts)
- Creating a Space-compatible configuration

Usage:
    python scripts/deploy_to_hf_space.py --space-id j2damax/boking-cancelation-api [--token YOUR_TOKEN]

Authentication: 
    - Use --token flag, or
    - Set HF_TOKEN environment variable, or
    - Run `huggingface-cli login` beforehand
"""

from __future__ import annotations
import argparse
import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from typing import List
import json


def get_latest_model_dir(models_base: Path) -> Path | None:
    """Find the most recent timestamped model directory."""
    remote_dir = models_base / "remote"
    if not remote_dir.exists():
        return None
    
    subdirs = [d for d in remote_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    
    # Sort by name (timestamp format makes this work)
    latest = sorted(subdirs, reverse=True)[0]
    return latest


def prepare_space_files(staging_dir: Path, repo_root: Path) -> None:
    """Prepare all files needed for Hugging Face Space in staging directory."""
    
    # 1. Copy app directory
    app_src = repo_root / "app"
    app_dst = staging_dir / "app"
    shutil.copytree(app_src, app_dst)
    
    # 2. Copy src directory (needed by model_loader)
    src_src = repo_root / "src"
    src_dst = staging_dir / "src"
    shutil.copytree(src_src, src_dst)
    
    # 3. Copy main.py
    shutil.copy(repo_root / "main.py", staging_dir / "main.py")
    
    # 4. Create models directory and copy model files
    models_dst = staging_dir / "models"
    models_dst.mkdir(exist_ok=True)
    
    # Find and copy the latest model artifacts
    latest_model_dir = get_latest_model_dir(repo_root / "models")
    if latest_model_dir and latest_model_dir.exists():
        for filename in ["champion_model.pkl", "preprocessor.pkl", "champion_meta.json"]:
            src_file = latest_model_dir / filename
            if src_file.exists():
                shutil.copy(src_file, models_dst / filename)
                print(f"✓ Copied {filename}")
    else:
        print("⚠ No model files found in models/remote/")
    
    # 5. Copy artifacts directory (if exists)
    artifacts_src = repo_root / "artifacts"
    if artifacts_src.exists():
        artifacts_dst = staging_dir / "artifacts"
        shutil.copytree(artifacts_src, artifacts_dst)
        print(f"✓ Copied artifacts directory")
    
    # 6. Create requirements.txt for Spaces
    requirements_content = """fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
pandas>=2.0.0
scikit-learn==1.7.2
xgboost>=2.0.0
joblib>=1.3.0
numpy>=1.24.0
python-dotenv>=1.0.0
huggingface_hub>=0.23.0
"""
    (staging_dir / "requirements.txt").write_text(requirements_content)
    print("✓ Created requirements.txt")
    
    # 7. Create README.md for the Space
    readme_content = """---
title: Hotel Booking Cancellation Prediction API
emoji: 🏨
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Hotel Booking Cancellation Prediction API

This is a FastAPI-based prediction service that estimates the probability of hotel booking cancellations.

## Features

- **POST /predict** - Predict cancellation probability for a single booking
- **GET /health** - Health check endpoint
- **GET /** - API information

## Example Usage

```python
import requests

payload = {
    "lead_time": 30,
    "arrival_month": 7,
    "adults": 2,
    "children": 0,
    "adr": 120.0
}

response = requests.post("https://huggingface.co/spaces/j2damax/boking-cancelation-api/predict", json=payload)
print(response.json())
```

## Model Information

The API uses a machine learning model trained on hotel booking data with features like:
- Lead time (days before arrival)
- Guest composition (adults, children)
- Pricing (average daily rate)
- Stay duration
- And more...

Check `/docs` for the interactive API documentation.
"""
    (staging_dir / "README.md").write_text(readme_content)
    print("✓ Created README.md")
    
    # 8. Create Dockerfile for Spaces
    dockerfile_content = """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Set environment variable for Hugging Face Spaces
ENV PORT=7860

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
"""
    (staging_dir / "Dockerfile").write_text(dockerfile_content)
    print("✓ Created Dockerfile")
    
    # 9. Create .env for production
    env_content = """# Production configuration
MODEL_PATH=models/
PREPROCESSOR_PATH=models/preprocessor.pkl
ARTIFACT_DIR=artifacts
LOCAL_MODEL_PATH=models/champion_model.pkl
LOCAL_PREPROCESSOR_PATH=models/preprocessor.pkl
ALLOW_START_WITHOUT_MODEL=false
"""
    (staging_dir / ".env").write_text(env_content)
    print("✓ Created .env")


def clear_space_repo(api: HfApi, space_id: str) -> None:
    """Delete all files from the space repository (force clear)."""
    try:
        # List all files in the space
        files = api.list_repo_files(space_id, repo_type="space")
        
        # Delete all files except .gitattributes (protected by HF)
        for file in files:
            if file != ".gitattributes":
                try:
                    api.delete_file(
                        path_in_repo=file,
                        repo_id=space_id,
                        repo_type="space",
                        commit_message="Clear space for fresh deployment"
                    )
                    print(f"✓ Deleted {file}")
                except Exception as e:
                    print(f"⚠ Could not delete {file}: {e}")
        
        print("✓ Space cleared")
    except Exception as e:
        print(f"⚠ Error clearing space: {e}")


def deploy_to_space(
    space_id: str,
    repo_root: Path,
    token: str | None = None,
    clear_cache: bool = True
) -> None:
    """Main deployment function."""
    
    # Initialize API
    api = HfApi(token=token)
    
    # Ensure space exists
    try:
        api.repo_info(space_id, repo_type="space")
        print(f"✓ Found existing space: {space_id}")
    except Exception:
        print(f"Creating new space: {space_id}")
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            token=token
        )
    
    # Clear existing files if requested
    if clear_cache:
        print("\n🧹 Clearing space...")
        clear_space_repo(api, space_id)
    
    # Create temporary staging directory
    with tempfile.TemporaryDirectory() as temp_dir:
        staging_dir = Path(temp_dir) / "space_content"
        staging_dir.mkdir(exist_ok=True)
        
        print("\n📦 Preparing files...")
        prepare_space_files(staging_dir, repo_root)
        
        # Upload everything
        print("\n🚀 Uploading to Hugging Face Space...")
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=space_id,
            repo_type="space",
            commit_message="Deploy app with models and artifacts (force update)",
        )
        
        print(f"\n✅ Deployment complete!")
        print(f"🔗 Space URL: https://huggingface.co/spaces/{space_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy app to Hugging Face Space"
    )
    parser.add_argument(
        "--space-id",
        default="j2damax/boking-cancelation-api",
        help="Hugging Face Space ID (default: j2damax/boking-cancelation-api)"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Skip clearing the space before deployment"
    )
    
    args = parser.parse_args()
    
    # Get repo root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    
    # Get token from args or environment
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("⚠ No token provided. Attempting to use cached credentials...")
        print("  Run 'huggingface-cli login' if authentication fails.")
    
    # Deploy
    deploy_to_space(
        space_id=args.space_id,
        repo_root=repo_root,
        token=token,
        clear_cache=not args.no_clear
    )


if __name__ == "__main__":
    main()
