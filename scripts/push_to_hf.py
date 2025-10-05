#!/usr/bin/env python
"""Utility script to push current champion artifacts to a Hugging Face model repository.

Usage:
  python scripts/push_to_hf.py \
      --repo-id <username/model-repo-name> \
      [--private] \
      [--commit-message "Add new champion"]

Environment:
  HF_TOKEN  (optional if you have already run `huggingface-cli login`)

It will upload the following required files:
  - champion_model.pkl
  - preprocessor.pkl
  - champion_meta.json

And optionally (if present):
  - feature_importance.json
  - shap_values_sample.json
  - feature_name_map.json

It also creates/updates a README.md model card if one does not exist.
"""

from __future__ import annotations
import argparse
import os
import json
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, create_repo, upload_file

CORE_MODEL_FILES = [
    ("models/champion_model.pkl", "champion_model.pkl"),
    ("models/preprocessor.pkl", "preprocessor.pkl"),
    ("artifacts/champion_meta.json", "champion_meta.json"),
]

OPTIONAL_FILES = [
    ("artifacts/feature_importance.json", "feature_importance.json"),
    ("artifacts/shap_values_sample.json", "shap_values_sample.json"),
    ("artifacts/feature_name_map.json", "feature_name_map.json"),
]

MODEL_CARD_TEMPLATE = """---
language: en
license: mit
tags:
    - tabular-classification
    - hospitality
    - cancellations
    - risk-prediction
model_index:
    - name: Hotel Cancellation Predictor
        results:
            - task:
                    type: tabular-classification
                    name: Hotel Booking Cancellation
                metrics:
                    - type: f1
                        value: n/a
                        name: F1 (champion)
                    - type: roc_auc
                        value: n/a
                        name: ROC-AUC (champion)
---

# Hotel Booking Cancellation Predictor

This repository hosts the champion model artifacts for predicting hotel booking cancellations.

## Contents
- `champion_model.pkl` : Serialized champion model (scikit-learn / XGBoost / PyTorch wrapped)
- `preprocessor.pkl` : Preprocessing pipeline (scaling, encoding, feature engineering)
- `champion_meta.json` : Metadata (model type, metrics, decision threshold)
- (Optional) Explainability artifacts.

## Inference Usage (Python)
```python
from huggingface_hub import snapshot_download
import joblib, json, pandas as pd

local_dir = snapshot_download(repo_id="{repo_id}")
model = joblib.load(f"{{local_dir}}/champion_model.pkl")
preprocessor = joblib.load(f"{{local_dir}}/preprocessor.pkl")
meta = json.load(open(f"{{local_dir}}/champion_meta.json"))

sample = pd.DataFrame([{{
        "lead_time": 34,
        "avg_price_per_room": 125.0,
        "no_of_special_requests": 1,
        "market_segment_type": "Online",
        "arrival_month": 7,
        "adults": 2,
        "children": 0,
        "weekend_nights": 1,
        "week_nights": 3,
        "meal_plan": "Meal Plan 1",
        "required_car_parking_space": 0,
        "repeated_guest": 0
}}])

X = preprocessor.transform(sample)
proba = model.predict_proba(X)[:,1][0]
print("Cancellation probability:", proba)
```

## Metadata
Auto-updated: {timestamp}

## Citation
Academic coursework NIB 7072 — Sri Lankan hospitality cancellation risk analysis.
"""


def ensure_repo(repo_id: str, private: bool) -> None:
    api = HfApi()
    try:
        api.repo_info(repo_id)
    except Exception:
        create_repo(repo_id=repo_id, private=private, exist_ok=True)


def upload(path_local: str, path_in_repo: str, repo_id: str, commit_message: str) -> None:
    if not Path(path_local).exists():
        raise FileNotFoundError(f"Missing required file: {path_local}")
    upload_file(
        path_or_fileobj=path_local,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        commit_message=commit_message,
    )


def maybe_upload_optional(path_local: str, path_in_repo: str, repo_id: str, commit_message: str) -> None:
    if Path(path_local).exists():
        upload_file(
            path_or_fileobj=path_local,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            commit_message=commit_message,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="<namespace>/<model_repo>")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Update champion model")
    parser.add_argument("--force-readme", action="store_true", help="Overwrite existing README.md with template")
    args = parser.parse_args()

    ensure_repo(args.repo_id, args.private)

    # Validate core files
    for local, target in CORE_MODEL_FILES:
        if not Path(local).exists():
            raise SystemExit(f"ERROR: Required artifact missing: {local}")

    # Upload required
    for local, target in CORE_MODEL_FILES:
        upload(local, target, args.repo_id, args.commit_message)

    # Upload optional
    for local, target in OPTIONAL_FILES:
        maybe_upload_optional(local, target, args.repo_id, args.commit_message)

    # Create/update model card if not present
    api = HfApi()
    files = api.list_repo_files(args.repo_id)
    from datetime import UTC
    if "README.md" not in files or args.force_readme:
        content = MODEL_CARD_TEMPLATE.format(repo_id=args.repo_id, timestamp=datetime.now(UTC).isoformat())
        tmp_readme = Path("/tmp/model_card.md")
        tmp_readme.write_text(content)
        upload_file(
            path_or_fileobj=str(tmp_readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            commit_message=args.commit_message,
        )
        print("Wrote model card README.md (created or overwritten)")
    else:
        print("Model card already exists; not overwriting.")

    print("Hugging Face model push completed.")


if __name__ == "__main__":
    main()
