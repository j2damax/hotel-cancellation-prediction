#!/usr/bin/env python
"""
Uploads required:
    * models/champion_model.pkl
    * models/preprocessor.pkl
    * artifacts/champion_meta.json

Optionally (if present):
    * artifacts/feature_importance.json (top features)
    * artifacts/shap_values_sample.json (sample SHAP rows)
    * artifacts/feature_name_map.json

Generates (or overwrites with --force-readme) a concise README model card
using metrics in champion_meta.json. Designed for tabular hospitality
cancellation prediction (Sri Lankan tourism context).

Usage:
    python scripts/push_to_hf.py --repo-id <user/model> [--private] \
             [--commit-message "Update"] [--force-readme]

Authentication: ensure `huggingface-cli login` (or set HF_TOKEN env var).
"""

from __future__ import annotations
import argparse
import os
import json
from pathlib import Path
from datetime import datetime, UTC
from huggingface_hub import HfApi, create_repo, upload_file
from typing import Any, Dict, List

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


def _safe(val: Any) -> Any:
    return val if isinstance(val, (int, float)) else "n/a"


def build_model_card(repo_id: str, meta: Dict[str, Any], top_features: List[str]) -> str:
    holdout = meta.get("holdout_metrics", {})
    agg = meta.get("aggregate", {})
    model_name = meta.get("model_name", "Champion Model")
    threshold = meta.get("decision_threshold")
    metrics = {
        "f1": holdout.get("f1_score") or agg.get("f1_score_mean"),
        "roc_auc": holdout.get("roc_auc") or agg.get("roc_auc_mean"),
        "precision": holdout.get("precision") or agg.get("precision_mean"),
        "recall": holdout.get("recall") or agg.get("recall_mean"),
        "accuracy": holdout.get("accuracy") or agg.get("accuracy_mean"),
    }
    metrics_yaml = []
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            metrics_yaml.append(f"    - type: {k}\n      value: {v}")
    top_feat_md = ("\n".join([f"- {f}" for f in top_features])) if top_features else "(Not available)"
    usage_block = (
        "```python\n"
        "from huggingface_hub import snapshot_download\n"
        "import joblib, json, pandas as pd\n\n"
        f"local_dir = snapshot_download(repo_id=\"{repo_id}\")\n"
        "model = joblib.load(f\"{local_dir}/champion_model.pkl\")\n"
        "preprocessor = joblib.load(f\"{local_dir}/preprocessor.pkl\")\n"
        "meta = json.load(open(f\"{local_dir}/champion_meta.json\"))\n\n"
        "sample = pd.DataFrame([{\n"
        "    'lead_time': 45, 'arrival_month': 7, 'adults': 2, 'children': 0, 'adr': 110.0\n"
        "}])\n\n"
        "X = preprocessor.transform(sample)\n"
        "proba = float(model.predict_proba(X)[:,1][0])\n"
        "print('Cancellation probability:', round(proba, 4))\n"
        "```\n"
    )
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    front = [
        "---",
        "language: en",
        "license: mit",
        "tags:",
        "- tabular-classification",
        "- hospitality",
        "- cancellations",
        "- sri-lanka",
        "- mlops",
        "- shap",
        "model-index:",
        "- name: hotel-cancellation-predictor",
        "  results:",
        "  - task:",
        "      type: tabular-classification",
        "      name: Hotel Booking Cancellation",
    ]
    if metrics_yaml:
        front.append("    metrics:")
        front.extend(metrics_yaml)
    front.append("---\n")
    body = f"""# Hotel Booking Cancellation Predictor\n\nPredicts probability that a hotel booking will be cancelled (Sri Lankan hospitality context). The champion model is **{model_name}**; threshold based decisions currently use **{threshold}** (see `champion_meta.json`).\n\n_Last updated: {timestamp}_\n\n## Key Metrics (Holdout)\n| Metric | Value |\n|--------|-------|\n| F1 | {_safe(metrics['f1'])} |\n| ROC-AUC | {_safe(metrics['roc_auc'])} |\n| Precision | {_safe(metrics['precision'])} |\n| Recall | {_safe(metrics['recall'])} |\n| Accuracy | {_safe(metrics['accuracy'])} |\n\n## Top Features (SHAP importance)\n{top_feat_md}\n\n## Quickstart\n{usage_block}## Files\n- `champion_model.pkl` – serialized champion estimator\n- `preprocessor.pkl` – unified preprocessing / feature pipeline\n- `champion_meta.json` – metrics & threshold\n- Optional SHAP / feature importance JSON artifacts\n\n## Notes\nModel trained with stratified 5-fold CV; primary selection metric: F1; tie-breaker: ROC-AUC. Class imbalance handled via class weights.\n\n## Citation\nAcademic coursework (NIB 7072) — Sri Lankan tourism cancellation risk analysis.\n"""
    return "\n".join(front) + body


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
    parser.add_argument("--private", action="store_true", help="Create repo as private if new")
    parser.add_argument("--commit-message", default="Update champion model")
    parser.add_argument("--force-readme", action="store_true", help="Overwrite existing README.md")
    parser.add_argument("--skip-optional", action="store_true", help="Do not attempt to upload optional explainability artifacts")
    args = parser.parse_args()

    ensure_repo(args.repo_id, args.private)

    # Load champion meta early for model card
    meta_path = Path("artifacts/champion_meta.json")
    if not meta_path.exists():
        raise SystemExit("champion_meta.json missing; run training before push")
    meta = json.loads(meta_path.read_text())

    # Validate & upload required artifacts
    for local, target in CORE_MODEL_FILES:
        if not Path(local).exists():
            raise SystemExit(f"Required artifact missing: {local}")
    for local, target in CORE_MODEL_FILES:
        upload(local, target, args.repo_id, args.commit_message)

    # Optional uploads
    top_features: List[str] = []
    if not args.skip_optional:
        for local, target in OPTIONAL_FILES:
            if Path(local).exists():
                maybe_upload_optional(local, target, args.repo_id, args.commit_message)
        fi_path = Path("artifacts/feature_importance.json")
        if fi_path.exists():
            try:
                raw_fi = json.loads(fi_path.read_text())
                # Accept either list[dict{name,importance}] or list of tuples; adapt generically
                for item in raw_fi[:10]:
                    if isinstance(item, dict):
                        # possible keys: feature / name
                        fname = item.get("feature") or item.get("name") or next(iter(item.keys()))
                        top_features.append(str(fname))
                    else:
                        top_features.append(str(item[0]))
            except Exception:
                pass

    # Determine if README exists
    api = HfApi()
    existing_files = set(api.list_repo_files(args.repo_id))
    write_readme = args.force_readme or ("README.md" not in existing_files)
    if write_readme:
        card = build_model_card(args.repo_id, meta, top_features)
        tmp_path = Path("/tmp/README.md")
        tmp_path.write_text(card)
        upload_file(
            path_or_fileobj=str(tmp_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            commit_message=args.commit_message,
        )
        print("Model card written (created/overwritten).")
    else:
        print("Existing README.md preserved (use --force-readme to overwrite).")

    print("Push complete: required artifacts + model card handled.")


if __name__ == "__main__":
    main()
