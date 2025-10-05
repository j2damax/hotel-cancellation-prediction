"""Exploratory fairness / subgroup performance analysis.

Produces:
  artifacts/fairness_group_metrics.json
  artifacts/fairness_summary.md

Current heuristic grouping (adjust based on available engineered features):
  - lead_time buckets
  - is_repeated_guest
  - total_of_special_requests buckets

Assumptions:
  - Training data (post feature engineering) is available at data/processed/hotel_booking_preprocessed.csv
  - Target column is 'is_canceled' (0/1)
  - Champion model + preprocessor at models/champion_model.pkl / models/preprocessor.pkl

Note: This is an exploratory diagnostic, not a formal bias audit. It surfaces disparate precision/recall across simple segments.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/processed/hotel_booking_preprocessed.csv")
MODEL_PATH = Path("models/champion_model.pkl")
PREPROC_PATH = Path("models/preprocessor.pkl")
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)


def bucket_lead_time(x: float) -> str:
    if x < 30:
        return "LT_<30"
    if x < 90:
        return "LT_30_89"
    if x < 180:
        return "LT_90_179"
    return "LT_180+"


def bucket_special(req: float) -> str:
    if req == 0:
        return "SR_0"
    if req == 1:
        return "SR_1"
    if req <= 3:
        return "SR_2_3"
    return "SR_4+"


def load():
    if not DATA_PATH.exists():
        raise SystemExit("Processed dataset not found; run preprocessing pipeline.")
    df = pd.read_csv(DATA_PATH)
    if 'is_canceled' not in df.columns:
        raise SystemExit("Expected target column 'is_canceled' missing.")
    model = joblib.load(MODEL_PATH)
    preproc_obj = joblib.load(PREPROC_PATH)
    return df, model, preproc_obj


def compute_group_metrics(df: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> pd.DataFrame:
    y_pred = (y_prob >= threshold).astype(int)

    groups = {
        'lead_time_bucket': df['lead_time_bucket'],
        'is_repeated_guest': df['is_repeated_guest'].astype(str),
        'special_requests_bucket': df['special_requests_bucket'],
    }

    rows = []
    for gname, series in groups.items():
        for val, subset_idx in series.groupby(series).groups.items():
            idx = list(subset_idx)
            yt = y_true[idx]
            yp = y_pred[idx]
            if len(yt) == 0:
                continue
            prec = precision_score(yt, yp, zero_division=0)
            rec = recall_score(yt, yp, zero_division=0)
            f1 = f1_score(yt, yp, zero_division=0)
            support = len(yt)
            pos_rate = yt.mean()
            rows.append({
                'group': gname,
                'value': val,
                'support': support,
                'positive_rate': round(float(pos_rate), 4),
                'precision': round(float(prec), 4),
                'recall': round(float(rec), 4),
                'f1': round(float(f1), 4)
            })
    return pd.DataFrame(rows)


def main():
    df, model, preproc_obj = load()

    # Prepare buckets (avoid modifying original target leakage columns if any)
    df['lead_time_bucket'] = df['lead_time'].apply(bucket_lead_time)
    if 'total_of_special_requests' in df.columns:
        df['special_requests_bucket'] = df['total_of_special_requests'].apply(bucket_special)
    else:
        df['special_requests_bucket'] = 'UNKNOWN'

    feature_cols = [c for c in df.columns if c not in {'is_canceled'}]
    X = df[feature_cols]
    y = df['is_canceled'].values

    # Minimal reconstruction: assume numeric features already scaled via stored scaler if available
    scaler: StandardScaler | None = None
    if isinstance(preproc_obj, dict) and 'scaler' in preproc_obj and isinstance(preproc_obj['scaler'], StandardScaler):
        scaler = preproc_obj['scaler']

    # Use only numeric columns for fairness diagnostic to avoid full encoding rebuild
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X_num = X[num_cols].fillna(0)
    if scaler is not None:
        try:
            X_proc = scaler.transform(X_num)
        except Exception:
            # Fallback if feature name mismatch (since original pipeline used encoded features)
            X_proc = X_num.to_numpy(dtype=float)
    else:
        X_proc = X_num.to_numpy(dtype=float)
    # Model may expose predict_proba or decision_function
    try:
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_proc)[:, 1]
        elif hasattr(model, 'decision_function'):
            logits = model.decision_function(X_proc)
            y_prob = 1 / (1 + np.exp(-logits))
        else:
            raise RuntimeError("Model does not support probability outputs.")
    except Exception as e:
        placeholder = {
            'error': 'feature_shape_mismatch',
            'message': str(e),
            'note': 'Full encoded feature space not reconstructed; run full pipeline-based fairness later.'
        }
        (ARTIFACTS / 'fairness_group_metrics.json').write_text(json.dumps(placeholder, indent=2))
        (ARTIFACTS / 'fairness_summary.md').write_text('# Fairness Analysis\nPipeline feature mismatch prevented scoring. Run full encoding-aware fairness module in future.')
        print('Fairness analysis skipped due to feature mismatch.')
        return

    # Load threshold if available
    threshold = 0.5
    champ_meta = ARTIFACTS / 'champion_meta.json'
    if champ_meta.exists():
        meta = json.loads(champ_meta.read_text())
        threshold = float(meta.get('decision_threshold', 0.5))

    group_df = compute_group_metrics(df, y, y_prob, threshold)
    group_df.to_json(ARTIFACTS / 'fairness_group_metrics.json', orient='records', indent=2)

    # Summary markdown
    piv = group_df.pivot_table(index=['group', 'value'], values=['support', 'positive_rate', 'precision', 'recall', 'f1'])
    md_lines = ["# Fairness / Subgroup Performance (Exploratory)", f"Threshold used: {threshold:.2f}", "", "| Group | Value | Support | Pos Rate | Precision | Recall | F1 |", "|-------|-------|---------|----------|-----------|--------|----|"]
    for (g, v), row in piv.iterrows():
        md_lines.append(f"| {g} | {v} | {int(row['support'])} | {row['positive_rate']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |")
    md_lines.append("\nNotes: Metrics are descriptive; no statistical parity or equalized odds tests applied yet.")
    (ARTIFACTS / 'fairness_summary.md').write_text("\n".join(md_lines))

    print("Fairness analysis complete: artifacts/fairness_group_metrics.json, artifacts/fairness_summary.md")


if __name__ == '__main__':  # pragma: no cover
    main()
