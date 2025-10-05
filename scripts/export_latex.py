"""Export LaTeX tables for academic report from existing artifact JSON/CSV files.

Generates:
  report/latex/cv_metrics_table.tex
  report/latex/holdout_metrics_table.tex
  report/latex/threshold_table.tex
  report/latex/feature_importance_table.tex

Assumptions:
  - artifacts/cv_metrics.json contains per-model aggregate metrics
  - artifacts/champion_meta.json contains holdout + threshold metrics
  - artifacts/threshold_sweep.csv contains threshold sweep data
  - artifacts/feature_importance.json contains SHAP mean abs values
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, Any

ARTIFACTS = Path("artifacts")
LATEX_DIR = Path("report/latex")
LATEX_DIR.mkdir(parents=True, exist_ok=True)


def _latex_escape(s: str) -> str:
    return s.replace('_', '\\_')


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def export_cv_metrics(cv_metrics: Dict[str, Any]):
    models = cv_metrics.get("models", {})
    header = ["Model", "Accuracy (µ±σ)", "Precision (µ±σ)", "Recall (µ±σ)", "F1 (µ±σ)", "ROC-AUC (µ±σ)"]
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
    " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for name, data in models.items():
        acc = f"{data['accuracy_mean']:.4f} $\\pm$ {data['accuracy_std']:.4f}"
        prec = f"{data['precision_mean']:.4f} $\\pm$ {data['precision_std']:.4f}"
        rec = f"{data['recall_mean']:.4f} $\\pm$ {data['recall_std']:.4f}"
        f1 = f"{data['f1_score_mean']:.4f} $\\pm$ {data['f1_score_std']:.4f}"
        roc = f"{data['roc_auc_mean']:.4f} $\\pm$ {data['roc_auc_std']:.4f}"
        lines.append(f"{_latex_escape(name)} & {acc} & {prec} & {rec} & {f1} & {roc} \\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\caption{Cross-validation aggregate metrics (mean $\\pm$ std across folds).}", "\\label{tab:cv-metrics}", "\\end{table}"]
    (LATEX_DIR / "cv_metrics_table.tex").write_text("\n".join(lines))


def export_holdout(champion_meta: Dict[str, Any]):
    hold = champion_meta.get("holdout_metrics", {})
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
    "Metric & Accuracy & Precision & Recall & F1 & ROC-AUC \\\\",
        "\\midrule",
        f"Holdout & {hold.get('accuracy', float('nan')):.4f} & {hold.get('precision', float('nan')):.4f} & {hold.get('recall', float('nan')):.4f} & {hold.get('f1_score', float('nan')):.4f} & {hold.get('roc_auc', float('nan')):.4f} \\",
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Champion holdout performance (model: {champion_meta.get('model_name')}).}}",
        "\\label{tab:holdout-metrics}",
        "\\end{table}",
    ]
    (LATEX_DIR / "holdout_metrics_table.tex").write_text("\n".join(lines))


def export_threshold(champion_meta: Dict[str, Any]):
    thresh_metrics = champion_meta.get("decision_threshold_metrics", {})
    threshold = champion_meta.get("decision_threshold")
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
    "Threshold & Precision & Recall & F1 & Note \\\\",
        "\\midrule",
        f"{threshold:.2f} & {thresh_metrics.get('precision', float('nan')):.4f} & {thresh_metrics.get('recall', float('nan')):.4f} & {thresh_metrics.get('f1_score', float('nan')):.4f} & F1-optimal \\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Selected operating threshold and associated metrics.}",
        "\\label{tab:threshold}",
        "\\end{table}",
    ]
    (LATEX_DIR / "threshold_table.tex").write_text("\n".join(lines))


def export_feature_importance(feature_importance, top_n: int = 15):
    # Accept either dict {feature: value} or list of {"feature": name, "importance": val}
    if isinstance(feature_importance, list):
        pairs = []
        for item in feature_importance:
            if isinstance(item, dict):
                # attempt generic key retrieval
                if 'feature' in item and 'importance' in item:
                    pairs.append((item['feature'], item['importance']))
                else:
                    # fall back: first two values
                    keys = list(item.keys())
                    if len(keys) >= 2:
                        pairs.append((item[keys[0]], float(item[keys[1]])))
        ordered = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_n]
    else:
        ordered = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcc}",
        "\\toprule",
    "Rank & Feature & Mean $|$SHAP$|$ \\\\",
        "\\midrule",
    ]
    for i, (feat, val) in enumerate(ordered, 1):
        lines.append(f"{i} & {_latex_escape(feat)} & {val:.6f} \\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Top feature attributions by mean absolute SHAP value (top 15).}",
        "\\label{tab:feature-importance}",
        "\\end{table}",
    ]
    (LATEX_DIR / "feature_importance_table.tex").write_text("\n".join(lines))


def main():
    cv_path = ARTIFACTS / "cv_metrics.json"
    champion_path = ARTIFACTS / "champion_meta.json"
    fi_path = ARTIFACTS / "feature_importance.json"

    if not cv_path.exists() or not champion_path.exists():
        raise SystemExit("Required artifact files not found. Run training first.")

    cv_metrics = load_json(cv_path)
    champion_meta = load_json(champion_path)
    feature_importance = load_json(fi_path) if fi_path.exists() else {}

    export_cv_metrics(cv_metrics)
    export_holdout(champion_meta)
    export_threshold(champion_meta)
    if feature_importance:
        export_feature_importance(feature_importance)

    print("LaTeX tables exported to report/latex/")


if __name__ == "__main__":  # pragma: no cover
    main()
