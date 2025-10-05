# LaTeX Export

Generate LaTeX tables for academic reports from training artifacts.

## Usage

```bash
python scripts/export_latex.py
```

## Generated Files

- `cv_metrics_table.tex` - Cross-validation metrics
- `holdout_metrics_table.tex` - Holdout performance
- `threshold_table.tex` - Operating threshold metrics
- `feature_importance_table.tex` - SHAP feature importance

## Requirements

Training artifacts must exist:
- `artifacts/cv_metrics.json`
- `artifacts/champion_meta.json`
- `artifacts/feature_importance.json`

Run training first: `python scripts/train.py --cv-folds 5`
