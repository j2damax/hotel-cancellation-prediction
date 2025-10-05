import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS = Path('artifacts')
OUTPUT_DIR = Path('artifacts')
CV_FILE = ARTIFACTS / 'cv_metrics.json'

# Figure settings
plt.style.use('seaborn-v0_8-muted')

BAR_WIDTH = 0.35
FIGSIZE = (8, 4.5)
DPI = 150

METRICS = [
    ("f1_score_mean", "F1"),
    ("roc_auc_mean", "ROC-AUC")
]

COLORS = {
    'LogisticRegression': '#6baed6',
    'RandomForest': '#74c476',
    'XGBoost': '#fd8d3c',
    'MLP': '#9e9ac8'
}

def load_cv_metrics():
    if not CV_FILE.exists():
        raise FileNotFoundError(f"Missing {CV_FILE}. Run training with CV first.")
    data = json.loads(CV_FILE.read_text())
    return data['results']


def build_dataframe(results):
    rows = []
    for model_name, detail in results.items():
        agg = detail['aggregate']
        rows.append({
            'model': model_name,
            'accuracy': agg['accuracy_mean'],
            'precision': agg['precision_mean'],
            'recall': agg['recall_mean'],
            'f1': agg['f1_score_mean'],
            'roc_auc': agg['roc_auc_mean']
        })
    # Sort by F1 descending
    rows.sort(key=lambda r: r['f1'], reverse=True)
    return rows


def plot_comparison(rows):
    models = [r['model'] for r in rows]
    f1_values = [r['f1'] for r in rows]
    roc_values = [r['roc_auc'] for r in rows]

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    bars1 = ax.bar(x - BAR_WIDTH/2, f1_values, BAR_WIDTH, label='F1',
                   color=[COLORS.get(m, '#3182bd') for m in models])
    bars2 = ax.bar(x + BAR_WIDTH/2, roc_values, BAR_WIDTH, label='ROC-AUC',
                   color=[COLORS.get(m, '#9ecae1') for m in models], alpha=0.85)

    # Annotate bars
    for bars in (bars1, bars2):
        for b in bars:
            height = b.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(b.get_x() + b.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    champion_idx = 0  # after sorting by F1 descending
    ax.axvline(champion_idx, color='#ffeda0', linewidth=6, alpha=0.3)
    ax.text(champion_idx, max(max(f1_values), max(roc_values)) + 0.01,
            'Champion', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison (Cross-Validation)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'model_comparison.png'
    plt.tight_layout()
    fig.savefig(out_path)
    print(f"Saved {out_path}")


def main():
    results = load_cv_metrics()
    rows = build_dataframe(results)
    plot_comparison(rows)


if __name__ == '__main__':
    main()
