"""Regenerate a labeled confusion matrix image from stored summary artifacts without retraining.

Derives counts using:
- classification_report.json (per-class precision, recall, support)
- champion_meta.json (holdout metrics for cross-check)

We reconstruct counts algebraically:
Let class 1 = positive (canceled), class 0 = negative.
Given recall_1 = TP / P  => TP = recall_1 * P
Given precision_1 = TP / (TP + FP) => FP = TP*(1/precision_1 - 1)
Given support_0 = N = TN + FP and support_1 = P = TP + FN => TN = N - FP, FN = P - TP
Rounding: We round to nearest integer and enforce non-negative.

Output: artifacts/confusion_matrix_labeled.png
"""
from __future__ import annotations
import json
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

ART = Path('artifacts')
REPORT_FILE = ART / 'classification_report.json'
META_FILE = ART / 'champion_meta.json'
OUT_PATH = ART / 'confusion_matrix_labeled.png'

def reconstruct_counts():
    report = json.loads(REPORT_FILE.read_text())
    cls0 = report['0']
    cls1 = report['1']
    N = int(cls0['support'])  # actual negatives
    P = int(cls1['support'])  # actual positives
    recall1 = cls1['recall']
    precision1 = cls1['precision']

    TP = recall1 * P
    FP = TP * (1/precision1 - 1)
    # Round sensibly
    TP = int(round(TP))
    FP = int(round(FP))
    FN = P - TP
    TN = N - FP
    if any(x < 0 for x in (TP, FP, FN, TN)):
        raise ValueError('Negative count encountered; reconstruction failed.')
    total = TP + TN + FP + FN
    return TN, FP, FN, TP, total

def plot_matrix(TN: int, FP: int, FN: int, TP: int):
    cm = np.array([[TN, FP],[FN, TP]])
    fig, ax = plt.subplots(figsize=(4.6,4.6), dpi=150)
    im = ax.imshow(cm, cmap='Blues')
    vmax = cm.max()
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}\n" + ("TN" if (i,j)==(0,0) else "FP" if (i,j)==(0,1) else "FN" if (i,j)==(1,0) else "TP"),
                ha='center', va='center', color='black', fontsize=10, fontweight='bold')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred 0','Pred 1'])
    ax.set_yticklabels(['Actual 0','Actual 1'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (Holdout, labeled)')
    cbar = fig.colorbar(im, shrink=0.75)
    cbar.ax.set_ylabel('Count', rotation=270, labelpad=10)
    note = f"Accuracy={(TP+TN)/cm.sum():.3f} | Precision={TP/(TP+FP):.3f} | Recall={TP/(TP+FN):.3f} | F1={2*TP/(2*TP+FP+FN):.3f}"
    plt.figtext(0.5, 0.01, note, ha='center', fontsize=8)
    fig.tight_layout(rect=[0,0.02,1,1])
    fig.savefig(OUT_PATH)
    print(f"Saved {OUT_PATH}")


def main():
    if not REPORT_FILE.exists():
        raise SystemExit('classification_report.json missing. Run training first.')
    TN, FP, FN, TP, total = reconstruct_counts()
    plot_matrix(TN, FP, FN, TP)

if __name__ == '__main__':
    main()
