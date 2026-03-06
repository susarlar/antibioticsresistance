# src/plotting.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def plot_cv_bars(
    comparison_csv: str = "reports/model_comparison.csv",
    out_dir: str = "reports/figures",
) -> None:
    """
    Plots:
      - Mean CV AUC (with std error bars)
      - Mean CV Accuracy (with std error bars)
    Uses data produced by src.train (reports/model_comparison.csv).
    """
    _ensure_dir(out_dir)

    df = pd.read_csv(comparison_csv)
    if df.empty:
        raise ValueError(f"No rows found in {comparison_csv}")

    df = df.sort_values("cv_auc_mean", ascending=True)  # nicer horizontal bars
    models = df["model"].tolist()

    # AUC
    plt.figure()
    plt.barh(models, df["cv_auc_mean"].values, xerr=df["cv_auc_std"].values)
    plt.xlabel("CV ROC-AUC (mean ± std)")
    plt.ylabel("Model")
    plt.title("10-fold CV ROC-AUC by Model")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_auc.png"), dpi=200)
    plt.close()

    # Accuracy
    plt.figure()
    plt.barh(models, df["cv_acc_mean"].values, xerr=df["cv_acc_std"].values)
    plt.xlabel("CV Accuracy (mean ± std)")
    plt.ylabel("Model")
    plt.title("10-fold CV Accuracy by Model")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_accuracy.png"), dpi=200)
    plt.close()

def plot_confusion_matrix_from_metadata(
    metadata_json: str = "model_artifacts/metadata.json",
    out_dir: str = "reports/figures",
) -> None:
    """
    Plots confusion matrix using model_artifacts/metadata.json output by training.
    """
    _ensure_dir(out_dir)

    with open(metadata_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cm = np.array(meta["confusion_matrix"], dtype=int)
    if cm.shape != (2, 2):
        raise ValueError("Expected 2x2 confusion matrix in metadata.json")

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Hold-out Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0, 1], ["Non-Resistant (0)", "Resistant (1)"])
    plt.yticks([0, 1], ["Non-Resistant (0)", "Resistant (1)"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

def plot_roc_curve_from_saved_probs(
    y_true_path: str,
    y_proba_path: str,
    out_dir: str = "reports/figures",
    title: str = "ROC Curve (Hold-out Test Set)",
) -> None:
    """
    Optional: If you save y_true and y_proba arrays from test-time evaluation,
    this will plot an ROC curve.

    Expected file formats:
      - y_true_path: CSV with one column (0/1)
      - y_proba_path: CSV with one column (probabilities for class 1)
    """
    _ensure_dir(out_dir)

    y_true = pd.read_csv(y_true_path).iloc[:, 0].astype(int).values
    y_proba = pd.read_csv(y_proba_path).iloc[:, 0].astype(float).values

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=200)
    plt.close()

def main():
    # Basic plots that should always work:
    plot_cv_bars()
    # Confusion matrix requires metadata.json to exist:
    if os.path.exists("model_artifacts/metadata.json"):
        plot_confusion_matrix_from_metadata()

if __name__ == "__main__":
    main()
