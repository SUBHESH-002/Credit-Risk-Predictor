"""
src/evaluate.py
----------------
Evaluation module for TabTransformer experiments on German Credit.

Computes for each trained model:
  - Accuracy
  - Precision, Recall, F1 per class
  - Macro-F1, Macro-Recall
  - F2 score (beta=2, as discussed in the paper)
  - Confusion matrix
  - ROC curve + AUC (binary)

Saves:
  - outputs/metrics/results.csv          - all metrics in one table
  - outputs/plots/confusion_matrix_expX.png
  - outputs/plots/roc_curve_expX.png
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data import CATEGORICAL_COLS, NUMERICAL_COLS, load_german_credit
from model import DEVICE, build_tabtransformer
from preprocessing import GermanCreditPreprocessor, run_preprocessing

MODELS_DIR  = Path(__file__).parent.parent / "outputs" / "models"
PLOTS_DIR   = Path(__file__).parent.parent / "outputs" / "plots"
METRICS_DIR = Path(__file__).parent.parent / "outputs" / "metrics"


# --- Prediction helpers -------------------------------------------------------

def load_checkpoint(experiment: str, cardinalities: list) -> torch.nn.Module:
    """Load a saved TabTransformer checkpoint."""
    ckpt_path = str(MODELS_DIR / f"tabtransformer_exp{experiment}.pt")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run train.py first.")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = build_tabtransformer(cardinalities)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[evaluate.py] Loaded Exp {experiment} from epoch {ckpt['epoch']}")
    return model


@torch.no_grad()
def predict(model, X, batch_size=256):
    """Return predicted class indices and class probabilities."""
    model.eval()
    all_preds, all_probs = [], []
    n = len(X)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = X.iloc[start:end]
        cat_t = torch.tensor(batch[CATEGORICAL_COLS].values.astype(int),
                              dtype=torch.long, device=DEVICE)
        num_t = torch.tensor(batch[NUMERICAL_COLS].values.astype(float),
                              dtype=torch.float32, device=DEVICE)
        logits = model(cat_t, num_t)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_probs)


# --- Metrics ------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_probs, experiment: str, split: str = "test") -> dict:
    """
    Compute all required metrics and return as a flat dict.
    """
    from sklearn.metrics import (
        accuracy_score, confusion_matrix,
        f1_score, fbeta_score, precision_score,
        recall_score, roc_auc_score,
    )

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    rec  = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    f1   = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    macro_f1  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f2        = fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0)

    # ROC AUC - use probability of class 1 (good credit)
    auc = roc_auc_score(y_true, y_probs[:, 1])

    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "experiment": experiment,
        "split": split,
        "accuracy": round(acc, 4),
        "precision_class0": round(prec[0], 4),
        "precision_class1": round(prec[1], 4),
        "recall_class0": round(rec[0], 4),
        "recall_class1": round(rec[1], 4),
        "f1_class0": round(f1[0], 4),
        "f1_class1": round(f1[1], 4),
        "macro_f1": round(macro_f1, 4),
        "macro_recall": round(macro_rec, 4),
        "f2_score": round(f2, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
    }
    return metrics


# --- Plots --------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, experiment: str, split: str = "test"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Bad (0)", "Good (1)"],
                    yticklabels=["Bad (0)", "Good (1)"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - Exp {experiment} ({split})")
        fig.tight_layout()
        os.makedirs(str(PLOTS_DIR), exist_ok=True)
        out = str(PLOTS_DIR / f"confusion_matrix_exp{experiment}_{split}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Confusion matrix saved: {out}")
    except Exception as e:
        print(f"  [warn] Confusion matrix plot failed: {e}")


def plot_roc_curve(y_true, y_probs, experiment: str, split: str = "test"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, roc_auc_score

        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        auc = roc_auc_score(y_true, y_probs[:, 1])

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - Exp {experiment} ({split})")
        ax.legend(); fig.tight_layout()
        os.makedirs(str(PLOTS_DIR), exist_ok=True)
        out = str(PLOTS_DIR / f"roc_curve_exp{experiment}_{split}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  ROC curve saved: {out}")
    except Exception as e:
        print(f"  [warn] ROC curve plot failed: {e}")


def plot_all_roc_curves(results_by_exp: dict, split: str = "test"):
    """Overlay ROC curves for all experiments on one chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, roc_auc_score

        fig, ax = plt.subplots(figsize=(6, 5))
        for exp, data in results_by_exp.items():
            fpr, tpr, _ = roc_curve(data["y_true"], data["y_probs"][:, 1])
            auc = roc_auc_score(data["y_true"], data["y_probs"][:, 1])
            ax.plot(fpr, tpr, label=f"Exp {exp} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC Curves Comparison ({split})")
        ax.legend(); fig.tight_layout()
        os.makedirs(str(PLOTS_DIR), exist_ok=True)
        out = str(PLOTS_DIR / f"roc_curves_all_{split}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Combined ROC curves saved: {out}")
    except Exception as e:
        print(f"  [warn] Combined ROC plot failed: {e}")


# --- Main evaluation ----------------------------------------------------------

def evaluate_all_experiments(result: dict = None) -> pd.DataFrame:
    """
    Evaluate all three experiments and save metrics + plots.

    Returns a DataFrame with all metrics.
    """
    if result is None:
        result = run_preprocessing(save=False)

    X_test  = result["X_test"]
    y_test  = result["y_test"]
    cardinalities = result["cardinalities"]

    all_metrics    = []
    roc_data       = {}

    for experiment in ["A", "B", "C"]:
        print(f"\n--- Evaluating Experiment {experiment} ---")
        try:
            model = load_checkpoint(experiment, cardinalities)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue

        y_pred, y_probs = predict(model, X_test)
        y_true = y_test.values

        metrics = compute_metrics(y_true, y_pred, y_probs, experiment, split="test")
        all_metrics.append(metrics)
        roc_data[experiment] = {"y_true": y_true, "y_probs": y_probs}

        # Print summary
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Macro-F1 : {metrics['macro_f1']:.4f}")
        print(f"  F2 Score : {metrics['f2_score']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"  Confusion Matrix: {metrics['confusion_matrix']}")

        # Plots
        plot_confusion_matrix(y_true, y_pred, experiment, "test")
        plot_roc_curve(y_true, y_probs, experiment, "test")

    # Combined ROC
    if roc_data:
        plot_all_roc_curves(roc_data, "test")

    # Save results CSV
    df = pd.DataFrame(all_metrics)
    os.makedirs(str(METRICS_DIR), exist_ok=True)
    results_path = str(METRICS_DIR / "results.csv")
    df.to_csv(results_path, index=False)
    print(f"\n[evaluate.py] Metrics saved to {results_path}")
    print(df[["experiment", "accuracy", "macro_f1", "f2_score", "roc_auc"]].to_string(index=False))

    return df


if __name__ == "__main__":
    evaluate_all_experiments()
