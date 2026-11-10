"""
src/explain.py
---------------
SHAP-based explainability for the trained TabTransformer.

Implements:
  1. A prediction wrapper that accepts original feature columns and applies
     the saved preprocessing internally before calling the model.
  2. A model-agnostic SHAP explainer (KernelExplainer) using a small
     background sample from training data.
  3. Global feature importance (mean |SHAP|) chart.
  4. Individual prediction explanations (waterfall plots) for a few held-out records.

Paper reference targets for German Credit:
  Top features: Credit amount, Age, Existing credit status

NOTE: SHAP values describe the model's behavior and do not prove causation.
The background sample and the class probability being explained are documented below.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from data import CATEGORICAL_COLS, NUMERICAL_COLS, load_german_credit
from model import DEVICE, build_tabtransformer
from preprocessing import GermanCreditPreprocessor, run_preprocessing

MODELS_DIR = Path(__file__).parent.parent / "outputs" / "models"
SHAP_DIR   = Path(__file__).parent.parent / "outputs" / "plots" / "shap"

BACKGROUND_SIZE   = 50    # size of SHAP background sample from training data
N_EXPLAIN_SAMPLES = 25    # number of test records to explain
N_INDIVIDUAL_PLOTS = 3    # number of individual waterfall plots to save
BEST_EXPERIMENT   = "B"   # experiment whose model is explained (paper's main model)


# --- Prediction wrapper -------------------------------------------------------

class TabTransformerPredictor:
    """
    Prediction wrapper that:
      - Accepts a 2-D numpy array of ORIGINAL feature order (cat + num mixed)
      - Applies the saved preprocessing internally
      - Returns class probabilities as a numpy array (N, 2)

    This is the interface expected by shap.KernelExplainer.
    """

    ALL_COLS = CATEGORICAL_COLS + NUMERICAL_COLS

    def __init__(self, model: torch.nn.Module, preprocessor: GermanCreditPreprocessor):
        self.model        = model
        self.preprocessor = preprocessor
        self.model.eval()

    def __call__(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X_raw : np.ndarray, shape (N, n_features)
            Raw (original) feature values in the order ALL_COLS.

        Returns
        -------
        probs : np.ndarray, shape (N, 2)
        """
        df = pd.DataFrame(X_raw, columns=self.ALL_COLS)

        # Restore original dtypes for categoricals
        for col in CATEGORICAL_COLS:
            df[col] = df[col].astype(str)

        df_processed = self.preprocessor.transform(df)

        cat_t = torch.tensor(
            df_processed[CATEGORICAL_COLS].values.astype(int),
            dtype=torch.long, device=DEVICE,
        )
        num_t = torch.tensor(
            df_processed[NUMERICAL_COLS].values.astype(float),
            dtype=torch.float32, device=DEVICE,
        )

        with torch.no_grad():
            logits = self.model(cat_t, num_t)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

        return probs


# --- SHAP analysis ------------------------------------------------------------

def run_shap_analysis(
    result: dict = None,
    experiment: str = BEST_EXPERIMENT,
) -> None:
    """
    Full SHAP analysis pipeline.

    1. Loads the trained model checkpoint for the given experiment.
    2. Builds the prediction wrapper.
    3. Creates a KernelExplainer on a background sample from training data.
    4. Explains a subset of test records.
    5. Saves global importance chart and individual waterfall plots.
    """
    import shap

    if result is None:
        result = run_preprocessing(save=False)

    X_train = result["X_train"]
    X_test  = result["X_test"]
    y_test  = result["y_test"]
    cardinalities = result["cardinalities"]
    preprocessor  = result["preprocessor"]

    print(f"\n[explain.py] Loading model for Experiment {experiment}...")
    ckpt_path = str(MODELS_DIR / f"tabtransformer_exp{experiment}.pt")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Run train.py first."
        )
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = build_tabtransformer(cardinalities)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    predictor = TabTransformerPredictor(model, preprocessor)
    ALL_COLS  = TabTransformerPredictor.ALL_COLS

    # Reconstruct raw (pre-processed) data by loading original features
    X_orig, y_orig = load_german_credit()

    # Re-split to get original (non-scaled) training and test features
    from sklearn.model_selection import train_test_split
    X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
        X_orig, y_orig, test_size=0.20, stratify=y_orig, random_state=42
    )
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
        X_temp_raw, y_temp_raw, test_size=0.50, stratify=y_temp_raw, random_state=42
    )

    # Build raw arrays in ALL_COLS order
    X_train_arr = X_train_raw[ALL_COLS].values
    X_test_arr  = X_test_raw[ALL_COLS].values

    # Background sample (stratified subset of training data)
    np.random.seed(42)
    bg_idx = np.random.choice(len(X_train_arr), size=BACKGROUND_SIZE, replace=False)
    background = X_train_arr[bg_idx]

    # Explain a subset of test samples
    explain_idx = np.random.choice(len(X_test_arr), size=N_EXPLAIN_SAMPLES, replace=False)
    X_explain   = X_test_arr[explain_idx]

    print(f"[explain.py] Running KernelExplainer "
          f"(background={BACKGROUND_SIZE}, explain={N_EXPLAIN_SAMPLES})...")
    print("  [running KernelExplainer for Good-credit class probability]")

    # Explain class 1 (Good credit) probability directly
    def predict_class1(x):
        return predictor(x)[:, 1]

    explainer = shap.KernelExplainer(predict_class1, background)
    sv = explainer.shap_values(X_explain, nsamples=100)

    os.makedirs(str(SHAP_DIR), exist_ok=True)

    # -- Global feature importance ----------------------------------------------
    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": ALL_COLS,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    importance_path = str(SHAP_DIR / "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"\n[explain.py] Global feature importance (top 10):")
    print(importance_df.head(10).to_string(index=False))

    _plot_global_importance(importance_df, experiment)

    # -- Individual explanations ------------------------------------------------
    _plot_individual_explanations(
        explainer, X_explain, sv, ALL_COLS, y_test_raw.iloc[explain_idx], experiment
    )

    print(f"\n[explain.py] SHAP analysis complete. Outputs in {SHAP_DIR}")


def _plot_global_importance(importance_df: pd.DataFrame, experiment: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = importance_df.head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="steelblue")
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title(f"Global Feature Importance - Exp {experiment}\n(class 1 = Good credit)")
        fig.tight_layout()
        out = str(SHAP_DIR / f"global_importance_exp{experiment}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Global importance chart saved: {out}")
    except Exception as e:
        print(f"  [warn] Global importance plot failed: {e}")


def _plot_individual_explanations(
    explainer, X_explain, sv, feature_names, y_true, experiment
):
    """Save waterfall-style bar charts for a few individual records."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for i in range(min(N_INDIVIDUAL_PLOTS, len(X_explain))):
            shap_row = sv[i]
            sorted_idx = np.argsort(np.abs(shap_row))[::-1][:10]
            labels = [feature_names[j] for j in sorted_idx]
            values = shap_row[sorted_idx]
            colors = ["red" if v < 0 else "green" for v in values]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(labels[::-1], values[::-1], color=colors[::-1])
            ax.axvline(0, color="black", linewidth=0.8)
            true_label = "Good" if int(y_true.iloc[i]) == 1 else "Bad"
            ax.set_title(f"SHAP Explanation - Sample {i+1} (True: {true_label})\nExp {experiment}")
            ax.set_xlabel("SHAP Value (toward Good credit)")
            fig.tight_layout()
            out = str(SHAP_DIR / f"individual_exp{experiment}_sample{i+1}.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"  Individual explanation {i+1} saved: {out}")
    except Exception as e:
        print(f"  [warn] Individual explanation plot failed: {e}")


if __name__ == "__main__":
    result = run_preprocessing(save=False)
    run_shap_analysis(result, experiment=BEST_EXPERIMENT)
