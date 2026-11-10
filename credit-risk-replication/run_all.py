"""
run_all.py
-----------
Master script that runs all phases end-to-end:

  Phase 1  -> Data audit
  Phase 2  -> Preprocessing
  Phase 3  -> Model sanity check
  Phase 4  -> Train unweighted baseline (Experiment A)
  Phase 5  -> Train weighted experiments (Experiments B and C)
  Phase 6  -> Evaluate all experiments
  Phase 7  -> SHAP explanations
  Phase 9  -> Final summary

Usage:
    python run_all.py
    python run_all.py --epochs 50    (faster run for testing)
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output to avoid cp1252 encoding errors on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import audit_dataset, load_german_credit
from evaluate import evaluate_all_experiments
from explain import run_shap_analysis
from losses import EXPERIMENT_REGISTRY
from model import DEVICE, batch_to_tensors, build_tabtransformer
from preprocessing import run_preprocessing
from train import EPOCHS, LR, train, train_logistic_regression

LOG_DIR = Path(__file__).parent / "outputs" / "metrics"


def log(msg: str):
    print(f"\n{'-'*65}")
    print(f"  {msg}")
    print(f"{'-'*65}")


def run_all(epochs: int = EPOCHS, lr: float = LR):
    start_total = time.time()
    run_log = {}

    # -- Phase 1: Data Audit ------------------------------------------------
    log("Phase 1: Data Audit")
    t0 = time.time()
    X, y = load_german_credit()
    audit = audit_dataset(
        X, y,
        output_path=str(Path(__file__).parent / "outputs" / "data_audit.json")
    )
    print(f"  Records   : {audit['n_samples']}")
    print(f"  Features  : {audit['n_features']}")
    print(f"  Bad  (0)  : {audit['class_distribution']['0_bad']}")
    print(f"  Good (1)  : {audit['class_distribution']['1_good']}")
    print(f"  Missing   : {audit['total_missing']}")
    run_log["phase1_data_audit"] = audit
    run_log["phase1_time_s"] = round(time.time() - t0, 2)

    # -- Phase 2: Preprocessing ---------------------------------------------
    log("Phase 2: Split & Preprocessing")
    t0 = time.time()
    result = run_preprocessing(save=True)
    run_log["phase2_time_s"] = round(time.time() - t0, 2)

    # -- Phase 3: Model sanity check ----------------------------------------
    log("Phase 3: TabTransformer Sanity Check")
    import torch
    t0 = time.time()
    cardinalities = result["cardinalities"]
    model = build_tabtransformer(cardinalities)
    batch_X = result["X_train"].iloc[:4]
    batch_y = result["y_train"].iloc[:4]
    cat_t, num_t, y_t = batch_to_tensors(batch_X, batch_y)
    with torch.no_grad():
        logits = model(cat_t, num_t)
    assert logits.shape == (4, 2), f"Unexpected logits shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "Non-finite logits detected."
    print(f"  [OK] Forward pass OK - logits shape: {logits.shape}")
    del model  # free memory before training
    run_log["phase3_time_s"] = round(time.time() - t0, 2)

    # -- Phase 4: Logistic Regression Baseline -----------------------------
    log("Phase 4: Logistic Regression Baseline")
    t0 = time.time()
    lr_result = train_logistic_regression(result)
    run_log["phase4_lr_val_acc"]  = round(lr_result["val_acc"], 4)
    run_log["phase4_lr_test_acc"] = round(lr_result["test_acc"], 4)
    run_log["phase4_time_s"] = round(time.time() - t0, 2)

    # -- Phase 4+5: Train all three TabTransformer experiments -------------
    log("Phase 4/5: Training TabTransformer (Experiments A, B, C)")
    train_results = {}
    for exp in ["A", "B", "C"]:
        t0 = time.time()
        log(f"  Starting Experiment {exp}: {EXPERIMENT_REGISTRY[exp]['name']}")
        tr = train(exp, result=result, epochs=epochs, lr=lr, verbose=True)
        train_results[exp] = tr
        run_log[f"phase{'4' if exp=='A' else '5'}_exp{exp}_best_val_loss"] = round(tr["best_val_loss"], 4)
        run_log[f"phase{'4' if exp=='A' else '5'}_exp{exp}_best_epoch"]    = tr["best_epoch"]
        run_log[f"phase{'4' if exp=='A' else '5'}_exp{exp}_time_s"]        = round(time.time() - t0, 2)

    # -- Phase 6: Evaluation ------------------------------------------------
    log("Phase 6: Evaluation")
    t0 = time.time()
    metrics_df = evaluate_all_experiments(result)
    run_log["phase6_time_s"] = round(time.time() - t0, 2)

    # -- Phase 7: SHAP Explanations -----------------------------------------
    log("Phase 7: SHAP Explanations")
    t0 = time.time()
    try:
        run_shap_analysis(result, experiment="B")
        run_log["phase7_status"] = "success"
    except Exception as e:
        print(f"  [warn] SHAP failed: {e}")
        run_log["phase7_status"] = f"failed: {e}"
    run_log["phase7_time_s"] = round(time.time() - t0, 2)

    # -- Phase 9: Final summary ---------------------------------------------
    log("Phase 9: Final Summary")
    total_time = round(time.time() - start_total, 2)
    run_log["total_time_s"] = total_time

    os.makedirs(str(LOG_DIR), exist_ok=True)
    run_log_path = str(LOG_DIR / "run_log.json")
    with open(run_log_path, "w") as f:
        json.dump(run_log, f, indent=2, default=str)
    print(f"\n  Run log saved to {run_log_path}")

    print(f"\n  {'Experiment':<15} {'Accuracy':>10} {'Macro-F1':>10} {'F2 Score':>10} {'ROC-AUC':>10}")
    print(f"  {'-'*55}")
    for _, row in metrics_df.iterrows():
        print(f"  {row['experiment']:<15} {row['accuracy']:>10.4f} {row['macro_f1']:>10.4f} "
              f"{row['f2_score']:>10.4f} {row['roc_auc']:>10.4f}")

    print(f"\n  Total runtime: {total_time:.1f}s")
    print(f"\n  {'='*55}")
    print(f"  ALL PHASES COMPLETE")
    print(f"  {'='*55}\n")

    return run_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"Learning rate (default: {LR})")
    args = parser.parse_args()
    run_all(epochs=args.epochs, lr=args.lr)
