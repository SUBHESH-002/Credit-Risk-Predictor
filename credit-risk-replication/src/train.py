"""
src/train.py
-------------
PyTorch training loop for TabTransformer on German Credit.

Supports:
  - Reproducible training (SEED=42)
  - Any loss function (CrossEntropyLoss or weighted variant)
  - Best-val-loss checkpoint saving
  - Training/validation history logging
  - Logistic regression sanity-check baseline

Usage:
    python train.py --experiment A   (unweighted baseline)
    python train.py --experiment B   (weighted - full dataset)
    python train.py --experiment C   (weighted - train only)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))

from data import CATEGORICAL_COLS, NUMERICAL_COLS, load_german_credit
from losses import EXPERIMENT_REGISTRY, make_unweighted_loss, make_weighted_loss
from model import DEVICE, batch_to_tensors, build_tabtransformer
from preprocessing import GermanCreditPreprocessor, run_preprocessing

# --- Reproducibility ----------------------------------------------------------
SEED = 42
LR   = 1e-3      # AdamW learning rate (implementation choice - paper unspecified)
EPOCHS = 100     # Max epochs (implementation choice - paper unspecified)
BATCH_SIZE = 64  # From paper Table 4

MODELS_DIR  = Path(__file__).parent.parent / "outputs" / "models"
PLOTS_DIR   = Path(__file__).parent.parent / "outputs" / "plots"
METRICS_DIR = Path(__file__).parent.parent / "outputs" / "metrics"


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Dataset helpers ---------------------------------------------------------

def make_tensor_dataset(X, y, device=DEVICE):
    cat_t = torch.tensor(X[CATEGORICAL_COLS].values.astype(int), dtype=torch.long)
    num_t = torch.tensor(X[NUMERICAL_COLS].values.astype(float), dtype=torch.float32)
    y_t   = torch.tensor(y.values.astype(int), dtype=torch.long)
    return TensorDataset(cat_t, num_t, y_t)


def make_loader(dataset, batch_size=BATCH_SIZE, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# --- Training loop ------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for cat_t, num_t, y_t in loader:
        cat_t, num_t, y_t = cat_t.to(DEVICE), num_t.to(DEVICE), y_t.to(DEVICE)
        optimizer.zero_grad()
        logits = model(cat_t, num_t)
        loss   = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_t)
        preds  = logits.argmax(dim=1)
        correct += (preds == y_t).sum().item()
        total  += len(y_t)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for cat_t, num_t, y_t in loader:
        cat_t, num_t, y_t = cat_t.to(DEVICE), num_t.to(DEVICE), y_t.to(DEVICE)
        logits = model(cat_t, num_t)
        loss   = criterion(logits, y_t)
        total_loss += loss.item() * len(y_t)
        preds  = logits.argmax(dim=1)
        correct += (preds == y_t).sum().item()
        total  += len(y_t)
    return total_loss / total, correct / total


def train(
    experiment: str,
    result: dict = None,
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    verbose: bool = True,
):
    """
    Full training run for one experiment (A, B, or C).

    Parameters
    ----------
    experiment : str - 'A', 'B', or 'C'
    result     : dict (optional) - output of run_preprocessing() to avoid re-loading
    Returns dict with training history and model checkpoint path.
    """
    set_seed(SEED)
    exp_info = EXPERIMENT_REGISTRY[experiment]
    print(f"\n{'='*60}")
    print(f"  Experiment {experiment}: {exp_info['name']}")
    print(f"  {exp_info['description']}")
    print(f"{'='*60}")

    if result is None:
        result = run_preprocessing(save=True)

    X_train = result["X_train"]
    X_val   = result["X_val"]
    X_test  = result["X_test"]
    y_train = result["y_train"]
    y_val   = result["y_val"]
    y_test  = result["y_test"]
    cardinalities = result["cardinalities"]

    # Build loss
    if not exp_info["weighted"]:
        criterion = make_unweighted_loss()
    elif exp_info["use_full_dataset_weights"]:
        X_full, y_full = load_german_credit()
        criterion = make_weighted_loss(y_full, num_classes=2, device=DEVICE)
    else:
        criterion = make_weighted_loss(y_train, num_classes=2, device=DEVICE)

    criterion = criterion.to(DEVICE)

    # Build model + optimizer
    model     = build_tabtransformer(cardinalities)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # DataLoaders
    train_ds = make_tensor_dataset(X_train, y_train)
    val_ds   = make_tensor_dataset(X_val,   y_val)
    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader   = make_loader(val_ds,   batch_size, shuffle=False)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch    = 0
    ckpt_path = str(MODELS_DIR / f"tabtransformer_exp{experiment}.pt")
    os.makedirs(str(MODELS_DIR), exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = evaluate(model, val_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch    = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": va_loss,
                "cardinalities": cardinalities,
            }, ckpt_path)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train loss: {tr_loss:.4f} acc: {tr_acc:.4f} | "
                  f"Val loss: {va_loss:.4f} acc: {va_acc:.4f}")

    print(f"\n  Best checkpoint: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
    print(f"  Saved to: {ckpt_path}")

    # Save history
    hist_path = str(METRICS_DIR / f"history_exp{experiment}.json")
    os.makedirs(str(METRICS_DIR), exist_ok=True)
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save loss curves plot
    _plot_history(history, experiment)

    return dict(
        history=history,
        ckpt_path=ckpt_path,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        model=model,
        cardinalities=cardinalities,
        experiment=experiment,
    )


def _plot_history(history: dict, experiment: str):
    """Save training/validation loss curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs, history["train_loss"], label="Train")
        axes[0].plot(epochs, history["val_loss"],   label="Val")
        axes[0].set_title(f"Loss - Experiment {experiment}")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(epochs, history["train_acc"], label="Train")
        axes[1].plot(epochs, history["val_acc"],   label="Val")
        axes[1].set_title(f"Accuracy - Experiment {experiment}")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].legend()

        fig.tight_layout()
        os.makedirs(str(PLOTS_DIR), exist_ok=True)
        out = str(PLOTS_DIR / f"loss_curves_exp{experiment}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Loss curves saved to {out}")
    except Exception as e:
        print(f"  [warn] Could not save plot: {e}")


# --- Logistic Regression baseline --------------------------------------------

def train_logistic_regression(result: dict = None) -> dict:
    """
    Quick sanity-check baseline with Logistic Regression.
    Helps identify problems with the data pipeline.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    if result is None:
        result = run_preprocessing(save=False)

    X_train = result["X_train"]
    X_val   = result["X_val"]
    X_test  = result["X_test"]
    y_train = result["y_train"]
    y_val   = result["y_val"]
    y_test  = result["y_test"]

    lr_model = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_model.fit(X_train, y_train)

    val_acc  = accuracy_score(y_val,  lr_model.predict(X_val))
    test_acc = accuracy_score(y_test, lr_model.predict(X_test))

    print(f"\n[Logistic Regression Baseline]")
    print(f"  Val  accuracy: {val_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Test report:\n{classification_report(y_test, lr_model.predict(X_test))}")

    # Save
    import pickle
    lr_path = str(MODELS_DIR / "logistic_regression.pkl")
    os.makedirs(str(MODELS_DIR), exist_ok=True)
    with open(lr_path, "wb") as f:
        pickle.dump(lr_model, f)
    print(f"  Saved to {lr_path}")

    return {"val_acc": val_acc, "test_acc": test_acc, "model": lr_model}


# --- Entrypoint --------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TabTransformer on German Credit")
    parser.add_argument("--experiment", choices=["A", "B", "C", "all"], default="all",
                        help="Which experiment to run (A=unweighted, B=full-weighted, C=train-weighted)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    result = run_preprocessing(save=True)

    # Always run LR baseline
    train_logistic_regression(result)

    if args.experiment == "all":
        for exp in ["A", "B", "C"]:
            train(exp, result=result, epochs=args.epochs, lr=args.lr)
    else:
        train(args.experiment, result=result, epochs=args.epochs, lr=args.lr)

    print("\n[train.py] All experiments complete.")
