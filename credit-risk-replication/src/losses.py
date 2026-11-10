"""
src/losses.py
--------------
Weighted cross-entropy loss for imbalanced credit risk classification.

The paper defines the class weight as:
    w_i = N / N_i

where N is the total number of samples and N_i is the number of samples in class i.

This module provides:
  - make_weighted_loss()  -> returns nn.CrossEntropyLoss with class weights
  - Three experiment modes:
      A: Ordinary cross-entropy (no weighting) - baseline
      B: Weights from the FULL dataset class counts (paper's stated procedure)
      C: Weights from TRAINING labels only (leakage-conscious comparison)
"""

import numpy as np
import torch
import torch.nn as nn


def make_weighted_loss(
    y_for_weights,
    num_classes: int,
    device: torch.device,
) -> nn.CrossEntropyLoss:
    """
    Create a CrossEntropyLoss with inverse-frequency class weights.

    Parameters
    ----------
    y_for_weights : array-like of int
        Labels used to compute class frequencies (full dataset for Exp B,
        training set for Exp C).
    num_classes : int
        Total number of classes.
    device : torch.device
        Device where the weights tensor will be placed.

    Returns
    -------
    criterion : nn.CrossEntropyLoss with weight parameter set.
    """
    counts = np.bincount(
        np.asarray(y_for_weights, dtype=int),
        minlength=num_classes,
    )

    if np.any(counts == 0):
        raise ValueError(
            f"Every class needs at least one sample. Got counts: {counts}"
        )

    # w_i = N / N_i
    weights = len(y_for_weights) / counts
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print(f"[losses.py] Class weights: {weights.tolist()}")
    return nn.CrossEntropyLoss(weight=weights)


def make_unweighted_loss() -> nn.CrossEntropyLoss:
    """Ordinary cross-entropy loss - Experiment A (baseline)."""
    return nn.CrossEntropyLoss()


EXPERIMENT_REGISTRY = {
    "A": {
        "name": "Unweighted (Baseline)",
        "description": "Ordinary cross-entropy loss. No class weighting.",
        "weighted": False,
        "use_full_dataset_weights": False,
    },
    "B": {
        "name": "Weighted (Full Dataset)",
        "description": (
            "Class weights computed from full dataset class counts, "
            "following the paper's stated weighting procedure."
        ),
        "weighted": True,
        "use_full_dataset_weights": True,
    },
    "C": {
        "name": "Weighted (Train Only)",
        "description": (
            "Class weights computed from training labels only. "
            "More conservative - avoids using held-out label information."
        ),
        "weighted": True,
        "use_full_dataset_weights": False,
    },
}


if __name__ == "__main__":
    import torch

    device = torch.device("cpu")
    y_train = [0] * 240 + [1] * 560   # approximate train split (30% bad, 70% good)
    y_full  = [0] * 300 + [1] * 700   # full German Credit distribution

    print("=== Experiment A (Unweighted) ===")
    loss_A = make_unweighted_loss()
    print(f"  Loss: {loss_A}")

    print("\n=== Experiment B (Full dataset weights) ===")
    loss_B = make_weighted_loss(y_full, num_classes=2, device=device)
    print(f"  Loss: {loss_B}")

    print("\n=== Experiment C (Train-only weights) ===")
    loss_C = make_weighted_loss(y_train, num_classes=2, device=device)
    print(f"  Loss: {loss_C}")
