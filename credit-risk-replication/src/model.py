"""
src/model.py
-------------
TabTransformer model builder for German Credit.

Uses the lucidrains/tab-transformer-pytorch implementation.

Hyperparameters from paper Table 4:
  Embedding dimension : 64   (German Credit)
  Transformer depth   : 2
  Attention heads     : 2
  Attention dropout   : 0.1
  FF dropout          : 0.1
  Output classes      : 2    (binary: 0=bad, 1=good)
  Batch size          : 64

NOTE: Optimizer settings (AdamW, lr=0.001, 100 epochs) are our implementation
choices because the paper does not fully specify them.
"""

import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer

from data import CATEGORICAL_COLS, NUMERICAL_COLS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_tabtransformer(
    category_cardinalities: list[int],
    num_continuous: int = len(NUMERICAL_COLS),
    dim: int = 64,
    depth: int = 2,
    heads: int = 2,
    attn_dropout: float = 0.1,
    ff_dropout: float = 0.1,
    num_classes: int = 2,
) -> TabTransformer:
    """
    Build a TabTransformer model with the paper's German Credit hyperparameters.

    Parameters
    ----------
    category_cardinalities : list[int]
        Number of unique categories (+ unknown slot) for each categorical column.
    num_continuous : int
        Number of numerical (continuous) features.
    dim : int
        Embedding dimension for categorical features.
    depth : int
        Number of Transformer blocks.
    heads : int
        Number of self-attention heads.
    attn_dropout : float
        Dropout in attention layers.
    ff_dropout : float
        Dropout in feed-forward layers.
    num_classes : int
        Number of output classes (2 for binary credit risk).

    Returns
    -------
    model : TabTransformer
    """
    model = TabTransformer(
        categories=tuple(category_cardinalities),
        num_continuous=num_continuous,
        dim=dim,
        dim_out=num_classes,
        depth=depth,
        heads=heads,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        mlp_act=nn.ReLU(),
    )
    model = model.to(DEVICE)
    print(f"[model.py] TabTransformer built on device: {DEVICE}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model.py] Trainable parameters: {n_params:,}")
    return model


def batch_to_tensors(
    X_batch: "pd.DataFrame",
    y_batch: "pd.Series" = None,
    device: torch.device = DEVICE,
) -> tuple:
    """
    Convert a DataFrame batch to the tensors expected by TabTransformer.

    Returns (cat_tensor, num_tensor, [y_tensor])
    """
    cat_data = torch.tensor(
        X_batch[CATEGORICAL_COLS].values.astype(int),
        dtype=torch.long,
        device=device,
    )
    num_data = torch.tensor(
        X_batch[NUMERICAL_COLS].values.astype(float),
        dtype=torch.float32,
        device=device,
    )
    if y_batch is not None:
        y_tensor = torch.tensor(
            y_batch.values.astype(int),
            dtype=torch.long,
            device=device,
        )
        return cat_data, num_data, y_tensor
    return cat_data, num_data


if __name__ == "__main__":
    # Sanity check: pass one batch through the model
    import pandas as pd
    import numpy as np
    from preprocessing import run_preprocessing

    result = run_preprocessing(save=False)
    cardinalities = result["cardinalities"]
    X_train = result["X_train"]
    y_train = result["y_train"]

    model = build_tabtransformer(cardinalities)

    # Take a small batch
    batch_X = X_train.iloc[:8]
    batch_y = y_train.iloc[:8]

    cat_t, num_t, y_t = batch_to_tensors(batch_X, batch_y)
    with torch.no_grad():
        logits = model(cat_t, num_t)

    print(f"\n[model.py] Batch logits shape: {logits.shape}")   # (8, 2)
    print(f"[model.py] Sample logits:\n{logits}")
    print("\n[model.py] [OK] Sanity check passed - model forward pass works.")
