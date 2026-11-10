"""
src/data.py
-----------
Loads the German Credit dataset from the local raw file.
Falls back to ucimlrepo if local file is not available.

Label mapping (Paper convention):
  UCI original: 1 = Good, 2 = Bad
  Paper:        1 = Good (class 1), 0 = Bad (class 0)
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Column names from german.doc (Attribute 1 to 20 + target)
COLUMN_NAMES = [
    "checking_status",         # A1  - categorical
    "duration",                # A2  - numerical
    "credit_history",          # A3  - categorical
    "purpose",                 # A4  - categorical
    "credit_amount",           # A5  - numerical
    "savings_status",          # A6  - categorical
    "employment",              # A7  - categorical
    "installment_commitment",  # A8  - numerical
    "personal_status",         # A9  - categorical
    "other_parties",           # A10 - categorical
    "residence_since",         # A11 - numerical
    "property_magnitude",      # A12 - categorical
    "age",                     # A13 - numerical
    "other_payment_plans",     # A14 - categorical
    "housing",                 # A15 - categorical
    "existing_credits",        # A16 - numerical
    "job",                     # A17 - categorical
    "num_dependents",          # A18 - numerical
    "own_telephone",           # A19 - categorical
    "foreign_worker",          # A20 - categorical
    "target",                  # Target: 1 = Good, 2 = Bad (UCI) -> 1/0 (paper)
]

CATEGORICAL_COLS = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

NUMERICAL_COLS = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]


def load_german_credit(raw_dir: str = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load German Credit dataset from the local raw file.

    Returns
    -------
    X : pd.DataFrame  - feature matrix (1000   20)
    y : pd.Series     - binary target (0 = Bad, 1 = Good)
    """
    if raw_dir is None:
        raw_dir = Path(__file__).parent.parent / "data" / "raw"
    raw_dir = Path(raw_dir)
    data_file = raw_dir / "german.data"

    if not data_file.exists():
        raise FileNotFoundError(
            f"german.data not found at {data_file}. "
            "Place the raw dataset in data/raw/ or provide raw_dir."
        )

    df = pd.read_csv(data_file, sep=" ", header=None, names=COLUMN_NAMES)

    X = df.drop(columns=["target"]).copy()
    y_raw = df["target"].copy()

    # Map UCI labels to paper labels: 1(Good)->1, 2(Bad)->0
    y = y_raw.astype(int).map({1: 1, 2: 0})
    assert y.notna().all(), "Label mapping produced NaN values."
    assert set(y.unique()) == {0, 1}, f"Unexpected label values: {set(y.unique())}"

    return X, y


def audit_dataset(X: pd.DataFrame, y: pd.Series, output_path: str = None) -> dict:
    """
    Produce a data audit report and optionally save it to JSON.

    Returns a dict with dataset statistics.
    """
    class_counts = y.value_counts().sort_index().to_dict()
    class_counts = {int(k): int(v) for k, v in class_counts.items()}

    missing = X.isna().sum().to_dict()
    missing = {k: int(v) for k, v in missing.items()}

    dtypes = {col: str(dtype) for col, dtype in X.dtypes.items()}

    audit = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "categorical_features": CATEGORICAL_COLS,
        "numerical_features": NUMERICAL_COLS,
        "dtypes": dtypes,
        "missing_values": missing,
        "total_missing": int(X.isna().sum().sum()),
        "class_distribution": {
            "0_bad": class_counts.get(0, 0),
            "1_good": class_counts.get(1, 0),
        },
        "class_ratio_bad_to_good": round(
            class_counts.get(0, 0) / max(class_counts.get(1, 1), 1), 4
        ),
        "label_mapping": "UCI 1=Good->1, UCI 2=Bad->0",
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(audit, f, indent=2)
        print(f"[data.py] Audit saved to {output_path}")

    return audit


if __name__ == "__main__":
    X, y = load_german_credit()
    audit = audit_dataset(
        X, y,
        output_path=str(Path(__file__).parent.parent / "outputs" / "data_audit.json")
    )
    print("\n=== DATA AUDIT ===")
    for k, v in audit.items():
        print(f"  {k}: {v}")
