"""
src/preprocessing.py
---------------------
Implements the full feature preprocessing pipeline for German Credit:

  - Categorical features  -> integer label-encoded IDs (for TabTransformer embeddings)
  - Numerical features    -> MinMaxScaler(feature_range=(-1, 1)) fit on training data only
  - Missing values        -> SimpleImputer fit on training data only
  - Unseen categories     -> reserved index 0 (all known categories start at 1)

NOTE: The paper mentions one-hot encoding, but TabTransformer uses categorical embeddings.
We use integer IDs for the embedding lookup. This methodological ambiguity is documented
in REPRODUCIBILITY_NOTES.md.

All fitted objects are saved to data/processed/ as .pkl files so downstream code can load
them without re-fitting.
"""

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from data import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    load_german_credit,
    audit_dataset,
)

SEED = 42
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


# ------------------------------------------------------------------------------
# Split
# ------------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.10,
    test_size: float = 0.10,
    seed: int = SEED,
) -> tuple:
    """
    80 / 10 / 10 stratified split as described in the paper.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split off the 20% temp set (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=val_size + test_size,
        stratify=y,
        random_state=seed,
    )
    # Split temp 50/50 into val and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed,
    )
    print(f"[preprocessing.py] Split sizes -> Train: {len(X_train)}, "
          f"Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ------------------------------------------------------------------------------
# Categorical encoding
# ------------------------------------------------------------------------------

class CategoricalEncoder:
    """
    Encodes each categorical column as integer IDs.

    Index 0 is reserved for UNKNOWN / unseen categories.
    Known categories start at 1.
    """

    def __init__(self):
        self.encoders: dict[str, LabelEncoder] = {}
        self.cardinalities: dict[str, int] = {}   # including the unknown slot

    def fit(self, X: pd.DataFrame) -> "CategoricalEncoder":
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
            # +1 because index 0 is reserved for unknown
            self.cardinalities[col] = len(le.classes_) + 1
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in CATEGORICAL_COLS:
            le = self.encoders[col]
            known = set(le.classes_)
            X[col] = X[col].astype(str).apply(
                lambda v: int(le.transform([v])[0]) + 1  # shift: 1..N
                if v in known else 0                       # 0 = unknown
            )
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

    def category_cardinalities(self) -> list[int]:
        """Return list of cardinalities in CATEGORICAL_COLS order."""
        return [self.cardinalities[col] for col in CATEGORICAL_COLS]


# ------------------------------------------------------------------------------
# Numerical scaling
# ------------------------------------------------------------------------------

class NumericalScaler:
    """MinMaxScaler(feature_range=(-1, 1)) fit on training data only."""

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.imputer = SimpleImputer(strategy="median")

    def fit(self, X: pd.DataFrame) -> "NumericalScaler":
        vals = X[NUMERICAL_COLS].values
        vals = self.imputer.fit_transform(vals)
        self.scaler.fit(vals)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        vals = X[NUMERICAL_COLS].values.astype(float)
        vals = self.imputer.transform(vals)
        vals = self.scaler.transform(vals)
        X[NUMERICAL_COLS] = vals
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)


# ------------------------------------------------------------------------------
# Full pipeline
# ------------------------------------------------------------------------------

class GermanCreditPreprocessor:
    """Combines CategoricalEncoder + NumericalScaler into a single object."""

    def __init__(self):
        self.cat_encoder = CategoricalEncoder()
        self.num_scaler = NumericalScaler()
        self.fitted = False

    def fit(self, X_train: pd.DataFrame) -> "GermanCreditPreprocessor":
        self.cat_encoder.fit(X_train)
        self.num_scaler.fit(X_train)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit() first.")
        X = self.cat_encoder.transform(X)
        X = self.num_scaler.transform(X)
        return X

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        self.fit(X_train)
        return self.transform(X_train)

    def category_cardinalities(self) -> list[int]:
        return self.cat_encoder.category_cardinalities()

    def save(self, path: str = None):
        if path is None:
            path = str(PROCESSED_DIR / "preprocessor.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[preprocessing.py] Preprocessor saved to {path}")

    @staticmethod
    def load(path: str = None) -> "GermanCreditPreprocessor":
        if path is None:
            path = str(PROCESSED_DIR / "preprocessor.pkl")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[preprocessing.py] Preprocessor loaded from {path}")
        return obj


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------

def run_preprocessing(save: bool = True) -> dict[str, Any]:
    """
    Full preprocessing run. Loads data, splits, preprocesses and saves artifacts.

    Returns a dict with all processed splits and the preprocessor.
    """
    X, y = load_german_credit()
    audit_dataset(
        X, y,
        output_path=str(Path(__file__).parent.parent / "outputs" / "data_audit.json")
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    preprocessor = GermanCreditPreprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p   = preprocessor.transform(X_val)
    X_test_p  = preprocessor.transform(X_test)

    if save:
        preprocessor.save()

        # Save splits as CSV for reference
        for name, Xd, yd in [
            ("train", X_train_p, y_train),
            ("val",   X_val_p,   y_val),
            ("test",  X_test_p,  y_test),
        ]:
            out = PROCESSED_DIR / f"{name}.csv"
            Xd.assign(target=yd.values).to_csv(out, index=False)

        print(f"[preprocessing.py] Processed splits saved to {PROCESSED_DIR}")

    # Validate class proportions across splits
    for name, yd in [("train", y_train), ("val", y_val), ("test", y_test)]:
        counts = yd.value_counts().sort_index()
        print(f"  {name} class counts -> {counts.to_dict()}")

    cardinalities = preprocessor.category_cardinalities()
    print(f"[preprocessing.py] Category cardinalities: {cardinalities}")

    return dict(
        X_train=X_train_p, X_val=X_val_p, X_test=X_test_p,
        y_train=y_train,   y_val=y_val,   y_test=y_test,
        preprocessor=preprocessor,
        cardinalities=cardinalities,
    )


if __name__ == "__main__":
    result = run_preprocessing(save=True)
    print("\n[preprocessing.py] Done.")
