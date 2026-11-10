# Reproducibility Notes

This document records all methodological ambiguities, implementation choices,
and deviations from the paper that must be understood when interpreting results.

---

## Paper Reference

> *A Novel Weighted Loss TabTransformer Integrating Explainable AI for Imbalanced
> Credit Risk Datasets*

---

## Known Ambiguities and Our Resolutions

### 1. Categorical Feature Encoding

**Paper states:** One-hot encoding for categorical features.

**Our implementation:** Integer label encoding with embedding lookup (TabTransformer
native representation). Index 0 is reserved for unseen/unknown categories; known
categories start at index 1.

**Reason:** The `tab-transformer-pytorch` (lucidrains) implementation uses learnable
embeddings, not one-hot vectors, as its core mechanism. Using one-hot encoding would
bypass the Transformer's embedding layer, which is architecturally central to the paper.

**Impact:** This is the most significant methodological discrepancy. The paper's one-hot
mention likely refers to how categoricals are handled before passing to the transformer,
not the final representation used in the architecture.

---

### 2. Optimizer and Learning Rate

**Paper states:** Not fully specified.

**Our implementation:** AdamW, learning rate = 0.001, up to 100 epochs.

**Reason:** AdamW is the standard choice for transformer-based models. The LR of 0.001
is a common default for AdamW on tabular data.

---

### 3. Class Weight Computation (Experiments B vs C)

**Paper states:** Class weights are computed using the original dataset distribution
before splitting.

**Experiment B:** Follows the paper — weights computed from the full 1000-sample dataset.
This means validation and test label counts influence the weights, which constitutes
a minor form of label leakage.

**Experiment C:** Conservative alternative — weights computed from training labels only
(800 samples). This avoids any leakage.

Both results are reported separately. Experiment B is the closer replication of the paper.

---

### 4. Random Seed

**Paper states:** Not mentioned.

**Our implementation:** SEED = 42 is used for all random operations:
- `random.seed(42)`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)`
- `sklearn.model_selection.train_test_split(..., random_state=42)`

---

### 5. Data Split Proportions

**Paper states:** 80:10:10 train/validation/test split with stratification.

**Our implementation:** Exactly as stated. Stratified by target label using
`sklearn.model_selection.train_test_split` with `stratify=y`.

---

### 6. Missing Values

**Paper states:** Not mentioned.

**Our finding:** German Credit has no missing values in the raw data file.
A `SimpleImputer(strategy='median')` is fitted on training data as a
defensive measure.

---

### 7. TabTransformer Implementation

**Paper states:** TabTransformer (Huang et al., 2020).

**Our implementation:** `tab-transformer-pytorch` by lucidrains (GitHub).
This is the most widely used PyTorch implementation.

---

### 8. Hyperparameters

From paper Table 4 (German Credit):

| Hyperparameter | Paper Value | Our Value |
|---------------|-------------|-----------|
| Embedding dim | 64 | 64 |
| Transformer depth | 2 | 2 |
| Attention heads | 2 | 2 |
| Attention dropout | 0.1 | 0.1 |
| FF dropout | 0.1 | 0.1 |
| Output classes | 2 | 2 |
| Batch size | 64 | 64 |

---

### 9. SHAP Explainer Choice

**Paper states:** SHAP is used for explanations.

**Our implementation:** `shap.KernelExplainer` (model-agnostic). This works with
any prediction function. We explain class 1 (Good credit) probability.

Background sample: 100 randomly selected training records (SEED=42).
Number of SHAP samples per record: 200 (nsamples parameter).

KernelExplainer is computationally expensive; we explain 50 test records.

---

### 10. Cross-Validation for Hyperparameter Tuning

**Paper states:** Ten-fold cross-validation during hyperparameter tuning.

**Our implementation:** Not implemented in the main run. The paper does not
fully specify the search grid. The hyperparameters from Table 4 are used directly
without tuning, as the goal is replication, not discovery.

A future extension could add cross-validation using `sklearn.model_selection.StratifiedKFold`
on the training set only (test set kept untouched).

---

## Performance Expectations

The paper reports:
- German Credit Unweighted: **93% accuracy**
- German Credit Weighted:   **95% accuracy**

Due to the ambiguities above (especially encoding choice and optimizer settings),
we do not expect an exact match. Results within ±3% of the paper's figures should
be considered a successful replication.

---

## BISAID Dataset

BISAID is not implemented in this replication. The paper does not make the four-class
target semantics sufficiently clear to reconstruct them from the text alone. Proceeding
without verified class definitions would produce meaningless results.

---

*Last updated: 2026-10-13*
*Author: Antigravity Agent*
