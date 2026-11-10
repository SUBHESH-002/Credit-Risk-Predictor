# Credit Risk Replication

Replication of the paper:

> *A Novel Weighted Loss TabTransformer Integrating Explainable AI for Imbalanced Credit Risk Datasets*

This repository reproduces the TabTransformer + weighted loss experiments on the **German Credit** dataset and explains predictions using **SHAP**.

---

## Project Structure

```
credit-risk-replication/
├── data/
│   ├── raw/          ← german.data (place dataset here)
│   └── processed/    ← preprocessed splits and fitted objects
├── src/
│   ├── data.py           ← data loading + label mapping
│   ├── preprocessing.py  ← 80:10:10 split + encoding + scaling
│   ├── model.py          ← TabTransformer builder
│   ├── train.py          ← training loop (experiments A, B, C)
│   ├── losses.py         ← weighted cross-entropy
│   ├── evaluate.py       ← metrics + confusion matrix + ROC
│   └── explain.py        ← SHAP feature explanations
├── configs/
│   └── german_credit.yaml
├── outputs/
│   ├── data_audit.json
│   ├── metrics/
│   │   ├── results.csv
│   │   ├── history_expA.json
│   │   └── run_log.json
│   ├── plots/
│   │   ├── loss_curves_expA.png
│   │   ├── confusion_matrix_expA_test.png
│   │   ├── roc_curve_expA_test.png
│   │   └── shap/
│   └── models/
│       ├── tabtransformer_expA.pt
│       ├── tabtransformer_expB.pt
│       ├── tabtransformer_expC.pt
│       └── logistic_regression.pkl
├── run_all.py         ← master pipeline script
├── requirements.txt
└── REPRODUCIBILITY_NOTES.md
```

---

## Setup

### 1. Install Python 3.10+

### 2. Create virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Obtain the dataset
Place `german.data` into `data/raw/`. The file comes from the UCI Statlog German Credit dataset:
- [UCI ML Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

---

## Running the Experiments

### Run everything (all phases) — recommended
```bash
python run_all.py
```

For a quick test run (fewer epochs):
```bash
python run_all.py --epochs 10
```

### Run individual steps
```bash
# Phase 1-2: Data audit and preprocessing
python src/data.py
python src/preprocessing.py

# Phase 3: Verify model
python src/model.py

# Phase 4: Train unweighted baseline (Experiment A)
python src/train.py --experiment A

# Phase 5: Train weighted experiments
python src/train.py --experiment B
python src/train.py --experiment C

# Phase 5 (all at once):
python src/train.py --experiment all

# Phase 6: Evaluate
python src/evaluate.py

# Phase 7: SHAP
python src/explain.py
```

---

## Experiments

| Experiment | Loss | Description |
|-----------|------|-------------|
| **A** | CrossEntropy (unweighted) | Baseline — standard training |
| **B** | Weighted CrossEntropy (full dataset) | Paper's stated procedure: weights from full dataset class counts |
| **C** | Weighted CrossEntropy (train only) | Conservative: weights from training labels only |

**Weight formula:** `w_i = N / N_i`

---

## Paper Reference Targets

| Dataset | Unweighted | Weighted |
|---------|-----------|---------|
| German Credit | 93% | 95% |

*Exact matches are not expected due to unspecified optimizer details, preprocessing ambiguities, and random initialization.*

---

## Key Results

See [`outputs/metrics/results.csv`](outputs/metrics/results.csv) for full metrics after running.

---

## SHAP Feature Importance

Explanation uses Experiment B (weighted model), explaining the probability of **Good credit (class 1)**.

Paper reference features for German Credit: **Credit amount, Age, Existing credit status**.

See [`outputs/plots/shap/`](outputs/plots/shap/) for charts.

---

## Notes on Reproducibility

See [`REPRODUCIBILITY_NOTES.md`](REPRODUCIBILITY_NOTES.md) for all documented methodological ambiguities and implementation choices.

---

## Citation

```
UCI Statlog German Credit Dataset
  Prof. Dr. Hans Hofmann, Universität Hamburg
```
