# Credit Risk Predictor

Replication of the research paper:
> **"A Novel Weighted Loss TabTransformer Integrating Explainable AI for Imbalanced Credit Risk Datasets"**

This project replicates the TabTransformer architecture with class-weighted loss functions to tackle credit risk classification on imbalanced financial datasets (specifically the UCI Statlog German Credit dataset), incorporating Explainable AI via SHAP (SHapley Additive exPlanations).

---

## 📌 Repository Overview

```
.
├── credit-risk-replication/      # Core replication codebase
│   ├── configs/                  # Experiment configuration (YAML)
│   ├── data/                     # Raw and preprocessed dataset splits
│   ├── notebooks/                # Exploratory & analysis notebooks
│   ├── outputs/                  # Model checkpoints, evaluation metrics, SHAP plots
│   ├── src/                      # Source modules (data, preprocessing, model, train, evaluate, explain)
│   ├── run_all.py                # End-to-end pipeline runner
│   ├── requirements.txt          # Python dependencies
│   ├── README.md                 # Detailed replication instructions
│   └── REPRODUCIBILITY_NOTES.md  # Experimental details and replication notes
├── dataset/                      # Original German Credit dataset files
├── TabTransformer_GermanCredit_Replication.pptx  # Summary presentation deck
├── TabTransformer_Professional_Redesign.pptx     # Redesigned presentation deck
├── implementation plan.pdf                       # Project execution roadmap
├── project_understanding.md                      # Academic paper context & breakdown
└── walkthrough.md                               # Project walkthrough & results
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
cd credit-risk-replication
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Full Pipeline
```bash
python run_all.py
```

This will run all experiment phases (data audit, preprocessing, model verification, training experiments A/B/C, evaluation, and SHAP explanations).

---

## 📊 Key Experiments & Results

- **Experiment A**: Baseline TabTransformer with standard unweighted Cross-Entropy Loss.
- **Experiment B**: TabTransformer with Class-Weighted Cross-Entropy Loss (weights calculated from full dataset).
- **Experiment C**: TabTransformer with Class-Weighted Cross-Entropy Loss (weights calculated from train split only to avoid data leakage).

Metrics, loss curves, ROC curves, and SHAP feature importance plots are generated under `credit-risk-replication/outputs/`.

---

## 📄 License & Attribution

- Dataset: UCI Machine Learning Repository (Statlog German Credit Data).
- Replication based on published literature on Weighted Loss TabTransformers for imbalanced tabular credit scoring.
