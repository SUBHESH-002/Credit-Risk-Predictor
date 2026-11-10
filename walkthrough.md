# Replication Walkthrough: Weighted Loss TabTransformer & Explainable AI (German Credit)

## Executive Summary

We have successfully implemented and executed **Phases 0 through 7 and Phase 9** of the paper replication:

> **"A Novel Weighted Loss TabTransformer Integrating Explainable AI for Imbalanced Credit Risk Datasets"**

The entire pipeline is packaged, self-contained, reproducible, and evaluated on the **German Credit dataset** (1,000 instances, 20 features).

---

## 1. Experimental Results & Performance Comparison

All experiments were evaluated on an identical **unseen test set of 100 samples** (stratified 80:10:10 train/val/test split).

| Model / Experiment | Loss Function | Accuracy | Macro-F1 | Macro-Recall | F2-Score | ROC-AUC | Bad Credit Recall (Class 0) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** (Sanity Baseline) | Logistic Loss | 76.0% | 0.7000 | 0.6900 | 0.6912 | 0.7724 | 50.0% (15/30) |
| **Exp A: TabTransformer** (Unweighted Baseline) | Standard Cross-Entropy | 74.0% | 0.5776 | 0.5857 | 0.5742 | 0.8148 | 20.0% (6/30) |
| **Exp B: TabTransformer** (Full-Dataset Weights) | Weighted Cross-Entropy ($w_i = N/N_i$) | **75.0%** | **0.7234** | **0.7452** | **0.7345** | **0.8157** | **73.3% (22/30)** |
| **Exp C: TabTransformer** (Train-Only Weights) | Weighted Cross-Entropy ($w_i = N_{tr}/N_{tr,i}$) | **75.0%** | **0.7234** | **0.7452** | **0.7345** | **0.8157** | **73.3% (22/30)** |

---

## 2. Key Findings & Paper Validation

### 1. Significant Minority Class Catch (+53.3% Recall Improvement)
In credit risk, class 0 (Bad Credit) is the minority (30% prevalence) but carries the catastrophic default risk.
- **Unweighted Model (Exp A)**: Predicted good credit almost indiscriminately, capturing only **6 out of 30** defaulters (80% missed defaults).
- **Weighted Models (Exp B & C)**: Caught **22 out of 30** defaulters (**73.3% recall**), cutting missed loan defaults from 24 down to 8.

### 2. Credit Risk Cost Matrix Impact
According to the Statlog German Credit cost matrix:
$$\text{Cost} = 5 \times \text{False Positives (bad approved)} + 1 \times \text{False Negatives (good denied)}$$

- **Experiment A Cost**: $5 \times 24 + 1 \times 2 = \mathbf{122}$
- **Experiment B/C Cost**: $5 \times 8 + 1 \times 17 = \mathbf{57}$
- **Cost Reduction**: **-53.3% financial penalty** using the weighted TabTransformer.

### 3. Metric Uplift
- **Macro-F1**: Rose from **0.5776 $\rightarrow$ 0.7234** (+14.58 percentage points)
- **F2-Score**: Rose from **0.5742 $\rightarrow$ 0.7345** (+16.03 percentage points)

---

## 3. Visualizations

### Confusion Matrices: Unweighted vs. Weighted

````carousel
![Confusion Matrix - Experiment A (Unweighted Baseline)](C:\Users\subhe\.gemini\antigravity\brain\80df0391-a595-465d-9565-257ac448e073\confusion_matrix_expA_test.png)
<!-- slide -->
![Confusion Matrix - Experiment B (Weighted Loss)](C:\Users\subhe\.gemini\antigravity\brain\80df0391-a595-465d-9565-257ac448e073\confusion_matrix_expB_test.png)
<!-- slide -->
![Combined ROC Curves](C:\Users\subhe\.gemini\antigravity\brain\80df0391-a595-465d-9565-257ac448e073\roc_curves_all_test.png)
````

### Training Dynamics (Experiment B)
![Loss Curves - Experiment B](C:\Users\subhe\.gemini\antigravity\brain\80df0391-a595-465d-9565-257ac448e073\loss_curves_expB.png)

---

## 4. Explainable AI: SHAP Feature Importance

SHAP values were estimated using `shap.KernelExplainer` on the best weighted TabTransformer model (Experiment B) predicting **Good credit probability (Class 1)**.

### Top 10 Influential Features (Mean |SHAP Value|)
1. **`checking_status`** ($0.1122$) — Status of existing checking account (primary driver)
2. **`purpose`** ($0.0397$) — Purpose of the loan
3. **`credit_history`** ($0.0382$) — Prior credit payment history
4. **`property_magnitude`** ($0.0375$) — Real estate, savings agreement, or property owned
5. **`savings_status`** ($0.0329$) — Savings account balances
6. **`employment`** ($0.0240$) — Current employment duration
7. **`other_payment_plans`** ($0.0178$) — Bank/store installment plans
8. **`own_telephone`** ($0.0163$) — Telephone registered
9. **`personal_status`** ($0.0163$) — Marital status and sex
10. **`housing`** ($0.0110$) — Rent vs. own vs. free housing

![Global Feature Importance - Experiment B](C:\Users\subhe\.gemini\antigravity\brain\80df0391-a595-465d-9565-257ac448e073\global_importance_expB.png)

### Individual Prediction Waterfall Explanation
![Individual Prediction Explanation Sample 1](C:\Users\subhe\.gemini\antigravity\brain\80df0391-a595-465d-9565-257ac448e073\individual_expB_sample1.png)

---

## 5. Repository Artifacts & Code Structure

The implementation is structured under [`credit-risk-replication/`](file:///o:/PROJECTS/Credit%20Risk%20Predictor/credit-risk-replication):

- **Data Pipeline**: [`src/data.py`](file:///o:/PROJECTS/Credit%20Risk%20Predictor/credit-risk-replication/src/data.py) & [`src/preprocessing.py`](file:///o:/PROJECTS/Credit%20Risk%20Predictor/credit-risk-replication/src/preprocessing.py)
- **Model**: [`src/model.py`](file:///o:/PROJECTS/Credit%20Risk%20Predictor/credit-risk-replication/src/model.py)
- **Loss Functions**: [`src/losses.py`](file:///o:/PROJECTS/Credit%20Risk%20Predictor/credit-risk-replication/src/losses.py)
- **Training & Evaluation**: [`src/train.py`](file:///o:/PROJECTS/Credit%20Risk%20Predictor/credit-risk-replication/src/train.py) & [`src/evaluate.py`](file:///o:/PROJECTS/Credit%20Risk%20Predictor/credit-risk-replication/src/evaluate.py)
- **Explainability**: [`src/explain.py`](file:///o:/PROJECTS/Credit%20Risk%20Predictor/credit-risk-replication/src/explain.py)
- **Orchestration**: [`run_all.py`](file:///o:/PROJECTS/Credit%20Risk%20Predictor/credit-risk-replication/run_all.py)
- **Configuration**: [`configs/german_credit.yaml`](file:///o:/PROJECTS/Credit%20Risk%20Predictor/credit-risk-replication/configs/german_credit.yaml)
- **Documentation**: [`README.md`](file:///o:/PROJECTS/Credit%20Risk%20Predictor/credit-risk-replication/README.md) & [`REPRODUCIBILITY_NOTES.md`](file:///o:/PROJECTS/Credit%20Risk%20Predictor/credit-risk-replication/REPRODUCIBILITY_NOTES.md)
- **Data & Metric Audits**: [`outputs/data_audit.json`](file:///o:/PROJECTS/Credit%20Risk%20Predictor/credit-risk-replication/outputs/data_audit.json), [`outputs/metrics/results.csv`](file:///o:/PROJECTS/Credit%20Risk%20Predictor/credit-risk-replication/outputs/metrics/results.csv), [`outputs/metrics/run_log.json`](file:///o:/PROJECTS/Credit%20Risk%20Predictor/credit-risk-replication/outputs/metrics/run_log.json)
