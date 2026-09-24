# Home Credit Default Risk Prediction

Predicting loan default risk using real-world application data, with a focus on rigorous evaluation, imbalance handling, and explainability — not just training a model.

---

## Problem Statement

Lenders can't manually review every loan application. This project builds a model that predicts whether an applicant will **default** (fail to repay) based on their application data — helping flag high-risk applicants for review while approving safe ones faster.

**Target:** `TARGET` — `1` = defaulted, `0` = repaid.

---

## Dataset

- **Source:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle, 2018)
- **File used:** `application_train.csv` — 307,511 applicants, 122 raw features
- Real-world data: heavily imbalanced target (~92% repaid / ~8% defaulted), significant missingness, mixed categorical/numeric features

---

## Approach

### Phase 1 — Data Understanding, Cleaning & Baseline ✅ Complete

- **EDA:** Identified severe class imbalance (92/8), mapped missing-value patterns, checked data types
- **Cleaning:** Dropped 40 low-value, high-missing columns (property details); kept `EXT_SOURCE_1/2/3` despite missingness because they're known strong predictors; used context-aware imputation (0 for "no inquiry made" columns, median for continuous features, "Unknown" category for missing categoricals)
- **Encoding:** One-hot encoding for low-cardinality categoricals, frequency encoding for high-cardinality ones (`ORGANIZATION_TYPE`, `OCCUPATION_TYPE`) to avoid sparse one-hot bloat
- **Split:** Stratified 80/20 train/test split to preserve class ratio
- **Baseline model:** Logistic Regression with feature scaling and `class_weight='balanced'`

**Baseline results:**
| Metric | Score |
|---|---|
| ROC-AUC | 0.7475 |
| Recall (defaulters) | 0.67 |
| Precision (defaulters) | 0.16 |

Full write-up: [`docs/phase1_documentation.md`](docs/Phase1_Documentation.md)

**Why this baseline matters:** it establishes the floor every later model must beat, and the recall-over-precision tradeoff was a deliberate choice — in credit risk, missing a real defaulter costs the lender far more than a false alarm.

### Phase 2 — Modeling & Explainability (In Progress)

- [ ] Random Forest
- [ ] XGBoost
- [ ] Hyperparameter tuning (GridSearchCV/Optuna)
- [ ] K-fold cross-validation
- [ ] SHAP explainability (global + local)

### Phase 3 — Deployment (Planned)

- [ ] FastAPI prediction endpoint
- [ ] Streamlit frontend
- [ ] Free-tier deployment (Render/HF Spaces)
- [ ] Model card & final documentation

---

## Tech Stack

Python, pandas, scikit-learn, XGBoost, SHAP, FastAPI, Streamlit

---

## Repository Structure

```
home-credit-default-risk-prediction/
├── README.md
├── notebooks/
│   └── phase1_eda_baseline.ipynb
├── docs/
│   └── phase1_documentation.md
└── requirements.txt
```

---

## Status

🚧 Actively being built — Phase 1 complete, Phase 2 in progress.
