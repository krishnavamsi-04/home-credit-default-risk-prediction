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
- **Cleaning:** Dropped 48 low-value, high-missing property/building columns (40 with over 50% missing, plus 8 more with roughly 47–50% missing); kept `EXT_SOURCE_1/2/3` despite missingness because they're known strong predictors; used context-aware imputation (0 for "no inquiry made" columns, median for continuous features, "Unknown" category for missing categoricals)
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

### Phase 2 — Modeling & Explainability ✅ Complete
Full write-up: [`docs/Phase2_Documentation.md`](docs/Phase2_Documentation.md)

- [x] Random Forest
- [x] XGBoost
- [x] Hyperparameter tuning (RandomizedSearchCV)
- [x] K-fold cross-validation
- [x] SHAP explainability (global)

**SHAP explainability (global):**
Ran `TreeExplainer` on the tuned XGBoost model to verify *why* it makes its predictions, not just that it performs well.

- **`EXT_SOURCE_1/2/3` rank 5th, 2nd and 1st** — external credit scores are the strongest signals. Low scores push toward default, high scores toward repayment.
- **`AMT_CREDIT` (4th)** pushes toward default as it grows, while **`AMT_GOODS_PRICE` (3rd)** pushes the opposite way. The two are almost redundant (correlation 0.987), so the model likely uses the gap between them. Their SHAP values should be read together, not one at a time.
- **`DAYS_EMPLOYED` (7th)**: longer employment lowers predicted risk.  
- **Fairness note:** `CODE_GENDER_M` ranks 6th, with male applicants pushed toward higher predicted risk. This reflects a pattern in the historical data, not a deliberate design choice. A real deployment would need a fairness audit before use, and simply dropping the column would not be enough, since other features can act as proxies.
  
**Model comparison (test set):**
| Model | ROC-AUC | Recall (defaulters) | Precision (defaulters) | Pipeline |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.7475 | 0.67 | 0.16 | original |
| Logistic Regression (baseline) | 0.7476 | 0.67 | 0.16 | corrected |
| Random Forest (default) | 0.7282 | 0.00* | 0.53* | original |
| XGBoost (default) | 0.7489 | 0.62 | 0.17 | original |
| XGBoost (tuned) | 0.7605 | 0.67 | 0.17 | original |
| **XGBoost (tuned), final model** | **0.7611** | **0.67** | **0.17** | **corrected** |

"Original" means preprocessing statistics were learned before the split. "Corrected" means they were learned from the training set only (see the leakage audit below). Random Forest and default XGBoost were not re-run, because the leakage effect proved negligible.

*Random Forest's default 0.5 threshold produced near-zero recall despite `class_weight='balanced'`. Unlike Logistic Regression, where class weighting directly reweights the loss function, Random Forest only reweights split quality within each tree — the minority-class signal gets diluted when averaging predictions across 100 trees. Adjusting the decision threshold to 0.15 partially recovered recall (0.34) but still underperformed the baseline. **Finding:** class imbalance handling behaves inconsistently across model architectures — a technique that works for one model type isn't guaranteed to transfer to another.

**Tuning:** Used `RandomizedSearchCV` (20 combinations, 3-fold CV, optimizing for ROC-AUC) on XGBoost. Best parameters: `max_depth=5`, `learning_rate=0.05`, `n_estimators=300`, `subsample=0.9`, `colsample_bytree=0.8`. This improved ROC-AUC from 0.7489 → 0.7605 while matching the baseline's recall (0.67) — a genuine improvement in ranking ability with no cost to the defaulter-catch rate.

**Cross-validation (original pipeline: 5-fold, tuned XGBoost, full dataset):**
Mean ROC-AUC: **0.7581** | Std deviation: **0.0050** (scores ranged 0.749–0.764 across folds)

Tight, consistent scores across all 5 folds confirm the 0.7605 test-set result is a stable, reliable estimate of model performance — not an artifact of one particular train/test split.

### Leakage audit & corrected pipeline
After Phase 2, I audited the pipeline and found that medians and frequency maps were learned before the split. I rebuilt preprocessing so that the split happens first, every statistic is learned from the training set only, and the test set is aligned to the training columns. The same applicants are in train and test as before, so old and new results are directly comparable. All learned values (`medians`, `freq_maps`, `train_columns`, `config`) and the final model are saved as artifacts for the Phase 3 API.

| Model | ROC-AUC (original) | ROC-AUC (corrected) |
|---|---|---|
| Logistic Regression | 0.7475 | 0.7476 |
| XGBoost (tuned) | 0.7605 | 0.7611 |

5-fold CV on the training set (corrected pipeline): mean 0.7561, std 0.0013.

### Phase 3 — Deployment (Planned)

- [ ] FastAPI prediction endpoint
- [ ] Streamlit frontend
- [ ] Free-tier deployment (Render/HF Spaces)
- [ ] Model card & final documentation

---

## How to Reproduce

1. Download `application_train.csv` from the [Kaggle competition page](https://www.kaggle.com/c/home-credit-default-risk/data) (a Kaggle account is required).
2. Notebooks were developed on Google Colab. Phase 1 reads the CSV from `/content/` and saves the processed splits (`X_train.pkl`, etc.) to Google Drive. Phase 2 loads them from Drive. Update the paths if you run locally.
3. Install dependencies: `pip install -r requirements.txt`
4. Run `notebooks/phase1_eda_baseline.ipynb`, then `notebooks/phase2_modeling.ipynb`.

All random seeds are fixed (`random_state=42`).

## Known Limitations

- **Fairness:** `CODE_GENDER_M` is among the influential features. Using a gender-correlated feature in real lending decisions would require a formal fairness audit.
- **Preprocessing leakage (found and fixed):** In the first version, median imputation values and frequency-encoding maps were computed on the full dataset before the train/test split. This was corrected in `phase1b_leakage_fix.ipynb`, where all learned statistics come from the training set only. Re-measured impact: ROC-AUC changed from 0.7475 to 0.7476 (Logistic Regression) and from 0.7605 to 0.7611 (tuned XGBoost), so the leakage had no meaningful effect at this dataset size. The cross-validation score (0.7561) still uses preprocessing fitted on all of the training set; a full fix would refit it inside each fold with a Pipeline.
- **Single table only:** Only `application_train.csv` is used. The other Home Credit tables (bureau, previous applications) are not included.
- **Low precision:** Precision for defaulters is about 0.17, so many flagged applicants would not actually default. This is a deliberate recall-over-precision tradeoff, but a real system would need a manual-review process.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, SHAP, FastAPI, Streamlit

---

## Repository Structure

```
home-credit-default-risk-prediction/
├── README.md
├── notebooks/
│   ├── phase1_eda_baseline.ipynb
│   └── phase2_modeling.ipynb
├── docs/
│   ├── Phase1_Documentation.md
│   └── Phase2_Documentation.md
└── requirements.txt
```

## Status

🚧 Phase 1 and Phase 2 complete. Final model: tuned XGBoost on the corrected pipeline, ROC-AUC 0.7611 on the test set (5-fold CV mean 0.7561). Phase 3 (deployment) is next.
