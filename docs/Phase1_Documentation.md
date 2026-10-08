# Home Credit Default Risk — Phase 1 Documentation
### Data Understanding, Cleaning & Baseline Model

---

## 1. Problem Statement

**Goal:** Predict whether a loan applicant will default (fail to repay) or repay successfully, using their application data.

**Why it matters:** Lenders can't manually review every application. A model that scores applicants by risk lets a bank approve safe applicants faster and flag risky ones for review — a real, widely used application of ML in fintech/banking.

**Target variable:** `TARGET` — `1` = defaulted, `0` = repaid.

---

## 2. Dataset

- **Source:** Home Credit Default Risk (Kaggle competition, 2018) — `application_train.csv`
- **Size:** 307,511 rows × 122 columns (before cleaning)
- **One row = one loan applicant.** Columns describe income, employment, family status, education, housing, credit bureau inquiries, and more.

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Class Balance
```
TARGET
0 (repaid)     91.93%
1 (defaulted)   8.07%
```
**Finding:** Severe class imbalance. This single fact drove several later decisions:
- Plain accuracy is a misleading metric — a model predicting "no default" for everyone would score ~92% accuracy while catching zero real defaulters.
- Chose **ROC-AUC, precision, recall** as the real evaluation metrics instead.
- Used `class_weight='balanced'` in the model to force it to pay attention to the minority (defaulter) class.

### 3.2 Missing Values
- 41 columns had >50% missing values — almost all were building/property detail columns (`COMMONAREA_*`, `LIVINGAPARTMENTS_*`, `FLOORSMIN_*`, `ELEVATORS_*`, etc.), likely missing together because certain applicant/housing types never had this data collected.
- **Exception:** `EXT_SOURCE_1/2/3` (external credit score sources) were kept even though `EXT_SOURCE_1` was 56% missing, because these are known to be strong predictors — dropping a highly predictive column just because it's incomplete would hurt the model more than the missingness itself.

### 3.3 Data Types
```
float64: 65 columns
int64:   41 columns
object:  16 columns (categorical/text — needed encoding)
```

---

## 4. Data Cleaning Decisions

| Step | Action | Reasoning |
|---|---|---|
| Drop >50% missing (property detail columns) | Dropped 40 columns | Too sparse to impute reliably; low information value |
| Drop remaining columns with high-gap, low-value data | Dropped 8 more: `FLOORSMAX_AVG/MODE/MEDI`, `YEARS_BEGINEXPLUATATION_AVG/MODE/MEDI`, `TOTALAREA_MODE`, `EMERGENCYSTATE_MODE` | Still ~47-50% missing, building-related, and not strong predictors |
| `EXT_SOURCE_1/2/3` | Filled with **median** | Median is robust to skew, unlike mean; kept because these are strong predictors despite missingness |
| `OCCUPATION_TYPE`, `NAME_TYPE_SUITE` | Filled with `"Unknown"` category | The absence of this data may itself be meaningful (e.g. unemployed applicants); safer than guessing a job |
| `AMT_REQ_CREDIT_BUREAU_*` (credit inquiry counts) | Filled with **0** | Missing almost certainly means "no inquiry was made" — filling with median would be factually wrong here |
| Remaining small-gap numeric columns (<1,021 missing rows each) | Filled with **median** | Negligible volume; median is a safe default |

**Result:** 307,511 rows × 122 columns → 82 after the first drop (40 columns) → **74** after the second drop (8 columns) and all imputation, with zero missing values.

---

## 5. Categorical Encoding

**Concept:** Models require numeric input — text categories must be converted to numbers.

| Encoding type | Applied to | Why |
|---|---|---|
| **One-hot encoding** (`pd.get_dummies`, `drop_first=True`) | `NAME_CONTRACT_TYPE`, `CODE_GENDER`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`, `NAME_TYPE_SUITE`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `WEEKDAY_APPR_PROCESS_START` (≤10 categories each) | Low cardinality — one-hot keeps column count manageable. `drop_first=True` avoids the dummy-variable trap for binary columns |
| **Frequency encoding** (category → its % frequency in data) | `OCCUPATION_TYPE` (19 categories), `ORGANIZATION_TYPE` (58 categories) | One-hot would have created 58+ mostly-empty sparse columns; frequency encoding compresses this into one informative numeric column |

**Result:** 307,511 rows × 103 fully numeric columns. This includes `TARGET` and `SK_ID_CURR`, which are removed before modeling, so the model trains on **101 features**.

---

## 6. Train/Test Split

- **Method:** Stratified 80/20 split (`train_test_split(..., stratify=y, random_state=42)`)
- **Why stratified:** A random split could accidentally produce a test set with a different default rate than the true 92/8 ratio, making evaluation unreliable. Stratification forces both sets to preserve the original class balance.
- **Result (verified identical ratios):**
  - Train: 246,008 rows — 91.93% / 8.07%
  - Test: 61,503 rows — 91.93% / 8.07%
    
---

## 7. Baseline Model — Logistic Regression

**Why start with a simple model:** Establishes a performance floor. Any more complex model (Random Forest, XGBoost) built later must beat this number to justify its added complexity — a key point to make in interviews.

**Preprocessing specific to this model:**
- **Feature scaling** (`StandardScaler`) — Logistic Regression is sensitive to feature magnitude differences; tree-based models later won't need this step.
- Scaler was **fit only on training data**, then applied to test data — fitting on test data would cause data leakage (letting test information influence training).
- **`class_weight='balanced'`** — forces the model to weigh the minority (defaulter) class more heavily, countering the 92/8 imbalance.

---

## 8. Baseline Results

```
ROC-AUC: 0.7475

              precision    recall  f1-score   support
0 (repaid)       0.96       0.69      0.80     56538
1 (default)      0.16       0.67      0.26      4965

Confusion Matrix:
                Predicted: 0     Predicted: 1
Actual: 0          38,997           17,541
Actual: 1           1,615            3,350
```

### Interpretation
- **ROC-AUC of 0.7475** is a reasonable baseline for a single-table logistic model. Top Kaggle solutions scored higher, but they used many additional tables and heavy feature engineering, so they are not a fair comparison.
- **Recall for defaulters = 0.67:** the model correctly identified 3,350 of 4,965 actual defaulters.
- **Precision for defaulters = 0.16:** many false alarms (17,541 non-defaulters flagged as risky).
- **Why this tradeoff is acceptable here:** In credit risk, missing an actual defaulter (false negative) is typically far more costly to a lender than an unnecessary manual review (false positive). Prioritizing recall over precision on the minority class is the defensible choice.
- **Accuracy (69%) is intentionally lower** than a naive "always predict repay" model (~92%) — because that naive model would have 0% recall on defaulters and be useless in practice. The 69% here reflects an honest, imbalance-aware model.

---

## 9. Phase 1 Deliverables (Complete)

- ✅ Cleaned dataset: 307,511 rows × 103 numeric columns (101 model features after removing `TARGET` and `SK_ID_CURR`), zero missing values
- ✅ Documented, reasoned cleaning decisions (not blanket drop/fill rules)
- ✅ Fully encoded categorical features (one-hot + frequency encoding)
- ✅ Stratified train/test split, verified balanced
- ✅ Baseline Logistic Regression model with imbalance-aware metrics
- ✅ Baseline established: **ROC-AUC = 0.7475**, Recall (defaulters) = 0.67

**Next (Phase 2):** Random Forest and XGBoost, hyperparameter tuning, cross-validation, and SHAP explainability — all benchmarked against this 0.7475 baseline.

---

## 10. Later Update: Leakage Audit

The pipeline described above computed medians and frequency maps on the full dataset before the train/test split. This was later corrected in `notebooks/phase1b_leakage_fix.ipynb`, where the split happens first and every statistic is learned from the training set only. Everything above, including the Section 8 results, is from the original run.

**Corrected baseline (same applicants in train and test):** ROC-AUC 0.7476, recall 0.67, precision 0.16. Confusion matrix: 39,007 / 17,531 / 1,618 / 3,347. The change from 0.7475 is negligible.
