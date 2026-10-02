# Home Credit Default Risk — Phase 2 Documentation
### Modeling, Tuning, Cross-Validation & Explainability

---

## 1. Goal

Beat the Phase 1 baseline (Logistic Regression, ROC-AUC = 0.7475) using more powerful models, then validate and explain the winning model — not just chase a higher number, but understand and justify every step.

**Models tried, in order:** Random Forest → XGBoost → Hyperparameter-tuned XGBoost.

---

## 2. Random Forest

**Setup:** `RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)`, trained on unscaled features (tree-based models don't need feature scaling — they split on raw thresholds, unlike Logistic Regression's weighted-sum math).

**Result (default 0.5 threshold):**

| Metric | Score |
|---|---|
| ROC-AUC | 0.7282 |
| Recall (defaulters) | 0.00 |
| Precision (defaulters) | 0.53 (misleading — only 15 total positive predictions) |

**What went wrong:** `class_weight='balanced'` did not translate into useful predictions. In Logistic Regression, class weighting directly reweights the loss function used to fit the single model. In Random Forest, class weighting only affects how *each individual tree* evaluates split quality (Gini impurity) — the final prediction is still a majority vote/averaged probability across 100 trees. Averaging dilutes the minority-class signal: even though individual trees "cared" more about defaulters during splitting, the aggregated probability for most applicants rarely crossed the default 0.5 cutoff.

**Threshold adjustment attempt:** Lowering the decision threshold from 0.5 to 0.15 (flagging anyone with ≥15% predicted default probability as risky) partially recovered recall to 0.34 — better, but still well below the Logistic Regression baseline's 0.67.

*Note:* The 0.15 threshold was chosen by inspecting results on the test set, so it is an illustration of the effect, not a properly tuned value. A tuned threshold should be selected on validation data.

**Conclusion:** Random Forest, as configured, does not beat the baseline on the metric that matters most for this problem (recall on defaulters). This is a genuine, documented finding: **class imbalance handling behaves inconsistently across model architectures** — a technique effective for one model type isn't guaranteed to transfer to another. Rather than force Random Forest to compete, focus shifted to XGBoost.

---

## 3. XGBoost (Default Settings)

**Setup:** XGBoost doesn't support `class_weight='balanced'` — instead, it uses `scale_pos_weight`, a manually-supplied ratio of majority-to-minority class counts.

```python
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
# = 11.39, matching the known ~92/8 imbalance
```

**Result:**

| Metric | Score |
|---|---|
| ROC-AUC | 0.7489 |
| Recall (defaulters) | 0.62 |
| Precision (defaulters) | 0.17 |

**Why this worked better than Random Forest:** XGBoost is a **boosting** method — trees are built sequentially, each one correcting the errors of the previous trees, rather than being trained independently in parallel and averaged (bagging, as in Random Forest). This sequential error-correction keeps the model "paying attention" to harder, rarer cases like defaulters across rounds, instead of averaging their signal away.

**Already a marginal win over baseline on ROC-AUC** (0.7489 vs 0.7475), even before any tuning — a promising candidate to carry forward.

---

## 4. Hyperparameter Tuning (XGBoost)

**Method:** `RandomizedSearchCV` — chosen over exhaustive `GridSearchCV` for speed on a dataset this size (307K rows). Randomly samples a fixed number of hyperparameter combinations rather than testing every possible combination.

**Search space:**
```python
param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}
```
- **`max_depth`** — controls how deep each tree grows; too shallow underfits, too deep overfits (memorizes noise).
- **`learning_rate`** — how strongly each new tree corrects previous mistakes; lower values learn more cautiously but need more trees.
- **`subsample` / `colsample_bytree`** — fraction of rows/features each tree sees; introduces randomness to reduce overfitting, similar in spirit to Random Forest's bagging but applied within a boosting framework.

**Search configuration:** 20 randomly sampled combinations, each evaluated with 3-fold cross-validation (60 total model fits), optimizing for ROC-AUC.

**Best parameters found:**
```
max_depth: 5
learning_rate: 0.05
n_estimators: 300
subsample: 0.9
colsample_bytree: 0.8
```

**Best cross-validated ROC-AUC (mean score on the validation folds, computed inside the training set): 0.7551**

**Final evaluation on held-out test set (never seen during tuning):**

| Metric | Score |
|---|---|
| ROC-AUC | **0.7605** |
| Recall (defaulters) | **0.67** |
| Precision (defaulters) | 0.17 |

**Result:** Tuning improved ROC-AUC from 0.7489 → 0.7605 (about +1.2 points over default XGBoost, and about +1.3 points over the Logistic Regression baseline of 0.7475). It also *matched* the baseline's recall exactly (0.67), meaning the tuned model catches just as many real defaulters as the baseline, with better overall ranking ability and no added cost in missed defaulters.

---

## 5. Cross-Validation (Stability Check)

**Why this step, on top of the tuning search's own CV:** The tuning search's 3-fold CV was used *during the search* to pick hyperparameters. This is a separate, final check — 5-fold cross-validation on the **entire dataset**, using the already-tuned model, purely to confirm the test-set result (0.7605) isn't an artifact of one lucky 80/20 split.

```python
cv_scores = cross_val_score(best_xgb, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
```

**Result:**
- Fold scores ranged narrowly: **0.749 – 0.764**
- **Mean ROC-AUC: 0.7581**
- **Standard deviation: 0.0050**
- 
**Interpretation:** The very small standard deviation (0.0050) indicates the model's performance is stable and consistent regardless of which 20% of applicants end up in the test set. This confirms 0.7605 is a reliable estimate of real-world performance, not a fluke of one particular split.
---

## 6. SHAP Explainability

**Why this step matters:** A model that performs well but can't be explained is a liability in credit risk — regulators and lenders need to know *why* a model denies someone credit, not just that it's statistically accurate.

**Method:** `shap.TreeExplainer` on the tuned XGBoost model, computing SHAP values for the test set to see how each feature pushes individual predictions toward "default" or "repay."

**Global findings (top features):**

| Feature | Why it matters |
|---|---|
| `EXT_SOURCE_3`, `EXT_SOURCE_2`, `EXT_SOURCE_1` | Top 5 most important — external credit bureau scores. Low values push predictions toward default; high values push toward repayment. Confirms these known strong predictors were used correctly and the model learned the expected real-world direction. |
| `AMT_GOODS_PRICE`, `AMT_CREDIT` | Larger loan amounts increase predicted risk — sensible, since bigger loans carry more exposure. |
| `DAYS_BIRTH`, `DAYS_EMPLOYED` | Longer employment history and older age reduce predicted risk — consistent with real-world credit intuition (stability signals lower risk). |

**Fairness observation:** `CODE_GENDER_M` appears among the top 10 most influential features, with male applicants (encoded as 1) associated with higher predicted default risk. This is a pattern learned from the historical training data, not something intentionally engineered. **This is flagged as a limitation, not resolved in this project** — a real production deployment of a credit model would need a formal fairness audit before using gender-correlated features to influence lending decisions, regardless of their historical predictive power. Documenting this openly is itself part of responsible ML practice.

---

## 7. Phase 2 Summary

**Full model comparison:**

| Model | ROC-AUC | Recall (defaulters) | Precision (defaulters) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.7475 | 0.67 | 0.16 |
| Random Forest (default) | 0.7282 | 0.00* | 0.53* |
| XGBoost (default) | 0.7489 | 0.62 | 0.17 |
| **XGBoost (tuned)** | **0.7605** | **0.67** | 0.17 |

*See Section 2 — Random Forest's precision/recall at default threshold are not meaningful due to near-zero positive predictions.

**Phase 2 deliverables (complete):**
- ✅ Random Forest trained and evaluated — documented underperformance and the architectural reason for it
- ✅ XGBoost trained with correctly-calculated `scale_pos_weight`
- ✅ Hyperparameter tuning via RandomizedSearchCV (20 combinations × 3-fold CV)
- ✅ Final model validated via 5-fold cross-validation (mean 0.758, std 0.0050 — stable)
- ✅ SHAP global explainability, including an explicit fairness observation

**Final model carried into Phase 3:** Tuned XGBoost (`max_depth=5, learning_rate=0.05, n_estimators=300, subsample=0.9, colsample_bytree=0.8`), ROC-AUC 0.7605, Recall 0.67.

**Next (Phase 3):** Save this model, build a FastAPI prediction endpoint, a Streamlit frontend, deploy both free-tier, and write a model card documenting assumptions, limitations (including the fairness note above), and intended use.
