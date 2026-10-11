"""Score one loan applicant using the saved model and preprocessing artifacts."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# Learned from the training set only (phase1b_leakage_fix.ipynb)
medians = joblib.load(ARTIFACTS / "medians.pkl")
freq_maps = joblib.load(ARTIFACTS / "freq_maps.pkl")
train_columns = joblib.load(ARTIFACTS / "train_columns.pkl")
config = joblib.load(ARTIFACTS / "config.pkl")

model = XGBClassifier()
model.load_model(ARTIFACTS / "xgb_model.json")

# Raw input columns the pipeline expects (after the 48 dropped ones).
# One-hot columns look like "CODE_GENDER_M", so strip those and add back "CODE_GENDER".
_dummy_prefixes = tuple(c + "_" for c in config["onehot_cols"])
RAW_COLUMNS = [c for c in train_columns if not c.startswith(_dummy_prefixes)] + config["onehot_cols"]


def preprocess(raw: dict) -> pd.DataFrame:
    """Turn one raw applicant (dict of original column names) into the 101 model features."""
    raw = {k: (np.nan if v is None else v) for k, v in raw.items()}
    row = pd.DataFrame([raw]).reindex(columns=RAW_COLUMNS)   # missing fields -> NaN, extra keys ignored

    row = row.fillna(medians)                                 # learned medians (train only)
    for col in config["unknown_cols"]:
        row[col] = row[col].astype("object").fillna("Unknown")
    row[config["bureau_cols"]] = row[config["bureau_cols"]].fillna(0)

    for col in config["freq_cols"]:                           # unseen category -> 0
        row[col] = row[col].map(freq_maps[col]).fillna(0)

    row = pd.get_dummies(row, columns=config["onehot_cols"], drop_first=False, dtype=int)
    return row.reindex(columns=train_columns, fill_value=0)   # same 101 columns, same order


def predict_score(raw: dict) -> float:
    """Risk score between 0 and 1. Higher = riskier. Not a calibrated probability."""
    return float(model.predict_proba(preprocess(raw))[0, 1])


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sample_applicant.json"
    with open(path) as f:
        applicant = json.load(f)
    print("Risk score:", round(predict_score(applicant), 6))
