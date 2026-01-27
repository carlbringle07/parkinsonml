# FILE: scripts/27_eval_repeated_splits_vowels_pack.py

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA_INDEX = ROOT / "data_index"

PERSON_CSV = DATA_INDEX / "features_vowels_pack_person.csv"
OUT_CSV = DATA_INDEX / "eval_repeated_splits_vowels_pack.csv"

N_SPLITS = 50
TEST_SIZE = 0.30
SEED = 42


def _get_feature_cols(df: pd.DataFrame):
    exclude = {"speaker", "group", "label"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def main():
    if not PERSON_CSV.exists():
        raise FileNotFoundError(f"Missing: {PERSON_CSV} (run scripts/26_extract_features_vowels_pack.py first)")

    df = pd.read_csv(PERSON_CSV)
    df["speaker"] = df["speaker"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    df = df[df["label"].isin(["pd", "control_elderly"])].copy()

    feat_cols = _get_feature_cols(df)
    y = (df["label"] == "pd").astype(int).values
    groups = df["speaker"].values

    print(f"Persons: {df['speaker'].nunique()}  PD: {(y==1).sum()}  CTRL: {(y==0).sum()}")
    print(f"Features: {len(feat_cols)}  Splits: {N_SPLITS}  Test size: {TEST_SIZE}")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
        ]
    )

    splitter = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)

    rows = []
    aucs = []
    skipped = 0

    for split_id, (tr, te) in enumerate(splitter.split(df[feat_cols], y, groups)):
        y_tr = y[tr]
        y_te = y[te]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            skipped += 1
            continue

        model.fit(df.iloc[tr][feat_cols], y_tr)
        p = model.predict_proba(df.iloc[te][feat_cols])[:, 1]
        auc = roc_auc_score(y_te, p)
        aucs.append(float(auc))
        rows.append({"split_id": split_id, "auc": float(auc), "n_train": int(len(tr)), "n_test": int(len(te))})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    aucs = np.array(aucs) if len(aucs) else np.array([np.nan])

    print("\n=== DONE ===")
    print("Saved:", OUT_CSV)
    print(f"Splits used: {len(out)}  skipped: {skipped}")
    print(f"AUC mean: {np.nanmean(aucs):.3f}")
    print(f"AUC std:  {np.nanstd(aucs):.3f}")
    print(f"AUC min:  {np.nanmin(aucs):.3f}")
    print(f"AUC max:  {np.nanmax(aucs):.3f}")


if __name__ == "__main__":
    main()
