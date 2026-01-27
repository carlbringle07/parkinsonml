# FILE: scripts/13_eval_repeated_splits_va_baseline_vs_praat.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[1]
DATA_INDEX = ROOT / "data_index"

BASELINE_CSV = DATA_INDEX / "features_va_person.csv"
PRAAT_CSV = DATA_INDEX / "features_va_person_plus_praat.csv"
OUT_CSV = DATA_INDEX / "compare_baseline_vs_praat_repeated.csv"


def _get_feature_cols(df: pd.DataFrame, exclude=("speaker", "group", "label")):
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _make_binary_y(df: pd.DataFrame, name: str) -> np.ndarray:
    """
    y in {0,1} where 1 = PD.
    Priority:
      A) label numeric 0/1
      B) label text contains 'pd' (robust)
      C) group == 'pd' (fallback)
    """
    # A) Numeric label
    if "label" in df.columns and pd.api.types.is_numeric_dtype(df["label"]):
        y = df["label"].astype(int).values
        if set(np.unique(y)).issubset({0, 1}):
            return y

    # B) Text label
    if "label" in df.columns:
        s = df["label"].astype(str).str.strip().str.lower()
        # Common in your data: 'pd' vs 'control_elderly'
        y = (s == "pd").astype(int).values
        # If that didn't find any PD, fallback to contains('pd')
        if y.sum() == 0 and s.str.contains("pd").any():
            y = s.str.contains("pd").astype(int).values
        return y

    # C) Group fallback
    if "group" in df.columns:
        g = df["group"].astype(str).str.strip().str.lower()
        return (g == "pd").astype(int).values

    raise RuntimeError(f"[{name}] Could not construct binary y: no usable label/group.")


def eval_repeated_auc(df: pd.DataFrame, feature_cols: list[str], name: str, n_splits=50, test_size=0.30, seed=42):
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)

    X = df[feature_cols].copy()
    y = _make_binary_y(df, name=name)
    groups = df["speaker"].astype(str).values

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
        ]
    )

    aucs = []
    n_train = []
    n_test = []
    skipped_small = 0

    for train_idx, test_idx in gss.split(X, y, groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            skipped_small += 1
            continue

        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, p)

        aucs.append(float(auc))
        n_train.append(int(len(y_tr)))
        n_test.append(int(len(y_te)))

    return np.array(aucs), np.array(n_train), np.array(n_test), skipped_small


def summarize(name: str, aucs: np.ndarray, n_train: np.ndarray, n_test: np.ndarray, skipped_small: int):
    return {
        "name": name,
        "n_splits_used": int(len(aucs)),
        "splits_skipped_no_both_classes": int(skipped_small),
        "auc_mean": float(np.mean(aucs)) if len(aucs) else np.nan,
        "auc_std": float(np.std(aucs)) if len(aucs) else np.nan,
        "auc_min": float(np.min(aucs)) if len(aucs) else np.nan,
        "auc_max": float(np.max(aucs)) if len(aucs) else np.nan,
        "train_n_mean": float(np.mean(n_train)) if len(n_train) else np.nan,
        "test_n_mean": float(np.mean(n_test)) if len(n_test) else np.nan,
    }


def main():
    df_base = pd.read_csv(BASELINE_CSV)
    df_praat = pd.read_csv(PRAAT_CSV)

    df_base["speaker"] = df_base["speaker"].astype(str)
    df_praat["speaker"] = df_praat["speaker"].astype(str)

    base_cols = _get_feature_cols(df_base)
    praat_cols = _get_feature_cols(df_praat)

    print("Baseline persons:", len(df_base), "features:", len(base_cols))
    print("Praat+Baseline persons:", len(df_praat), "features:", len(praat_cols))

    # Debug label/group distribution for BOTH
    if "label" in df_base.columns:
        print("Baseline label sample:", df_base["label"].astype(str).value_counts().to_dict())
    if "group" in df_base.columns:
        print("Baseline group sample:", df_base["group"].astype(str).value_counts().to_dict())

    if "label" in df_praat.columns:
        print("Praat label sample:", df_praat["label"].astype(str).value_counts().to_dict())
    if "group" in df_praat.columns:
        print("Praat group sample:", df_praat["group"].astype(str).value_counts().to_dict())

    # DEBUG NaNs per column for Praat dataframe (top 10)
    na_counts = df_praat[praat_cols].isna().sum().sort_values(ascending=False)
    print("\nPraat NaN counts (top 10):")
    print(na_counts.head(10).to_string())

    # Also debug y distributions
    y_base = _make_binary_y(df_base, "baseline")
    y_praat = _make_binary_y(df_praat, "baseline_plus_praat")
    print("\nY distribution:")
    print(" baseline:        n1 =", int(y_base.sum()), "n0 =", int((1 - y_base).sum()))
    print(" baseline+praat:  n1 =", int(y_praat.sum()), "n0 =", int((1 - y_praat).sum()))

    auc_b, tr_b, te_b, sk_b = eval_repeated_auc(df_base, base_cols, "baseline", n_splits=50, test_size=0.30, seed=42)
    auc_p, tr_p, te_p, sk_p = eval_repeated_auc(df_praat, praat_cols, "baseline_plus_praat", n_splits=50, test_size=0.30, seed=42)

    out = pd.DataFrame(
        [
            summarize("baseline", auc_b, tr_b, te_b, sk_b),
            summarize("baseline_plus_praat", auc_p, tr_p, te_p, sk_p),
        ]
    )
    out.to_csv(OUT_CSV, index=False)

    print("\n=== RESULTS ===")
    print(out.to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
