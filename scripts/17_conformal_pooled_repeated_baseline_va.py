# FILE: scripts/17_conformal_pooled_repeated_baseline_va.py

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

PERSON_CSV = DATA_INDEX / "features_va_person.csv"
OUT_CSV = DATA_INDEX / "conformal_pooled_repeated_baseline_va_alpha0.10.csv"


def _get_feature_cols(df: pd.DataFrame, exclude=("speaker", "group", "label")):
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _y(df: pd.DataFrame) -> np.ndarray:
    s = df["label"].astype(str).str.strip().str.lower()
    return (s == "pd").astype(int).values


def _prob_pos(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def split_conformal_pooled(
    model: Pipeline,
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_cal: pd.DataFrame, y_cal: np.ndarray,
    X_te: pd.DataFrame, y_te: np.ndarray,
    alpha: float
):
    model.fit(X_tr, y_tr)

    # nonconformity for class k: s_k(x) = 1 - p_k(x)
    p_cal = model.predict_proba(X_cal)
    s_cal_0 = 1.0 - p_cal[:, 0]
    s_cal_1 = 1.0 - p_cal[:, 1]

    # For each calibration example i, take score for its true label
    s_true = np.where(y_cal == 1, s_cal_1, s_cal_0)

    n = len(s_true)
    # conformal quantile with +1 correction
    q = np.quantile(s_true, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")

    # Predict sets for test
    p_te = model.predict_proba(X_te)
    s0 = 1.0 - p_te[:, 0]
    s1 = 1.0 - p_te[:, 1]

    pred0 = s0 <= q
    pred1 = s1 <= q

    # set size: 0..2
    set_size = pred0.astype(int) + pred1.astype(int)

    # coverage: true label included
    true_in = np.where(y_te == 1, pred1, pred0)
    coverage = float(np.mean(true_in))

    # singleton decisions: exactly one label in set
    singleton = set_size == 1
    singleton_rate = float(np.mean(singleton))

    # accuracy among singleton decisions
    if singleton.any():
        pred_label = np.where(pred1, 1, 0)  # if {1} then 1 else 0 (only valid when singleton)
        singleton_acc = float(np.mean(pred_label[singleton] == y_te[singleton]))
    else:
        singleton_acc = np.nan

    avg_set_size = float(np.mean(set_size))

    auc = roc_auc_score(y_te, p_te[:, 1]) if len(np.unique(y_te)) == 2 else np.nan

    return {
        "q": float(q),
        "coverage": coverage,
        "singleton_rate": singleton_rate,
        "singleton_acc": singleton_acc,
        "avg_set_size": avg_set_size,
        "auc": float(auc),
        "n_test": int(len(y_te)),
        "n_cal": int(len(y_cal)),
        "n_train": int(len(y_tr)),
    }


def main():
    df = pd.read_csv(PERSON_CSV)
    df["speaker"] = df["speaker"].astype(str)

    feat_cols = _get_feature_cols(df)
    y_all = _y(df)
    groups = df["speaker"].values

    print("Persons:", len(df), "PD:", int(y_all.sum()), "CTRL:", int((1 - y_all).sum()))
    print("Features:", len(feat_cols))

    # Base model
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
        ]
    )

    alpha = 0.10
    outer = GroupShuffleSplit(n_splits=50, test_size=0.30, random_state=42)

    rows = []

    for split_id, (train_idx, test_idx) in enumerate(outer.split(df[feat_cols], y_all, groups)):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # within-train: split into proper-train and calibration (speaker split again)
        inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=split_id + 123)
        X_train_all = train_df[feat_cols]
        y_train_all = _y(train_df)
        g_train_all = train_df["speaker"].values

        (tr2_idx, cal_idx) = next(inner.split(X_train_all, y_train_all, g_train_all))

        proper = train_df.iloc[tr2_idx]
        cal = train_df.iloc[cal_idx]

        y_proper = _y(proper)
        y_cal = _y(cal)
        y_test = _y(test_df)

        # Need both classes in cal and test
        if len(np.unique(y_cal)) < 2 or len(np.unique(y_test)) < 2 or len(np.unique(y_proper)) < 2:
            continue

        res = split_conformal_pooled(
            model,
            proper[feat_cols], y_proper,
            cal[feat_cols], y_cal,
            test_df[feat_cols], y_test,
            alpha=alpha,
        )
        res["split_id"] = int(split_id)
        rows.append(res)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    print("\n=== DONE ===")
    print("Saved:", OUT_CSV)

    def pm(x):
        return (float(np.mean(x)), float(np.std(x, ddof=0)))

    print("\nSummary (mean ± std):")
    for col in ["coverage", "singleton_rate", "singleton_acc", "avg_set_size", "auc"]:
        m, s = pm(out[col].dropna().values)
        print(f"{col}: {m:.3f} ± {s:.3f}  (n={out[col].dropna().shape[0]})")


if __name__ == "__main__":
    main()
