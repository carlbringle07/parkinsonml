# FILE: scripts/18_conformal_alpha_sweep_baseline_va.py

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
OUT_CSV = DATA_INDEX / "conformal_alpha_sweep_baseline_va.csv"


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


def split_conformal_pooled(model, X_tr, y_tr, X_cal, y_cal, X_te, y_te, alpha: float):
    model.fit(X_tr, y_tr)

    p_cal = model.predict_proba(X_cal)
    s_cal_0 = 1.0 - p_cal[:, 0]
    s_cal_1 = 1.0 - p_cal[:, 1]
    s_true = np.where(y_cal == 1, s_cal_1, s_cal_0)

    n = int(len(s_true))
    if n <= 0:
        return None

    # Conformal quantile level with +1 correction, then clamp to [0,1]
    level = (np.ceil((n + 1) * (1.0 - alpha)) / n)
    level = float(np.clip(level, 0.0, 1.0))

    # Use "higher" (discrete) quantile
    q = np.quantile(s_true, level, method="higher")

    p_te = model.predict_proba(X_te)
    s0 = 1.0 - p_te[:, 0]
    s1 = 1.0 - p_te[:, 1]

    pred0 = s0 <= q
    pred1 = s1 <= q
    set_size = pred0.astype(int) + pred1.astype(int)

    true_in = np.where(y_te == 1, pred1, pred0)
    coverage = float(np.mean(true_in))

    singleton = set_size == 1
    singleton_rate = float(np.mean(singleton))

    if singleton.any():
        pred_label = np.where(pred1, 1, 0)
        singleton_acc = float(np.mean(pred_label[singleton] == y_te[singleton]))
    else:
        singleton_acc = np.nan

    avg_set_size = float(np.mean(set_size))
    auc = roc_auc_score(y_te, p_te[:, 1]) if len(np.unique(y_te)) == 2 else np.nan

    return coverage, singleton_rate, singleton_acc, avg_set_size, auc


def _pm(x: np.ndarray):
    return float(np.mean(x)), float(np.std(x, ddof=0))


def main():
    df = pd.read_csv(PERSON_CSV)
    df["speaker"] = df["speaker"].astype(str)

    feat_cols = _get_feature_cols(df)
    y_all = _y(df)
    groups = df["speaker"].values

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
        ]
    )

    alphas = [0.05, 0.10, 0.20]
    outer = GroupShuffleSplit(n_splits=50, test_size=0.30, random_state=42)

    rows = []

    for alpha in alphas:
        covs, srs, saccs, ssizes, aucs = [], [], [], [], []
        used = 0
        skipped = 0

        for split_id, (train_idx, test_idx) in enumerate(outer.split(df[feat_cols], y_all, groups)):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=split_id + 123)
            X_train_all = train_df[feat_cols]
            y_train_all = _y(train_df)
            g_train_all = train_df["speaker"].values

            tr2_idx, cal_idx = next(inner.split(X_train_all, y_train_all, g_train_all))

            proper = train_df.iloc[tr2_idx]
            cal = train_df.iloc[cal_idx]

            y_proper = _y(proper)
            y_cal = _y(cal)
            y_test = _y(test_df)

            if len(np.unique(y_cal)) < 2 or len(np.unique(y_test)) < 2 or len(np.unique(y_proper)) < 2:
                skipped += 1
                continue

            res = split_conformal_pooled(
                model,
                proper[feat_cols], y_proper,
                cal[feat_cols], y_cal,
                test_df[feat_cols], y_test,
                alpha=alpha,
            )

            if res is None:
                skipped += 1
                continue

            coverage, singleton_rate, singleton_acc, avg_set_size, auc = res
            covs.append(coverage)
            srs.append(singleton_rate)
            if not np.isnan(singleton_acc):
                saccs.append(singleton_acc)
            ssizes.append(avg_set_size)
            aucs.append(auc)
            used += 1

        cov_m, cov_s = _pm(np.array(covs))
        sr_m, sr_s = _pm(np.array(srs))
        ss_m, ss_s = _pm(np.array(ssizes))
        auc_m, auc_s = _pm(np.array(aucs))
        if len(saccs) > 0:
            sa_m, sa_s = _pm(np.array(saccs))
        else:
            sa_m, sa_s = np.nan, np.nan

        rows.append(
            {
                "alpha": alpha,
                "n_splits_used": int(used),
                "n_splits_skipped": int(skipped),
                "coverage_mean": cov_m,
                "coverage_std": cov_s,
                "singleton_rate_mean": sr_m,
                "singleton_rate_std": sr_s,
                "unsure_rate_mean": 1.0 - sr_m,
                "singleton_acc_mean": sa_m,
                "singleton_acc_std": sa_s,
                "avg_set_size_mean": ss_m,
                "avg_set_size_std": ss_s,
                "auc_mean": auc_m,
                "auc_std": auc_s,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    print("=== DONE ===")
    print("Saved:", OUT_CSV)
    print("\nAlpha sweep summary:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
