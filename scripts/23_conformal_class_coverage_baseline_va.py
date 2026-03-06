# FILE: scripts/23_conformal_class_coverage_baseline_va.py

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
OUT_CSV = DATA_INDEX / "conformal_class_coverage_baseline_va_alpha0.10.csv"

N_SPLITS = 50
TEST_SIZE = 0.30
SEED = 42
ALPHA = 0.10
CAL_WITHIN_TRAIN = 0.50


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


def _quantile_level(n: int, alpha: float) -> float:
    level = (np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.clip(level, 0.0, 1.0))


def main():
    if not PERSON_CSV.exists():
        raise FileNotFoundError(f"Missing: {PERSON_CSV} (run scripts/01_extract_features_va.py first)")

    df = pd.read_csv(PERSON_CSV)
    df["speaker"] = df["speaker"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["pd", "control_elderly"])].copy()

    feat_cols = _get_feature_cols(df)
    y_all = _y(df)
    groups = df["speaker"].values

    print(f"Persons: {df['speaker'].nunique()}  PD: {(y_all==1).sum()}  CTRL: {(y_all==0).sum()}")
    print(f"Features: {len(feat_cols)}  Splits: {N_SPLITS}  Test size: {TEST_SIZE}  alpha: {ALPHA}")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
        ]
    )

    outer = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)

    rows = []
    used = 0
    skipped = 0

    for split_id, (train_idx, test_idx) in enumerate(outer.split(df[feat_cols], y_all, groups)):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        inner = GroupShuffleSplit(n_splits=1, test_size=CAL_WITHIN_TRAIN, random_state=split_id + 123)
        X_train_all = train_df[feat_cols]
        y_train_all = _y(train_df)
        g_train_all = train_df["speaker"].values

        tr2_idx, cal_idx = next(inner.split(X_train_all, y_train_all, g_train_all))
        proper = train_df.iloc[tr2_idx]
        cal = train_df.iloc[cal_idx]

        y_proper = _y(proper)
        y_cal = _y(cal)
        y_test = _y(test_df)

        # need both classes everywhere
        if len(np.unique(y_proper)) < 2 or len(np.unique(y_cal)) < 2 or len(np.unique(y_test)) < 2:
            skipped += 1
            continue

        model.fit(proper[feat_cols], y_proper)

        # calibration scores
        p_cal = model.predict_proba(cal[feat_cols])
        s_cal_0 = 1.0 - p_cal[:, 0]
        s_cal_1 = 1.0 - p_cal[:, 1]
        s_true = np.where(y_cal == 1, s_cal_1, s_cal_0)

        n = len(s_true)
        level = _quantile_level(n, ALPHA)
        q = np.quantile(s_true, level, method="higher")

        # test
        p_te = model.predict_proba(test_df[feat_cols])
        p_pd = p_te[:, 1]

        s0 = 1.0 - p_te[:, 0]
        s1 = 1.0 - p_te[:, 1]

        pred0 = s0 <= q
        pred1 = s1 <= q
        set_size = pred0.astype(int) + pred1.astype(int)

        # overall coverage
        true_in = np.where(y_test == 1, pred1, pred0)
        cov_all = float(np.mean(true_in))

        # per-class coverage
        mask_pd = y_test == 1
        mask_ctrl = y_test == 0
        cov_pd = float(np.mean(pred1[mask_pd])) if mask_pd.any() else np.nan
        cov_ctrl = float(np.mean(pred0[mask_ctrl])) if mask_ctrl.any() else np.nan

        # unsure & decided
        decided = set_size == 1
        decided_rate = float(np.mean(decided))
        unsure_rate = float(np.mean(~decided))

        # AUC for reference (probability)
        auc = roc_auc_score(y_test, p_pd)

        rows.append(
            {
                "split_id": int(split_id),
                "n_test": int(len(test_df)),
                "cov_all": cov_all,
                "cov_pd": cov_pd,
                "cov_ctrl": cov_ctrl,
                "decided_rate": decided_rate,
                "unsure_rate": unsure_rate,
                "auc": float(auc),
            }
        )
        used += 1

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    def pm(series: pd.Series):
        x = series.dropna().values
        return float(np.mean(x)), float(np.std(x, ddof=0))

    print("\n=== DONE ===")
    print(f"Splits used: {used}   skipped: {skipped}")
    print("Saved:", OUT_CSV)
    print("\nSummary (mean ± std over splits):")
    for col in ["cov_all", "cov_pd", "cov_ctrl", "decided_rate", "unsure_rate", "auc"]:
        m, s = pm(out[col])
        print(f"{col}: {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()
