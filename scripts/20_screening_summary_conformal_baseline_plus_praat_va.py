# FILE: scripts/20_screening_summary_conformal_baseline_plus_praat_va.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[1]
DATA_INDEX = ROOT / "data_index"

PERSON_CSV = DATA_INDEX / "features_va_person_plus_praat.csv"
OUT_CSV = DATA_INDEX / "screening_summary_conformal_baseline_plus_praat_va_alpha0.10.csv"


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
        raise FileNotFoundError(f"Missing: {PERSON_CSV} (run scripts/12_extract_praat_features_va.py first)")

    df = pd.read_csv(PERSON_CSV)
    df["speaker"] = df["speaker"].astype(str)

    # Some merges may create duplicate/unnamed columns; drop any fully-empty columns
    df = df.dropna(axis=1, how="all")

    # Keep only pd vs control_elderly (safety)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["pd", "control_elderly"])].copy()

    feat_cols = _get_feature_cols(df)
    y_all = _y(df)
    groups = df["speaker"].values

    alpha = 0.10

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
        ]
    )

    outer = GroupShuffleSplit(n_splits=50, test_size=0.30, random_state=42)

    rows = []

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
            continue

        # Fit
        model.fit(proper[feat_cols], y_proper)

        # Cal scores
        p_cal = model.predict_proba(cal[feat_cols])
        s_cal_0 = 1.0 - p_cal[:, 0]
        s_cal_1 = 1.0 - p_cal[:, 1]
        s_true = np.where(y_cal == 1, s_cal_1, s_cal_0)

        n = len(s_true)
        level = _quantile_level(n, alpha)
        q = np.quantile(s_true, level, method="higher")

        # Test sets
        p_te = model.predict_proba(test_df[feat_cols])
        s0 = 1.0 - p_te[:, 0]
        s1 = 1.0 - p_te[:, 1]

        pred0 = s0 <= q
        pred1 = s1 <= q
        set_size = pred0.astype(int) + pred1.astype(int)

        # Map to 3 outcomes
        decided = set_size == 1
        pred_label = np.where(pred1, 1, 0)  # valid when decided

        n_test = len(y_test)
        n_dec = int(decided.sum())
        n_unsure = int((~decided).sum())

        # Coverage
        true_in = np.where(y_test == 1, pred1, pred0)
        coverage = float(np.mean(true_in))

        # Accuracy on decided
        acc_dec = float(np.mean(pred_label[decided] == y_test[decided])) if n_dec > 0 else np.nan

        # Decision counts
        n_pd_dec = int(np.sum(decided & (pred_label == 1)))
        n_ctrl_dec = int(np.sum(decided & (pred_label == 0)))

        # Precision on decided
        tp = int(np.sum(decided & (pred_label == 1) & (y_test == 1)))
        fp = int(np.sum(decided & (pred_label == 1) & (y_test == 0)))
        tn = int(np.sum(decided & (pred_label == 0) & (y_test == 0)))
        fn = int(np.sum(decided & (pred_label == 0) & (y_test == 1)))

        prec_pd = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        prec_ctrl = tn / (tn + fn) if (tn + fn) > 0 else np.nan

        rows.append(
            {
                "split_id": int(split_id),
                "n_test": int(n_test),
                "n_features": int(len(feat_cols)),
                "coverage": coverage,
                "decided_rate": n_dec / n_test,
                "unsure_rate": n_unsure / n_test,
                "acc_decided": acc_dec,
                "prec_pd_decided": prec_pd,
                "prec_ctrl_decided": prec_ctrl,
                "pd_decision_rate": n_pd_dec / n_test,
                "ctrl_decision_rate": n_ctrl_dec / n_test,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    def pm(x):
        x = x.dropna().values
        return float(np.mean(x)), float(np.std(x, ddof=0))

    print("=== DONE ===")
    print("Saved:", OUT_CSV)
    print("Features used:", int(len(feat_cols)))
    print("\nSummary (mean ± std over splits):")
    for col in [
        "coverage",
        "decided_rate",
        "unsure_rate",
        "acc_decided",
        "prec_pd_decided",
        "prec_ctrl_decided",
        "pd_decision_rate",
        "ctrl_decision_rate",
    ]:
        m, s = pm(out[col])
        print(f"{col}: {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()
