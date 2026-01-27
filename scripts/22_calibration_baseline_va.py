# FILE: scripts/22_calibration_baseline_va.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss


ROOT = Path(__file__).resolve().parents[1]
DATA_INDEX = ROOT / "data_index"

PERSON_CSV = DATA_INDEX / "features_va_person.csv"

OUT_METRICS = DATA_INDEX / "calibration_baseline_va_metrics.csv"
OUT_BINS = DATA_INDEX / "calibration_baseline_va_bins.csv"
OUT_FIG = DATA_INDEX / "calibration_baseline_va_reliability.png"

N_SPLITS = 50
TEST_SIZE = 0.30
SEED = 42
N_BINS = 10


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


def _bin_stats(y_true: np.ndarray, p: np.ndarray, n_bins: int):
    """
    Fixed bins on [0,1]. Returns per-bin:
    - n
    - mean_pred (avg predicted probability)
    - frac_pos (empirical positive rate)
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_n = np.zeros(n_bins, dtype=int)
    mean_pred = np.full(n_bins, np.nan, dtype=float)
    frac_pos = np.full(n_bins, np.nan, dtype=float)

    # Put p==1.0 in last bin
    idx = np.minimum(np.digitize(p, edges, right=False) - 1, n_bins - 1)
    idx = np.maximum(idx, 0)

    for b in range(n_bins):
        mask = idx == b
        nb = int(mask.sum())
        bin_n[b] = nb
        if nb > 0:
            mean_pred[b] = float(p[mask].mean())
            frac_pos[b] = float(y_true[mask].mean())

    return edges, bin_n, mean_pred, frac_pos


def _ece(bin_n: np.ndarray, mean_pred: np.ndarray, frac_pos: np.ndarray):
    n_tot = int(bin_n.sum())
    if n_tot == 0:
        return np.nan
    ece = 0.0
    for nb, mp, fp in zip(bin_n, mean_pred, frac_pos):
        if nb <= 0:
            continue
        ece += (nb / n_tot) * abs(fp - mp)
    return float(ece)


def _pm(x: np.ndarray):
    x = x[~np.isnan(x)]
    return float(np.mean(x)), float(np.std(x, ddof=0))


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
    print(f"Features: {len(feat_cols)}  Splits: {N_SPLITS}  Test size: {TEST_SIZE}  Bins: {N_BINS}")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
        ]
    )

    outer = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)

    metrics_rows = []
    bins_rows = []

    used = 0
    skipped = 0

    for split_id, (train_idx, test_idx) in enumerate(outer.split(df[feat_cols], y_all, groups)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        y_tr = _y(train_df)
        y_te = _y(test_df)

        # Need both classes in both sets
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            skipped += 1
            continue

        X_tr = train_df[feat_cols]
        X_te = test_df[feat_cols]

        model.fit(X_tr, y_tr)
        p_te = model.predict_proba(X_te)[:, 1]
        p_te = np.clip(p_te, 1e-6, 1 - 1e-6)

        auc = roc_auc_score(y_te, p_te)
        ll = log_loss(y_te, p_te, labels=[0, 1])
        brier = float(np.mean((p_te - y_te) ** 2))

        edges, bin_n, mean_pred, frac_pos = _bin_stats(y_te, p_te, N_BINS)
        ece = _ece(bin_n, mean_pred, frac_pos)

        metrics_rows.append(
            {
                "split_id": split_id,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "auc": float(auc),
                "log_loss": float(ll),
                "brier": float(brier),
                "ece": float(ece),
            }
        )

        for b in range(N_BINS):
            bins_rows.append(
                {
                    "split_id": split_id,
                    "bin": b,
                    "bin_left": float(edges[b]),
                    "bin_right": float(edges[b + 1]),
                    "n": int(bin_n[b]),
                    "mean_pred": float(mean_pred[b]) if not np.isnan(mean_pred[b]) else np.nan,
                    "frac_pos": float(frac_pos[b]) if not np.isnan(frac_pos[b]) else np.nan,
                }
            )

        used += 1

    mdf = pd.DataFrame(metrics_rows)
    bdf = pd.DataFrame(bins_rows)

    mdf.to_csv(OUT_METRICS, index=False)
    bdf.to_csv(OUT_BINS, index=False)

    # Aggregate bin stats across splits (mean ± std)
    agg = (
        bdf.groupby("bin", as_index=False)
        .agg(
            bin_left=("bin_left", "first"),
            bin_right=("bin_right", "first"),
            n_mean=("n", "mean"),
            mean_pred_mean=("mean_pred", "mean"),
            mean_pred_std=("mean_pred", "std"),
            frac_pos_mean=("frac_pos", "mean"),
            frac_pos_std=("frac_pos", "std"),
        )
    )

    # Plot reliability diagram using mean points + vertical error bars
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.errorbar(
        agg["mean_pred_mean"].values,
        agg["frac_pos_mean"].values,
        yerr=agg["frac_pos_std"].fillna(0.0).values,
        fmt="o-",
        capsize=3,
    )
    plt.xlabel("Predicted probability (mean per bin)")
    plt.ylabel("Observed fraction PD (mean per bin)")
    plt.title("Reliability diagram (baseline VA, repeated speaker-splits)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200)
    plt.close()

    # Print summary
    auc_m, auc_s = _pm(mdf["auc"].values)
    ll_m, ll_s = _pm(mdf["log_loss"].values)
    br_m, br_s = _pm(mdf["brier"].values)
    ece_m, ece_s = _pm(mdf["ece"].values)

    print("\n=== DONE ===")
    print(f"Splits used: {used}   skipped: {skipped}")
    print(f"Saved metrics: {OUT_METRICS}")
    print(f"Saved bins:    {OUT_BINS}")
    print(f"Saved figure:  {OUT_FIG}")
    print("\nSummary (mean ± std over splits):")
    print(f"AUC:      {auc_m:.3f} ± {auc_s:.3f}")
    print(f"LogLoss:  {ll_m:.3f} ± {ll_s:.3f}")
    print(f"Brier:    {br_m:.3f} ± {br_s:.3f}")
    print(f"ECE:      {ece_m:.3f} ± {ece_s:.3f}")


if __name__ == "__main__":
    main()
