# scripts/39_conformal_class_coverage_vowels_pack_plus_praat.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def quantile_high(scores: np.ndarray, q: float) -> float:
    """
    Returnerar 'higher'-kvantil så att vi får en konservativ tröskel.
    q = 1 - alpha (t.ex. 0.9)
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    if scores.size == 0:
        return float("inf")
    q = float(np.clip(q, 0.0, 1.0))
    try:
        return float(np.quantile(scores, q, method="higher"))
    except TypeError:
        # numpy < 1.22
        return float(np.quantile(scores, q, interpolation="higher"))


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"speaker", "label", "group", "y", "tasks_present", "n_tasks_present", "n_tasks_missing"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def main() -> None:
    root = project_root()
    data_dir = root / "data_index"

    # Input: person-tabellen med baseline+praat
    in_path = data_dir / "features_vowels_pack_person_plus_praat.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Hittar inte: {in_path}")

    df = pd.read_csv(in_path)

    # label -> y
    if "label" not in df.columns or "speaker" not in df.columns:
        raise ValueError("CSV måste innehålla kolumnerna 'label' och 'speaker'.")

    y = df["label"].map({"pd": 1, "control_elderly": 0})
    if y.isna().any():
        bad = df.loc[y.isna(), "label"].unique().tolist()
        raise ValueError(f"Oväntade label-värden: {bad}")
    df = df.copy()
    df["y"] = y.astype(int)

    feat_cols = get_feature_cols(df)
    X = df[feat_cols].to_numpy(float)
    y = df["y"].to_numpy(int)
    groups = df["speaker"].astype(str).to_numpy()

    # Inställningar
    alpha = 0.10
    n_splits = 50
    test_size = 0.30          # matchar ditt script 28-output (test size 0.3)
    calib_size_within_trainval = 0.25  # av återstående (trainval) går 25% till calibration

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear")),
    ])

    rows = []
    skipped = 0

    for seed in range(n_splits):
        # Outer split: trainval vs test (speaker-split)
        gss_outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(gss_outer.split(X, y, groups=groups))

        X_trainval, y_trainval, g_trainval = X[trainval_idx], y[trainval_idx], groups[trainval_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Inner split: train vs calib (speaker-split inom trainval)
        gss_inner = GroupShuffleSplit(n_splits=1, test_size=calib_size_within_trainval, random_state=seed + 10_000)
        train_idx_rel, calib_idx_rel = next(gss_inner.split(X_trainval, y_trainval, groups=g_trainval))

        X_train, y_train = X_trainval[train_idx_rel], y_trainval[train_idx_rel]
        X_cal, y_cal = X_trainval[calib_idx_rel], y_trainval[calib_idx_rel]

        # Säkerhetscheck: behöver minst lite av varje klass i calibration
        if len(np.unique(y_cal)) < 2:
            skipped += 1
            continue

        # Fit på train
        model.fit(X_train, y_train)

        # Predict probs
        p_cal = model.predict_proba(X_cal)[:, 1]      # P(PD)
        p_test = model.predict_proba(X_test)[:, 1]

        # Pooled conformal: nonconformity score s = 1 - p(true_label)
        p_true_cal = np.where(y_cal == 1, p_cal, 1.0 - p_cal)
        scores_cal = 1.0 - p_true_cal

        q = quantile_high(scores_cal, 1.0 - alpha)

        # För test: inkludera label i pred_set om score(label) <= q
        # score(PD)=1-p_pd ; score(CTRL)=1-(1-p_pd)=p_pd
        score_pd = 1.0 - p_test
        score_ctrl = p_test

        in_pd = score_pd <= q
        in_ctrl = score_ctrl <= q

        # coverage overall
        covered = np.where(y_test == 1, in_pd, in_ctrl)

        # classwise coverage
        mask_pd = (y_test == 1)
        mask_ctrl = (y_test == 0)

        cov_all = float(np.mean(covered)) if covered.size else np.nan
        cov_pd = float(np.mean(covered[mask_pd])) if np.any(mask_pd) else np.nan
        cov_ctrl = float(np.mean(covered[mask_ctrl])) if np.any(mask_ctrl) else np.nan

        rows.append({
            "split": seed,
            "alpha": alpha,
            "n_test": int(len(y_test)),
            "n_test_pd": int(np.sum(mask_pd)),
            "n_test_ctrl": int(np.sum(mask_ctrl)),
            "q": float(q),
            "cov_all": cov_all,
            "cov_pd": cov_pd,
            "cov_ctrl": cov_ctrl,
        })

    out_df = pd.DataFrame(rows)
    out_path = data_dir / f"conformal_class_coverage_vowels_pack_plus_praat_alpha{alpha:.2f}.csv"
    out_df.to_csv(out_path, index=False)

    # Summary
    def ms(col: str) -> tuple[float, float]:
        v = pd.to_numeric(out_df[col], errors="coerce").to_numpy(float)
        return float(np.nanmean(v)), float(np.nanstd(v, ddof=1))

    cov_all_m, cov_all_s = ms("cov_all")
    cov_pd_m, cov_pd_s = ms("cov_pd")
    cov_ctrl_m, cov_ctrl_s = ms("cov_ctrl")

    print("=== Conformal class coverage: vowels_pack + Praat (pooled) ===")
    print(f"Input: {in_path.name}")
    print(f"Features: {X.shape[1]}  Speakers: {len(np.unique(groups))}")
    print(f"alpha: {alpha}  splits: {n_splits}  test_size: {test_size}  inner_calib_frac: {calib_size_within_trainval}")
    print(f"Saved: {out_path.name}")
    print(f"Splits used: {len(out_df)}  skipped: {skipped}\n")
    print("Summary (mean ± std over splits):")
    print(f"cov_all : {cov_all_m:.3f} ± {cov_all_s:.3f}")
    print(f"cov_pd  : {cov_pd_m:.3f} ± {cov_pd_s:.3f}")
    print(f"cov_ctrl: {cov_ctrl_m:.3f} ± {cov_ctrl_s:.3f}")


if __name__ == "__main__":
    main()
