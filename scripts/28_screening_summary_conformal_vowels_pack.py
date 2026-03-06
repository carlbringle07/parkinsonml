# FILE: scripts/28_screening_summary_conformal_vowels_pack.py

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
OUT_CSV = DATA_INDEX / "screening_summary_conformal_vowels_pack_alpha0.10.csv"

N_SPLITS = 50
TEST_SIZE = 0.30
CAL_WITHIN_TRAIN = 0.50  # split train into (proper-train) and calibration
ALPHA = 0.10
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


def _split_conformal_pooled(model, X_train, y_train, X_test, y_test, alpha: float, rng: np.random.RandomState):
    """
    Pooled split-conformal for binary classification using nonconformity = 1 - p_true.
    Returns: coverage, decided_rate, unsure_rate, acc_decided, prec_pd_decided, prec_ctrl_decided, auc
    """
    # split X_train into proper-train and calibration
    n = len(X_train)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(np.floor((1.0 - CAL_WITHIN_TRAIN) * n))
    proper_idx = idx[:cut]
    cal_idx = idx[cut:]

    X_prop = X_train.iloc[proper_idx]
    y_prop = y_train[proper_idx]
    X_cal = X_train.iloc[cal_idx]
    y_cal = y_train[cal_idx]

    # need both classes in proper train and calibration to be meaningful
    if len(np.unique(y_prop)) < 2 or len(np.unique(y_cal)) < 2:
        return None

    model.fit(X_prop, y_prop)

    # calibration scores
    p_cal = model.predict_proba(X_cal)
    s_true = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]  # 1 - p_true

    n_cal = len(s_true)
    # conformal quantile level: ceil((n+1)*(1-alpha))/n  clipped to [0,1]
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = float(np.clip(q_level, 0.0, 1.0))
    q = float(np.quantile(s_true, q_level, method="higher"))

    # test prediction sets
    p_test = model.predict_proba(X_test)
    # include label k if 1 - p_k <= q  => p_k >= 1 - q
    thresh = 1.0 - q
    pred_set_pd = p_test[:, 1] >= thresh
    pred_set_ctrl = p_test[:, 0] >= thresh

    set_size = pred_set_pd.astype(int) + pred_set_ctrl.astype(int)

    # coverage: true label is in set
    y = y_test
    covered = np.where(y == 1, pred_set_pd, pred_set_ctrl)
    coverage = float(np.mean(covered))

    decided = set_size == 1
    decided_rate = float(np.mean(decided))
    unsure_rate = 1.0 - decided_rate

    # accuracy on decided: must map singleton set to label
    acc_decided = np.nan
    prec_pd = np.nan
    prec_ctrl = np.nan

    if np.any(decided):
        y_hat = np.where(pred_set_pd[decided], 1, 0)
        y_true_d = y[decided]
        acc_decided = float(np.mean(y_hat == y_true_d))

        # precision per predicted class (on decided)
        pred_pd = y_hat == 1
        pred_ctrl = y_hat == 0
        if np.any(pred_pd):
            prec_pd = float(np.mean(y_true_d[pred_pd] == 1))
        if np.any(pred_ctrl):
            prec_ctrl = float(np.mean(y_true_d[pred_ctrl] == 0))

    auc = float(roc_auc_score(y_test, p_test[:, 1]))

    return coverage, decided_rate, unsure_rate, acc_decided, prec_pd, prec_ctrl, auc


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
    print(f"Features: {len(feat_cols)}  Splits: {N_SPLITS}  Test size: {TEST_SIZE}  alpha: {ALPHA}")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
        ]
    )

    splitter = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)

    rows = []
    rng_master = np.random.RandomState(SEED)

    used = 0
    skipped = 0

    for split_id, (tr, te) in enumerate(splitter.split(df[feat_cols], y, groups)):
        y_tr = y[tr]
        y_te = y[te]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            skipped += 1
            continue

        # independent rng per split
        rng = np.random.RandomState(rng_master.randint(0, 10**9))

        res = _split_conformal_pooled(
            model=model,
            X_train=df.iloc[tr][feat_cols],
            y_train=y_tr,
            X_test=df.iloc[te][feat_cols],
            y_test=y_te,
            alpha=ALPHA,
            rng=rng,
        )
        if res is None:
            skipped += 1
            continue

        coverage, decided_rate, unsure_rate, acc_decided, prec_pd, prec_ctrl, auc = res
        rows.append(
            {
                "split_id": split_id,
                "coverage": coverage,
                "decided_rate": decided_rate,
                "unsure_rate": unsure_rate,
                "acc_decided": acc_decided,
                "prec_pd_decided": prec_pd,
                "prec_ctrl_decided": prec_ctrl,
                "auc": auc,
            }
        )
        used += 1

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    def ms(x):
        x = pd.to_numeric(x, errors="coerce")
        return float(np.nanmean(x)), float(np.nanstd(x))

    cov_m, cov_s = ms(out["coverage"])
    dec_m, dec_s = ms(out["decided_rate"])
    uns_m, uns_s = ms(out["unsure_rate"])
    acc_m, acc_s = ms(out["acc_decided"])
    pp_m, pp_s = ms(out["prec_pd_decided"])
    pc_m, pc_s = ms(out["prec_ctrl_decided"])
    auc_m, auc_s = ms(out["auc"])

    print("\n=== DONE ===")
    print("Saved:", OUT_CSV)
    print(f"Splits used: {used}  skipped: {skipped}")
    print("\nSummary (mean ± std over splits):")
    print(f"coverage:        {cov_m:.3f} ± {cov_s:.3f}")
    print(f"decided_rate:    {dec_m:.3f} ± {dec_s:.3f}")
    print(f"unsure_rate:     {uns_m:.3f} ± {uns_s:.3f}")
    print(f"acc_decided:     {acc_m:.3f} ± {acc_s:.3f}")
    print(f"prec_pd_decided: {pp_m:.3f} ± {pp_s:.3f}")
    print(f"prec_ctrl_decided:{pc_m:.3f} ± {pc_s:.3f}")
    print(f"auc:             {auc_m:.3f} ± {auc_s:.3f}")


if __name__ == "__main__":
    main()
