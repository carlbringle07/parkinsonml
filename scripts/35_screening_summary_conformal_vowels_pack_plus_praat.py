# scripts/35_screening_summary_conformal_vowels_pack_plus_praat.py
"""
Pooled split-conformal screening (alpha=0.10) för vowels_pack + Praat.

Input:
- data_index/features_vowels_pack_person_plus_praat.csv

Output:
- data_index/screening_summary_conformal_vowels_pack_plus_praat_alpha0.10.csv

Metrics (mean ± std över splits):
- coverage
- decided_rate / unsure_rate
- acc_decided
- auc
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = df["label"].astype(str).str.strip()
    df["speaker"] = df["speaker"].astype(str)
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"speaker", "label", "group", "tasks_present", "n_tasks_present", "n_tasks_missing"}
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("Hittade inga numeriska features.")
    return cols


def y_from_label(lbl: pd.Series) -> np.ndarray:
    mapping = {"pd": 1, "control_elderly": 0}
    y = lbl.map(mapping)
    if y.isna().any():
        bad = sorted(lbl[y.isna()].unique().tolist())
        raise ValueError(f"Okända label-värden: {bad}")
    return y.to_numpy(dtype=int)


def make_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])


def pooled_q(cal_probs: np.ndarray, cal_y: np.ndarray, alpha: float) -> float:
    p_true = np.where(cal_y == 1, cal_probs, 1.0 - cal_probs)
    scores = 1.0 - p_true
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def pred_set(prob_pd: float, q: float) -> set[int]:
    thr = 1.0 - q
    s = set()
    if prob_pd >= thr:
        s.add(1)
    if (1.0 - prob_pd) >= thr:
        s.add(0)
    return s


def run(alpha: float = 0.10, n_splits: int = 50, test_size: float = 0.25, cal_size: float = 0.25, seed: int = 42) -> pd.DataFrame:
    root = project_root()
    data_dir = root / "data_index"
    path = data_dir / "features_vowels_pack_person_plus_praat.csv"
    if not path.exists():
        raise FileNotFoundError(f"Hittar inte {path}")

    df = load_df(path)

    cols = feature_cols(df)
    X = df[cols].to_numpy(float)
    y = y_from_label(df["label"])
    groups = df["speaker"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)

    rows = []
    for split_id, (traincal_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        X_tc, y_tc, g_tc = X[traincal_idx], y[traincal_idx], groups[traincal_idx]

        cal_splitter = GroupShuffleSplit(n_splits=1, test_size=cal_size, random_state=seed + split_id)
        tr_rel, cal_rel = next(cal_splitter.split(X_tc, y_tc, groups=g_tc))

        tr_idx = traincal_idx[tr_rel]
        cal_idx = traincal_idx[cal_rel]

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        model = make_model()
        model.fit(X_tr, y_tr)

        p_cal = model.predict_proba(X_cal)[:, 1]
        q = pooled_q(p_cal, y_cal, alpha)

        p_te = model.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, p_te))

        sets = [pred_set(float(p), q) for p in p_te]
        covered = [int(yt in s) for yt, s in zip(y_te, sets)]
        coverage = float(np.mean(covered))

        singleton = [int(len(s) == 1) for s in sets]
        decided_rate = float(np.mean(singleton))
        unsure_rate = 1.0 - decided_rate

        decided_idx = [i for i, s in enumerate(sets) if len(s) == 1]
        if decided_idx:
            y_dec = y_te[decided_idx]
            y_hat = np.array([next(iter(sets[i])) for i in decided_idx], dtype=int)
            acc_decided = float(accuracy_score(y_dec, y_hat))
        else:
            acc_decided = float("nan")

        rows.append({
            "split": split_id,
            "auc": auc,
            "coverage": coverage,
            "decided_rate": decided_rate,
            "unsure_rate": unsure_rate,
            "acc_decided": acc_decided,
            "q": float(q),
            "n_train": int(len(tr_idx)),
            "n_cal": int(len(cal_idx)),
            "n_test": int(len(test_idx)),
        })

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    for c in ["auc", "coverage", "decided_rate", "unsure_rate", "acc_decided"]:
        out[f"{c}_mean"] = float(np.nanmean(df[c].to_numpy(float)))
        out[f"{c}_std"] = float(np.nanstd(df[c].to_numpy(float), ddof=1))
    return out


def main() -> None:
    alpha = 0.10
    res = run(alpha=alpha)

    summ = summarize(res)
    out = pd.DataFrame([{"variant": "vowels_pack_plus_praat", "alpha": alpha, **summ}])

    root = project_root()
    out_path = root / "data_index" / f"screening_summary_conformal_vowels_pack_plus_praat_alpha{alpha:.2f}.csv"
    out.to_csv(out_path, index=False)

    print("=== Conformal screening summary: vowels_pack + Praat ===")
    print(f"Saved: {out_path.name}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
