# scripts/32_compare_conformal_screening_vowels_pack_baseline_vs_hierarchical.py
"""
Jämför pooled split-conformal screening (alpha=0.10) för vowels_pack:
- baseline person features
- hierarchical person features

Mäter över repeated speaker-splits:
- coverage
- decided_rate (singleton_rate)
- unsure_rate
- acc_decided
- auc

Output:
- data_index/compare_conformal_screening_vowels_pack_baseline_vs_hierarchical_alpha0.10.csv
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


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"speaker", "label", "group", "tasks_present", "n_tasks_present", "n_tasks_missing"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
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
        )),
    ])


def pooled_conformal_threshold(cal_probs: np.ndarray, cal_y: np.ndarray, alpha: float) -> float:
    """
    Pooled conformal för binär klassificering via sannolikheter:
    Vi använder nonconformity score = 1 - p_true.
    Tröskel q = quantile_{ceil((n+1)*(1-alpha))/n} av scores.
    """
    p_true = np.where(cal_y == 1, cal_probs, 1.0 - cal_probs)
    scores = 1.0 - p_true  
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    q = np.sort(scores)[k - 1]
    return float(q)


def predict_set(prob_pd: float, q: float) -> set[int]:
    """
    Returnerar prediktionsmängd {0}, {1}, {0,1} beroende på conformal-villkor.
    Inkludera klass y om 1 - p_y <= q  <=> p_y >= 1 - q
    """
    threshold = 1.0 - q
    pred = set()
    if prob_pd >= threshold:
        pred.add(1)
    if (1.0 - prob_pd) >= threshold:
        pred.add(0)
    return pred


def eval_variant(df: pd.DataFrame, alpha: float = 0.10, n_splits: int = 50, test_size: float = 0.25, cal_size: float = 0.25, seed: int = 42) -> pd.DataFrame:
    """
    Per split:
      Train speakers -> delas i train+cal (på speaker-nivå)
      Fit model på train
      Calibrera q på cal
      Testa på test
    """
    feat_cols = get_feature_cols(df)
    X = df[feat_cols].to_numpy(float)
    y = y_from_label(df["label"])
    groups = df["speaker"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)

    rows = []
    for split_id, (traincal_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        # split traincal speakers vidare till train/cal
        X_traincal = X[traincal_idx]
        y_traincal = y[traincal_idx]
        g_traincal = groups[traincal_idx]

        # cal split på speaker-nivå
        cal_splitter = GroupShuffleSplit(n_splits=1, test_size=cal_size, random_state=seed + split_id)
        train_idx_rel, cal_idx_rel = next(cal_splitter.split(X_traincal, y_traincal, groups=g_traincal))

        train_idx = traincal_idx[train_idx_rel]
        cal_idx = traincal_idx[cal_idx_rel]

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        model = make_model()
        model.fit(X_tr, y_tr)

        p_cal = model.predict_proba(X_cal)[:, 1]
        q = pooled_conformal_threshold(p_cal, y_cal, alpha=alpha)

        p_te = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, p_te)

        # prediktionsmängder och metrics
        sets = [predict_set(float(p), q) for p in p_te]

        covered = [int(yt in s) for yt, s in zip(y_te, sets)]
        coverage = float(np.mean(covered))

        singleton = [int(len(s) == 1) for s in sets]
        decided_rate = float(np.mean(singleton))
        unsure_rate = 1.0 - decided_rate

        # accuracy på de decided fallen
        decided_idx = [i for i, s in enumerate(sets) if len(s) == 1]
        if len(decided_idx) > 0:
            y_dec = y_te[decided_idx]
            y_hat = np.array([next(iter(sets[i])) for i in decided_idx], dtype=int)
            acc_decided = float(accuracy_score(y_dec, y_hat))
        else:
            acc_decided = float("nan")

        rows.append({
            "split": split_id,
            "auc": float(auc),
            "coverage": coverage,
            "decided_rate": decided_rate,
            "unsure_rate": unsure_rate,
            "acc_decided": acc_decided,
            "q": float(q),
            "n_test": int(len(test_idx)),
            "n_cal": int(len(cal_idx)),
            "n_train": int(len(train_idx)),
        })

    return pd.DataFrame(rows)


def summarize(df_res: pd.DataFrame) -> dict:
    def mean_std(x):
        return float(np.nanmean(x)), float(np.nanstd(x, ddof=1))

    out = {}
    for col in ["auc", "coverage", "decided_rate", "unsure_rate", "acc_decided"]:
        m, s = mean_std(df_res[col].to_numpy(float))
        out[f"{col}_mean"] = m
        out[f"{col}_std"] = s
    return out


def main() -> None:
    root = project_root()
    data_dir = root / "data_index"

    base_path = data_dir / "features_vowels_pack_person.csv"
    hier_path = data_dir / "features_vowels_pack_person_hierarchical.csv"

    if not base_path.exists():
        raise FileNotFoundError(f"Hittar inte {base_path}")
    if not hier_path.exists():
        raise FileNotFoundError(f"Hittar inte {hier_path}")

    base = load_df(base_path)
    hier = load_df(hier_path)

    alpha = 0.10

    res_base = eval_variant(base, alpha=alpha, n_splits=50, test_size=0.25, cal_size=0.25, seed=42)
    res_hier = eval_variant(hier, alpha=alpha, n_splits=50, test_size=0.25, cal_size=0.25, seed=42)

    sum_base = summarize(res_base)
    sum_hier = summarize(res_hier)

    out = pd.DataFrame([
        {"variant": "baseline_person", "alpha": alpha, **sum_base},
        {"variant": "hierarchical_person", "alpha": alpha, **sum_hier},
    ])

    out_path = data_dir / f"compare_conformal_screening_vowels_pack_baseline_vs_hierarchical_alpha{alpha:.2f}.csv"
    out.to_csv(out_path, index=False)

    print("=== Conformal screening comparison (vowels_pack) ===")
    print(f"Saved: {out_path.name}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
