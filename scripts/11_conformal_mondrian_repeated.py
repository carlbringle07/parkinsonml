from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# =========================
# CONFIG
# =========================
FEATURES_CSV = Path("data_index/features_va_person.csv")

ID_COL_CANDIDATES = ["person_id", "speaker", "subject", "person", "id"]
LABEL_COL_CANDIDATES = ["label", "y", "target", "class"]

POS_LABEL = "pd"
NEG_LABEL = "control_elderly"
ALLOWED_LABELS = {POS_LABEL, NEG_LABEL}

N_SPLITS = 50
SEED = 42

TEST_FRAC_PER_CLASS = 0.20
CAL_FRAC_WITHIN_TRAIN_PER_CLASS = 0.50  # av resterande

ALPHA = 0.10

MODEL = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=SEED
    ))
])


def _pick_col(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Kunde inte hitta {kind}-kolumn. Testade: {candidates}")


def _maybe_aggregate_duplicates(df: pd.DataFrame, id_col: str, label_col: str) -> pd.DataFrame:
    
    if df[id_col].duplicated().any():
        feat_cols = [c for c in df.columns if c not in (id_col, label_col)]
        check = df.groupby(id_col)[label_col].nunique()
        if (check > 1).any():
            bad = check[check > 1].index.tolist()[:5]
            raise ValueError(f"Olika labels inom samma person (ex {bad}).")
        df = (
            df.groupby(id_col, as_index=False)
              .agg({**{label_col: "first"}, **{c: "mean" for c in feat_cols}})
        )
    return df


def fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    m = MODEL
    m.fit(X_train, y_train)
    return m


def split_by_class(y: np.ndarray, rng: np.random.Generator):
    """Returnerar proper_idx, cal_idx, test_idx med samma fraktion per klass."""
    idx_all = np.arange(len(y))
    idx_pd = idx_all[y == 1]
    idx_ctrl = idx_all[y == 0]

    rng.shuffle(idx_pd)
    rng.shuffle(idx_ctrl)

    n_test_pd = max(1, int(round(len(idx_pd) * TEST_FRAC_PER_CLASS)))
    n_test_ctrl = max(1, int(round(len(idx_ctrl) * TEST_FRAC_PER_CLASS)))

    test_pd = idx_pd[:n_test_pd]
    test_ctrl = idx_ctrl[:n_test_ctrl]

    remain_pd = idx_pd[n_test_pd:]
    remain_ctrl = idx_ctrl[n_test_ctrl:]

    n_cal_pd = max(1, int(round(len(remain_pd) * CAL_FRAC_WITHIN_TRAIN_PER_CLASS)))
    n_cal_ctrl = max(1, int(round(len(remain_ctrl) * CAL_FRAC_WITHIN_TRAIN_PER_CLASS)))

    cal_pd = remain_pd[:n_cal_pd]
    cal_ctrl = remain_ctrl[:n_cal_ctrl]

    proper_pd = remain_pd[n_cal_pd:]
    proper_ctrl = remain_ctrl[n_cal_ctrl:]

    test_idx = np.concatenate([test_pd, test_ctrl])
    cal_idx = np.concatenate([cal_pd, cal_ctrl])
    proper_idx = np.concatenate([proper_pd, proper_ctrl])

    return proper_idx, cal_idx, test_idx


def conformal_sets_pooled(proba_cal: np.ndarray, y_cal: np.ndarray, proba_test: np.ndarray, alpha: float):
    """
    Pooled split-conformal p-values för klassificering:
    score(x,y)=1-P(y|x)
    pval_y(x) = ( #{i: score_i >= score(x,y)} + 1 ) / (n_cal + 1)
    inkludera y om pval_y(x) > alpha
    """
    scores_true = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]
    sets = []
    for i in range(proba_test.shape[0]):
        S = set()
        for y in (0, 1):
            score_xy = 1.0 - proba_test[i, y]
            pval = (np.sum(scores_true >= score_xy) + 1.0) / (len(scores_true) + 1.0)
            if pval > alpha:
                S.add(y)
        sets.append(S)
    return sets


def main():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Hittar inte: {FEATURES_CSV.resolve()}")

    df = pd.read_csv(FEATURES_CSV)
    id_col = _pick_col(df, ID_COL_CANDIDATES, "ID/person")
    label_col = _pick_col(df, LABEL_COL_CANDIDATES, "label")

    df = df[df[label_col].isin(ALLOWED_LABELS)].copy()
    df = _maybe_aggregate_duplicates(df, id_col, label_col)

    y = (df[label_col].values == POS_LABEL).astype(int)
    feat_cols = [c for c in df.columns if c not in (id_col, label_col)]
    X = df[feat_cols].values.astype(float)

    rows = []

    for s in range(1, N_SPLITS + 1):
        
        ok = False
        for attempt in range(50):
            rng = np.random.default_rng(SEED + s * 1000 + attempt)
            proper_idx, cal_idx, test_idx = split_by_class(y, rng)

            if len(np.unique(y[proper_idx])) < 2:
                continue
            if len(np.unique(y[test_idx])) < 2:
                continue
            ok = True
            break

        if not ok:
            continue

        X_proper, y_proper = X[proper_idx], y[proper_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = fit_model(X_proper, y_proper)
        proba_cal = model.predict_proba(X_cal)
        proba_test = model.predict_proba(X_test)

        sets = conformal_sets_pooled(proba_cal, y_cal, proba_test, ALPHA)

        set_sizes = np.array([len(ss) for ss in sets], dtype=int)
        covered = np.array([y_test[i] in sets[i] for i in range(len(y_test))], dtype=bool)

        coverage = float(covered.mean())
        cov_pd = float(covered[y_test == 1].mean()) if np.any(y_test == 1) else np.nan
        cov_ctrl = float(covered[y_test == 0].mean()) if np.any(y_test == 0) else np.nan

        singleton_mask = (set_sizes == 1)
        singleton_rate = float(singleton_mask.mean())
        if singleton_mask.any():
            pred_single = np.array([list(sets[i])[0] for i in np.where(singleton_mask)[0]], dtype=int)
            singleton_acc = float((pred_single == y_test[singleton_mask]).mean())
        else:
            singleton_acc = np.nan

        avg_set_size = float(set_sizes.mean())
        auc = float(roc_auc_score(y_test, proba_test[:, 1]))

        rows.append({
            "split": s,
            "n_test": len(y_test),
            "n_cal": len(y_cal),
            "coverage": coverage,
            "coverage_pd": cov_pd,
            "coverage_control": cov_ctrl,
            "singleton_rate": singleton_rate,
            "singleton_acc": singleton_acc,
            "avg_set_size": avg_set_size,
            "auc": auc,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError("Inga splits kördes.")

    out_path = Path("data_index") / f"conformal_pooled_stratified_alpha{ALPHA:.2f}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)

    def ms(x: pd.Series) -> str:
        x = x.dropna()
        return f"{x.mean():.3f} ± {x.std(ddof=1):.3f} (min {x.min():.3f}, max {x.max():.3f})"

    print("\n=== Conformal (POOLED + stratified splits) ===")
    print(f"alpha = {ALPHA}")
    print(f"test_frac_per_class = {TEST_FRAC_PER_CLASS} | cal_frac_within_train_per_class = {CAL_FRAC_WITHIN_TRAIN_PER_CLASS}")
    print(f"splits used: {len(res)}")

    print("\nMetrics:")
    print(f"coverage:         {ms(res['coverage'])}")
    print(f"coverage_pd:      {ms(res['coverage_pd'])}")
    print(f"coverage_control: {ms(res['coverage_control'])}")
    print(f"singleton_rate:   {ms(res['singleton_rate'])}")
    print(f"singleton_acc:    {ms(res['singleton_acc'])}")
    print(f"avg_set_size:     {ms(res['avg_set_size'])}")
    print(f"auc:              {ms(res['auc'])}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
