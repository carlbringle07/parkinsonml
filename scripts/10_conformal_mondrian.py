from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

FEAT_PATH = Path("data_index/features_va_person.csv")

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    # finite-sample quantile (garanti-variant)
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = np.clip(k, 0, n - 1)
    return float(np.sort(scores)[k])

def main(alpha: float = 0.10) -> None:
    df = pd.read_csv(FEAT_PATH)
    df = df[df["label"].isin(["pd", "control_elderly"])].copy()
    df["y"] = (df["label"] == "pd").astype(int)

    # Ta bort duration (extra trovärdighet)
    feature_cols = [c for c in df.columns if c not in {"speaker", "label", "y", "duration_s"}]

    X = df[feature_cols]
    y = df["y"].to_numpy()
    groups = df["speaker"].to_numpy()

    # 70% train, 30% rest
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, rest_idx = next(splitter1.split(X, y, groups=groups))

    X_train, X_rest = X.iloc[train_idx], X.iloc[rest_idx]
    y_train, y_rest = y[train_idx], y[rest_idx]
    g_rest = groups[rest_idx]

    # Rest -> 50% cal, 50% test
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=123)
    cal_rel, test_rel = next(splitter2.split(X_rest, y_rest, groups=g_rest))

    X_cal, X_test = X_rest.iloc[cal_rel], X_rest.iloc[test_rel]
    y_cal, y_test = y_rest[cal_rel], y_rest[test_rel]

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=0)),
    ])
    model.fit(X_train, y_train)

    p_cal = model.predict_proba(X_cal)[:, 1]   # p = P(PD)
    p_test = model.predict_proba(X_test)[:, 1]

    # Klass-villkorade scores
    scores_pd = 1 - p_cal[y_cal == 1]   # PD: vill att p är hög
    scores_ctrl = p_cal[y_cal == 0]     # Control: vill att p är låg

    q_pd = conformal_quantile(scores_pd, alpha=alpha)
    q_ctrl = conformal_quantile(scores_ctrl, alpha=alpha)

    high = 1 - q_pd      # PD om p >= high
    low = q_ctrl         # Control om p <= low

    # Beslut: singel om tydligt, annars osäker
    decisions = np.full_like(p_test, "uncertain", dtype=object)
    decisions[p_test <= low] = "control"
    decisions[p_test >= high] = "pd"

    # Coverage: "uncertain" räknas som att båda labels ingår -> alltid korrekt täckt
    covered = (decisions == "uncertain") | ((decisions == "pd") & (y_test == 1)) | ((decisions == "control") & (y_test == 0))
    coverage = covered.mean()

    decided = decisions != "uncertain"
    decided_rate = decided.mean()
    if decided.any():
        pred = (decisions[decided] == "pd").astype(int)
        acc_decided = (pred == y_test[decided]).mean()
    else:
        acc_decided = float("nan")

    print(f"Mondrian split-conformal (alpha={alpha})")
    print("low (control if p<=low):", round(float(low), 3))
    print("high (pd if p>=high):  ", round(float(high), 3))
    print("Coverage (mål ~", 1-alpha, "):", round(float(coverage), 3))
    print("Andel singel-beslut:", round(float(decided_rate), 3))
    print("Accuracy på singel-beslut:", round(float(acc_decided), 3))

    uniq, counts = np.unique(decisions, return_counts=True)
    print("\nFördelning beslut på TEST:")
    for u, c in zip(uniq, counts):
        print(u, ":", int(c))

if __name__ == "__main__":
    main(alpha=0.10)
