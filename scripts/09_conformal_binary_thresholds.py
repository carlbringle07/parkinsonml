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
    # finite-sample quantile
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = np.clip(k, 0, n - 1)
    return float(np.sort(scores)[k])

def main(alpha: float = 0.10) -> None:
    df = pd.read_csv(FEAT_PATH)
    df = df[df["label"].isin(["pd", "control_elderly"])].copy()
    df["y"] = (df["label"] == "pd").astype(int)

    # Ta bort duration
    feature_cols = [c for c in df.columns if c not in {"speaker", "label", "y", "duration_s"}]
    X = df[feature_cols]
    y = df["y"].to_numpy()
    groups = df["speaker"].to_numpy()

    # Split: 70% train, 30% (cal+test)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, rest_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_rest = X.iloc[train_idx], X.iloc[rest_idx]
    y_train, y_rest = y[train_idx], y[rest_idx]
    g_rest = groups[rest_idx]

    # Split rest into calibration and test (halva/halva)
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

    # p = P(PD)
    p_cal = model.predict_proba(X_cal)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    # Nonconformity scores:
    # - om y=1 (PD): score = 1 - p
    # - om y=0 (control): score = p
    scores = np.where(y_cal == 1, 1 - p_cal, p_cal)

    q = conformal_quantile(scores, alpha=alpha)

    # Beslutsregler:
    # PD om p >= 1 - q
    # Control om p <= q
    # annars osäker
    low = q
    high = 1 - q

    decisions = np.full_like(p_test, "uncertain", dtype=object)
    decisions[p_test <= low] = "control"
    decisions[p_test >= high] = "pd"

    # Coverage: sann label ska hamna inom setet
    # Setet är {control} om p<=low; {pd} om p>=high; annars {control,pd}
    in_set = np.ones_like(y_test, dtype=bool)  # osäker innehåller alltid båda
    in_set[(y_test == 1) & (p_test < high)] = True
    in_set[(y_test == 0) & (p_test > low)] = True
   

    coverage = in_set.mean()

    decided = decisions != "uncertain"
    decided_rate = decided.mean()
    if decided.any():
        pred = (decisions[decided] == "pd").astype(int)
        acc_decided = (pred == y_test[decided]).mean()
    else:
        acc_decided = float("nan")

    print("Binary split-conformal (alpha =", alpha, ")")
    print("low =", round(float(low), 3), "high =", round(float(high), 3))
    print("Coverage (teoretiskt ~", 1-alpha, "):", round(float(coverage), 3))
    print("Andel singel-beslut:", round(float(decided_rate), 3))
    print("Accuracy på singel-beslut:", round(float(acc_decided), 3))

    uniq, counts = np.unique(decisions, return_counts=True)
    print("\nFördelning beslut på TEST:")
    for u, c in zip(uniq, counts):
        print(u, ":", int(c))

if __name__ == "__main__":
    main(alpha=0.10)
