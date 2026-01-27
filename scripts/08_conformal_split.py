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

def quantile(scores: np.ndarray, alpha: float) -> float:
    # conformal-kvantilen (finite-sample)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = np.clip(k, 0, n - 1)
    return float(np.sort(scores)[k])

def main(alpha: float = 0.10) -> None:
    df = pd.read_csv(FEAT_PATH)
    df = df[df["label"].isin(["pd", "control_elderly"])].copy()
    df["y"] = (df["label"] == "pd").astype(int)

    # Ta bort duration för trovärdighet
    feature_cols = [c for c in df.columns if c not in {"speaker", "label", "y", "duration_s"}]

    X = df[feature_cols]
    y = df["y"].to_numpy()
    groups = df["speaker"].to_numpy()

    # 1) Train/Temp split
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(splitter1.split(X, y, groups=groups))

    X_train, X_temp = X.iloc[train_idx], X.iloc[temp_idx]
    y_train, y_temp = y[train_idx], y[temp_idx]
    g_temp = groups[temp_idx]

    # 2) Calibration/Test split (från temp)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=123)
    cal_rel, test_rel = next(splitter2.split(X_temp, y_temp, groups=g_temp))

    X_cal, X_test = X_temp.iloc[cal_rel], X_temp.iloc[test_rel]
    y_cal, y_test = y_temp[cal_rel], y_temp[test_rel]

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=0)),
    ])
    model.fit(X_train, y_train)

    # Sannolikhet för PD
    p_cal = model.predict_proba(X_cal)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    # Nonconformity scores:
    # score för sann klass = 1 - p(y_true)
    # där p(y_true) = p om y=1 annars (1-p)
    p_true_cal = np.where(y_cal == 1, p_cal, 1 - p_cal)
    scores = 1 - p_true_cal

    q = quantile(scores, alpha=alpha)

    # Prediktionsmängd för varje testpunkt:
    # inkludera label om (1 - p(label)) <= q  <=> p(label) >= 1 - q
    threshold = 1 - q
    # p(control) = 1 - p_pd
    set_is_pd = (p_test >= threshold)
    set_is_ctrl = ((1 - p_test) >= threshold)

    # Bygg beslut
    decisions = []
    for pd_ok, ctrl_ok in zip(set_is_pd, set_is_ctrl):
        if pd_ok and ctrl_ok:
            decisions.append("uncertain")
        elif pd_ok:
            decisions.append("pd")
        elif ctrl_ok:
            decisions.append("control")
        else:
            decisions.append("uncertain")  # borde vara sällsynt, men vi mappar till osäker

    decisions = np.array(decisions, dtype=object)

    # Coverage = hur ofta sann label finns i mängden
    true_in_set = np.where(y_test == 1, set_is_pd, set_is_ctrl)
    coverage = true_in_set.mean()

    # Hur ofta vi får ett singel-beslut (inte osäker)
    decided = decisions != "uncertain"
    decided_rate = decided.mean()

    # Accuracy på singel-beslut
    if decided.any():
        pred = (decisions[decided] == "pd").astype(int)
        acc_decided = (pred == y_test[decided]).mean()
    else:
        acc_decided = float("nan")

    print("Split-conformal (alpha =", alpha, ")")
    print("Threshold for including label:", round(float(threshold), 3))
    print("Coverage (bör vara ~", 1-alpha, "):", round(float(coverage), 3))
    print("Andel singel-beslut (ej osäker):", round(float(decided_rate), 3))
    print("Accuracy på singel-beslut:", round(float(acc_decided), 3))

    # Fördelning
    uniq, counts = np.unique(decisions, return_counts=True)
    print("\nFördelning beslut på TEST:")
    for u, c in zip(uniq, counts):
        print(u, ":", int(c))

if __name__ == "__main__":
    main(alpha=0.10)
