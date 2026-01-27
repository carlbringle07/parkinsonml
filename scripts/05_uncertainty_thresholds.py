from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

FEAT_PATH = Path("data_index/features_va_person.csv")

def metrics_for_thresholds(y: np.ndarray, p: np.ndarray, t_low: float, t_high: float):
    decision = np.full_like(p, "uncertain", dtype=object)
    decision[p <= t_low] = "control"
    decision[p >= t_high] = "pd"

    decided = decision != "uncertain"
    coverage = decided.mean()

    # precision på respektive säkert beslut
    pd_mask = decision == "pd"
    ctrl_mask = decision == "control"
    pd_prec = (y[pd_mask] == 1).mean() if pd_mask.any() else np.nan
    ctrl_prec = (y[ctrl_mask] == 0).mean() if ctrl_mask.any() else np.nan

    # accuracy på de som är beslutade
    if decided.any():
        pred = (decision[decided] == "pd").astype(int)
        acc = (pred == y[decided]).mean()
    else:
        acc = np.nan

    return coverage, pd_prec, ctrl_prec, acc, decision

def pick_thresholds(y: np.ndarray, p: np.ndarray, target_precision: float = 0.90):
    """
    Sök över ett rutnät och välj (t_low, t_high) som:
      1) t_low < t_high
      2) precision_pd >= target_precision och precision_control >= target_precision
      3) maximerar coverage
    """
    grid = np.linspace(0.0, 1.0, 501)  # 0.002 steg ungefär
    best = None  # (coverage, t_low, t_high, pd_prec, ctrl_prec)

    for t_low in grid:
        for t_high in grid:
            if not (t_low < t_high):
                continue
            coverage, pd_prec, ctrl_prec, acc, _ = metrics_for_thresholds(y, p, t_low, t_high)

            if np.isnan(pd_prec) or np.isnan(ctrl_prec):
                continue
            if pd_prec < target_precision or ctrl_prec < target_precision:
                continue

            cand = (coverage, t_low, t_high, pd_prec, ctrl_prec, acc)
            if best is None or cand[0] > best[0]:
                best = cand

    return best  # kan bli None om kravet är för hårt

def main() -> None:
    df = pd.read_csv(FEAT_PATH)
    df = df[df["label"].isin(["pd", "control_elderly"])].copy()
    df["y"] = (df["label"] == "pd").astype(int)

    # (Valfritt men rekommenderat) ta bort duration för extra trovärdighet
    feature_cols = [c for c in df.columns if c not in {"speaker", "label", "y", "duration_s"}]

    X = df[feature_cols]
    y = df["y"].to_numpy()
    groups = df["speaker"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=0)),
    ])

    model.fit(X_train, y_train)
    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    target = 0.90
    best = pick_thresholds(y_train, p_train, target_precision=target)

    # Om 0.90 är för hårt, backa automatiskt lite
    if best is None:
        target = 0.85
        best = pick_thresholds(y_train, p_train, target_precision=target)

    if best is None:
        raise RuntimeError("Hittade inga trösklar som ger meningsfull osäker-zon. Data kan vara för liten.")

    coverage_tr, t_low, t_high, pd_prec_tr, ctrl_prec_tr, acc_tr = best

    print("Valda trösklar (från TRAIN)")
    print("Mål-precision:", target)
    print("t_low :", round(float(t_low), 3), "(<= t_low => Kontroll)")
    print("t_high:", round(float(t_high), 3), "(>= t_high => PD)")
    print("TRAIN coverage:", round(float(coverage_tr), 3))
    print("TRAIN precision control:", round(float(ctrl_prec_tr), 3))
    print("TRAIN precision pd     :", round(float(pd_prec_tr), 3))

    decision, coverage_te, acc_te, = None, None, None
    coverage_te, pd_prec_te, ctrl_prec_te, acc_te, decision = metrics_for_thresholds(y_test, p_test, t_low, t_high)

    print("\nPå TEST")
    print("Coverage (andel som får säkert beslut):", round(float(coverage_te), 3))
    print("Accuracy på säkra beslut:", round(float(acc_te), 3))
    print("Precision control (på säkra):", round(float(ctrl_prec_te), 3) if not np.isnan(ctrl_prec_te) else "NA")
    print("Precision pd      (på säkra):", round(float(pd_prec_te), 3) if not np.isnan(pd_prec_te) else "NA")

    uniq, counts = np.unique(decision, return_counts=True)
    print("\nFördelning beslut på TEST:")
    for u, c in zip(uniq, counts):
        print(u, ":", int(c))

    decided = decision != "uncertain"
    if decided.any():
        pred = (decision[decided] == "pd").astype(int)
        cm = confusion_matrix(y_test[decided], pred)
        print("\nConfusion matrix (bara säkra beslut):")
        print(cm)

if __name__ == "__main__":
    main()
