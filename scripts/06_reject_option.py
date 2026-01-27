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

def run_once(reject_rate: float = 0.2) -> None:
    df = pd.read_csv(FEAT_PATH)
    df = df[df["label"].isin(["pd", "control_elderly"])].copy()
    df["y"] = (df["label"] == "pd").astype(int)

    # Ta bort duration för trovärdighet
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

    p = model.predict_proba(X_test)[:, 1]

    # "Osäkerhet" = hur nära 0.5 man ligger
    uncertainty = np.abs(p - 0.5)

    n_test = len(p)
    n_reject = int(round(reject_rate * n_test))

    # Index på de mest osäkra (minst uncertainty)
    reject_idx = np.argsort(uncertainty)[:n_reject]

    decision = np.array(["decide"] * n_test, dtype=object)
    decision[reject_idx] = "uncertain"

    decided_mask = decision != "uncertain"
    coverage = decided_mask.mean()

    # Prediktion på de som är beslutade (tröskel 0.5)
    pred = (p >= 0.5).astype(int)

    if decided_mask.any():
        acc = (pred[decided_mask] == y_test[decided_mask]).mean()
        cm = confusion_matrix(y_test[decided_mask], pred[decided_mask])
    else:
        acc = float("nan")
        cm = None

    # Precision för PD och kontroll på beslutade
    if decided_mask.any():
        # PD-precision: bland de vi predikterar som PD, hur många är verkligen PD?
        pred_pd = decided_mask & (pred == 1)
        pred_ctrl = decided_mask & (pred == 0)
        prec_pd = (y_test[pred_pd] == 1).mean() if pred_pd.any() else float("nan")
        prec_ctrl = (y_test[pred_ctrl] == 0).mean() if pred_ctrl.any() else float("nan")
    else:
        prec_pd = prec_ctrl = float("nan")

    print(f"Reject rate: {reject_rate}  (andel osäkra)")
    print("Coverage (andel säkra beslut):", round(float(coverage), 3))
    print("Accuracy på säkra beslut:", round(float(acc), 3))
    print("Precision control:", round(float(prec_ctrl), 3) if not np.isnan(prec_ctrl) else "NA")
    print("Precision pd     :", round(float(prec_pd), 3) if not np.isnan(prec_pd) else "NA")
    print("Osäkra antal:", int((~decided_mask).sum()), "av", n_test)

    if cm is not None:
        print("\nConfusion matrix (bara säkra beslut):")
        print(cm)

def main() -> None:
    for rr in [0.0, 0.1, 0.2, 0.3]:
        print("\n" + "="*40)
        run_once(reject_rate=rr)

if __name__ == "__main__":
    main()
