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

def eval_one_split(p: np.ndarray, y: np.ndarray, reject_rate: float):
    n = len(p)
    n_reject = int(round(reject_rate * n))

    # osäkerhet = |p - 0.5|
    uncertainty = np.abs(p - 0.5)
    reject_idx = np.argsort(uncertainty)[:n_reject]

    decided = np.ones(n, dtype=bool)
    decided[reject_idx] = False

    coverage = decided.mean()

    pred = (p >= 0.5).astype(int)

    if decided.any():
        acc = (pred[decided] == y[decided]).mean()

        pred_pd = decided & (pred == 1)
        pred_ctrl = decided & (pred == 0)

        prec_pd = (y[pred_pd] == 1).mean() if pred_pd.any() else np.nan
        prec_ctrl = (y[pred_ctrl] == 0).mean() if pred_ctrl.any() else np.nan
    else:
        acc = np.nan
        prec_pd = np.nan
        prec_ctrl = np.nan

    return coverage, acc, prec_pd, prec_ctrl

def summarize(arr: np.ndarray):
    return float(np.nanmean(arr)), float(np.nanstd(arr)), float(np.nanmin(arr)), float(np.nanmax(arr))

def main() -> None:
    df = pd.read_csv(FEAT_PATH)
    df = df[df["label"].isin(["pd", "control_elderly"])].copy()
    df["y"] = (df["label"] == "pd").astype(int)

    # Ta bort duration för trovärdighet
    feature_cols = [c for c in df.columns if c not in {"speaker", "label", "y", "duration_s"}]

    X = df[feature_cols]
    y = df["y"].to_numpy()
    groups = df["speaker"].to_numpy()

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=0)),
    ])

    reject_rates = [0.0, 0.1, 0.2, 0.3]
    splitter = GroupShuffleSplit(n_splits=50, test_size=0.25, random_state=42)

    results = {rr: {"coverage": [], "acc": [], "prec_pd": [], "prec_ctrl": []} for rr in reject_rates}

    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        p_test = model.predict_proba(X_test)[:, 1]

        for rr in reject_rates:
            cov, acc, prec_pd, prec_ctrl = eval_one_split(p_test, y_test, rr)
            results[rr]["coverage"].append(cov)
            results[rr]["acc"].append(acc)
            results[rr]["prec_pd"].append(prec_pd)
            results[rr]["prec_ctrl"].append(prec_ctrl)

    print("Repeated reject-option evaluation (50 speaker splits)")
    for rr in reject_rates:
        cov = np.array(results[rr]["coverage"])
        acc = np.array(results[rr]["acc"])
        ppd = np.array(results[rr]["prec_pd"])
        pct = np.array(results[rr]["prec_ctrl"])

        cov_m, cov_s, cov_min, cov_max = summarize(cov)
        acc_m, acc_s, acc_min, acc_max = summarize(acc)
        ppd_m, ppd_s, ppd_min, ppd_max = summarize(ppd)
        pct_m, pct_s, pct_min, pct_max = summarize(pct)

        print("\n" + "="*50)
        print(f"Reject rate = {rr}")
        print(f"Coverage: mean {cov_m:.3f} ± {cov_s:.3f} (min {cov_min:.3f}, max {cov_max:.3f})")
        print(f"Accuracy (decided): mean {acc_m:.3f} ± {acc_s:.3f} (min {acc_min:.3f}, max {acc_max:.3f})")
        print(f"Precision PD: mean {ppd_m:.3f} ± {ppd_s:.3f} (min {ppd_min:.3f}, max {ppd_max:.3f})")
        print(f"Precision Control: mean {pct_m:.3f} ± {pct_s:.3f} (min {pct_min:.3f}, max {pct_max:.3f})")

if __name__ == "__main__":
    main()
