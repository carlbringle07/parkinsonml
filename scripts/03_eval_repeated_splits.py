from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEAT_PATH = Path("data_index/features_va_person.csv")

def main() -> None:
    df = pd.read_csv(FEAT_PATH)

    # Bara pd vs äldre kontroller
    df = df[df["label"].isin(["pd", "control_elderly"])].copy()
    df["y"] = (df["label"] == "pd").astype(int)

    feature_cols = [c for c in df.columns if c not in {"speaker", "label", "y"}]
    X = df[feature_cols]
    y = df["y"].to_numpy()
    groups = df["speaker"].to_numpy()

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=0)),
    ])

    aucs = []
    splitter = GroupShuffleSplit(n_splits=50, test_size=0.25, random_state=42)

    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, proba))

    aucs = np.array(aucs)

    print("Repeated speaker-split AUC (n=50)")
    print("Mean:", float(aucs.mean()))
    print("Std :", float(aucs.std()))
    print("Min :", float(aucs.min()))
    print("Max :", float(aucs.max()))

if __name__ == "__main__":
    main()
