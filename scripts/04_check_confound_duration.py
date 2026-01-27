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

def eval_auc(df: pd.DataFrame, feature_cols: list[str], n_splits: int = 50) -> tuple[float,float,float,float]:
    X = df[feature_cols]
    y = df["y"].to_numpy()
    groups = df["speaker"].to_numpy()

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=0)),
    ])

    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=42)
    aucs = []
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, proba))

    aucs = np.array(aucs)
    return float(aucs.mean()), float(aucs.std()), float(aucs.min()), float(aucs.max())

def main() -> None:
    df = pd.read_csv(FEAT_PATH)
    df = df[df["label"].isin(["pd", "control_elderly"])].copy()
    df["y"] = (df["label"] == "pd").astype(int)

    all_features = [c for c in df.columns if c not in {"speaker", "label", "y"}]

    # 1) Endast duration
    mean1, std1, mn1, mx1 = eval_auc(df, ["duration_s"])
    print("\nA) Only duration_s")
    print("Mean:", mean1, "Std:", std1, "Min:", mn1, "Max:", mx1)

    # 2) Alla features
    mean2, std2, mn2, mx2 = eval_auc(df, all_features)
    print("\nB) All features")
    print("Mean:", mean2, "Std:", std2, "Min:", mn2, "Max:", mx2)

    # 3) Alla utom duration
    no_duration = [c for c in all_features if c != "duration_s"]
    mean3, std3, mn3, mx3 = eval_auc(df, no_duration)
    print("\nC) All features EXCEPT duration_s")
    print("Mean:", mean3, "Std:", std3, "Min:", mn3, "Max:", mx3)

if __name__ == "__main__":
    main()
