from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt

FEAT_PATH = Path("data_index/features_va_person.csv")
OUT_ROC = Path("data_index/roc_baseline_va.png")

def main() -> None:
    if not FEAT_PATH.exists():
        raise FileNotFoundError(
            f"Hittar inte {FEAT_PATH}. Kör scripts/01_extract_features_va.py först."
        )

    df = pd.read_csv(FEAT_PATH)

    # Binary label: pd=1, control_elderly=0
    df = df[df["label"].isin(["pd", "control_elderly"])].copy()
    df["y"] = (df["label"] == "pd").astype(int)

    # Features 
    drop_cols = {"speaker", "label", "y"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["y"].to_numpy()
    groups = df["speaker"].to_numpy()

    # Speaker-split: samma person får inte hamna i både train och test
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("Antal personer totalt:", len(df))
    print("Train personer:", len(np.unique(groups[train_idx])))
    print("Test personer:", len(np.unique(groups[test_idx])))
    print("Train class balance (pd=1):", y_train.mean())
    print("Test  class balance (pd=1):", y_test.mean())

    # Modell 1: Logistic Regression (skalning behövs)
    logreg = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
    ])

    # Modell 2: Random Forest (ingen skalning behövs)
    rf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced_subsample"
        )),
    ])

    models = {
        "LogReg": logreg,
        "RandomForest": rf,
    }

    plt.figure()
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, proba)
        fpr, tpr, _ = roc_curve(y_test, proba)

        print(f"\n=== {name} ===")
        print("AUC:", round(float(auc), 3))

        # Enkel threshold 0.5 för första baseline 
        pred = (proba >= 0.5).astype(int)
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred, target_names=["control_elderly", "pd"]))

        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC (VA tasks, person-level)")
    plt.legend(loc="lower right")

    OUT_ROC.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_ROC, dpi=200, bbox_inches="tight")
    plt.close()

    print("\n✅ Sparade ROC-figur:", OUT_ROC)

if __name__ == "__main__":
    main()
