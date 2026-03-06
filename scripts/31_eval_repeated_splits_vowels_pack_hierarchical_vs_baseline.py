# scripts/31_eval_repeated_splits_vowels_pack_hierarchical_vs_baseline.py
"""
Jämför repeated speaker-split AUC för vowels_pack:
- Baseline person-agg: data_index/features_vowels_pack_person.csv
- Hierarchical person-agg: data_index/features_vowels_pack_person_hierarchical.csv

Output:
- data_index/eval_repeated_splits_vowels_pack_hierarchical_vs_baseline.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_person_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # standardisera label
    df["label"] = df["label"].astype(str).str.strip()
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {
        "speaker",
        "label",
        "group",
        "tasks_present",
        "n_tasks_present",
        "n_tasks_missing",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("Hittade inga numeriska features i person-tabellen.")
    return cols


def make_model() -> Pipeline:
    # Stabil logreg
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="liblinear",
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )),
        ]
    )


def label_to_y(series: pd.Series) -> np.ndarray:
    # pd -> 1, control_elderly -> 0
    mapping = {"pd": 1, "control_elderly": 0}
    y = series.map(mapping)
    if y.isna().any():
        bad = sorted(series[y.isna()].unique().tolist())
        raise ValueError(f"Okända label-värden: {bad}. Förväntar 'pd' och 'control_elderly'.")
    return y.to_numpy(dtype=int)


def eval_repeated(df: pd.DataFrame, n_splits: int = 50, test_size: float = 0.25, seed: int = 42) -> dict:
    X_cols = get_feature_cols(df)
    X = df[X_cols].to_numpy(dtype=float)
    y = label_to_y(df["label"])
    groups = df["speaker"].astype(str).to_numpy()

    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    aucs = []

    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = make_model()
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_te, p)
        aucs.append(float(auc))

    aucs = np.array(aucs, dtype=float)
    return {
        "n_splits": n_splits,
        "test_size": test_size,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)),
        "auc_min": float(np.min(aucs)),
        "auc_max": float(np.max(aucs)),
        "aucs": aucs,
        "n_speakers": int(df["speaker"].nunique()),
        "n_pd": int((df["label"] == "pd").sum()),
        "n_ctrl": int((df["label"] == "control_elderly").sum()),
        "n_features": int(len(X_cols)),
    }


def main() -> None:
    root = project_root()
    data_dir = root / "data_index"

    baseline_path = data_dir / "features_vowels_pack_person.csv"
    hier_path = data_dir / "features_vowels_pack_person_hierarchical.csv"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Hittar inte {baseline_path}. Kör script 26 först.")
    if not hier_path.exists():
        raise FileNotFoundError(f"Hittar inte {hier_path}. Kör script 30 först.")

    df_base = load_person_features(baseline_path)
    df_hier = load_person_features(hier_path)

    res_base = eval_repeated(df_base, n_splits=50, test_size=0.25, seed=42)
    res_hier = eval_repeated(df_hier, n_splits=50, test_size=0.25, seed=42)

    # skillnad per split
    
    delta_mean = res_hier["auc_mean"] - res_base["auc_mean"]

    out = pd.DataFrame([
        {"variant": "baseline_person", **{k: v for k, v in res_base.items() if k != "aucs"}},
        {"variant": "hierarchical_person", **{k: v for k, v in res_hier.items() if k != "aucs"}},
    ])
    out["auc_mean_delta_vs_baseline"] = [0.0, float(delta_mean)]

    out_path = data_dir / "eval_repeated_splits_vowels_pack_hierarchical_vs_baseline.csv"
    out.to_csv(out_path, index=False)

    print("=== Repeated speaker-split AUC comparison (vowels_pack) ===")
    print(f"Saved: {out_path.name}")
    print("")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
