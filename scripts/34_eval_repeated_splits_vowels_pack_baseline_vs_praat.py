# scripts/34_eval_repeated_splits_vowels_pack_baseline_vs_praat.py
"""
Repeated speaker-splits (n=50) för vowels_pack:
- Baseline: features_vowels_pack_person.csv
- Baseline + Praat: features_vowels_pack_person_plus_praat.csv

Output:
- data_index/eval_repeated_splits_vowels_pack_baseline_vs_praat.csv
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


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = df["label"].astype(str).str.strip()
    df["speaker"] = df["speaker"].astype(str)
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"speaker", "label", "group", "tasks_present", "n_tasks_present", "n_tasks_missing"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("Hittade inga numeriska features.")
    return cols


def y_from_label(lbl: pd.Series) -> np.ndarray:
    mapping = {"pd": 1, "control_elderly": 0}
    y = lbl.map(mapping)
    if y.isna().any():
        bad = sorted(lbl[y.isna()].unique().tolist())
        raise ValueError(f"Okända label-värden: {bad}")
    return y.to_numpy(dtype=int)


def make_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])


def eval_repeated(df: pd.DataFrame, n_splits: int = 50, test_size: float = 0.25, seed: int = 42) -> dict:
    cols = feature_cols(df)
    X = df[cols].to_numpy(float)
    y = y_from_label(df["label"])
    groups = df["speaker"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    aucs = []

    for tr_idx, te_idx in splitter.split(X, y, groups=groups):
        model = make_model()
        model.fit(X[tr_idx], y[tr_idx])
        p = model.predict_proba(X[te_idx])[:, 1]
        aucs.append(float(roc_auc_score(y[te_idx], p)))

    aucs = np.array(aucs, dtype=float)
    return {
        "n_splits": n_splits,
        "test_size": test_size,
        "n_speakers": int(df["speaker"].nunique()),
        "n_pd": int((df["label"] == "pd").sum()),
        "n_ctrl": int((df["label"] == "control_elderly").sum()),
        "n_features": int(len(cols)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)),
        "auc_min": float(np.min(aucs)),
        "auc_max": float(np.max(aucs)),
    }


def main() -> None:
    root = project_root()
    data_dir = root / "data_index"

    base_path = data_dir / "features_vowels_pack_person.csv"
    praat_path = data_dir / "features_vowels_pack_person_plus_praat.csv"

    if not base_path.exists():
        raise FileNotFoundError(f"Hittar inte {base_path}")
    if not praat_path.exists():
        raise FileNotFoundError(f"Hittar inte {praat_path}")

    base = load_df(base_path)
    plus = load_df(praat_path)

    res_base = eval_repeated(base)
    res_plus = eval_repeated(plus)

    out = pd.DataFrame([
        {"variant": "baseline", **res_base},
        {"variant": "baseline_plus_praat", **res_plus},
    ])
    out["auc_mean_delta_vs_baseline"] = [0.0, res_plus["auc_mean"] - res_base["auc_mean"]]

    out_path = data_dir / "eval_repeated_splits_vowels_pack_baseline_vs_praat.csv"
    out.to_csv(out_path, index=False)

    print("=== Repeated speaker-splits AUC: vowels_pack baseline vs +Praat ===")
    print(f"Saved: {out_path.name}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
