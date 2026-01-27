# scripts/37_calibration_vowels_pack_baseline_vs_praat.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, brier_score_loss


# -----------------------------
# Paths / utils
# -----------------------------
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# -----------------------------
# Calibration metrics
# -----------------------------
def ece_from_bins(bin_acc: np.ndarray, bin_conf: np.ndarray, bin_count: np.ndarray) -> float:
    n = float(np.sum(bin_count))
    if n <= 0:
        return float("nan")
    w = bin_count / n
    return float(np.sum(w * np.abs(bin_acc - bin_conf)))


def reliability_bins_uniform(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """
    Uniform bins i [0,1] (behåller ditt gamla upplägg exakt)
    Return:
      edges (n_bins+1),
      bin_count (n_bins),
      bin_acc (n_bins),
      bin_conf (n_bins),
      ece
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin_ids: 0..n_bins-1
    bin_ids = np.digitize(y_prob, edges[1:-1], right=False)

    bin_count = np.zeros(n_bins, dtype=int)
    bin_acc = np.full(n_bins, np.nan, dtype=float)
    bin_conf = np.full(n_bins, np.nan, dtype=float)

    for b in range(n_bins):
        idx = np.where(bin_ids == b)[0]
        bin_count[b] = len(idx)
        if len(idx) > 0:
            bin_acc[b] = float(np.mean(y_true[idx]))
            bin_conf[b] = float(np.mean(y_prob[idx]))

    ece = ece_from_bins(
        np.nan_to_num(bin_acc, nan=0.0),
        np.nan_to_num(bin_conf, nan=0.0),
        bin_count.astype(float),
    )
    return edges, bin_count, bin_acc, bin_conf, ece


def reliability_bins_from_raw_df(df_raw: pd.DataFrame, n_bins: int = 20, strategy: str = "quantile") -> pd.DataFrame:
    """
    Skapar en 'bins-tabell' med fler datapunkter (n_bins).
    df_raw måste ha kolumner: y_true (0/1), y_prob (float 0..1)

    strategy:
      - "quantile": lika många observationer per bin (rekommenderas för snyggare kurva)
      - "uniform": lika breda bins i [0,1] (bättre om du vill merge:a baseline+praat i samma tabell)
    """
    df = df_raw.copy()
    df["y_true"] = df["y_true"].astype(int)
    df["y_prob"] = df["y_prob"].astype(float)

    if strategy == "quantile":
        df["bin"] = pd.qcut(df["y_prob"], q=n_bins, duplicates="drop")
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        df["bin"] = pd.cut(df["y_prob"], bins=edges, include_lowest=True)
    else:
        raise ValueError(f"Okänd strategy: {strategy}")

    out = (
        df.groupby("bin", observed=True)
        .agg(
            bin_left=("y_prob", "min"),
            bin_right=("y_prob", "max"),
            count=("y_prob", "size"),
            conf=("y_prob", "mean"),
            acc=("y_true", "mean"),
        )
        .reset_index(drop=True)
    )
    out["bin_mid"] = 0.5 * (out["bin_left"] + out["bin_right"])
    return out


def save_raw_preds(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, variant: str) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {"variant": variant, "y_true": y_true.astype(int), "y_prob": y_prob.astype(float)}
    )
    df.to_csv(out_path, index=False)
    return df


# -----------------------------
# Data loading / feature columns
# -----------------------------
def load_person_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" not in df.columns or "speaker" not in df.columns:
        raise ValueError(f"{path.name} saknar 'label' eller 'speaker'.")

    y = df["label"].map({"pd": 1, "control_elderly": 0})
    if y.isna().any():
        bad = df.loc[y.isna(), "label"].unique().tolist()
        raise ValueError(f"Oväntade label-värden i {path.name}: {bad}")

    df = df.copy()
    df["y"] = y.astype(int)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"speaker", "label", "group", "y", "tasks_present", "n_tasks_present", "n_tasks_missing"}
    cols: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


# -----------------------------
# Model: repeated speaker split -> OOS probabilities
# -----------------------------
def fit_predict_probs(df: pd.DataFrame, feat_cols: list[str], seed: int, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    X = df[feat_cols].to_numpy(float)
    y = df["y"].to_numpy(int)
    groups = df["speaker"].astype(str).to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="liblinear")),
        ]
    )
    model.fit(X[train_idx], y[train_idx])
    prob = model.predict_proba(X[test_idx])[:, 1]
    return y[test_idx], prob


def aggregate_probs_over_splits(df: pd.DataFrame, feat_cols: list[str], n_splits: int = 50, test_size: float = 0.30) -> tuple[np.ndarray, np.ndarray]:
    ys = []
    ps = []
    for seed in range(n_splits):
        y_t, p_t = fit_predict_probs(df, feat_cols, seed=seed, test_size=test_size)
        ys.append(y_t)
        ps.append(p_t)
    return np.concatenate(ys), np.concatenate(ps)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    root = project_root()
    data_dir = root / "data_index"

    base_path = data_dir / "features_vowels_pack_person.csv"
    praat_path = data_dir / "features_vowels_pack_person_plus_praat.csv"

    base = load_person_table(base_path)
    praat = load_person_table(praat_path)

    base_cols = get_feature_cols(base)
    praat_cols = get_feature_cols(praat)

    print("=== Calibration (repeated speaker-split): vowels_pack baseline vs +Praat ===")
    print(f"Baseline features: {len(base_cols)}")
    print(f"Praat+baseline features: {len(praat_cols)}")

    # Samla out-of-sample probabilities över splits
    yb, pb = aggregate_probs_over_splits(base, base_cols, n_splits=50, test_size=0.30)
    yp, pp = aggregate_probs_over_splits(praat, praat_cols, n_splits=50, test_size=0.30)

    # --- Behåll din gamla bins-fil (uniform, 10 bins) + ECE ---
    edges, cnt_b, acc_b, conf_b, ece_b = reliability_bins_uniform(yb, pb, n_bins=10)
    edges2, cnt_p, acc_p, conf_p, ece_p = reliability_bins_uniform(yp, pp, n_bins=10)
    # sanity: edges ska matcha om samma n_bins
    if not np.allclose(edges, edges2):
        print("VARNING: bin-edges skiljer (oväntat).")

    # Övriga metrics
    ll_b = float(log_loss(yb, pb, labels=[0, 1]))
    ll_p = float(log_loss(yp, pp, labels=[0, 1]))
    br_b = float(brier_score_loss(yb, pb))
    br_p = float(brier_score_loss(yp, pp))

    metrics = pd.DataFrame(
        [
            {"variant": "vowels_pack_baseline", "ece": ece_b, "logloss": ll_b, "brier": br_b, "n_points": int(len(pb))},
            {"variant": "vowels_pack_plus_praat", "ece": ece_p, "logloss": ll_p, "brier": br_p, "n_points": int(len(pp))},
        ]
    )

    bins = pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count_baseline": cnt_b,
            "acc_baseline": acc_b,
            "conf_baseline": conf_b,
            "count_praat": cnt_p,
            "acc_praat": acc_p,
            "conf_praat": conf_p,
        }
    )
    bins["bin_mid"] = 0.5 * (bins["bin_left"] + bins["bin_right"])

    metrics_path = data_dir / "calibration_vowels_pack_baseline_vs_praat_metrics.csv"
    bins_path = data_dir / "calibration_vowels_pack_baseline_vs_praat_bins.csv"
    metrics.to_csv(metrics_path, index=False)
    bins.to_csv(bins_path, index=False)

    print("\nSaved:", metrics_path.name)
    print("Saved:", bins_path.name)
    print("\nMetrics:")
    print(metrics.to_string(index=False))

    # --- NYTT: spara råprediktioner + skapa fler bins (riktiga fler datapunkter) ---
    # Du kan ändra dessa två:
    N_BINS = 50
    STRATEGY = "quantile"   # "quantile" rekommenderas, "uniform" om du vill ha samma bin-gränser

    raw_base_path = data_dir / "calibration_vowels_pack_baseline_raw_preds.csv"
    raw_praat_path = data_dir / "calibration_vowels_pack_plus_praat_raw_preds.csv"

    df_raw_base = save_raw_preds(yb, pb, raw_base_path, variant="vowels_pack_baseline")
    df_raw_praat = save_raw_preds(yp, pp, raw_praat_path, variant="vowels_pack_plus_praat")

    # fler-punkts bins per variant (sparas som separata filer)
    b_base = reliability_bins_from_raw_df(df_raw_base[["y_true", "y_prob"]], n_bins=N_BINS, strategy=STRATEGY)
    b_praat = reliability_bins_from_raw_df(df_raw_praat[["y_true", "y_prob"]], n_bins=N_BINS, strategy=STRATEGY)

    base_bins_path = data_dir / f"calibration_vowels_pack_baseline_bins_{STRATEGY}_{N_BINS}.csv"
    praat_bins_path = data_dir / f"calibration_vowels_pack_plus_praat_bins_{STRATEGY}_{N_BINS}.csv"

    b_base.to_csv(base_bins_path, index=False)
    b_praat.to_csv(praat_bins_path, index=False)

    print("\nSaved raw:", raw_base_path.name)
    print("Saved raw:", raw_praat_path.name)
    print("Saved bins (more points):", base_bins_path.name)
    print("Saved bins (more points):", praat_bins_path.name)


if __name__ == "__main__":
    main()
