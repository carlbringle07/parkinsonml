# scripts/42_robustness_summary_vowels_pack_baseline_vs_praat.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def project_root() -> Path:
    # scripts/.. -> project root
    return Path(__file__).resolve().parents[1]


def safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def main():
    root = project_root()
    data_dir = root / "data_index"

    inp = data_dir / "robustness_waveform_vowels_pack_baseline_vs_praat.csv"
    if not inp.exists():
        raise FileNotFoundError(
            f"Hittar inte inputfilen:\n  {inp}\n"
            "Kör steg 41 först så att robustness_waveform_*.csv skapas i data_index."
        )

    df = pd.read_csv(inp)

    required_cols = {"variant", "condition", "auc_mean"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input saknar kolumner {missing}. Finns: {list(df.columns)}")

    # Coerce numerics
    for c in ["auc_mean", "auc_std", "auc_min", "auc_max", "auc_mean_delta_vs_clean"]:
        if c in df.columns:
            df[c] = safe_float_series(df[c])

    # Basic cleaning
    df["variant"] = df["variant"].astype(str)
    df["condition"] = df["condition"].astype(str)

    # Keep one row per (variant, condition) – om df råkar innehålla dubbletter
    # tar vi medel 
    agg_cols = {}
    for c in ["n_splits", "test_size", "n_speakers", "auc_mean", "auc_std", "auc_min", "auc_max", "auc_mean_delta_vs_clean"]:
        if c in df.columns:
            agg_cols[c] = "mean"
    df = df.groupby(["variant", "condition"], as_index=False).agg(agg_cols)

    # Variant ordering (nice prints)
    def variant_key(v: str) -> int:
        if v == "baseline":
            return 0
        if v == "baseline_plus_praat":
            return 1
        return 2

    variants = sorted(df["variant"].unique().tolist(), key=variant_key)
    conditions = df["condition"].unique().tolist()
    # Put clean first if present
    conditions = sorted(conditions, key=lambda x: (0 if x == "clean" else 1, x))

    # ----- Build pivot tables -----
    auc_pivot = df.pivot(index="condition", columns="variant", values="auc_mean").reindex(index=conditions, columns=variants)

    # delta vs clean (if not provided, compute it)
    if "auc_mean_delta_vs_clean" in df.columns and df["auc_mean_delta_vs_clean"].notna().any():
        delta_pivot = df.pivot(index="condition", columns="variant", values="auc_mean_delta_vs_clean").reindex(index=conditions, columns=variants)
    else:
        # compute per variant: auc_mean(condition) - auc_mean(clean)
        delta_pivot = auc_pivot.copy()
        for v in variants:
            if "clean" in auc_pivot.index:
                delta_pivot[v] = auc_pivot[v] - auc_pivot.loc["clean", v]
            else:
                delta_pivot[v] = np.nan

    # praat - baseline (if both exist)
    has_base = "baseline" in auc_pivot.columns
    has_praat = "baseline_plus_praat" in auc_pivot.columns

    compare = pd.DataFrame(index=conditions)
    compare.index.name = "condition"

    if has_base:
        compare["auc_baseline"] = auc_pivot["baseline"]
        compare["delta_vs_clean_baseline"] = delta_pivot["baseline"]
    else:
        compare["auc_baseline"] = np.nan
        compare["delta_vs_clean_baseline"] = np.nan

    if has_praat:
        compare["auc_plus_praat"] = auc_pivot["baseline_plus_praat"]
        compare["delta_vs_clean_plus_praat"] = delta_pivot["baseline_plus_praat"]
    else:
        compare["auc_plus_praat"] = np.nan
        compare["delta_vs_clean_plus_praat"] = np.nan

    if has_base and has_praat:
        compare["praat_minus_baseline_auc"] = compare["auc_plus_praat"] - compare["auc_baseline"]
        compare["praat_minus_baseline_delta_vs_clean"] = (
            compare["delta_vs_clean_plus_praat"] - compare["delta_vs_clean_baseline"]
        )
    else:
        compare["praat_minus_baseline_auc"] = np.nan
        compare["praat_minus_baseline_delta_vs_clean"] = np.nan

    # ----- Save outputs -----
    out_summary = data_dir / "robustness_summary_vowels_pack_baseline_vs_praat.csv"
    out_auc_table = data_dir / "robustness_auc_mean_table_vowels_pack_baseline_vs_praat.csv"
    out_delta_table = data_dir / "robustness_delta_vs_clean_table_vowels_pack_baseline_vs_praat.csv"

    compare.reset_index().to_csv(out_summary, index=False)
    auc_pivot.reset_index().to_csv(out_auc_table, index=False)
    delta_pivot.reset_index().to_csv(out_delta_table, index=False)

    # ----- Print quick human-readable summary -----
    print("=== Robustness summary: vowels_pack baseline vs +Praat ===")
    print(f"Input:  {inp.name}  (rows={len(df)})")
    print(f"Saved:  {out_summary.name}")
    print(f"Saved:  {out_auc_table.name}")
    print(f"Saved:  {out_delta_table.name}")
    print()

    # Show the main comparison table (rounded)
    show = compare.copy()
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print("Comparison (per condition):")
        print(show.round(6))

    # Also show the key “worst robustness” case (most negative delta vs clean)
    if "clean" in compare.index:
        # worst condition for each variant (most negative delta)
        for label, col in [
            ("baseline", "delta_vs_clean_baseline"),
            ("baseline_plus_praat", "delta_vs_clean_plus_praat"),
        ]:
            if col in compare.columns and compare[col].notna().any():
                worst_cond = compare[col].idxmin()
                worst_val = compare.loc[worst_cond, col]
                print(f"\nWorst delta vs clean for {label}: {worst_cond}  ({worst_val:.6f})")

    print("\nDONE.")


if __name__ == "__main__":
    main()
