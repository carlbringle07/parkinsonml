# FILE: scripts/29_compare_conformal_screening_va_vs_vowels_pack.py

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_INDEX = ROOT / "data_index"

VA_SCREEN_CSV = DATA_INDEX / "screening_summary_conformal_baseline_va_alpha0.10.csv"
VP_SCREEN_CSV = DATA_INDEX / "screening_summary_conformal_vowels_pack_alpha0.10.csv"

# Fallback: baseline AUC finns i conformal-run-csv (från steg 17)
VA_AUC_FALLBACK_CSV = DATA_INDEX / "conformal_pooled_repeated_baseline_va_alpha0.10.csv"

OUT_CSV = DATA_INDEX / "compare_conformal_screening_va_vs_vowels_pack_alpha0.10.csv"


METRICS = [
    "coverage",
    "decided_rate",
    "unsure_rate",
    "acc_decided",
    "prec_pd_decided",
    "prec_ctrl_decided",
    "auc",
]


def ms(series: pd.Series):
    x = pd.to_numeric(series, errors="coerce").astype(float).values
    return float(np.nanmean(x)), float(np.nanstd(x))


def fmt(mu, sd):
    if np.isnan(mu):
        return "NaN"
    return f"{mu:.3f} ± {sd:.3f}"


def summarize(df: pd.DataFrame, name: str):
    row = {"model": name, "n_splits": int(len(df))}
    for m in METRICS:
        if m in df.columns:
            mu, sd = ms(df[m])
            row[f"{m}_mean"] = mu
            row[f"{m}_std"] = sd
        else:
            row[f"{m}_mean"] = np.nan
            row[f"{m}_std"] = np.nan
    return row


def inject_auc_from_fallback(row: dict, fallback_csv: Path):
    """If auc missing, try to compute it from a fallback per-split csv that has 'auc'."""
    if not np.isnan(row.get("auc_mean", np.nan)):
        return row  # already has auc

    if not fallback_csv.exists():
        return row

    df_fb = pd.read_csv(fallback_csv)
    if "auc" not in df_fb.columns:
        return row

    mu, sd = ms(df_fb["auc"])
    row["auc_mean"] = mu
    row["auc_std"] = sd
    return row


def main():
    if not VA_SCREEN_CSV.exists():
        raise FileNotFoundError(f"Missing: {VA_SCREEN_CSV} (run scripts/19_screening_summary_conformal_baseline_va.py first)")
    if not VP_SCREEN_CSV.exists():
        raise FileNotFoundError(f"Missing: {VP_SCREEN_CSV} (run scripts/28_screening_summary_conformal_vowels_pack.py first)")

    df_va = pd.read_csv(VA_SCREEN_CSV)
    df_vp = pd.read_csv(VP_SCREEN_CSV)

    r_va = summarize(df_va, "VA_only")
    r_va = inject_auc_from_fallback(r_va, VA_AUC_FALLBACK_CSV)

    r_vp = summarize(df_vp, "Vowels_pack")

    out = pd.DataFrame([r_va, r_vp])
    out.to_csv(OUT_CSV, index=False)

    print("\n=== CONFORMAL SCREENING COMPARISON (alpha=0.10) ===")
    print("Saved:", OUT_CSV)
    print()

    pretty = pd.DataFrame(
        [
            {
                "model": r_va["model"],
                "n_splits": r_va["n_splits"],
                "coverage": fmt(r_va["coverage_mean"], r_va["coverage_std"]),
                "decided_rate": fmt(r_va["decided_rate_mean"], r_va["decided_rate_std"]),
                "unsure_rate": fmt(r_va["unsure_rate_mean"], r_va["unsure_rate_std"]),
                "acc_decided": fmt(r_va["acc_decided_mean"], r_va["acc_decided_std"]),
                "prec_pd": fmt(r_va["prec_pd_decided_mean"], r_va["prec_pd_decided_std"]),
                "prec_ctrl": fmt(r_va["prec_ctrl_decided_mean"], r_va["prec_ctrl_decided_std"]),
                "auc": fmt(r_va["auc_mean"], r_va["auc_std"]),
            },
            {
                "model": r_vp["model"],
                "n_splits": r_vp["n_splits"],
                "coverage": fmt(r_vp["coverage_mean"], r_vp["coverage_std"]),
                "decided_rate": fmt(r_vp["decided_rate_mean"], r_vp["decided_rate_std"]),
                "unsure_rate": fmt(r_vp["unsure_rate_mean"], r_vp["unsure_rate_std"]),
                "acc_decided": fmt(r_vp["acc_decided_mean"], r_vp["acc_decided_std"]),
                "prec_pd": fmt(r_vp["prec_pd_decided_mean"], r_vp["prec_pd_decided_std"]),
                "prec_ctrl": fmt(r_vp["prec_ctrl_decided_mean"], r_vp["prec_ctrl_decided_std"]),
                "auc": fmt(r_vp["auc_mean"], r_vp["auc_std"]),
            },
        ]
    )

    with pd.option_context("display.max_columns", 50, "display.width", 200):
        print(pretty.to_string(index=False))

    print("\nNotes:")
    print(f"- decided_rate: VA_only {r_va['decided_rate_mean']:.3f} vs Vowels_pack {r_vp['decided_rate_mean']:.3f} (higher = fewer 'Osäker').")
    print("- coverage should be >= ~0.90 for alpha=0.10 (higher = more conservative).")
    if np.isnan(r_va["auc_mean"]):
        print(f"- NOTE: VA_only auc missing and fallback not available: {VA_AUC_FALLBACK_CSV}")


if __name__ == "__main__":
    main()
