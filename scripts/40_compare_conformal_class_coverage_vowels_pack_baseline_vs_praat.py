# scripts/40_compare_conformal_class_coverage_vowels_pack_baseline_vs_praat.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def mean_std(df: pd.DataFrame, col: str) -> tuple[float, float]:
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=1))


def summarize(path: Path, variant_name: str) -> dict:
    df = pd.read_csv(path)
    return {
        "variant": variant_name,
        "alpha": float(df["alpha"].iloc[0]) if "alpha" in df.columns else float("nan"),
        "cov_all_mean": mean_std(df, "cov_all")[0],
        "cov_all_std": mean_std(df, "cov_all")[1],
        "cov_pd_mean": mean_std(df, "cov_pd")[0],
        "cov_pd_std": mean_std(df, "cov_pd")[1],
        "cov_ctrl_mean": mean_std(df, "cov_ctrl")[0],
        "cov_ctrl_std": mean_std(df, "cov_ctrl")[1],
        "n_splits": int(len(df)),
    }


def main() -> None:
    root = project_root()
    data_dir = root / "data_index"

    alpha = 0.10
    baseline_path = data_dir / f"conformal_class_coverage_vowels_pack_alpha{alpha:.2f}.csv"
    praat_path = data_dir / f"conformal_class_coverage_vowels_pack_plus_praat_alpha{alpha:.2f}.csv"
    out_path = data_dir / f"compare_conformal_class_coverage_vowels_pack_baseline_vs_praat_alpha{alpha:.2f}.csv"

    print("=== Compare classwise conformal coverage: vowels_pack baseline vs +Praat ===")
    print("Baseline file:", baseline_path.name, "(exists)" if baseline_path.exists() else "(MISSING)")
    print("Praat file:   ", praat_path.name, "(exists)" if praat_path.exists() else "(MISSING)")

    if not praat_path.exists():
        raise FileNotFoundError(f"Hittar inte Praat-filen: {praat_path}")

    rows = []
    if baseline_path.exists():
        rows.append(summarize(baseline_path, "vowels_pack_baseline"))
    else:
        # placeholder 
        rows.append({
            "variant": "vowels_pack_baseline",
            "alpha": alpha,
            "cov_all_mean": float("nan"),
            "cov_all_std": float("nan"),
            "cov_pd_mean": float("nan"),
            "cov_pd_std": float("nan"),
            "cov_ctrl_mean": float("nan"),
            "cov_ctrl_std": float("nan"),
            "n_splits": 0,
        })

    rows.append(summarize(praat_path, "vowels_pack_plus_praat"))

    out = pd.DataFrame(rows)

    
    if baseline_path.exists():
        b = out.loc[out["variant"] == "vowels_pack_baseline"].iloc[0]
        p = out.loc[out["variant"] == "vowels_pack_plus_praat"].iloc[0]
        delta = {
            "variant": "delta_(praat_minus_baseline)",
            "alpha": alpha,
            "cov_all_mean": p["cov_all_mean"] - b["cov_all_mean"],
            "cov_all_std": float("nan"),
            "cov_pd_mean": p["cov_pd_mean"] - b["cov_pd_mean"],
            "cov_pd_std": float("nan"),
            "cov_ctrl_mean": p["cov_ctrl_mean"] - b["cov_ctrl_mean"],
            "cov_ctrl_std": float("nan"),
            "n_splits": 0,
        }
        out = pd.concat([out, pd.DataFrame([delta])], ignore_index=True)

    out.to_csv(out_path, index=False)
    print("Saved:", out_path.name)
    print("\n", out.to_string(index=False))


if __name__ == "__main__":
    main()
