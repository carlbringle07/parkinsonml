# scripts/36_compare_screening_vowels_pack_baseline_vs_praat.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_summary(df: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    """
    Standardiserar en summary till kolumner:
    auc_mean, auc_std, coverage_mean, coverage_std, decided_rate_mean, decided_rate_std,
    unsure_rate_mean, unsure_rate_std, acc_decided_mean, acc_decided_std, alpha, variant

    Hanterar två format:
    A) 1-radigt format med *_mean (t.ex. praat-summary)
    B) per-split-format med kolumner som coverage/auc/... (t.ex. baseline från script 28)
       -> då aggregerar vi mean/std över alla rader.
    """
    out = pd.DataFrame(index=[0])
    out["variant"] = variant_name

    # alpha
    if "alpha" in df.columns and pd.notna(df.loc[df.index[0], "alpha"]):
        out["alpha"] = float(df.loc[df.index[0], "alpha"])
    else:
        out["alpha"] = 0.10

    # Fall A: redan mean/std-format
    if ("coverage_mean" in df.columns) or ("auc_mean" in df.columns):
        for c in [
            "auc_mean", "auc_std",
            "coverage_mean", "coverage_std",
            "decided_rate_mean", "decided_rate_std",
            "unsure_rate_mean", "unsure_rate_std",
            "acc_decided_mean", "acc_decided_std",
        ]:
            out[c] = float(df.loc[df.index[0], c]) if c in df.columns else np.nan
        return out

    # Fall B: per-split-format
    
    metrics = ["auc", "coverage", "decided_rate", "unsure_rate", "acc_decided"]
    for m in metrics:
        if m in df.columns:
            vals = pd.to_numeric(df[m], errors="coerce").to_numpy(float)
            out[f"{m}_mean"] = float(np.nanmean(vals))
            out[f"{m}_std"] = float(np.nanstd(vals, ddof=1))
        else:
            out[f"{m}_mean"] = np.nan
            out[f"{m}_std"] = np.nan

    # byt namn auc_mean osv till standardnamn
    rename = {
        "auc_mean": "auc_mean",
        "auc_std": "auc_std",
        "coverage_mean": "coverage_mean",
        "coverage_std": "coverage_std",
        "decided_rate_mean": "decided_rate_mean",
        "decided_rate_std": "decided_rate_std",
        "unsure_rate_mean": "unsure_rate_mean",
        "unsure_rate_std": "unsure_rate_std",
        "acc_decided_mean": "acc_decided_mean",
        "acc_decided_std": "acc_decided_std",
    }
    

 
    for c in [
        "auc_mean", "auc_std",
        "coverage_mean", "coverage_std",
        "decided_rate_mean", "decided_rate_std",
        "unsure_rate_mean", "unsure_rate_std",
        "acc_decided_mean", "acc_decided_std",
    ]:
        if c not in out.columns:
            out[c] = np.nan

    return out


def main() -> None:
    root = project_root()
    data_dir = root / "data_index"

    base_path = data_dir / "screening_summary_conformal_vowels_pack_alpha0.10.csv"
    praat_path = data_dir / "screening_summary_conformal_vowels_pack_plus_praat_alpha0.10.csv"

    if not base_path.exists():
        raise FileNotFoundError(f"Hittar inte baseline summary: {base_path.name}")
    if not praat_path.exists():
        raise FileNotFoundError(f"Hittar inte praat summary: {praat_path.name}")

    base_raw = pd.read_csv(base_path)
    praat_raw = pd.read_csv(praat_path)

    base = normalize_summary(base_raw, "vowels_pack_baseline")
    praat = normalize_summary(praat_raw, "vowels_pack_plus_praat")

    keep_cols = [
        "variant", "alpha",
        "auc_mean", "auc_std",
        "coverage_mean", "coverage_std",
        "decided_rate_mean", "decided_rate_std",
        "unsure_rate_mean", "unsure_rate_std",
        "acc_decided_mean", "acc_decided_std",
    ]

    out = pd.concat([base[keep_cols], praat[keep_cols]], ignore_index=True)

    delta = {"variant": "delta_(praat_minus_baseline)", "alpha": float(out.loc[0, "alpha"])}
    for c in ["auc_mean", "coverage_mean", "decided_rate_mean", "unsure_rate_mean", "acc_decided_mean"]:
        delta[c] = float(out.loc[1, c] - out.loc[0, c])
    out = pd.concat([out, pd.DataFrame([delta])], ignore_index=True)

    out_path = data_dir / "compare_screening_vowels_pack_baseline_vs_praat_alpha0.10.csv"
    out.to_csv(out_path, index=False)

    print("=== Compare conformal screening: vowels_pack baseline vs +Praat ===")
    print(f"Baseline file: {base_path.name}  (rows={len(base_raw)})")
    print(f"Praat file:    {praat_path.name} (rows={len(praat_raw)})")
    print(f"Saved: {out_path.name}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
