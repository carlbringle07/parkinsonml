# FILE: scripts/16_summarize_robustness_waveform_va.py

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_INDEX = ROOT / "data_index"

IN_CSV = DATA_INDEX / "robustness_waveform_va.csv"
OUT_CSV = DATA_INDEX / "robustness_waveform_va_summary.csv"


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing: {IN_CSV} (run scripts/15_robustness_waveform_va.py first)")

    df = pd.read_csv(IN_CSV)

    # Normalize snr_db: 'clean' -> string kept, numeric stays numeric
    # Compute clean AUC per split for gain=1.0
    clean = df[(df["snr_db"].astype(str) == "clean") & (df["gain"] == 1.0)][["split_id", "auc"]].copy()
    clean = clean.rename(columns={"auc": "auc_clean"})

    df2 = df.merge(clean, on="split_id", how="inner")
    df2["auc_drop"] = df2["auc"] - df2["auc_clean"]

    # Summaries per condition
    def summarize(g: pd.DataFrame):
        return pd.Series(
            {
                "n": int(len(g)),
                "auc_mean": float(g["auc"].mean()),
                "auc_std": float(g["auc"].std(ddof=0)),
                "drop_mean": float(g["auc_drop"].mean()),
                "drop_std": float(g["auc_drop"].std(ddof=0)),
            }
        )

    out = df2.groupby(["snr_db", "gain"], as_index=False).apply(summarize).reset_index(drop=True)

    # Sort: clean first, then by snr, then gain
    def sort_key(row):
        snr = row["snr_db"]
        if str(snr) == "clean":
            snr_val = -1.0
        else:
            snr_val = float(snr)
        return (snr_val, float(row["gain"]))

    out = out.sort_values(by=["snr_db", "gain"], key=lambda col: col)  # stable
    # manual reorder using key
    out["_k"] = out.apply(sort_key, axis=1)
    out = out.sort_values("_k").drop(columns=["_k"])

    out.to_csv(OUT_CSV, index=False)

    print("=== DONE ===")
    print("Saved:", OUT_CSV)
    print("\nRobustness summary (AUC and drop vs clean per split):")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
