# FILE: scripts/21_final_project_summary_va.py

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_INDEX = ROOT / "data_index"

BASE_SCREEN = DATA_INDEX / "screening_summary_conformal_baseline_va_alpha0.10.csv"
PRAAT_SCREEN = DATA_INDEX / "screening_summary_conformal_baseline_plus_praat_va_alpha0.10.csv"
ROB_SUM = DATA_INDEX / "robustness_waveform_va_summary.csv"

OUT_TXT = DATA_INDEX / "final_summary_va.txt"


def pm(series: pd.Series):
    x = series.dropna().values
    return float(np.mean(x)), float(np.std(x, ddof=0))


def load_summary(path: Path, name: str):
    df = pd.read_csv(path)
    out = {"model": name, "n_splits": len(df)}
    for col in [
        "coverage",
        "decided_rate",
        "unsure_rate",
        "acc_decided",
        "prec_pd_decided",
        "prec_ctrl_decided",
    ]:
        m, s = pm(df[col])
        out[col] = f"{m:.3f} ± {s:.3f}"
    return out


def main():
    lines = []

    rows = []
    if BASE_SCREEN.exists():
        rows.append(load_summary(BASE_SCREEN, "baseline"))
    if PRAAT_SCREEN.exists():
        rows.append(load_summary(PRAAT_SCREEN, "baseline_plus_praat"))

    if rows:
        tab = pd.DataFrame(rows)[
            ["model", "n_splits", "coverage", "decided_rate", "unsure_rate", "acc_decided", "prec_pd_decided", "prec_ctrl_decided"]
        ]
        lines.append("SCREENING (alpha=0.10), mean ± std\n")
        lines.append(tab.to_string(index=False))
        lines.append("\n")
        print("\n=== SCREENING (alpha=0.10), mean ± std ===")
        print(tab.to_string(index=False))
    else:
        print("Missing screening csv files.")
        lines.append("Missing screening csv files.\n")

    if ROB_SUM.exists():
        rob = pd.read_csv(ROB_SUM)
        rob["snr_db"] = rob["snr_db"].astype(str)
        keep = rob[((rob["snr_db"] == "clean") & (rob["gain"] == 1.0)) |
                   ((rob["snr_db"].isin(["10.0", "20.0"])) & (rob["gain"] == 1.0))].copy()

        keep2 = keep[["snr_db", "gain", "auc_mean", "auc_std", "drop_mean", "drop_std"]].copy()
        for c in ["auc_mean", "auc_std", "drop_mean", "drop_std"]:
            keep2[c] = keep2[c].map(lambda v: f"{float(v):.3f}")

        lines.append("ROBUSTNESS (AUC drop vs clean gain=1.0)\n")
        lines.append(keep2.to_string(index=False))
        lines.append("\n")

        print("\n=== ROBUSTNESS (AUC drop vs clean gain=1.0) ===")
        print(keep2.to_string(index=False))
    else:
        print("\nMissing robustness summary csv.")
        lines.append("Missing robustness summary csv.\n")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\nSaved pretty summary:", OUT_TXT)


if __name__ == "__main__":
    main()
