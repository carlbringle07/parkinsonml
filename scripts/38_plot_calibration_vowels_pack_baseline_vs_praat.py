# scripts/38_plot_calibration_vowels_pack_baseline_vs_praat.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = project_root()
    data_dir = root / "data_index"

    bins_path = data_dir / "calibration_vowels_pack_baseline_vs_praat_bins.csv"
    metrics_path = data_dir / "calibration_vowels_pack_baseline_vs_praat_metrics.csv"
    out_png = data_dir / "calibration_vowels_pack_baseline_vs_praat_reliability.png"

    bins = pd.read_csv(bins_path)
    metrics = pd.read_csv(metrics_path)

    # x = mean confidence, y = empirical accuracy
    x_b = bins["conf_baseline"].to_numpy(float)
    y_b = bins["acc_baseline"].to_numpy(float)
    w_b = bins["count_baseline"].to_numpy(int)

    x_p = bins["conf_praat"].to_numpy(float)
    y_p = bins["acc_praat"].to_numpy(float)
    w_p = bins["count_praat"].to_numpy(int)

    # filtrera bort tomma bins
    mb = w_b > 0
    mp = w_p > 0

    ece_b = float(metrics.loc[metrics["variant"] == "vowels_pack_baseline", "ece"].iloc[0])
    ece_p = float(metrics.loc[metrics["variant"] == "vowels_pack_plus_praat", "ece"].iloc[0])

    plt.figure()
    # perfekt kalibrering
    plt.plot([0, 1], [0, 1])

    # baseline
    plt.plot(x_b[mb], y_b[mb], marker="o")
    # praat
    plt.plot(x_p[mp], y_p[mp], marker="o")

    plt.xlabel("Predicted probability (mean in bin)")
    plt.ylabel("Observed frequency (mean in bin)")
    plt.title(f"Reliability: vowels_pack baseline vs +Praat (ECE {ece_b:.3f} vs {ece_p:.3f})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print("Saved:", out_png.name)


if __name__ == "__main__":
    main()
