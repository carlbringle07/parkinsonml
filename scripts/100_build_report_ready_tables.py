# scripts/100_build_report_ready_tables.py
# Creates report-ready CSV tables + copies key plots into report_artifacts/REPORT_READY

from __future__ import annotations

import shutil
from pathlib import Path
import pandas as pd


# -------- Paths --------
ROOT = Path(".")
RA = ROOT / "report_artifacts"
OUT = RA / "REPORT_READY"

# Preferred source files (in report_artifacts). If missing, we try data_index as fallback.
PREFERRED_FILES = {
    # dataset stats
    "dataset_stats": ["dataset_stats_vowels_pack.csv"],

    # AUC eval (baseline vs praat)
    "auc_eval": [
        "eval_repeated_splits_vowels_pack_baseline_vs_praat.csv",
        "eval_repeated_splits_vowels_pack.csv",
        "eval_repeated_splits_vowels_pack_hierarchical_vs_baseline.csv",
    ],

    # screening summary (conformal)
    "screening": [
        "screening_summary_conformal_vowels_pack_alpha0.10.csv",
        "screening_summary_conformal_vowels_pack_plus_praat_alpha0.10.csv",
        "compare_screening_vowels_pack_baseline_vs_praat_alpha0.10.csv",
    ],

    # conformal class coverage
    "class_coverage": [
        "conformal_class_coverage_vowels_pack_alpha0.10.csv",
        "conformal_class_coverage_vowels_pack_plus_praat_alpha0.10.csv",
        "compare_conformal_class_coverage_vowels_pack_baseline_vs_praat_alpha0.10.csv",
    ],

    # robustness
    "robustness": [
        "robustness_summary_vowels_pack_baseline_vs_praat.csv",
        "robustness_auc_mean_table_vowels_pack_baseline_vs_praat.csv",
        "robustness_conformal_screening_vowels_pack_baseline_vs_praat_alpha0.10_summary.csv",
        "robustness_conformal_screening_vowels_pack_baseline_vs_praat_alpha0.10_per_split.csv",
        "robustness_delta_vs_clean_table_vowels_pack_baseline_vs_praat.csv",
        "robustness_waveform_vowels_pack_baseline_vs_praat.csv",
        "per_split_clean_summary_from_robustness.csv",
    ],

    # calibration
    "calibration": [
        "calibration_vowels_pack_baseline_vs_praat_bins.csv",
        "calibration_vowels_pack_baseline_vs_praat_metrics.csv",
        "calibration_vowels_pack_baseline_vs_praat_reliability.png",
    ],
}


def find_file(name: str) -> Path | None:
    """Find a file in report_artifacts first, else in data_index."""
    cand1 = RA / name
    if cand1.exists():
        return cand1
    cand2 = ROOT / "data_index" / name
    if cand2.exists():
        return cand2
    return None


def safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def maybe_add_delta_row(df_base: pd.DataFrame, df_praat: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """
    Creates a delta row (praat - baseline) for numeric columns.
    Assumes each DF is (usually) 1-row summary; if more, we take mean over rows.
    """
    if df_base.empty or df_praat.empty:
        return pd.DataFrame()

    base_num = df_base.select_dtypes(include="number").mean(numeric_only=True)
    praat_num = df_praat.select_dtypes(include="number").mean(numeric_only=True)
    delta_num = (praat_num - base_num)

    # Try to carry over non-numeric fields sensibly:
    delta = {}
    for c in key_cols:
        if c in df_base.columns:
            if c.lower() in ["variant", "model"]:
                delta[c] = "delta_(praat_minus_baseline)"
            else:
                delta[c] = df_base[c].iloc[0]

    # Put numeric deltas
    for c, v in delta_num.items():
        delta[c] = v

    return pd.DataFrame([delta])


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    produced = []

    # ---- 1) REPORT_table_dataset.csv ----
    ds_path = None
    for fn in PREFERRED_FILES["dataset_stats"]:
        ds_path = find_file(fn)
        if ds_path:
            break

    if ds_path:
        ds = safe_read_csv(ds_path)
        # expected columns: metric,value (from your earlier command)
        # if not, keep raw
        save_csv(ds, OUT / "REPORT_table_dataset.csv")
        produced.append(("REPORT_table_dataset.csv", ds_path.name))
    else:
        print("[WARN] Could not find dataset_stats_vowels_pack.csv (in report_artifacts or data_index).")

    # ---- 2) REPORT_table_auc.csv ----
    auc_frames = []
    for fn in PREFERRED_FILES["auc_eval"]:
        p = find_file(fn)
        if not p:
            continue
        df = safe_read_csv(p)
        df.insert(0, "source_file", p.name)
        auc_frames.append(df)

    if auc_frames:
        auc = pd.concat(auc_frames, ignore_index=True)
        # Keep a clean subset if present, else export all columns
        preferred_cols = [
            "source_file", "variant", "model", "n_splits", "test_size", "n_speakers",
            "n_pd", "n_ctrl", "n_features", "auc_mean", "auc_std", "auc_min", "auc_max",
            "auc_mean_delta_vs_baseline", "auc_mean_delta_v"
        ]
        cols = [c for c in preferred_cols if c in auc.columns]
        if cols:
            auc = auc[cols]
        save_csv(auc, OUT / "REPORT_table_auc.csv")
        produced.append(("REPORT_table_auc.csv", "multiple"))
    else:
        print("[WARN] No AUC eval files found. (eval_repeated_splits_*.csv)")

    # ---- 3) REPORT_table_screening.csv ----
    base_scr = find_file("screening_summary_conformal_vowels_pack_alpha0.10.csv")
    praat_scr = find_file("screening_summary_conformal_vowels_pack_plus_praat_alpha0.10.csv")
    compare_scr = find_file("compare_screening_vowels_pack_baseline_vs_praat_alpha0.10.csv")

    scr_out = []
    if base_scr:
        dfb = safe_read_csv(base_scr)
        dfb.insert(0, "source_file", base_scr.name)
        scr_out.append(dfb)
    if praat_scr:
        dfp = safe_read_csv(praat_scr)
        dfp.insert(0, "source_file", praat_scr.name)
        scr_out.append(dfp)
    # If both exist, add delta row ourselves (sometimes compare file exists too)
    if base_scr and praat_scr:
        dfb0 = safe_read_csv(base_scr)
        dfp0 = safe_read_csv(praat_scr)
        delta = maybe_add_delta_row(dfb0, dfp0, key_cols=["alpha", "variant"])
        if not delta.empty:
            delta.insert(0, "source_file", "computed_delta")
            scr_out.append(delta)

    if compare_scr:
        dfc = safe_read_csv(compare_scr)
        dfc.insert(0, "source_file", compare_scr.name)
        # Keep it as extra evidence if it contains useful stuff
        scr_out.append(dfc)

    if scr_out:
        scr = pd.concat(scr_out, ignore_index=True)

        preferred_cols = [
            "source_file", "variant", "alpha",
            "auc_mean", "auc_std",
            "coverage_mean", "coverage_std",
            "decided_rate_mean", "decided_rate_std",
            "unsure_rate_mean", "unsure_rate_std", "unsure_rate",
            "acc_decided_mean", "acc_decided_std",
        ]
        cols = [c for c in preferred_cols if c in scr.columns]
        if cols:
            scr = scr[cols]
        save_csv(scr, OUT / "REPORT_table_screening.csv")
        produced.append(("REPORT_table_screening.csv", "screening summaries"))
    else:
        print("[WARN] No screening summary files found.")

    # ---- 4) REPORT_table_class_coverage.csv ----
    cc_frames = []
    for fn in PREFERRED_FILES["class_coverage"]:
        p = find_file(fn)
        if not p:
            continue
        df = safe_read_csv(p)
        df.insert(0, "source_file", p.name)
        cc_frames.append(df)

    if cc_frames:
        cc = pd.concat(cc_frames, ignore_index=True)
        # Try keep typical columns
        preferred_cols = [
            "source_file", "variant", "alpha", "label", "class", "y", "coverage",
            "coverage_mean", "coverage_std", "n"
        ]
        cols = [c for c in preferred_cols if c in cc.columns]
        if cols:
            cc = cc[cols]
        save_csv(cc, OUT / "REPORT_table_class_coverage.csv")
        produced.append(("REPORT_table_class_coverage.csv", "multiple"))
    else:
        print("[WARN] No conformal class coverage files found.")

    # ---- 5) REPORT_table_robustness.csv ----
    rb_frames = []
    for fn in PREFERRED_FILES["robustness"]:
        p = find_file(fn)
        if not p:
            continue
        df = safe_read_csv(p)
        df.insert(0, "source_file", p.name)
        rb_frames.append(df)

    if rb_frames:
        rb = pd.concat(rb_frames, ignore_index=True)
        save_csv(rb, OUT / "REPORT_table_robustness.csv")
        produced.append(("REPORT_table_robustness.csv", "multiple"))
    else:
        print("[WARN] No robustness files found.")

    # ---- Copy calibration reliability plot + calibration tables (if present) ----
    for fn in PREFERRED_FILES["calibration"]:
        p = find_file(fn)
        if not p:
            continue
        dest = OUT / p.name
        try:
            shutil.copy2(p, dest)
            produced.append((p.name, p.name))
        except Exception as e:
            print(f"[WARN] Could not copy {p} -> {dest}: {e}")

    # ---- Write a short README to explain what’s inside ----
    readme = OUT / "README_REPORT_READY.txt"
    lines = [
        "REPORT_READY folder contents",
        "============================",
        "",
        "This folder is generated by scripts/100_build_report_ready_tables.py",
        "It collects report-friendly CSVs and copies key calibration plot(s).",
        "",
        "Core tables:",
        "- REPORT_table_dataset.csv        : dataset counts/statistics (from dataset_stats_vowels_pack.csv)",
        "- REPORT_table_auc.csv            : AUC evaluation tables (repeated speaker splits)",
        "- REPORT_table_screening.csv      : conformal screening summaries (+ optional delta row)",
        "- REPORT_table_class_coverage.csv : class-wise coverage tables (conformal)",
        "- REPORT_table_robustness.csv     : robustness outputs (noise/volume/waveform etc.)",
        "",
        "Also copied (if available):",
        "- calibration_*_reliability.png and calibration CSVs",
        "",
        "Provenance (what source files were used):",
        "",
    ]
    for out_name, src in produced:
        lines.append(f"- {out_name}   <=   {src}")
    readme.write_text("\n".join(lines), encoding="utf-8")

    print("=== DONE ===")
    print(f"Output folder: {OUT.resolve()}")
    print("Produced:")
    for out_name, src in produced:
        print(f" - {out_name} (from {src})")


if __name__ == "__main__":
    main()
