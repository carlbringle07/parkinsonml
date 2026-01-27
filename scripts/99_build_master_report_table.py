# scripts/99_build_master_report_table.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


def _read_csv_safely(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop accidental unnamed index columns
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def _tag_from_filename(name: str) -> str:
    n = name.lower()
    if "dataset_stats" in n:
        return "dataset_stats"
    if "calibration" in n and "bins" in n:
        return "calibration_bins"
    if "calibration" in n and "metrics" in n:
        return "calibration_metrics"
    if "class_coverage" in n and "compare" in n:
        return "compare_class_coverage"
    if "conformal_class_coverage" in n:
        return "conformal_class_coverage"
    if "per_split_clean_summary_from_robustness" in n:
        return "robustness_per_split_clean_summary"
    if "robustness_conformal_screening" in n and "per_split" in n:
        return "robustness_conformal_per_split"
    if "robustness_conformal_screening" in n and "summary" in n:
        return "robustness_conformal_summary"
    if "robustness_summary" in n:
        return "robustness_summary"
    if "robustness_waveform" in n:
        return "robustness_waveform"
    if "compare_screening" in n:
        return "compare_screening"
    if "screening_summary_conformal" in n:
        return "screening_summary_conformal"
    if "eval_repeated_splits" in n:
        return "eval_repeated_splits"
    if n.startswith("features_"):
        return "features_table"
    if n.startswith("praat_"):
        return "praat_features_table"
    return "misc"


def build_master(report_dir: Path) -> pd.DataFrame:
    # Explicit allowlist (stable + relevant for report)
    want = [
        # Core “headline” outputs
        "dataset_stats_vowels_pack.csv",
        "eval_repeated_splits_vowels_pack_baseline_vs_praat.csv",
        "screening_summary_conformal_vowels_pack_alpha0.10.csv",
        "screening_summary_conformal_vowels_pack_plus_praat_alpha0.10.csv",
        "compare_screening_vowels_pack_baseline_vs_praat_alpha0.10.csv",
        "robustness_summary_vowels_pack_baseline_vs_praat.csv",
        "conformal_alpha_sweep_baseline_va.csv",  # already in artifacts

        # Robustness conformal artifacts
        "robustness_conformal_screening_vowels_pack_baseline_vs_praat_alpha0.10_per_split.csv",
        "robustness_conformal_screening_vowels_pack_baseline_vs_praat_alpha0.10_summary.csv",
        "robustness_delta_vs_clean_table_vowels_pack_baseline_vs_praat.csv",

        # Calibration artifacts
        "calibration_vowels_pack_baseline_vs_praat_bins.csv",
        "calibration_vowels_pack_baseline_vs_praat_metrics.csv",

        # Class-coverage artifacts
        "conformal_class_coverage_vowels_pack_alpha0.10.csv",
        "conformal_class_coverage_vowels_pack_plus_praat_alpha0.10.csv",
        "compare_conformal_class_coverage_vowels_pack_baseline_vs_praat_alpha0.10.csv",

        # Helpful “compressed” robustness summary you created
        "per_split_clean_summary_from_robustness.csv",

        # (Optional but often useful to reference in method/appendix)
        "features_vowels_pack_file.csv",
        "features_vowels_pack_person.csv",
        "features_vowels_pack_person_plus_praat.csv",
        "praat_vowels_pack_file.csv",
        "praat_vowels_pack_person.csv",
    ]

    # Only keep files that actually exist
    existing = [report_dir / f for f in want if (report_dir / f).exists()]

    # If you want to be extra safe: also include any compare_* / screening_* / robustness_* CSVs
    # that might exist, without exploding the table.
    extra_patterns = [
        "compare_*vowels_pack*.csv",
        "screening_summary_conformal_*vowels_pack*.csv",
        "robustness_*vowels_pack*.csv",
        "conformal_class_coverage_*vowels_pack*.csv",
        "calibration_*vowels_pack*.csv",
    ]
    for pat in extra_patterns:
        for p in report_dir.glob(pat):
            if p not in existing:
                existing.append(p)

    # Deterministic order
    existing = sorted(existing, key=lambda p: p.name.lower())

    frames: list[pd.DataFrame] = []
    for p in existing:
        df = _read_csv_safely(p)

        # Add metadata columns (always first)
        df.insert(0, "artifact_tag", _tag_from_filename(p.name))
        df.insert(0, "source_file", p.name)

        # Try to normalize a couple of common keys if present (optional, safe)
        # (No hard requirements: just helps sorting/filtering later)
        for col in list(df.columns):
            cl = str(col).lower()
            if cl in ("variant", "model", "model_variant"):
                # keep as is, but ensure string type
                df[col] = df[col].astype(str)
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No CSV files found in {report_dir} matching allowlist/patterns.")

    master = pd.concat(frames, ignore_index=True, sort=False)
    return master


def main() -> None:
    ra = Path("report_artifacts")
    ra.mkdir(parents=True, exist_ok=True)

    master = build_master(ra)

    out_csv = ra / "master_summary_for_report.csv"
    master.to_csv(out_csv, index=False)

    # Small audit file for reproducibility
    used = sorted(set(master["source_file"].astype(str).tolist()))
    (ra / "master_summary_files_used.txt").write_text(
        "Files used:\n" + "\n".join(used) + "\n", encoding="utf-8"
    )

    print(f"Saved: {out_csv}")
    print(f"Rows: {len(master)}")
    print(f"Files used: {used}")


if __name__ == "__main__":
    main()
