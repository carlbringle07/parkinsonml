# FILE: scripts/12_extract_praat_features_va.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import parselmouth
from parselmouth.praat import call


ROOT = Path(__file__).resolve().parents[1]
DATA_INDEX = ROOT / "data_index"

FILE_INDEX_CSV = DATA_INDEX / "file_index.csv"
OUT_FILE_CSV = DATA_INDEX / "praat_va_file.csv"
OUT_PERSON_CSV = DATA_INDEX / "praat_va_person.csv"
BASELINE_PERSON_CSV = DATA_INDEX / "features_va_person.csv"
OUT_MERGED_PERSON_CSV = DATA_INDEX / "features_va_person_plus_praat.csv"


def _safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _finite(x) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except Exception:
        return False


def is_va_row(row: pd.Series) -> bool:
    tc = str(row.get("task_code", "")).strip().upper()
    fn = str(row.get("filename", "")).strip().upper()
    return tc.startswith("VA") or fn.startswith("VA")


def extract_praat_features_for_file(wav_path: Path) -> dict:
    if not wav_path.exists():
        return {
            "f0_mean_hz": np.nan,
            "f0_sd_hz": np.nan,
            "f0_min_hz": np.nan,
            "f0_max_hz": np.nan,
            "hnr_mean_db": np.nan,
            "jitter_local": np.nan,
            "shimmer_local": np.nan,
            "pitch_floor_hz_used": np.nan,
            "pitch_ceiling_hz_used": np.nan,
            "file_missing": 1,
        }

    try:
        sound = parselmouth.Sound(str(wav_path))
    except Exception:
        return {
            "f0_mean_hz": np.nan,
            "f0_sd_hz": np.nan,
            "f0_min_hz": np.nan,
            "f0_max_hz": np.nan,
            "hnr_mean_db": np.nan,
            "jitter_local": np.nan,
            "shimmer_local": np.nan,
            "pitch_floor_hz_used": np.nan,
            "pitch_ceiling_hz_used": np.nan,
            "file_missing": 0,
        }

    pitch_ranges = [
        (75, 500),
        (50, 300),
        (100, 600),
    ]

    best = None
    best_score = -1

    for floor, ceiling in pitch_ranges:
        f0_mean = f0_sd = f0_min = f0_max = np.nan
        hnr_mean = np.nan
        jitter_local = np.nan
        shimmer_local = np.nan

        # Pitch (robust: use "To Pitch" not "To Pitch (ac)")
        try:
            pitch = call(sound, "To Pitch", 0.0, floor, ceiling)
            f0_mean = _safe_float(call(pitch, "Get mean", 0, 0, "Hertz"))
            f0_sd = _safe_float(call(pitch, "Get standard deviation", 0, 0, "Hertz"))
            f0_min = _safe_float(call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic"))
            f0_max = _safe_float(call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic"))
        except Exception:
            pass

        # HNR
        try:
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, floor, 0.1, 1.0)
            hnr_mean = _safe_float(call(harmonicity, "Get mean", 0, 0))
        except Exception:
            pass

        # Jitter / Shimmer
        try:
            point_process = call(sound, "To PointProcess (periodic, cc)", floor, ceiling)
            n_points = int(call(point_process, "Get number of points"))
            if n_points >= 10:
                jitter_local = _safe_float(call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
                shimmer_local = _safe_float(
                    call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                )
        except Exception:
            pass

        candidate = {
            "f0_mean_hz": f0_mean,
            "f0_sd_hz": f0_sd,
            "f0_min_hz": f0_min,
            "f0_max_hz": f0_max,
            "hnr_mean_db": hnr_mean,
            "jitter_local": jitter_local,
            "shimmer_local": shimmer_local,
            "pitch_floor_hz_used": float(floor),
            "pitch_ceiling_hz_used": float(ceiling),
            "file_missing": 0,
        }

        score = int(_finite(f0_mean)) + int(_finite(hnr_mean)) + int(_finite(jitter_local)) + int(_finite(shimmer_local))
        if score > best_score:
            best_score = score
            best = candidate

        # if we already got jitter or shimmer, good enough
        if _finite(jitter_local) or _finite(shimmer_local):
            return candidate

    return best if best is not None else {
        "f0_mean_hz": np.nan,
        "f0_sd_hz": np.nan,
        "f0_min_hz": np.nan,
        "f0_max_hz": np.nan,
        "hnr_mean_db": np.nan,
        "jitter_local": np.nan,
        "shimmer_local": np.nan,
        "pitch_floor_hz_used": np.nan,
        "pitch_ceiling_hz_used": np.nan,
        "file_missing": 0,
    }


def main():
    if not FILE_INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing: {FILE_INDEX_CSV}")

    df = pd.read_csv(FILE_INDEX_CSV)
    print("Loaded file_index.csv rows:", len(df))

    if "speaker" not in df.columns:
        raise RuntimeError("file_index.csv missing required column: speaker")

    df["speaker"] = df["speaker"].astype(str)

    allowed_speakers = None
    if BASELINE_PERSON_CSV.exists():
        base = pd.read_csv(BASELINE_PERSON_CSV)
        allowed_speakers = set(base["speaker"].astype(str))
        print("Baseline speakers loaded:", len(allowed_speakers))

    df_va = df[df.apply(is_va_row, axis=1)].copy()
    print("Rows matching VA via task_code/filename:", len(df_va))

    if allowed_speakers is not None:
        df_va = df_va[df_va["speaker"].isin(allowed_speakers)].copy()
        print("Rows after restricting to baseline speakers:", len(df_va))

    if df_va.empty:
        raise RuntimeError("No VA files found after filtering.")

    rows = []
    for _, r in tqdm(df_va.iterrows(), total=len(df_va), desc="Praat VA"):
        path = Path(r["path"])
        feats = extract_praat_features_for_file(path)
        out = {
            "path": str(path),
            "filename": r.get("filename", path.name),
            "speaker": r["speaker"],
            "group": r.get("group", ""),
            "label": r.get("label", np.nan),
            "task_code": r.get("task_code", ""),
        }
        out.update(feats)
        rows.append(out)

    df_out = pd.DataFrame(rows)

    df_out.to_csv(OUT_FILE_CSV, index=False)

    feature_cols = [
        "f0_mean_hz",
        "f0_sd_hz",
        "f0_min_hz",
        "f0_max_hz",
        "hnr_mean_db",
        "jitter_local",
        "shimmer_local",
    ]
    group_cols = [c for c in ["speaker", "group", "label"] if c in df_out.columns]
    agg = df_out.groupby(group_cols, as_index=False)[feature_cols].mean(numeric_only=True)
    agg.to_csv(OUT_PERSON_CSV, index=False)

    merged_ok = False
    if BASELINE_PERSON_CSV.exists():
        base = pd.read_csv(BASELINE_PERSON_CSV)
        keys = ["speaker"]
        for k in ["group", "label"]:
            if k in base.columns and k in agg.columns:
                keys.append(k)
        merged = base.merge(agg, on=keys, how="left")
        merged.to_csv(OUT_MERGED_PERSON_CSV, index=False)
        merged_ok = True

    non_na = df_out[feature_cols].notna().sum().to_dict()
    missing_files = int(df_out.get("file_missing", pd.Series([0]*len(df_out))).sum())

    print("\n=== DONE ===")
    print(f"VA files processed: {len(df_out)}")
    print(f"Missing files on disk: {missing_files}")
    print("Non-NaN counts (file-level):")
    for k, v in non_na.items():
        print(f"  {k}: {int(v)}")

    print(f"\nSaved file-level: {OUT_FILE_CSV}")
    print(f"Saved person-level: {OUT_PERSON_CSV}")
    if merged_ok:
        print(f"Merged: {OUT_MERGED_PERSON_CSV}")


if __name__ == "__main__":
    main()
