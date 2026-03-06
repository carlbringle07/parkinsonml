from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa


ROOT = Path(__file__).resolve().parents[1]
DATA_INDEX = ROOT / "data_index"

FILE_INDEX_CSV = DATA_INDEX / "file_index.csv"
OUT_FILE_CSV = DATA_INDEX / "features_va_file.csv"
OUT_PERSON_CSV = DATA_INDEX / "features_va_person.csv"


def _safe_mean_std(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (np.nan, np.nan)
    return (float(np.nanmean(x)), float(np.nanstd(x)))


def _label_from_row(row: pd.Series) -> str:
    # Prefer 'label' if it's already in expected format
    lab = str(row.get("label", "")).strip().lower()
    if lab in {"pd", "control_elderly", "control_young", "control"}:
        return lab

    # Fallback: infer from group text
    grp = str(row.get("group", "")).strip().lower()
    if "parkinson" in grp or grp == "pd":
        return "pd"
    if "elderly" in grp:
        return "control_elderly"
    if "young" in grp:
        return "control_young"

    # Last fallback: keep original
    return lab if lab else grp


def _is_va_row(row: pd.Series) -> bool:
    tc = str(row.get("task_code", "")).strip().upper()
    fn = str(row.get("filename", "")).strip().upper()
    return tc.startswith("VA") or fn.startswith("VA")


def extract_baseline_features(y: np.ndarray, sr: int) -> dict:
    # --- Duration
    duration_s = float(len(y) / sr) if sr > 0 else np.nan

    # --- Frame-based features
    # Use default hop_length (512) for consistency
    rms = librosa.feature.rms(y=y)[0]  # shape (frames,)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]

    rms_mean, rms_std = _safe_mean_std(rms)
    zcr_mean, zcr_std = _safe_mean_std(zcr)
    centroid_mean, centroid_std = _safe_mean_std(centroid)
    rolloff_mean, rolloff_std = _safe_mean_std(rolloff)

    # --- MFCCs (13) -> mean/std each
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # (13, frames)
    feats = {
        "duration_s": duration_s,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "centroid_mean": centroid_mean,
        "centroid_std": centroid_std,
        "rolloff_mean": rolloff_mean,
        "rolloff_std": rolloff_std,
    }

    for i in range(13):
        m, s = _safe_mean_std(mfcc[i, :])
        feats[f"mfcc{i+1}_mean"] = m
        feats[f"mfcc{i+1}_std"] = s

    return feats


def main():
    if not FILE_INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing: {FILE_INDEX_CSV} (run scripts/00_build_index.py first)")

    df = pd.read_csv(FILE_INDEX_CSV)

    # Filter to VA
    df = df[df.apply(_is_va_row, axis=1)].copy()

    # Only PD vs elderly control for this stage
    df["label_norm"] = df.apply(_label_from_row, axis=1)
    df = df[df["label_norm"].isin(["pd", "control_elderly"])].copy()

    if df.empty:
        raise RuntimeError("No VA rows found for labels pd/control_elderly after filtering.")

    # Ensure columns exist
    if "path" not in df.columns or "speaker" not in df.columns:
        raise RuntimeError("file_index.csv must contain at least columns: path, speaker")

    df["speaker"] = df["speaker"].astype(str)

    rows = []
    failed = 0

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Extract VA baseline"):
        wav_path = Path(r["path"])
        try:
            y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
            if y is None or len(y) < 10:
                raise RuntimeError("Too short/empty audio.")
            feats = extract_baseline_features(y, sr)
        except Exception:
            failed += 1
            feats = {k: np.nan for k in (
                ["duration_s", "rms_mean", "rms_std", "zcr_mean", "zcr_std",
                 "centroid_mean", "centroid_std", "rolloff_mean", "rolloff_std"]
                + [f"mfcc{i}_mean" for i in range(1, 14)]
                + [f"mfcc{i}_std" for i in range(1, 14)]
            )}

        out = {
            "path": str(wav_path),
            "filename": r.get("filename", wav_path.name),
            "speaker": r["speaker"],
            "group": r.get("group", ""),
            "label": r["label_norm"],
            "task_code": r.get("task_code", ""),
        }
        out.update(feats)
        rows.append(out)

    df_file = pd.DataFrame(rows)

    # Save file-level
    df_file.to_csv(OUT_FILE_CSV, index=False)

    # Aggregate to person-level (mean over files per speaker)
    feat_cols = [c for c in df_file.columns if c not in ["path", "filename", "speaker", "group", "label", "task_code"]]
    df_person = (
        df_file.groupby(["speaker", "label"], as_index=False)[feat_cols]
        .mean(numeric_only=True)
    )

    # Add group label 
    grp_map = df_file.groupby("speaker")["group"].first().reset_index()
    df_person = df_person.merge(grp_map, on="speaker", how="left")

    df_person.to_csv(OUT_PERSON_CSV, index=False)

    print("\n=== DONE ===")
    print("VA files processed:", len(df_file))
    print("Failed loads:", failed)
    print("Saved file-level:", OUT_FILE_CSV)
    print("Saved person-level:", OUT_PERSON_CSV)
    print("Persons:", len(df_person), " (pd:", int((df_person['label']=='pd').sum()), ", control_elderly:", int((df_person['label']=='control_elderly').sum()), ")")

    # Quick NaN check
    non_na = df_file[feat_cols].notna().sum().sort_values(ascending=True).head(10)
    print("\nLowest non-NaN counts (file-level, bottom 10):")
    print(non_na.to_string())


if __name__ == "__main__":
    main()
