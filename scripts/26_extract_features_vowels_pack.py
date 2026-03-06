# FILE: scripts/26_extract_features_vowels_pack.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import librosa
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA_INDEX = ROOT / "data_index"

FILE_INDEX = DATA_INDEX / "file_index.csv"

OUT_FILE = DATA_INDEX / "features_vowels_pack_file.csv"
OUT_PERSON = DATA_INDEX / "features_vowels_pack_person.csv"

# PD vs control_elderly
KEEP_LABELS = {"pd", "control_elderly"}

# “Vowel-pack” (fokus vokaler)
TASKS = {"VA1", "VA2", "VE1", "VE2", "VI1", "VI2", "VO1", "VO2", "VU1", "VU2"}

SR = 16000


def base_task(code: str) -> str:
    # Ex: "VA2G" -> "VA2"
    s = str(code).strip().upper()
    # matchar bokstäver + siffror i början (VA2)
    out = ""
    for ch in s:
        if ch.isalnum():
            out += ch
        else:
            break
    # ta bara prefix av formen LETTERS+DIGITS, enklast:
   
    seen_digit = False
    base = ""
    for ch in out:
        if ch.isdigit():
            seen_digit = True
            base += ch
        else:
            if seen_digit:
            
                break
            base += ch
    return base


def extract_features(path: str) -> dict:
    y, sr = librosa.load(path, sr=SR, mono=True)
    if y is None or len(y) < int(0.2 * sr):
        return {}

    # trimma tystnad lite
    y, _ = librosa.effects.trim(y, top_db=30)

    if len(y) < int(0.2 * sr):
        return {}

    duration_s = len(y) / sr

    # frame features
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    feats = {
        "duration_s": float(duration_s),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "rolloff_mean": float(np.mean(rolloff)),
        "rolloff_std": float(np.std(rolloff)),
    }

    for i in range(mfcc.shape[0]):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))

    return feats


def main():
    if not FILE_INDEX.exists():
        raise FileNotFoundError(f"Missing: {FILE_INDEX} (run scripts/00_build_index.py first)")

    df = pd.read_csv(FILE_INDEX)
    for c in ["path", "label", "group", "speaker", "task_code"]:
        if c not in df.columns:
            raise RuntimeError(f"file_index.csv missing column: {c}")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["speaker"] = df["speaker"].astype(str)
    df["task_code"] = df["task_code"].astype(str)

    df = df[df["label"].isin(KEEP_LABELS)].copy()
    df["base_task"] = df["task_code"].apply(base_task)

    df = df[df["base_task"].isin(TASKS)].copy()

    if len(df) == 0:
        raise RuntimeError("No files matched vowel-pack tasks. Check task codes in file_index.csv.")

    print(f"Files matched: {len(df)}")
    print(f"Speakers: {df['speaker'].nunique()}  PD: {(df['label']=='pd').sum()}  CTRL: {(df['label']=='control_elderly').sum()}")
    print("Tasks found:", sorted(df["base_task"].unique().tolist()))

    rows = []
    failed = 0

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Extract vowel-pack baseline"):
        p = str(r["path"])
        feats = extract_features(p)
        if not feats:
            failed += 1
            continue

        out = {
            "path": p,
            "speaker": r["speaker"],
            "label": r["label"],
            "group": r["group"],
            "task_code": r["task_code"],
            "base_task": r["base_task"],
        }
        out.update(feats)
        rows.append(out)

    fdf = pd.DataFrame(rows)

    if len(fdf) == 0:
        raise RuntimeError("All feature extraction failed. Check audio paths / librosa install.")

    # person-level aggregation (mean across files for each speaker)
    num_cols = [c for c in fdf.columns if pd.api.types.is_numeric_dtype(fdf[c])]
    meta_cols = ["speaker", "label", "group"]

    pdf = (
        fdf.groupby("speaker", as_index=False)
        .agg({**{c: "mean" for c in num_cols}, "label": "first", "group": "first"})
    )

    fdf.to_csv(OUT_FILE, index=False)
    pdf.to_csv(OUT_PERSON, index=False)

    print("\n=== DONE ===")
    print(f"Files processed (with features): {len(fdf)}")
    print(f"Failed loads: {failed}")
    print("Saved file-level:", OUT_FILE)
    print("Saved person-level:", OUT_PERSON)
    print("Persons:", pdf["speaker"].nunique(), " (pd:", int((pdf["label"]=="pd").sum()), ", ctrl:", int((pdf["label"]=="control_elderly").sum()), ")")
    print("Feature columns (person):", len([c for c in pdf.columns if c not in meta_cols]))


if __name__ == "__main__":
    main()
