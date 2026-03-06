# scripts/41_robustness_waveform_vowels_pack_baseline_vs_praat.py
"""
Robustness test (waveform perturbations) for vowels_pack:
- Builds per-speaker feature tables under several waveform conditions (clean/noise/volume)
- Compares baseline (librosa spectral/MFCC) vs baseline+Praat (parselmouth) using repeated splits
- Saves a single CSV with AUC stats per condition + variant.

IMPORTANT:
- Reads file list from data_index/features_vowels_pack_file.csv (already matched 495 files).
  This avoids fragile task filtering in file_index.csv.
- Caches per-condition per-speaker tables in data_index/robustness_cache_vowels_pack/
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import librosa

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---- Optional Praat/parselmouth ----
try:
    import parselmouth  # type: ignore
    PARSELMOUTH_OK = True
except Exception:
    PARSELMOUTH_OK = False


# =========================
# Config
# =========================
ROOT = Path(__file__).resolve().parents[1]  # project root (GA_ML)
DATA_INDEX = ROOT / "data_index"

INPUT_FILE_LIST = DATA_INDEX / "features_vowels_pack_file.csv"  # file-level list with paths, labels, speakers
CACHE_DIR = DATA_INDEX / "robustness_cache_vowels_pack"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = DATA_INDEX / "robustness_waveform_vowels_pack_baseline_vs_praat.csv"

SR = 16000
N_SPLITS = 50
TEST_SIZE = 0.30
RANDOM_STATE = 42

LABEL_MAP = {"pd": 1, "control_elderly": 0}

# Conditions to test
CONDITIONS = [
    {"name": "clean", "type": "clean"},
    {"name": "snr20", "type": "snr", "snr_db": 20.0},
    {"name": "snr10", "type": "snr", "snr_db": 10.0},
    {"name": "vol0p5", "type": "volume", "gain": 0.5},
    {"name": "vol2p0", "type": "volume", "gain": 2.0},
]


# =========================
# Audio perturbations
# =========================
def apply_volume(y: np.ndarray, gain: float) -> np.ndarray:
    y2 = y * float(gain)
    # clip to [-1, 1] to avoid absurd values
    return np.clip(y2, -1.0, 1.0)


def apply_snr_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise to reach target SNR (dB)."""
    y = y.astype(np.float32, copy=False)
    sig_power = float(np.mean(y * y) + 1e-12)
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_power = sig_power / snr_linear
    noise = rng.normal(0.0, math.sqrt(noise_power), size=y.shape).astype(np.float32)
    y2 = y + noise
    return np.clip(y2, -1.0, 1.0)


# =========================
# Feature extraction (baseline)
# =========================
def baseline_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Produces 35 baseline features:
    - duration
    - rms_mean/std, zcr_mean/std
    - centroid_mean/std, bandwidth_mean/std, rolloff_mean/std, flatness_mean/std
    - mfcc1..13 mean  (13)
    - mfcc1..10 std   (10)
    Total: 1 + 2+2 + 2+2+2+2 + 13 + 10 = 35
    """
    eps = 1e-12
    y = y.astype(np.float32, copy=False)

    # Avoid empty / ultra-short
    if y.size < sr // 20:  # <50ms
        y = np.pad(y, (0, sr // 20 - y.size), mode="constant")

    duration = float(len(y) / sr)

    # Frame-based measures
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    feats: Dict[str, float] = {}
    feats["duration_s"] = duration

    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"] = float(np.std(rms))
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"] = float(np.std(zcr))

    feats["centroid_mean"] = float(np.mean(centroid))
    feats["centroid_std"] = float(np.std(centroid))
    feats["bandwidth_mean"] = float(np.mean(bandwidth))
    feats["bandwidth_std"] = float(np.std(bandwidth))
    feats["rolloff_mean"] = float(np.mean(rolloff))
    feats["rolloff_std"] = float(np.std(rolloff))
    feats["flatness_mean"] = float(np.mean(flatness))
    feats["flatness_std"] = float(np.std(flatness))

    for i in range(13):
        feats[f"mfcc{i+1}_mean"] = float(mfcc_mean[i])
    for i in range(10):
        feats[f"mfcc{i+1}_std"] = float(mfcc_std[i])

    # Safety: replace any nan/inf with finite
    for k, v in list(feats.items()):
        if not np.isfinite(v):
            feats[k] = 0.0

    return feats


# =========================
# Feature extraction (Praat)
# =========================
def praat_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Returns a small Praat set.
    If parselmouth is unavailable, returns NaNs.

    We use Sound.to_pitch(pitch_floor=..., pitch_ceiling=...) WITHOUT time_step=0.0,
    because some parselmouth versions reject 0.0 and require None or positive step.
    """
    keys = [
        "f0_mean_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz",
        "hnr_mean_db",
        "jitter_local", "shimmer_local",
    ]

    if not PARSELMOUTH_OK:
        return {k: np.nan for k in keys}

    # parselmouth expects float64
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)

    # Pitch: safe call (no time_step=0.0)
    try:
        pitch = snd.to_pitch(pitch_floor=60.0, pitch_ceiling=400.0)
    except TypeError:
        # fallback if signature differs
        pitch = snd.to_pitch()

    f0 = pitch.selected_array["frequency"]
    f0 = f0[np.isfinite(f0)]
    f0 = f0[f0 > 0]

    if f0.size == 0:
        f0_mean = f0_std = f0_min = f0_max = np.nan
    else:
        f0_mean = float(np.mean(f0))
        f0_std = float(np.std(f0))
        f0_min = float(np.min(f0))
        f0_max = float(np.max(f0))

    # HNR
    try:
        harmonicity = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=60.0)
        hnr_vals = harmonicity.values
        hnr_vals = hnr_vals[np.isfinite(hnr_vals)]
        hnr_mean = float(np.mean(hnr_vals)) if hnr_vals.size else np.nan
    except Exception:
        hnr_mean = np.nan

    # Jitter/Shimmer using PointProcess (standard approach)
    jitter_local = np.nan
    shimmer_local = np.nan
    try:
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 60.0, 400.0)
        jitter_local = float(parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 60.0, 400.0, 1.3, 1.6))
        shimmer_local = float(parselmouth.praat.call(
            [snd, point_process], "Get shimmer (local)", 0, 0, 60.0, 400.0, 1.3, 1.6, 0.03, 0.45
        ))
    except Exception:
        # leave as NaN
        pass

    out = {
        "f0_mean_hz": f0_mean,
        "f0_std_hz": f0_std,
        "f0_min_hz": f0_min,
        "f0_max_hz": f0_max,
        "hnr_mean_db": hnr_mean,
        "jitter_local": jitter_local,
        "shimmer_local": shimmer_local,
    }

    return out


# =========================
# Loading file list
# =========================
def load_file_list() -> pd.DataFrame:
    if not INPUT_FILE_LIST.exists():
        raise FileNotFoundError(f"Hittar inte {INPUT_FILE_LIST}. Kör script 26 igen om den saknas.")
    df = pd.read_csv(INPUT_FILE_LIST)

    required = {"path", "speaker", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{INPUT_FILE_LIST.name} saknar kolumner: {sorted(missing)}")

    # Keep only expected labels
    df = df[df["label"].isin(LABEL_MAP.keys())].copy()
    df["path"] = df["path"].astype(str)

    if df.empty:
        raise ValueError("Inga rader kvar efter label-filter. Kontrollera label-kodning i features_vowels_pack_file.csv")

    print(f"Files in vowels_pack list: {len(df)}")
    return df


# =========================
# Build per-condition speaker tables
# =========================
def build_condition_tables(condition: Dict, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      baseline_person_df: speaker-level baseline features
      plus_praat_person_df: speaker-level baseline + praat features
    Uses caching on disk.
    """
    cname = condition["name"]
    cache_base = CACHE_DIR / f"{cname}_person_baseline.csv"
    cache_plus = CACHE_DIR / f"{cname}_person_plus_praat.csv"

    if cache_base.exists() and cache_plus.exists():
        base_df = pd.read_csv(cache_base)
        plus_df = pd.read_csv(cache_plus)
        return base_df, plus_df

    files = load_file_list()

    rows: List[Dict[str, object]] = []

    for i, r in enumerate(files.itertuples(index=False), start=1):
        path = getattr(r, "path")
        speaker = getattr(r, "speaker")
        label = getattr(r, "label")

        try:
            y, _ = librosa.load(path, sr=SR, mono=True)
        except Exception:
            # Skip unreadable file
            continue

        # Apply condition
        if condition["type"] == "clean":
            y2 = y
        elif condition["type"] == "volume":
            y2 = apply_volume(y, float(condition["gain"]))
        elif condition["type"] == "snr":
            y2 = apply_snr_noise(y, float(condition["snr_db"]), rng)
        else:
            raise ValueError(f"Unknown condition type: {condition['type']}")

        b = baseline_features(y2, SR)
        p = praat_features(y2, SR)

        row: Dict[str, object] = {"speaker": speaker, "label": label}
        row.update(b)
        row.update(p)
        rows.append(row)

        if i % 50 == 0 or i == len(files):
            print(f"  [{cname}] processed {i}/{len(files)} files")

    if not rows:
        raise RuntimeError(f"Inga features extraherades för condition={cname}")

    file_df = pd.DataFrame(rows)

    # Aggregate per speaker
    meta_cols = ["speaker", "label"]
    feat_cols = [c for c in file_df.columns if c not in meta_cols]

    # numeric mean per speaker
    agg = file_df.groupby(["speaker", "label"], as_index=False)[feat_cols].mean(numeric_only=True)

    # Build baseline-only and plus-praat tables
    praat_cols = [
        "f0_mean_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz",
        "hnr_mean_db", "jitter_local", "shimmer_local"
    ]
    base_feat_cols = [c for c in agg.columns if c not in meta_cols and c not in praat_cols]
    plus_feat_cols = [c for c in agg.columns if c not in meta_cols]  # includes praat

    baseline_person = agg[meta_cols + base_feat_cols].copy()
    plus_person = agg[meta_cols + plus_feat_cols].copy()

    baseline_person.to_csv(cache_base, index=False)
    plus_person.to_csv(cache_plus, index=False)

    return baseline_person, plus_person


# =========================
# Evaluation (repeated splits)
# =========================
def repeated_auc(person_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[float, float, float, float, int]:
    """
    Returns mean, std, min, max, n_used
    """
    y = person_df["label"].map(LABEL_MAP).astype(int).to_numpy()
    X = person_df[feature_cols].to_numpy(dtype=float)

    # Replace inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    splitter = StratifiedShuffleSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, solver="liblinear")),
    ])

    aucs: List[float] = []
    for train_idx, test_idx in splitter.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, proba)
        aucs.append(float(auc))

    arr = np.array(aucs, dtype=float)
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max()), len(aucs)


# =========================
# Main
# =========================
def main() -> None:
    print("=== Building/Loading robustness cache (person tables per condition) ===")
    rng = np.random.default_rng(RANDOM_STATE)

    results: List[Dict[str, object]] = []

    for cond in CONDITIONS:
        cname = cond["name"]
        print(f"\n--- Building cache for condition={cname} ---")
        base_df, plus_df = build_condition_tables(cond, rng)

        # Identify feature columns
        base_feat_cols = [c for c in base_df.columns if c not in ["speaker", "label"]]
        plus_feat_cols = [c for c in plus_df.columns if c not in ["speaker", "label"]]

        # Baseline
        m, s, mn, mx, n_used = repeated_auc(base_df, base_feat_cols)
        results.append({
            "variant": "baseline",
            "condition": cname,
            "n_splits": n_used,
            "test_size": TEST_SIZE,
            "n_speakers": int(base_df.shape[0]),
            "n_pd": int((base_df["label"] == "pd").sum()),
            "n_ctrl": int((base_df["label"] == "control_elderly").sum()),
            "n_features": int(len(base_feat_cols)),
            "auc_mean": m,
            "auc_std": s,
            "auc_min": mn,
            "auc_max": mx,
        })

        # Baseline + Praat
        m, s, mn, mx, n_used = repeated_auc(plus_df, plus_feat_cols)
        results.append({
            "variant": "baseline_plus_praat",
            "condition": cname,
            "n_splits": n_used,
            "test_size": TEST_SIZE,
            "n_speakers": int(plus_df.shape[0]),
            "n_pd": int((plus_df["label"] == "pd").sum()),
            "n_ctrl": int((plus_df["label"] == "control_elderly").sum()),
            "n_features": int(len(plus_feat_cols)),
            "auc_mean": m,
            "auc_std": s,
            "auc_min": mn,
            "auc_max": mx,
        })

    out = pd.DataFrame(results)

    # Add delta vs clean for each variant
    out["auc_mean_delta_vs_clean"] = np.nan
    for v in out["variant"].unique():
        clean_val = out[(out["variant"] == v) & (out["condition"] == "clean")]["auc_mean"]
        if len(clean_val) == 1:
            clean_auc = float(clean_val.iloc[0])
            out.loc[out["variant"] == v, "auc_mean_delta_vs_clean"] = out.loc[out["variant"] == v, "auc_mean"] - clean_auc

    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV.name}")
    print(out)

    print("\nDONE.")


if __name__ == "__main__":
    main()
