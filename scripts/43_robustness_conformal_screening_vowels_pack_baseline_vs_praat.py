"""
Step 43 (rebuild): Robustness + conformal screening for vowels_pack (baseline vs baseline+Praat).

Goals
-----
1) Reproducible speaker-wise repeated splits.
2) Robust handling of noisy conditions (clean, snr20, snr10).
3) Reliable outputs for both model variants:
   - baseline
   - baseline_plus_praat
4) Explicit uncertainty accounting (decided vs unsure) + confusion matrices.

This script intentionally reads the canonical file-level table created in step 26:
  data_index/features_vowels_pack_file.csv
which already contains a filtered vowels_pack cohort and avoids fragile filtering from file_index.csv.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import parselmouth
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# Config
# =========================
ROOT = Path(__file__).resolve().parents[1]
DATA_INDEX = ROOT / "data_index"

INPUT_FILE_LIST = DATA_INDEX / "features_vowels_pack_file.csv"
CACHE_DIR = DATA_INDEX / "robustness_cache_vowels_pack_43"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.10
N_SPLITS = 50
TEST_SIZE = 0.25
CALIB_FRAC_OF_TRAIN = 0.25
RANDOM_STATE = 123
SR = 16000

LABEL_MAP = {"control_elderly": 0, "pd": 1}
INV_LABEL_MAP = {0: "control", 1: "pd"}

CONDITIONS: list[dict[str, Any]] = [
    {"name": "clean", "type": "clean"},
    {"name": "snr20", "type": "snr", "snr_db": 20.0},
    {"name": "snr10", "type": "snr", "snr_db": 10.0},
]

OUT_PER_SPLIT = DATA_INDEX / f"robustness_conformal_screening_vowels_pack_baseline_vs_praat_alpha{ALPHA:.2f}_per_split.csv"
OUT_SUMMARY = DATA_INDEX / f"robustness_conformal_screening_vowels_pack_baseline_vs_praat_alpha{ALPHA:.2f}_summary.csv"
OUT_CONFUSION = DATA_INDEX / f"robustness_conformal_screening_vowels_pack_baseline_vs_praat_alpha{ALPHA:.2f}_confusion_counts.csv"
OUT_RUNINFO = DATA_INDEX / f"robustness_conformal_screening_vowels_pack_baseline_vs_praat_alpha{ALPHA:.2f}_runinfo.json"


# =========================
# Utilities
# =========================
def stable_seed(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def apply_condition(y: np.ndarray, condition: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    y = y.astype(np.float32, copy=False)

    if condition["type"] == "clean":
        return y

    if condition["type"] == "snr":
        snr_db = float(condition["snr_db"])
        sig_power = float(np.mean(y * y) + 1e-12)
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(0.0, math.sqrt(noise_power), size=y.shape).astype(np.float32)
        return np.clip(y + noise, -1.0, 1.0)

    raise ValueError(f"Unknown condition type: {condition['type']}")


def baseline_features(y: np.ndarray, sr: int) -> dict[str, float]:
    if y.size < sr // 20:
        y = np.pad(y, (0, sr // 20 - y.size), mode="constant")

    feats: dict[str, float] = {}

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    feats["duration_s"] = float(len(y) / sr)
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"] = float(np.std(rms))
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"] = float(np.std(zcr))
    feats["centroid_mean"] = float(np.mean(centroid))
    feats["centroid_std"] = float(np.std(centroid))
    feats["rolloff_mean"] = float(np.mean(rolloff))
    feats["rolloff_std"] = float(np.std(rolloff))

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    for i in range(13):
        feats[f"mfcc{i+1}_mean"] = float(mfcc_mean[i])
        feats[f"mfcc{i+1}_std"] = float(mfcc_std[i])

    for k, v in list(feats.items()):
        if not np.isfinite(v):
            feats[k] = np.nan

    return feats


def praat_features(y: np.ndarray, sr: int) -> dict[str, float]:
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)

    pitch = snd.to_pitch(pitch_floor=60.0, pitch_ceiling=400.0)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[np.isfinite(f0)]
    f0 = f0[f0 > 0]

    f0_mean = float(np.mean(f0)) if f0.size else np.nan
    f0_std = float(np.std(f0)) if f0.size else np.nan
    f0_min = float(np.min(f0)) if f0.size else np.nan
    f0_max = float(np.max(f0)) if f0.size else np.nan

    harmonicity = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=60.0)
    hnr_vals = harmonicity.values
    hnr_vals = hnr_vals[np.isfinite(hnr_vals)]
    hnr_mean = float(np.mean(hnr_vals)) if hnr_vals.size else np.nan

    jitter_local = np.nan
    shimmer_local = np.nan
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 60.0, 400.0)
    try:
        jitter_local = float(parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 60.0, 400.0, 1.3, 1.6))
    except Exception:
        jitter_local = np.nan
    try:
        shimmer_local = float(parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 60.0, 400.0, 1.3, 1.6, 0.03, 0.45))
    except Exception:
        shimmer_local = np.nan

    out = {
        "praat_f0_mean_hz": f0_mean,
        "praat_f0_std_hz": f0_std,
        "praat_f0_min_hz": f0_min,
        "praat_f0_max_hz": f0_max,
        "praat_hnr_mean_db": hnr_mean,
        "praat_jitter_local": jitter_local,
        "praat_shimmer_local": shimmer_local,
    }

    return out


def load_file_list() -> pd.DataFrame:
    if not INPUT_FILE_LIST.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_FILE_LIST}")

    df = pd.read_csv(INPUT_FILE_LIST)
    required = {"path", "speaker", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{INPUT_FILE_LIST.name} missing required columns: {sorted(missing)}")

    df = df[df["label"].isin(LABEL_MAP.keys())].copy()
    df["path"] = df["path"].astype(str)
    df["speaker"] = df["speaker"].astype(str)

    if df.empty:
        raise ValueError("No rows left after label filter (expected pd/control_elderly).")

    return df


def build_person_tables_for_condition(condition: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    name = condition["name"]
    cache_base = CACHE_DIR / f"{name}_person_baseline.csv"
    cache_plus = CACHE_DIR / f"{name}_person_plus_praat.csv"
    cache_qc = CACHE_DIR / f"{name}_qc.json"

    if cache_base.exists() and cache_plus.exists() and cache_qc.exists():
        return pd.read_csv(cache_base), pd.read_csv(cache_plus), json.loads(cache_qc.read_text(encoding="utf-8"))

    files = load_file_list()
    rows: list[dict[str, Any]] = []
    n_read_fail = 0

    for r in files.itertuples(index=False):
        path = str(r.path)
        speaker = str(r.speaker)
        label = str(r.label)

        try:
            y, _ = librosa.load(path, sr=SR, mono=True)
        except Exception:
            n_read_fail += 1
            continue

        seed = stable_seed(f"{name}::{path}")
        rng = np.random.default_rng(seed)
        y2 = apply_condition(y, condition, rng)

        b = baseline_features(y2, SR)
        p = praat_features(y2, SR)

        row: dict[str, Any] = {"speaker": speaker, "label": label}
        row.update(b)
        row.update(p)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No feature rows extracted for condition={name}")

    file_df = pd.DataFrame(rows)
    feat_cols = [c for c in file_df.columns if c not in ["speaker", "label"]]
    person_df = file_df.groupby(["speaker", "label"], as_index=False)[feat_cols].mean(numeric_only=True)

    praat_cols = [c for c in person_df.columns if c.startswith("praat_")]
    base_cols = [c for c in person_df.columns if c not in ["speaker", "label"] + praat_cols]

    baseline_person = person_df[["speaker", "label"] + base_cols].copy()
    plus_person = person_df[["speaker", "label"] + base_cols + praat_cols].copy()

    def nan_stats(df_: pd.DataFrame, cols: list[str]) -> dict[str, float]:
        if not cols:
            return {"max_nan_rate": 1.0, "median_nan_rate": 1.0}
        nan_rates = df_[cols].isna().mean(axis=0)
        return {
            "max_nan_rate": float(nan_rates.max()),
            "median_nan_rate": float(nan_rates.median()),
        }

    qc = {
        "condition": name,
        "n_input_files": int(len(files)),
        "n_feature_rows": int(len(file_df)),
        "n_read_fail": int(n_read_fail),
        "n_speakers": int(len(person_df)),
        "n_pd": int((person_df["label"] == "pd").sum()),
        "n_control": int((person_df["label"] == "control_elderly").sum()),
        "baseline_nan": nan_stats(baseline_person, base_cols),
        "praat_nan": nan_stats(plus_person, praat_cols),
    }

    if qc["n_pd"] < 2 or qc["n_control"] < 2:
        raise RuntimeError(f"Too few speakers per class for condition={name}: {qc}")

    # Some Praat descriptors (especially jitter/shimmer) can fail on certain
    # recordings/conditions while core Praat descriptors (f0/HNR) remain valid.
    # Fail only if the core Praat block is almost entirely missing.
    core_praat_cols = [c for c in praat_cols if c in {"praat_f0_mean_hz", "praat_f0_std_hz", "praat_hnr_mean_db"}]
    core_nan = plus_person[core_praat_cols].isna().mean(axis=0) if core_praat_cols else pd.Series(dtype=float)
    core_max_nan = float(core_nan.max()) if not core_nan.empty else 1.0
    qc["praat_nan"]["core_max_nan_rate"] = core_max_nan

    if core_max_nan > 0.95:
        raise RuntimeError(
            f"Core Praat features are almost entirely missing for condition={name}. "
            f"Check audio quality/parselmouth setup. QC={qc['praat_nan']}"
        )

    baseline_person.to_csv(cache_base, index=False)
    plus_person.to_csv(cache_plus, index=False)
    cache_qc.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    return baseline_person, plus_person, qc


def conformal_sets_binary(proba_test: np.ndarray, proba_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> list[set[int]]:
    classes = np.array([0, 1], dtype=int)
    score_true = np.array([proba_cal[i, int(y_cal[i])] for i in range(len(y_cal))], dtype=float)
    n = len(score_true)

    out: list[set[int]] = []
    for i in range(proba_test.shape[0]):
        s: set[int] = set()
        for k in classes:
            pval = (np.sum(score_true <= proba_test[i, k]) + 1.0) / (n + 1.0)
            if pval > alpha:
                s.add(int(k))
        out.append(s)
    return out


def evaluate_condition_variant(df: pd.DataFrame, condition_name: str, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in df.columns if c not in ["speaker", "label"]]
    # Guard against columns that are entirely NaN (common for some Praat columns
    # in noisy conditions). Median imputation cannot recover an all-NaN column.
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]
    if not feature_cols:
        raise RuntimeError(f"No usable feature columns left for condition={condition_name}, variant={variant}")
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].map(LABEL_MAP).astype(int).to_numpy()

    sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    split_rows: list[dict[str, Any]] = []
    conf_rows: list[dict[str, Any]] = []

    for split_idx, (traincal_idx, test_idx) in enumerate(sss.split(X, y), start=1):
        X_traincal = X[traincal_idx]
        y_traincal = y[traincal_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        inner = StratifiedShuffleSplit(n_splits=1, test_size=CALIB_FRAC_OF_TRAIN, random_state=RANDOM_STATE + 10000 + split_idx)
        train_rel, cal_rel = next(inner.split(X_traincal, y_traincal))

        X_train = X_traincal[train_rel]
        y_train = y_traincal[train_rel]
        X_cal = X_traincal[cal_rel]
        y_cal = y_traincal[cal_rel]

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, solver="liblinear")),
            ]
        )
        model.fit(X_train, y_train)

        proba_cal = model.predict_proba(X_cal)
        proba_test = model.predict_proba(X_test)

        auc = float(roc_auc_score(y_test, proba_test[:, 1]))
        pred_sets = conformal_sets_binary(proba_test, proba_cal, y_cal, ALPHA)

        sizes = np.array([len(s) for s in pred_sets], dtype=int)
        decided_mask = sizes == 1
        unsure_mask = ~decided_mask

        covered = np.array([int(y_test[i] in pred_sets[i]) for i in range(len(y_test))], dtype=int)

        pred_decided = np.array([next(iter(pred_sets[i])) if decided_mask[i] else -1 for i in range(len(y_test))], dtype=int)

        acc_decided = np.nan
        if np.any(decided_mask):
            acc_decided = float(np.mean(pred_decided[decided_mask] == y_test[decided_mask]))

        split_rows.append(
            {
                "condition": condition_name,
                "variant": variant,
                "split": split_idx,
                "n_test": int(len(y_test)),
                "n_decided": int(np.sum(decided_mask)),
                "n_unsure": int(np.sum(unsure_mask)),
                "auc": auc,
                "coverage": float(np.mean(covered)),
                "decided_rate": float(np.mean(decided_mask)),
                "unsure_rate": float(np.mean(unsure_mask)),
                "acc_decided": acc_decided,
            }
        )

        # Confusion on all forced predictions (argmax) for comparability
        pred_all = np.argmax(proba_test, axis=1)
        for yt, yp in zip(y_test, pred_all):
            conf_rows.append(
                {
                    "condition": condition_name,
                    "variant": variant,
                    "split": split_idx,
                    "matrix": "all_forced",
                    "true_label": INV_LABEL_MAP[int(yt)],
                    "pred_label": INV_LABEL_MAP[int(yp)],
                    "count": 1,
                }
            )

        # Confusion on decided-only predictions
        for yt, yp in zip(y_test[decided_mask], pred_decided[decided_mask]):
            conf_rows.append(
                {
                    "condition": condition_name,
                    "variant": variant,
                    "split": split_idx,
                    "matrix": "decided_only",
                    "true_label": INV_LABEL_MAP[int(yt)],
                    "pred_label": INV_LABEL_MAP[int(yp)],
                    "count": 1,
                }
            )

        # Unsure accounting (kept separately)
        for yt in y_test[unsure_mask]:
            conf_rows.append(
                {
                    "condition": condition_name,
                    "variant": variant,
                    "split": split_idx,
                    "matrix": "unsure",
                    "true_label": INV_LABEL_MAP[int(yt)],
                    "pred_label": "unsure",
                    "count": 1,
                }
            )

    split_df = pd.DataFrame(split_rows)
    conf_df = pd.DataFrame(conf_rows)

    return split_df, conf_df


def summarize_splits(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["condition", "variant"]
    rows: list[dict[str, Any]] = []
    for (condition, variant), g in df.groupby(keys):
        rows.append(
            {
                "condition": condition,
                "variant": variant,
                "n_splits": int(len(g)),
                "n_test_total": int(g["n_test"].sum()),
                "n_decided_total": int(g["n_decided"].sum()),
                "n_unsure_total": int(g["n_unsure"].sum()),
                "auc_mean": float(g["auc"].mean()),
                "auc_std": float(g["auc"].std(ddof=1)),
                "coverage_mean": float(g["coverage"].mean()),
                "coverage_std": float(g["coverage"].std(ddof=1)),
                "decided_rate_mean": float(g["decided_rate"].mean()),
                "decided_rate_std": float(g["decided_rate"].std(ddof=1)),
                "unsure_rate_mean": float(g["unsure_rate"].mean()),
                "unsure_rate_std": float(g["unsure_rate"].std(ddof=1)),
                "acc_decided_mean": float(g["acc_decided"].mean()),
                "acc_decided_std": float(g["acc_decided"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    print("=== Step 43 rebuild: robustness + conformal screening (vowels_pack) ===")
    print(f"alpha={ALPHA}, n_splits={N_SPLITS}, test_size={TEST_SIZE}, calib_frac={CALIB_FRAC_OF_TRAIN}")

    split_parts: list[pd.DataFrame] = []
    conf_parts: list[pd.DataFrame] = []
    qc_parts: list[dict[str, Any]] = []

    for condition in CONDITIONS:
        cname = condition["name"]
        print(f"\n[Condition] {cname}")
        base_df, plus_df, qc = build_person_tables_for_condition(condition)
        qc_parts.append(qc)

        s_base, c_base = evaluate_condition_variant(base_df, cname, "baseline")
        s_plus, c_plus = evaluate_condition_variant(plus_df, cname, "baseline_plus_praat")

        split_parts.extend([s_base, s_plus])
        conf_parts.extend([c_base, c_plus])

    per_split = pd.concat(split_parts, ignore_index=True)
    summary = summarize_splits(per_split)

    conf_long = pd.concat(conf_parts, ignore_index=True)
    conf_agg = (
        conf_long.groupby(["condition", "variant", "matrix", "true_label", "pred_label"], as_index=False)["count"]
        .sum()
        .sort_values(["condition", "variant", "matrix", "true_label", "pred_label"])
    )

    per_split.to_csv(OUT_PER_SPLIT, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    conf_agg.to_csv(OUT_CONFUSION, index=False)

    runinfo = {
        "alpha": ALPHA,
        "n_splits": N_SPLITS,
        "test_size": TEST_SIZE,
        "calib_frac_of_train": CALIB_FRAC_OF_TRAIN,
        "random_state": RANDOM_STATE,
        "conditions": CONDITIONS,
        "input_file": str(INPUT_FILE_LIST),
        "cache_dir": str(CACHE_DIR),
        "qc": qc_parts,
        "outputs": {
            "per_split": str(OUT_PER_SPLIT),
            "summary": str(OUT_SUMMARY),
            "confusion_counts": str(OUT_CONFUSION),
        },
    }
    OUT_RUNINFO.write_text(json.dumps(runinfo, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f" - {OUT_PER_SPLIT}")
    print(f" - {OUT_SUMMARY}")
    print(f" - {OUT_CONFUSION}")
    print(f" - {OUT_RUNINFO}")


if __name__ == "__main__":
    main()
