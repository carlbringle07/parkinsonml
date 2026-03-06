# FILE: scripts/15_robustness_waveform_va.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[1]
DATA_INDEX = ROOT / "data_index"

FILE_INDEX_CSV = DATA_INDEX / "file_index.csv"
OUT_CSV = DATA_INDEX / "robustness_waveform_va.csv"


def _safe_mean_std(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (np.nan, np.nan)
    return (float(np.nanmean(x)), float(np.nanstd(x)))


def extract_baseline_features(y: np.ndarray, sr: int) -> dict:
    duration_s = float(len(y) / sr) if sr > 0 else np.nan

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]

    rms_mean, rms_std = _safe_mean_std(rms)
    zcr_mean, zcr_std = _safe_mean_std(zcr)
    centroid_mean, centroid_std = _safe_mean_std(centroid)
    rolloff_mean, rolloff_std = _safe_mean_std(rolloff)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

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


def _is_va_row(row: pd.Series) -> bool:
    tc = str(row.get("task_code", "")).strip().upper()
    fn = str(row.get("filename", "")).strip().upper()
    return tc.startswith("VA") or fn.startswith("VA")


def _label_from_row(row: pd.Series) -> str:
    lab = str(row.get("label", "")).strip().lower()
    if lab in {"pd", "control_elderly"}:
        return lab

    grp = str(row.get("group", "")).strip().lower()
    if "parkinson" in grp or grp == "pd":
        return "pd"
    if "elderly" in grp:
        return "control_elderly"

    return lab if lab else grp


def _make_binary_y_from_label(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    return (s == "pd").astype(int).values


def _stable_seed(s: str) -> int:
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h & 0x7FFFFFFF)


def _add_white_noise(y: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    x = y.astype(np.float32)
    rng = np.random.default_rng(seed)

    sig_rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
    noise_rms = sig_rms / (10 ** (snr_db / 20.0))
    noise = rng.normal(0.0, noise_rms, size=x.shape).astype(np.float32)
    return x + noise


def main():
    if not FILE_INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing: {FILE_INDEX_CSV} (run scripts/00_build_index.py)")

    df = pd.read_csv(FILE_INDEX_CSV)

    df = df[df.apply(_is_va_row, axis=1)].copy()
    df["label_norm"] = df.apply(_label_from_row, axis=1)
    df = df[df["label_norm"].isin(["pd", "control_elderly"])].copy()

    if df.empty:
        raise RuntimeError("No VA rows found for pd/control_elderly.")

    df["speaker"] = df["speaker"].astype(str)

    # Conditions
    snr_list = [None, 20.0, 10.0]   # None = clean
    gain_list = [1.0, 0.5, 2.0]

    # Model on person-level
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
        ]
    )

    # speaker-level table for splitting
    sp = df.groupby("speaker")["label_norm"].first().reset_index()
    sp["y"] = _make_binary_y_from_label(sp["label_norm"])

    gss = GroupShuffleSplit(n_splits=20, test_size=0.30, random_state=42)

    results = []

    print("Files:", len(df), "Speakers:", len(sp), "PD:", int(sp["y"].sum()), "CTRL:", int((1 - sp["y"]).sum()))
    print("Running waveform robustness over splits...")

    for split_id, (tr_idx, te_idx) in enumerate(gss.split(sp[["speaker"]], sp["y"], sp["speaker"])):
        train_speakers = set(sp.iloc[tr_idx]["speaker"].astype(str))
        test_speakers = set(sp.iloc[te_idx]["speaker"].astype(str))

        df_tr = df[df["speaker"].isin(train_speakers)].copy()
        df_te = df[df["speaker"].isin(test_speakers)].copy()

        # --- Extract CLEAN train features (file-level -> person-level)
        tr_rows = []
        for _, r in df_tr.iterrows():
            wav = Path(r["path"])
            y, sr = librosa.load(str(wav), sr=16000, mono=True)
            feats = extract_baseline_features(y, sr)
            out = {"speaker": r["speaker"], "label": r["label_norm"]}
            out.update(feats)
            tr_rows.append(out)

        tr_file = pd.DataFrame(tr_rows)
        feat_cols = [c for c in tr_file.columns if c not in ["speaker", "label"]]
        tr_person = tr_file.groupby(["speaker", "label"], as_index=False)[feat_cols].mean(numeric_only=True)

        y_tr = _make_binary_y_from_label(tr_person["label"])
        if len(np.unique(y_tr)) < 2:
            continue

        model.fit(tr_person[feat_cols], y_tr)

        # --- For each condition, perturb TEST waveforms, extract, aggregate, score
        for snr_db in snr_list:
            for gain in gain_list:
                te_rows = []
                for _, r in df_te.iterrows():
                    wav = Path(r["path"])
                    y, sr = librosa.load(str(wav), sr=16000, mono=True)

                    # apply gain
                    if gain != 1.0:
                        y = y * float(gain)

                    # apply noise
                    if snr_db is not None:
                        seed = _stable_seed(f"{split_id}|{wav.name}|{snr_db}|{gain}")
                        y = _add_white_noise(y, float(snr_db), seed)

                    feats = extract_baseline_features(y, sr)
                    out = {"speaker": r["speaker"], "label": r["label_norm"]}
                    out.update(feats)
                    te_rows.append(out)

                te_file = pd.DataFrame(te_rows)
                te_person = te_file.groupby(["speaker", "label"], as_index=False)[feat_cols].mean(numeric_only=True)
                y_te = _make_binary_y_from_label(te_person["label"])

                if len(np.unique(y_te)) < 2:
                    continue

                p = model.predict_proba(te_person[feat_cols])[:, 1]
                auc = roc_auc_score(y_te, p)

                results.append(
                    {
                        "split_id": int(split_id),
                        "snr_db": "clean" if snr_db is None else float(snr_db),
                        "gain": float(gain),
                        "auc": float(auc),
                        "n_test_persons": int(len(te_person)),
                    }
                )

    out = pd.DataFrame(results)
    out.to_csv(OUT_CSV, index=False)

    print("\n=== DONE ===")
    print("Saved:", OUT_CSV)
    print("\nSummary (mean AUC by condition):")
    print(out.groupby(["snr_db", "gain"])["auc"].mean().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
