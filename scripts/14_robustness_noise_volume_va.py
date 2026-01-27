# FILE: scripts/14_robustness_noise_volume_va.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[1]
DATA_INDEX = ROOT / "data_index"

BASELINE_PERSON_CSV = DATA_INDEX / "features_va_person.csv"
BASELINE_FILE_CSV = DATA_INDEX / "features_va_file.csv"
OUT_CSV = DATA_INDEX / "robustness_va_noise_volume.csv"


def _get_feature_cols(df: pd.DataFrame, exclude=("path", "filename", "speaker", "group", "label", "task_code")):
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _make_binary_y(df: pd.DataFrame) -> np.ndarray:
    # label is text: 'pd' vs 'control_elderly'
    s = df["label"].astype(str).str.strip().str.lower()
    return (s == "pd").astype(int).values


def _snr_db_to_noise_scale(signal_rms: float, snr_db: float) -> float:
    # snr_db = 20*log10(signal_rms/noise_rms)
    # => noise_rms = signal_rms / (10^(snr_db/20))
    return signal_rms / (10 ** (snr_db / 20.0))


def _set_seed_from_str(s: str) -> int:
    # stable hash to int32
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h & 0x7FFFFFFF)


def _apply_noise_and_gain(y: np.ndarray, snr_db: float | None, gain: float | None, seed: int) -> np.ndarray:
    x = y.astype(np.float32).copy()

    if gain is not None:
        x = x * float(gain)

    if snr_db is not None:
        rng = np.random.default_rng(seed)
        sig_rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
        noise_rms = _snr_db_to_noise_scale(sig_rms, float(snr_db))
        noise = rng.normal(0.0, noise_rms, size=x.shape).astype(np.float32)
        x = x + noise

    return x


def main():
    # We will do file-level robustness, then aggregate to person-level within each split.
    if not BASELINE_FILE_CSV.exists():
        raise FileNotFoundError(f"Missing: {BASELINE_FILE_CSV} (run scripts/01_extract_features_va.py first)")

    df_file = pd.read_csv(BASELINE_FILE_CSV)

    # Keep VA and only PD vs elderly control
    df_file = df_file[df_file["task_code"].astype(str).str.upper().str.startswith("VA")].copy()
    df_file = df_file[df_file["label"].astype(str).str.lower().isin(["pd", "control_elderly"])].copy()

    df_file["speaker"] = df_file["speaker"].astype(str)

    feat_cols = _get_feature_cols(df_file)

    print("File-level rows:", len(df_file))
    print("Speakers:", df_file["speaker"].nunique())
    print("Features:", len(feat_cols))

    # Model
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
        ]
    )

    # Conditions to test
    snr_list = [None, 20.0, 10.0]     # None = clean
    gain_list = [None, 0.5, 2.0]      # None = clean gain

    # Repeated splits by speaker
    gss = GroupShuffleSplit(n_splits=30, test_size=0.30, random_state=42)

    results = []

    speakers = df_file["speaker"].values
    labels_person = df_file.groupby("speaker")["label"].first().reset_index()
    labels_person["y"] = (labels_person["label"].astype(str).str.lower() == "pd").astype(int)

    for split_id, (train_sp_idx, test_sp_idx) in enumerate(gss.split(labels_person[["speaker"]], labels_person["y"], labels_person["speaker"])):
        train_speakers = set(labels_person.iloc[train_sp_idx]["speaker"].astype(str))
        test_speakers = set(labels_person.iloc[test_sp_idx]["speaker"].astype(str))

        df_tr = df_file[df_file["speaker"].isin(train_speakers)].copy()
        df_te = df_file[df_file["speaker"].isin(test_speakers)].copy()

        # Clean person aggregation for train
        tr_person = df_tr.groupby(["speaker", "label"], as_index=False)[feat_cols].mean(numeric_only=True)
        te_person_clean = df_te.groupby(["speaker", "label"], as_index=False)[feat_cols].mean(numeric_only=True)

        y_tr = _make_binary_y(tr_person)
        y_te_clean = _make_binary_y(te_person_clean)

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te_clean)) < 2:
            continue

        # Fit on clean train
        model.fit(tr_person[feat_cols], y_tr)

        # Evaluate conditions by perturbing test FEATURES in a simple way:
        # We'll perturb ONLY the amplitude-related features as a proxy (rms_mean, rms_std),
        # and add noise proxy by jittering mfcc means slightly (simple, fast, reproducible).
        #
        # This is a FEATURE-SPACE robustness proxy (not waveform-level).
        # Next iteration can do waveform-level, but this is a safe first step.

        for snr_db in snr_list:
            for gain in gain_list:
                te_person = te_person_clean.copy()

                # Gain proxy: scale rms features
                if gain is not None:
                    for c in ["rms_mean", "rms_std"]:
                        if c in te_person.columns:
                            te_person[c] = te_person[c] * float(gain)

                # Noise proxy: add small noise to MFCC mean/std columns (deterministic seed)
                if snr_db is not None:
                    seed = _set_seed_from_str(f"{split_id}|{snr_db}|{gain}")
                    rng = np.random.default_rng(seed)
                    # stronger noise for lower SNR
                    scale = 0.02 if snr_db >= 20 else 0.05
                    mfcc_cols = [c for c in feat_cols if c.startswith("mfcc") and (c.endswith("_mean") or c.endswith("_std"))]
                    if mfcc_cols:
                        te_person[mfcc_cols] = te_person[mfcc_cols] + rng.normal(0.0, scale, size=(len(te_person), len(mfcc_cols)))

                y_te = _make_binary_y(te_person)
                p = model.predict_proba(te_person[feat_cols])[:, 1]
                auc = roc_auc_score(y_te, p)

                results.append(
                    {
                        "split_id": split_id,
                        "snr_db": "clean" if snr_db is None else float(snr_db),
                        "gain": "clean" if gain is None else float(gain),
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
