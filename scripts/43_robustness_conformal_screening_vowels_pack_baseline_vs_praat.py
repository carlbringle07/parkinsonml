"""
scripts/43_robustness_conformal_screening_vowels_pack_baseline_vs_praat.py

Robustness + Conformal screening (vowels_pack): baseline vs baseline+Praat

- Normaliserar task_code -> base_task (VA1, VE2, ...)
- Lägger till SimpleImputer(strategy="median") i pipeline (fixar NaN)
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import librosa
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
def merge_praat_person_features(df_person, praat_person_csv=r"data_index\praat_vowels_pack_person.csv", merge_key="speaker"):
    """
    Merge person-level Praat features into df_person.

    Expects praat_person_csv to have one row per speaker/person, with columns like:
      f0_mean_hz, f0_std_hz, f0_min_hz, f0_max_hz, hnr_mean_db, jitter_local, shimmer_local
    (either prefixed with praat_ already, or not). We will prefix with praat_ if missing.

    Sanity-check: Warn (not crash) if the *feature* columns (f0/hnr/jitter/shimmer) look all-NaN or constant.
    """
    import pandas as pd
    import numpy as np

    out = df_person.copy()

    if praat_person_csv is None:
        return out

    praat_path = Path(praat_person_csv)
    if not praat_path.exists():
        # If file doesn't exist, just return baseline (but warn)
        print(f"[WARN] Praat person CSV not found: {praat_person_csv}. Continuing without Praat.")
        return out

    praat = pd.read_csv(praat_path)

    # --- Find merge key in both tables ---
    def _find_key(df):
        for cand in ["speaker", "person", "subject", "id", "ID", "Speaker", "Person"]:
            if cand in df.columns:
                return cand
        return None

    key_a = merge_key if merge_key in out.columns else _find_key(out)
    key_b = merge_key if merge_key in praat.columns else _find_key(praat)

    if key_a is None or key_b is None:
        print(f"[WARN] Could not find merge keys (df_person key={key_a}, praat key={key_b}). Continuing without Praat.")
        return out

    if key_a != merge_key:
        out = out.rename(columns={key_a: merge_key})
    if key_b != merge_key:
        praat = praat.rename(columns={key_b: merge_key})

    # --- Ensure praat columns are prefixed with 'praat_' (except merge_key) ---
    new_cols = {}
    for c in praat.columns:
        if c == merge_key:
            continue
        cl = str(c)
        if cl.lower().startswith("praat_"):
            new_cols[c] = cl
        else:
            new_cols[c] = "praat_" + cl
    praat = praat.rename(columns=new_cols)

    # If duplicates per speaker, aggregate safely
    if praat[merge_key].duplicated().any():
        num_cols = [c for c in praat.columns if c != merge_key and pd.api.types.is_numeric_dtype(praat[c])]
        other_cols = [c for c in praat.columns if c != merge_key and c not in num_cols]
        agg = {}
        for c in num_cols:
            agg[c] = "mean"
        for c in other_cols:
            agg[c] = "first"
        praat = praat.groupby(merge_key, as_index=False).agg(agg)

    # Merge
    out = out.merge(praat, on=merge_key, how="left")

    # --- Sanity-check (features only, ignore metadata like praat_label/praat_n_files/praat_ok_rate) ---
    praat_cols = [c for c in out.columns if str(c).lower().startswith("praat_")]
    praat_feat_cols = [c for c in praat_cols if any(k in str(c).lower() for k in ["f0", "hnr", "jitter", "shimmer"])]

    if len(praat_feat_cols) == 0:
        print("[WARN] No praat feature columns (f0/hnr/jitter/shimmer) found after merge. +Praat may be ≈ baseline.")
        return out

    nan_max = float(out[praat_feat_cols].isna().mean().max())
    nun = out[praat_feat_cols].nunique(dropna=False).sort_values()
    nun_min = int(nun.min()) if len(nun) else 0

    if nan_max > 0.0 or nun_min <= 1:
        print(
            "[WARN] Praat sanity-check triggade (endast riktiga praat-features). "
            "Om praat_* är NaN/konstanta kommer +Praat ≈ baseline.\n"
            f"nan_max={nan_max}\n"
            "nunique head:\n" + str(nun.head(20))
        )

    return out



def stable_u32(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def load_audio(path: str | Path, sr: int = SR) -> tuple[np.ndarray, int]:
    y, sr2 = librosa.load(str(path), sr=sr, mono=True)
    if y is None or len(y) == 0:
        return np.zeros(1, dtype=np.float32), sr2
    y = y.astype(np.float32, copy=False)
    y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    if len(y) == 0:
        y = np.zeros(1, dtype=np.float32)
    return y, sr2


def apply_condition(y: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    y2 = y.astype(np.float32, copy=True)

    if condition == "clean":
        return y2

    if condition.startswith("snr"):
        snr_db = 20.0 if condition == "snr20" else 10.0
        sig_power = float(np.mean(y2 ** 2)) + 1e-12
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(0.0, np.sqrt(noise_power), size=y2.shape).astype(np.float32)
        return np.clip(y2 + noise, -1.0, 1.0)

    if condition == "vol0p5":
        return np.clip(y2 * 0.5, -1.0, 1.0)

    if condition == "vol2p0":
        return np.clip(y2 * 2.0, -1.0, 1.0)

    raise ValueError(f"Okänd condition: {condition}")


def baseline_features(y: np.ndarray, sr: int) -> dict[str, float]:
    feats: dict[str, float] = {}

    # Pitch (yin)
    try:
        f0 = librosa.yin(y=y, fmin=50, fmax=500, sr=sr)
        f0 = f0[np.isfinite(f0)]
        f0 = f0[f0 > 0]
        feats["f0_median"] = float(np.median(f0)) if len(f0) else np.nan
        feats["f0_mean"] = float(np.mean(f0)) if len(f0) else np.nan
        feats["f0_std"] = float(np.std(f0)) if len(f0) else np.nan
    except Exception:
        feats["f0_median"] = np.nan
        feats["f0_mean"] = np.nan
        feats["f0_std"] = np.nan

    # HNR approx
    try:
        harm, perc = librosa.effects.hpss(y)
        ph = float(np.sum(harm**2)) + 1e-12
        pn = float(np.sum(perc**2)) + 1e-12
        feats["hnr_mean_db"] = float(10.0 * np.log10(ph / pn))
    except Exception:
        feats["hnr_mean_db"] = np.nan

    # Spectral centroid/rolloff
    try:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        feats["spectral_centroid_mean"] = float(np.mean(centroid))
        feats["spectral_centroid_std"] = float(np.std(centroid))
    except Exception:
        feats["spectral_centroid_mean"] = np.nan
        feats["spectral_centroid_std"] = np.nan

    try:
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        feats["spectral_rolloff_mean"] = float(np.mean(rolloff))
        feats["spectral_rolloff_std"] = float(np.std(rolloff))
    except Exception:
        feats["spectral_rolloff_mean"] = np.nan
        feats["spectral_rolloff_std"] = np.nan

    # ZCR
    try:
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        feats["zcr_mean"] = float(np.mean(zcr))
    except Exception:
        feats["zcr_mean"] = np.nan

    # MFCC 13 (mean + std)
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        for i in range(13):
            feats[f"mfcc_mean_{i+1}"] = float(mfcc_mean[i])
        for i in range(13):
            feats[f"mfcc_std_{i+1}"] = float(mfcc_std[i])
    except Exception:
        for i in range(13):
            feats[f"mfcc_mean_{i+1}"] = np.nan
        for i in range(13):
            feats[f"mfcc_std_{i+1}"] = np.nan

    return feats


def praat_features(y: np.ndarray, sr: int) -> dict[str, float]:
    feats: dict[str, float] = {
        "praat_f0_mean_hz": np.nan,
        "praat_f0_std_hz": np.nan,
        "praat_f0_min_hz": np.nan,
        "praat_f0_max_hz": np.nan,
        "praat_f0_median_hz": np.nan,
        "praat_f0_range_hz": np.nan,
        "praat_hnr_mean_db": np.nan,
        "praat_jitter_local": np.nan,
        "praat_shimmer_local": np.nan,
    }

    if not PARSELMOUTH_OK:
        return feats

    try:
        snd = parselmouth.Sound(y, sampling_frequency=sr)

        pitch = snd.to_pitch(time_step=0.0, pitch_floor=75, pitch_ceiling=500)
        f0 = pitch.selected_array["frequency"]
        f0 = f0[np.isfinite(f0)]
        f0 = f0[f0 > 0]
        if len(f0) > 0:
            feats["praat_f0_mean_hz"] = float(np.mean(f0))
            feats["praat_f0_std_hz"] = float(np.std(f0))
            feats["praat_f0_min_hz"] = float(np.min(f0))
            feats["praat_f0_max_hz"] = float(np.max(f0))
            feats["praat_f0_median_hz"] = float(np.median(f0))
            feats["praat_f0_range_hz"] = float(np.max(f0) - np.min(f0))

        try:
            harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=75)
            hvals = np.array(harm.values).flatten()
            hvals = hvals[np.isfinite(hvals)]
            if len(hvals) > 0:
                feats["praat_hnr_mean_db"] = float(np.mean(hvals))
        except Exception:
            pass

        try:
            pp = praat_call(snd, "To PointProcess (periodic, cc)", 75, 500)
            feats["praat_jitter_local"] = safe_float(
                praat_call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            )
            feats["praat_shimmer_local"] = safe_float(
                praat_call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            )
        except Exception:
            pass

    except Exception:
        return feats

    return feats


def infer_label(row: pd.Series) -> str | None:
    if "label" in row and isinstance(row["label"], str) and row["label"].strip():
        return row["label"].strip()

    g = str(row.get("group", "")).lower()
    if "parkinson" in g:
        return "pd"
    if "elderly" in g and "control" in g:
        return "control_elderly"
    if "young" in g and "control" in g:
        return "control_young"
    return None


def normalize_task(s: str) -> str | None:
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    m = re.match(r"^([A-Za-z]{1,2}\d)", s)
    if m:
        return m.group(1).upper()
    if len(s) >= 3:
        return s[:3].upper()
    return s.upper()


def infer_task_code(row: pd.Series) -> str | None:
    if "task_code" in row and pd.notna(row["task_code"]):
        t = normalize_task(row["task_code"])
        if t:
            return t
    fn = str(row.get("filename", "")) or Path(str(row.get("path", ""))).name
    return normalize_task(fn)


def infer_speaker(row: pd.Series) -> str | None:
    if "speaker" in row and isinstance(row["speaker"], str) and row["speaker"].strip():
        return row["speaker"].strip()
    p = Path(str(row.get("path", "")))
    if p.exists():
        return p.parent.name
    return None


def build_person_table_for_condition(condition: str, with_praat: bool) -> pd.DataFrame:
    tag = "plus_praat" if with_praat else "baseline"
    cache_path = CACHE_DIR / f"{condition}_{tag}_person.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    if not FILE_INDEX_PATH.exists():
        raise FileNotFoundError(f"Hittar inte {FILE_INDEX_PATH}. Kör 00_build_index.py först.")

    idx = pd.read_csv(FILE_INDEX_PATH)

    idx["label_std"] = idx.apply(infer_label, axis=1)
    idx["task_std"] = idx.apply(infer_task_code, axis=1)
    idx["speaker_std"] = idx.apply(infer_speaker, axis=1)

    if condition == "clean" and tag == "baseline":
        print("\n[DEBUG] Top group values:")
        if "group" in idx.columns:
            print(idx["group"].value_counts().head(5).to_string())
        print("\n[DEBUG] Top inferred task_std values:")
        print(idx["task_std"].value_counts().head(15).to_string())
        print("\n[DEBUG] Top inferred label_std values:")
        print(idx["label_std"].value_counts().head(10).to_string())
        print()

    idx = idx[idx["label_std"].isin(["pd", "control_elderly"])].copy()
    idx = idx[idx["task_std"].isin(VOWELS_PACK_TASKS)].copy()
    idx = idx.dropna(subset=["speaker_std", "path"]).copy()
    idx["y"] = (idx["label_std"] == "pd").astype(int)

    print(f"[{condition} | {tag}] Matched files after filters: {len(idx)}")

    rows = []
    missing = 0
    for r in idx.itertuples(index=False):
        path = getattr(r, "path")
        speaker = getattr(r, "speaker_std")
        ylab = int(getattr(r, "y"))

        seed = stable_u32(f"{condition}::{path}")
        rng = np.random.default_rng(seed)

        try:
            ysig, sr = load_audio(path, sr=SR)
            ysig = apply_condition(ysig, condition, rng)
            feats = baseline_features(ysig, sr)
            if with_praat:
                feats.update(praat_features(ysig, sr))
            feats["speaker"] = speaker
            feats["y"] = ylab
            rows.append(feats)
        except Exception:
            missing += 1

    df_file = pd.DataFrame(rows)
    print(f"[{condition} | {tag}] Feature rows built: {len(df_file)} | missing: {missing}")

    if df_file.empty:
        raise RuntimeError(f"Inga features skapades för condition={condition}, tag={tag}.")

    feature_cols = [c for c in df_file.columns if c not in ["speaker", "y"]]
    df_person = df_file.groupby(["speaker", "y"], as_index=False)[feature_cols].mean(numeric_only=True)

    df_person.to_csv(cache_path, index=False)
    print(f"[{condition} | {tag}] Saved cache: {cache_path} (rows={len(df_person)})")
    # --- PATCH: robust merge of Praat person features (fail-fast if missing) ---
    if with_praat:
        df_person = merge_praat_person_features(df_person, praat_person_csv=r"data_index\praat_vowels_pack_person.csv")
    # --- END PATCH ---

    return df_person


def pooled_conformal_prediction_sets(
    proba_test: np.ndarray,
    proba_cal: np.ndarray,
    y_cal: np.ndarray,
    alpha: float,
    classes: np.ndarray,
) -> list[set[int]]:
    class_to_col = {int(c): j for j, c in enumerate(classes)}
    true_scores = np.array([proba_cal[i, class_to_col[int(y_cal[i])]] for i in range(len(y_cal))], dtype=float)
    n = len(true_scores)

    pred_sets: list[set[int]] = []
    for i in range(proba_test.shape[0]):
        s = set()
        for k in classes:
            pk = proba_test[i, class_to_col[int(k)]]
            pval = (np.sum(true_scores <= pk) + 1.0) / (n + 1.0)
            if pval > alpha:
                s.add(int(k))
        pred_sets.append(s)
    return pred_sets


def eval_conformal_screening_repeated(
    df_person: pd.DataFrame,
    feature_cols: list[str],
    alpha: float,
    n_splits: int,
    test_size: float,
    calib_frac_of_train: float,
    seed0: int,
) -> pd.DataFrame:
    X = df_person[feature_cols].to_numpy(dtype=float)
    y = df_person["y"].to_numpy(dtype=int)

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed0)

    rows = []
    for split_idx, (traincal_idx, test_idx) in enumerate(sss.split(X, y), start=1):
        X_traincal, y_traincal = X[traincal_idx], y[traincal_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=calib_frac_of_train, random_state=seed0 + 10_000 + split_idx
        )
        train_idx_rel, calib_idx_rel = next(sss2.split(X_traincal, y_traincal))
        X_train, y_train = X_traincal[train_idx_rel], y_traincal[train_idx_rel]
        X_cal, y_cal = X_traincal[calib_idx_rel], y_traincal[calib_idx_rel]

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, solver="liblinear")),
            ]
        )
        model.fit(X_train, y_train)

        proba_cal = model.predict_proba(X_cal)
        proba_test = model.predict_proba(X_test)

        try:
            pos_col = list(model.named_steps["clf"].classes_).index(1)
            auc = float(roc_auc_score(y_test, proba_test[:, pos_col]))
        except Exception:
            auc = np.nan

        classes = model.named_steps["clf"].classes_
        pred_sets = pooled_conformal_prediction_sets(
            proba_test=proba_test,
            proba_cal=proba_cal,
            y_cal=y_cal,
            alpha=alpha,
            classes=classes,
        )

        covered = [int(y_test[i] in pred_sets[i]) for i in range(len(y_test))]
        sizes = [len(s) for s in pred_sets]
        decided_mask = np.array([sz == 1 for sz in sizes], dtype=bool)

        acc_decided = np.nan
        if decided_mask.any():
            pred_labels = []
            for s in pred_sets:
                pred_labels.append(next(iter(s)) if len(s) == 1 else -1)
            pred_labels = np.array(pred_labels, dtype=int)
            acc_decided = float(np.mean(pred_labels[decided_mask] == y_test[decided_mask]))

        rows.append({
            "split": split_idx,
            "n_test": int(len(y_test)),
            "auc": auc,
            "coverage": float(np.mean(covered)),
            "decided_rate": float(np.mean(decided_mask)),
            "unsure_rate": float(np.mean(~decided_mask)),
            "acc_decided": acc_decided,
        })

    return pd.DataFrame(rows)


def summarize_metrics(df_splits: pd.DataFrame) -> dict[str, float]:
    out = {}
    for col in ["auc", "coverage", "decided_rate", "unsure_rate", "acc_decided"]:
        out[f"{col}_mean"] = float(df_splits[col].mean())
        out[f"{col}_std"] = float(df_splits[col].std(ddof=1))
    return out


def main() -> None:
    print("=== Robustness + Conformal screening (vowels_pack) ===")
    print(f"alpha={ALPHA} | n_splits={N_SPLITS} | test_size={TEST_SIZE} | calib_frac_of_train={CALIB_FRAC_OF_TRAIN}")
    if not PARSELMOUTH_OK:
        print("OBS: parselmouth saknas -> kommer bara köra baseline (ingen +Praat).")

    all_split_rows = []
    summary_rows = []

    for condition in CONDITIONS:
        df_base = build_person_table_for_condition(condition, with_praat=False)
        base_feature_cols = [c for c in df_base.columns if c not in ["speaker", "y"]]

        splits_base = eval_conformal_screening_repeated(
            df_person=df_base,
            feature_cols=base_feature_cols,
            alpha=ALPHA,
            n_splits=N_SPLITS,
            test_size=TEST_SIZE,
            calib_frac_of_train=CALIB_FRAC_OF_TRAIN,
            seed0=123,
        )
        splits_base["condition"] = condition
        splits_base["model"] = "baseline"
        all_split_rows.append(splits_base)

        summary_rows.append({
            "condition": condition,
            "model": "baseline",
            **summarize_metrics(splits_base),
        })

        if PARSELMOUTH_OK:
            df_praat = build_person_table_for_condition(condition, with_praat=True)
            praat_feature_cols = [c for c in df_praat.columns if c not in ["speaker", "y"]]

            splits_praat = eval_conformal_screening_repeated(
                df_person=df_praat,
                feature_cols=praat_feature_cols,
                alpha=ALPHA,
                n_splits=N_SPLITS,
                test_size=TEST_SIZE,
                calib_frac_of_train=CALIB_FRAC_OF_TRAIN,
                seed0=123,
            )
            splits_praat["condition"] = condition
            splits_praat["model"] = "plus_praat"
            all_split_rows.append(splits_praat)

            summary_rows.append({
                "condition": condition,
                "model": "plus_praat",
                **summarize_metrics(splits_praat),
            })

    df_all_splits = pd.concat(all_split_rows, ignore_index=True)
    df_summary = pd.DataFrame(summary_rows)

    df_all_splits.to_csv(OUT_PER_SPLIT, index=False)
    df_summary.to_csv(OUT_SUMMARY, index=False)

    comp_rows = []
    for condition in CONDITIONS:
        b = df_summary[(df_summary["condition"] == condition) & (df_summary["model"] == "baseline")]
        p = df_summary[(df_summary["condition"] == condition) & (df_summary["model"] == "plus_praat")]
        if b.empty:
            continue

        row = {"condition": condition}
        for metric in ["auc", "coverage", "decided_rate", "unsure_rate", "acc_decided"]:
            row[f"baseline_{metric}_mean"] = float(b[f"{metric}_mean"].iloc[0])
            row[f"baseline_{metric}_std"] = float(b[f"{metric}_std"].iloc[0])
            if not p.empty:
                row[f"plus_praat_{metric}_mean"] = float(p[f"{metric}_mean"].iloc[0])
                row[f"plus_praat_{metric}_std"] = float(p[f"{metric}_std"].iloc[0])
                row[f"delta_{metric}_mean"] = row[f"plus_praat_{metric}_mean"] - row[f"baseline_{metric}_mean"]
        comp_rows.append(row)

    df_compare = pd.DataFrame(comp_rows)
    df_compare.to_csv(OUT_COMPARE, index=False)

    print("\n=== KLART ===")
    print(f"Saved per-split: {OUT_PER_SPLIT}")
    print(f"Saved summary  : {OUT_SUMMARY}")
    print(f"Saved compare : {OUT_COMPARE}")

    with pd.option_context("display.max_columns", 999, "display.width", 160):
        print("\nSummary (first rows):")
        print(df_summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
