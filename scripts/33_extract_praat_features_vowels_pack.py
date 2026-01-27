# scripts/33_extract_praat_features_vowels_pack.py
"""
Extraherar Praat-features för vowels_pack (VA/VE/VI/VO/VU, 1 & 2) och aggregerar per speaker.
Merge sker med baseline person-tabellen (features_vowels_pack_person.csv).

Input:
- data_index/file_index.csv
- data_index/features_vowels_pack_person.csv

Output:
- data_index/praat_vowels_pack_file.csv
- data_index/praat_vowels_pack_person.csv
- data_index/features_vowels_pack_person_plus_praat.csv
"""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import parselmouth
    from parselmouth.praat import call
except Exception as e:
    raise ImportError(
        "Kunde inte importera parselmouth.\n"
        "Installera med: pip install praat-parselmouth\n"
        f"Originalfel: {e}"
    )

VOWELS_PACK_TASKS = ["VA1", "VA2", "VE1", "VE2", "VI1", "VI2", "VO1", "VO2", "VU1", "VU2"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def base_task(task_code: str) -> str | None:
    m = re.search(r"(V[AEIOU][12])", str(task_code))
    return m.group(1) if m else None


def safe_float(x) -> float:
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return float("nan")
        return x
    except Exception:
        return float("nan")


def extract_praat(wav_path: Path) -> dict:
    out = {
        "f0_mean_hz": np.nan,
        "f0_std_hz": np.nan,
        "f0_min_hz": np.nan,
        "f0_max_hz": np.nan,
        "hnr_mean_db": np.nan,
        "jitter_local": np.nan,
        "shimmer_local": np.nan,
        "praat_ok": 0,
    }
    try:
        snd = parselmouth.Sound(str(wav_path))

        # bred pitch range
        pitch = call(snd, "To Pitch", 0.0, 75, 600)
        f0 = pitch.selected_array["frequency"]
        voiced = f0[f0 > 0]

        if voiced.size >= 3:
            out["f0_mean_hz"] = safe_float(np.mean(voiced))
            out["f0_std_hz"] = safe_float(np.std(voiced, ddof=1) if voiced.size > 1 else 0.0)
            out["f0_min_hz"] = safe_float(np.min(voiced))
            out["f0_max_hz"] = safe_float(np.max(voiced))

        harm = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        out["hnr_mean_db"] = safe_float(call(harm, "Get mean", 0, 0))

        point = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        out["jitter_local"] = safe_float(call(point, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        out["shimmer_local"] = safe_float(call([snd, point], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))

        if (not np.isnan(out["hnr_mean_db"])) or (not np.isnan(out["f0_mean_hz"])):
            out["praat_ok"] = 1

        return out
    except Exception:
        return out


def main() -> None:
    root = project_root()
    data_dir = root / "data_index"

    index_path = data_dir / "file_index.csv"
    base_person_path = data_dir / "features_vowels_pack_person.csv"

    if not index_path.exists():
        raise FileNotFoundError(f"Hittar inte {index_path}")
    if not base_person_path.exists():
        raise FileNotFoundError(f"Hittar inte {base_person_path}")

    idx = pd.read_csv(index_path)
    for c in ["path", "speaker", "label", "task_code"]:
        if c not in idx.columns:
            raise ValueError(f"file_index.csv saknar kolumn: {c}")

    idx["label"] = idx["label"].astype(str).str.strip()
    idx["speaker"] = idx["speaker"].astype(str)
    idx["base_task"] = idx["task_code"].apply(base_task)

    df = idx[idx["base_task"].isin(VOWELS_PACK_TASKS)].copy()
    total = len(df)

    rows = []
    missing_files = 0

    for i, r in enumerate(df.itertuples(index=False), start=1):
        p = Path(r.path)
        if not p.exists():
            missing_files += 1
            feats = {k: np.nan for k in ["f0_mean_hz","f0_std_hz","f0_min_hz","f0_max_hz","hnr_mean_db","jitter_local","shimmer_local"]}
            feats["praat_ok"] = 0
        else:
            feats = extract_praat(p)

        rows.append({
            "path": str(r.path),
            "speaker": r.speaker,
            "label": r.label,
            "task_code": r.task_code,
            "base_task": r.base_task,
            **feats
        })

        if i % 50 == 0 or i == total:
            print(f"Processed {i}/{total} files...")

    praat_file = pd.DataFrame(rows)
    out_file = data_dir / "praat_vowels_pack_file.csv"
    praat_file.to_csv(out_file, index=False)

    feat_cols = ["f0_mean_hz","f0_std_hz","f0_min_hz","f0_max_hz","hnr_mean_db","jitter_local","shimmer_local"]
    g = praat_file.groupby(["speaker","label"], as_index=False)
    praat_person = g[feat_cols].median(numeric_only=True)
    ok_rate = g["praat_ok"].mean().rename(columns={"praat_ok":"praat_ok_rate"})
    n_files = g.size().rename(columns={"size":"n_files"})
    praat_person = praat_person.merge(ok_rate, on=["speaker","label"], how="left")
    praat_person = praat_person.merge(n_files, on=["speaker","label"], how="left")

    out_person = data_dir / "praat_vowels_pack_person.csv"
    praat_person.to_csv(out_person, index=False)

    base_person = pd.read_csv(base_person_path)
    base_person["label"] = base_person["label"].astype(str).str.strip()
    base_person["speaker"] = base_person["speaker"].astype(str)

    merged = base_person.merge(praat_person, on=["speaker","label"], how="left")
    out_merged = data_dir / "features_vowels_pack_person_plus_praat.csv"
    merged.to_csv(out_merged, index=False)

    print("\n=== Praat vowels_pack extraction summary ===")
    print(f"Matched files: {total}")
    print(f"Missing files on disk: {missing_files}")
    print(f"Saved file-level:   {out_file.name} (rows={len(praat_file)})")
    print(f"Saved person-level: {out_person.name} (rows={len(praat_person)})")
    print(f"Saved merged:       {out_merged.name} (rows={len(merged)})")

    non_nan = praat_file[feat_cols].notna().sum().to_dict()
    print("Non-NaN counts (file-level):")
    for k, v in non_nan.items():
        print(f"  {k}: {int(v)} / {total}")

    ok = float(praat_file["praat_ok"].mean())
    print(f"praat_ok rate (file-level mean): {ok:.3f}")


if __name__ == "__main__":
    main()
