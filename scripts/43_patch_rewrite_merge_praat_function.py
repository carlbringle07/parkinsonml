# scripts/43_patch_rewrite_merge_praat_function.py
from __future__ import annotations

import re
from pathlib import Path

TARGET = Path("scripts/43_robustness_conformal_screening_vowels_pack_baseline_vs_praat.py")

NEW_FUNC = r'''
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
'''.lstrip("\n")


def main():
    if not TARGET.exists():
        raise SystemExit(f"Hittar inte {TARGET.as_posix()}")

    txt = TARGET.read_text(encoding="utf-8")

    # Replace the whole function def merge_praat_person_features(...)
    pattern = re.compile(
        r"(?ms)^\s*def\s+merge_praat_person_features\s*\(.*?\):\n.*?(?=^\s*def\s+\w+\s*\(|\Z)"
    )

    m = pattern.search(txt)
    if not m:
        raise SystemExit(
            "Kunde inte hitta funktionen merge_praat_person_features(...) att ersätta.\n"
            "Sök i filen efter 'def merge_praat_person_features' och kontrollera att den finns."
        )

    backup = TARGET.with_suffix(TARGET.suffix + ".bak_before_rewrite_merge")
    backup.write_text(txt, encoding="utf-8")

    new_txt = txt[:m.start()] + NEW_FUNC + "\n\n" + txt[m.end():]
    TARGET.write_text(new_txt, encoding="utf-8")

    print("Patch: merge_praat_person_features(...) omskriven (fixar indent + robust sanity-check).")
    print(f"Backup sparad: {backup.as_posix()}")
    print(f"Patchad fil : {TARGET.as_posix()}")


if __name__ == "__main__":
    main()
