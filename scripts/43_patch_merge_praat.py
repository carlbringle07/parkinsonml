# scripts/43_patch_merge_praat.py
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime

# Target script to patch
SCRIPT_PATH = Path("scripts/43_robustness_conformal_screening_vowels_pack_baseline_vs_praat.py")

# Default Praat person CSV (you can change this if needed)
PRAAT_PERSON_CSV_DEFAULT = r"data_index\praat_vowels_pack_person.csv"

MERGE_HELPER = r'''
def merge_praat_person_features(df_person, praat_person_csv=r"data_index\praat_vowels_pack_person.csv"):
    """
    Robust merge of Praat person-level features into df_person.

    - Auto-detect merge key
    - Drops any existing praat_* columns before merging
    - Prefixes merged columns with praat_
    - Fail-fast if merge produces all-NaN / constant Praat columns
    """
    import pandas as pd

    praat = pd.read_csv(praat_person_csv)

    # Drop existing (possibly broken) praat columns if present
    existing_praat_cols = [c for c in df_person.columns if str(c).lower().startswith("praat_")]
    if existing_praat_cols:
        df_person = df_person.drop(columns=existing_praat_cols)

    # Find a common merge key automatically
    candidate_keys = [
        "speaker_id", "person_id", "subject_id", "participant_id",
        "speaker", "person", "subject", "participant",
        "id", "ID",
    ]
    key = None
    for k in candidate_keys:
        if k in df_person.columns and k in praat.columns:
            key = k
            break

    if key is None:
        inter = [c for c in df_person.columns if c in praat.columns]
        id_like = [c for c in inter if ("id" in str(c).lower() or "speaker" in str(c).lower() or "subject" in str(c).lower())]
        if id_like:
            key = id_like[0]

    if key is None:
        raise RuntimeError(
            "Could not find a common merge key between df_person and praat table.\n"
            + "df_person cols: " + str(list(df_person.columns)) + "\n"
            + "praat cols: " + str(list(praat.columns)) + "\n"
        )

    praat_feat_cols = [c for c in praat.columns if c != key]

    # Prefix praat columns to avoid collisions
    praat2 = praat[[key] + praat_feat_cols].copy()
    rename_map = {}
    for c in praat_feat_cols:
        rename_map[c] = "praat_" + str(c)
    praat2 = praat2.rename(columns=rename_map)

    out = df_person.merge(praat2, on=key, how="left")

    praat_cols = [c for c in out.columns if str(c).lower().startswith("praat_")]
    if not praat_cols:
        raise RuntimeError("No praat_* columns present after merge (unexpected).")

    # Fail-fast if merge didn't match (all NaN)
    nan_rate = out[praat_cols].isna().mean().sort_values(ascending=False)
    if float(nan_rate.max()) >= 0.999:
        raise RuntimeError(
            "Praat merge failed: praat_* columns are ~all NaN after merge.\n"
            + "Merge key used: " + str(key) + "\n"
            + "Top NaN rates:\n" + str(nan_rate.head(20)) + "\n"
            + "Hint: df_person[" + str(key) + "] values may not match praat[" + str(key) + "]."
        )

    # Fail-fast if constant (e.g. one unique value everywhere)
    nun = out[praat_cols].nunique(dropna=False).sort_values()
    if int(nun.min()) <= 1:
        raise RuntimeError(
            "Praat merge suspicious: praat_* columns look constant/empty after merge.\n"
            + "Merge key used: " + str(key) + "\n"
            + "nunique head:\n" + str(nun.head(20))
        )

    return out
'''.strip("\n")


def insert_merge_helper(text: str) -> str:
    if "def merge_praat_person_features" in text:
        return text

    lines = text.splitlines(True)

    insert_at = 0
    for i, line in enumerate(lines[:400]):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1

    helper = "\n\n" + MERGE_HELPER + "\n\n"
    lines.insert(insert_at, helper)
    return "".join(lines)


def patch_build_person_table(text: str) -> str:
    # Much looser match: just find the line that starts the function (signature can span multiple lines)
    m = re.search(r"(?m)^\s*def\s+build_person_table_for_condition\b", text)
    if not m:
        # Print a hint by trying to find any similar function names
        candidates = re.findall(r"(?m)^\s*def\s+([a-zA-Z0-9_]+)\s*\(", text)
        nearby = [c for c in candidates if "person" in c.lower() or "condition" in c.lower()]
        raise RuntimeError(
            "Could not find function build_person_table_for_condition(...) in script 43.\n"
            + "Similar defs found: " + str(nearby[:25])
        )

    start = m.start()

    # Find end of this function: next top-level 'def ' after start
    m2 = re.search(r"(?m)^\s*def\s+", text[m.end():])
    end = len(text) if not m2 else m.end() + m2.start()

    block = text[start:end]

    # If already patched, skip
    if "merge_praat_person_features(df_person" in block:
        return text

    # Find a return line that returns df_person (allow spaces)
    r = re.search(r"(?m)^\s*return\s+df_person\b", block)
    if not r:
        raise RuntimeError(
            "Could not find a line 'return df_person' inside build_person_table_for_condition.\n"
            "Open script 43 and search inside that function for its return statement."
        )

    return_line = block[r.start():].splitlines()[0]
    indent = return_line.split("return")[0]

    injection = (
        f"{indent}# --- PATCH: robust merge of Praat person features (fail-fast if missing) ---\n"
        f"{indent}if with_praat:\n"
        f"{indent}    df_person = merge_praat_person_features(df_person, praat_person_csv=r\"{PRAAT_PERSON_CSV_DEFAULT}\")\n"
        f"{indent}# --- END PATCH ---\n\n"
    )

    block2 = block[:r.start()] + injection + block[r.start():]
    return text[:start] + block2 + text[end:]


def main():
    if not SCRIPT_PATH.exists():
        raise SystemExit(f"Could not find: {SCRIPT_PATH}")

    orig = SCRIPT_PATH.read_text(encoding="utf-8")

    patched = insert_merge_helper(orig)
    patched = patch_build_person_table(patched)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = SCRIPT_PATH.with_suffix(SCRIPT_PATH.suffix + f".bak_{ts}")
    bak.write_text(orig, encoding="utf-8")

    SCRIPT_PATH.write_text(patched, encoding="utf-8")
    print(f"Patched OK. Backup saved to: {bak}")


if __name__ == "__main__":
    main()
