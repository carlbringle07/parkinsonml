# scripts/33b_rebuild_features_vowels_pack_person_plus_praat.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_index"

BASE_PERSON = DATA / "features_vowels_pack_person.csv"
PRAAT_PERSON = DATA / "praat_vowels_pack_person.csv"
OUT = DATA / "features_vowels_pack_person_plus_praat.csv"

def main():
    df_base = pd.read_csv(BASE_PERSON)
    df_praat = pd.read_csv(PRAAT_PERSON)

    if "speaker" not in df_base.columns or "speaker" not in df_praat.columns:
        raise ValueError("Både baseline och praat måste ha kolumnen 'speaker'.")

    # välj praatkolumner (i din fil heter de f0_*, hnr_*, jitter_*, shimmer_*)
    praat_cols = [c for c in df_praat.columns if any(
        key in c.lower() for key in ["f0_", "hnr", "jitter", "shimmer"]
    )]

    keep = ["speaker"] + (["label"] if "label" in df_praat.columns else []) + praat_cols
    df_praat2 = df_praat[keep].copy()

    print("baseline rows:", len(df_base), "cols:", df_base.shape[1])
    print("praat rows   :", len(df_praat2), "praat_cols:", len(praat_cols))
    if len(praat_cols) == 0:
        raise RuntimeError("Inga praat-kolumner hittades i praat_vowels_pack_person.csv.")

    # merge på speaker
    df_out = df_base.merge(df_praat2, on="speaker", how="left", suffixes=("", "_praatdup"))

    # om label finns i båda, behåll baseline-label
    if "label_praatdup" in df_out.columns:
        df_out.drop(columns=["label_praatdup"], inplace=True)

    # sanity: nunique på praat-features (ska vara >1 för de flesta)
    nunq = df_out[praat_cols].nunique(dropna=False).sort_values()
    print("Praat nunique (min 12):")
    print(nunq.head(12))

    df_out.to_csv(OUT, index=False)
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
