# scripts/30_hierarchical_agg_vowels_pack.py
"""

"""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import numpy as np


VOWELS_PACK_TASKS = ["VA1", "VA2", "VE1", "VE2", "VI1", "VI2", "VO1", "VO2", "VU1", "VU2"]


def project_root() -> Path:
    # scripts/ -> projektrot
    return Path(__file__).resolve().parents[1]


def infer_base_task(df: pd.DataFrame) -> pd.Series:
    """
    Försök hitta base_task.
    Prioritet:
      1) om kolumn 'base_task' finns -> använd
      2) annars extrahera från 'task_code' (t.ex. VA1, VE2 ...) även om suffix finns
         (t.ex. "VA1G" -> "VA1")
    """
    if "base_task" in df.columns:
        return df["base_task"].astype(str)

    if "task_code" not in df.columns:
        raise ValueError(
            "Kunde inte hitta 'base_task' eller 'task_code' i features_vowels_pack_file.csv. "
            "Öppna filen och kolla vilka kolumnnamn som finns."
        )

    s = df["task_code"].astype(str)

    # Fångar VA1, VA2, VE1, VE2, VI1, VI2, VO1, VO2, VU1, VU2 var som helst i strängen
    pat = re.compile(r"(V[AEIOU]([12]))")
    extracted = s.apply(lambda x: (m.group(1) if (m := pat.search(x)) else np.nan))
    return extracted


def get_id_cols(df: pd.DataFrame) -> tuple[str, str]:
    if "speaker" not in df.columns:
        raise ValueError("Saknar kolumnen 'speaker' i features-filen.")
    if "label" not in df.columns:
        raise ValueError("Saknar kolumnen 'label' i features-filen.")
    return "speaker", "label"


def main() -> None:
    root = project_root()
    data_dir = root / "data_index"
    in_path = data_dir / "features_vowels_pack_file.csv"

    if not in_path.exists():
        raise FileNotFoundError(
            f"Hittar inte {in_path}. Kör script 26_extract_features_vowels_pack.py först."
        )

    df = pd.read_csv(in_path)
    speaker_col, label_col = get_id_cols(df)

    # Standardisera label om någon råkat bli konstig
    df[label_col] = df[label_col].astype(str).str.strip()

    # Derivera base_task och filtrera till vowels_pack
    df["base_task"] = infer_base_task(df)
    before = len(df)
    df = df[df["base_task"].isin(VOWELS_PACK_TASKS)].copy()
    after = len(df)

    # Feature-kolumner: alla numeriska utom id-kolumner och uppenbara meta-kolumner
    exclude = {
        speaker_col,
        label_col,
        "group",
        "path",
        "filename",
        "ext",
        "task_code",
        "base_task",
    }
    # välj numeriska kolumner (inkl float/int) som inte är exkluderade
    numeric_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    if len(numeric_cols) == 0:
        raise ValueError(
            "Hittade inga numeriska feature-kolumner att aggregera. "
            "Kolla features_vowels_pack_file.csv."
        )

    # 1) Fil -> (speaker, base_task): median
    # Behåll även counts för diagnostik
    g1 = df.groupby([speaker_col, label_col, "base_task"], as_index=False)

    # median för features
    speaker_task = g1[numeric_cols].median(numeric_only=True)
    # antal filer som ingick i varje (speaker, base_task)
    counts = g1.size().rename(columns={"size": "n_files_in_task"})
    speaker_task = speaker_task.merge(counts, on=[speaker_col, label_col, "base_task"], how="left")

    # 2) (speaker, base_task) -> speaker: mean över tasks (lika vikt per task)
    g2 = speaker_task.groupby([speaker_col, label_col], as_index=False)

    person = g2[numeric_cols].mean(numeric_only=True)

    # Lägg till diagnostik: antal tasks per speaker och vilka tasks som saknas
    task_counts = g2["base_task"].nunique().rename(columns={"base_task": "n_tasks_present"})
    task_list = g2["base_task"].apply(lambda x: ",".join(sorted(set(map(str, x))))).rename(columns={"base_task": "tasks_present"})
    person = person.merge(task_counts, on=[speaker_col, label_col], how="left")
    person = person.merge(task_list, on=[speaker_col, label_col], how="left")

    all_tasks = set(VOWELS_PACK_TASKS)
    person["n_tasks_missing"] = person["tasks_present"].apply(
        lambda s: len(all_tasks - set(s.split(","))) if isinstance(s, str) and s else len(all_tasks)
    )

    # Spara outputs
    out_speaker_task = data_dir / "features_vowels_pack_speaker_task_hierarchical.csv"
    out_person = data_dir / "features_vowels_pack_person_hierarchical.csv"

    speaker_task.to_csv(out_speaker_task, index=False)
    person.to_csv(out_person, index=False)

    # Print sammanfattning som du kan klistra in här
    n_speakers = person[speaker_col].nunique()
    label_counts = person[label_col].value_counts(dropna=False).to_dict()

    print("=== Hierarchical aggregation: vowels_pack ===")
    print(f"Input:  {in_path.name}")
    print(f"Rows before filter: {before}")
    print(f"Rows after filter (vowels_pack tasks): {after}")
    print(f"Feature columns aggregated: {len(numeric_cols)}")
    print(f"Speakers in output: {n_speakers}")
    print(f"Label counts (per speaker): {label_counts}")
    print(f"Saved: {out_speaker_task.name}  (rows={len(speaker_task)})")
    print(f"Saved: {out_person.name}        (rows={len(person)})")

    # Snabb check: hur många speakers saknar tasks?
    missing_stats = person["n_tasks_missing"].value_counts().sort_index()
    print("Tasks missing per speaker (count -> #speakers):")
    for k, v in missing_stats.items():
        print(f"  missing {int(k)} -> {int(v)}")

    # Visa topp 5 med flest missing
    worst = person.sort_values("n_tasks_missing", ascending=False).head(5)[
        [speaker_col, label_col, "n_tasks_present", "n_tasks_missing", "tasks_present"]
    ]
    print("Top 5 speakers with most missing tasks:")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
