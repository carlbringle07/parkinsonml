from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Ändra denna till DIN rotmapp (mappen som innehåller de tre gruppmapparna)
DATA_ROOT = Path(r"C:\Users\ah87324\Downloads\GA\Röstinspelningar")  # <-- ändra vid behov

# Vilka ljudformat vi accepterar
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

# Grupp -> label
GROUP_TO_LABEL = {
    "15 Young Healthy Control": "control_young",
    "22 Elderly Healthy Control": "control_elderly",
    "28 People with Parkinson's disease": "pd",
}

# Taskkod sitter i början av filnamnet: t.ex. VA1..., B1L..., PR1...
TASK_RE = re.compile(r"^([A-Za-z]{1,3}\d{1,2}[A-Za-z]?)")

def extract_task_code(stem: str) -> str | None:
    m = TASK_RE.match(stem)
    return m.group(1).upper() if m else None

def main() -> None:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Hittar inte DATA_ROOT: {DATA_ROOT}")

    rows = []

    group_dirs = [p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name in GROUP_TO_LABEL]
    if not group_dirs:
        raise RuntimeError(
            "Hittade inga gruppmappar. Kolla att DATA_ROOT pekar på mappen som innehåller "
            "'15 Young Healthy Control', '22 Elderly Healthy Control', '28 People with Parkinson's disease'."
        )

    for group_dir in group_dirs:
        label = GROUP_TO_LABEL[group_dir.name]

        # Rekursivt: funkar även om PD har extra undermappar (1-5, 6-10, ...)
        all_files = [p for p in group_dir.rglob("*") if p.is_file()]

        for f in tqdm(all_files, desc=f"Scanning {group_dir.name}"):
            ext = f.suffix.lower()

            # Hoppa över textfiler etc.
            if ext and ext not in AUDIO_EXTS:
                continue

            # Speaker = närmaste mapp ovanför filen (personmappen)
            speaker = f.parent.name

            stem = f.stem
            task_code = extract_task_code(stem)

            rows.append({
                "path": str(f),
                "group": group_dir.name,
                "label": label,
                "speaker": speaker,
                "filename": f.name,
                "ext": ext if ext else "",
                "task_code": task_code if task_code else "",
            })

    df = pd.DataFrame(rows)
    out_dir = Path("data_index")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "file_index.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print("\n✅ Skapade:", out_path)
    print("Antal filer (alla ljudfiler som hittades):", len(df))

    print("\n--- Per label ---")
    print(df["label"].value_counts(dropna=False))

    print("\n--- Vanligaste task_code ---")
    print(df["task_code"].value_counts().head(20))

    no_ext = (df["ext"] == "").sum()
    no_task = (df["task_code"] == "").sum()
    print(f"\n⚠️ Filer utan ändelse: {no_ext}")
    print(f"⚠️ Filer utan task_code: {no_task}")

if __name__ == "__main__":
    main()
