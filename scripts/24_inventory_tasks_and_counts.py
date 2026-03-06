# FILE: scripts/24_inventory_tasks_and_counts.py

from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_INDEX = ROOT / "data_index"

FILE_INDEX = DATA_INDEX / "file_index.csv"

def main():
    if not FILE_INDEX.exists():
        raise FileNotFoundError(f"Missing: {FILE_INDEX} (run scripts/00_build_index.py first)")

    df = pd.read_csv(FILE_INDEX)
    for c in ["task_code", "group", "label", "speaker"]:
        if c not in df.columns:
            raise RuntimeError(f"file_index.csv missing column: {c}")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["speaker"] = df["speaker"].astype(str)

    # Focus on pd vs control_elderly only
    df = df[df["label"].isin(["pd", "control_elderly"])].copy()

    # Overall by task_code
    task_file_counts = df.groupby("task_code").size().sort_values(ascending=False)
    task_speaker_counts = df.groupby("task_code")["speaker"].nunique().sort_values(ascending=False)

    # PD vs CTRL speakers per task
    speaker_by_task_label = (
        df.groupby(["task_code", "label"])["speaker"]
        .nunique()
        .unstack(fill_value=0)
        .sort_values(by=["pd", "control_elderly"], ascending=False)
    )

    print("=== TASK INVENTORY (pd vs control_elderly) ===\n")
    print("Files per task_code:")
    print(task_file_counts.to_string())
    print("\nSpeakers per task_code:")
    print(task_speaker_counts.to_string())
    print("\nSpeakers per task_code split by label:")
    print(speaker_by_task_label.to_string())

if __name__ == "__main__":
    main()
