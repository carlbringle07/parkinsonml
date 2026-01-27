# scripts/31b_check_person_tables_identical.py
from __future__ import annotations

import numpy as np
import pandas as pd

base = pd.read_csv("data_index/features_vowels_pack_person.csv")
hier = pd.read_csv("data_index/features_vowels_pack_person_hierarchical.csv")

base = base.sort_values(["speaker", "label"]).reset_index(drop=True)
hier = hier.sort_values(["speaker", "label"]).reset_index(drop=True)

exclude = {"speaker", "label", "group", "tasks_present", "n_tasks_present", "n_tasks_missing"}
feat_cols = [
    c for c in base.columns
    if c in hier.columns
    and c not in exclude
    and pd.api.types.is_numeric_dtype(base[c])
]

diff = base[feat_cols].to_numpy() - hier[feat_cols].to_numpy()
max_abs = float(np.nanmax(np.abs(diff)))
n_nonzero = int(np.sum(np.abs(diff) > 1e-12))

print("=== Person-table diff check ===")
print("n_features:", len(feat_cols))
print("max_abs_diff:", max_abs)
print("count_abs_diff_gt_1e-12:", n_nonzero)
