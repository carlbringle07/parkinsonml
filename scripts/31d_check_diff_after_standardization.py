# scripts/31d_check_diff_after_standardization.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

base = pd.read_csv("data_index/features_vowels_pack_person.csv").sort_values(["speaker","label"]).reset_index(drop=True)
hier = pd.read_csv("data_index/features_vowels_pack_person_hierarchical.csv").sort_values(["speaker","label"]).reset_index(drop=True)

assert base[["speaker","label"]].equals(hier[["speaker","label"]]), "Keys mismatch"

exclude = {"speaker","label","group","tasks_present","n_tasks_present","n_tasks_missing"}
feat_cols = [
    c for c in base.columns
    if c in hier.columns and c not in exclude and pd.api.types.is_numeric_dtype(base[c])
]

Xb = base[feat_cols].to_numpy(float)
Xh = hier[feat_cols].to_numpy(float)

# Standardisera separat (som i pipeline, fit på train brukar göras)
# För att bara se om de är "samma upp till shift/scale" gör vi här fit på ALLA.
sb = StandardScaler().fit(Xb)
sh = StandardScaler().fit(Xh)

Zb = sb.transform(Xb)
Zh = sh.transform(Xh)

diffZ = Zb - Zh
max_abs = float(np.nanmax(np.abs(diffZ)))
mean_abs = float(np.nanmean(np.abs(diffZ)))
n_big = int(np.sum(np.abs(diffZ) > 1e-6))

print("=== Diff AFTER standardization (z-scores) ===")
print("n_features:", len(feat_cols))
print("max_abs_diff_z:", max_abs)
print("mean_abs_diff_z:", mean_abs)
print("count_abs_diff_z_gt_1e-6:", n_big)
