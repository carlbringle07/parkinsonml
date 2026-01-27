# scripts/31c_debug_person_table_diffs.py
from __future__ import annotations
import numpy as np
import pandas as pd

base = pd.read_csv("data_index/features_vowels_pack_person.csv")
hier = pd.read_csv("data_index/features_vowels_pack_person_hierarchical.csv")

base = base.sort_values(["speaker","label"]).reset_index(drop=True)
hier = hier.sort_values(["speaker","label"]).reset_index(drop=True)

# sanity: samma keys?
keys_ok = (base[["speaker","label"]].equals(hier[["speaker","label"]]))
print("keys_match:", keys_ok)
if not keys_ok:
    # visa första mismatch
    merged = base[["speaker","label"]].merge(hier[["speaker","label"]], how="outer", indicator=True)
    print(merged["_merge"].value_counts())
    print("Example rows not matching:")
    print(merged[merged["_merge"]!="both"].head(10).to_string(index=False))
    raise SystemExit(1)

exclude = {"speaker","label","group","tasks_present","n_tasks_present","n_tasks_missing"}
feat_cols = [
    c for c in base.columns
    if c in hier.columns and c not in exclude and pd.api.types.is_numeric_dtype(base[c])
]

# per-column max abs diff
diffs = {}
for c in feat_cols:
    d = base[c].to_numpy() - hier[c].to_numpy()
    diffs[c] = float(np.nanmax(np.abs(d)))

s = pd.Series(diffs).sort_values(ascending=False)

print("\nTop 10 columns by max abs diff:")
print(s.head(10).to_string())

# visa en rad där största diff uppstår
top_col = s.index[0]
d = np.abs(base[top_col].to_numpy() - hier[top_col].to_numpy())
i = int(np.nanargmax(d))

print("\nWorst row example:")
print("column:", top_col)
print("speaker:", base.loc[i, "speaker"], "label:", base.loc[i, "label"])
print("baseline value:", base.loc[i, top_col])
print("hier value:", hier.loc[i, top_col])
print("abs diff:", d[i])

# extra: sammanfattning
print("\nSummary:")
print("n_features:", len(feat_cols))
print("max_abs_diff_overall:", float(s.iloc[0]))
print("n_cols_with_diff>1e-12:", int((s > 1e-12).sum()))
