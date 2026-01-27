# scripts/36_export_results_34_35_to_index.py
from __future__ import annotations

import json
import os
import glob
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

import pandas as pd


DATA_INDEX = "data_index"


def _pick_latest(patterns: List[str]) -> Optional[str]:
    """Pick newest file matching any glob pattern."""
    cands: List[str] = []
    for pat in patterns:
        cands.extend(glob.glob(pat))
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        if x is None:
            return "NA"
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


@dataclass
class ResultRow:
    source: str               # "script34" / "script35"
    name: str                 # e.g. "baseline", "baseline_plus_praat", "conformal_plus_praat"
    metric: str               # e.g. "auc_mean"
    value: float | None
    extra: Dict[str, Any]


def export_34(eval_csv_path: str) -> List[ResultRow]:
    df = pd.read_csv(eval_csv_path)

    # Normalisera kolumnnamn vi förväntar oss
    # (script 34 brukar ha: variant, auc_mean, auc_std, auc_min, auc_max, n_features, n_speakers, n_pd, n_ctrl, test_size, n_splits)
    rows: List[ResultRow] = []

    for _, r in df.iterrows():
        variant = str(r.get("variant", "unknown"))
        extra = {
            "n_splits": r.get("n_splits"),
            "test_size": r.get("test_size"),
            "n_speakers": r.get("n_speakers"),
            "n_pd": r.get("n_pd"),
            "n_ctrl": r.get("n_ctrl"),
            "n_features": r.get("n_features"),
            "auc_std": r.get("auc_std"),
            "auc_min": r.get("auc_min"),
            "auc_max": r.get("auc_max"),
        }
        rows.append(ResultRow("script34", variant, "auc_mean", r.get("auc_mean"), extra))

        # Om delta-kolumn finns, spara den också
        for k in ["auc_mean_delta_vs_baseline", "auc_mean_delta_v", "auc_delta", "auc_mean_delta"]:
            if k in df.columns:
                rows.append(ResultRow("script34", variant, k, r.get(k), {"note": "delta vs baseline if applicable"}))
                break

    return rows


def export_35(screen_csv_path: str) -> List[ResultRow]:
    df = pd.read_csv(screen_csv_path)

    # script 35 brukar ha: variant, alpha, auc_mean, auc_std, coverage_mean, decided_rate_mean, unsure_rate_mean, acc_decided_mean + std-kolumner
    rows: List[ResultRow] = []

    for _, r in df.iterrows():
        variant = str(r.get("variant", "unknown"))
        alpha = r.get("alpha", None)
        extra_common = {"alpha": alpha}

        metrics = [
            "auc_mean",
            "auc_std",
            "coverage_mean",
            "coverage_std",
            "decided_rate_mean",
            "decided_rate_std",
            "unsure_rate_mean",
            "unsure_rate_std",
            "acc_decided_mean",
            "acc_decided_std",
        ]

        for m in metrics:
            if m in df.columns:
                rows.append(ResultRow("script35", variant, m, r.get(m), dict(extra_common)))

    return rows


def write_index(rows: List[ResultRow], out_csv: str, out_txt: str, out_json: str) -> None:
    # CSV
    flat = []
    for rr in rows:
        d = {
            "source": rr.source,
            "name": rr.name,
            "metric": rr.metric,
            "value": rr.value,
        }
        # Flatta extra
        for k, v in (rr.extra or {}).items():
            d[f"extra_{k}"] = v
        flat.append(d)

    out_df = pd.DataFrame(flat)
    out_df.to_csv(out_csv, index=False)

    # TXT (lättläst)
    lines = []
    lines.append("=== RESULTS INDEX (scripts 34 & 35) ===")
    lines.append("")
    lines.append(f"Saved CSV : {out_csv}")
    lines.append(f"Saved TXT : {out_txt}")
    lines.append(f"Saved JSON: {out_json}")
    lines.append("")

    # Gruppvis print
    for src in ["script34", "script35"]:
        sub = out_df[out_df["source"] == src].copy()
        if sub.empty:
            continue
        lines.append(f"--- {src.upper()} ---")
        for name in sorted(sub["name"].unique()):
            lines.append(f"* {name}")
            s2 = sub[sub["name"] == name]
            # sortera lite: auc först
            order = ["auc_mean", "auc_std", "auc_min", "auc_max", "coverage_mean", "decided_rate_mean", "unsure_rate_mean", "acc_decided_mean"]
            s2["_ord"] = s2["metric"].apply(lambda x: order.index(x) if x in order else 999)
            s2 = s2.sort_values(["_ord", "metric"])
            for _, r in s2.iterrows():
                lines.append(f"  - {r['metric']}: {_fmt(r['value'])}")
            lines.append("")
        lines.append("")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")

    # JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)


def main() -> None:
    # Hitta senaste output för 34 och 35
    f34 = _pick_latest([
        os.path.join(DATA_INDEX, "eval_repeated_splits_vowels_pack_baseline_vs_praat*.csv"),
        os.path.join(DATA_INDEX, "eval_repeated_splits_*baseline_vs_praat*.csv"),
        os.path.join(DATA_INDEX, "eval_repeated_splits_*praat*.csv"),
    ])

    f35 = _pick_latest([
        os.path.join(DATA_INDEX, "screening_summary_conformal_vowels_pack_plus_praat*.csv"),
        os.path.join(DATA_INDEX, "screening_summary_conformal_*plus_praat*.csv"),
        os.path.join(DATA_INDEX, "screening_summary_conformal_*praat*.csv"),
    ])

    if not f34:
        raise FileNotFoundError("Could not find script 34 output CSV in data_index (eval_repeated_splits_*praat*.csv).")
    if not f35:
        raise FileNotFoundError("Could not find script 35 output CSV in data_index (screening_summary_conformal_*praat*.csv).")

    rows: List[ResultRow] = []
    rows.extend(export_34(f34))
    rows.extend(export_35(f35))

    out_csv = os.path.join(DATA_INDEX, "results_index_34_35.csv")
    out_txt = os.path.join(DATA_INDEX, "results_index_34_35.txt")
    out_json = os.path.join(DATA_INDEX, "results_index_34_35.json")

    write_index(rows, out_csv, out_txt, out_json)

    print("=== KLART ===")
    print("Input 34:", f34)
    print("Input 35:", f35)
    print("Saved:", out_csv)
    print("Saved:", out_txt)
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
