# scripts/43_patch_fix_praat_sanity_check.py
from __future__ import annotations
import re
from pathlib import Path

TARGET = Path("scripts/43_robustness_conformal_screening_vowels_pack_baseline_vs_praat.py")

REPL = r"""
    # Fail-fast / warn if Praat features look empty or constant.
    # NOTE: Ignore metadata columns like praat_praat_ok_rate / praat_label / praat_n_files.
    praat_feat_cols = [
        c for c in praat_cols
        if any(k in c.lower() for k in ["f0", "hnr", "jitter", "shimmer"])
    ]
    if len(praat_feat_cols) > 0:
        # If any praat feature column is all-NaN or constant -> suspicious
        nan_max = float(out[praat_feat_cols].isna().mean().max())
        nun = out[praat_feat_cols].nunique(dropna=False).sort_values()
        if nan_max > 0.0 or int(nun.min()) <= 1:
            print(
                "[WARN] Praat sanity-check triggade (endast riktiga praat-features). "
                "Om praat_feat_cols är NaN/konstanta kommer +Praat ≈ baseline.\n"
                f"nan_max={nan_max}\n"
                "nunique head:\n" + str(nun.head(20))
            )
    else:
        print("[WARN] Inga praat_feat_cols hittades (f0/hnr/jitter/shimmer). +Praat kan bli ≈ baseline.")
"""

def main():
    if not TARGET.exists():
        raise SystemExit(f"Hittar inte {TARGET.as_posix()}")

    txt = TARGET.read_text(encoding="utf-8")

    # Replace the existing sanity-check block: from the comment line to the 'return out'
    pattern = re.compile(
        r"""
(?P<indent>\s*)#\s*Fail-fast\s*if\s*constant.*?\n   # start marker
.*?\n
(?P=indent)return\s+out\s*\n
""",
        re.VERBOSE | re.DOTALL,
    )

    m = pattern.search(txt)
    if not m:
        raise SystemExit(
            "Kunde inte hitta sanity-check-blocket att ersätta. "
            "Sök i filen efter raden '# Fail-fast if constant' och kolla att den finns."
        )

    start, end = m.span()
    indent = m.group("indent") or ""
    # Ensure replacement is indented like original block
    rep_lines = [(indent + line if line.strip() else line) for line in REPL.strip("\n").splitlines()]
    rep = "\n".join(rep_lines) + "\n\n" + indent + "return out\n"

    new_txt = txt[:start] + rep + txt[end:]

    # Backup then write
    backup = TARGET.with_suffix(TARGET.suffix + ".bak_before_fix_sanity")
    backup.write_text(txt, encoding="utf-8")
    TARGET.write_text(new_txt, encoding="utf-8")

    print("Patch: sanity-check uppdaterad så den bara kollar f0/hnr/jitter/shimmer.")
    print(f"Backup sparad: {backup.as_posix()}")
    print(f"Patchad fil : {TARGET.as_posix()}")

if __name__ == "__main__":
    main()
