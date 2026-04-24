"""Measure how much content each truncation candidate actually lost.

For each PT file flagged as truncated, try to find an EN counterpart in the
same library's /en dir by best-effort name matching, then report
  - PT chars (post-repair)
  - EN chars
  - ratio
  - verdict: severe (<60%), moderate (60-80%), false-positive (>80%)

Also reports the last 200 chars of the PT extract so we can see where it ends.
"""
from __future__ import annotations

import difflib
import json
import sys
import unicodedata
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import trafilatura  # noqa: E402
from src.glossary_repair import repair_pt_text  # noqa: E402

DATA = ROOT / "data"
AUDIT = ROOT / "docs" / "pt_html_audit.json"


def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    # strip common boilerplate
    for junk in [".html", "manual do numpy", "manual numpy", "documentacao matplotlib",
                 "documentacao seaborn", "documentacao do pandas", "documentacao pandas",
                 "v2.4", "3.10.8", "0.13.2", "3.0.2", "  ", "—", "-", "_", " "]:
        s = s.replace(junk, "")
    return s


def find_en_counterpart(lib: str, pt_filename: str) -> Path | None:
    en_dir = DATA / lib / "en"
    if not en_dir.exists():
        return None
    en_files = list(en_dir.glob("*.html"))
    if not en_files:
        return None
    pt_key = normalize_name(pt_filename)
    best = None
    best_score = 0.0
    for en in en_files:
        en_key = normalize_name(en.name)
        score = difflib.SequenceMatcher(None, pt_key, en_key).ratio()
        if score > best_score:
            best_score = score
            best = en
    # Require at least 0.5 overlap to call it a match.
    return best if best_score >= 0.5 else None


def extract_text(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    return trafilatura.extract(html, include_comments=False, include_tables=True) or ""


def main():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    rows = []
    for lib, files in audit["libs"].items():
        for f in files:
            if not f["truncated"]:
                continue
            pt_path = DATA / lib / "pt" / f["file"]
            pt_text = repair_pt_text(extract_text(pt_path))
            en_path = find_en_counterpart(lib, f["file"])
            en_chars = len(extract_text(en_path)) if en_path else None
            ratio = (len(pt_text) / en_chars) if en_chars else None
            if ratio is None:
                verdict = "no-en-counterpart"
            elif ratio < 0.60:
                verdict = "SEVERE"
            elif ratio < 0.80:
                verdict = "moderate"
            elif ratio < 1.30:
                verdict = "false-positive (pt ~ en)"
            else:
                verdict = "pt-much-larger?"
            rows.append({
                "lib": lib,
                "pt": f["file"],
                "pt_chars": len(pt_text),
                "en_match": en_path.name if en_path else None,
                "en_chars": en_chars,
                "ratio": round(ratio, 2) if ratio else None,
                "verdict": verdict,
                "pt_tail": pt_text[-200:].replace("\n", " "),
            })
    rows.sort(key=lambda r: (r["ratio"] if r["ratio"] is not None else 99, -r["pt_chars"]))

    # Print report
    print(f"{'verdict':<28} {'lib':<10} {'ratio':>6} {'pt':>7} {'en':>7}  file")
    print("-" * 120)
    for r in rows:
        ratio = f"{r['ratio']:.2f}" if r["ratio"] is not None else "   - "
        en = f"{r['en_chars']:,}" if r["en_chars"] else "   -"
        pt = f"{r['pt_chars']:,}"
        print(f"{r['verdict']:<28} {r['lib']:<10} {ratio:>6} {pt:>7} {en:>7}  {r['pt'][:60]}")

    out = ROOT / "docs" / "truncation_measure.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetail: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
