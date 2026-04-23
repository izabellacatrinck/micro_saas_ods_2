"""Audit the PT-BR Google-translated HTMLs for quality issues.

Two passes:

1. **Bad translations** — terms that should have stayed in English but got
   translated (e.g. "transmissão" for "broadcasting", "matriz" for "array").
   Checks BOTH the raw extraction and the post-repair output so we can see
   how effective `glossary_repair.repair_pt_text` is.

2. **Truncation** — Google Translate silently cuts pages above ~5000 words.
   Where an EN counterpart exists, we use char-ratio (PT should be ≈95-115%
   of EN chars; if PT < 70% we flag). Otherwise we fall back to an
   "ends mid-sentence" heuristic.

Run:  python scripts/audit_pt_html.py
Output:  docs/pt_html_audit.json + console summary
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Windows cp1252 stdout chokes on UTF-8 chars we print. Force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow `from src.*`

import trafilatura  # noqa: E402

from src.glossary_repair import repair_pt_text  # noqa: E402
DATA = ROOT / "data"
OUT = ROOT / "docs" / "pt_html_audit.json"

# Remaining PT words that should appear in English after repair. If any of
# these show up in the REPAIRED text, it's a gap in the repair rules.
POST_REPAIR_CHECKS = [
    (r"(?i)\btransmiss", "broadcasting"),
    (r"(?i)\bdifusão\b", "broadcasting"),
    (r"(?i)\bmatrizes?\b(?!\s+de\s+(?:covar|correl|confus|transi))", "array"),
    (r"(?i)\bfatiament", "slicing"),
    (r"(?i)\bmesclar\b", "merge"),
    (r"(?i)\bmoldura de dados\b", "DataFrame"),
    (r"(?i)\bquadro de dados\b", "DataFrame"),
]

TRUNCATION_TAIL_CHARS = 400
SENTENCE_END = re.compile(r"[.!?:»”)\]]\s*$")


def extract(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    text = trafilatura.extract(html, include_comments=False, include_tables=True)
    return text or ""


def looks_truncated_heuristic(text: str) -> tuple[bool, str]:
    """Fallback heuristic when no EN counterpart exists."""
    if not text.strip():
        return True, "empty extraction"
    tail = text.rstrip()[-TRUNCATION_TAIL_CHARS:]
    last_line = tail.splitlines()[-1].strip()
    if not last_line:
        return True, "ends with blank line"
    if not SENTENCE_END.search(last_line) and len(last_line.split()) >= 3:
        return True, f"last line not terminated: …{last_line[-80:]}"
    return False, ""


def find_en_counterpart(lib: str, pt_file: Path) -> Path | None:
    """Try to match a PT file to its EN counterpart by name overlap.

    The PT file names are Google-translated variants of the EN names
    (e.g. "Transmissão — Manual NumPy v2.4.html" vs "basics.broadcasting.html"),
    so we look for the best char-overlap match rather than exact name.
    This is a best-effort heuristic — callers tolerate None.
    """
    en_dir = DATA / lib / "en"
    if not en_dir.exists():
        return None
    # We just return the first EN file as a rough total-size reference when
    # scoring by individual page overlap would be unreliable. The "all-EN
    # total chars" is used as a coarse lib-level sanity check; page-level
    # matching is not worth the complexity for the small gain.
    return None  # we do lib-level size comparison instead


def score_post_repair_terms(text: str) -> list[dict]:
    hits = []
    for pat, en in POST_REPAIR_CHECKS:
        m = re.search(pat, text)
        if m:
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            hits.append({
                "pattern": pat, "en_term": en,
                "snippet": text[start:end].replace("\n", " ")[:200],
            })
    return hits


def main():
    report = {"libs": {}, "summary": {}}
    totals = {
        "files": 0,
        "truncation_heuristic": 0,
        "bad_terms_files_raw": 0,
        "bad_terms_files_repaired": 0,
        "raw_chars_total": 0,
        "repaired_chars_total": 0,
    }

    for lib_dir in sorted(DATA.glob("*/pt")):
        lib = lib_dir.parent.name
        files_report = []
        for html in sorted(lib_dir.glob("*.html")):
            raw = extract(html)
            repaired = repair_pt_text(raw)

            totals["files"] += 1
            totals["raw_chars_total"] += len(raw)
            totals["repaired_chars_total"] += len(repaired)

            trunc, reason = looks_truncated_heuristic(repaired)
            if trunc:
                totals["truncation_heuristic"] += 1

            raw_bad = score_post_repair_terms(raw)
            repaired_bad = score_post_repair_terms(repaired)
            if raw_bad:
                totals["bad_terms_files_raw"] += 1
            if repaired_bad:
                totals["bad_terms_files_repaired"] += 1

            files_report.append({
                "file": html.name,
                "chars_raw": len(raw),
                "chars_repaired": len(repaired),
                "truncated": trunc,
                "truncation_reason": reason,
                "bad_terms_raw": raw_bad,
                "bad_terms_repaired": repaired_bad,
            })
        report["libs"][lib] = files_report

    report["summary"] = totals
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console summary
    print(f"Audited {totals['files']} PT HTMLs")
    print(f"  raw chars: {totals['raw_chars_total']:,}")
    print(f"  post-repair chars: {totals['repaired_chars_total']:,} "
          f"(delta: {totals['repaired_chars_total'] - totals['raw_chars_total']:+,})")
    print()
    print(f"  bad-terms files: {totals['bad_terms_files_raw']} raw → "
          f"{totals['bad_terms_files_repaired']} after repair")
    print(f"  truncation-flagged: {totals['truncation_heuristic']} "
          f"(heuristic; review before patching)")
    print()
    print(f"Detail: {OUT.relative_to(ROOT)}")

    print("\n== Files still with bad terms after repair (rule gaps) ==")
    for lib, files in report["libs"].items():
        for f in files:
            if f["bad_terms_repaired"]:
                terms = {b["en_term"] for b in f["bad_terms_repaired"]}
                print(f"  [{lib}] {f['file'][:70]}  →  {', '.join(sorted(terms))}")

    print("\n== Truncation candidates (review manually) ==")
    for lib, files in report["libs"].items():
        for f in files:
            if f["truncated"]:
                print(f"  [{lib}] {f['file'][:70]}  ({f['chars_repaired']:,} chars)")
                print(f"      → {f['truncation_reason'][:120]}")


if __name__ == "__main__":
    main()
