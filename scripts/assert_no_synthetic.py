#!/usr/bin/env python3
"""Fail if synthetic-data generation appears on any AAAI reported-number path.

Per the hard data policy in OUTLINE.md, no synthetic data may live in
``src/cascade/data/`` or ``experiments/aaai/`` (the loaders and the experiments
that feed reported AAAI numbers). This script greps those paths for
synthetic-generation patterns and exits non-zero if any appear.

Scanned paths:
- ``src/cascade/data/`` (the real-data loaders)
- ``experiments/aaai/`` (the real-data experiments and the aggregator)

Flagged patterns (case-insensitive):
- random generation used to build features / labels / edges: ``np.random``,
  ``default_rng``, ``standard_normal``, ``rng.uniform``, ``rng.integers``,
  ``rng.choice``, ``rng.random``,
- the words ``synthetic``, ``plant``, ``proxy``, ``toy``.

A line is exempt only if it carries the marker ``# real-data-policy: allow``
(for the rare legitimate mention, e.g. an error message that names the policy).
The honest disclaimer "does not fabricate baseline numbers" is intentionally NOT
flagged: only data-construction patterns are.
Synthetic CERTIFICATE-SOUNDNESS verification lives under ``experiments/soundness/``
and is intentionally NOT scanned: it verifies the math on random graphs and is
never a reported biological number.

Usage: ``python scripts/assert_no_synthetic.py`` (exit 0 clean, 1 on any hit).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCAN_DIRS = [REPO / "src" / "cascade" / "data", REPO / "experiments" / "aaai"]

PATTERNS = [
    r"np\.random",
    r"default_rng",
    r"standard_normal",
    r"rng\.uniform",
    r"rng\.integers",
    r"rng\.choice",
    r"rng\.random",
    r"\bsynthetic\b",
    r"\bplant\b",
    r"\bproxy\b",
    r"\btoy\b",
]
_RE = re.compile("|".join(PATTERNS), re.IGNORECASE)
ALLOW_MARKER = "# real-data-policy: allow"


def scan_file(path: Path) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return hits
    for lineno, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line:
            continue
        if _RE.search(line):
            hits.append((lineno, line.strip()))
    return hits


def main() -> int:
    all_hits: list[tuple[Path, int, str]] = []
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for lineno, line in scan_file(path):
                all_hits.append((path.relative_to(REPO), lineno, line))

    if all_hits:
        print("HARD DATA POLICY VIOLATION: synthetic-generation patterns on a reported-number path")
        for path, lineno, line in all_hits:
            print(f"  {path}:{lineno}: {line}")
        print(
            f"\n{len(all_hits)} violation(s). Move synthetic verification to experiments/soundness/"
        )
        print(
            "or load real data and RAISE FileNotFoundError / NotImplementedError when it is absent."
        )
        return 1
    print("OK: no synthetic-generation patterns in src/cascade/data/ or experiments/aaai/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
