#!/usr/bin/env python3
"""Forbid em-dashes (U+2014) and en-dashes-as-em (U+2013 between words).

Per OUTLINE.md rule 4, em-dashes are not allowed anywhere in this repo.
This script is wired into pre-commit and CI. It exits non-zero on the first
hit and prints file:line for each offense.
"""

from __future__ import annotations

import sys
from pathlib import Path

FORBIDDEN = {
    "—": "em-dash (U+2014)",
    "―": "horizontal bar (U+2015)",
}


def scan(path: Path) -> list[tuple[int, str, str]]:
    hits: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return hits
    for lineno, line in enumerate(text.splitlines(), start=1):
        for ch, label in FORBIDDEN.items():
            if ch in line:
                hits.append((lineno, label, line.rstrip()))
                break
    return hits


def main(argv: list[str]) -> int:
    rc = 0
    for arg in argv:
        p = Path(arg)
        if not p.is_file():
            continue
        # Skip self so the literal forbidden-list does not trip the hook.
        if p.resolve() == Path(__file__).resolve():
            continue
        for lineno, label, line in scan(p):
            print(f"{p}:{lineno}: {label}: {line}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
