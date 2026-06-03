#!/usr/bin/env python3
"""GO/NO-GO gate entrypoint: CASCADE vs additive-linear and mean baselines.

Thin wrapper around :mod:`cascade.eval.baseline_gonogo` (the multi-seed gate),
kept for the ``experiments/`` convention. Writes results/baseline_gonogo.json
with a `go` boolean (CASCADE at least ties the baselines on a held-out split,
on every seed). This number gates the empirical sections of both papers.

Equivalent to: python -m cascade.eval.baseline_gonogo --seeds ...

Example:
    python experiments/run_baseline_gonogo.py --seeds 0 1 2
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from cascade.eval.baseline_gonogo import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
