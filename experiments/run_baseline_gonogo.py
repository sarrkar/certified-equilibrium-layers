#!/usr/bin/env python3
"""GO/NO-GO gate entrypoint: CASCADE vs additive-linear and mean baselines.

Writes results/baseline_gonogo.json with a `go` boolean (CASCADE at least ties
the baselines on a held-out split). This number gates the empirical sections of
both papers. Accepts --seed and logs it (CLAUDE.md rule 3).

Example:
    python experiments/run_baseline_gonogo.py --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from cascade.eval.gonogo import run_gonogo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (logged)")
    parser.add_argument("--out-dir", type=Path, default=REPO / "results")
    args = parser.parse_args()

    print(f"[gonogo] seed={args.seed} out_dir={args.out_dir}")
    result = run_gonogo(seed=args.seed, out_dir=args.out_dir)
    # Non-zero exit on NO-GO so CI / make can gate on it.
    return 0 if result["go"] else 1


if __name__ == "__main__":
    sys.exit(main())
