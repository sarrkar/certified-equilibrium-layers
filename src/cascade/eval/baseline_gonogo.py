"""Multi-seed GO/NO-GO gate entrypoint.

Runs the CASCADE-vs-baselines gate (:mod:`cascade.eval.gonogo`) across one or
more seeds and writes a single aggregated ``results/baseline_gonogo.json``. The
aggregate verdict is conservative: GO only when CASCADE at least ties the best
baseline on every seed. This number gates the empirical sections of both papers
(CLAUDE.md "Empirical gate").

Run:
    python -m cascade.eval.baseline_gonogo --seeds 0 1 2

Each seed is logged (CLAUDE.md rule 3). Until the real GSE184929 / STRING join
is wired (a tracked data gap), the gate runs on the synthetic propagation proxy
and records that provenance in the JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cascade.eval.gonogo import TIE_BAND, run_gonogo


def aggregate_gonogo(seeds: list[int], out_dir: str | Path) -> dict[str, Any]:
    """Run the gate per seed and write the aggregated verdict to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_gonogo_run"

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[gonogo] === seed {seed} ===")
        per_seed.append(run_gonogo(seed=seed, out_dir=scratch, run_dir=scratch / f"seed_{seed}"))

    cascade_mses = [r["cascade"]["test_mse"] for r in per_seed]
    baseline_mses = [r["best_baseline_test_mse"] for r in per_seed]
    n_go = sum(bool(r["go"]) for r in per_seed)

    aggregate = {
        "go": bool(n_go == len(seeds)),
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "n_go": n_go,
        "tie_band": TIE_BAND,
        "metric": "test_mse (lower is better), aggregated over seeds",
        "cascade_test_mse_mean": float(np.mean(cascade_mses)),
        "cascade_test_mse_std": float(np.std(cascade_mses)),
        "best_baseline_test_mse_mean": float(np.mean(baseline_mses)),
        "beats_all_baselines_every_seed": bool(all(r["beats_best_baseline"] for r in per_seed)),
        "dataset_provenance": per_seed[0]["dataset_provenance"],
        "per_seed": per_seed,
    }

    out_path = out_dir / "baseline_gonogo.json"
    out_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    verdict = "GO" if aggregate["go"] else "NO-GO"
    print(
        f"[gonogo] AGGREGATE {verdict}: {n_go}/{len(seeds)} seeds GO; "
        f"cascade_test_mse={aggregate['cascade_test_mse_mean']:.5f} "
        f"+/- {aggregate['cascade_test_mse_std']:.5f} vs best baseline "
        f"{aggregate['best_baseline_test_mse_mean']:.5f}; wrote {out_path}"
    )
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0], help="one or more RNG seeds (logged)"
    )
    repo_results = Path(__file__).resolve().parents[3] / "results"
    parser.add_argument("--out-dir", type=Path, default=repo_results)
    args = parser.parse_args(argv)

    print(f"[gonogo] seeds={args.seeds} out_dir={args.out_dir}")
    result = aggregate_gonogo(args.seeds, args.out_dir)
    # Non-zero exit on NO-GO so CI / make can gate on it.
    return 0 if result["go"] else 1


if __name__ == "__main__":
    sys.exit(main())
