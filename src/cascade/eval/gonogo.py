"""GO/NO-GO empirical gate: CASCADE vs the additive-linear and mean baselines.

Runs the CASCADE off-target model and the two baselines on the SAME held-out
split (built deterministically from the same seed) and decides GO when CASCADE
at least ties the best baseline test MSE within a small tie band. This number
gates the empirical sections of both papers (CLAUDE.md "Empirical gate").

Until the real GSE184929 / STRING join is wired (a tracked data gap), the gate
runs on the synthetic propagation proxy and records that provenance in the JSON.
torch is imported lazily inside :func:`run_gonogo` so this module imports in the
torch-free CI image; the baseline halves stay numpy.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cascade.baselines.linear import (
    AdditiveLinearBaseline,
    MeanBaseline,
    regression_metrics,
)
from cascade.synthetic_dataset import build_cascade_data

# CASCADE counts as tying a baseline if its test MSE is within this relative band.
TIE_BAND = 0.02


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats with None so the output is valid JSON."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def default_gonogo_cfg(seed: int = 0) -> dict[str, Any]:
    """Config for the gate: a synthetic propagation proxy with a real held-out set."""
    return {
        "seed": seed,
        "device": "cpu",
        "model": {
            "hidden_dim": 32,
            "concept_emb_dim": 32,
            "n_layers": 2,
            "head_hidden": 16,
            "dropout": 0.0,
            "prop_steps": 10,
            "alpha_init": 0.15,
            "learnable_alpha": True,
        },
        "dataset": {
            "graph": "synthetic",
            "n_nodes": 64,
            "edge_p": 0.25,
            "synthetic": True,
            "synthetic_alpha": 0.2,
            "synthetic_noise": 0.05,
            "split": {"mode": "standard", "val_frac": 0.2, "test_frac": 0.2, "n_genes": 8},
        },
        "training": {"epochs": 200, "lr": 0.01, "log_every": 50},
    }


def _baseline_test_metrics(data, model) -> dict[str, float]:
    x, y = data.features, data.labels
    model.fit(x[data.train_mask], y[data.train_mask])
    pred = model.predict(x[data.test_mask])
    return regression_metrics(pred, y[data.test_mask])


def run_gonogo(
    cfg: Mapping[str, Any] | None = None,
    seed: int = 0,
    out_dir: str | Path = "results",
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the gate and write ``baseline_gonogo.json`` under ``out_dir``.

    Returns the result dict including the ``go`` boolean.
    """
    cfg = dict(cfg) if cfg is not None else default_gonogo_cfg(seed)
    cfg["seed"] = seed
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_dir) if run_dir is not None else out_dir / "_gonogo_run"

    # Same split for all models: build the data once from the shared seed.
    data = build_cascade_data(cfg["dataset"], seed)

    baselines = {
        "mean": _baseline_test_metrics(data, MeanBaseline()),
        "additive_linear": _baseline_test_metrics(data, AdditiveLinearBaseline()),
    }

    # CASCADE trains on the identical config + seed, so it sees the same split.
    from cascade.training import train

    cascade_summary = train(cfg, out_dir=run_dir)
    cascade_test_mse = float(cascade_summary["test_mse"])

    best_baseline = min(baselines, key=lambda k: baselines[k]["mse"])
    best_baseline_mse = baselines[best_baseline]["mse"]
    ties = cascade_test_mse <= best_baseline_mse * (1.0 + TIE_BAND)
    beats = cascade_test_mse < best_baseline_mse

    result = {
        "go": bool(ties),
        "beats_best_baseline": bool(beats),
        "seed": seed,
        "tie_band": TIE_BAND,
        "metric": "test_mse (lower is better)",
        "cascade": {
            "test_mse": cascade_test_mse,
            "test_pearson": float(cascade_summary["test_pearson"]),
            "alpha": float(cascade_summary["alpha"]),
        },
        "baselines": baselines,
        "best_baseline": best_baseline,
        "best_baseline_test_mse": best_baseline_mse,
        "dataset_provenance": "synthetic propagation proxy (real GSE184929/STRING join is a tracked gap)",
        "num_nodes": int(data.num_nodes),
        "split_mode": data.meta["split_mode"],
    }

    out_path = out_dir / "baseline_gonogo.json"
    out_path.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    verdict = "GO" if result["go"] else "NO-GO"
    print(
        f"[gonogo] {verdict}: cascade_test_mse={cascade_test_mse:.5f} vs "
        f"best baseline ({best_baseline})={best_baseline_mse:.5f}; wrote {out_path}"
    )
    return result
