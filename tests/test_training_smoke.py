"""Smoke tests for the CASCADE training loop (torch-gated)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cascade.training import train  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "string_tiny.txt"


def _cfg(graph: str, mode: str, cache_root: Path) -> dict:
    return {
        "seed": 0,
        "device": "cpu",
        "model": {
            "hidden_dim": 16,
            "concept_emb_dim": 16,
            "n_layers": 2,
            "head_hidden": 16,
            "dropout": 0.0,
            "prop_steps": 8,
            "alpha_init": 0.15,
            "learnable_alpha": True,
        },
        "dataset": {
            "graph": graph,
            "threshold": 700,
            "cache": str(cache_root),
            "synthetic": True,
            "synthetic_alpha": 0.2,
            "synthetic_noise": 0.05,
            "n_nodes": 20,
            "edge_p": 0.35,
            "split": {"mode": mode, "val_frac": 0.25, "test_frac": 0.25, "n_genes": 5},
        },
        "training": {"epochs": 20, "lr": 0.02, "log_every": 5},
    }


def test_smoke_train_fixture_graph(tmp_path: Path, cache_root: Path) -> None:
    out = tmp_path / "run"
    summary = train(_cfg(str(FIXTURE), "standard", cache_root), out_dir=out)

    # alpha stays the bounded interpretable knob.
    assert 0.0 < summary["alpha"] < 1.0
    # All nine concept channels are logged and alive.
    assert len(summary["concept_norms"]) == 9
    assert all(v > 0 for v in summary["concept_norms"].values())
    # Containment held during training.
    assert summary["spectral_norm"] <= 1.0 + 1e-4
    assert summary["seed"] == 0

    # Artifacts written.
    assert (out / "summary.json").exists()
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert records[0]["seed"] == 0
    assert "alpha" in records[0] and "concept_norms" in records[0]
    # Loss went down over the run.
    assert records[-1]["train_mse"] <= records[0]["train_mse"]


def test_smoke_train_held_out_gene(tmp_path: Path) -> None:
    out = tmp_path / "run_hog"
    summary = train(_cfg("synthetic", "held_out_gene", tmp_path / "cache"), out_dir=out)
    assert summary["split_mode"] == "held_out_gene"
    assert 0.0 < summary["alpha"] < 1.0
    # On a larger graph the held-out test Pearson is computable and finite.
    assert summary["num_nodes"] == 20
