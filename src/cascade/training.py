"""Training loop for CASCADE (regression to measured log2 fold-change).

Hydra-free and omegaconf-free on purpose: :func:`train` takes a plain nested
mapping (an omegaconf ``DictConfig`` is one, and so is a dict), so the smoke
test can drive it without hydra installed. The hydra entrypoint lives in the
top-level ``train.py``.

Each run logs the seed it ran with, the propagation teleport alpha, and the
per-concept embedding norms (evidence the K=9 channels stay alive and
separated), to stdout and to ``metrics.jsonl`` / ``summary.json`` in the run
directory.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from cascade.data.cascade_dataset import CascadeData, build_cascade_data
from cascade.models.cascade import CASCADE
from cascade.models.ccmp_backbone import CCMPBackbone, default_concept_dims
from cascade.models.propagation_head import PropagationHead
from cascade.verify.containment import operator_norm, spectral_clip


def _get(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_cascade(model_cfg: Mapping[str, Any], concept_dims: dict[str, int]) -> CASCADE:
    """Build a CASCADE model (CCMP backbone + propagation head) from a config."""
    backbone = CCMPBackbone(
        concept_dims=concept_dims,
        hidden_dim=int(_get(model_cfg, "hidden_dim", 32)),
        concept_emb_dim=int(_get(model_cfg, "concept_emb_dim", 32)),
        n_layers=int(_get(model_cfg, "n_layers", 2)),
        head_hidden=int(_get(model_cfg, "head_hidden", 16)),
        dropout=float(_get(model_cfg, "dropout", 0.0)),
    )
    propagation = PropagationHead(
        num_steps=int(_get(model_cfg, "prop_steps", 10)),
        alpha_init=float(_get(model_cfg, "alpha_init", 0.15)),
        learnable_alpha=bool(_get(model_cfg, "learnable_alpha", True)),
    )
    return CASCADE(backbone, propagation)


def _pearson(pred: Tensor, target: Tensor) -> float:
    if pred.numel() < 2:
        return float("nan")
    p = pred - pred.mean()
    t = target - target.mean()
    denom = p.norm() * t.norm()
    if float(denom) < 1e-12:
        return float("nan")
    return float((p @ t) / denom)


def per_concept_norms(model: CASCADE, features: Tensor) -> dict[str, float]:
    """Mean L2 norm of each concept's embedding across nodes."""
    model.eval()
    with torch.no_grad():
        embs = model.get_per_concept_embeddings(features)
    return {name: float(z.norm(dim=-1).mean()) for name, z in embs.items()}


@torch.no_grad()
def evaluate(
    model: CASCADE,
    features: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor,
    n: int,
    labels: Tensor,
    mask: Tensor,
) -> dict[str, float]:
    model.eval()
    pred = model(features, edge_index, edge_weight, num_nodes=n, assert_contained=False)
    mse = float(((pred[mask] - labels[mask]) ** 2).mean())
    return {"mse": mse, "pearson": _pearson(pred[mask], labels[mask])}


def _to_tensors(data: CascadeData, device: torch.device):
    # Spectral-clip the GRN once so ||S||_2 <= 1 holds for every forward, then
    # turn off per-forward clipping for deterministic gradients.
    clipped, sigma = spectral_clip(data.S)
    coo = clipped.tocoo()
    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    edge_weight = torch.tensor(coo.data, dtype=torch.float32, device=device)
    features = torch.tensor(data.features, dtype=torch.float32, device=device)
    labels = torch.tensor(data.labels, dtype=torch.float32, device=device)
    masks = {
        name: torch.tensor(getattr(data, f"{name}_mask"), dtype=torch.bool, device=device)
        for name in ("train", "val", "test")
    }
    return features, edge_index, edge_weight, labels, masks, float(sigma)


def train(cfg: Mapping[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Run the CASCADE training loop; return a summary dict and write metrics.

    ``cfg`` is a nested mapping with ``seed``, ``device``, and ``model`` /
    ``dataset`` / ``training`` blocks.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(_get(cfg, "seed", 0))
    set_seed(seed)
    device = torch.device(str(_get(cfg, "device", "cpu")))
    model_cfg = _get(cfg, "model", {})
    dataset_cfg = _get(cfg, "dataset", {})
    train_cfg = _get(cfg, "training", {})
    print(f"[train] seed={seed} device={device}")

    data = build_cascade_data(dataset_cfg, seed)
    features, edge_index, edge_weight, labels, masks, sigma = _to_tensors(data, device)
    n = data.num_nodes
    print(
        f"[train] graph={data.meta['graph']} nodes={n} edges={data.meta['num_edges']} "
        f"||S||_2={sigma:.4f} split={data.meta['split_mode']} "
        f"train/val/test={int(masks['train'].sum())}/{int(masks['val'].sum())}/{int(masks['test'].sum())}"
    )
    # Containment by construction: assert the propagated operator cannot amplify.
    assert operator_norm(data.S) <= 1.0 + 1e-4, "GRN spectral norm exceeds 1; containment broken"

    if bool(_get(dataset_cfg, "synthetic", True)):
        concept_dims = default_concept_dims()
    else:
        concept_dims = {name: 1 for name in data.feature_names}  # overridden by node_table widths
    model = build_cascade(model_cfg, concept_dims).to(device)
    # Pre-clipped weights mean per-forward clipping is unnecessary and would add
    # nondeterminism, so disable it.
    model.propagation.clip = False

    lr = float(_get(train_cfg, "lr", 1e-2))
    weight_decay = float(_get(train_cfg, "weight_decay", 0.0))
    epochs = int(_get(train_cfg, "epochs", 30))
    log_every = int(_get(train_cfg, "log_every", 5))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    metrics_path = out_dir / "metrics.jsonl"
    metrics: list[dict[str, Any]] = []
    best_val = float("inf")
    best_state: dict[str, Tensor] | None = None

    with metrics_path.open("w", encoding="utf-8") as fh:
        for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            pred = model(
                features, edge_index, edge_weight, num_nodes=n, assert_contained=(epoch == 0)
            )
            loss = ((pred[masks["train"]] - labels[masks["train"]]) ** 2).mean()
            loss.backward()
            opt.step()

            if epoch % log_every == 0 or epoch == epochs - 1:
                val = evaluate(model, features, edge_index, edge_weight, n, labels, masks["val"])
                norms = per_concept_norms(model, features)
                record = {
                    "epoch": epoch,
                    "seed": seed,
                    "train_mse": float(loss.detach()),
                    "val_mse": val["mse"],
                    "val_pearson": val["pearson"],
                    "alpha": float(model.alpha.detach()),
                    "concept_norms": norms,
                }
                metrics.append(record)
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                print(
                    f"[train] epoch {epoch:3d} train_mse={record['train_mse']:.4f} "
                    f"val_mse={val['mse']:.4f} val_r={val['pearson']:.3f} "
                    f"alpha={record['alpha']:.4f}"
                )
                if val["mse"] < best_val:
                    best_val = val["mse"]
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, features, edge_index, edge_weight, n, labels, masks["test"])
    final_norms = per_concept_norms(model, features)

    summary = {
        "seed": seed,
        "epochs": epochs,
        "alpha": float(model.alpha.detach()),
        "best_val_mse": best_val,
        "test_mse": test["mse"],
        "test_pearson": test["pearson"],
        "spectral_norm": sigma,
        "concept_norms": final_norms,
        "split_mode": data.meta["split_mode"],
        "num_nodes": n,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[train] done: alpha={summary['alpha']:.4f} best_val_mse={best_val:.4f} "
        f"test_mse={test['mse']:.4f} test_r={test['pearson']:.3f}"
    )
    print(f"[train] wrote {out_dir / 'summary.json'} and {metrics_path}")
    return summary
