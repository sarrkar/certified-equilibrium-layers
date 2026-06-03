#!/usr/bin/env python3
"""Hydra entrypoint for CASCADE training.

Thin wrapper around :func:`cascade.training.train`. All configuration lives under
``configs/`` (model / dataset / training groups composed by ``configs/config.yaml``).

Examples:
    python train.py                          # smoke train on the fixture graph
    python train.py seed=1 training.epochs=100
    python train.py dataset.split.mode=held_out_gene
    python train.py training=default         # full schedule
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Make the package importable when running ``python train.py`` without install.
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from cascade.training import train  # noqa: E402


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    out_dir = HydraConfig.get().runtime.output_dir
    train(cfg, out_dir=out_dir)


if __name__ == "__main__":
    main()
