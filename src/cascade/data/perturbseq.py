"""GEARS-based Perturb-seq loaders (TRACKED GAP, stubbed).

These cover Replogle 2022 (genome-scale Perturb-seq, DOI
10.1016/j.cell.2022.05.013) and Norman 2019 (combinatorial Perturb-seq, DOI
10.1126/science.aax4438). They supply the interventional data for the
identifiability project (single-gene hard and combinatorial soft interventions)
and the downstream-ripple validation of CASCADE's containment claim (see
docs/DATA_CARD.md section 4).

The target implementation loads each atlas through GEARS ``PertData`` (the
GEARS data utility downloads and parses both datasets and exposes the
perturbation labels and the cell-by-gene expression), then packs the result
into :class:`PerturbSeqDataset`. The dataclass shape and loader signatures are
fixed so downstream code can be written against them now.

TODO: wire GEARS ``PertData`` (``pip install cell-gears``); until then the
loaders raise :class:`NotImplementedError` and ``tests/test_shared_core.py``
carries a visible xfail-strict gap test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Bump alongside the (future) parser implementation.
PARSER_VERSION = "perturbseq-stub-v0"


@dataclass(frozen=True)
class PerturbSeqDataset:
    """Placeholder container for a parsed Perturb-seq atlas.

    Fields are intentionally minimal until the full loader lands. ``name``
    identifies the source atlas (``"replogle2022"`` / ``"norman2019"``);
    ``perturbations`` is the per-cell guide-target label vector and
    ``expression`` the cell-by-gene matrix once populated.
    """

    name: str
    perturbations: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    expression: np.ndarray | None = None
    gene_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.perturbations)


def _not_implemented(name: str) -> NotImplementedError:
    return NotImplementedError(
        f"{name} GEARS-based Perturb-seq loader is a tracked gap; wire GEARS "
        "PertData to populate PerturbSeqDataset. See cascade.data.perturbseq "
        "and the gap test in tests/test_shared_core.py."
    )


def load_replogle2022(
    raw_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    rebuild: bool = False,
) -> PerturbSeqDataset:
    """Stub: load the Replogle 2022 genome-scale Perturb-seq atlas via GEARS PertData."""
    raise _not_implemented("Replogle 2022")


def load_norman2019(
    raw_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    rebuild: bool = False,
) -> PerturbSeqDataset:
    """Stub: load the Norman 2019 combinatorial Perturb-seq atlas via GEARS PertData."""
    raise _not_implemented("Norman 2019")
