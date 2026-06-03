"""Stub loaders for the Perturb-seq downstream-ripple proxies.

These cover Replogle 2022 (genome-scale Perturb-seq, DOI
10.1016/j.cell.2022.05.013) and Norman 2019 (combinatorial Perturb-seq, DOI
10.1126/science.aax4438). They are intended for held-out downstream-ripple
validation of CASCADE's containment claim (see docs/DATA_CARD.md section 4).

Full parsing lands in a follow-up commit. For now the dataclass shape and the
loader signatures are fixed so downstream code can be written against them; the
loaders raise :class:`NotImplementedError` until the parser arrives.
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
        f"{name} Perturb-seq loader is a stub; full integration lands in a "
        "follow-up commit. See docs/DATA_CARD.md section 4."
    )


def load_replogle2022(
    raw_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    rebuild: bool = False,
) -> PerturbSeqDataset:
    """Stub: load the Replogle 2022 genome-scale Perturb-seq atlas."""
    raise _not_implemented("Replogle 2022")


def load_norman2019(
    raw_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    rebuild: bool = False,
) -> PerturbSeqDataset:
    """Stub: load the Norman 2019 combinatorial Perturb-seq atlas."""
    raise _not_implemented("Norman 2019")
