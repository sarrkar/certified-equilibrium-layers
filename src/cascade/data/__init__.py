"""Data loaders and caching for CASCADE.

Each loader parses an open-access source into a small dataclass and caches the
parsed form under ``data/cache/`` (idempotent). See docs/DATA_CARD.md for
provenance and the rat-liver specificity caveat on GSE184929.
"""

from .cache import CacheKey, cache_dir_for, sha256_file
from .geo import (
    DEFAULT_CONTRASTS,
    GEOOffTargetDataset,
    load_gse184929,
)
from .huesken import (
    HueskenDataset,
    canonical_split,
    load_huesken,
)
from .paths import cache_dir, data_dir, raw_dir, repo_root
from .perturbseq import (
    PerturbSeqDataset,
    load_norman2019,
    load_replogle2022,
)
from .reactome import load_reactome_graph
from .string_grn import (
    DEFAULT_THRESHOLD,
    StringGRN,
    load_string_grn,
)

__all__ = [
    # cache primitives
    "CacheKey",
    "cache_dir_for",
    "sha256_file",
    # paths
    "repo_root",
    "data_dir",
    "raw_dir",
    "cache_dir",
    # Huesken efficacy
    "HueskenDataset",
    "load_huesken",
    "canonical_split",
    # GSE184929 off-target
    "GEOOffTargetDataset",
    "load_gse184929",
    "DEFAULT_CONTRASTS",
    # STRING GRN
    "StringGRN",
    "load_string_grn",
    "DEFAULT_THRESHOLD",
    # Reactome GRN (tracked gap)
    "load_reactome_graph",
    # Perturb-seq stubs (tracked gap)
    "PerturbSeqDataset",
    "load_replogle2022",
    "load_norman2019",
]
