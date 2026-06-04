"""Data loaders and caching for CASCADE.

Each loader parses an open-access source into a small dataclass and caches the
parsed form under ``data/cache/`` (idempotent). See docs/DATA_CARD.md for
provenance and the rat-liver specificity caveat on GSE184929.
"""

from .cache import CacheKey, cache_dir_for, cached_frame, sha256_file
from .efficacy import (
    EFFICACY_SOURCES,
    MIX_STUDIES,
    EfficacyDataset,
    load_oligoformer_efficacy,
)
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
from .offtarget import (
    OFFTARGET_SOURCES,
    OffTargetDataset,
    load_birmingham2006,
    load_jackson2003,
    load_jackson2006,
    load_seedseq,
)
from .paths import cache_dir, data_dir, raw_dir, repo_root
from .perturbseq import (
    PerturbSeqDataset,
    load_norman2019,
    load_replogle2022,
)
from .reactome import load_reactome_graph
from .roles import (
    DATASET_ROLES,
    EFFICACY_BACKBONE,
    OFFTARGET_CERTIFICATION,
    SYNTHETIC_COVERAGE,
    dataset_role,
    datasets_with_role,
)
from .string_grn import (
    DEFAULT_THRESHOLD,
    StringGRN,
    load_string_confidence,
    load_string_grn,
)
from .synthetic import SyntheticCoverageDataset, make_synthetic_coverage

__all__ = [
    # roles
    "EFFICACY_BACKBONE",
    "OFFTARGET_CERTIFICATION",
    "SYNTHETIC_COVERAGE",
    "DATASET_ROLES",
    "dataset_role",
    "datasets_with_role",
    # cache primitives
    "CacheKey",
    "cache_dir_for",
    "cached_frame",
    "sha256_file",
    # paths
    "repo_root",
    "data_dir",
    "raw_dir",
    "cache_dir",
    # on-target efficacy backbone (OligoFormer suite)
    "EfficacyDataset",
    "load_oligoformer_efficacy",
    "EFFICACY_SOURCES",
    "MIX_STUDIES",
    # Huesken efficacy (one efficacy_backbone source)
    "HueskenDataset",
    "load_huesken",
    "canonical_split",
    # off-target certification loaders
    "OffTargetDataset",
    "load_jackson2003",
    "load_jackson2006",
    "load_birmingham2006",
    "load_seedseq",
    "OFFTARGET_SOURCES",
    # synthetic coverage ensembles
    "SyntheticCoverageDataset",
    "make_synthetic_coverage",
    # GSE184929 off-target
    "GEOOffTargetDataset",
    "load_gse184929",
    "DEFAULT_CONTRASTS",
    # STRING GRN
    "StringGRN",
    "load_string_grn",
    "load_string_confidence",
    "DEFAULT_THRESHOLD",
    # Reactome GRN (tracked gap)
    "load_reactome_graph",
    # Perturb-seq stubs (tracked gap)
    "PerturbSeqDataset",
    "load_replogle2022",
    "load_norman2019",
]
