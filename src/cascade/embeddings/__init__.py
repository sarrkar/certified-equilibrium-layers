"""RNA foundation-model embeddings for CASCADE (gated off the lightweight import path).

The RNA-FM extractor (:mod:`cascade.embeddings.rnafm_extract`) needs torch and the
``fm`` (rna-fm) package; both are imported lazily inside the functions so that merely
importing this package never requires them.
"""

from .rnafm_extract import (
    RNAFM_DIM,
    RNAFM_MAX_SEQ_LEN,
    extract_embeddings,
    require_rnafm,
    write_embeddings,
)

__all__ = [
    "RNAFM_DIM",
    "RNAFM_MAX_SEQ_LEN",
    "extract_embeddings",
    "require_rnafm",
    "write_embeddings",
]
