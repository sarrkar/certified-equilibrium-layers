"""Content-addressed cache helpers for parsed datasets.

A loader builds a :class:`CacheKey` from its inputs and a short version
tag. The key hashes to a fixed-length directory name under
``data/cache/<dataset>/<hash>/``. Loaders are responsible for writing and
reading their own artifact files inside that directory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CacheKey:
    """Identity of a parsed-dataset cache entry.

    ``name`` is the dataset name (used as the parent directory). ``version``
    is an opaque tag the loader bumps when its parser changes. ``params``
    holds any inputs that affect the parsed output (paths, thresholds, etc).
    """

    name: str
    version: str
    params: dict[str, Any]

    def hexdigest(self) -> str:
        payload = json.dumps(
            {"name": self.name, "version": self.version, "params": self.params},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


def cache_dir_for(base: Path, key: CacheKey) -> Path:
    """Return (and create) the cache directory for ``key`` under ``base``."""
    p = Path(base) / key.name / key.hexdigest()
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(p: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
