"""Measured off-target signature loaders (role: offtarget_certification).

These are the datasets the off-target containment certificate is evaluated
against (the efficacy datasets are NOT; see docs/DATA_CARD.md and
:mod:`cascade.data.roles`).

- GSE184929 (RNA-seq) is wired separately in :mod:`cascade.data.geo`.
- Jackson 2003 (Nat Biotechnol) and Jackson 2006 (RNA): microarray off-target
  signatures, down-regulated transcripts with 3'UTR seed complementarity. Some
  tables live in the paper supplements rather than a clean GEO matrix, so this
  module parses supplement tables and records the provenance; where a GEO
  accession exists, point the loader at it.
- Birmingham 2006 (Nat Methods): seed-match off-target associations.
- SeedSeq (Das 2013): the aggregated off-target transcriptome table, a
  convenience union over the above with a ``source`` flag on every call.

Each loader returns an :class:`OffTargetDataset` whose frame has columns
siRNA id, antisense, seed, off-target transcript, effect size, cell line or
tissue, assay, and source. Microarray sets are lower resolution and use older
annotation than the GSE184929 RNA-seq; this is recorded honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .cache import CacheKey, cached_frame, sha256_file
from .paths import cache_dir as default_cache_dir
from .paths import raw_dir as default_raw_dir
from .roles import OFFTARGET_CERTIFICATION

PARSER_VERSION = "offtarget-v1"

COLUMNS = [
    "sirna_id",
    "antisense",
    "seed",
    "off_target_transcript",
    "effect_size",
    "cell_line",
    "assay",
    "source",
]

_SIRNA = ("sirna_id", "sirna", "guide_id", "sirna_name", "oligo", "id")
_ANTISENSE = ("antisense", "guide", "antisense_sequence", "as_seq")
_SEED = ("seed", "seed_region", "seed_match", "seed_seq")
_TRANSCRIPT = (
    "off_target_transcript",
    "transcript",
    "off_target",
    "target_transcript",
    "refseq",
    "probe",
    "gene",
)
_EFFECT = (
    "effect_size",
    "log2fc",
    "log2_fold_change",
    "fold_change",
    "logratio",
    "ratio",
    "zscore",
    "score",
)
_CELL = ("cell_line", "cell", "tissue", "cellline")
_ASSAY = ("assay", "assay_type", "platform")
_SOURCE = ("source", "origin", "dataset", "study")

# Fixed provenance per dataset.
OFFTARGET_SOURCES: dict[str, dict] = {
    "jackson2003": {
        "cell_line": "HeLa",
        "assay": "microarray",
        "provenance": "Jackson 2003 (Nat Biotechnol); supplement tables (no clean GEO matrix)",
    },
    "jackson2006": {
        "cell_line": "HeLa",
        "assay": "microarray",
        "provenance": "Jackson 2006 (RNA); supplement tables",
    },
    "birmingham2006": {
        "cell_line": "various",
        "assay": "microarray",
        "provenance": "Birmingham 2006 (Nat Methods); seed-match off-target associations",
    },
    "seedseq": {
        "cell_line": "various",
        "assay": "microarray (aggregated)",
        "provenance": "SeedSeq / Das 2013; aggregated union over Jackson and Birmingham",
    },
}


@dataclass(frozen=True)
class OffTargetDataset:
    """Measured off-target calls for one source (role: offtarget_certification)."""

    name: str
    frame: pd.DataFrame
    assay: str
    provenance: str
    role: str = field(default=OFFTARGET_CERTIFICATION)

    def __len__(self) -> int:
        return len(self.frame)

    def sirnas(self) -> list[str]:
        return sorted(self.frame["sirna_id"].unique().tolist())

    def sources(self) -> list[str]:
        return sorted(self.frame["source"].unique().tolist())

    def transcripts(self) -> list[str]:
        return sorted(self.frame["off_target_transcript"].unique().tolist())


def _pick(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    for cand in candidates:
        for low, orig in lowered.items():
            if cand in low:
                return orig
    return None


def _seed_from_antisense(antisense: pd.Series) -> pd.Series:
    """Seed = guide positions 2-8 (RNA), derived when no seed column is present."""
    rna = antisense.astype(str).str.strip().str.upper().str.replace("T", "U", regex=False)
    return rna.str.slice(1, 8)


def _parse_offtarget(path: Path, name: str) -> pd.DataFrame:
    meta = OFFTARGET_SOURCES[name]
    df = pd.read_csv(path, sep=None, engine="python", compression="infer")
    cols = list(df.columns)
    transcript = _pick(cols, _TRANSCRIPT)
    if transcript is None:
        raise ValueError(
            f"{path}: need an off-target transcript column ({_TRANSCRIPT}); found {cols}"
        )
    n = len(df)
    sirna = _pick(cols, _SIRNA)
    anti = _pick(cols, _ANTISENSE)
    seed = _pick(cols, _SEED)
    effect = _pick(cols, _EFFECT)
    cell = _pick(cols, _CELL)
    assay = _pick(cols, _ASSAY)
    src = _pick(cols, _SOURCE)

    antisense = df[anti].astype(str).str.strip().str.upper() if anti else pd.Series([""] * n)
    if seed:
        seed_series = df[seed].astype(str).str.strip().str.upper()
    elif anti:
        seed_series = _seed_from_antisense(df[anti])
    else:
        seed_series = pd.Series([""] * n)

    return pd.DataFrame(
        {
            "sirna_id": (df[sirna].astype(str) if sirna else [f"{name}_sirna"] * n),
            "antisense": antisense.to_numpy(),
            "seed": seed_series.to_numpy(),
            "off_target_transcript": df[transcript].astype(str).to_numpy(),
            "effect_size": (
                pd.to_numeric(df[effect], errors="coerce").to_numpy()
                if effect
                else [float("nan")] * n
            ),
            "cell_line": (df[cell].astype(str) if cell else [meta["cell_line"]] * n),
            "assay": (df[assay].astype(str) if assay else [meta["assay"]] * n),
            "source": (df[src].astype(str) if src else [name] * n),
        }
    )[COLUMNS]


def _default_path(name: str) -> Path:
    return default_raw_dir() / "offtarget" / f"{name}.csv"


def _load_one(
    name: str, raw_path: str | Path | None, cache_root: str | Path | None, rebuild: bool
) -> OffTargetDataset:
    meta = OFFTARGET_SOURCES[name]
    raw_path = Path(raw_path) if raw_path is not None else _default_path(name)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{name} off-target table not found at {raw_path}. Place the parsed supplement "
            "table there; see docs/DATA_CARD.md."
        )
    cache_root = Path(cache_root) if cache_root is not None else default_cache_dir()
    key = CacheKey(name=name, version=PARSER_VERSION, params={"sha256": sha256_file(raw_path)})
    frame = cached_frame(key, cache_root, lambda: _parse_offtarget(raw_path, name), rebuild=rebuild)
    return OffTargetDataset(
        name=name, frame=frame, assay=meta["assay"], provenance=meta["provenance"]
    )


def load_jackson2003(raw_path=None, cache_root=None, rebuild: bool = False) -> OffTargetDataset:
    """Jackson 2003 microarray off-target signatures (supplement tables)."""
    return _load_one("jackson2003", raw_path, cache_root, rebuild)


def load_jackson2006(raw_path=None, cache_root=None, rebuild: bool = False) -> OffTargetDataset:
    """Jackson 2006 microarray off-target signatures (supplement tables)."""
    return _load_one("jackson2006", raw_path, cache_root, rebuild)


def load_birmingham2006(raw_path=None, cache_root=None, rebuild: bool = False) -> OffTargetDataset:
    """Birmingham 2006 seed-match off-target associations (microarray)."""
    return _load_one("birmingham2006", raw_path, cache_root, rebuild)


def load_seedseq(raw_path=None, cache_root=None, rebuild: bool = False) -> OffTargetDataset:
    """SeedSeq (Das 2013) aggregated off-target union with a per-call source flag.

    Reads the aggregated table at ``data/raw/offtarget/seedseq.csv`` if present.
    Otherwise builds the convenience union from whichever individual sources
    (Jackson 2003/2006, Birmingham 2006) are available, tagging each call with
    its ``source``.
    """
    meta = OFFTARGET_SOURCES["seedseq"]
    raw_path = Path(raw_path) if raw_path is not None else _default_path("seedseq")
    if raw_path.exists():
        return _load_one("seedseq", raw_path, cache_root, rebuild)

    members = []
    for loader in (load_jackson2003, load_jackson2006, load_birmingham2006):
        try:
            members.append(loader(cache_root=cache_root, rebuild=rebuild).frame)
        except FileNotFoundError:
            continue
    if not members:
        raise FileNotFoundError(
            f"no SeedSeq table at {raw_path} and no member off-target sources available to "
            "build the union; see docs/DATA_CARD.md."
        )
    union = pd.concat(members, ignore_index=True)[COLUMNS]
    return OffTargetDataset(
        name="seedseq",
        frame=union,
        assay=meta["assay"],
        provenance=meta["provenance"] + " (built from available members)",
    )
