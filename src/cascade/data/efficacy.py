"""Unified on-target siRNA efficacy loader (role: efficacy_backbone).

Loads the OligoFormer nine-dataset suite (3714 siRNAs across 75 mRNAs) into one
tidy table. Each row carries the antisense and sense sequences, the target mRNA
id, the measured on-target efficacy in [0, 1], and provenance (source study,
cell line, assay type).

Source files. Prefer the parsed CSVs released with OligoFormer or siRNADiscovery
(siRNADiscovery uses a 2816-pair HUVK set from Huesken, Harborth, Ui-Tei,
Vickers, Khvorova plus a 322-siRNA Simone validation set). Reuse those rather
than re-parsing the raw journal supplements. Expected layout:
``data/raw/oligoformer/<source>.csv`` for source in
{huesken, katoh, mix, shabalina}; the ``mix`` file carries a per-row source
study (Amarzguioui 2003, Harborth 2003, Hsieh 2004, Khvorova 2003, Reynolds
2004, Vickers 2003, Ui-Tei 2004).

These datasets measure ON-TARGET potency. They do NOT validate the off-target
containment certificate (see docs/DATA_CARD.md). The loader is column-name
tolerant and caches a tidy parquet under ``data/cache/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import CacheKey, cached_frame, sha256_file
from .paths import cache_dir as default_cache_dir
from .paths import raw_dir as default_raw_dir
from .roles import EFFICACY_BACKBONE

PARSER_VERSION = "efficacy-v1"

# OligoFormer source files and their fixed provenance. The ``mix`` file pools
# seven studies that carry their own per-row source_study.
EFFICACY_SOURCES: dict[str, dict] = {
    "huesken": {
        "study": "Huesken 2005",
        "cell_line": "H1299",
        "assay": "branched-DNA residual mRNA",
        "n_sirna": 2431,
        "n_mrna": 34,
    },
    "katoh": {
        "study": "Katoh (Taka)",
        "cell_line": "HeLa",
        "assay": "luciferase reporter",
        "n_sirna": 702,
        "n_mrna": 1,
    },
    "mix": {
        "study": "Mix",
        "cell_line": "various",
        "assay": "various",
        "n_sirna": None,
        "n_mrna": None,
    },
    "shabalina": {
        "study": "Shabalina 2006",
        "cell_line": "various",
        "assay": "various (thermodynamic-filtered compilation, 13 publications)",
        "n_sirna": 653,
        "n_mrna": None,
    },
}

# Studies pooled inside the OligoFormer "mix" file.
MIX_STUDIES = (
    "Amarzguioui 2003",
    "Harborth 2003",
    "Hsieh 2004",
    "Khvorova 2003",
    "Reynolds 2004",
    "Vickers 2003",
    "Ui-Tei 2004",
)

_ANTISENSE = ("antisense", "guide", "as", "antisense_sequence", "sirna_antisense", "as_seq")
_SENSE = ("sense", "passenger", "ss", "sense_sequence", "sirna_sense", "ss_seq")
_MRNA = ("mrna_id", "mrna", "target", "gene", "transcript", "target_id")
_EFF = ("efficacy", "inhibition", "activity", "knockdown", "norm_inhibition", "label", "value")
_SOURCE = ("source_study", "source", "study", "dataset", "publication")
_CELL = ("cell_line", "cell", "cellline")
_ASSAY = ("assay", "assay_type", "readout")

COLUMNS = ["antisense", "sense", "mrna_id", "efficacy", "source_study", "cell_line", "assay"]


@dataclass(frozen=True)
class EfficacyDataset:
    """Unified on-target efficacy table (role: efficacy_backbone)."""

    frame: pd.DataFrame
    role: str = field(default=EFFICACY_BACKBONE)

    def __len__(self) -> int:
        return len(self.frame)

    def sources(self) -> list[str]:
        return sorted(self.frame["source_study"].unique().tolist())

    def cell_lines(self) -> list[str]:
        return sorted(self.frame["cell_line"].unique().tolist())

    def assays(self) -> list[str]:
        return sorted(self.frame["assay"].unique().tolist())

    def filter_study(self, study: str) -> EfficacyDataset:
        return EfficacyDataset(
            self.frame[self.frame["source_study"] == study].reset_index(drop=True)
        )


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


def _normalize_efficacy(values: np.ndarray) -> np.ndarray:
    eff = np.asarray(values, dtype=float)
    finite = eff[np.isfinite(eff)]
    if finite.size and finite.max() > 1.0:  # percent-scaled -> [0, 1]
        eff = eff / 100.0
    return np.clip(eff, 0.0, 1.0)


def _parse_source(path: Path, source_key: str) -> pd.DataFrame:
    meta = EFFICACY_SOURCES[source_key]
    df = pd.read_csv(path, sep=None, engine="python", compression="infer")
    cols = list(df.columns)
    anti = _pick(cols, _ANTISENSE)
    eff = _pick(cols, _EFF)
    if anti is None or eff is None:
        raise ValueError(
            f"{path}: need an antisense column ({_ANTISENSE}) and an efficacy column "
            f"({_EFF}); found {cols}"
        )
    sense = _pick(cols, _SENSE)
    mrna = _pick(cols, _MRNA)
    src = _pick(cols, _SOURCE)
    cell = _pick(cols, _CELL)
    assay = _pick(cols, _ASSAY)
    n = len(df)
    out = pd.DataFrame(
        {
            "antisense": df[anti].astype(str).str.strip().str.upper(),
            "sense": (df[sense].astype(str).str.strip().str.upper() if sense else [""] * n),
            "mrna_id": (df[mrna].astype(str) if mrna else [f"{source_key}_mrna"] * n),
            "efficacy": _normalize_efficacy(df[eff].to_numpy()),
            "source_study": (df[src].astype(str) if src else [meta["study"]] * n),
            "cell_line": (df[cell].astype(str) if cell else [meta["cell_line"]] * n),
            "assay": (df[assay].astype(str) if assay else [meta["assay"]] * n),
        }
    )
    return out


def default_raw_dir_efficacy() -> Path:
    return default_raw_dir() / "oligoformer"


def load_oligoformer_efficacy(
    raw_dir: str | Path | None = None,
    cache_root: str | Path | None = None,
    sources: list[str] | None = None,
    rebuild: bool = False,
) -> EfficacyDataset:
    """Load the OligoFormer efficacy suite into one tidy table.

    Args:
        raw_dir: directory holding ``<source>.csv`` files. Default
            ``data/raw/oligoformer``.
        cache_root: parquet cache root. Default ``data/cache``.
        sources: subset of {huesken, katoh, mix, shabalina}. Default: all present.
        rebuild: reparse and overwrite the cache.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else default_raw_dir_efficacy()
    cache_root = Path(cache_root) if cache_root is not None else default_cache_dir()
    keys = sources if sources is not None else list(EFFICACY_SOURCES)

    present = [(k, raw_dir / f"{k}.csv") for k in keys if (raw_dir / f"{k}.csv").exists()]
    if not present:
        raise FileNotFoundError(
            f"no OligoFormer efficacy CSVs found under {raw_dir} (expected "
            f"{[k + '.csv' for k in keys]}). Pull the parsed CSVs from the OligoFormer "
            "or siRNADiscovery release; see docs/DATA_CARD.md."
        )

    digest = {k: sha256_file(p) for k, p in present}
    key = CacheKey(name="efficacy", version=PARSER_VERSION, params={"sources": digest})

    def build() -> pd.DataFrame:
        frames = [_parse_source(p, k) for k, p in present]
        return pd.concat(frames, ignore_index=True)[COLUMNS]

    return EfficacyDataset(cached_frame(key, cache_root, build, rebuild=rebuild))
