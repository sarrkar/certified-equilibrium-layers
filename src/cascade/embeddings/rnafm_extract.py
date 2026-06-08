"""RNA-FM per-nucleotide and mean-pooled embeddings for RNA sequences.

Loads the RNA-FM foundation model via ``fm.pretrained.rna_fm_t12()`` (Chen et al. 2022,
the 12-layer, 640-dimensional RNA language model) and embeds a list of RNA sequences
(siRNA guides of 19 to 23 nt and 3'UTR seed-match windows, each well under the 1024-token
limit). For each sequence it returns the 640-d per-nucleotide embeddings (one vector per
base, the special BOS / EOS tokens stripped) and the mean-pooled 640-d sequence vector,
as numpy arrays, and can write them to a JSON sidecar plus a .npy file.

REAL DATA ONLY: there is NO synthetic fallback. A missing, empty, or non-RNA sequence
RAISES; the model is never substituted by random vectors. torch and the ``fm`` (rna-fm)
package are required to CALL the extractor and are imported lazily, so importing this
module stays light (the lightweight numpy import path never pulls in torch / fm).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RNAFM_DIM = 640  # RNA-FM rna_fm_t12 hidden dimension
RNAFM_MAX_SEQ_LEN = 1022  # 1024-token limit minus the BOS and EOS tokens
RNAFM_INSTALL = "pip install rna-fm torch  (the 'fm' package; see github.com/ml4bio/RNA-FM)"
_REPR_LAYER = 12  # final layer of rna_fm_t12

_MODEL_CACHE: dict = {}


def require_rnafm() -> None:
    """Raise ImportError (with the install command) if torch or RNA-FM are unavailable."""
    try:
        import fm  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"RNA-FM embeddings need torch and the 'fm' (rna-fm) package, which are not "
            f"installed. {RNAFM_INSTALL}"
        ) from exc


def _clean_sequences(sequences: list[str]) -> list[str]:
    """Validate and normalize RNA sequences; RAISE on any missing / invalid one."""
    if sequences is None or len(sequences) == 0:
        raise ValueError("no sequences provided")
    clean: list[str] = []
    for i, s in enumerate(sequences):
        if s is None or not str(s).strip():
            raise ValueError(f"sequence {i} is missing or empty; REAL DATA ONLY, no fallback")
        seq = str(s).strip().upper().replace("T", "U")
        bad = sorted({c for c in seq if c not in "ACGUN"})
        if bad:
            raise ValueError(f"sequence {i} has non-RNA characters {bad}: {seq!r}")
        if len(seq) > RNAFM_MAX_SEQ_LEN:
            raise ValueError(
                f"sequence {i} length {len(seq)} exceeds the RNA-FM limit {RNAFM_MAX_SEQ_LEN}"
            )
        clean.append(seq)
    return clean


def load_model():
    """Load (and cache) RNA-FM rna_fm_t12: returns (model, alphabet, batch_converter)."""
    require_rnafm()
    if "rna_fm_t12" not in _MODEL_CACHE:
        import fm

        model, alphabet = fm.pretrained.rna_fm_t12()
        model.eval()
        _MODEL_CACHE["rna_fm_t12"] = (model, alphabet, alphabet.get_batch_converter())
    return _MODEL_CACHE["rna_fm_t12"]


def extract_embeddings(
    sequences: list[str], labels: list[str] | None = None, batch_size: int = 8
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return (per_nucleotide, mean_pooled) RNA-FM embeddings for ``sequences``.

    per_nucleotide is a list of float32 arrays, one (L_i, 640) array per sequence (the
    special tokens stripped); mean_pooled is a float32 (n, 640) array. Raises on any
    missing / invalid sequence (no synthetic fallback).
    """
    import torch

    clean = _clean_sequences(sequences)
    if labels is None:
        labels = [f"seq{i}" for i in range(len(clean))]
    elif len(labels) != len(clean):
        raise ValueError(f"got {len(labels)} labels for {len(clean)} sequences")
    model, _alphabet, batch_converter = load_model()

    per_nt: list[np.ndarray] = []
    pooled: list[np.ndarray] = []
    for start in range(0, len(clean), batch_size):
        chunk = clean[start : start + batch_size]
        chunk_labels = labels[start : start + batch_size]
        _, _, tokens = batch_converter(list(zip(chunk_labels, chunk, strict=True)))
        with torch.no_grad():
            reps = model(tokens, repr_layers=[_REPR_LAYER])["representations"][_REPR_LAYER]
        for j, seq in enumerate(chunk):
            # tokens are [BOS, base_1 ... base_L, EOS, pad ...]; keep the L base positions.
            emb = reps[j, 1 : len(seq) + 1, :].cpu().numpy().astype(np.float32)
            if emb.shape != (len(seq), RNAFM_DIM):
                raise ValueError(
                    f"unexpected RNA-FM output {emb.shape} for a {len(seq)} nt sequence"
                )
            per_nt.append(emb)
            pooled.append(emb.mean(axis=0))
    return per_nt, np.stack(pooled).astype(np.float32)


def write_embeddings(
    sequences: list[str], out_prefix: str | Path, labels: list[str] | None = None
) -> tuple[list[np.ndarray], np.ndarray]:
    """Extract and write the embeddings to ``out_prefix`` + .npy and + .json.

    The .npy holds the per-nucleotide embeddings as a length-n object array (one (L_i, 640)
    array per sequence); the JSON sidecar holds the metadata and the mean-pooled vectors.
    Returns (per_nucleotide, mean_pooled).
    """
    out_prefix = Path(out_prefix)
    clean = _clean_sequences(sequences)
    if labels is None:
        labels = [f"seq{i}" for i in range(len(clean))]
    per_nt, pooled = extract_embeddings(clean, labels=labels)

    obj = np.empty(len(per_nt), dtype=object)  # ragged-safe 1-D object array
    for i, emb in enumerate(per_nt):
        obj[i] = emb
    npy_path = out_prefix.with_suffix(".npy")
    np.save(npy_path, obj, allow_pickle=True)

    sidecar = {
        "model": "RNA-FM rna_fm_t12",
        "embedding_dim": RNAFM_DIM,
        "n_sequences": len(clean),
        "labels": labels,
        "sequences": clean,
        "lengths": [int(emb.shape[0]) for emb in per_nt],
        "mean_pooled": pooled.tolist(),
        "per_nucleotide_npy": npy_path.name,
        "note": "per-nucleotide 640-d embeddings are in the .npy (object array, one (L,640) "
        "per sequence); the mean-pooled 640-d vectors are listed here.",
    }
    out_prefix.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return per_nt, pooled
