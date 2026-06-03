"""Thermodynamic features for siRNA seed pairing and target-site accessibility.

When the ViennaRNA Python bindings are installed (the pinned dependency), this
module uses them directly:

- seed-pairing stability via ``RNA.duplexfold`` (hybridization MFE of the seed
  against its best target site, and of the 3' supplementary region),
- target-site accessibility via the partition-function base-pair probabilities
  (``RNA.fold_compound(...).bpp()``), from which per-nucleotide unpaired
  probabilities and a site opening energy are derived.

When ViennaRNA is not importable (for example in a lightweight CI image), a
deterministic nearest-neighbor fallback is used instead: duplex free energies
come from the Turner / Xia nearest-neighbor stacking table (Xia et al. 1998,
Biochemistry; Turner and Mathews 2010), and accessibility is approximated from
local AU content. The fallback keeps the feature blocks well-defined and the
tests runnable without ViennaRNA; it is documented as an approximation and is
not used when ViennaRNA is present.
"""

from __future__ import annotations

import math

from .seedmatch import Site, revcomp, to_rna

try:  # ViennaRNA is the preferred backend; fall back when unavailable.
    import RNA as _RNA

    HAS_VIENNA = True
except ImportError:  # pragma: no cover - exercised only in ViennaRNA-free envs
    _RNA = None
    HAS_VIENNA = False

# Turner / Xia nearest-neighbor stacking free energies, dG37 in kcal/mol, keyed
# by the 5' to 3' top-strand dinucleotide (the complementary bottom strand is
# implied). Reverse-complement-symmetric duplicates share a value.
_NN_STACK: dict[str, float] = {
    "AA": -0.93, "AC": -2.24, "AG": -2.08, "AU": -1.10,
    "CA": -2.11, "CC": -3.26, "CG": -2.36, "CU": -2.08,
    "GA": -2.35, "GC": -3.42, "GG": -3.26, "GU": -2.24,
    "UA": -1.33, "UC": -2.35, "UG": -2.11, "UU": -0.93,
}  # fmt: skip
_HELIX_INIT = 4.09  # helix initiation penalty (kcal/mol)
_TERMINAL_AU = 0.45  # per-end penalty for a terminal A-U pair
_RT_37 = 0.6156  # gas constant times 310.15 K, kcal/mol
_P_FLOOR = 1e-6  # clamp for unpaired probabilities before taking a log


def _stack_energy(seq: str) -> float:
    """Sum of nearest-neighbor stacking energies along a perfectly paired strand."""
    return sum(_NN_STACK.get(seq[i : i + 2], 0.0) for i in range(len(seq) - 1))


def nn_duplex_mfe(strand: str) -> float:
    """Nearest-neighbor duplex free energy of a fully Watson-Crick region.

    The strand is assumed paired to its reverse complement (the seed-match
    site is exactly that complement). Returns 0.0 for strands shorter than a
    single stack.
    """
    s = to_rna(strand)
    if len(s) < 2:
        return 0.0
    dg = _HELIX_INIT + _stack_energy(s)
    for end in (s[0], s[-1]):
        if end in ("A", "U"):
            dg += _TERMINAL_AU
    return dg


def _vienna_duplex_mfe(strand_a: str, strand_b: str) -> float:
    duplex = _RNA.duplexfold(to_rna(strand_a), to_rna(strand_b))
    return float(duplex.energy)


def _duplex_mfe(strand: str, partner: str | None = None) -> float:
    """Hybridization MFE of ``strand`` to ``partner`` (its complement if None)."""
    s = to_rna(strand)
    p = to_rna(partner) if partner is not None else revcomp(s)
    if len(s) < 2 or len(p) < 2:
        return 0.0
    if HAS_VIENNA:
        return _vienna_duplex_mfe(s, p)
    return nn_duplex_mfe(s)


def seed_pairing_stability(
    guide: str, target: str, sites: list[Site]
) -> tuple[float, float, float]:
    """Return (seed duplex MFE, MFE per base pair, 3' supplementary MFE).

    The seed duplex is guide positions 2 to 8 hybridized to its best site. The
    supplementary term is guide positions 13 to 16 (the TargetScan 3'
    supplementary region) hybridized to its best complementary match in the
    target, or 0.0 when absent.
    """
    g = to_rna(guide)
    t = to_rna(target)
    seed = g[1:8]  # positions 2 to 8
    n_bp = max(len(seed), 1)

    if sites:
        site = sites[0]
        site_word = t[site.start : site.end]
        seed_mfe = _duplex_mfe(seed, site_word)
    else:
        seed_mfe = _duplex_mfe(seed)
    seed_mfe_per_bp = seed_mfe / n_bp

    supp_mfe = 0.0
    if len(g) >= 16:
        supp = g[12:16]  # positions 13 to 16
        if revcomp(supp) in t:
            supp_mfe = _duplex_mfe(supp)
    return seed_mfe, seed_mfe_per_bp, supp_mfe


def _au_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(b in ("A", "U") for b in seq) / len(seq)


def _vienna_unpaired(target: str, lo: int, hi: int, pad: int = 50) -> list[float]:
    """Per-position unpaired probabilities over target[lo:hi] from the pf model."""
    a = max(0, lo - pad)
    b = min(len(target), hi + pad)
    window = target[a:b]
    fc = _RNA.fold_compound(window)
    fc.pf()
    bpp = fc.bpp()  # 1-indexed [n+1][n+1]
    n = len(window)
    unpaired: list[float] = []
    for idx in range(lo - a, hi - a):
        i = idx + 1
        paired = 0.0
        for j in range(1, n + 1):
            paired += bpp[i][j] if i < j else (bpp[j][i] if j < i else 0.0)
        unpaired.append(max(0.0, 1.0 - paired))
    return unpaired


def target_accessibility(target: str, site: Site | None) -> tuple[float, float, float]:
    """Return (site unpaired prob, opening energy, flank unpaired prob).

    Opening energy is ``-RT * sum(log p_unpaired)`` over the site (a positive
    cost to expose the site). With ViennaRNA the probabilities come from the
    partition function; otherwise they are approximated by local AU content.
    """
    t = to_rna(target)
    if not t:
        return 0.0, 0.0, 0.0
    if site is None:
        lo, hi = 0, min(len(t), 7)
    else:
        lo, hi = site.start, site.end
    flo, fhi = max(0, lo - 7), min(len(t), hi + 7)

    if HAS_VIENNA:
        site_p = _vienna_unpaired(t, lo, hi)
        flank_p = _vienna_unpaired(t, flo, fhi)
    else:  # pragma: no cover - AU-content proxy, ViennaRNA-free path
        site_p = [_au_fraction(t[lo:hi])] * max(hi - lo, 1)
        flank_p = [_au_fraction(t[flo:fhi])] * max(fhi - flo, 1)

    site_unpaired = sum(site_p) / len(site_p) if site_p else 0.0
    flank_unpaired = sum(flank_p) / len(flank_p) if flank_p else 0.0
    opening = -_RT_37 * sum(math.log(max(p, _P_FLOOR)) for p in site_p)
    return site_unpaired, opening, flank_unpaired


def nn_end_energy(strand: str) -> float:
    """Nearest-neighbor stacking energy of a short terminal region (no init)."""
    return _stack_energy(to_rna(strand))


def thermo_asymmetry(guide: str, end_len: int = 4) -> tuple[float, float, float]:
    """Return (5'-end dG, 3'-end dG, asymmetry) of the guide duplex ends.

    Captures the Khvorova / Schwarz thermodynamic end-asymmetry rule: a guide
    whose 5' end is less stable than its 3' end is preferentially loaded. The
    asymmetry score is ``dG_5p - dG_3p`` from the nearest-neighbor table.
    """
    g = to_rna(guide)
    if len(g) < end_len + 1:
        return 0.0, 0.0, 0.0
    dg_5p = nn_end_energy(g[:end_len])
    dg_3p = nn_end_energy(g[-end_len:])
    return dg_5p, dg_3p, dg_5p - dg_3p
