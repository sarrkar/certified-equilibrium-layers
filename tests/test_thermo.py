"""Tests for the thermodynamic feature module (ViennaRNA + NN fallback)."""

from __future__ import annotations

from cascade.features import thermo
from cascade.features.seedmatch import find_sites
from cascade.features.thermo import (
    nn_duplex_mfe,
    seed_pairing_stability,
    target_accessibility,
    thermo_asymmetry,
)

GUIDE = "UGAGGUAGUAGGUUGUAUAGU"


def test_nn_duplex_mfe_gc_more_stable_than_au() -> None:
    # A GC-rich duplex is more stable (more negative dG) than an AU-rich one.
    assert nn_duplex_mfe("GGGGGG") < nn_duplex_mfe("AAAAAA")
    # Too-short strands contribute no stacking energy.
    assert nn_duplex_mfe("G") == 0.0


def test_thermo_asymmetry_sign() -> None:
    # Weak (AU) 5' end and strong (GC) 3' end => positive asymmetry score.
    guide = "AAUU" + "A" * 13 + "GGCC"
    dg5, dg3, asym = thermo_asymmetry(guide)
    assert dg5 > dg3  # 5' end less stable (less negative)
    assert asym > 0.0
    assert abs(asym - (dg5 - dg3)) < 1e-9


def test_seed_pairing_stability_finite_and_negative() -> None:
    words_target = "CCCAUAUA" + "CUACCUCA" + "GGGCAUUAAC"
    sites = find_sites(GUIDE, words_target)
    mfe, per_bp, supp = seed_pairing_stability(GUIDE, words_target, sites)
    assert mfe < 0.0
    assert abs(per_bp - mfe / 7.0) < 1e-9
    assert supp <= 0.0  # 0 when absent, negative when a supplementary match exists


def test_target_accessibility_in_range() -> None:
    target = "GGGCAUUAACCGGUUAAUGCCCAAAUUUGGGCCCAAAUUU"
    sites = find_sites(GUIDE, target)
    site = sites[0] if sites else None
    up, opening, flank = target_accessibility(target, site)
    assert 0.0 <= up <= 1.0
    assert 0.0 <= flank <= 1.0
    assert opening >= 0.0  # opening a structured site costs energy


def test_fallback_path_without_vienna(monkeypatch) -> None:
    # Force the ViennaRNA-free branch and confirm features stay well-defined.
    monkeypatch.setattr(thermo, "HAS_VIENNA", False)
    monkeypatch.setattr(thermo, "_RNA", None)
    target = "GGGCAUUAACCGGUUAAUGCCCAAAUUUGGG" + "CUACCUCA" + "GGG"
    sites = find_sites(GUIDE, target)
    mfe, per_bp, _supp = seed_pairing_stability(GUIDE, target, sites)
    assert mfe < 0.0  # NN duplex energy of the seed
    up, opening, flank = target_accessibility(target, sites[0] if sites else None)
    assert 0.0 <= up <= 1.0
    assert 0.0 <= flank <= 1.0
    assert opening >= 0.0
