"""Project 2 (AAAI): probabilistic certified L2 region for off-target prediction.

Certifies an L2 region for the diffusion-propagated prediction y = P_alpha(S) x0
under a confidence-calibrated random-edge noise model on the inferred GRN
(:mod:`noise_model`), via a finite-sample Monte-Carlo radius and an analytic
resolvent-perturbation bound (:mod:`certify`), with coverage calibration
(:mod:`calibration`).

The package reuses the CASCADE P_alpha operator and the spectral clip from
``cascade.verify.containment`` and never re-implements them (CLAUDE.md rule 8).
Pure numpy / scipy, so it runs in the lightweight CI image.
"""

from .calibration import calibrate_base_std, empirical_coverage
from .certify import CertResult, certify_l2_region
from .noise_model import RandomEdgeNoise, confidence_from_weights, upper_edges

__all__ = [
    "RandomEdgeNoise",
    "confidence_from_weights",
    "upper_edges",
    "CertResult",
    "certify_l2_region",
    "empirical_coverage",
    "calibrate_base_std",
]
