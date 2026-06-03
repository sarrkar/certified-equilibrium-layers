"""Baseline models: seed-match, APPNP, sequence-only, linear.

The linear baselines (additive OLS and mean) are pure numpy and back the
GO/NO-GO gate. The remaining baselines land in follow-up commits.
"""

from .linear import AdditiveLinearBaseline, MeanBaseline, regression_metrics

__all__ = ["AdditiveLinearBaseline", "MeanBaseline", "regression_metrics"]
