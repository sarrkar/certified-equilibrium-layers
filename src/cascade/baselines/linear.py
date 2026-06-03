"""Additive-linear and mean baselines for off-target log2FC regression.

These are the reference baselines the CASCADE off-target model must at least tie
on a held-out split (the GO/NO-GO gate, :mod:`cascade.eval.gonogo`). Both are
per-node predictors of the feature matrix only; neither uses the GRN, so the
gap between them and CASCADE measures what the certified propagation buys.

Pure numpy (no torch) so they run in the lightweight CI image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MeanBaseline:
    """Predict the training-set mean label for every node (the trivial baseline)."""

    mean_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> MeanBaseline:
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.mean_, dtype=float)


@dataclass
class AdditiveLinearBaseline:
    """Ordinary least squares with intercept: y ~ x . w + b (additive over features)."""

    coef_: np.ndarray | None = None
    intercept_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> AdditiveLinearBaseline:
        x = np.asarray(x, dtype=float)
        design = np.hstack([x, np.ones((x.shape[0], 1))])
        sol, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=float), rcond=None)
        self.coef_ = sol[:-1]
        self.intercept_ = float(sol[-1])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("AdditiveLinearBaseline must be fit before predict")
        return np.asarray(x, dtype=float) @ self.coef_ + self.intercept_


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return MSE, MAE, and Pearson r (r is NaN when a side is constant)."""
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    if pred.size < 2:
        return {"mse": mse, "mae": mae, "pearson": float("nan")}
    p = pred - pred.mean()
    t = target - target.mean()
    denom = float(np.linalg.norm(p) * np.linalg.norm(t))
    pearson = float((p @ t) / denom) if denom > 1e-12 else float("nan")
    return {"mse": mse, "mae": mae, "pearson": pearson}
