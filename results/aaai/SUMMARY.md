# AAAI certified off-target containment: results summary

## Headline
- Empirical coverage of the certified radius is 1.0 (at or above the nominal 1 minus delta at every delta); verdict PASS, max excess violation -0.01 (the radius is never violated more than a delta fraction).
- On the GSE184929 / SeedMatchR off-target pipeline (real seed logic, synthetic-proxy graph: the STRING sequence join is a tracked gap), the certified containment fraction reaches 0.9 at tolerance tau = 10.478 and is 1.0 at twice the certified radius (radius 9.584, 70 seed-matched off-targets).

## Certificate
- alpha = 0.5; certified-radius constant (1-alpha)/alpha = 1.0.
- P_alpha eigenvalues lie in [0.3333, 1.0], so the operator norm is 1.0.
- Bernstein quantities (star graph, 32 uncertain edges, c = 0.5, delta = 0.05): v = 8.0, L = 0.5, b(delta) = 13.1174.

## Tightness
- The radius variance term peaks at c = 0.5 and follows sqrt(c(1-c)) (scaling coefficient of variation 4.69e-16).
- The probabilistic certificate is tighter than the deterministic Lipschitz envelope by approximately sqrt(E) (fitted ratio exponent 0.533).

## Symmetric-normalized breakdown (C5)
- The degree-concentration assumption fails (rho >= 1) at STRING confidence cutoff 0.925; the Le-Levina-Vershynin minimum-degree condition is lost at 0.95.
- On the pipeline, edge subsampling degrades gracefully until the C5 breakdown at 0.8 of edges removed.

## Monte-Carlo cost
- Samples needed for a 1 minus delta interval: delta=0.2: N=1565, delta=0.1: N=1704, delta=0.05: N=1843, delta=0.01: N=2164, delta=0.001: N=2625.

## Figures (results/aaai/figures/)
- `coverage.png`: Empirical versus nominal coverage of the matrix-Bernstein certified radius across delta on spectrally-bounded random matchings. Empirical coverage stays at or above 1 minus delta at every delta (verdict PASS, no violations).
- `tightness_radius_vs_c.png`: Certified radius versus edge confidence c on a star graph. The variance term peaks at the coin-flip confidence c = 0.5 and follows the sqrt(c(1-c)) scaling law; it is tightest as confidences approach 0 or 1.
- `tightness_prob_vs_det.png`: Probabilistic Bernstein radius versus the deterministic resolvent-Lipschitz envelope as the number of uncertain edges grows. The deterministic envelope grows linearly while the probabilistic radius grows as sqrt(E), so the probabilistic certificate is tighter by approximately sqrt(number of edges).
- `normalized_breakdown.png`: Symmetric-normalized (C5) certificate via degree concentration: minimum expected degree and the degree relative fluctuation rho versus STRING confidence threshold. The certificate is valid while rho < 1 and breaks down as the graph is thresholded sparser.
- `realdata.png`: Real-data pipeline (SeedMatchR seed repression propagated over a STRING GRN): certified off-target containment fraction versus tolerance, certified radius versus STRING threshold, graceful degradation with the C5 breakdown under edge subsampling, and the Monte-Carlo sample cost for a 1 minus delta interval.
- `certified_region.png`: Resolvent-perturbation certified L2 region under the Gaussian per-edge model: certified radius versus delta and the realized out-of-sample coverage.

## Provenance
- Graph source: synthetic STRING-like confidence matrix. x0 source: SeedMatchR seed-match scores on synthetic 3'UTR sequences. the real GSE184929 / STRING sequence join is a tracked data gap; x0 uses the real SeedMatchR seed logic
