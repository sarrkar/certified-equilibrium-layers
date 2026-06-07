# AAAI results summary

## Headline
- Certificate soundness (random-graph verification, no biological claim): empirical coverage of the certified radius is 1.0 (>= nominal 1 minus delta at every delta); verdict PASS, max excess violation -0.01 (never violated more than delta).
- Application (real off-target data): PENDING real data. exp_realdata.py runs the Theorem 4 certificate the moment the real off-target / STRING join is wired; no non-real number is reported.
- Backbone competitiveness (efficacy): PENDING real data. exp_backbone_efficacy.py trains on the real OligoFormer suite when the CSVs are present; no non-real number is reported.

## 1. Certificate soundness (random-graph verification, no biological claim)
- alpha = 0.5; certified-radius constant (1-alpha)/alpha = 1.0; P_alpha eigenvalues in [0.3333, 1.0] (operator norm 1.0).
- Bernstein quantities (star graph, 32 uncertain edges, c = 0.5, delta = 0.05): v = 8.0, L = 0.5, b(delta) = 13.1174.
- Tightness (C4): the variance term peaks at c = 0.5 and follows sqrt(c(1-c)) (CV 4.69e-16); probabilistic vs deterministic ratio exponent 0.533 (approx sqrt(E)).
- Coverage (C3): empirical vs nominal across delta on random matchings, verdict PASS.
- Symmetric-normalized breakdown (C5): degree concentration fails (rho >= 1) at STRING cutoff 0.925; LLV min-degree condition lost at 0.95.

## 2. Application results (real off-target data)
PENDING real data. Per the hard data policy, exp_realdata.py uses only real measured off-target calls and a real STRING GRN and raises until that join is wired, so no containment numbers are reported here yet.

## 3. Backbone competitiveness (efficacy data, on-target only)
PENDING real data. exp_backbone_efficacy.py trains the CCMP backbone on the real OligoFormer suite and raises when the CSVs are absent, so no efficacy numbers are reported here yet.

## Figures (results/aaai/figures/)
- `coverage.png`: Empirical versus nominal coverage of the matrix-Bernstein certified radius across delta on spectrally-bounded random matchings. Empirical coverage stays at or above 1 minus delta at every delta (verdict PASS, no violations).
- `tightness_radius_vs_c.png`: Certified radius versus edge confidence c on a star graph. The variance term peaks at the coin-flip confidence c = 0.5 and follows the sqrt(c(1-c)) scaling law; it is tightest as confidences approach 0 or 1.
- `tightness_prob_vs_det.png`: Probabilistic Bernstein radius versus the deterministic resolvent-Lipschitz envelope as the number of uncertain edges grows. The deterministic envelope grows linearly while the probabilistic radius grows as sqrt(E), so the probabilistic certificate is tighter by approximately sqrt(number of edges).
- `normalized_breakdown.png`: Symmetric-normalized (C5) certificate via degree concentration: minimum expected degree and the degree relative fluctuation rho versus STRING confidence threshold. The certificate is valid while rho < 1 and breaks down as the graph is thresholded sparser.
- `certified_region.png`: Resolvent-perturbation certified L2 region under the Gaussian per-edge model: certified radius versus delta and the realized out-of-sample coverage.
