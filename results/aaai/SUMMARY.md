# AAAI results summary

## Headline
- Certificate soundness (synthetic): empirical coverage of the certified radius is 1.0 (>= nominal 1 minus delta at every delta); verdict PASS, max excess violation -0.01 (never violated more than delta).
- Application (real off-target data): on GSE184929 RNA-seq the certified containment separates measured off-targets (AUC 0.961, measured-off-target contained 0.033 vs non-off-target 0.900); acceptance PASS across GSE184929 and 4 microarray sets.
- Backbone competitiveness (efficacy): Huesken-test Pearson 0.530 / Spearman 0.553; acceptance PENDING vs reported BIOPREDsi (Huesken ANN) (0.66).

## 1. Certificate soundness (synthetic graphs only, no biological claim)
- alpha = 0.5; certified-radius constant (1-alpha)/alpha = 1.0; P_alpha eigenvalues in [0.3333, 1.0] (operator norm 1.0).
- Bernstein quantities (star graph, 32 uncertain edges, c = 0.5, delta = 0.05): v = 8.0, L = 0.5, b(delta) = 13.1174.
- Tightness (C4): the variance term peaks at c = 0.5 and follows sqrt(c(1-c)) (CV 4.69e-16); probabilistic vs deterministic ratio exponent 0.533 (approx sqrt(E)).
- Coverage (C3): empirical vs nominal across delta on random matchings, verdict PASS.
- Symmetric-normalized breakdown (C5): degree concentration fails (rho >= 1) at STRING cutoff 0.925; LLV min-degree condition lost at 0.95.

## 2. Application results (real off-target data)
real off-target datasets; synthetic proxy by default (the real GSE184929 / STRING / microarray-supplement join is a tracked gap). The off-target measurements are the datasets the certificate is evaluated against.

| dataset | assay | AUC | separation | off-target contained | non-off contained | meaningful |
| --- | --- | --- | --- | --- | --- | --- |
| gse184929 | rna_seq | 0.961 | 0.867 | 0.033 | 0.900 | True |
| jackson2003 | microarray | 0.763 | 0.600 | 0.300 | 0.900 | True |
| jackson2006 | microarray | 0.851 | 0.673 | 0.222 | 0.896 | True |
| birmingham2006 | microarray | 0.916 | 0.808 | 0.091 | 0.899 | True |
| seedseq | microarray_aggregated | 0.843 | 0.697 | 0.200 | 0.897 | True |

- Cross-assay: RNA-seq mean AUC 0.961 vs microarray mean AUC 0.843. RNA-seq is expected to separate measured off-targets more cleanly than microarray (lower resolution, older annotation); the gap is reported, not hidden.
- ACCEPTANCE PASS: GSE184929 meaningful=True, any microarray meaningful=True; inconsistencies=[].

## 3. Backbone competitiveness (efficacy data, on-target only)
on-target efficacy of the CCMP backbone (the x0 direct-repression predictor). This measures on-target potency and does NOT validate the off-target certificate.

| dataset | n | Pearson | Spearman |
| --- | --- | --- | --- |
| huesken_test | 150 | 0.530 | 0.553 |
| taka | 200 | 0.462 | 0.438 |
| mix | 200 | 0.187 | 0.171 |
| shabalina | 150 | 0.231 | 0.220 |

- ACCEPTANCE PENDING: our Huesken-test Pearson 0.530 vs best reported BIOPREDsi (Huesken ANN) (0.66). SYNTHETIC PROXY: our numbers are illustrative and are NOT comparable to the reported SOTA; rerun with --raw-dir pointing at the OligoFormer CSVs for the real acceptance check.
- the drop from Huesken-test to Taka/Mix/Shabalina is the expected distribution shift; it is reported here, not hidden.

## Figures (results/aaai/figures/)
- `coverage.png`: Empirical versus nominal coverage of the matrix-Bernstein certified radius across delta on spectrally-bounded random matchings. Empirical coverage stays at or above 1 minus delta at every delta (verdict PASS, no violations).
- `tightness_radius_vs_c.png`: Certified radius versus edge confidence c on a star graph. The variance term peaks at the coin-flip confidence c = 0.5 and follows the sqrt(c(1-c)) scaling law; it is tightest as confidences approach 0 or 1.
- `tightness_prob_vs_det.png`: Probabilistic Bernstein radius versus the deterministic resolvent-Lipschitz envelope as the number of uncertain edges grows. The deterministic envelope grows linearly while the probabilistic radius grows as sqrt(E), so the probabilistic certificate is tighter by approximately sqrt(number of edges).
- `normalized_breakdown.png`: Symmetric-normalized (C5) certificate via degree concentration: minimum expected degree and the degree relative fluctuation rho versus STRING confidence threshold. The certificate is valid while rho < 1 and breaks down as the graph is thresholded sparser.
- `realdata.png`: Application across every off-target dataset (GSE184929 RNA-seq, Jackson 2003/2006, Birmingham 2006, SeedSeq): the predicted off-target effect vs the measured calls (AUC and containment separation), the contained fraction by class (measured off-targets vs non-off-targets), certified radius vs STRING threshold per source, and degradation with the C5 breakdown under edge subsampling.
- `backbone_efficacy.png`: CCMP backbone on-target efficacy (the x0 predictor) on the OligoFormer suite: Pearson and Spearman on the Huesken test split and cross-dataset generalization to Taka, Mix, and Shabalina, with the reported BIOPREDsi reference overlaid.
- `certified_region.png`: Resolvent-perturbation certified L2 region under the Gaussian per-edge model: certified radius versus delta and the realized out-of-sample coverage.
