# AAAI results summary

## Acceptance status per theorem (real data only)

| theorem | status | one-line basis |
| --- | --- | --- |
| T1 (leakage-free per-concept identifiability) | **PASS** | disjoint CCMP per-concept embeddings change at machine zero on real Huesken; the shared-encoder CBM and the Concept Embedding Model leak. |
| T2 (P_alpha contraction / stability on the signed/directed GRN) | **RESTRICTED** | on the real signed/directed TRRUST GRN the rho-based L2 contraction constant is exceeded by directedness non-normality (||S||_2 ~ 8.31 >> rho ~ 1); the inf-norm certificate is non-expansive for every alpha, and the magnetic operator (q-swept) repairs L2 contraction but stays secondary to the real-valued P_alpha. |
| T3 (KKT mass-action / thermodynamic constraint) | **PASS** | KKT mass-action residual at machine precision on every seed with no accuracy regression; the composed-error bound holds and the IFT gradient through the equality+box ADMM matches finite differences below 1e-6. |
| T4 (certified off-target bound) | **RESTRICTED** | the certified L2 bound is REAL and BINDS on GSE184929 via the TRRUST P_alpha join, and the measured seed off-target KS is real in rat (Schlegel) AND human (Patel GSE164289 / GSE196839); restricted by cross-species ortholog alignment, assumed uncertainty budgets, and the microarray cross-assay (Jackson 2006 / Birmingham 2006) which is PENDING (the GEO deposits lack the siRNA sequences and Birmingham is not in GEO). |
| T5 (cross-species transfer) | **DEMOTED** | Cross-species AUROC stays below 0.6: T5 is DEMOTED from a theorem to a NUMERICAL OBSERVATION; it must NOT be framed as a theorem. |

Legend: PASS = holds on real data; RESTRICTED = holds with a documented restriction; PENDING = real data not yet available; DEMOTED = an empirical observation, not a theorem.

## Headline (real data only)
- Baselines (real Huesken efficacy): CASCADE Pearson 0.605 vs ridge/linear 0.628 on the Huesken test split. Beats the mean baseline=True; beats the ridge/linear baseline=False (a near-tie, the gap is just outside the 0.02 band). Out of distribution CASCADE beats ridge on the Katoh transfer set (CASCADE 0.168 vs ridge 0.065).
- T2 (real TRRUST GRN): rho(S)=1.000, certified L2 contraction constant 1.000; verdict RESTRICTED (rho-based L2 bound holds=False, inf-norm non-expansive=True for every alpha; ||S||_2=8.313).
- T3 (real Huesken + ViennaRNA): mass-action constraint residual 1.67e-16 (< 1e-08, machine precision on every seed); KKT-constrained Pearson 0.386 +/- 0.051 vs unconstrained 0.417 +/- 0.033 (within one std: regression beyond 1 std=False); verdict PASS.
- T4 (real GSE184929 RNA-seq): the certified L2 bound is REAL via the TRRUST GRN-propagation join (alpha=0.5, 2072 nodes, 1908 aligned). It BINDS on all 3 Ttr contrasts (binds_all=True); D1_30mkg bound 74.4 vs measured deviation 25.2 (vacuous=False). The measured seed off-target KS is also real (D1 present, D4 abolished).

## T1 leakage-free per-concept identifiability
- Provenance: Huesken 2005 on-target efficacy (OligoFormer suite); disjoint CCMP backbone vs a shared-encoder CBM and a Concept Embedding Model on the SAME real data
- Disjoint cross-channel change 0.00 (machine zero=True); shared encoder 0.700, CEM 1.338. Verdict PASS.

## T2 P_alpha contraction / stability (real signed/directed GRN)
- rho(S)=1.000, ||S||_2=8.313, ||S||_inf=1.000; max true ||P_alpha||_2 2.365, max measured ratio 0.953; alpha sweep over 11 values; signed spectral gap 0.293 (min real eig -0.522).
- rho-based L2 constant is exact for the symmetric variant and exceeded on the signed/directed GRN by non-normality; the inf-norm certificate holds for every alpha and the L2 ||S||_2 bound holds for alpha > alpha_min.
- Magnetic-operator grid (q in [0.0, 0.1, 0.2, 0.25, 0.5]): the magnetic signed-Laplacian operator has a PSD spectrum in [0,2] for every q (True) so its 2-norm is 1-alpha (q-independent) and <= 1 wherever ||P_alpha||_2 > 1. It does NOT dominate on every metric (False); outcome: p_alpha_primary__magnetic_is_l2_repair_only (real P_alpha primary, magnetic = L2 repair).

## T3 KKT mass-action / thermodynamic constraint
- Provenance: Huesken 2005 efficacy + ViennaRNA seed-duplex free energy dG (real)
- Constraint residual 1.67e-16 (< 1e-08, below tol on every seed=True); constrained Pearson 0.386 +/- 0.051 vs unconstrained 0.417 +/- 0.033 (accuracy gate is std-based: regression beyond 1 std=False). Verdict PASS.
- dG diagnostic (one-time, real data): total seed dG vs knockdown Pearson 0.135 (slope sign +1); the duplex-end asymmetry feature is more predictive (Pearson -0.278). dG is not standardized before the mass-action fit (raw slope b=0.033, sign +1), so the positive b is a real property of on-target efficacy, not a feature-scaling artifact.
- Composed-error bound (real Huesken): forward-map Lipschitz L=0.148, dist(y_*,P)=0.0032; the bound ||Pi_P(f(x))-y_*|| <= L||x-x_*|| + dist(y_*,P) holds across all probes=True. The genuine ADMM (equality + occupancy box) holds equality residual 5.55e-17 / active-set residual 1.01e-16; the IFT gradient through the eq+box projection matches finite differences at 5.51e-07 (path held: ift).

## T4 certified off-target bound (per real dataset)
- Certified bound REAL via the TRRUST GRN-propagation join: e propagated over P_alpha (alpha=0.5), bound = ||P_a(S)|| ||e-e_true|| + ((1-a)/a) ||P_a(S)||^2 ||S-S'|| ||e||. TRRUST is human, GSE184929 is rat; aligned by orthologous gene symbol. e is the seed-match-derived direct repression of the ViennaRNA-confirmed Ttr seed.

| contrast | certified bound | measured deviation | binds | vacuous | rowwise max |
| --- | --- | --- | --- | --- | --- |
| Schlegel_2022_Ttr_D1_30mkg | 74.406 | 25.222 | True | False | 31.465 |
| Schlegel_2022_Ttr_D4_30mkg | 74.406 | 11.545 | True | False | 31.465 |
| Schlegel_2022_Ttr_D1_10mkg | 74.406 | 14.695 | True | False | 31.465 |

- Cross-assay: RNA-seq (GSE184929) mean KS D=0.0715 (REAL); microarray mean KS D=pending (pending_real_data: Jackson 2006 / Birmingham 2006 lack public per-gene fold-changes / siRNA sequences).
- Thermodynamic-input discrimination: per-site seed dG D1 -5.606 vs D4 -2.584 (guides differ=True); e_dG separates D1 from D4 by predicted magnitude=True where the count cannot, and the rank-based KS is saturated for both.
- Real HUMAN off-target KS (Patel 2022, hg38 = GRCh38, confirmed from each GEO Sample_data_processing (Genome_build: hg38); not assumed, Ensembl release 104):

| dataset | contrast | seed | KS D | p |
| --- | --- | --- | --- | --- |
| GSE164289 | siGGGGGC | GGGGGC | 0.0264 | 7.53e-04 |
| GSE164289 | siGGCAGU | GGCAGU | 0.0323 | 9.50e-06 |
| GSE196839 | siGGGGGC | GGGGGC | 0.1882 | 2.23e-147 |

## T5 cross-species transfer (rat -> human zero-shot)
- Status **DEMOTED**: best cross-species AUROC 0.5941 (threshold 0.6); survives as theorem=False. Cross-species AUROC stays below 0.6: T5 is DEMOTED from a theorem to a NUMERICAL OBSERVATION; it must NOT be framed as a theorem.
- 1-Wasserstein(rat vs human 3'UTR seed-match abundance) = 0.3432.

| q | GSE164289_siGGGGGC | GSE196839_siGGGGGC |
| --- | --- | --- |
| 0.0 | 0.4800 | 0.5792 |
| 0.1 | 0.4802 | 0.5793 |
| 0.2 | 0.4800 | 0.5791 |
| 0.25 | 0.4800 | 0.5793 |
| 0.5 | 0.4819 | 0.5941 |

## Figures (results/aaai/figures/)
- `t1_leakage.png`: Theorem 1 leakage audit on real efficacy data: maximum cross-channel embedding change per model (the disjoint CCMP backbone is at machine zero; the shared-encoder CBM and the Concept Embedding Model leak), and the Havasi probe leakage gap and Yeh completeness.
- `t2_stability.png`: Theorem 2 on the real TRRUST signed/directed GRN: the true L2 operator norm of P_alpha versus the certified rho-based and inf-norm constants across alpha (the rho-based L2 constant is exceeded by non-normality; the inf-norm certificate holds for every alpha), and the Dirichlet-energy oversmoothing curve.
- `t3_constraint.png`: Theorem 3 on real Huesken efficacy with real ViennaRNA dG: the mass-action constraint residual (KKT-projected at machine precision versus the unconstrained model) and the held-out accuracy with and without the projection.
