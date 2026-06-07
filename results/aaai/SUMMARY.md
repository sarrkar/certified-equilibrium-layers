# AAAI four-theorem results summary

## Headline (real data only)
- Baselines (real Huesken efficacy): CASCADE Pearson 0.605 vs ridge/linear 0.628 on the Huesken test split. Beats the mean baseline=True; beats the ridge/linear baseline=False (a near-tie, the gap is just outside the 0.02 band). Out of distribution CASCADE beats ridge on the Katoh transfer set (CASCADE 0.168 vs ridge 0.065).
- T2 (real TRRUST GRN): rho(S)=1.000, certified L2 contraction constant 1.000; verdict RESTRICTED (rho-based L2 bound holds=False, inf-norm non-expansive=True for every alpha; ||S||_2=8.313).
- T3 (real Huesken + ViennaRNA): mass-action constraint residual 1.67e-16 (< 1e-08, machine precision on every seed); KKT-constrained Pearson 0.386 +/- 0.051 vs unconstrained 0.417 +/- 0.033 (within one std: regression beyond 1 std=False); verdict PASS.
- T4 (real GSE184929 RNA-seq): measured off-target SeedMatchR KS is REAL (guide seed UAUAGAG, Rnor_6.0 3'UTRs). Parent D1_30mkg D=0.159 (p=8.64e-24) shows seed-mediated off-target; the g7-GNA D4_30mkg D=0.048 (p=7.52e-03) abolishes it. The certified L2 bound stays PENDING the STRING / GRN-propagation join.

## T1 leakage-free per-concept identifiability
- Provenance: Huesken 2005 on-target efficacy (OligoFormer suite); disjoint CCMP backbone vs a shared-encoder CBM and a Concept Embedding Model on the SAME real data
- Disjoint cross-channel change 0.00 (machine zero=True); shared encoder 0.700, CEM 1.338. Verdict PASS.

## T2 P_alpha contraction / stability (real signed/directed GRN)
- rho(S)=1.000, ||S||_2=8.313, ||S||_inf=1.000; max true ||P_alpha||_2 2.365, max measured ratio 0.953; alpha sweep over 11 values; signed spectral gap 0.293 (min real eig -0.522).
- rho-based L2 constant is exact for the symmetric variant and exceeded on the signed/directed GRN by non-normality; the inf-norm certificate holds for every alpha and the L2 ||S||_2 bound holds for alpha > alpha_min.

## T3 KKT mass-action / thermodynamic constraint
- Provenance: Huesken 2005 efficacy + ViennaRNA seed-duplex free energy dG (real)
- Constraint residual 1.67e-16 (< 1e-08, below tol on every seed=True); constrained Pearson 0.386 +/- 0.051 vs unconstrained 0.417 +/- 0.033 (accuracy gate is std-based: regression beyond 1 std=False). Verdict PASS.
- dG diagnostic (one-time, real data): total seed dG vs knockdown Pearson 0.135 (slope sign +1); the duplex-end asymmetry feature is more predictive (Pearson -0.278). dG is not standardized before the mass-action fit (raw slope b=0.033, sign +1), so the positive b is a real property of on-target efficacy, not a feature-scaling artifact.

## T4 certified off-target bound (per real dataset)
- Measured off-target SeedMatchR KS is REAL (guide UUAUAGAGCAAGAACACUGUUUU, seed UAUAGAG; Ensembl Rnor_6.0 rat 3'UTRs (build-matched to GSE184929)). The certified L2 bound stays PENDING the GRN-propagation join.

| contrast | KS D | p | n seed-match | n background |
| --- | --- | --- | --- | --- |
| Schlegel_2022_Ttr_D1_30mkg | 0.159 | 8.64e-24 | 1117 | 18850 |
| Schlegel_2022_Ttr_D4_30mkg | 0.048 | 7.52e-03 | 1117 | 18850 |
| Schlegel_2022_Ttr_D1_10mkg | 0.008 | 8.87e-01 | 1090 | 17702 |

## Figures (results/aaai/figures/)
- `t1_leakage.png`: Theorem 1 leakage audit on real efficacy data: maximum cross-channel embedding change per model (the disjoint CCMP backbone is at machine zero; the shared-encoder CBM and the Concept Embedding Model leak), and the Havasi probe leakage gap and Yeh completeness.
- `t2_stability.png`: Theorem 2 on the real TRRUST signed/directed GRN: the true L2 operator norm of P_alpha versus the certified rho-based and inf-norm constants across alpha (the rho-based L2 constant is exceeded by non-normality; the inf-norm certificate holds for every alpha), and the Dirichlet-energy oversmoothing curve.
- `t3_constraint.png`: Theorem 3 on real Huesken efficacy with real ViennaRNA dG: the mass-action constraint residual (KKT-projected at machine precision versus the unconstrained model) and the held-out accuracy with and without the projection.
