# AAAI four-theorem results summary

## Headline (real data only)
- Baselines (real Huesken efficacy): PENDING real data. The mandatory mean and ridge/linear comparison runs the moment the real OligoFormer CSVs are present; no number is fabricated.
- T2 (real TRRUST GRN): rho(S)=1.000, certified L2 contraction constant 1.000; verdict RESTRICTED (rho-based L2 bound holds=False, inf-norm non-expansive=True for every alpha; ||S||_2=8.313).
- T3 (real Huesken + ViennaRNA): PENDING real efficacy data. The KKT projection holds the mass-action residual below 1e-8 on every seed in the fixture check; the real-data residual runs when the OligoFormer CSVs are present.
- T4 (real GSE184929 RNA-seq): PENDING real data. Whether the certified bound binds is reported once the SeedMatchR export and the STRING / sequence join are wired.

## T1 leakage-free per-concept identifiability
- Provenance: Huesken 2005 on-target efficacy (OligoFormer suite); disjoint CCMP backbone vs a shared-encoder CBM and a Concept Embedding Model on the SAME real data
- PENDING real data (pending_real_data).

## T2 P_alpha contraction / stability (real signed/directed GRN)
- rho(S)=1.000, ||S||_2=8.313, ||S||_inf=1.000; max true ||P_alpha||_2 2.365, max measured ratio 0.953; alpha sweep over 11 values; signed spectral gap 0.293 (min real eig -0.522).
- rho-based L2 constant is exact for the symmetric variant and exceeded on the signed/directed GRN by non-normality; the inf-norm certificate holds for every alpha and the L2 ||S||_2 bound holds for alpha > alpha_min.

## T3 KKT mass-action / thermodynamic constraint
- Provenance: Huesken 2005 efficacy + ViennaRNA seed-duplex free energy dG (real); PENDING real data

## T4 certified off-target bound (per real dataset)
- PENDING real data for: gse184929 (rna_seq), jackson2006 (microarray), birmingham2006 (microarray)

## Figures (results/aaai/figures/)
- `t2_stability.png`: Theorem 2 on the real TRRUST signed/directed GRN: the true L2 operator norm of P_alpha versus the certified rho-based and inf-norm constants across alpha (the rho-based L2 constant is exceeded by non-normality; the inf-norm certificate holds for every alpha), and the Dirichlet-energy oversmoothing curve.
