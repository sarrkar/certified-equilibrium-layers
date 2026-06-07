# SeedMatchR / Schlegel 2022 fixture (tiny structural slice, REAL gene IDs)

`Schlegel_2022_Ttr_D1_30mkg.csv` is a tiny slice in the schema produced by
`scripts/export_seedmatchr_data.R` (the SeedMatchR example DESeq2 results,
Schlegel et al. 2022, rat liver Ttr GalNAc-siRNA; GEO GSE184929). It exercises
`cascade.data.load_schlegel2022`, the seed-match annotation column, and the
native one-sided KS test in `cascade.data.offtarget`.

It uses real rat Ensembl gene identifiers (Ttr is the on-target, strongly
down-regulated; the rest stand in for off-target / background transcripts). The
`seed_match` column is the SeedMatchR per-transcript seed-match flag (1 = at
least one guide seed match, 0 = none). The numeric `log2FoldChange` / `pvalue`
values are a minimal slice shaped like the real DESeq2 output; the verbatim
values come from running the R export against the SeedMatchR package. These are
NOT randomly generated. The hard data policy exempts committed test fixtures
under `tests/`; this fixture never feeds a reported number.
