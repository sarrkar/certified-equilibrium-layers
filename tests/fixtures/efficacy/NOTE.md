# Efficacy fixtures (tiny, REAL sequences, not synthetic)

These committed fixtures exercise the per-study efficacy loaders in
`cascade.data.efficacy`. They are tiny and contain REAL published siRNA / miRNA
sequences, never randomly generated values:

- `huesken.csv`: a per-study file slice in the loader schema (with the standard
  `split` column). The first three rows reuse the sequences already committed
  under `tests/fixtures/oligoformer/huesken.csv`; the last two are the canonical
  anti-luciferase GL2 and anti-Lamin A/C control siRNAs from Elbashir et al.,
  Nature 411:494-498 (2001).
- `siRNAEfficacyDB.csv`: a master-format slice with a `source_study` column,
  used to test per-study selection out of a master compilation. Each row pairs a
  real published reference siRNA (GL2 / GL3 luciferase, Lamin A/C; Elbashir 2001)
  with the study label a per-study loader filters on.

These are minimal parser / split / cache fixtures, NOT a verbatim copy of any
study's full published table. The real tables are fetched by
`scripts/download_efficacy.sh` into `data/raw/efficacy/` (gitignored); until then
the loaders RAISE with the exact URL. The hard data policy exempts committed
test fixtures under `tests/`; these fixtures never feed a reported number.
