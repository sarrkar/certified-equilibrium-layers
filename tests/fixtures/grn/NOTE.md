# GRN fixture (tiny REAL TRRUST slice)

`trrust_rawdata.human.tsv` is a tiny slice in the TRRUST v2 raw format
(`TF<TAB>Target<TAB>Mode<TAB>PMID`). The rows are real curated human TF-target
regulations, chosen to exercise `cascade.data.load_trrust`:

- `TP53 -> CDKN1A`, `TP53 -> MDM2`, `TP53 -> BAX` (Activation),
- `MDM2 -> TP53` (Repression): the real p53 / MDM2 negative-feedback loop, which
  makes the network cyclic so the spectral radius rho(S) is positive (here 1.0),
- `TP53 -> MYC` (Repression), `MYC -> TERT`, `RELA -> IL6` (Activation),
- `TP53 -> EGFR` (Unknown): dropped from the signed network by default, retained
  with `include_unknown=True`.

These are real TRRUST relationships used to test the signed/directed transition,
the sign pattern, rho(S), and P_alpha propagation. The full TRRUST v2 tables are
fetched by `scripts/download_grn.sh`; until then the loader RAISES with the exact
URL. The hard data policy exempts committed test fixtures under `tests/`.
