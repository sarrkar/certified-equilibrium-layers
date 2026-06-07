# DATA CARD

This card documents every dataset used by CASCADE and the two theory projects.
Per the project contract (see [CLAUDE.md](../CLAUDE.md)), raw data is fetched
into `data/raw/` and parsed objects are cached under `data/cache/`; neither is
tracked in git. Tiny committed fixtures under `tests/fixtures/` exercise each
loader.

## Role tags

The data layer holds ONLY real measured data. Per the hard data policy
([CLAUDE.md](../CLAUDE.md)), no generated data sits on any AAAI reported-number
path; `scripts/assert_no_synthetic.py` enforces this in CI. Every dataset carries
one of two real-data roles (`cascade.data.roles`):

- `efficacy_backbone`: on-target siRNA potency datasets. They measure how well a
  guide knocks down its INTENDED target. They do NOT validate the off-target
  containment certificate (Project 2); they are the on-target backbone only.
- `offtarget_certification`: measured off-target signatures. These are the
  datasets the off-target certificate is evaluated against.

Random-graph certificate-soundness verification (coverage / tightness / the C5
breakdown) is NOT a dataset and carries no role: it verifies the math, not
biology, and lives under `experiments/soundness/` with results labeled "no
biological claim".

## Master table

| Dataset | Citation | Size | Organism / cell line | Assay | Target | Role |
| --- | --- | --- | --- | --- | --- | --- |
| Huesken | Huesken 2005, 10.1038/nbt1118 | 2431 siRNA / 34 mRNA | human H1299 | branched-DNA residual mRNA | on-target | efficacy_backbone |
| Katoh (Taka) | Katoh, via OligoFormer | 702 siRNA | human HeLa | luciferase reporter | on-target | efficacy_backbone |
| Mix | Amarzguioui 2003, Harborth 2003 (44), Hsieh 2004, Khvorova 2003, Reynolds 2004 (240), Vickers 2003 (76), Ui-Tei 2004 (62) | pooled | various | various | on-target | efficacy_backbone |
| Shabalina | Shabalina 2006 (13-publication compilation) | 653 siRNA | various | various (thermo-filtered) | on-target | efficacy_backbone |
| GSE184929 | NCBI GEO GSE184929 | per-transcript | rat liver (Rattus norvegicus) | RNA-seq (DESeq2 log2FC) | off-target | offtarget_certification |
| Jackson 2003 | Jackson 2003 (Nat Biotechnol) | signature | human HeLa | microarray | off-target | offtarget_certification |
| Jackson 2006 | Jackson 2006 (RNA) | signature | human HeLa | microarray | off-target | offtarget_certification |
| Birmingham 2006 | Birmingham 2006 (Nat Methods) | seed-match calls | various | microarray | off-target | offtarget_certification |
| SeedSeq | Das 2013 (SeedSeq) | aggregated union | various | microarray (aggregated) | off-target | offtarget_certification |

The OligoFormer suite totals 3714 siRNAs across 75 mRNAs. The whole suite is
loaded by `cascade.data.load_oligoformer_efficacy` into one tidy table with the
antisense and sense sequences, target mRNA id, measured efficacy in [0, 1],
source study, cell line, and assay.

## Role: efficacy_backbone (on-target potency)

HONEST SCOPE: these datasets measure on-target knockdown potency. They are the
backbone for the CASCADE on-target / CCMP work and the efficacy benchmarks; they
do NOT provide off-target measurements and therefore do NOT validate the
off-target containment certificate of Project 2. Do not report them as
off-target evidence.

- **Huesken 2005.** 2431 siRNAs across 34 mRNA targets, human H1299, branched-DNA
  residual-mRNA readout. Normalized knockdown efficacy in [0, 1]. Standard split
  2182 train / 249 test (`cascade.data.load_huesken`). Originally a journal
  supplementary table; place at `data/raw/Huesken/huesken_2005.tsv` or pull the
  parsed CSV from the OligoFormer / siRNADiscovery release.
- **Katoh (Taka).** 702 siRNAs, luciferase reporter in HeLa.
- **Mix.** Amarzguioui 2003, Harborth 2003 (44), Hsieh 2004, Khvorova 2003,
  Reynolds 2004 (240), Vickers 2003 (76), Ui-Tei 2004 (62). The OligoFormer
  `mix` file carries a per-row `source_study`.
- **Shabalina 2006.** 653 siRNAs, a thermodynamic-filtered compilation over 13
  publications.
- **Loader input.** `data/raw/oligoformer/<source>.csv` for source in
  {huesken, katoh, mix, shabalina}. Prefer the parsed CSVs from the OligoFormer
  or siRNADiscovery public releases (siRNADiscovery uses a 2816-pair HUVK set
  from Huesken, Harborth, Ui-Tei, Vickers, Khvorova plus a 322-siRNA Simone
  validation set) rather than re-parsing the raw journal supplements. The loader
  is column-name tolerant and percent efficacies are rescaled to [0, 1].

### Per-study REAL loaders (`cascade.data.efficacy`)

REAL DATA ONLY. Each per-study loader reads `data/raw/efficacy/<study>.csv` (or
selects the study's rows out of a `siRNAEfficacyDB` master table) and RAISES
`FileNotFoundError` with the exact URL when the file is absent. Fetch with
`scripts/download_efficacy.sh` (writes `data/raw/efficacy/` + a sha256 manifest).
Sources: the OligoFormer release `data/` directory
(https://github.com/lulab/OligoFormer) and the siRNAEfficacyDB supplement
(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11665841/).

| Study | Loader | Citation | Size | Cell line | Assay | Real? |
| --- | --- | --- | --- | --- | --- | --- |
| Huesken 2005 | `load_huesken_efficacy` (split 2182/249) | Huesken 2005, 10.1038/nbt1118 | 2431 siRNA / 34 mRNA | H1299 | branched-DNA residual mRNA | REAL |
| Reynolds 2004 | `load_reynolds` | Reynolds 2004, 10.1038/nbt936 | 240 siRNA | HEK293 / HeLa | reporter / endogenous mRNA | REAL |
| Vickers 2003 | `load_vickers` | Vickers 2003, 10.1074/jbc.M210326200 | 76 siRNA | T24 / HeLa | endogenous mRNA | REAL |
| Harborth 2003 | `load_harborth` | Harborth 2003, 10.1089/108729003321629638 | 44 siRNA | HeLa | endogenous mRNA / protein | REAL |
| Ui-Tei 2004 | `load_uitei` | Ui-Tei 2004, 10.1093/nar/gkh247 | 62 siRNA | various | reporter | REAL |
| Khvorova 2003 | `load_khvorova` | Khvorova 2003, 10.1016/S0092-8674(03)00759-1 | ~90 siRNA | various | reporter | REAL |
| Katoh (Taka) | `load_katoh` | Katoh / Takayuki, via OligoFormer | 702 siRNA | HeLa | luciferase reporter | REAL |

`load_efficacy_suite` concatenates several studies and raises listing the missing
files and their URLs if any requested study has no real file. There is no
synthetic fallback on any of these paths.

## Role: offtarget_certification (measured off-target)

- **GSE184929 (RNA-seq, primary off-target ground truth).** GalNAc-siRNA
  seed-destabilization RNA-seq in rat liver, Ttr-targeting guides D1 and D4 at
  10 and 30 mg/kg; per-transcript DESeq2 log2 fold-change versus vehicle across
  the four contrasts `D1_10mg`, `D1_30mg`, `D4_10mg`, `D4_30mg`
  (`cascade.data.load_gse184929`). URL
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184929 . This is the
  highest-resolution off-target set used here. REAL.

  - **SeedMatchR export (recommended path).** The SeedMatchR R package ships the
    real DESeq2 results as example data: `Schlegel_2022_Ttr_D1_30mkg`,
    `Schlegel_2022_Ttr_D4_30mkg`, `Schlegel_2022_Ttr_D1_10mkg` (Schlegel et al.
    2022, the GSE184929 study). Run `scripts/export_seedmatchr_data.R` (needs R +
    `install.packages("SeedMatchR")`) to write per-contrast CSVs with
    log2FoldChange, pvalue, padj, and the seed-match annotation to
    `data/raw/offtarget/seedmatchr/`. `cascade.data.load_schlegel2022(dataset)`
    loads one contrast (per-transcript `log2fc`, `pvalue`, `seed_match`) and
    raises with the exact source if absent. REAL DATA ONLY: no fallback.
  - **Off-target evaluation metric (Theorem 4).** `seedmatch_ks_test` /
    `SeedMatchRDataset.ks()` is the SeedMatchR metric implemented natively in
    Python: a one-sided two-sample Kolmogorov-Smirnov test comparing the log2FC
    ECDF of seed-matched vs non-seed-matched transcripts, returning the D
    statistic and p-value. The D statistic matches `scipy.stats.ks_2samp`; the
    one-sided p uses the plain Smirnov tail `exp(-2 ne D^2)`, matching R's
    `ks.test` (the SeedMatchR backend).

  Rat-liver specificity caveat (READ THIS): GSE184929 measures off-target
  effects in a single tissue (rat liver), a single species, and a single
  delivery modality (GalNAc). Effects here do not directly extrapolate to other
  tissues, species, non-GalNAc delivery, or other chemical modification
  patterns. The containment theorem bounds off-target propagation in the modeled
  GRN; it does not lift the species/tissue scope of this ground truth.

- **Jackson 2003 and Jackson 2006 (microarray).** Microarray off-target
  signatures: down-regulated transcripts with 3'UTR seed complementarity. Some
  tables live in the paper supplements rather than a clean GEO matrix, so
  `cascade.data.load_jackson2003` / `load_jackson2006` parse the supplement
  tables and record the provenance string; where a GEO accession exists, point
  the loader at it instead.
- **Birmingham 2006 (microarray).** Seed-match off-target associations
  (`cascade.data.load_birmingham2006`).
- **SeedSeq (Das 2013).** The aggregated off-target transcriptome table, loaded
  by `cascade.data.load_seedseq` as a convenience union over the above with a
  `source` flag on every off-target call. If no aggregated table is present, the
  loader builds the union from whichever member sources are available.
- **Loader output.** Each off-target loader returns the siRNA id, antisense,
  seed (guide positions 2-8, derived from the antisense when no seed column is
  given), the off-target transcript, the effect size, the cell line or tissue,
  the assay (microarray vs RNA-seq), and the source.

HONEST RESOLUTION NOTE: Jackson and Birmingham are MICROARRAY sets and are
partly extracted from paper supplements. They are lower resolution and use older
transcript annotation than the GSE184929 RNA-seq ground truth. Treat them as
corroborating older evidence, not as a like-for-like substitute for the RNA-seq
measurements.

## Certificate soundness (random graphs, no biological claim, no role)

Certificate-soundness verification (coverage, tightness, the C5 breakdown) runs
on random-graph confidence ensembles built inline by the soundness experiments
under `experiments/soundness/` (matchings for the coverage regime, stars for the
tightness regime, Erdos-Renyi for the symmetric-normalized breakdown). These
verify the math, not biology; they are not datasets, carry no role, and are
never reported as measured efficacy or off-target evidence. They live outside
the data layer and outside `experiments/aaai/`, so the hard data policy
(`scripts/assert_no_synthetic.py`) does not scan them.

## Graph backbones

- **STRING v12.0** (https://string-db.org). `protein.links.v12.0.txt.gz` for the
  relevant organism (default rat, taxonomy 10116, matching GSE184929).
  Space-separated `protein1 protein2 combined_score` (0 to 1000). Edges with
  `combined_score >= threshold` (default 700) are kept; the adjacency is
  symmetric (max over both directions), weighted by `combined_score / 1000`, and
  symmetric-normalized as S = D^(-1/2) A D^(-1/2) (`cascade.data.load_string_grn`).
  `cascade.data.load_string_confidence` returns the raw confidence adjacency
  A_bar used by the random-subgraph noise model.
- **Reactome** and the **GEARS Perturb-seq atlases** (Replogle 2022,
  DOI 10.1016/j.cell.2022.05.013; Norman 2019, DOI 10.1126/science.aax4438) are
  tracked gaps (stub loaders with visible xfail tests); see CLAUDE.md.

## Licensing and reuse

- GEO and STRING are open-access; cite the original deposits when
  redistributing parsed forms.
- The efficacy tables and the microarray off-target supplements are reused under
  the respective journals' terms; we redistribute loader code and tiny synthetic
  fixtures only, not the raw tables.
- Cached parquet / npz artifacts under `data/cache/` are derived works and are
  not committed.
