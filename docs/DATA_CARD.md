# DATA CARD

This card documents the datasets used by CASCADE. Per the project contract
(see [CLAUDE.md](../CLAUDE.md)), raw data is fetched by
[scripts/download_data.sh](../scripts/download_data.sh) into `data/raw/`,
and parsed objects are cached under `data/cache/`. Neither directory is
tracked in git.

## 1. GSE184929 (primary off-target ground truth)

- **Source.** NCBI GEO accession GSE184929.
- **URL.** https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184929
- **FTP.** https://ftp.ncbi.nlm.nih.gov/geo/series/GSE184nnn/GSE184929/
- **Modality.** Bulk liver RNA-seq following in-vivo administration of
  GalNAc-conjugated siRNAs.
- **Organism.** Rattus norvegicus (rat, NCBI taxonomy ID 10116).
- **Tissue.** Liver (hepatocytes; GalNAc targets ASGPR-expressing cells).
- **Target gene.** Ttr (transthyretin).
- **Guides.** D1 and D4 (two seed-destabilization designs against Ttr).
- **Doses.** 10 mg/kg and 30 mg/kg. Four primary contrast columns:
  `D1_10mg`, `D1_30mg`, `D4_10mg`, `D4_30mg`.
- **Label exposed by loader.** Per-transcript log2 fold-change versus
  vehicle control.
- **Loader input.** `load_gse184929` reads a processed table (one transcript
  per row, one column per contrast). Default path
  `data/raw/GSE184929/GSE184929_processed_log2fc.tsv`. Contrast columns are
  auto-detected (names carrying a log2FC marker or matching `D<g>_<dose>mg`)
  or can be pinned via the `contrasts` argument.
- **Splits.** Held-out splits are constructed downstream of the loader.

### Rat-liver specificity caveat (READ THIS)

GSE184929 measures off-target effects in a single tissue (rat liver) of a
single species (rat) under a single delivery modality (GalNAc conjugation).
Effects observed here do not directly extrapolate to:

- other tissues (CNS, kidney, tumor) where uptake and turnover differ,
- other species (mouse, NHP, human) with different 3'UTR landscapes,
- non-GalNAc delivery (LNP, naked, conjugates to other receptors),
- chemical modifications outside the modification pattern used in this study.

CASCADE's containment theorem (Contribution 2) bounds off-target propagation
in the modeled GRN; it does not lift the species/tissue scope of this
ground truth. Headline numbers are reported with this caveat made explicit.

## 2. Huesken et al. 2005 (siRNA efficacy benchmark)

- **Source.** Huesken D et al., Nature Biotechnology 23, 995-1001 (2005).
- **DOI.** 10.1038/nbt1118
- **Size.** 2431 siRNAs across 34 mRNA targets.
- **Standard split.** 2182 train / 249 test (the conventional split used
  by downstream efficacy benchmarks).
- **Label exposed by loader.** Normalized knockdown efficacy in [0, 1].
- **Distribution.** Originally distributed as a journal supplementary
  table. No long-lived canonical mirror; set `HUESKEN_URL` for the
  download script or place the file at
  `data/raw/Huesken/huesken_2005.tsv` manually.

## 3. STRING v12 (gene regulatory / interaction backbone)

- **Source.** STRING v12.0, https://string-db.org
- **File.** `protein.links.v12.0.txt.gz` for the relevant organism.
  Default: rat (taxonomy ID 10116), matching GSE184929.
- **Format.** Space-separated triples `protein1 protein2 combined_score`
  with `combined_score` in 0 to 1000.
- **Filter.** Edges with `combined_score >= threshold`. Default threshold
  is 700 (STRING's "high confidence" cutoff). Configurable.
- **Adjacency.** Symmetric (the loader canonicalizes both edge directions
  and takes the max combined score), weighted by `combined_score / 1000`,
  then symmetric-normalized as S = D^(-1/2) A D^(-1/2).
- **Loader output.** scipy.sparse CSR `S`, ordered node id list, and a
  `to_edge_index()` helper that returns a torch_geometric edge_index
  (torch is imported lazily).

## 4. Replogle 2022 + Norman 2019 (Perturb-seq, downstream proxy)

- **Status.** Stub loaders only. Full integration lands in a follow-up
  commit.
- **Replogle 2022.** DOI 10.1016/j.cell.2022.05.013. Genome-scale
  Perturb-seq in K562 / RPE1.
- **Norman 2019.** DOI 10.1126/science.aax4438. Combinatorial Perturb-seq
  in K562.
- **Intended use.** Downstream-ripple validation: check whether
  CASCADE's containment claim holds against an out-of-distribution
  perturbation atlas.

## Licensing and reuse

- GEO and STRING are open-access. Cite the original deposits when
  redistributing parsed forms.
- The Huesken table is reused under the journal's terms; we do not
  redistribute the raw file, only loader code.
- Cached parquet / npz artifacts under `data/cache/` are derived works
  and are not committed.
