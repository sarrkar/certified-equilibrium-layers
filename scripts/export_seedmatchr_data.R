#!/usr/bin/env Rscript
# Export the REAL SeedMatchR example DESeq2 results (Schlegel et al. 2022, rat
# liver Ttr GalNAc-siRNA; GEO GSE184929) to CSV for the Python loader in
# cascade.data.offtarget (load_schlegel2022). This only re-serializes the
# package's example data and, where a reference and guide are configured, the
# real SeedMatchR seed-match annotation. It generates no data.
#
# Requirements: R with the SeedMatchR package.
#   install.packages("SeedMatchR")        # CRAN
# Usage:
#   Rscript scripts/export_seedmatchr_data.R [out_dir]
# Default out_dir: data/raw/offtarget/seedmatchr
#
# Output per dataset: <dataset>.csv with the DESeq2 columns (gene id,
# log2FoldChange, pvalue, padj, baseMean) and, when the reference + guide are
# available, a seed_match count column. The Python loader reads log2FoldChange,
# pvalue, and the seed_match annotation and runs the one-sided KS test natively.

suppressPackageStartupMessages(library(SeedMatchR))

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else "data/raw/offtarget/seedmatchr"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

datasets <- c(
  "Schlegel_2022_Ttr_D1_30mkg",
  "Schlegel_2022_Ttr_D4_30mkg",
  "Schlegel_2022_Ttr_D1_10mkg"
)

# Ttr guide antisense sequences (Schlegel et al. 2022). CONFIRM against the paper
# / SeedMatchR vignette before relying on the seed-match annotation. Leave a
# guide as NA to export the DESeq2 results for that contrast WITHOUT a seed-match
# column rather than annotating with an unverified guide.
guides <- list(
  D1 = "UUAUAGAGCAAGAACACUGUUUU",
  D4 = NA_character_
)

# Optional rat reference for the real seed-match annotation. The SeedMatchR
# vignette loads an Ensembl annotation + sequence; set this up if available.
species <- tryCatch(load_species("rat"), error = function(e) NULL)

annotate_seed <- function(res, guide_label) {
  g <- guides[[guide_label]]
  if (is.null(species) || is.na(g)) {
    message("  no reference/guide for ", guide_label,
            "; exporting DESeq2 results only (seed-match column omitted)")
    return(res)
  }
  seed <- get_seed(g, "mer7")
  SeedMatchR(res, indices.name = "seed_match",
             gtf = species$gtf.path, seqs = species$seqs, seed = seed)
}

for (name in datasets) {
  e <- new.env()
  data(list = name, package = "SeedMatchR", envir = e)
  res <- get(name, envir = e)
  guide_label <- if (grepl("D4", name)) "D4" else "D1"
  res <- annotate_seed(res, guide_label)
  df <- as.data.frame(res)
  out <- file.path(out_dir, paste0(name, ".csv"))
  write.csv(df, out, row.names = TRUE)
  message("wrote ", out, " (", nrow(df), " transcripts)")
}

message("done. The Python loader is cascade.data.load_schlegel2022.")
