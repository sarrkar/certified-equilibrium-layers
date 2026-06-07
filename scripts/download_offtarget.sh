#!/usr/bin/env bash
# Fetch the REAL Schlegel et al. 2022 off-target DESeq2 tables (GEO GSE184929,
# rat liver Ttr GalNAc-siRNA) into data/raw/offtarget/seedmatchr/ as the CSVs the
# Python loader (cascade.data.load_schlegel2022) reads. No R and no SeedMatchR are
# needed: these are the authoritative GEO supplementary files (SeedMatchR's
# AnnotationHub copy derives from the same deposit). No data is generated here.
#
# Usage:
#   scripts/download_offtarget.sh            # download missing tables, then checksum
#   scripts/download_offtarget.sh --verify   # only verify the manifest matches disk
#
# Source: NCBI GEO GSE184929 supplementary,
#   https://ftp.ncbi.nlm.nih.gov/geo/series/GSE184nnn/GSE184929/suppl/
#
# The seed-match annotation (which transcripts carry a Ttr seed site) is NOT
# produced here: it needs a rat 3'UTR reference and the verified guide, and is
# computed natively in Python (cascade.features.seedmatch) as a separate step.

set -euo pipefail

VERIFY_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --verify) VERIFY_ONLY=1 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$REPO_ROOT/data/raw/offtarget/seedmatchr"
MANIFEST="$RAW/CHECKSUMS.sha256"
GEO="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE184nnn/GSE184929/suppl"

mkdir -p "$RAW"

# dataset (loader name) | GEO supplementary DESeq2 file (gzipped)
DATASETS=(
    "Schlegel_2022_Ttr_D1_30mkg|GSE184929_Fig3-TTR-2_groupD1_30mkg.vs.Mock.DESeq2.txt.gz"
    "Schlegel_2022_Ttr_D4_30mkg|GSE184929_Fig3-TTR-2_groupD4_30mkg.vs.Mock.DESeq2.txt.gz"
    "Schlegel_2022_Ttr_D1_10mkg|GSE184929_Fig3-TTR-1_groupD1_10mkg.vs.Mock.DESeq2.txt.gz"
)

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
    [[ -f "$MANIFEST" ]] || { echo "no manifest at $MANIFEST" >&2; exit 1; }
    ( cd "$RAW" && sha256sum --check --strict CHECKSUMS.sha256 )
    exit $?
fi

for spec in "${DATASETS[@]}"; do
    name="${spec%%|*}"
    remote="${spec##*|}"
    dest="$RAW/$name.csv"
    if [[ -s "$dest" ]]; then
        echo "have   $name.csv"
        continue
    fi
    echo "fetch  $name.csv  <-  $remote"
    if curl -fsSL "$GEO/$remote" | gunzip > "$dest"; then
        echo "       $(wc -l < "$dest") rows"
    else
        rm -f "$dest"
        echo "MISS   could not fetch $remote" >&2
        echo "       place it by hand from $GEO/$remote (gunzip to $dest)" >&2
    fi
done

# sha256 manifest of the real tables (idempotent; for --verify).
( cd "$RAW" && sha256sum ./*.csv > CHECKSUMS.sha256 2>/dev/null ) || true
echo "wrote manifest $MANIFEST"
