#!/usr/bin/env bash
# Fetch the REAL rat (Rattus norvegicus) 3'UTR sequences for the seed-match
# annotation of the GSE184929 off-target DESeq2 tables. No data is generated.
#
# Build: the GSE184929 rat-liver DESeq2 tables are aligned to Rnor_6.0 (GEO
# Sample_data_processing: "Assembly: Rnor_6.0"). The build-matched 3'UTRs come
# from the Ensembl apr2020 archive (release 100, the Rnor_6.0 assembly); the
# current Ensembl site serves mRatBN7.2 and is the WRONG build for these IDs.
#
# Usage:
#   scripts/download_rat_3utr.sh            # download if absent, then checksum
#   scripts/download_rat_3utr.sh --verify   # only verify the manifest matches disk
#
# Output: data/raw/reference/rat_3utr_rnor6.fa (Ensembl gene id + 3'UTR, FASTA).
# Consumed by scripts/annotate_seedmatch.py.

set -euo pipefail

VERIFY_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --verify) VERIFY_ONLY=1 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REF="$REPO_ROOT/data/raw/reference"
DEST="$REF/rat_3utr_rnor6.fa"
MANIFEST="$REF/CHECKSUMS.sha256"
ARCHIVE="${ENSEMBL_ARCHIVE:-https://apr2020.archive.ensembl.org}"  # Ensembl 100, Rnor_6.0

mkdir -p "$REF"

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
    [[ -f "$MANIFEST" ]] || { echo "no manifest at $MANIFEST" >&2; exit 1; }
    ( cd "$REF" && sha256sum --check --strict CHECKSUMS.sha256 )
    exit $?
fi

if [[ -s "$DEST" ]]; then
    echo "have rat_3utr_rnor6.fa ($(grep -c '^>' "$DEST") sequences)"
else
    read -r -d '' QUERY <<'XML' || true
<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="FASTA" header="0" uniqueRows="1" count="" datasetConfigVersion="0.6">
<Dataset name="rnorvegicus_gene_ensembl" interface="default">
<Attribute name="ensembl_gene_id"/><Attribute name="3utr"/>
</Dataset></Query>
XML
    echo "fetch rat 3'UTRs (Rnor_6.0) from $ARCHIVE ..."
    if curl -fsS -G "$ARCHIVE/biomart/martservice" --data-urlencode "query=$QUERY" -o "$DEST"; then
        echo "       $(grep -c '^>' "$DEST") sequences"
    else
        rm -f "$DEST"
        echo "MISS   could not fetch from $ARCHIVE (archive BioMart is sometimes slow; retry," >&2
        echo "       or set ENSEMBL_ARCHIVE to another Rnor_6.0 release archive)" >&2
        exit 1
    fi
fi

( cd "$REF" && sha256sum ./*.fa > CHECKSUMS.sha256 2>/dev/null ) || true
echo "wrote manifest $MANIFEST"
