#!/usr/bin/env bash
# Fetch the REAL Patel et al. 2022 (PMID 35332222) human 6mer-seed off-target tables
# from NCBI GEO into data/raw/offtarget/patel/. No data is generated here.
#
# Datasets (genome build confirmed hg38 = GRCh38 from each GEO record's
# Sample_data_processing "Genome_build: hg38"):
#   GSE164289 (HeyA8 ovarian): authors' DESeq2 differential-expression tables.
#     - siMxy  = siGGGGGC (transfection: 10 nM siGGGGGC; seed GGGGGC, 3'UTR site GCCCCC)
#     - si34a  = siGGCAGU (transfection: 10 nM siGGCAGU; seed GGCAGU, 3'UTR site ACTGCC)
#     vs siNT1 control.
#   GSE196839 (PC3 + HeyA8): raw HTSeq/featureCounts counts only (siMxy = siGGGGGC
#     confirmed; single replicate per condition; NO authors' DE table).
#
# Usage: scripts/download_patel.sh            # download missing files, then checksum
#        scripts/download_patel.sh --verify   # only verify the manifest matches disk

set -euo pipefail
VERIFY_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --verify) VERIFY_ONLY=1 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$REPO_ROOT/data/raw/offtarget/patel"
MANIFEST="$RAW/CHECKSUMS.sha256"
G164="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164289/suppl"
G196="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE196nnn/GSE196839/suppl"
mkdir -p "$RAW"

# local name | remote gzipped URL
FILES=(
    "GSE164289_siGGGGGC_vs_siNT1_48h.csv|$G164/GSE164289_siMxy_48h_versus_siNT1_S2_OM_48h_deseq_diff_expression.csv.gz"
    "GSE164289_siGGCAGU_vs_siNT1_48h.csv|$G164/GSE164289_si34a_48h_versus_siNT1_S2_OM_48h_deseq_diff_expression.csv.gz"
    "GSE196839_counts.txt|$G196/GSE196839_DISE-059.htseq.all.counts.txt.gz"
)

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }

if [ "$VERIFY_ONLY" -eq 1 ]; then
    [ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST; run without --verify first" >&2; exit 1; }
    ( cd "$RAW" && if command -v sha256sum >/dev/null 2>&1; then sha256sum -c CHECKSUMS.sha256; else shasum -a 256 -c CHECKSUMS.sha256; fi )
    exit $?
fi

missing=0
for entry in "${FILES[@]}"; do
    IFS='|' read -r fname url <<< "$entry"
    dest="$RAW/$fname"
    if [ -s "$dest" ]; then echo "[have] $fname"; continue; fi
    echo "[get ] $fname  <-  $url"
    if curl -fsSL "$url" 2>/dev/null | gunzip > "$dest.tmp" 2>/dev/null && [ -s "$dest.tmp" ]; then
        mv "$dest.tmp" "$dest"; echo "       $(wc -l < "$dest") rows"
    else
        rm -f "$dest.tmp"; echo "[MISS] could not fetch $fname from $url" >&2; missing=$((missing + 1))
    fi
done

( cd "$RAW" && find . -maxdepth 1 -type f ! -name 'CHECKSUMS.sha256' -printf '%P\n' | sort | while read -r f; do
    echo "$(sha "$RAW/$f")  $f"
done > CHECKSUMS.sha256 )
echo "wrote $MANIFEST"
[ "$missing" -gt 0 ] && { echo "$missing file(s) missing" >&2; exit 1; }
echo "all Patel files present under $RAW"
