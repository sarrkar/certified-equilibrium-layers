#!/usr/bin/env bash
# Fetch CASCADE raw datasets into data/raw/ and write a sha256 manifest.
#
# Usage:
#   scripts/download_data.sh              # download missing files, then checksum
#   scripts/download_data.sh --verify     # only verify manifest matches disk
#   STRING_TAXID=9606 scripts/download_data.sh   # override organism (default: rat)
#   HUESKEN_URL=https://... scripts/download_data.sh
#
# Datasets: GSE184929 (GEO), STRING v12 (string-db.org), Huesken 2005.
# Idempotent: existing files are not re-downloaded.

set -euo pipefail

VERIFY_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --verify) VERIFY_ONLY=1 ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$REPO_ROOT/data/raw"
MANIFEST="$RAW/CHECKSUMS.sha256"

STRING_TAXID="${STRING_TAXID:-10116}"
STRING_VERSION="v12.0"
STRING_BASE="https://stringdb-downloads.org/download"
STRING_LINKS_URL="$STRING_BASE/protein.links.$STRING_VERSION/$STRING_TAXID.protein.links.$STRING_VERSION.txt.gz"
STRING_INFO_URL="$STRING_BASE/protein.info.$STRING_VERSION/$STRING_TAXID.protein.info.$STRING_VERSION.txt.gz"

GSE_BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE184nnn/GSE184929"
GSE_SOFT_URL="$GSE_BASE/soft/GSE184929_family.soft.gz"
GSE_SUPPL_URL="$GSE_BASE/suppl/"

HUESKEN_URL="${HUESKEN_URL:-}"

mkdir -p "$RAW/GSE184929" "$RAW/STRING" "$RAW/Huesken"

fetch() {
    local url="$1"; local out="$2"
    if [[ -f "$out" ]]; then
        echo "[skip] $out"
        return 0
    fi
    if [[ "$VERIFY_ONLY" -eq 1 ]]; then
        echo "[missing] $out"
        return 0
    fi
    echo "[fetch] $url -> $out"
    if ! curl -fL --retry 3 --retry-delay 5 -o "$out.part" "$url"; then
        rm -f "$out.part"
        echo "[fail]  $url" >&2
        return 1
    fi
    mv "$out.part" "$out"
}

if [[ "$VERIFY_ONLY" -eq 0 ]]; then
    fetch "$STRING_LINKS_URL" "$RAW/STRING/$STRING_TAXID.protein.links.$STRING_VERSION.txt.gz" || true
    fetch "$STRING_INFO_URL"  "$RAW/STRING/$STRING_TAXID.protein.info.$STRING_VERSION.txt.gz"  || true
    fetch "$GSE_SOFT_URL"     "$RAW/GSE184929/GSE184929_family.soft.gz" || true
    # GEO supplementary directory: mirror the listing so the user can pick the
    # processed file. Avoid recursive mirror to keep this script polite.
    if [[ ! -f "$RAW/GSE184929/suppl_index.html" ]]; then
        curl -fL --retry 3 --retry-delay 5 -o "$RAW/GSE184929/suppl_index.html" "$GSE_SUPPL_URL" || true
    fi
    if [[ -n "$HUESKEN_URL" ]]; then
        fetch "$HUESKEN_URL" "$RAW/Huesken/huesken_2005.tsv" || true
    else
        echo "[warn] HUESKEN_URL not set."
        echo "       Place the Huesken 2005 supplementary table at:"
        echo "       $RAW/Huesken/huesken_2005.tsv"
    fi
fi

# Write or verify the sha256 manifest.
if [[ "$VERIFY_ONLY" -eq 1 ]]; then
    if [[ ! -f "$MANIFEST" ]]; then
        echo "[err] no manifest at $MANIFEST" >&2
        exit 3
    fi
    ( cd "$RAW" && sha256sum -c "$(basename "$MANIFEST")" )
    exit 0
fi

(
    cd "$RAW"
    find . -type f \
        ! -name "$(basename "$MANIFEST")" \
        ! -name "*.part" \
        ! -name "suppl_index.html" \
        -print0 | sort -z | xargs -0 sha256sum
) > "$MANIFEST"

echo "[ok] manifest written: $MANIFEST"
wc -l "$MANIFEST"
