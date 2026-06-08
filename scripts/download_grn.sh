#!/usr/bin/env bash
# Fetch the REAL signed/directed regulatory networks into data/raw/grn/ and write
# a sha256 manifest. This only downloads published files; it generates no data.
#
# Usage:
#   scripts/download_grn.sh            # download missing files, then checksum
#   scripts/download_grn.sh --verify   # only verify the manifest matches disk
#
# URLs can be overridden:
#   TRRUST_HUMAN_URL=https://... TRRUST_MOUSE_URL=https://... scripts/download_grn.sh
#
# Sources:
#   TRRUST v2 (curated, signed, directed TF-target): https://www.grnpedia.org/trrust/
#   CollecTRI (signed, directed TF-target) via OmniPath REST: https://omnipathdb.org/
#   DoRothEA regulons (confidence A-E, mode of regulation): https://saezlab.github.io/dorothea/
# STRING v12 (undirected, unsigned) is handled by scripts/download_data.sh.

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
RAW="$REPO_ROOT/data/raw/grn"
MANIFEST="$RAW/CHECKSUMS.sha256"
TRRUST="https://www.grnpedia.org/trrust/data"

mkdir -p "$RAW"

OMNIPATH="https://omnipathdb.org/interactions?datasets=collectri&genesymbols=yes&fields=sources,references"

# name | local filename | URL | source page
FILES=(
    "trrust_human|trrust_rawdata.human.tsv|${TRRUST_HUMAN_URL:-$TRRUST/trrust_rawdata.human.tsv}|https://www.grnpedia.org/trrust/"
    "trrust_mouse|trrust_rawdata.mouse.tsv|${TRRUST_MOUSE_URL:-$TRRUST/trrust_rawdata.mouse.tsv}|https://www.grnpedia.org/trrust/"
    "collectri_mouse|collectri_mouse.tsv|${COLLECTRI_MOUSE_URL:-$OMNIPATH&organisms=10090}|https://omnipathdb.org/CollecTRI"
)

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }

if [ "$VERIFY_ONLY" -eq 1 ]; then
    [ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST; run without --verify first" >&2; exit 1; }
    ( cd "$RAW" && if command -v sha256sum >/dev/null 2>&1; then sha256sum -c CHECKSUMS.sha256; else shasum -a 256 -c CHECKSUMS.sha256; fi )
    exit $?
fi

missing=0
for entry in "${FILES[@]}"; do
    IFS='|' read -r name fname url page <<< "$entry"
    dest="$RAW/$fname"
    if [ -f "$dest" ]; then
        echo "[have] $name -> $fname"
        continue
    fi
    echo "[get ] $name -> $url"
    if curl -fsSL "$url" -o "$dest.tmp" 2>/dev/null && [ -s "$dest.tmp" ]; then
        mv "$dest.tmp" "$dest"
    else
        rm -f "$dest.tmp"
        echo "[MISS] could not fetch $name from $url" >&2
        echo "       obtain it from: $page  and place it at $dest" >&2
        missing=$((missing + 1))
    fi
done

echo "[note] DoRothEA: export a CSV (tf,target,confidence,mor) to $RAW/dorothea.csv"
echo "       from https://saezlab.github.io/dorothea/ (decoupleR / OmniPath)."

( cd "$RAW" && find . -maxdepth 1 -type f ! -name 'CHECKSUMS.sha256' -printf '%P\n' | sort | while read -r f; do
    echo "$(sha "$RAW/$f")  $f"
done > CHECKSUMS.sha256 )
echo "wrote $MANIFEST"

if [ "$missing" -gt 0 ]; then
    echo "$missing file(s) missing; the loaders RAISE with the exact URL until placed." >&2
    exit 1
fi
echo "all TRRUST files present under $RAW"
