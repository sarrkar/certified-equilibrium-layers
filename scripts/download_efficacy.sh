#!/usr/bin/env bash
# Fetch the REAL on-target siRNA efficacy compilation into data/raw/efficacy/
# and write a sha256 manifest. No data is generated here: this only downloads
# published files. If a file cannot be fetched, the script prints the source page
# so it can be placed by hand.
#
# Usage:
#   scripts/download_efficacy.sh            # download missing files, then checksum
#   scripts/download_efficacy.sh --verify   # only verify the manifest matches disk
#
# Per-study URLs can be overridden (the upstream layout may move):
#   HUESKEN_URL=https://... REYNOLDS_URL=https://... scripts/download_efficacy.sh
#
# Sources:
#   OligoFormer release, data/ directory: https://github.com/lulab/OligoFormer
#   siRNAEfficacyDB supplement, PMC11665841:
#     https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11665841/
#
# Studies: Huesken, Reynolds, Vickers, Harborth, Ui-Tei, Khvorova, Katoh.

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
RAW="$REPO_ROOT/data/raw/efficacy"
MANIFEST="$RAW/CHECKSUMS.sha256"
# OligoFormer parsed datasets (Comparison methods/Monopoli-RF/datasets/). The
# 19mer column is the SENSE strand; cascade.data.efficacy recovers the antisense
# guide as its reverse complement and efficacy as 1 - expression/100.
OF="https://raw.githubusercontent.com/lulab/OligoFormer/main/Comparison%20methods/Monopoli-RF/datasets"
DB_PAGE="https://github.com/lulab/OligoFormer (Comparison methods/Monopoli-RF/datasets)"

mkdir -p "$RAW"

# study | local filename | default URL | source page (for manual fetch)
STUDIES=(
    "huesken|huesken.csv|${HUESKEN_URL:-$OF/Hu.csv}|$DB_PAGE"
    "katoh|katoh.csv|${KATOH_URL:-$OF/Taka.csv}|$DB_PAGE"
    "mix|mix.csv|${MIX_URL:-$OF/Mixset.csv}|$DB_PAGE"
)

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }

if [ "$VERIFY_ONLY" -eq 1 ]; then
    [ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST; run without --verify first" >&2; exit 1; }
    ( cd "$RAW" && if command -v sha256sum >/dev/null 2>&1; then sha256sum -c CHECKSUMS.sha256; else shasum -a 256 -c CHECKSUMS.sha256; fi )
    exit $?
fi

missing=0
for entry in "${STUDIES[@]}"; do
    IFS='|' read -r study fname url page <<< "$entry"
    dest="$RAW/$fname"
    if [ -f "$dest" ]; then
        echo "[have] $study -> $fname"
        continue
    fi
    echo "[get ] $study -> $url"
    if curl -fsSL "$url" -o "$dest.tmp" 2>/dev/null && [ -s "$dest.tmp" ]; then
        mv "$dest.tmp" "$dest"
    else
        rm -f "$dest.tmp"
        echo "[MISS] could not fetch $study from $url" >&2
        echo "       obtain it from: $page  and place it at $dest" >&2
        missing=$((missing + 1))
    fi
done

# Write a sha256 manifest over whatever real files are present.
( cd "$RAW" && find . -maxdepth 1 -type f ! -name 'CHECKSUMS.sha256' -printf '%P\n' | sort | while read -r f; do
    echo "$(sha "$RAW/$f")  $f"
done > CHECKSUMS.sha256 )
echo "wrote $MANIFEST"

if [ "$missing" -gt 0 ]; then
    echo "$missing file(s) still missing; the loaders will RAISE with the exact URL until placed." >&2
    exit 1
fi
echo "all efficacy files present under $RAW"
