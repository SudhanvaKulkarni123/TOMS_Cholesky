#!/usr/bin/env bash
# Download SPD test matrices from the SuiteSparse Matrix Collection.
#
# Usage:
#   ./download_spd_matrices.sh [output_dir]
#
# Default output_dir is ./mtx_matrices.  Files end up as:
#   <output_dir>/bcsstk14.mtx
#   <output_dir>/bcsstk16.mtx
#   ...
# (one flat directory of .mtx files, ready to feed to the benchmark)

set -euo pipefail

OUT_DIR="${1:-mtx_matrices}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

BASE_URL="https://suitesparse-collection-website.herokuapp.com/MM"

# Format: <Group>/<Name> -- order is roughly ascending size.
matrices=(
  "HB/bcsstk14"            #  n=1806     structural
  "Boeing/crystm01"        #  n=4875     crystal (well-conditioned)
  "HB/bcsstk16"            #  n=4884     structural (well-conditioned)
  "Cylshell/s3rmt3m3"      #  n=5357     cylindrical shell
  "Boeing/msc10848"        #  n=10848    structural
  "HB/bcsstk17"            #  n=10974    structural
  "HB/bcsstk18"            #  n=11948    structural (very ill-conditioned)
  "Boeing/cbuckle"         #  n=13681    buckling analysis
  "Boeing/crystm02"        #  n=13965    crystal (well-conditioned)
  "ACUSIM/Pres_Poisson"    #  n=14822    CFD pressure Poisson
  "UTEP/Dubcova1"          #  n=16129    PDE
  "Oberwolfach/gyro_K"     #  n=17361    MEMS gyroscope
  "ND/nd6k"                #  n=18000    3D mesh
  "Boeing/crystm03"        #  n=24696    crystal (well-conditioned)
  "GHS_psdef/wathen100"    #  n=30401    Wathen FEM
  "ND/nd12k"               #  n=36000    3D mesh (largest in set)
)

# Pretty-printed status helpers
ok()    { printf '  \033[32m[ok]\033[0m   %s\n' "$*"; }
skip()  { printf '  \033[33m[skip]\033[0m %s\n' "$*"; }
warn()  { printf '  \033[33m[warn]\033[0m %s\n' "$*"; }
fail()  { printf '  \033[31m[fail]\033[0m %s\n' "$*" >&2; }

downloaded=0
skipped=0
failed=0

for entry in "${matrices[@]}"; do
  group="${entry%/*}"
  name="${entry#*/}"
  tarball="${name}.tar.gz"
  final="${name}.mtx"

  if [[ -f "$final" ]]; then
    skip "$final  (already present)"
    skipped=$((skipped+1))
    continue
  fi

  printf '  [..]   %s ... ' "$entry"
  if ! curl -fsSL --retry 3 --retry-delay 2 \
        "${BASE_URL}/${entry}.tar.gz" -o "$tarball"; then
    printf '\n'
    fail "  download failed for $entry"
    rm -f "$tarball"
    failed=$((failed+1))
    continue
  fi
  printf 'downloaded, '

  if ! tar xzf "$tarball" 2>/dev/null; then
    printf '\n'
    fail "  extraction failed for $tarball"
    rm -f "$tarball"
    failed=$((failed+1))
    continue
  fi
  printf 'extracted.\n'

  # Move the main .mtx out of the per-matrix subdir, drop any *_b.mtx
  # right-hand-side files we don't need.
  if [[ -f "${name}/${name}.mtx" ]]; then
    mv "${name}/${name}.mtx" "./${final}"
    rm -rf "${name}/" "$tarball"
    ok "$final  ($(du -h "$final" | cut -f1))"
    downloaded=$((downloaded+1))
  else
    warn "expected ${name}/${name}.mtx not found inside tarball"
    rm -f "$tarball"
    failed=$((failed+1))
  fi
done

echo
echo "================================================================"
printf "  Downloaded: %d   Skipped: %d   Failed: %d\n" \
       "$downloaded" "$skipped" "$failed"
echo "  Output dir: $(pwd)"
echo "  Total size: $(du -sh . 2>/dev/null | cut -f1)"
echo "================================================================"

if (( failed > 0 )); then
  exit 1
fi