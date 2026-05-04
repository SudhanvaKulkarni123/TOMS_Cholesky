#!/bin/bash
# download_spd_matrices.sh
#
# Downloads a curated set of SPD matrices from SuiteSparse Matrix Collection.
# All are <40k rows, suitable for dense Cholesky benchmarking on H200.
#
# Memory usage: 8 * N^2 bytes per matrix (FP64 dense)
#   1k  →   8 MB       5k  → 200 MB      10k → 800 MB
#   20k → 3.2 GB      30k → 7.2 GB       40k → 12.8 GB
#
# Usage:
#   chmod +x download_spd_matrices.sh
#   ./download_spd_matrices.sh [output_dir]
#   ./pcg_bench --mtx matrices/bcsstk17.mtx --nb 256 --adapt_eps_list 0.5,0.1,0.01,0.001

set -e

OUTDIR="${1:-matrices}"
mkdir -p "$OUTDIR"

BASE_URL="https://suitesparse-collection-website.herokuapp.com/MM"

# ── Small (N < 5k): quick tests ──────────────────────────────────
declare -A SMALL=(
    # Structural mechanics — well-conditioned
    ["HB/bcsstk15"]="bcsstk15"       # 3948 x 3948, stiffness
    ["HB/bcsstk16"]="bcsstk16"       # 4884 x 4884, stiffness
    ["HB/bcsstk17"]="bcsstk17"       # 10974 x 10974, stiffness
    # Thermal
    ["Nasa/nasa2146"]="nasa2146"     # 2146 x 2146
    ["HB/nos4"]="nos4"               # 100 x 100, quick sanity check
    ["HB/nos7"]="nos7"               # 729 x 729
)

# ── Medium (5k–15k): main benchmark tier ─────────────────────────
declare -A MEDIUM=(
    ["DNVS/shipsec1"]="shipsec1"             # 140874 — too big, skip
    ["Rothberg/cfd1"]="cfd1"                 # 70656 — too big, skip
    ["Boeing/bcsstk35"]="bcsstk35"           # 30237 — fits
    ["Boeing/bcsstk36"]="bcsstk36"           # 23052
    ["Boeing/bcsstk38"]="bcsstk38"           # 8032
    ["Cunningham/qa8fk"]="qa8fk"             # 66127 — too big
    ["Wissgott/parabolic_fem"]="parabolic_fem" # 525825 — too big for dense
    ["GHS_psdef/wathen100"]="wathen100"      # 30401
    ["GHS_psdef/wathen120"]="wathen120"      # 36441
    ["Oberwolfach/t2dah_e"]="t2dah_e"        # 11445
)

# ── Curated final list (all confirmed <40k, SPD) ─────────────────
declare -A MATRICES=(
    # Tiny (sanity check)
    ["HB/nos4"]="nos4"                       #     100
    ["HB/nos7"]="nos7"                       #     729

    # Small structural
    ["HB/bcsstk15"]="bcsstk15"               #   3,948
    ["HB/bcsstk16"]="bcsstk16"               #   4,884

    # Medium structural
    ["Boeing/bcsstk38"]="bcsstk38"           #   8,032
    ["HB/bcsstk17"]="bcsstk17"               #  10,974

    # Thermal / FEM
    ["Oberwolfach/t2dah_e"]="t2dah_e"        #  11,445

    # Larger structural (still <40k)
    ["Boeing/bcsstk36"]="bcsstk36"           #  23,052
    ["Boeing/bcsstk35"]="bcsstk35"           #  30,237

    # GHS positive definite set
    ["GHS_psdef/wathen100"]="wathen100"      #  30,401
    ["GHS_psdef/wathen120"]="wathen120"      #  36,441
)

echo "Downloading ${#MATRICES[@]} SPD matrices to $OUTDIR ..."
echo ""

for group_name in "${!MATRICES[@]}"; do
    short="${MATRICES[$group_name]}"
    tarball="${short}.tar.gz"
    url="${BASE_URL}/${group_name}.tar.gz"
    mtx_file="$OUTDIR/${short}.mtx"

    if [ -f "$mtx_file" ]; then
        echo "  [skip] $short (already exists)"
        continue
    fi

    echo -n "  [download] $short ... "

    if curl -fsSL -o "$OUTDIR/$tarball" "$url" 2>/dev/null; then
        # The tarball contains: <group>/<name>/<name>.mtx
        # Extract just the .mtx file
        tar xzf "$OUTDIR/$tarball" -C "$OUTDIR" --strip-components=1 2>/dev/null || \
        tar xzf "$OUTDIR/$tarball" -C "$OUTDIR" 2>/dev/null

        # Find the .mtx file (might be nested)
        found=$(find "$OUTDIR" -name "${short}.mtx" -type f 2>/dev/null | head -1)
        if [ -n "$found" ] && [ "$found" != "$mtx_file" ]; then
            mv "$found" "$mtx_file"
        fi

        # Cleanup extracted dirs and tarball
        rm -f "$OUTDIR/$tarball"
        rm -rf "$OUTDIR/$short" 2>/dev/null

        if [ -f "$mtx_file" ]; then
            # Print matrix dimensions from the header
            dims=$(grep -v '^%' "$mtx_file" | head -1)
            echo "ok  ($dims)"
        else
            echo "FAILED (mtx not found in archive)"
        fi
    else
        echo "FAILED (download error)"
    fi
done

echo ""
echo "Done. Run benchmarks with:"
echo "  ./pcg_bench --mtx $OUTDIR/bcsstk17.mtx --nb 256 --adapt_eps_list 0.5,0.1,0.01,0.001 --csv results_mtx.csv"
echo ""
echo "Or run all at once:"
echo -n "  ./pcg_bench --nb 256 --adapt_eps_list 0.5,0.1,0.01,0.001 --csv results_mtx.csv"
for f in "$OUTDIR"/*.mtx; do
    [ -f "$f" ] && echo -n " --mtx $f"
done
echo ""