#!/bin/bash
# run_gpu_pcg.sh — Runs ALL PCG task files sequentially on GPU 0.
# Usage: run_gpu_pcg.sh <binary> <task_dir> <output_csv>
#
# Task file format (one experiment per line, 7 space-separated fields):
#   n nb cond shape_p clusters eps_list mtx_path
#
# eps_list is a comma-separated list of adapt_eps_prime values for
# --adapt_eps_list (e.g. "0.5,0.1,0.05,0.01"), or "-" to use the binary's
# single default (adapt_eps_prime=0.1).
#
# mtx_path is "-" for synthetic runs, or a path to a .mtx file for
# SuiteSparse runs. When set, n/cond/shape_p/clusters are placeholders the
# binary ignores (it loads the matrix's actual dimensions from disk); nb
# and eps_list are still honored.

set -euo pipefail

BINARY=$1
TASK_DIR=$2
OUTPUT_CSV=$3

export CUDA_VISIBLE_DEVICES=0

echo "[GPU 0] Running all PCG tasks sequentially -> ${OUTPUT_CSV}"

OVERALL_COUNT=0
OVERALL_FAILED=0

for TASK_FILE in "${TASK_DIR}"/tasks_gpu*.txt; do
    [ -f "${TASK_FILE}" ] || continue

    TOTAL=$(wc -l < "${TASK_FILE}")
    COUNT=0
    FAILED=0

    echo "[GPU 0] Processing ${TASK_FILE} (${TOTAL} experiments)"

    # Read 7 fields; eps_list and mtx are "-" placeholders when not used.
    while IFS=' ' read -r n nb cond shape_p clusters eps_list mtx; do
        # Skip blank lines and comments
        [[ -z "${n}" || "${n}" == "#"* ]] && continue
        COUNT=$((COUNT + 1))

        ARGS=(--n "${n}" --nb "${nb}" --cond "${cond}" \
              --shape_p "${shape_p}" --num_clusters "${clusters}" \
              --csv "${OUTPUT_CSV}")
        [[ -n "${eps_list}" && "${eps_list}" != "-" ]] && ARGS+=(--adapt_eps_list "${eps_list}")

        if [[ -z "${mtx}" || "${mtx}" == "-" ]]; then
            echo "[GPU 0] (${COUNT}/${TOTAL}) n=${n} nb=${nb} cond=${cond}"
        else
            echo "[GPU 0] (${COUNT}/${TOTAL}) MTX: $(basename "${mtx}")"
            ARGS+=(--mtx "${mtx}")
        fi

        # Wrap in `if` so set -e doesn't kill the loop on a failed experiment;
        # capture the binary's exit code from PIPESTATUS[0].
        if "${BINARY}" "${ARGS[@]}" 2>&1 | sed "s/^/  [GPU0] /"; then
            :
        else
            EXIT_CODE=${PIPESTATUS[0]}
            echo "[GPU 0] WARNING: experiment failed (exit ${EXIT_CODE}, mtx=${mtx}), continuing..."
            FAILED=$((FAILED + 1))
        fi
    done < "${TASK_FILE}"

    if [ "${FAILED}" -gt 0 ]; then
        echo "[GPU 0] Finished ${TASK_FILE}. ${COUNT}/${TOTAL} completed (${FAILED} failed)."
    else
        echo "[GPU 0] Finished ${TASK_FILE}. ${COUNT}/${TOTAL} completed."
    fi

    OVERALL_COUNT=$((OVERALL_COUNT + COUNT))
    OVERALL_FAILED=$((OVERALL_FAILED + FAILED))
done

NROWS=0
if [ -f "${OUTPUT_CSV}" ]; then
    NROWS=$(( $(wc -l < "${OUTPUT_CSV}") - 1 ))
fi

if [ "${OVERALL_FAILED}" -gt 0 ]; then
    echo "[GPU 0] All tasks done. ${OVERALL_COUNT} experiments total (${OVERALL_FAILED} failed), ${NROWS} CSV rows -> ${OUTPUT_CSV}"
else
    echo "[GPU 0] All tasks done. ${OVERALL_COUNT} experiments total, ${NROWS} CSV rows -> ${OUTPUT_CSV}"
fi
