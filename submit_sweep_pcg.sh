#!/bin/bash
#SBATCH --job-name=pcg-bench-sweep
#SBATCH --output=pcg_sweep_%j.log
#SBATCH --error=pcg_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4         # 4 tasks × 4 CPUs = 16 CPUs total
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --partition=normal
## Uncomment and set your project account:
## #SBATCH --account=<your_project_id>

# ===========================================================================
#  PCG benchmark sweep  —  4 GH200 GPUs
#
#  Compares None / Dpotrf-FP64 / Mixed-FP16 / Adaptive preconditioners for
#  FP64 PCG. Each pcg_bench invocation internally sweeps all built-in
#  eigenvalue distributions and the full --adapt_eps_list, so a "task" here
#  is one (n, nb, cond[, shape_p, clusters, eps_list]) point, or one MTX file.
#
#  Synthetic:  2 sizes x 3 condition numbers x 2 FP8 scaling modes = 12 tasks
#  MTX:        up to 15 SuiteSparse SPD matrices with n <= 16384, x 2 FP8
#              scaling modes (skipped if not on disk); each is factored and
#              solved with PCG like a synthetic matrix.
#
#  Each srun gets --ntasks=1 --gpus=1 --cpus-per-task=4 so there is
#  zero CPU/GPU contention between the 4 workers.
#
#  Pipeline:
#    1. generate_pcg_tasks.py  → tasks_gpu{0..3}.txt     (HOST)
#       (skipped when TASK_DIR is exported: those task files are used as-is)
#    2. inner_pcg_sweep.sh ×4  → results_gpu{0..3}.csv  (CONTAINER)
#    3. merge CSVs             → results_all.csv          (HOST)
#    4. plot_iters_precision.py → iters_precision_*.{png,pdf}  (HOST, uenv)
#       saved under ${SCRATCH}/slurm_pcg_results/<jobid>/, next to the Ozaki
#       sweep results.
# ===========================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths  (adjust these to your setup)
# ---------------------------------------------------------------------------
SCRIPT_DIR="${HOME}/bin/Cholesky_SC"       # source + scripts on host
# Workers run under uenv on the host filesystem (no container), so they see
# SCRIPT_DIR directly.  CONTAINER_DIR is kept as an alias for that.
CONTAINER_DIR="${SCRIPT_DIR}"
UENV_CMD=(uenv run prgenv-gnu/25.6:v2 --view=modules --)
# pcg_bench source and binary (export PCG_SRC / PCG_BIN before sbatch to run a
# different source, e.g. a baseline, concurrently with another sweep).
PCG_SRC="${PCG_SRC:-${SCRIPT_DIR}/pcg_bench.cu}"
PCG_BIN="${PCG_BIN:-${SCRIPT_DIR}/pcg_bench_uenv}"
# Final results land in the project directory (/bin/Cholesky_SC) by default.
# Export RESULTS_ROOT before sbatch to put them elsewhere (e.g. on scratch).
RESULTS_DIR="${RESULTS_ROOT:-${SCRIPT_DIR}/pcg_results}/${SLURM_JOB_ID}"
# Workers run inside the cuda13.1 container, which sees SCRATCH but not necessarily
# HOME, so they write per-GPU CSVs to a container-visible scratch dir first; the
# host then merges and copies everything into RESULTS_DIR below.
WORK_DIR="${SCRATCH}/pcg_results/${SLURM_JOB_ID}"
# Host path to the task dir. By default the task lists are generated fresh for
# this job (Step 1) into the container-visible work dir. Export TASK_DIR before
# sbatch to run a pre-made set of tasks_gpu{0..3}.txt instead.
GENERATE_TASKS=1
[ -n "${TASK_DIR:-}" ] && GENERATE_TASKS=0
HOST_TASK_DIR="${TASK_DIR:-${WORK_DIR}/tasks}"
# The cuda13.1 container bind-mounts the project at CONTAINER_DIR (see
# ~/.edf/cuda13.1.toml: ${HOME}/bin/Cholesky_SC -> ${SCRATCH}/Cholesky_SC), and
# SCRATCH 1:1. Workers can't see HOME, so translate the host task dir to its
# container-visible equivalent: replace a SCRIPT_DIR prefix with CONTAINER_DIR;
# paths already under SCRATCH pass through unchanged.
CONTAINER_TASK_DIR="${HOST_TASK_DIR}"   # no container: same path
# Plots (and a copy of the merged CSVs they are made from) go on scratch, in the
# same place the Ozaki sweeps save theirs.
PLOT_DIR="${SCRATCH}/slurm_pcg_results/${SLURM_JOB_ID}"
UENV_IMG="prgenv-gnu/25.6:v2"

# Per-family pcg_bench flags (appended to every task of that family by
# inner_pcg_sweep.sh).  SuiteSparse matrices are equilibrated (A := D A D,
# power-of-two D) and use pivot floors tuned on the equilibrated hard set
# (t2dah_e, bcsstk38, s3rmt3m3, msc10848, bcsstk18).  Synthetic matrices are
# NOT equilibrated (it hurt FP32 at cond=1e8) and keep the original floors
# (2^-4, 2^-11, 2^-18): a delta sweep at n=16384 over geometric/arithmetic x
# cond {1e4,1e8,1e12} found no per-tier choice that beats them.
MTX_FLAGS="--equil --pad_maxdiag --fp8_delta 0.3 --fp16_delta 3e-3 --fp32_delta 3.814697265625e-6"
SYN_FLAGS="--fp8_delta 0.0625 --fp16_delta 4.8828125e-4 --fp32_delta 3.814697265625e-6"
mkdir -p "${RESULTS_DIR}" "${WORK_DIR}" "${PLOT_DIR}"

echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPUs:      4 (GH200)"
echo "Start:     $(date)"
echo "Tasks:     ${HOST_TASK_DIR} (host)  ->  ${CONTAINER_TASK_DIR} (container)"
echo "Work dir:  ${WORK_DIR} (container-visible scratch)"
echo "Results:   ${RESULTS_DIR} (final)"
echo "Plots:     ${PLOT_DIR}"
echo "Source:    ${PCG_SRC} -> ${PCG_BIN} (uenv, no container)"
echo "SYN flags: ${SYN_FLAGS}"
echo "MTX flags: ${MTX_FLAGS}"
echo "=========================================="

# ---------------------------------------------------------------------------
# Step 1: Generate (or validate) task lists (HOST — has Python)
#
# generate_pcg_tasks.py writes tasks_gpu{0..3}.txt: the synthetic (n, cond)
# grid plus one task per SuiteSparse matrix it finds under <its-own-dir>/
# mtx_matrices or <its-own-dir>/matrices, each once per FP8 scaling mode.
# MTX paths are written project-relative (inner_pcg_sweep.sh cd's to the
# project root); missing matrices are skipped and listed at the end of stdout.
#
# Each line is 9 space-separated fields:
#
#   n nb cond shape_p clusters eps_list mtx_path scale panel
#
# eps_list is a comma-separated --adapt_eps_list sweep (e.g. "0.5,0.1,0.05"),
# or "-" for the binary's single default. mtx_path is "-" for synthetic, or
# a path to a .mtx file (n/cond/shape_p/clusters are then placeholders).
# scale is "mx" or "rowwise" (FP8 Schur-update scaling mode). panel is "gmw81"
# (GMW81 modified-Cholesky panel) or "potrf" (potrf_panel + TRSM).
# ---------------------------------------------------------------------------
if [ "${GENERATE_TASKS}" -eq 1 ]; then
    echo ""
    echo "=== Step 1: Generating task lists ==="
    python3 "${SCRIPT_DIR}/generate_pcg_tasks.py" "${HOST_TASK_DIR}" 4
    # Keep a copy of the task lists with the results (unless they already
    # live there, i.e. RESULTS_DIR is the work dir).
    if [ ! "${HOST_TASK_DIR}" -ef "${RESULTS_DIR}/tasks" ]; then
        mkdir -p "${RESULTS_DIR}/tasks"
        cp -f "${HOST_TASK_DIR}"/tasks_gpu*.txt "${RESULTS_DIR}/tasks/"
    fi
fi

echo ""
echo "=== Step 1: Validating task lists ==="
MISSING=0
for gpu_id in 0 1 2 3; do
    TASK_FILE="${HOST_TASK_DIR}/tasks_gpu${gpu_id}.txt"
    if [ ! -f "${TASK_FILE}" ]; then
        echo "  MISSING: ${TASK_FILE}"
        MISSING=1
    else
        echo "  OK: ${TASK_FILE} ($(wc -l < "${TASK_FILE}") lines)"
    fi
done
if [ "${MISSING}" -eq 1 ]; then
    echo ""
    echo "ERROR: one or more tasks_gpu{0..3}.txt files are missing under ${HOST_TASK_DIR}."
    echo "       Unset TASK_DIR to have them generated, or point it at a directory"
    echo "       that holds all four files (format documented above)."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Launch 4 parallel GPU workers INSIDE the cuda13.1 container
#
#   Each srun gets:
#     --ntasks=1          one process
#     --gpus=1            one GPU (SLURM assigns distinct GPUs)
#     --cpus-per-task=4   four dedicated cores
#     --exact             no over-subscription
#     --exclusive         no resource sharing with other srun steps
#
#   inner_pcg_sweep.sh receives gpu_id (0-3) and reads the matching task file.
#   CUDA_VISIBLE_DEVICES is also set inside inner_pcg_sweep.sh for safety.
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Launching 4 GPU workers ==="

for gpu_id in 0 1 2 3; do
    srun --exact --ntasks=1 --gpus=1 --cpus-per-task=4 --exclusive \
         "${UENV_CMD[@]}" bash -c '
             module load gcc/14.2.0 cuda/12.9.0
             exec bash "$@"' _ \
              "${CONTAINER_DIR}/inner_pcg_sweep.sh" \
              "${CONTAINER_DIR}" "${WORK_DIR}" "${CONTAINER_TASK_DIR}" "${gpu_id}" \
              "${SYN_FLAGS}" "${MTX_FLAGS}" "${PCG_SRC}" "${PCG_BIN}" \
         > "${WORK_DIR}/gpu${gpu_id}.log" 2>&1 &
    echo "  GPU ${gpu_id}: launched (pid $!)"
done

echo "Waiting for all 4 GPU workers ..."
wait
echo "All GPU workers complete."

# ---------------------------------------------------------------------------
# Step 3: Merge per-GPU CSVs into one file (HOST)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: Merging CSV results into project dir ==="

# Merge a set of per-GPU CSVs (same header) into one file. $1 = output path,
# $2 = basename pattern suffix (e.g. ".csv" or "_panel.csv").
merge_csvs() {
    local out="$1" suffix="$2" first=1 gpu csv
    for gpu in 0 1 2 3; do
        csv="${WORK_DIR}/results_gpu${gpu}${suffix}"
        [ -f "${csv}" ] || { echo "  WARNING: ${csv} not found, skipping"; continue; }
        if [ ${first} -eq 1 ]; then cp "${csv}" "${out}"; first=0
        else tail -n +2 "${csv}" >> "${out}"; fi
        echo "  Merged ${csv} ($(( $(wc -l < "${csv}") - 1 )) rows)"
    done
    if [ -f "${out}" ]; then
        echo "  Total: $(( $(wc -l < "${out}") - 1 )) rows -> ${out}"
    fi
}

# Per-run and per-panel merges, written to the project results dir.
merge_csvs "${RESULTS_DIR}/results_all.csv"       ".csv"
merge_csvs "${RESULTS_DIR}/results_all_panel.csv" "_panel.csv"

# Copy the raw per-GPU CSVs and worker logs into the project dir too, so the
# complete job output lives under /bin/Cholesky_SC.
if [ ! "${WORK_DIR}" -ef "${RESULTS_DIR}" ]; then
    cp -f "${WORK_DIR}"/results_gpu*.csv "${RESULTS_DIR}/" 2>/dev/null || true
    cp -f "${WORK_DIR}"/gpu*.log         "${RESULTS_DIR}/" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Step 4: Plots (HOST, under uenv for matplotlib)
#
# plot_iters_precision.py reads figAB_run.csv / figAB_panel.csv from one
# directory, so the merged CSVs are copied to PLOT_DIR under those names. One
# grid figure per (spectrum, FP8 scaling mode) and one figure per (SuiteSparse
# matrix, FP8 scaling mode). The workers run with --fact_prec f32, hence
# --base-prec FP32. A plotting failure does not fail the job.
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4: Generating plots ==="
if [ -f "${RESULTS_DIR}/results_all.csv" ] && [ -f "${RESULTS_DIR}/results_all_panel.csv" ]; then
    cp -f "${RESULTS_DIR}/results_all.csv"       "${PLOT_DIR}/figAB_run.csv"
    cp -f "${RESULTS_DIR}/results_all_panel.csv" "${PLOT_DIR}/figAB_panel.csv"
    uenv run "${UENV_IMG}" --view=modules -- bash -c \
        'module load python; python3 "$1/plot_iters_precision.py" "$2" -o "$2" --base-prec FP32' \
        _ "${SCRIPT_DIR}" "${PLOT_DIR}" \
        || echo "  WARNING: plotting failed; CSVs are in ${PLOT_DIR}"
else
    echo "  WARNING: merged CSVs missing, skipping plots"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Results:  ${RESULTS_DIR}/"
echo "  Per-run CSV:   ${RESULTS_DIR}/results_all.csv"
echo "  Per-panel CSV: ${RESULTS_DIR}/results_all_panel.csv"
echo "  Logs:          ${RESULTS_DIR}/gpu{0..3}.log"
echo "  Plots:         ${PLOT_DIR}/iters_precision_*.{png,pdf}"
echo "=========================================="
