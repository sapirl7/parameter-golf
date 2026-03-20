#!/bin/bash
set -e

echo "=========================================================="
echo "STAGE 1: HARDWARE ALIGNMENT MICROBENCHMARK (8xH100)"
echo "=========================================================="
echo "This script profiles system speed (ms/step) across 3 "
echo "architecture widths. It does NOT evaluate val_bpb."
echo "Each configuration runs for ~180 seconds and is force-killed."
echo "Validation is DISABLED to prevent skewing the step times."
echo "=========================================================="

mkdir -p records/preflight_logs

export OMP_NUM_THREADS=8
# Default matches official baseline (NaiveBaseline README uses NCCL_IB_DISABLE=1).
# Override by setting NCCL_IB_DISABLE=0 in your environment if the cluster needs IB.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# Override via env: GPUS=1 MODE=smoke bash run_preflight.sh
GPUS="${GPUS:-8}"
MODE="${MODE:-full}"  # 'smoke' = exit after smoke check; 'full' = run all three benchmarks

# ============================================================
# CLUSTER ENVIRONMENT SNAPSHOT
# Run once at boot so we have a reproducible build fingerprint.
# ============================================================
echo ""
echo "--- Cluster Environment Snapshot ---"
git rev-parse HEAD 2>/dev/null && echo "[git commit captured]" || echo "[not a git repo context]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
python3 -V 2>&1 || true
python3 -c "import torch; print('torch:', torch.__version__)" 2>/dev/null || true
echo "------------------------------------"
echo ""

# ============================================================
# SMOKE CHECK (30 seconds on dim=512)
# Verifies that: (a) torchrun works, (b) 'step_avg:' appears
# in the log. If this fails, do NOT proceed to full benchmarks.
# ============================================================
SMOKE_LOG="records/preflight_logs/smoke.log"
echo "--- Smoke check (120s, dim=512, checking log format) ---"
echo "  (torch.compile takes ~60-90s cold start — waiting for first step_avg: line)"
set +e
MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=1 \
timeout 120s torchrun --standalone --nproc_per_node=$GPUS train_gpt.py > "$SMOKE_LOG" 2>&1
set -e

if grep -q 'step_avg:' "$SMOKE_LOG"; then
    echo "[PASS] 'step_avg:' found in smoke log."
    grep 'step_avg:' "$SMOKE_LOG" | tail -3
else
    echo "[FAIL] 'step_avg:' NOT found. Review log:"
    tail -10 "$SMOKE_LOG"
    echo "Aborting. Fix the log format parser before continuing."
    exit 1
fi
echo ""

if [ "$MODE" = "smoke" ]; then
    echo "[SMOKE MODE] Smoke check passed. Skipping full benchmarks."
    echo "When ready for 8xH100 preflight, run: bash run_preflight.sh"
    exit 0
fi

run_benchmark() {
    local dim=$1
    local heads=$2
    local kv=$3
    local name="dim${dim}_h${heads}_kv${kv}"
    local log="records/preflight_logs/${name}.log"

    echo ""
    echo "---> Starting benchmark: $name"
    echo "Wait ~180 seconds for profiling to complete..."

    # VAL_LOSS_EVERY=0 prevents the pipeline from pausing for validation.
    # TRAIN_LOG_EVERY=1 ensures we get a dense log of every step's dt (ms/step)
    set +e
    MODEL_DIM=$dim NUM_HEADS=$heads NUM_KV_HEADS=$kv VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=1 \
    timeout 180s torchrun --standalone --nproc_per_node=$GPUS train_gpt.py > "$log" 2>&1
    set -e

    # Parse results. Discard the first 50 steps (warmup/compile).
    # Official train_gpt.py log format: step_avg:X.XXms (e.g. "step_avg:43.50ms")
    local count=$(grep -c 'step_avg:' "$log" || true)
    
    if [ "$count" -lt 60 ]; then
        echo "[ERROR] Not enough steps completed ($count). Cluster might be hanging."
        echo "  (Tip: run 'grep step_avg records/preflight_logs/${name}.log | head -5' to verify log format)"
        return
    fi

    echo "Completed $count logged steps before 180s timeout."
    
    # Parse results — use sed (POSIX) instead of gawk 3-arg match().
    # Drop first 50 steps (torch.compile warmup noise), take the last stable step_avg.
    last_val=$(grep 'step_avg:' "$log" | tail -n +51 | tail -1 | sed 's/.*step_avg:\([0-9.]*\)ms.*/\1/')

    if [ -z "$last_val" ]; then
        echo "[ERROR] Failed to extract step_avg from log. Check: grep 'step_avg:' $log | head -3"
        return 1
    fi

    steps_in_600s=$(echo "$last_val" | awk '{printf "%.0f", (600.0 * 1000) / $1}')
    echo "=> step_avg (last stable value): ${last_val} ms/step"
    echo "=> ESTIMATED STEPS IN 600s:      ${steps_in_600s}"
}

run_benchmark 512 8 4
run_benchmark 544 8 4
run_benchmark 576 9 3

echo ""
echo "=========================================================="
echo "Preflight Stage 1 complete. Logs saved in records/preflight_logs/"
echo "=========================================================="
