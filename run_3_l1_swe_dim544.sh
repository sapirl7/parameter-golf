#!/usr/bin/env bash
# =============================================================
# RUN 3 — L1 + Sliding Window Eval (SWE), dim=544
# =============================================================
# Evaluation overlay only: SWE does NOT change training dynamics.
# Same training setup as Run 2, but eval_val is replaced with
# eval_val_swe (stride=512) so each token gets up to 512 tokens
# of extra left-context at eval time.
#
# Expected improvement over Run 2 (plain L1, dim=544): ~0.001–0.003 bpb
#
# Usage:
#   bash run_3_l1_swe_dim544.sh
#
# Requires: train_gpt_l1.py with SWE overlay (SWE_STRIDE env var supported)
# =============================================================
set -euo pipefail

RUN_ID="run3_l1_swe_dim544_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="records/run3_l1_swe_dim544"
mkdir -p "$LOG_DIR"

echo "======================================================="
echo "RUN 3 — L1 + SWE  |  dim=544, SWE_STRIDE=512"
echo "Log dir: $LOG_DIR"
echo "======================================================="

# GPU count from env or default (set to match your cluster)
GPUS="${GPUS:-5}"

# Core hyperparams (identical to Run 2 training)
export MODEL_DIM=544
export NUM_HEADS=8
export NUM_KV_HEADS=4
export LAMBDA_L1=1e-4
export L1_START_FRAC=0.80
export SEED=42

# SWE evaluation overlay
export SWE_STRIDE=512       # each val token gets up to 512 extra context tokens

# Training identical to baseline
export VAL_LOSS_EVERY=200   # evaluate every 200 steps
export TRAIN_LOG_EVERY=10

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

LOG_FILE="$LOG_DIR/${RUN_ID}.log"
echo "Starting Run 3 | $(date)"
echo "GPUS=$GPUS  MODEL_DIM=$MODEL_DIM  SWE_STRIDE=$SWE_STRIDE"

torchrun --standalone --nproc_per_node=$GPUS train_gpt_l1.py \
    2>&1 | tee "$LOG_FILE"

# ─── Parse results ────────────────────────────────────────────
echo ""
echo "======================================================="
echo "RUN 3 RESULTS"
echo "======================================================="

LAST_BPB=$(grep "val_bpb:" "$LOG_FILE" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
LAST_STEP_AVG=$(grep "step_avg:" "$LOG_FILE" | tail -1 | grep -oP 'step_avg:\K[0-9.]+' || echo "N/A")
BEST_BPB=$(grep "val_bpb:" "$LOG_FILE" | grep -oP 'val_bpb:\K[0-9.]+' | sort -n | head -1 || echo "N/A")

echo "Last val_bpb   : $LAST_BPB"
echo "Best val_bpb   : $BEST_BPB"
echo "Last step_avg  : ${LAST_STEP_AVG}ms"
echo "Full log       : $LOG_FILE"
echo "======================================================="
echo "Done: $(date)"
