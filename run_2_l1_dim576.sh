#!/usr/bin/env bash
# =============================================================
# RUN 2 — L1 + dim=576 (8×H100 target)
# =============================================================
# Main candidate configuration.
#
# Preflight result (4×H100, 2026-03-20):
#   dim576 h=9 kv=3  → 108.51ms/step (head_dim=64 ✅)
#   dim544 h=8 kv=4  → 125.66ms/step (head_dim=68, non-power-of-2 ❌)
#   Decision: dim576 faster AND has more capacity → main candidate
#
# Target: records/track_10min_16mb/<date>_dim576_l1/
# Usage:
#   bash run_2_l1_dim576.sh
#
# This script is for /records/... only, not a core train_gpt.py PR.
# =============================================================
set -euo pipefail

RUN_ID="run2_l1_dim576_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="records/run2_l1_dim576"
mkdir -p "$LOG_DIR"

echo "======================================================="
echo "RUN 2 — L1 compression  |  dim=576, h=9, kv=3"
echo "Candidate config: head_dim=64, target 8×H100 @ 600s"
echo "Log dir: $LOG_DIR"
echo "======================================================="

GPUS="${GPUS:-8}"

# Model shape — preflight winner
export MODEL_DIM=576
export NUM_HEADS=9
export NUM_KV_HEADS=3
# head_dim = 576/9 = 64 → FlashAttention-optimal ✅

# L1 sparsity (post-step proximal shrinkage)
export LAMBDA_L1=1e-4
export L1_START_FRAC=0.80

export SEED=42
export VAL_LOSS_EVERY=200
export TRAIN_LOG_EVERY=10

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# TTT disabled by default (needs separate eval timing proof on 8×H100)
export ENABLE_TTT=0

LOG_FILE="$LOG_DIR/${RUN_ID}.log"
echo "Starting Run 2 | $(date)"
echo "GPUS=$GPUS  MODEL_DIM=$MODEL_DIM  NUM_HEADS=$NUM_HEADS  NUM_KV_HEADS=$NUM_KV_HEADS"

torchrun --standalone --nproc_per_node=$GPUS train_gpt_l1.py \
    2>&1 | tee "$LOG_FILE"

# ─── Parse results ────────────────────────────────────────────
echo ""
echo "======================================================="
echo "RUN 2 RESULTS"
echo "======================================================="

LAST_BPB=$(grep "val_bpb:" "$LOG_FILE" | tail -1 | sed 's/.*val_bpb:\([0-9.]*\).*/\1/' || echo "N/A")
BEST_BPB=$(grep "val_bpb:" "$LOG_FILE" | sed 's/.*val_bpb:\([0-9.]*\).*/\1/' | sort -n | head -1 || echo "N/A")
LAST_STEP=$(grep "step_avg:" "$LOG_FILE" | tail -1 | sed 's/.*step_avg:\([0-9.]*\)ms.*/\1/' || echo "N/A")
FINAL_INT8=$(grep "final_int8_zlib_roundtrip_exact" "$LOG_FILE" | tail -1 | sed 's/.*val_bpb:\([0-9]*\.[0-9]*\).*/\1/' || echo "N/A")

echo "Last val_bpb   : $LAST_BPB"
echo "Best val_bpb   : $BEST_BPB"
echo "Final int8 bpb : $FINAL_INT8"
echo "Last step_avg  : ${LAST_STEP}ms"
echo "Full log       : $LOG_FILE"
echo "======================================================="
echo "Done: $(date)"
