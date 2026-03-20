#!/bin/bash
# =================================================================
# RUN 2: L1 Compression Branch @ dim=544
# Goal: Isolate our unique compression contribution.
# No SWE, no other levers. Clean ablation of L1 + width.
# =================================================================
#
# Usage (on 8xH100 cluster):
#   bash run_2_l1_dim544.sh 2>&1 | tee records/run2_l1_dim544/train.log
#
# SEED DISCIPLINE: 1 seed only. Multi-seed only if this qualifies for record.
# =================================================================

set -e

RUN_NAME="2026-03-XX_L1_Dim544"
RECORD_DIR="records/track_10min_16mb/${RUN_NAME}"
mkdir -p "$RECORD_DIR"

# --- Cluster environment snapshot ---
echo "=== RUN 2: L1 @ dim=544 ===" | tee -a "$RECORD_DIR/train.log"
git rev-parse HEAD 2>/dev/null | tee -a "$RECORD_DIR/train.log"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$RECORD_DIR/train.log"
python3 -c "import torch; print('torch:', torch.__version__)" | tee -a "$RECORD_DIR/train.log"
echo "---" | tee -a "$RECORD_DIR/train.log"

# --- Hyperparameters: dim=544, heads=8, kv=4 ---
# head_dim = 544/8 = 68 (non-power-of-2, acceptable)
# Estimated param budget: ~fits 16MB with L1 based on single-H100 experiments
export MODEL_DIM=544
export NUM_HEADS=8
export NUM_KV_HEADS=4

# L1 regularization: activate at 80% of training (proven config)
export LAMBDA_L1=1e-5
export L1_START_FRAC=0.80

# Official baseline-compatible settings
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export OMP_NUM_THREADS=8
export SEED=1337

GPUS=8

torchrun --standalone --nproc_per_node=$GPUS train_gpt_l1.py \
  2>&1 | tee -a "$RECORD_DIR/train.log"

echo "=== Run 2 complete. Artifact saved by train_gpt_l1.py ===" | tee -a "$RECORD_DIR/train.log"

# --- Stop/Go gate ---
echo ""
echo "CHECK: grep for 'val_bpb' in train.log to get the final metric."
echo "SUBMIT only if val_bpb improvement >= 0.005 nats over current SOTA."
echo "If artifact > 16MB, reduce LAMBDA_L1 to 2e-5 and retry."
