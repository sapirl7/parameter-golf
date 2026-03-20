#!/bin/bash
set -euo pipefail

# =============================================================
# RUN 3 — L1 + SWE + dim=576 (8×H100 target)
# =============================================================
# Hardware-aligned candidate: dim576/h9/kv3 won preflight
# on H100 (108ms/step vs 125ms for dim544, head_dim=64).
# Run 3 adds Sliding Window Eval as final-eval overlay.
#
# Ablation order:
#   Run 1 — official baseline
#   Run 2 — L1 + dim576 (no SWE)
#   Run 3 — L1 + dim576 + SWE  ← this script
#
# This script is for /records/... only, not a core train_gpt.py PR.
# =============================================================
GPUS="${GPUS:-8}"
SEED="${SEED:-1337}"

export RUN_ID="run3_l1_swe_dim576_seed${SEED}"
export SEED="${SEED}"

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# Candidate shape from preflight
export MODEL_DIM=576
export NUM_HEADS=9
export NUM_KV_HEADS=3
# head_dim = 576/9 = 64 → FlashAttention-optimal ✅

# Architecture
export NUM_LAYERS="${NUM_LAYERS:-9}"
export MLP_MULT="${MLP_MULT:-2}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"

# Training budget
export ITERATIONS="${ITERATIONS:-20000}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-1200}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"

# Optimizer
export MATRIX_LR="${MATRIX_LR:-0.04}"
export SCALAR_LR="${SCALAR_LR:-0.04}"
export EMBED_LR="${EMBED_LR:-0.6}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.05}"
export HEAD_LR="${HEAD_LR:-0.008}"
export QK_GAIN_INIT="${QK_GAIN_INIT:-1.5}"
export LOGIT_SOFTCAP="${LOGIT_SOFTCAP:-30.0}"
export ROPE_BASE="${ROPE_BASE:-10000.0}"

# L1 compression (post-step proximal shrinkage)
export LAMBDA_L1="${LAMBDA_L1:-3e-5}"
export L1_START_FRAC="${L1_START_FRAC:-0.80}"

# SWE: final-eval only (SWE_ON_PERIODIC_VAL=0 saves training budget)
export SWE_STRIDE="${SWE_STRIDE:-64}"
export SWE_ON_PERIODIC_VAL=0
export SWE_ON_FINAL_VAL=1

# TTT disabled (needs separate eval timing proof on 8×H100)
export ENABLE_TTT=0

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

echo "============================================================"
echo "RUN 3: L1 + SWE + dim576"
echo "GPUS=${GPUS}  SEED=${SEED}"
echo "MODEL_DIM=${MODEL_DIM}  NUM_HEADS=${NUM_HEADS}  NUM_KV_HEADS=${NUM_KV_HEADS}"
echo "LAMBDA_L1=${LAMBDA_L1}  L1_START_FRAC=${L1_START_FRAC}"
echo "SWE_STRIDE=${SWE_STRIDE}  SWE_ON_FINAL_VAL=${SWE_ON_FINAL_VAL}"
echo "MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}"
echo "============================================================"

torchrun --standalone --nproc_per_node="${GPUS}" train_gpt_l1.py
