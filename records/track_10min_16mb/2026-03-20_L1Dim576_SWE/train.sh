#!/bin/bash
set -euo pipefail

# ============================================================
# FINAL RECORD CANDIDATE: L1 + dim576 + SWE
# Run from inside this records folder:
#   bash train.sh
# Override: SEED=2026 GPUS=8 bash train.sh
# ============================================================

GPUS="${GPUS:-8}"
SEED="${SEED:-1337}"

export RUN_ID="${RUN_ID:-record_l1_dim576_swe_seed${SEED}}"
export SEED="${SEED}"

# ---- Resolve repo root relative to this records folder ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---- Data / tokenizer ----
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# ---- Candidate shape (selected from H100 preflight) ----
export MODEL_DIM="${MODEL_DIM:-576}"
export NUM_HEADS="${NUM_HEADS:-9}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-3}"

# ---- Architecture ----
export NUM_LAYERS="${NUM_LAYERS:-9}"
export MLP_MULT="${MLP_MULT:-2}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"

# ---- Training budget ----
export ITERATIONS="${ITERATIONS:-20000}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-1200}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"

# ---- Optimizer / model defaults ----
export MATRIX_LR="${MATRIX_LR:-0.04}"
export SCALAR_LR="${SCALAR_LR:-0.04}"
export EMBED_LR="${EMBED_LR:-0.6}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.05}"
export HEAD_LR="${HEAD_LR:-0.008}"
export QK_GAIN_INIT="${QK_GAIN_INIT:-1.5}"
export LOGIT_SOFTCAP="${LOGIT_SOFTCAP:-30.0}"
export ROPE_BASE="${ROPE_BASE:-10000.0}"

# ---- Compression branch ----
export LAMBDA_L1="${LAMBDA_L1:-3e-5}"
export L1_START_FRAC="${L1_START_FRAC:-0.80}"

# ---- Eval overlay: SWE final-only ----
export SWE_STRIDE="${SWE_STRIDE:-64}"
export SWE_ON_PERIODIC_VAL="${SWE_ON_PERIODIC_VAL:-0}"
export SWE_ON_FINAL_VAL="${SWE_ON_FINAL_VAL:-1}"

# ---- TTT off by default ----
export ENABLE_TTT="${ENABLE_TTT:-0}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

echo "============================================================"
echo "FINAL RECORD CANDIDATE: L1 + dim576 + SWE"
echo "RUN_ID=${RUN_ID}  GPUS=${GPUS}  SEED=${SEED}"
echo "MODEL_DIM=${MODEL_DIM}  NUM_HEADS=${NUM_HEADS}  NUM_KV_HEADS=${NUM_KV_HEADS}"
echo "head_dim=$(( MODEL_DIM / NUM_HEADS ))  (power-of-2 → FlashAttn-optimal)"
echo "LAMBDA_L1=${LAMBDA_L1}  L1_START_FRAC=${L1_START_FRAC}"
echo "SWE_STRIDE=${SWE_STRIDE}  SWE_ON_FINAL_VAL=${SWE_ON_FINAL_VAL}"
echo "ENABLE_TTT=${ENABLE_TTT}  MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}"
echo "============================================================"

echo ""
echo "--- Environment snapshot ---"
pwd
python3 -V
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "----------------------------"
echo ""

torchrun --standalone --nproc_per_node="${GPUS}" train_gpt.py
