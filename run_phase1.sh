#!/bin/bash
# Parameter Golf — Phase 1 RunPod Runner
# ========================================
#
# Uses train_gpt_phase1.py (direct fork, no auto-patcher needed).
#
# Usage:
#   1. SSH into your RunPod 1xH100 pod
#   2. Clone YOUR fork: git clone https://github.com/sapirl7/parameter-golf.git
#   3. Run: bash run_phase1.sh setup
#   4. Run: bash run_phase1.sh exp0   (baseline reproduction)
#   5. Run: bash run_phase1.sh exp1   (QAT only)
#   6. Run: bash run_phase1.sh exp2   (QAT + L1 sweep)
#   7. Run: bash run_phase1.sh results
#
# Total cost estimate: ~$5-8 on 1xH100 RunPod (~2 hours)

set -euo pipefail

REPO_DIR="/workspace/parameter-golf"
TRAIN_SCRIPT="train_gpt_phase1.py"

# Common env vars
export DATA_PATH="$REPO_DIR/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="$REPO_DIR/data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export VAL_LOSS_EVERY=200
export TRAIN_LOG_EVERY=50

run_experiment() {
    local run_id=$1
    shift
    
    echo "============================================"
    echo "  Experiment: $run_id"
    echo "  Time:       $(date)"
    echo "  Env overrides: $*"
    echo "============================================"
    
    cd "$REPO_DIR"
    mkdir -p logs
    
    # All positional args are KEY=VALUE env overrides
    env RUN_ID="$run_id" "$@" \
        torchrun --standalone --nproc_per_node=1 "$TRAIN_SCRIPT" \
        2>&1 | tee "logs/${run_id}_console.txt"
    
    echo ""
    echo "=== RESULTS for $run_id ==="
    local logfile="logs/${run_id}.txt"
    if [ -f "logs/${run_id}.txt" ]; then
        grep -E "(val_bpb|val_loss|final_int8|qat_alpha|model_params|weight_sparsity|Serialized model)" "logs/${run_id}.txt" | tail -20 || true
    else
        echo "(log file not found at $logfile)"
    fi
    echo "============================================"
    echo ""
}

case "${1:-help}" in
    setup)
        echo "=== Setting up Parameter Golf ==="
        cd /workspace
        
        if [ ! -d "$REPO_DIR" ]; then
            git clone https://github.com/sapirl7/parameter-golf.git
        fi
        cd "$REPO_DIR"
        
        echo "Checking train_gpt_phase1.py exists..."
        if [ ! -f "$TRAIN_SCRIPT" ]; then
            echo "ERROR: $TRAIN_SCRIPT not found. Did you push it to your fork?"
            exit 1
        fi
        
        echo "Smoke-checking Python syntax..."
        python3 -m py_compile "$TRAIN_SCRIPT"
        echo "py_compile: OK"
        
        echo ""
        echo "Downloading data (full validation + 10 training shards)..."
        python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
        
        echo ""
        echo "=== Setup complete! ==="
        echo "Next: bash run_phase1.sh exp0"
        ;;
    
    exp0)
        echo "=== Exp 0: Baseline reproduction (QAT disabled) ==="
        echo "Goal: Verify Phase 1 code reproduces baseline ~1.2244 BPB"
        echo "QAT_START_FRAC=2.0 disables QAT (alpha never activates)"
        echo ""
        run_experiment "exp0_baseline" \
            QAT_START_FRAC=2.0 \
            LAMBDA_L1=0.0
        ;;
    
    exp1)
        echo "=== Exp 1: QAT only (no L1) ==="
        echo "Goal: Measure quantization gap reduction"
        echo ""
        run_experiment "exp1_qat_080" \
            QAT_START_FRAC=0.80 \
            LAMBDA_L1=0.0
        ;;
    
    exp2)
        echo "=== Exp 2: QAT + L1 sweep (5 runs) ==="
        echo "Goal: Find optimal L1 coefficient for compression"
        echo ""
        
        for LAMBDA in 0.000001 0.000005 0.00001 0.00003 0.0001; do
            run_experiment "exp2_l1_${LAMBDA}" \
                QAT_START_FRAC=0.80 \
                LAMBDA_L1="${LAMBDA}" \
                L1_START_FRAC=0.85
        done
        
        echo ""
        echo "=== L1 Sweep Summary ==="
        for LAMBDA in 0.000001 0.000005 0.00001 0.00003 0.0001; do
            logfile="$REPO_DIR/logs/exp2_l1_${LAMBDA}.txt"
            if [ -f "$logfile" ]; then
                bpb=$(grep "final_int8_zlib_roundtrip_exact" "$logfile" | tail -1 | sed -n 's/.*val_bpb:\([0-9.]*\).*/\1/p')
                bytes=$(grep "Serialized model int8+zlib:" "$logfile" | tail -1 | sed -n 's/.*: \([0-9]*\) bytes.*/\1/p')
                echo "  lambda=$LAMBDA  bpb=${bpb:-?}  bytes=${bytes:-?}"
            fi
        done
        ;;
    
    results)
        echo "=== All Experiment Results ==="
        echo ""
        cd "$REPO_DIR"
        
        for logfile in logs/exp*.txt; do
            if [ -f "$logfile" ]; then
                run=$(basename "$logfile" .txt)
                echo "--- $run ---"
                grep -E "(final_int8_zlib_roundtrip|model_params|Serialized model|weight_sparsity)" "$logfile" | tail -5 || true
                echo ""
            fi
        done
        ;;
    
    compare)
        echo "=== Phase 1 Comparison Table ==="
        echo ""
        cd "$REPO_DIR"
        printf "%-25s %12s %12s %12s\n" "Experiment" "Pre-Q BPB" "Post-Q BPB" "Comp. Bytes"
        printf "%-25s %12s %12s %12s\n" "---------" "--------" "--------" "----------"
        
        for logfile in logs/exp*.txt; do
            if [ -f "$logfile" ]; then
                run=$(basename "$logfile" .txt)
                pre_bpb=$(grep "step:.*val_bpb:" "$logfile" | tail -1 | sed -n 's/.*val_bpb:\([0-9.]*\).*/\1/p')
                post_bpb=$(grep "final_int8_zlib_roundtrip_exact" "$logfile" | sed -n 's/.*val_bpb:\([0-9.]*\).*/\1/p')
                comp_bytes=$(grep "Serialized model int8+zlib:" "$logfile" | sed -n 's/.*: \([0-9]*\) bytes.*/\1/p')
                printf "%-25s %12s %12s %12s\n" "$run" "${pre_bpb:-?}" "${post_bpb:-?}" "${comp_bytes:-?}"
            fi
        done
        ;;
    
    *)
        echo "Parameter Golf — Phase 1 Runner"
        echo ""
        echo "Usage: bash run_phase1.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup    - Clone fork, download data, verify script"
        echo "  exp0     - Baseline reproduction (QAT disabled)"
        echo "  exp1     - QAT only (measure quant gap reduction)"
        echo "  exp2     - QAT + L1 sweep (5 runs, find optimal lambda)"
        echo "  results  - Show all experiment results"
        echo "  compare  - Show comparison table"
        echo ""
        echo "Run in order: setup → exp0 → exp1 → exp2"
        ;;
esac
