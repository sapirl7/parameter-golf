# L1 + dim576 + Sliding Window Eval

This record captures a hardware-aligned width-scaled candidate built on a validated L1 compression branch.

## Summary

- **Layout**: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=576 NUM_HEADS=9 NUM_KV_HEADS=3 MLP_MULT=2`
- **head_dim**: `576 / 9 = 64` (power-of-2, FlashAttention-optimal)
- **Tied embeddings**: `TIE_EMBEDDINGS=1`
- **Compression**: post-step proximal L1 soft-thresholding (`LAMBDA_L1=3e-5 L1_START_FRAC=0.80`)
- **Eval overlay**: Sliding Window Eval, final-only (`SWE_STRIDE=64 SWE_ON_PERIODIC_VAL=0 SWE_ON_FINAL_VAL=1`)
- **TTT**: disabled (`ENABLE_TTT=0`)
- **Export**: INT8 + zlib

## Why this run

A hardware preflight on H100 showed `dim576/h9/kv3` materially faster than `dim544/h8/kv4`:

| Config | ms/step | head_dim |
|---|---|---|
| dim512 h8/kv4 | 99.97ms | 64 ✅ |
| dim544 h8/kv4 | 125.66ms | 68 ⚠️ |
| dim576 h9/kv3 | 108.51ms | 64 ✅ |

`dim544` is slower despite being smaller because `head_dim=68` is a non-power-of-2, which is suboptimal for FlashAttention kernel alignment on H100.

## Ablation order

1. Official baseline
2. `L1 + dim576` (no SWE)  — `run_2_l1_dim576.sh`
3. `L1 + dim576 + SWE` ← **this record** — `train.sh`

## Launch

```bash
# From this folder:
bash train.sh

# Override:
SEED=2026 GPUS=8 bash train.sh

# Disable SWE temporarily:
SWE_STRIDE=0 SWE_ON_FINAL_VAL=0 bash train.sh
```

## Key metrics (fill after run)

- Steps completed (at wallclock cap): `REPLACE`
- Pre-quant `val_bpb`: `REPLACE`
- `final_int8_zlib_roundtrip_exact val_bpb`: `REPLACE`
- `step_avg`: `REPLACE ms`
- Serialized size (int8+zlib): `REPLACE bytes`
- Total submission size: `REPLACE bytes`

## Included files

- `train_gpt.py` — standalone snapshot of `train_gpt_l1.py` at this commit
- `train.sh` — launch script with all hyperparameters set
- `submission.json` — leaderboard metadata
- `train.log` — auto-generated during run
