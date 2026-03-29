#!/usr/bin/env bash
set -euo pipefail

cd /mnt/user-ssd/wangshengfan
source .venv/bin/activate

export PYTHONPATH=${PYTHONPATH:-$PWD}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}

: "${DP_SIZE:=2}"
: "${PP_SIZE:=2}"
: "${TP_SIZE:=2}"
: "${GLOBAL_SEED:=42}"
: "${HIDDEN_SIZE:=2048}"
: "${FFN_MULT:=2}"
: "${BATCH_SIZE:=128}"
: "${MICRO_BATCHES:=4}"
: "${LR:=1e-3}"
: "${STEPS:=10000}"
: "${NPROC_PER_NODE:=8}"

# Validate PP divides 4 (we have 4 MLP blocks)
if (( 4 % PP_SIZE != 0 )); then
  echo "Error: PP_SIZE must be a divisor of 4 (got PP_SIZE=${PP_SIZE})." >&2
  exit 1
fi

# Warn if process count mismatch
let PRODUCT=DP_SIZE*PP_SIZE*TP_SIZE
if (( PRODUCT != NPROC_PER_NODE )); then
  echo "Warning: DP*PP*TP=${PRODUCT} != NPROC_PER_NODE=${NPROC_PER_NODE}. Adjust NPROC_PER_NODE or parallel sizes." >&2
fi

exec python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} \
  -m dist_mlp.train \
  --dp-size ${DP_SIZE} --pp-size ${PP_SIZE} --tp-size ${TP_SIZE} \
  --global-seed ${GLOBAL_SEED} \
  --hidden-size ${HIDDEN_SIZE} --ffn-mult ${FFN_MULT} \
  --batch-size ${BATCH_SIZE} --micro-batches ${MICRO_BATCHES} \
  --lr ${LR} --steps ${STEPS} --mem-snapshot \
  --save-loss #--sp
#  --sp \
  #\
#  --profile-memory
 # --wandb-able true --wandb-project "MLP Lab" --wandb-entity "root" #\  
 #--save-loss 