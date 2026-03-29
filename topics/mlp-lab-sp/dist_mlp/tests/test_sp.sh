#!/usr/bin/env bash
set -euo pipefail

# 项目根目录
ROOT="/mnt/user-ssd/wangshengfan"
cd "$ROOT"
# 环境设置
export PYTHONPATH=${PYTHONPATH:-$PWD}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# 进程数（等于使用的 GPU 数），默认 2，可用 NPROC_PER_NODE 覆盖
NPROC=${NPROC_PER_NODE:-2}

# 如需指定 GPU，提前设置 CUDA_VISIBLE_DEVICES，例如：
# CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash test_tp.sh

echo "[test_tp] Launching with NPROC_PER_NODE=${NPROC}"
echo "[test_tp] PYTHONPATH=$PYTHONPATH"
echo "[test_tp] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"

# 运行测试
exec torchrun --nnodes=1 --nproc_per_node="${NPROC}" \
  dist_mlp/tests/test_sp.py