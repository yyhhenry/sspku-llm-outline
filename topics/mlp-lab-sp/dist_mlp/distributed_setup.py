import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist


@dataclass
class ParallelDims:
    data: int
    pipeline: int
    tensor: int


def _env_default_int(env_name: str, default: int) -> int:
    val = os.getenv(env_name)
    return int(val) if val is not None else default


def initialize_distributed(dp_size: int, pp_size: int, tp_size: int) -> Tuple[ParallelDims, int, int, int, int]:
    """
    Initialize torch.distributed default process group and create subgroups for
    Data Parallel (DP), Pipeline Parallel (PP), and Tensor Parallel (TP).

    Returns:
        (dims, global_rank, world_size, local_rank, device_index)
    """
    if not dist.is_initialized():
        backend = os.getenv("TORCH_DISTRIBUTED_BACKEND", "nccl")
        dist.init_process_group(backend=backend)

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    assert world_size % (dp_size * pp_size * tp_size) == 0, "World size must be multiple of dp*pp*tp"

    local_rank = _env_default_int("LOCAL_RANK", 0)
    device_index = local_rank
    torch.cuda.set_device(device_index)

    dims = ParallelDims(data=dp_size, pipeline=pp_size, tensor=tp_size)

    return dims, global_rank, world_size, local_rank, device_index


def build_parallel_groups(dims: ParallelDims):
    """
    Create and return dict of process groups according to mapping:
      rank = p*(DP*TP) + d*TP + t
    - dp_group: fix (p,t), vary d
    - pp_group: fix (d,t), vary p
    - tp_group: fix (d,p), vary t
    """
    world_size = dist.get_world_size()
    dp = dims.data
    pp = dims.pipeline
    tp = dims.tensor
    assert world_size % (dp * pp * tp) == 0

    num_replicas = world_size // (dp * pp * tp)
    assert num_replicas == 1, "This example assumes a single replica of (dp*pp*tp) mapping per node"

    dp_groups = []
    pp_groups = []
    tp_groups = []
    tmp_dp_groups = []
    tmp_pp_groups = []
    tmp_tp_groups = []

    # rank = p*(dp*tp) + d*tp + t
    for p in range(pp):
        for t in range(tp):
            ranks = [p * (dp * tp) + d * tp + t for d in range(dp)]
            dp_groups.append(dist.new_group(ranks))
            tmp_dp_groups.append(ranks)

    # PP groups: fix (d,t), vary p → size=PP, count=DP*TP
    for d in range(dp):
        for t in range(tp):
            ranks = [p * (dp * tp) + d * tp + t for p in range(pp)]
            pp_groups.append(dist.new_group(ranks))
            tmp_pp_groups.append(ranks)

    # TP groups: fix (d,p), vary t → size=TP, count=DP*PP
    for d in range(dp):
        for p in range(pp):
            ranks = [p * (dp * tp) + d * tp + t for t in range(tp)]
            tp_groups.append(dist.new_group(ranks))
            tmp_tp_groups.append(ranks)
    print(f"dp_groups={tmp_dp_groups} pp_groups={tmp_pp_groups} tp_groups={tmp_tp_groups}",flush=True)
    # Compute local (d,p,t) from rank
    rank = dist.get_rank()
    p_est = rank // (dp * tp)
    rem = rank % (dp * tp)
    d_est = rem // tp
    t_est = rem % tp

    # Pick containing groups by indices
    # dp_group with fixed (p_est, t_est)
    dp_group = None
    idx = 0
    for p in range(pp):
        for t in range(tp):
            if p == p_est and t == t_est:
                dp_group = dp_groups[idx]
                break
            idx += 1
        if dp_group is not None:
            break

    # pp_group with fixed (d_est, t_est)
    pp_group = None
    idx = 0
    for d in range(dp):
        for t in range(tp):
            if d == d_est and t == t_est:
                pp_group = pp_groups[idx]
                break
            idx += 1
        if pp_group is not None:
            break

    # tp_group with fixed (d_est, p_est)
    tp_group = None
    idx = 0
    for d in range(dp):
        for p in range(pp):
            if d == d_est and p == p_est:
                tp_group = tp_groups[idx]
                break
            idx += 1
        if tp_group is not None:
            break

    return {
        "dp_group": dp_group,
        "pp_group": pp_group,
        "tp_group": tp_group,
        "dp_index": d_est,
        "pp_index": p_est,
        "tp_index": t_est,
        "dp_size": dp,
        "pp_size": pp,
        "tp_size": tp,
    }

