import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.autograd import Function

from .tensor_parallel import ColumnParallelLinear, RowParallelLinear

class _Snapshot(Function):
    """
    一个自定义的 autograd.Function，用于在计算图中插入快照操作。
    前向传播时调用 snapshotter，反向传播时直接传递梯度。
    """
    @staticmethod
    def forward(ctx, x, snapshotter):
        ctx.snapshotter = snapshotter
        if snapshotter != None:
            snapshotter.dump_tmp()
        return x

    @staticmethod
    def backward(ctx, grad_output):
        snapshotter = ctx.snapshotter
        if snapshotter != None:
            snapshotter.dump_tmp()
        return grad_output, None

snapshot_function = _Snapshot.apply

@dataclass
class MLPConfig:
    hidden_size: int
    ffn_mult: int

def blocks_per_stage(pp_size: int) -> int:
    if 4 % pp_size != 0:
        raise ValueError("pp_size must be a divisor of 4 for a 4-layer MLP")
    return 4 // pp_size


class MLPBlock(nn.Module):
    def __init__(self, hidden_size: int, ffn_mult: int, tp_group,use_sp: bool=False):
        super().__init__()
        ffn = hidden_size * ffn_mult
        self.ln_in = nn.LayerNorm(hidden_size, eps=1e-5)
        self.fc1 = ColumnParallelLinear(hidden_size, ffn, tp_group, bias=True, use_sp=use_sp)
        self.act = nn.GELU()
        self.fc2 = RowParallelLinear(ffn, hidden_size, tp_group, bias=True, use_sp=use_sp)
        self.use_sp=use_sp

    def forward(self, x: torch.Tensor, tp_group,snapshotter=None):
        y = self.ln_in(x)
        # y = snapshot_function(y, snapshotter)
        y = self.fc1(y)
        # y = snapshot_function(y, snapshotter)
        y = self.act(y)
        # y = snapshot_function(y, snapshotter)
        out = self.fc2(y)
        # out = snapshot_function(out, snapshotter)
        return out


class MLPStage(nn.Module):
    def __init__(self, cfg: MLPConfig, tp_group, num_blocks: int, use_sp:bool=False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MLPBlock(cfg.hidden_size, cfg.ffn_mult, tp_group,use_sp) for _ in range(num_blocks)]
        )
        self.tp_group = tp_group
        self.use_sp=use_sp

    def forward(self, x: torch.Tensor, tp_group, snapshotter=None):
        if not self.use_sp:
            y=x
            for blk in self.blocks:
                y = blk(y, tp_group,snapshotter=snapshotter)
            return y
        y = x
        for blk in self.blocks:
            y = blk(y, tp_group,snapshotter=snapshotter)
        return y