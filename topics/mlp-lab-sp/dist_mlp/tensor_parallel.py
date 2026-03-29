from typing import Optional, Tuple

import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def get_world_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


class _AllReduce(Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, group):
        ctx.group = group
        output_tensor = input_tensor.clone()
        dist.all_reduce(output_tensor, group=group)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None

class _SequenceAllGather(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, sp_group):
        ctx.sp_group = sp_group
        world_size = dist.get_world_size(sp_group)
        if world_size == 1:
            return x
        ctx.input_shape_sp_dim= x.shape[1]
        output_list = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(output_list, x, group=sp_group)
        output = torch.cat(output_list, dim=1).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        sp_group = ctx.sp_group
        world_size = dist.get_world_size(sp_group)
        if world_size == 1:
            return grad_output, None
        input_shape_sp_dim = ctx.input_shape_sp_dim
        grad_local_reduced = torch.empty(
            (input_shape_sp_dim, grad_output.shape[0], grad_output.shape[2]), 
            dtype=grad_output.dtype, 
            device=grad_output.device
        ) 
        grad_output=grad_output.transpose(0, 1).contiguous()
        input_list = list(torch.chunk(grad_output, world_size, dim=0))
        dist.reduce_scatter(
            grad_local_reduced,  # output
            input_list,          # input
            group=sp_group
        )
        grad_local_reduced = grad_local_reduced.transpose(0, 1).contiguous()
        return grad_local_reduced, None

class _GradAllReduce(Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, group=ctx.group)
        return grad, None
    

class _GradAllReduce(Function):
    """Forward identity; backward all-reduce grad_input across TP group."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_output, None

class _AllGatherCatFeat(Function):
    """Autograd-aware all_gather along feature dim (dim=1) then cat.
    Backward uses reduce_scatter(sum) along dim=1 to return local chunk grads.
    """
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, group):
        ctx.group = group
        world_size = dist.get_world_size(group)
        tensor_list = [torch.empty_like(input_tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, input_tensor, group=group)
        return torch.cat(tensor_list, dim=2)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        group = ctx.group
        world_size = dist.get_world_size(group)
        chunks = torch.chunk(grad_output, world_size, dim=2)  # 按特征维切分
        in_list = [c.contiguous() for c in chunks]
        out = torch.empty_like(in_list[0])
        dist.reduce_scatter(out, in_list, op=dist.ReduceOp.SUM, group=group)
        return out, None

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp_group, bias: bool = True,use_sp: bool=False):
        super().__init__()
        self.tp_group = tp_group
        self.world_size = dist.get_world_size(tp_group)
        assert out_features % self.world_size == 0
        self.local_out = out_features // self.world_size
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.weight = nn.Parameter(torch.full((self.local_out, in_features), 0.00001, device=device))
        self.bias = nn.Parameter(torch.zeros(self.local_out, device=device)) if bias else None
        self.reduce_input_grad = True 
        self.use_sp=use_sp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_sp:
            x = _SequenceAllGather.apply(x, self.tp_group)
        elif self.reduce_input_grad:
            x = _GradAllReduce.apply(x, self.tp_group)
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, self.weight.shape[1])
        out_local=torch.addmm(self.bias, x, self.weight.t())
        if len(original_shape) > 2:
            out_local = out_local.view(*original_shape[:-1], self.local_out)
        # out_local=torch.matmul(x, self.weight.t())
        # if self.bias is not None:
        #     out_local = out_local + self.bias
        # x=F.linear(x, self.weight, self.bias)
        return out_local

class _ReduceScatterSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out_partial: torch.Tensor, group):
        ctx.group   = group
        ctx.S_full  = out_partial.shape[1]
        ctx.world_size = dist.get_world_size(group)
        S_local = ctx.S_full // ctx.world_size
        out_part = torch.empty(S_local, out_partial.size(0), out_partial.size(2),
                               dtype=out_partial.dtype, device=out_partial.device)
        out = out_partial.transpose(0, 1).contiguous() 
        input_list = list(torch.chunk(out, ctx.world_size, dim=0))
        dist.reduce_scatter(out_part, input_list, group=group)
        out_part = out_part.transpose(0, 1).contiguous()
        return out_part

    @staticmethod
    def backward(ctx, grad_out_part: torch.Tensor):
        output_list = [
            torch.empty_like(grad_out_part) for _ in range(ctx.world_size)
        ]
        dist.all_gather(
            output_list, 
            grad_out_part.contiguous(), 
            group=ctx.group
        )
        grad_full = torch.cat(output_list, dim=1).contiguous()
        return grad_full, None

class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp_group, bias: bool = True,use_sp: bool=False):
        super().__init__()
        self.tp_group = tp_group
        self.world_size = dist.get_world_size(tp_group)
        assert in_features % self.world_size == 0
        self.local_in = in_features // self.world_size
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.weight = nn.Parameter(torch.full((out_features, self.local_in), 0.00001, device=device))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device)) if bias else None
        self.use_sp=use_sp
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_partial= torch.matmul(x, self.weight.t())
        if not self.use_sp:
            out_full = _AllReduce.apply(out_partial, self.tp_group)
            if self.bias is not None:
                out_full = out_full + self.bias
            return out_full
        else:
            out_full = _ReduceScatterSeq.apply(out_partial, self.tp_group)
            if self.bias is not None:
                out_full = out_full + self.bias
            return out_full
