import torch
import torch.distributed as dist


def rank_from_indices(dp_index: int, pp_index: int, tp_index: int, dp_size: int, pp_size: int, tp_size: int) -> int:
    # rank = p*(dp*tp) + d*tp + t
    return pp_index * (dp_size * tp_size) + dp_index * tp_size + tp_index


def next_stage_rank(dp_index: int, pp_index: int, tp_index: int, dp_size: int, pp_size: int, tp_size: int):
    nxt = pp_index + 1
    if nxt < pp_size:
        return rank_from_indices(dp_index, nxt, tp_index, dp_size, pp_size, tp_size)
    return None


def prev_stage_rank(dp_index: int, pp_index: int, tp_index: int, dp_size: int, pp_size: int, tp_size: int):
    prv = pp_index - 1
    if prv >= 0:
        return rank_from_indices(dp_index, prv, tp_index, dp_size, pp_size, tp_size)
    return None


def send_activation(tensor: torch.Tensor, dst_rank: int, tag: int = 0):
    dist.send(tensor=tensor.contiguous(), dst=dst_rank, tag=tag)


def recv_activation(shape, dtype, device, src_rank: int, tag: int = 0):
    tensor = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(tensor=tensor, src=src_rank, tag=tag)
    return tensor


def send_grad(tensor: torch.Tensor, dst_rank: int, tag: int = 1):
    dist.send(tensor=tensor.contiguous(), dst=dst_rank, tag=tag)


def recv_grad(shape, dtype, device, src_rank: int, tag: int = 1):
    tensor = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(tensor=tensor, src=src_rank, tag=tag)
    return tensor

# Non-blocking variants

def isend_activation(tensor: torch.Tensor, dst_rank: int, tag: int = 0):
    return dist.isend(tensor=tensor.contiguous(), dst=dst_rank, tag=tag)


def irecv_activation(shape, dtype, device, src_rank: int, tag: int = 0):
    tensor = torch.empty(shape, dtype=dtype, device=device)
    work = dist.irecv(tensor=tensor, src=src_rank, tag=tag)
    return tensor, work


def isend_grad(tensor: torch.Tensor, dst_rank: int, tag: int = 1):
    return dist.isend(tensor=tensor.contiguous(), dst=dst_rank, tag=tag)


def irecv_grad(shape, dtype, device, src_rank: int, tag: int = 1):
    tensor = torch.empty(shape, dtype=dtype, device=device)
    work = dist.irecv(tensor=tensor, src=src_rank, tag=tag)
    return tensor, work

