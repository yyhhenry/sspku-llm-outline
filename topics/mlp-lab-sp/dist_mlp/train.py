import argparse
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import inspect

from .distributed_setup import initialize_distributed, build_parallel_groups
from .mlp_model import MLPConfig, blocks_per_stage, MLPStage
from .pipeline import (
    next_stage_rank,
    prev_stage_rank,
    isend_activation,
    irecv_activation,
    isend_grad,
    irecv_grad,
    rank_from_indices,
)
from .utils import get_snapshotter
from .tensor_parallel import _SequenceAllGather

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dp-size", type=int, required=True)
    p.add_argument("--pp-size", type=int, required=True)
    p.add_argument("--tp-size", type=int, required=True)
    p.add_argument("--global-seed", type=int, default=42)
    p.add_argument("--hidden-size", type=int, default=8192)
    p.add_argument("--seq-length", type=int, default=16, help="序列长度 S，使输入为 [B, S, H]")
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--micro-batches", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--data-source", type=str, default="generate", choices=["generate", "load"])
    p.add_argument("--data-path", type=str, default="./data")
    p.add_argument("--save-loss", action="store_true", help="Save loss history to file")
    p.add_argument("--loss-log-interval", type=int, default=1, help="Interval for logging loss")
    p.add_argument("--wandb-able", type=bool, default=False, help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name")
    p.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity name")
    p.add_argument("--mem-snapshot", action="store_true", help="启用 CUDA 内存快照与显存统计")
    p.add_argument("--mem-snapshot-dir", type=str, default="mem_snapshots", help="快照与统计输出目录")
    p.add_argument("--mem-snapshot-every", type=int, default=1, help="每多少步保存一次快照与统计")
    p.add_argument("--sp",action="store_true",help="use sp")
    return p.parse_args()

def set_seed(seed: int):

    torch.cuda.manual_seed_all(seed)


def generate_data(args, device, batch_size, hidden_size, seed):
    """Generate fixed deterministic data based on step"""
    torch.manual_seed(seed)
    S = args.seq_length
    # 输入: [B, S, H]
    inputs = torch.randn(batch_size, S, hidden_size, device=device, dtype=torch.bfloat16).requires_grad_(True)
    # 目标: 对每个 token 做 y = sin(sum(x_{token})) 并在特征维展开到 H，得到 [B, S, H]
    target_scalar = torch.sin(inputs.sum(dim=-1))  # [B, S]
    targets_full = target_scalar.unsqueeze(-1).expand(-1, -1, hidden_size).contiguous().to(dtype=torch.bfloat16)
    return inputs, targets_full


def load_data(args, device, batch_size, hidden_size):
    """Load data from local files"""
    try:
        data = torch.load(f"{args.data_path}/train_data.pt", map_location=device)
        inputs = data["inputs"][:batch_size].to(device=device, dtype=torch.bfloat16)
        targets_full = data["targets"][:batch_size].to(device=device, dtype=torch.bfloat16)
        # 适配老数据：如果是 [B, H]，提升到 [B, S, H]
        if inputs.ndim == 2:
            S = args.seq_length
            inputs = inputs.unsqueeze(1).expand(-1, S, -1).contiguous()
        if targets_full.ndim == 2:
            S = args.seq_length
            targets_full = targets_full.unsqueeze(1).expand(-1, S, -1).contiguous()
        inputs = inputs.requires_grad_(True)
        return inputs, targets_full
    except FileNotFoundError:
        print(f"Warning: Data file not found at {args.data_path}/train_data.pt, falling back to generation")
        return generate_data(args, device, batch_size, hidden_size)


def allreduce_params(module: nn.Module, group):
    for p in module.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, group=group)
            p.grad.mul_(1.0 / dist.get_world_size(group))


def main():
    args = parse_args()
    dims, rank, world_size, local_rank, device_index = initialize_distributed(
        args.dp_size, args.pp_size, args.tp_size
    )
    if rank != rank_from_indices(args.dp_size-1,args.pp_size-1,args.tp_size-1,args.dp_size,args.pp_size,args.tp_size) :
        args.wandb_able=False
    if rank != 1 : 
        args.mem_snapshot = False
    if args.wandb_able==True:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"TP_{args.tp_size}DP_{args.dp_size}PP_{args.pp_size}_bf16_lr_{args.lr}_curv_nograd_avg",
            config={
                "dp_size": args.dp_size,
                "pp_size": args.pp_size,
                "tp_size": args.tp_size,
                "global_seed": args.global_seed,
                "hidden_size": args.hidden_size,
                "ffn_mult": args.ffn_mult,
                "batch_size": args.batch_size,
                "micro_batches": args.micro_batches,
                "lr": args.lr,
                "steps": args.steps,
                "data_source": args.data_source,
            },
        )
    pg = build_parallel_groups(dims)
    set_seed(args.global_seed+pg["tp_index"]*args.pp_size+pg["pp_index"])
    device = torch.device("cuda", device_index)
    print(
        f"rank: {rank}, world_size: {world_size}, local_rank: {local_rank}, device_index: {device_index}",
        flush=True,
    )

    # 获取快照器实例（如果禁用，则返回一个空操作对象）
    snapshotter = get_snapshotter(args, rank, device)

    assert args.batch_size%args.dp_size == 0
    batch_size= args.batch_size // args.dp_size
    cfg = MLPConfig(hidden_size=args.hidden_size, ffn_mult=args.ffn_mult)

    pp_index =pg["pp_index"]

    if pp_index != args.pp_size-1 :
        args.save_loss = False
    model = MLPStage(cfg, pg["tp_group"], blocks_per_stage(args.pp_size),args.sp).to(device)
    # Switch to BF16 for model to match BF16 inputs/targets
    model = model.to(dtype=torch.bfloat16)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    micro_bsz = batch_size // args.micro_batches
    assert batch_size % args.micro_batches == 0

    criterion = nn.MSELoss()

    nb_next = next_stage_rank(
        pg["dp_index"], pg["pp_index"], pg["tp_index"], pg["dp_size"], pg["pp_size"], pg["tp_size"]
    )
    nb_prev = prev_stage_rank(
        pg["dp_index"], pg["pp_index"], pg["tp_index"], pg["dp_size"], pg["pp_size"], pg["tp_size"]
    )

    if args.sp == True:
        data_shape = (batch_size, args.seq_length // args.tp_size, args.hidden_size)
        recv_shapes = (micro_bsz, args.seq_length // args.tp_size, args.hidden_size)
    else :
        data_shape = (batch_size, args.seq_length, args.hidden_size)
        recv_shapes = (micro_bsz, args.seq_length, args.hidden_size)
    loss_history = []
    
    # 新建init_model文件夹，里面有init_model.pt文件
    # import os
    # os.makedirs("init_model", exist_ok=True)
    # torch.save(model.state_dict(), f"init_model/init_model_sum_{rank}.pt")
    # torch.save(model.state_dict(), f"init_model/init_model_retaingrad_{rank}.pt")
    # args.steps=1
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        # Initialize variables for all stages
        activations_stage0: List[torch.Tensor] = []
        outputs: List[torch.Tensor] = []
        received_activations: List[torch.Tensor] = []
        targets_full = None
        # data generate and distribute
        if pp_index == 0:
            if rank == 0:
                # Rank 0 generates full batch data
                if args.data_source == "generate":
                    inputs_full, targets_full = generate_data(args, device, args.batch_size, args.hidden_size, args.global_seed+step)
                else:
                    inputs_full, targets_full = load_data(args, device, args.batch_size, args.hidden_size)
                for dp_idx in range(args.dp_size):
                    inputs_slice = inputs_full[dp_idx*batch_size:(dp_idx+1)*batch_size]
                    for tp_idx in range(args.tp_size):
                        target_rank = rank_from_indices(dp_idx, 0, tp_idx, args.dp_size, args.pp_size, args.tp_size)
                        if target_rank != rank:
                            # print(f"input rank{rank} send target_rank:{target_rank},tag:{args.steps*args.micro_batches+step*2}", flush=True)
                            if not args.sp:
                                _ = isend_activation(inputs_slice.detach(), target_rank, tag=args.steps*args.micro_batches+step*2)
                            else:
                                tp_rank=dist.get_rank(pg["tp_group"])
                                split_size = args.seq_length // args.tp_size
                                input_slice_part=inputs_slice[:, tp_rank*split_size:(tp_rank+1)*split_size, :]
                                _=isend_activation(input_slice_part.detach(), target_rank, tag=args.steps*args.micro_batches+step*2)
                # Send targets to all ranks
                for dp_idx in range(args.dp_size):
                    targets_slice = targets_full[dp_idx*batch_size:(dp_idx+1)*batch_size]
                    for tp_idx in range(args.tp_size):
                        target_rank = rank_from_indices(dp_idx, args.pp_size-1, tp_idx, args.dp_size, args.pp_size, args.tp_size)
                        if target_rank!=rank:
                            # print(f"target rank{rank} send target_rank:{target_rank},tag:{args.steps*args.micro_batches+step*2+1}", flush=True)
                            # if not args.sp:
                            _ = isend_activation(targets_slice.detach(), target_rank, tag=args.steps*args.micro_batches+step*2+1)
                
                # Keep local data slice
                inputs = inputs_full[0:batch_size]
                targets_full = targets_full[0:batch_size]

                if args.sp:
                    tp_rank=dist.get_rank(pg["tp_group"])
                    split_size = args.seq_length // args.tp_size
                    inputs=inputs[:, tp_rank*split_size:(tp_rank+1)*split_size, :]
                    
                inputs=inputs.requires_grad_(True)
            else:
                inputs, a_work = irecv_activation(data_shape, torch.bfloat16, device, 0, tag=args.steps*args.micro_batches+step*2)
                a_work.wait()
                inputs = inputs.requires_grad_(True)
        if pp_index == args.pp_size-1:
            if rank != 0:
                targets_full, a_work = irecv_activation((batch_size, args.seq_length,args.hidden_size), torch.bfloat16, device, 0, tag=args.steps*args.micro_batches+step*2+1)
                a_work.wait()
        # if pp_index==0 :
        #     our_dir=f"input_dp{args.dp_size}_pp{args.pp_size}_tp{args.tp_size}_mb{args.micro_batches}"
        #     if not os.path.exists(our_dir):
        #         os.makedirs(our_dir)
        #     torch.save({"inputs": inputs}, f"{our_dir}/{pg['dp_index']}_{step}_input.pt")
        # if pp_index==args.pp_size-1:
        #     our_dir=f"input_dp{args.dp_size}_pp{args.pp_size}_tp{args.tp_size}_mb{args.micro_batches}"
        #     if not os.path.exists(our_dir):
        #         os.makedirs(our_dir)
        #     torch.save({"target": targets_full}, f"{our_dir}/{pg['dp_index']}_{step}_target.pt")
        if args.pp_size == 1:
            # loss_mb_list=[]
            total_loss = 0.0
            for mb in range(args.micro_batches):
                mb_start = mb * micro_bsz
                mb_end = (mb + 1) * micro_bsz
                targets_full_stage0 = targets_full[mb_start:mb_end].detach()
                x_mb=inputs[mb_start:mb_end].detach()
                a_mb=model(x_mb, pg["tp_group"],snapshotter=snapshotter)
                if args.sp:
                    a_mb = _SequenceAllGather.apply(a_mb, pg["tp_group"])
                loss_mb = criterion(a_mb, targets_full_stage0) / args.micro_batches
                total_loss += loss_mb.detach().item()
                # loss_mb_list.append(loss_mb.detach().item())
                loss_mb.backward()
                # 保存grad
                # grads_to_save = [p.grad.clone() for p in model.parameters() if p.grad is not None]
                # import os
                # os.makedirs(f"dp_test_dp{args.dp_size}_tp{args.tp_size}_pp{args.pp_size}", exist_ok=True)
                # save_path = f"dp_test_dp{args.dp_size}_tp{args.tp_size}_pp{args.pp_size}/grad_mb{mb}_step{step}_rank{rank}.pt"
                # torch.save(grads_to_save, save_path)

                # # 保存输出a_mb
                # os.makedirs(f"sp_test_dp{args.dp_size}_tp{args.tp_size}_pp{args.pp_size}", exist_ok=True)
                # activation_save_path = f"sp_test_dp{args.dp_size}_tp{args.tp_size}_pp{args.pp_size}/a_mb{mb}_step{step}_rank{rank}_sp{args.sp}.pt"
                # a_to_save = a_mb.detach()
                # torch.save(a_to_save, activation_save_path)

            total_loss_tensor = torch.tensor(total_loss, device=device)
            dist.all_reduce(total_loss_tensor, group=pg["dp_group"])
            total_loss = total_loss_tensor.item() / dist.get_world_size(pg["dp_group"])
            loss_history.append(total_loss)
            if rank==rank_from_indices(args.dp_size-1,args.pp_size-1,args.tp_size-1,args.dp_size,args.pp_size,args.tp_size):
                if args.wandb_able:
                    run.log({"total loss":min(10.0,total_loss)})
                if step%100==0 :
                    print(f"step:{step} loss:{total_loss}",flush=True)
        else:
            # forward
            if pp_index == 0:
                for mb in range(args.micro_batches):
                    mb_start = mb * micro_bsz
                    mb_end = (mb + 1) * micro_bsz
                    x_mb = inputs[mb_start:mb_end]  # [mb, S, H]
                    x_mb = x_mb.detach()
                    a_mb = model(x_mb, pg["tp_group"],snapshotter=snapshotter)  # [mb, S, H]
                    activations_stage0.append(a_mb)
                    assert nb_next is not None
                    _ = isend_activation(a_mb.detach(), nb_next, tag=step*args.micro_batches+mb)
                    # print(f"act rank{rank} send to rank{nb_next},tag:{step*args.micro_batches+mb}", flush=True)
            elif pp_index < args.pp_size - 1:
                for mb in range(args.micro_batches):
                    mb_start = mb * micro_bsz
                    mb_end = (mb + 1) * micro_bsz
                    a_buf, a_work = irecv_activation(recv_shapes, torch.bfloat16, device, nb_prev, tag=step*args.micro_batches+mb)
                    a_work.wait()
                    # print(f"act rank{rank} received rank{nb_prev},tag:{step*args.micro_batches+mb}", flush=True)
                    a_buf.requires_grad_(True)
                    received_activations.append(a_buf)
                    y_mb = model(a_buf, pg["tp_group"],snapshotter=snapshotter)
                    outputs.append(y_mb)
                    assert nb_next is not None
                    _ = isend_activation(y_mb.detach(), nb_next, tag=step*args.micro_batches+mb)
                    # print(f"act rank{rank} send to rank{nb_next},tag:{step*args.micro_batches+mb}", flush=True)
            elif pp_index == args.pp_size - 1:
                total_loss = 0.0
                for mb in range(args.micro_batches):
                    mb_start = mb * micro_bsz
                    mb_end = (mb + 1) * micro_bsz
                    a_buf, a_work = irecv_activation(recv_shapes, torch.bfloat16, device, nb_prev, tag=step*args.micro_batches+mb)
                    a_work.wait()
                    # print(f"act rank{rank} received rank{nb_prev},tag:{step*args.micro_batches+mb}", flush=True)
                    a_buf.requires_grad_(True)
                    y_mb = model(a_buf, pg["tp_group"],snapshotter=snapshotter)
                    if args.sp:
                        y_mb = _SequenceAllGather.apply(y_mb, pg["tp_group"])
                    targets_full_stage0 = targets_full[mb_start:mb_end]
                    loss_mb = criterion(y_mb, targets_full_stage0) / args.micro_batches
                    total_loss += loss_mb.item()
                    loss_mb.backward()
                    grad_a = a_buf.grad.detach()
                    _ = isend_grad(grad_a, nb_prev, tag=step*args.micro_batches+mb)
                    # print(f"grad rank{rank} send to rank{nb_prev},tag:{step*args.micro_batches+mb}", flush=True)
                # 对同一个dp组的total_loss进行all_reduce并记录
                total_loss_tensor = torch.tensor(total_loss, device=device)
                dist.all_reduce(total_loss_tensor, group=pg["dp_group"])
                total_loss = total_loss_tensor.item() / dist.get_world_size(pg["dp_group"])
                if args.wandb_able:
                    run.log({"total loss":min(10.0,total_loss)})
                    if step%100==0 :
                        print(f"step:{step} loss:{total_loss}",flush=True)
                # 计算平均loss并记录
                if args.save_loss:
                    loss_history.append(total_loss)
            
            # backward
            if pp_index == 0:
                if args.pp_size > 1:
                    for mb in range(args.micro_batches):
                        g_buf, g_work = irecv_grad(recv_shapes, torch.bfloat16, device, nb_next, tag=step*args.micro_batches+mb)
                        g_work.wait()
                        # print(f"grad rank{rank} received rank{nb_next},tag:{step*args.micro_batches+mb}", flush=True)
                        activations_stage0[mb].backward(g_buf)
            elif pp_index < args.pp_size - 1:
                for mb in range(args.micro_batches):
                    g_buf, g_work = irecv_grad(recv_shapes, torch.bfloat16, device, nb_next, tag=step*args.micro_batches+mb)
                    g_work.wait()
                    # print(f"grad rank{rank} received rank{nb_next},tag:{step*args.micro_batches+mb}", flush=True)
                    outputs[mb].backward(g_buf)
                    grad_a = received_activations[mb].grad.detach()
                    _ = isend_grad(grad_a, nb_prev, tag=step*args.micro_batches+mb)
                    # print(f"grad rank{rank} send to rank{nb_prev},tag:{step*args.micro_batches+mb}", flush=True)

        # DP gradient all-reduce
        allreduce_params(model, pg["dp_group"])
        optimizer.step()

        # print(f"step{step}rank{rank} over!",flush=True)
        dist.barrier()

        if step == args.steps-1:
            snapshotter.dump(step)


        # import os
        # os.makedirs(f"results{args.dp_size}{args.pp_size}{args.tp_size}_{args.micro_batches}sum", exist_ok=True)
        # if rank == 0:
        #     print(f"step {step} done", flush=True)
        # torch.save(model.state_dict(), f"results{args.dp_size}{args.pp_size}{args.tp_size}_{args.micro_batches}sum/step_{step}_model_{rank}.pt")
    
    # # 输出loss统计总结
    if args.save_loss:
        os.makedirs(f"sp_test_dp{args.dp_size}_tp{args.tp_size}_pp{args.pp_size}", exist_ok=True)
        loss_file = f"sp_test_dp{args.dp_size}_tp{args.tp_size}_pp{args.pp_size}/loss_rank{rank}_sp{args.sp}.txt"
        with open(loss_file, 'w') as f:
            for step, loss in enumerate(loss_history):
                f.write(f"{step}\t{loss}\n")
    if args.wandb_able==True:
        run.finish()
    # if rank==rank_from_indices(args.dp_size-1,args.pp_size-1,args.tp_size-1,args.dp_size,args.pp
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()