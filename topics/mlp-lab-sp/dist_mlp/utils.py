import os
import torch
import torch.distributed as dist
import json
import inspect

class _NullSnapshotter:
    """一个什么都不做的伪快照器，用于在禁用快照时避免代码中的 if 判断。"""
    def dump(self, step_idx: int):
        pass

    def dump_tmp(self):
        pass

class MemorySnapshotter:
    """
    封装 PyTorch 内存快照的逻辑。
    """
    def __init__(self, args, rank, device):
        self.enabled = args.mem_snapshot
        if not self.enabled:
            return

        self.dir = args.mem_snapshot_dir
        self.rank = rank
        self.device = device
        self.tmp_counter = 0

        os.makedirs(self.dir, exist_ok=True)

        torch.cuda.memory._record_memory_history(
            enabled='all', 
            context='all', 
            stacks='all',
            max_entries=1_000_000 
        )
        torch.cuda.reset_peak_memory_stats(self.device)

        self.prefix = f"dp{args.dp_size}_pp{args.pp_size}_tp{args.tp_size}_rank{rank}"
        self.csv_path = os.path.join(self.dir, f"mem_stats_{self.prefix}.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w") as f:
                f.write("step,allocated_B,reserved_B,peak_allocated_B,peak_reserved_B\n")

    def dump(self, step_idx: int):
        """在给定的步骤导出所有内存快照和统计信息。"""
        if not self.enabled:
            return

        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        peak_alloc = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        with open(self.csv_path, "a") as f:
            f.write(f"{step_idx},{allocated},{reserved},{peak_alloc},{peak_reserved}\n")

        try:
            snap_path = os.path.join(self.dir, f"step{step_idx}_{self.prefix}.pickle")
            torch.cuda.memory._dump_snapshot(snap_path)
        except Exception as e:
            print(f"[Rank {self.rank}] Failed to dump snapshot pickle: {e}", flush=True)

    def dump_tmp(self):
        """导出一个临时的、带唯一编号的快照。"""
        if not self.enabled:
            return

        torch.cuda.synchronize(self.device)
        self.tmp_counter += 1
        snap_path = os.path.join(self.dir, f"t{self.tmp_counter}.pickle")
        try:
            torch.cuda.memory._dump_snapshot(snap_path)
        except Exception as e:
            print(f"[Rank {self.rank}] Failed to dump temporary snapshot: {e}", flush=True)

def get_snapshotter(args, rank, device):
    """根据参数返回一个真实的快照器或一个空对象。"""
    if args.mem_snapshot :
        return MemorySnapshotter(args, rank, device)
    return _NullSnapshotter()