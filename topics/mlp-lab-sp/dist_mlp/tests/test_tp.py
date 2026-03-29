import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import math
from dist_mlp.tensor_parallel import ColumnParallelLinear, RowParallelLinear,_AllGatherCatFeat

def _unbiased_equal(a: torch.Tensor, b: torch.Tensor, z: float = 3.0):
    """基于误差无偏性：检验 mean(diff) 是否在 z*SE 范围内"""
    diff = (a - b).detach().float()
    N = diff.numel()
    mean = diff.mean()
    if N > 1:
        std = diff.std(unbiased=True)
        se = std / math.sqrt(N)
    else:
        std = torch.tensor(0.0, device=diff.device)
        se = torch.tensor(0.0, device=diff.device)
    ok = (mean.abs() <= z * se + 1e-12)  # N=1 时容错
    stats = {
        "mean": mean.item(),
        "std": std.item() if N > 1 else 0.0,
        "se": se.item() if N > 1 else 0.0,
        "max_abs": diff.abs().max().item(),
    }
    return ok.item(), stats
def _unbiased_over_trials(values, z: float = 3.0):
    """对跨迭代的均值列表做无偏性检验：|mean| <= z * SE"""
    t = torch.as_tensor(values, dtype=torch.float64)
    n = t.numel()
    if n == 0:
        return False, {"mean": 0.0, "std": 0.0, "se": 0.0}
    mean = t.mean()
    if n > 1:
        std = t.std(unbiased=True)
        se = std / math.sqrt(n)
    else:
        std = torch.tensor(0.0, dtype=torch.float64)
        se = torch.tensor(0.0, dtype=torch.float64)
    ok = (mean.abs() <= z * se + 1e-12)
    return bool(ok.item()), {"mean": float(mean), "std": float(std), "se": float(se)}

def run_test():
    # init from torchrun env
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    tp_group = dist.group.WORLD

    TRIALS = int(os.environ.get("TP_TEST_TRIALS", "100"))
    batch = int(os.environ.get("TP_TEST_BATCH", "4"))
    in_features = int(os.environ.get("TP_TEST_IN", "64"))
    out_features = int(os.environ.get("TP_TEST_OUT", "128"))
    assert out_features % world_size == 0 and in_features % world_size == 0, "Choose sizes divisible by TP"
    local_out = out_features // world_size
    local_in = in_features // world_size
    
    agg = {
        "col_fwd_mean_abs": 0.0, "col_fwd_max_abs": 0.0, "col_fwd_unbiased_ok": 0,
        "col_grad_mean_abs": 0.0, "col_grad_max_abs": 0.0, "col_grad_unbiased_ok": 0,
        "row_fwd_mean_abs": 0.0, "row_fwd_max_abs": 0.0, "row_fwd_unbiased_ok": 0,
        "row_grad_mean_abs": 0.0, "row_grad_max_abs": 0.0, "row_grad_unbiased_ok": 0,
    }
    records = []
    seed = 14324
    col_fwd_means=[]
    col_grad_means=[]
    row_fwd_means=[]
    row_grad_means=[]
    for it in range(TRIALS):
        seed = seed + it
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        full = nn.Linear(in_features, out_features, bias=True).to(device=device, dtype=torch.bfloat16)
        W_full = full.weight.detach().clone()
        b_full = full.bias.detach().clone()
        x = torch.randn(batch, in_features, device=device, dtype=torch.bfloat16, requires_grad=True)

        # ---------- ColumnParallel 前向 ----------
        col = ColumnParallelLinear(in_features, out_features, tp_group, bias=True).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            W_chunks = W_full.chunk(world_size, dim=0)
            b_chunks = b_full.chunk(world_size, dim=0)
            col.weight.copy_(W_chunks[rank].to(device=device, dtype=torch.bfloat16))
            if col.bias is not None:
                col.bias.copy_(b_chunks[rank].to(device=device, dtype=torch.bfloat16))
        out_local = col(x)  # [B, local_out]
        out_recon = _AllGatherCatFeat.apply(out_local, tp_group)  # [B, out_features]
        out_ref = full(x)

        ok, s = _unbiased_equal(out_recon, out_ref, z=3.0)

        agg["col_fwd_unbiased_ok"] += int(ok)
        agg["col_fwd_mean_abs"] += abs(s["mean"])
        agg["col_fwd_max_abs"] += s["max_abs"]

        # ---------- ColumnParallel 反向（权重梯度） ----------
        loss_ref = out_ref.pow(2).sum()
        loss_recon = out_recon.pow(2).sum()
        full.zero_grad(set_to_none=True)
        loss_ref.backward()
        grad_W_full = full.weight.grad.detach().clone()
        col.zero_grad(set_to_none=True)
        loss_recon.backward()
        grad_W_local = col.weight.grad.detach().clone()  # [local_out, in_features]
        # 重建完整梯度（按行拼接）
        gathered_grads = [torch.zeros_like(grad_W_local) for _ in range(world_size)]
        dist.all_gather(gathered_grads, grad_W_local, group=tp_group)
        grad_W_recon = torch.cat(gathered_grads, dim=0)

        okg, sg = _unbiased_equal(grad_W_recon, grad_W_full, z=3.0)
        agg["col_grad_unbiased_ok"] += int(okg)
        agg["col_grad_mean_abs"] += abs(sg["mean"])
        agg["col_grad_max_abs"] += sg["max_abs"]

        dist.barrier()

        # ---------- RowParallel 前向 ----------
        row = RowParallelLinear(in_features, out_features, tp_group, bias=True).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            row.weight.copy_(W_full[:, rank*local_in:(rank+1)*local_in].to(device=device, dtype=torch.bfloat16))
            if row.bias is not None:
                row.bias.copy_(b_full.to(device=device, dtype=torch.bfloat16))
        x_local = x.chunk(world_size, dim=1)[rank].contiguous()
        out_partial = row(x_local)  # 期望与 out_ref 一致（Row 内部做 all-reduce + bias 逻辑）

        ok2, s2 = _unbiased_equal(out_partial, out_ref, z=3.0)
        agg["row_fwd_unbiased_ok"] += int(ok2)
        agg["row_fwd_mean_abs"] += abs(s2["mean"])
        agg["row_fwd_max_abs"] += s2["max_abs"]

        # ---------- RowParallel 反向（权重梯度） ----------
        row.zero_grad(set_to_none=True)
        loss_row = out_partial.pow(2).sum()
        loss_row.backward()
        grad_W_local_row = row.weight.grad.detach().clone()  # [out_features, local_in]
        gathered_grads_row = [torch.zeros_like(grad_W_local_row) for _ in range(world_size)]
        dist.all_gather(gathered_grads_row, grad_W_local_row, group=tp_group)
        grad_W_recon_row = torch.cat(gathered_grads_row, dim=1)  # 列拼接 -> [out_features, in_features]

        okg2, sg2 = _unbiased_equal(grad_W_recon_row, grad_W_full, z=3.0)
        agg["row_grad_unbiased_ok"] += int(okg2)
        agg["row_grad_mean_abs"] += abs(sg2["mean"])
        agg["row_grad_max_abs"] += sg2["max_abs"]

        dist.barrier()
        
        if rank == 0:
            if (it+1) % 10 == 0: 
                print(f"Trial {it+1}/{TRIALS} done.")
            records.append({
            "iter": it,
            "col_fwd_unbiased_ok": int(ok),
            "col_fwd_mean": float(s["mean"]),
            "col_fwd_max_abs": float(s["max_abs"]),
            "col_grad_unbiased_ok": int(okg),
            "col_grad_mean": float(sg["mean"]),
            "col_grad_max_abs": float(sg["max_abs"]),
            "row_fwd_unbiased_ok": int(ok2),
            "row_fwd_mean": float(s2["mean"]),
            "row_fwd_max_abs": float(s2["max_abs"]),
            "row_grad_unbiased_ok": int(okg2),
            "row_grad_mean": float(sg2["mean"]),
            "row_grad_max_abs": float(sg2["max_abs"])
            })        
            col_fwd_means.append(float(s["mean"]))
            col_grad_means.append(float(sg["mean"]))
            row_fwd_means.append(float(s2["mean"]))
            row_grad_means.append(float(sg2["mean"]))
    # 汇总并打印（仅 rank0）
    if rank == 0 :
        def avg(key): return agg[key] / TRIALS
        def rate(key): return agg[key] / TRIALS

        print(f"=== TP test summary over {TRIALS} random seeds ===")
        print(f"- ColumnParallel Forward:  unbiased_pass={rate('col_fwd_unbiased_ok'):.3f}, avg|mean|={avg('col_fwd_mean_abs'):.3e}, avg max|diff|={avg('col_fwd_max_abs'):.3e}")
        print(f"- ColumnParallel Grad(W): unbiased_pass={rate('col_grad_unbiased_ok'):.3f}, avg|mean|={avg('col_grad_mean_abs'):.3e}, avg max|diff|={avg('col_grad_max_abs'):.3e}")
        print(f"- RowParallel    Forward:  unbiased_pass={rate('row_fwd_unbiased_ok'):.3f}, avg|mean|={avg('row_fwd_mean_abs'):.3e}, avg max|diff|={avg('row_fwd_max_abs'):.3e}")
        print(f"- RowParallel    Grad(W): unbiased_pass={rate('row_grad_unbiased_ok'):.3f}, avg|mean|={avg('row_grad_mean_abs'):.3e}, avg max|diff|={avg('row_grad_max_abs'):.3e}")
        z = 3.0
        ok_cf, st_cf = _unbiased_over_trials(col_fwd_means, z=z)
        ok_cg, st_cg = _unbiased_over_trials(col_grad_means, z=z)
        ok_rf, st_rf = _unbiased_over_trials(row_fwd_means, z=z)
        ok_rg, st_rg = _unbiased_over_trials(row_grad_means, z=z)

        print("=== Unbiasedness over trials (|mean| <= z*SE) ===")
        print(f"col_fwd_mean: ok={ok_cf} mean={st_cf['mean']:.3e} se={st_cf['se']:.3e} z={z}")
        print(f"col_grad_mean: ok={ok_cg} mean={st_cg['mean']:.3e} se={st_cg['se']:.3e} z={z}")
        print(f"row_fwd_mean: ok={ok_rf} mean={st_rf['mean']:.3e} se={st_rf['se']:.3e} z={z}")
        print(f"row_grad_mean: ok={ok_rg} mean={st_rg['mean']:.3e} se={st_rg['se']:.3e} z={z}")
        from pathlib import Path
        out_dir = Path("dist_mlp/test_result")
        out_dir.mkdir(parents=True, exist_ok=True)
        xlsx_path = out_dir / "result_test_tp.xlsx"
        headers = [
            "iter",
            "col_fwd_unbiased_ok","col_fwd_mean","col_fwd_max_abs",
            "col_grad_unbiased_ok","col_grad_mean","col_grad_max_abs",
            "row_fwd_unbiased_ok","row_fwd_mean","row_fwd_max_abs",
            "row_grad_unbiased_ok","row_grad_mean","row_grad_max_abs",
        ]
        summary = [
            ["col_fwd_unbiased_rate", float(rate("col_fwd_unbiased_ok"))],
            ["col_fwd_avg_abs_mean", float(avg("col_fwd_mean_abs"))],
            ["col_fwd_avg_max_abs", float(avg("col_fwd_max_abs"))],
            ["col_grad_unbiased_rate", float(rate("col_grad_unbiased_ok"))],
            ["col_grad_avg_abs_mean", float(avg("col_grad_mean_abs"))],
            ["col_grad_avg_max_abs", float(avg("col_grad_max_abs"))],
            ["row_fwd_unbiased_rate", float(rate("row_fwd_unbiased_ok"))],
            ["row_fwd_avg_abs_mean", float(avg("row_fwd_mean_abs"))],
            ["row_fwd_avg_max_abs", float(avg("row_fwd_max_abs"))],
            ["row_grad_unbiased_rate", float(rate("row_grad_unbiased_ok"))],
            ["row_grad_avg_abs_mean", float(avg("row_grad_mean_abs"))],
            ["row_grad_avg_max_abs", float(avg("row_grad_max_abs"))],
        ]
        try:
            from openpyxl import Workbook
            wb = Workbook()
            ws1 = wb.active
            ws1.title = "details"
            ws1.append(headers)
            for r in records:
                ws1.append([r[h] for h in headers])
            ws2 = wb.create_sheet("summary")
            ws2.append(["metric","value"])
            for m,v in summary:
                ws2.append([m, v])
            wb.save(str(xlsx_path))
            print(f"[TP Test] 结果已保存: {xlsx_path}", flush=True)
        except ImportError:
            import csv
            csv_path = out_dir / "result_test_tp.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(headers)
                for r in records:
                    w.writerow([r[h] for h in headers])
            print(f"[TP Test] openpyxl 未安装，结果已保存为 CSV: {csv_path}", flush=True)
        # 绘制误差随迭代的变化曲线
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,6))
            plt.plot(range(TRIALS), col_fwd_means, label="Col Fwd Mean")
            plt.plot(range(TRIALS), col_grad_means, label="Col Grad Mean")
            plt.plot(range(TRIALS), row_fwd_means, label="Row Fwd Mean")
            plt.plot(range(TRIALS), row_grad_means, label="Row Grad Mean")
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.xlabel("Trial")
            plt.ylabel("Mean Error")
            plt.title("TP Module Mean Error over Trials")
            plt.legend()
            plt.grid(True)
            fig_path = out_dir / "tp_test_mean_error.png"
            plt.savefig(fig_path)
            print(f"[TP Test] 误差曲线已保存: {fig_path}", flush=True)
        except ImportError:
            print(f"[TP Test] matplotlib 未安装，跳过绘图。", flush=True)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    run_test()