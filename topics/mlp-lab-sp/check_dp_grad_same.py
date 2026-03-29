import argparse
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch

# dp1 使用 rank0 文件
PAT_DP1 = re.compile(r"grad_mb(\d+)_step(\d+)_rank0\.pt$")
# 通用解析（用于 dp4）
PAT_ALL = re.compile(r"grad_mb(\d+)_step(\d+)_rank(\d+)\.pt$")

def load_obj(path: Path) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...], dict]:
    return torch.load(path, map_location="cpu")

def to_tensor_payload(obj) -> torch.Tensor:
    # 单 Tensor 直接返回；list/tuple（参数梯度列表）展平拼成 1D；dict 取第一个 Tensor 值
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        flat = []
        for x in obj:
            if isinstance(x, torch.Tensor):
                flat.append(x.detach().float().reshape(-1))
        if flat:
            return torch.cat(flat, dim=0)
        raise ValueError("List/tuple without tensors.")


def find_dp1_pairs(dp1_dir: Path) -> List[Tuple[int, int, Path]]:
    files = []
    for p in dp1_dir.glob("grad_mb*_step*_rank0.pt"):
        m = PAT_DP1.search(p.name)
        if m:
            mb = int(m.group(1)); step = int(m.group(2))
            files.append((step, mb, p))
    files.sort()  # (step, mb) 排序
    return files

def _parse_mb_step_rank(name: str) -> Tuple[int, int, int]:
    m = PAT_ALL.search(name)
    if not m:
        raise ValueError(f"Unrecognized filename: {name}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

def compare_tensors(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> Dict[str, float]:
    if a.shape != b.shape:
        return {"equal": 0.0, "allclose": 0.0, "max_abs": float("inf"), "mean_abs": float("inf"), "shape_equal": 0.0}
    if a.dtype != b.dtype:
        a = a.float(); b = b.float()
    diff = (a - b).abs()
    return {
        "equal": float(torch.equal(a, b)),
        "allclose": float(torch.allclose(a, b, rtol=rtol, atol=atol)),
        "max_abs": float(diff.max().item()) if diff.numel() > 0 else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() > 0 else 0.0,
        "shape_equal": 1.0,
    }

def _unbiased_equal(a: torch.Tensor, b: torch.Tensor, z: float = 3.0):
    if a.shape != b.shape:
        return False, {"mean": float("nan"), "std": float("nan"), "se": float("nan"), "max_abs": float("inf"), "n": 0}
    diff = (a - b).detach().float()
    n = diff.numel()
    if n == 0:
        return True, {"mean": 0.0, "std": 0.0, "se": 0.0, "max_abs": 0.0, "n": 0}
    mean = diff.mean()
    if n > 1:
        std = diff.std(unbiased=True); se = std / math.sqrt(n)
    else:
        std = torch.tensor(0.0, dtype=torch.float32); se = torch.tensor(0.0, dtype=torch.float32)
    ok = (mean.abs() <= z * se + 1e-12)
    return bool(ok.item()), {"mean": float(mean.item()), "std": float(std.item()), "se": float(se.item()),
                             "max_abs": float(diff.abs().max().item()), "n": int(n)}

def _unbiased_over_trials(values: List[float], z: float = 3.0):
    if len(values) == 0:
        return True, {"mean": 0.0, "std": 0.0, "se": 0.0, "n": 0}
    t = torch.as_tensor(values, dtype=torch.float64)
    n = t.numel(); mean = t.mean()
    if n > 1:
        std = t.std(unbiased=True); se = std / math.sqrt(n)
    else:
        std = torch.tensor(0.0, dtype=torch.float64); se = torch.tensor(0.0, dtype=torch.float64)
    ok = (mean.abs() <= z * se + 1e-12)
    return bool(ok.item()), {"mean": float(mean.item()), "std": float(std.item()), "se": float(se.item()), "n": int(n)}

def main():
    ap = argparse.ArgumentParser(description="比较 DP=1 与 DP=4 梯度：DP1.mbX vs DP4.rankX.(mb0+mb1+...)")
    ap.add_argument("--dp1-dir", type=str, default="dp_test_dp1_tp1_pp1", help="dp=1 梯度目录（rank0）")
    ap.add_argument("--dp4-dir", type=str, default="dp_test_dp4_tp1_pp1", help="dp=4 梯度目录")
    ap.add_argument("--dp4_world_size", type=int, default=4, help="dp=4 的 rank 数")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dp4_reduce", type=str, default="mean",
                    choices=["concat", "sum", "mean"],
                    help="dp4 多 rank 合并方式：concat(按行拼接，用于激活梯度)，sum/mean(用于参数梯度列表)")
    args = ap.parse_args()

    dp1_dir = Path(args.dp1_dir); dp4_dir = Path(args.dp4_dir)
    assert dp1_dir.is_dir(), f"dp1-dir not found: {dp1_dir}"
    assert dp4_dir.is_dir(), f"dp4-dir not found: {dp4_dir}"

    pairs = find_dp1_pairs(dp1_dir)
    if not pairs:
        print(f"[WARN] 未在 {dp1_dir} 找到 grad_mb*_step*_rank0.pt")
        raise SystemExit(1)

    total = ok_cnt = unbias_ok_cnt = 0
    means_over_pairs: List[float] = []

    for step, mb, p1 in pairs:
        raw1 = load_obj(p1)
        t1 = to_tensor_payload(raw1)

        mb_fixed = args.dp4_world_size - 1
        required_ranks = list(range(mb + 1))
        t4_accum = None
        base_shape = None
        missing = []
        for r in required_ranks:
            p4 = dp4_dir / f"grad_mb{mb_fixed}_step{step}_rank{r}.pt"
            if not p4.exists():
                missing.append(str(p4))
                continue
            obj = load_obj(p4)
            t = to_tensor_payload(obj)
            flat = t.detach().float().reshape(-1)
            if t4_accum is None:
                t4_accum = flat.clone()
                base_shape = t.shape
            else:
                if flat.numel() != t4_accum.numel():
                    print(f"[ERR ] step={step} mb={mb} 形状不匹配: {p4.name} {tuple(t.shape)} vs base {base_shape}")
                    t4_accum = None
                    break
                t4_accum.add_(flat)
        if missing:
            print(f"[MISS] step={step} mb={mb} 缺少 dp4 mb{mb_fixed} 文件: {missing}")
            total += 1
            continue
        if t4_accum is None:
            total += 1
            continue
        denom = float(len(required_ranks)) if len(required_ranks) > 0 else 1.0
        t4 = t4_accum.view(base_shape)
        t1.mul_(4) # 乘上mb数还原原来的结果
        # 后续保持不变：t1 vs t4 的误差与无偏性比较、打印与统计
        stats = compare_tensors(t1, t4, args.rtol, args.atol)
        equal = bool(stats["equal"] or stats["allclose"])
        unb_ok, unb = _unbiased_equal(t1, t4, z=args.z)
        if not math.isnan(unb["mean"]):
            means_over_pairs.append(unb["mean"])
        total += 1; ok_cnt += int(equal); unbias_ok_cnt += int(unb_ok)
        if args.verbose or (not equal) or (not unb_ok):
            print(
                f"[CHK ] step={step} mb={mb} "
                f"equal={bool(stats['equal'])} allclose={bool(stats['allclose'])} "
                f"shape_eq={bool(stats['shape_equal'])} "
                f"max|diff|={stats['max_abs']:.3e} mean|diff|={stats['mean_abs']:.3e} | "
                f"unbiased_ok={unb_ok} |mean(diff)|={abs(unb['mean']):.3e} "
                f"se={unb['se']:.3e} z={args.z} n={unb['n']}"
            )

    over_ok, over = _unbiased_over_trials(means_over_pairs, z=args.z)
    print(f"[SUM ] total={total} "
          f"passed_equal/allclose={ok_cnt} failed_equal/allclose={total - ok_cnt} "
          f"unbiased_pass={unbias_ok_cnt} unbiased_fail={total - unbias_ok_cnt} "
          f"(rtol={args.rtol}, atol={args.atol}, z={args.z})")
    print(f"[TRIALS] unbiased over pairs: ok={over_ok} "
          f"mean={over['mean']:.3e} se={over['se']:.3e} n={over['n']}")
    raise SystemExit(0 if ok_cnt == total else 2)

if __name__ == "__main__":
    main()