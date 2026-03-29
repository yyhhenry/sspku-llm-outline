import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import math

PAT = re.compile(r"a_mb(\d+)_step(\d+)_rank0\.pt$")

def load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        if len(obj) == 1 and isinstance(obj[0], torch.Tensor):
            return obj[0]
        raise ValueError(f"{path} contains list/tuple, expected single Tensor.")
    if isinstance(obj, dict):
        # try common keys or first tensor value
        for k in ["tensor", "data", "activation", "a_mb", "act", "out"]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k]
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                return v
        raise ValueError(f"{path} dict has no Tensor values.")
    raise ValueError(f"{path} contains unsupported type: {type(obj)}")

def find_dp1_pairs(dp1_dir: Path) -> List[Tuple[int, int, Path]]:
    files = []
    for p in dp1_dir.glob("a_mb*_step*_rank0.pt"):
        m = PAT.search(p.name)
        if m:
            mb = int(m.group(1))
            step = int(m.group(2))
            files.append((step, mb, p))
    files.sort()  # sort by step, then mb
    return files

def compare_tensors(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> Dict[str, float]:
    # 若形状不同，直接返回不相等，避免逐元素运算报错
    if a.shape != b.shape:
        return {
            "equal": 0.0,
            "allclose": 0.0,
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "shape_equal": 0.0,
        }
    if a.dtype != b.dtype:
        a = a.float()
        b = b.float()
    diff = (a - b).abs()
    return {
        "equal": float(torch.equal(a, b)),
        "allclose": float(torch.allclose(a, b, rtol=rtol, atol=atol)),
        "max_abs": float(diff.max().item()) if diff.numel() > 0 else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() > 0 else 0.0,
        "shape_equal": 1.0,
    }

def _unbiased_equal(a: torch.Tensor, b: torch.Tensor, z: float = 3.0):
    """无偏性检验：|mean(diff)| <= z * SE"""
    if a.shape != b.shape:
        return False, {"mean": float("nan"), "std": float("nan"), "se": float("nan"), "max_abs": float("inf"), "n": 0}
    diff = (a - b).detach().float()
    n = diff.numel()
    if n == 0:
        return True, {"mean": 0.0, "std": 0.0, "se": 0.0, "max_abs": 0.0, "n": 0}
    mean = diff.mean()
    if n > 1:
        std = diff.std(unbiased=True)
        se = std / math.sqrt(n)
    else:
        std = torch.tensor(0.0, dtype=torch.float32)
        se = torch.tensor(0.0, dtype=torch.float32)
    ok = (mean.abs() <= z * se + 1e-12)
    return bool(ok.item()), {
        "mean": float(mean.item()),
        "std": float(std.item()),
        "se": float(se.item()),
        "max_abs": float(diff.abs().max().item()),
        "n": int(n),
    }

def _unbiased_over_trials(values: List[float], z: float = 3.0):
    """对跨样本的 mean(diff) 列表做无偏性检验"""
    if len(values) == 0:
        return True, {"mean": 0.0, "std": 0.0, "se": 0.0, "n": 0}
    t = torch.as_tensor(values, dtype=torch.float64)
    n = t.numel()
    mean = t.mean()
    if n > 1:
        std = t.std(unbiased=True)
        se = std / math.sqrt(n)
    else:
        std = torch.tensor(0.0, dtype=torch.float64)
        se = torch.tensor(0.0, dtype=torch.float64)
    ok = (mean.abs() <= z * se + 1e-12)
    return bool(ok.item()), {
        "mean": float(mean.item()),
        "std": float(std.item()),
        "se": float(se.item()),
        "n": int(n),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp1-dir", type=str, default="dp_test_dp1_tp1_pp1", help="目录：dp=1 的激活文件")
    ap.add_argument("--dp4-dir", type=str, default="dp_test_dp4_tp1_pp1", help="目录：dp=4 的激活文件")
    ap.add_argument("--dp4-world-size", type=int, default=4, help="dp=4 的 rank 数")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--z", type=float, default=3.0, help="无偏性检验阈值系数 z")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    dp1_dir = Path(args.dp1_dir)
    dp4_dir = Path(args.dp4_dir)
    assert dp1_dir.is_dir(), f"dp1-dir not found: {dp1_dir}"
    assert dp4_dir.is_dir(), f"dp4-dir not found: {dp4_dir}"

    pairs = find_dp1_pairs(dp1_dir)
    if not pairs:
        print(f"[WARN] 未在 {dp1_dir} 找到匹配文件 a_mb*_step*_rank0.pt")
        raise SystemExit(1)

    total = 0
    ok_cnt = 0
    unbias_ok_cnt = 0
    means_over_pairs: List[float] = []
    for step, mb, p1 in pairs:
        # load dp1 tensor (rank0)
        t1 = load_tensor(p1)  # [B, H]
        # load dp4 tensors and concat on dim=0
        t_list = []
        missing = []
        for r in range(args.dp4_world_size):
            p4 = dp4_dir / f"a_mb{mb}_step{step}_rank{r}.pt"
            if not p4.exists():
                missing.append(str(p4))
            else:
                t_list.append(load_tensor(p4))
        if missing:
            print(f"[MISS] step={step} mb={mb} 缺少 dp4 文件: {missing}")
            total += 1
            continue
        try:
            t4 = torch.cat(t_list, dim=0)
        except Exception as e:
            print(f"[ERR ] step={step} mb={mb} 拼接失败: {e}")
            total += 1
            continue

        stats = compare_tensors(t1, t4, args.rtol, args.atol)
        equal = bool(stats["equal"] or stats["allclose"])
        unb_ok, unb = _unbiased_equal(t1, t4, z=args.z)
        if not math.isnan(unb["mean"]):
            means_over_pairs.append(unb["mean"])
        total += 1
        ok_cnt += int(equal)
        unbias_ok_cnt += int(unb_ok)
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