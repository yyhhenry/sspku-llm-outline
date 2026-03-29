import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict
import torch
import math

PAT_FALSE = re.compile(r"a_mb(\d+)_step(\d+)_rank0_spFalse\.pt$")
PAT_SP    = re.compile(r"a_mb(\d+)_step(\d+)_rank(\d+)\.pt$")

def load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                return v
    if isinstance(obj, (list, tuple)):
        for v in obj:
            if isinstance(v, torch.Tensor):
                return v
    raise ValueError(f"{path} 不含可识别 Tensor")

def find_false_files(root: Path) -> List[Tuple[int,int,Path]]:
    items = []
    for p in root.glob("a_mb*_step*_rank0_spFalse.pt"):
        m = PAT_FALSE.match(p.name)
        if m:
            mb = int(m.group(1)); step = int(m.group(2))
            items.append((step, mb, p))
    items.sort()
    return items

def build_sp_paths(root: Path, step: int, mb: int, ranks: List[int]) -> List[Path]:
    return [root / f"a_mb{mb}_step{step}_rank{r}.pt" for r in ranks]

def compare(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> Dict[str,float]:
    if a.shape != b.shape:
        return {
            "shape_equal": 0.0,
            "allclose": 0.0,
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
        }
    a_f = a.float(); b_f = b.float()
    diff = (a_f - b_f).abs()
    return {
        "shape_equal": 1.0,
        "allclose": float(torch.allclose(a_f, b_f, rtol=rtol, atol=atol)),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }

def unbiased_mean_zero(a: torch.Tensor, b: torch.Tensor, z: float = 3.0):
    if a.shape != b.shape:
        return False, {"mean": float("inf"), "se": float("inf"), "n": 0}
    diff = (a.float() - b.float())
    n = diff.numel()
    if n == 0:
        return True, {"mean": 0.0, "se": 0.0, "n": 0}
    mean = diff.mean()
    if n > 1:
        std = diff.std(unbiased=True)
        se = std / math.sqrt(n)
    else:
        std = torch.tensor(0.0); se = torch.tensor(0.0)
    ok = mean.abs() <= z * se + 1e-12
    return bool(ok.item()), {"mean": float(mean), "se": float(se), "n": n}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="sp_test_dp1_tp2_pp1")
    ap.add_argument("--ranks", type=int, nargs="+", default=[0,1], help="参与 SP 的 rank 列表（不含 _spFalse 的那个）")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.dir)
    assert root.is_dir(), f"目录不存在: {root}"

    false_files = find_false_files(root)
    if not false_files:
        print(f"[WARN] 未找到 *_spFalse.pt 文件")
        return

    total = 0
    pass_allclose = 0
    pass_unbiased = 0
    for step, mb, pf in false_files:
        total += 1
        try:
            t_false = load_tensor(pf)
        except Exception as e:
            print(f"[ERR ] 载入 spFalse 失败 step={step} mb={mb}: {e}")
            continue

        sp_paths = build_sp_paths(root, step, mb, args.ranks)
        missing = [str(p) for p in sp_paths if not p.exists()]
        if missing:
            print(f"[MISS] step={step} mb={mb} 缺少: {missing}")
            continue

        tensors = []
        ok_load = True
        for p in sp_paths:
            try:
                tensors.append(load_tensor(p))
            except Exception as e:
                print(f"[ERR ] 载入 sp 文件失败 {p}: {e}")
                ok_load = False
                break
        if not ok_load:
            continue

        try:
            t_cat = torch.cat(tensors, dim=1)
        except Exception as e:
            print(f"[ERR ] 拼接失败 step={step} mb={mb}: {e}")
            continue

        stats = compare(t_false, t_cat, args.rtol, args.atol)
        unbiased_ok, unbiased_stats = unbiased_mean_zero(t_false, t_cat, z=args.z)
        pass_allclose += int(stats["shape_equal"] == 1.0 and stats["allclose"] == 1.0)
        pass_unbiased += int(unbiased_ok)

        if args.verbose or stats["allclose"] != 1.0 or not unbiased_ok:
            print(
                f"[CHK ] step={step} mb={mb} "
                f"shape_eq={stats['shape_equal']} "
                f"allclose={int(stats['allclose'])} "
                f"max|diff|={stats['max_abs']:.3e} mean|diff|={stats['mean_abs']:.3e} | "
                f"unbiased_ok={unbiased_ok} mean(diff)={unbiased_stats['mean']:.3e} se={unbiased_stats['se']:.3e} n={unbiased_stats['n']}"
            )

    print(f"[SUM ] total={total} allclose_pass={pass_allclose} allclose_fail={total-pass_allclose} "
          f"unbiased_pass={pass_unbiased} unbiased_fail={total-pass_unbiased} "
          f"(rtol={args.rtol}, atol={args.atol}, z={args.z})")

if __name__ == "__main__":
    main()