import argparse
from pathlib import Path
from typing import List, Tuple
import math
import torch


def load_loss_file(path: Path) -> Tuple[List[int], List[float]]:
    """读取 loss 文件。支持两列(step, loss)或仅一列(loss)。返回(steps, losses)。"""
    steps, losses = [], []
    with path.open("r") as f:
        cur_step = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            try:
                if len(parts) == 1:
                    loss = float(parts[0])
                    steps.append(cur_step)
                    losses.append(loss)
                    cur_step += 1
                else:
                    # 取最后一列为 loss，第一列为 step（若可解析）
                    loss = float(parts[-1])
                    try:
                        step = int(float(parts[0]))
                    except ValueError:
                        step = cur_step
                    steps.append(step)
                    losses.append(loss)
                    cur_step = step + 1
            except Exception:
                # 跳过无法解析的行
                continue
    return steps, losses


def _unbiased_over_trials(values: torch.Tensor, z: float = 3.0):
    """无偏性检验：|mean(values)| <= z * SE(values)"""
    n = values.numel()
    if n == 0:
        return True, {"mean": 0.0, "std": 0.0, "se": 0.0, "n": 0}
    mean = values.mean()
    if n > 1:
        std = values.std(unbiased=True)
        se = std / math.sqrt(n)
    else:
        std = torch.tensor(0.0, dtype=values.dtype)
        se = torch.tensor(0.0, dtype=values.dtype)
    ok = (mean.abs() <= z * se + 1e-12)
    return bool(ok.item()), {
        "mean": float(mean.item()),
        "std": float(std.item()),
        "se": float(se.item()),
        "n": int(n),
    }


def main():
    ap = argparse.ArgumentParser(description="比较 dp1 与 dp4 的 loss 是否相同，并检查差值无偏性")
    ap.add_argument("--dp1", type=str, default="dp_test_dp1_tp1_pp1/loss_rank0.txt",
                    help="dp=1 的 loss 文件路径")
    ap.add_argument("--dp4", type=str, default="dp_test_dp4_tp1_pp1/loss_rank0.txt",
                    help="dp=4 的 loss 文件路径")
    ap.add_argument("--start", type=int, default=0, help="从该步开始比较（包含）")
    ap.add_argument("--end", type=int, default=None, help="比较到该步（不包含），默认到最末尾")
    ap.add_argument("--rtol", type=float, default=0.0, help="allclose 相对容差")
    ap.add_argument("--atol", type=float, default=0.0, help="allclose 绝对容差")
    ap.add_argument("--z", type=float, default=3.0, help="无偏性 z 系数")
    ap.add_argument("--verbose", action="store_true", help="打印逐步差异")
    args = ap.parse_args()

    p1 = Path(args.dp1)
    p4 = Path(args.dp4)
    if not p1.exists() or not p4.exists():
        print(f"[ERR] 文件不存在: {p1 if not p1.exists() else ''} {p4 if not p4.exists() else ''}")
        raise SystemExit(1)

    s1, l1 = load_loss_file(p1)
    s4, l4 = load_loss_file(p4)
    if len(l1) == 0 or len(l4) == 0:
        print("[ERR] 输入文件无有效数据")
        raise SystemExit(1)

    # 对齐范围
    L = min(len(l1), len(l4))
    start = max(0, args.start)
    end = min(L, args.end if args.end is not None else L)
    if end <= start:
        print(f"[ERR] 比较范围无效: start={start}, end={end}, L={L}")
        raise SystemExit(1)

    a = torch.tensor(l1[start:end], dtype=torch.float64)
    b = torch.tensor(l4[start:end], dtype=torch.float64)
    diff = b - a
    abs_diff = diff.abs()

    # allclose 判定（逐元素）
    allclose_mask = torch.isclose(a, b, rtol=args.rtol, atol=args.atol)
    allclose_all = bool(allclose_mask.all().item())

    # 统计
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    mse = float((diff.pow(2).mean().item()))
    unb_ok, unb = _unbiased_over_trials(diff, z=args.z)

    if args.verbose:
        for i in range(end - start):
            print(f"[STEP] idx={start+i} a={a[i].item():.6e} b={b[i].item():.6e} "
                  f"diff={diff[i].item():.3e} allclose={bool(allclose_mask[i].item())}")

    print(f"[RANGE] compare steps [{start}, {end}) out of {L}")
    print(f"[DIFF ] max|diff|={max_abs:.6e} mean|diff|={mean_abs:.6e} mse={mse:.6e} "
          f"(rtol={args.rtol}, atol={args.atol})")
    print(f"[ALLC ] allclose_all={allclose_all} passed={int(allclose_mask.sum().item())}/{end-start}")
    print(f"[UNB  ] unbiased_ok={unb_ok} mean(diff)={unb['mean']:.6e} se={unb['se']:.6e} "
          f"std={unb['std']:.6e} n={unb['n']} z={args.z}")

    # 通过条件：allclose_all 且 无偏性通过
    ok = allclose_all and unb_ok
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()