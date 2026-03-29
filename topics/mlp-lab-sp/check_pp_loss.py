import argparse
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch


def load_loss_file(path: Path) -> Tuple[List[Optional[int]], List[float]]:
    """
    读取 loss 文件。支持两列(step, loss)或仅一列(loss)。
    返回(steps, losses)，steps 中无 step 时为 None。
    """
    steps: List[Optional[int]] = []
    losses: List[float] = []
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
                    steps.append(None)
                    losses.append(loss)
                    cur_step += 1
                else:
                    loss = float(parts[-1])
                    try:
                        step = int(float(parts[0]))
                        steps.append(step)
                    except ValueError:
                        steps.append(None)
                    losses.append(loss)
            except Exception:
                # 跳过无法解析的行
                continue
    return steps, losses


def align_by_steps(s1: List[Optional[int]], l1: List[float],
                   s2: List[Optional[int]], l2: List[float]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    若两侧都有有效 step，按公共 step 对齐；否则按索引对齐（取最短长度）。
    返回 (a, b, used_steps)
    """
    has_step1 = any(x is not None for x in s1)
    has_step2 = any(x is not None for x in s2)

    if has_step1 and has_step2:
        # 构建 step -> loss 映射（遇重复保留最后一个）
        d1: Dict[int, float] = {}
        d2: Dict[int, float] = {}
        for st, v in zip(s1, l1):
            if st is not None:
                d1[st] = v
        for st, v in zip(s2, l2):
            if st is not None:
                d2[st] = v
        common = sorted(set(d1.keys()) & set(d2.keys()))
        a = torch.tensor([d1[k] for k in common], dtype=torch.float64)
        b = torch.tensor([d2[k] for k in common], dtype=torch.float64)
        return a, b, common
    else:
        L = min(len(l1), len(l2))
        a = torch.tensor(l1[:L], dtype=torch.float64)
        b = torch.tensor(l2[:L], dtype=torch.float64)
        idx_steps = list(range(L))
        return a, b, idx_steps


def main():
    ap = argparse.ArgumentParser(description="比较两个 PP 配置下的 loss 是否相同")
    ap.add_argument("--file1", type=str, default="pp_test_dp1_tp1_pp1_step1000_loss.txt",
                    help="文件1路径（默认：pp=1）")
    ap.add_argument("--file2", type=str, default="pp_test_dp1_tp1_pp4_step1000_loss.txt",
                    help="文件2路径（默认：pp=2）")
    ap.add_argument("--rtol", type=float, default=0.0, help="相对容差（allclose）")
    ap.add_argument("--atol", type=float, default=0.0, help="绝对容差（allclose）")
    ap.add_argument("--start", type=int, default=0, help="起始索引/步（含）")
    ap.add_argument("--end", type=int, default=None, help="结束索引/步（不含），默认到末尾")
    ap.add_argument("--verbose", action="store_true", help="打印逐项差异")
    args = ap.parse_args()

    p1 = Path(args.file1)
    p2 = Path(args.file2)
    if not p1.exists() or not p2.exists():
        print(f"[ERR] 文件不存在: {p1 if not p1.exists() else ''} {p2 if not p2.exists() else ''}")
        raise SystemExit(1)

    s1, l1 = load_loss_file(p1)
    s2, l2 = load_loss_file(p2)
    if len(l1) == 0 or len(l2) == 0:
        print("[ERR] 输入文件无有效数据")
        raise SystemExit(1)

    a_all, b_all, used_steps = align_by_steps(s1, l1, s2, l2)

    # 选择范围
    L = a_all.numel()
    start = max(0, args.start)
    end = L if args.end is None else min(L, args.end)
    if end <= start:
        print(f"[ERR] 比较范围无效: start={start}, end={end}, L={L}")
        raise SystemExit(1)

    a = a_all[start:end]
    b = b_all[start:end]
    diff = b - a
    abs_diff = diff.abs()

    allclose_mask = torch.isclose(a, b, rtol=args.rtol, atol=args.atol)
    allclose_all = bool(allclose_mask.all().item())

    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    mse = float((diff.pow(2).mean().item()))
    eq_all = bool(torch.equal(a, b))

    # 打印
    print(f"[INFO] file1={p1} 共有 {len(l1)} 条; file2={p2} 共有 {len(l2)} 条")
    print(f"[RANGE] 对齐后比较区间 [{start}, {end}) / {L} 条")
    print(f"[DIFF ] exact_equal={eq_all} allclose_all={allclose_all} "
          f"max|diff|={max_abs:.6e} mean|diff|={mean_abs:.6e} mse={mse:.6e} "
          f"(rtol={args.rtol}, atol={args.atol})")

    if args.verbose:
        for i in range(end - start):
            step_str = f"step={used_steps[start+i]}" if used_steps and isinstance(used_steps[0], int) else f"idx={start+i}"
            print(f"[ITEM] {step_str} a={a[i].item():.6e} b={b[i].item():.6e} "
                  f"diff={diff[i].item():.3e} allclose={bool(allclose_mask[i].item())}")

    # 通过条件：完全相等或全 allclose
    ok = eq_all or allclose_all
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()