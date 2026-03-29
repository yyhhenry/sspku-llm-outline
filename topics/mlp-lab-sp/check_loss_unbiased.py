import argparse
from pathlib import Path
import math
import re

FLOAT_PAT = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
LOSS_PAT  = re.compile(r"loss\s*[:=]\s*(" + FLOAT_PAT.pattern + r")", re.IGNORECASE)

def _pick_number(tokens):
    # 优先带小数点或科学计数
    for t in tokens:
        if "." in t or "e" in t.lower():
            return t
    # 否则取最后一个，避免把行号当作值
    return tokens[-1]

def load_losses(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")
    vals = []
    with path.open("r") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            m_loss = LOSS_PAT.search(s)
            if m_loss:
                tok = m_loss.group(1)
            else:
                toks = FLOAT_PAT.findall(s)
                if not toks:
                    continue
                tok = _pick_number(toks)
            try:
                v = float(tok)
            except ValueError:
                continue
            if math.isfinite(v):
                vals.append(v)
    if not vals:
        raise ValueError(f"文件为空或无有效数字: {path}")
    return vals

def unbiased_test(a, b, z=3.0):
    if len(a) != len(b):
        return {
            "same_length": False,
            "n": min(len(a), len(b)),
            "mean_diff": None,
            "std_diff": None,
            "se_diff": None,
            "z_score": None,
            "unbiased": False,
        }
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean_diff = sum(diffs) / n
    if n > 1:
        var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        std = math.sqrt(var)
        se = std / math.sqrt(n)
    else:
        std = 0.0
        se = 0.0
    z_score = mean_diff / (se + 1e-12) if se > 0 else 0.0
    unbiased = abs(mean_diff) <= z * se + 1e-12
    return {
        "same_length": True,
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std,
        "se_diff": se,
        "z_score": z_score,
        "unbiased": unbiased,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="sp_test_dp2_tp2_pp2")
    ap.add_argument("--ranks", type=int, nargs="+", default=[4,5,6,7])
    ap.add_argument("--z", type=float, default=3.0, help="无偏判定阈值倍数 (|mean| <= z*SE)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"目录不存在: {root}")
        return

    summary = []
    for r in args.ranks:
        f_false = root / f"loss_rank{r}_spFalse.txt"
        f_true  = root / f"loss_rank{r}_spTrue.txt"
        try:
            l_false = load_losses(f_false)
            l_true  = load_losses(f_true)
        except Exception as e:
            print(f"[rank {r}] 载入失败: {e}")
            continue
        stats = unbiased_test(l_false, l_true, z=args.z)
        summary.append((r, stats))
        if args.verbose:
            print(f"[rank {r}] n={stats['n']} same_len={stats['same_length']} "
                  f"mean_diff={stats['mean_diff']:.6g} std={stats['std_diff']:.6g} "
                  f"se={stats['se_diff']:.6g} z_score={stats['z_score']:.3f} "
                  f"unbiased={stats['unbiased']}")

    # 汇总
    total = len(summary)
    unbiased_cnt = sum(1 for _, s in summary if s["unbiased"])
    print(f"[SUM] ranks={total} unbiased={unbiased_cnt} biased={total - unbiased_cnt} (阈值 z={args.z})")
    for r, s in summary:
        print(f"rank {r}: unbiased={s['unbiased']} mean_diff={s['mean_diff']:.6g} se={s['se_diff']:.6g} z={s['z_score']:.3f}")

if __name__ == "__main__":
    main()