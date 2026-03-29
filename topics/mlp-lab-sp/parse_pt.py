import argparse
from pathlib import Path
import torch

def summarize_tensor(t: torch.Tensor, prefix=""):
    t = t.detach().cpu()
    flat = t.reshape(-1)
    print(f"{prefix}Tensor: shape={tuple(t.shape)}, dtype={t.dtype}")
    if flat.numel() > 0:
        print(f"{prefix}  stats: mean={flat.mean().item():.6e}, std={flat.std(unbiased=False).item():.6e}, "
              f"min={flat.min().item():.6e}, max={flat.max().item():.6e}, numel={flat.numel()}")
        preview = flat[:10].tolist()
        print(f"{prefix}  first10: {preview}")
    else:
        print(f"{prefix}  (empty)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default="dp_test_dp1_tp1_pp1/grad_mb0_step0_rank0.pt",
                    help="文件路径")
    ap.add_argument("--max-items", type=int, default=5, help="list/dict 打印的最大项数")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"[ERR] 文件不存在: {path}")
        return

    obj = torch.load(path, map_location="cpu")
    print(f"[INFO] 加载: {path}")
    if isinstance(obj, torch.Tensor):
        summarize_tensor(obj)
    elif isinstance(obj, (list, tuple)):
        print(f"List/Tuple，长度={len(obj)}")
        total = 0
        for i, v in enumerate(obj):
            if isinstance(v, torch.Tensor):
                total += v.numel()
        print(f"总元素数(numel) ≈ {total}")
        for i, v in enumerate(obj[:args.max_items]):
            print(f"[{i}] 类型={type(v).__name__}")
            if isinstance(v, torch.Tensor):
                summarize_tensor(v, prefix="  ")
            else:
                print(f"  值预览: {repr(v)}")
        if len(obj) > args.max_items:
            print(f"... 省略 {len(obj) - args.max_items} 项")
    elif isinstance(obj, dict):
        print(f"Dict，键数={len(obj)}，键列表（前{args.max_items}个）：{list(obj.keys())[:args.max_items]}")
        shown = 0
        for k, v in obj.items():
            if shown >= args.max_items:
                break
            print(f"[{k}] 类型={type(v).__name__}")
            if isinstance(v, torch.Tensor):
                summarize_tensor(v, prefix="  ")
            else:
                print(f"  值预览: {repr(v)}")
            shown += 1
        if len(obj) > args.max_items:
            print(f"... 省略 {len(obj) - args.max_items} 项")
    else:
        print(f"不支持的对象类型: {type(obj)}")
        print(repr(obj))

if __name__ == "__main__":
    main()