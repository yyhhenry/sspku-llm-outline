"""
KV-Cache量化模块 - 针对Transformer的推理优化

本模块实现KV-Cache量化，用于优化Transformer模型的推理内存和速度，包括：
1. 动态范围追踪
2. KV cache的逐层量化
3. 注意力计算中的在线量化/反量化
4. 内存占用优化

作者: AI Research Lab
难度: ⭐⭐⭐⭐ (较难)
预计完成时间: 4-5小时
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math

from .quantization_basics import QuantizationConfig, quantize_tensor, dequantize_tensor


class QuantizedKVCache:
    """
    量化的KV Cache
    
    学生任务:
        1. 实现KV cache的存储和量化
        2. 动态更新量化参数
        3. 在注意力计算时反量化
    """
    
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        config: QuantizationConfig,
        device: str = 'cpu',
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.config = config
        self.device = device
        
        # 预分配量化cache
        dtype = torch.int8 if config.n_bits == 8 else torch.int32
        self.k_cache = torch.zeros(
            (max_batch_size, num_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        
        # 量化参数
        self.k_scale = torch.ones((max_batch_size, num_heads, 1, 1), device=device)
        self.k_zero_point = torch.zeros((max_batch_size, num_heads, 1, 1), dtype=torch.int32, device=device)
        self.v_scale = torch.ones_like(self.k_scale)
        self.v_zero_point = torch.zeros_like(self.k_zero_point)
        
        self.current_length = 0
    
    def update(self, k: torch.Tensor, v: torch.Tensor, start_pos: int) -> None:
        """
        更新KV cache
        
        学生任务:
            1. 计算新的k, v的量化参数
            2. 量化并存储
            3. 更新current_length
        
        参数:
            k: (batch, num_heads, seq_len, head_dim)
            v: 同上
            start_pos: 当前序列起始位置
        """
        # ==================== YOUR CODE HERE (开始) ====================
        batch, _, seq_len, _ = k.shape
        
        # 计算量化参数（逐头）
        k_min = k.amin(dim=(2, 3), keepdim=True)
        k_max = k.amax(dim=(2, 3), keepdim=True)
        
        if self.config.symmetric:
            k_abs_max = torch.maximum(k_min.abs(), k_max.abs())
            qmax = 2 ** (self.config.n_bits - 1) - 1
            k_scale = k_abs_max / qmax
            k_zero_point = torch.zeros_like(k_scale, dtype=torch.int32)
        else:
            quant_range = self.config.quant_max - self.config.quant_min
            k_scale = (k_max - k_min) / quant_range
            k_zero_point = self.config.quant_min - torch.round(k_min / k_scale)
            k_zero_point = torch.clamp(k_zero_point, self.config.quant_min, self.config.quant_max).to(torch.int32)
        
        # V的量化参数
        v_min = v.amin(dim=(2, 3), keepdim=True)
        v_max = v.amax(dim=(2, 3), keepdim=True)
        
        if self.config.symmetric:
            v_abs_max = torch.maximum(v_min.abs(), v_max.abs())
            v_scale = v_abs_max / qmax
            v_zero_point = torch.zeros_like(v_scale, dtype=torch.int32)
        else:
            v_scale = (v_max - v_min) / quant_range
            v_zero_point = self.config.quant_min - torch.round(v_min / v_scale)
            v_zero_point = torch.clamp(v_zero_point, self.config.quant_min, self.config.quant_max).to(torch.int32)
        
        # 量化
        k_quant = quantize_tensor(k, k_scale, k_zero_point, self.config.quant_min, self.config.quant_max)
        v_quant = quantize_tensor(v, v_scale, v_zero_point, self.config.quant_min, self.config.quant_max)
        
        # 存储
        end_pos = start_pos + seq_len
        self.k_cache[:batch, :, start_pos:end_pos, :] = k_quant
        self.v_cache[:batch, :, start_pos:end_pos, :] = v_quant
        
        # 更新量化参数
        self.k_scale[:batch] = k_scale
        self.k_zero_point[:batch] = k_zero_point
        self.v_scale[:batch] = v_scale
        self.v_zero_point[:batch] = v_zero_point
        
        self.current_length = max(self.current_length, end_pos)
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def get(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取并反量化KV cache
        
        学生任务:
            1. 取出当前长度的cache
            2. 反量化
            3. 返回浮点KV
        """
        # ==================== YOUR CODE HERE (开始) ====================
        
        k_quant = self.k_cache[:batch_size, :, :self.current_length, :]
        v_quant = self.v_cache[:batch_size, :, :self.current_length, :]
        
        k = dequantize_tensor(k_quant, self.k_scale[:batch_size], self.k_zero_point[:batch_size])
        v = dequantize_tensor(v_quant, self.v_scale[:batch_size], self.v_zero_point[:batch_size])
        
        return k, v
        
        # ==================== YOUR CODE HERE (结束) ====================


def test_kv_cache():
    """测试KV Cache量化"""
    print("=" * 80)
    print("测试KV-Cache量化")
    print("=" * 80)
    
    config = QuantizationConfig(n_bits=8, symmetric=False)
    cache = QuantizedKVCache(
        max_batch_size=2,
        max_seq_len=128,
        num_heads=8,
        head_dim=64,
        config=config,
    )
    
    # 模拟自回归生成
    batch = 2
    for step in range(10):
        k = torch.randn(batch, 8, 1, 64)
        v = torch.randn(batch, 8, 1, 64)
        
        cache.update(k, v, start_pos=step)
    
    # 获取cache
    k_all, v_all = cache.get(batch)
    print(f"Cache长度: {k_all.shape[2]}")
    print(f"K范围: [{k_all.min():.4f}, {k_all.max():.4f}]")
    print(f"V范围: [{v_all.min():.4f}, {v_all.max():.4f}]")
    
    # 计算内存节省
    original_size = 2 * batch * 8 * 10 * 64 * 4  # FP32
    quantized_size = 2 * batch * 8 * 10 * 64 * 1  # INT8
    print(f"内存节省: {original_size / quantized_size:.2f}x")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_kv_cache()
