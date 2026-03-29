"""
GPTQ - 基于Hessian的权重量化算法

本模块实现GPTQ (Accurate Post-Training Quantization for GPT)，包括：
1. Hessian矩阵近似计算
2. 最优脑量化 (Optimal Brain Quantization)
3. 逐层、逐块量化
4. 分组量化策略
5. 动态规划优化

参考论文:
    GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
    https://arxiv.org/abs/2210.17323

作者: AI Research Lab
难度: ⭐⭐⭐⭐⭐ (高级)
预计完成时间: 6-8小时
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import math

from .quantization_basics import QuantizationConfig, quantize_tensor, dequantize_tensor


# ============================================================================
# 任务1: 实现Hessian矩阵计算
# ============================================================================

class HessianComputer:
    """
    Hessian矩阵计算器（使用Fisher信息矩阵近似）
    
    学生任务:
        1. 收集层输入的统计信息
        2. 计算Hessian近似: H ≈ 2 * X^T * X / n
        3. 支持分块计算以节省内存
        4. 计算Hessian的逆（用于GPTQ）
    
    关键概念:
        - Hessian矩阵表示损失函数对权重的二阶导数
        - 用于衡量每个权重的重要性
    """
    
    def __init__(self, layer: nn.Module):
        self.layer = layer
        self.inputs = []
        self.nsamples = 0
    
    def add_batch(self, inp: torch.Tensor) -> None:
        """
        添加一个batch的输入
        
        学生任务:
            1. 收集层的输入
            2. 对于Linear层，输入形状为 (batch, in_features)
            3. 对于Conv层，需要展开为 (batch*H*W, in_channels*K*K)
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现输入收集
        
        if isinstance(self.layer, nn.Linear):
            # Linear层：直接存储
            # inp shape: (batch_size, in_features)
            if len(inp.shape) == 3:  # (batch, seq, features)
                inp = inp.reshape(-1, inp.shape[-1])
            self.inputs.append(inp.cpu())
        elif isinstance(self.layer, nn.Conv2d):
            # Conv层：需要展开
            # inp shape: (batch, channels, height, width)
            # 使用unfold将卷积转换为矩阵乘法形式
            batch, channels, height, width = inp.shape
            kernel_size = self.layer.kernel_size
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            
            # Unfold: (batch, C*K*K, H_out*W_out)
            unfolded = torch.nn.functional.unfold(
                inp,
                kernel_size=kernel_size,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            # Reshape to (batch*H_out*W_out, C*K*K)
            unfolded = unfolded.transpose(1, 2).reshape(-1, unfolded.shape[1])
            self.inputs.append(unfolded.cpu())
        
        self.nsamples += inp.shape[0]
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def compute_hessian(self, device: str = 'cpu') -> torch.Tensor:
        """
        计算Hessian矩阵 (近似)
        
        学生任务:
            1. 将所有输入拼接: X = concat(inputs)
            2. 计算 H = 2 * X^T * X / n_samples
            3. 添加阻尼项避免奇异: H = H + lambda * I
        
        公式:
            H ≈ (2/n) * Σ(x_i * x_i^T)
        
        提示:
            - 使用torch.mm进行矩阵乘法
            - 阻尼系数lambda通常为1e-2到1e-1
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现Hessian计算
        
        if len(self.inputs) == 0:
            raise RuntimeError("No inputs collected")
        
        # 拼接所有输入
        X = torch.cat(self.inputs, dim=0).to(device)  # (n_samples, d)
        n, d = X.shape
        
        # 计算 H = 2 * X^T * X / n
        # 注意：实际可以不乘2，只是缩放因子
        H = 2.0 * torch.mm(X.t(), X) / n  # (d, d)
        
        # 添加阻尼项（正则化）
        damping = 0.01
        H += damping * torch.eye(d, device=device)
        
        # 清理内存
        del X
        self.inputs = []
        
        return H
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def compute_hessian_inv(self, device: str = 'cpu') -> torch.Tensor:
        """
        计算Hessian的逆矩阵
        
        学生任务:
            1. 计算Hessian矩阵
            2. 使用Cholesky分解计算逆: H = L * L^T, H^-1 = (L^-1)^T * L^-1
            3. 或使用torch.inverse（较慢但简单）
        
        提示:
            - torch.cholesky + torch.cholesky_inverse更快
            - 如果Cholesky失败，回退到torch.inverse
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现Hessian逆计算
        
        H = self.compute_hessian(device)
        
        try:
            # 尝试Cholesky分解（更快）
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        except RuntimeError:
            # 如果失败，使用普通求逆
            print("Warning: Cholesky decomposition failed, using torch.inverse")
            H_inv = torch.inverse(H)
        
        return H_inv
        
        # ==================== YOUR CODE HERE (结束) ====================


# ============================================================================
# 任务2: 实现GPTQ核心算法
# ============================================================================

class GPTQQuantizer:
    """
    GPTQ量化器
    
    学生任务:
        1. 实现逐列量化算法
        2. 使用Hessian逆矩阵计算最优量化
        3. 更新其他列以补偿量化误差
        4. 支持分组量化
    
    算法核心:
        For each column i in W:
            1. 量化 w_i
            2. 计算误差 e_i = w_i - quant(w_i)
            3. 更新剩余列: W[:, j>i] -= (H^-1[:, i] / H^-1[i, i]) * e_i
    """
    
    def __init__(
        self,
        layer: nn.Module,
        config: QuantizationConfig,
        group_size: int = 128,
    ):
        self.layer = layer
        self.config = config
        self.group_size = group_size
        
        self.hessian_computer = HessianComputer(layer)
        self.scale = None
        self.zero_point = None
    
    def add_batch(self, inp: torch.Tensor) -> None:
        """添加校准数据"""
        self.hessian_computer.add_batch(inp)
    
    def quantize_weight(self, device: str = 'cpu') -> torch.Tensor:
        """
        使用GPTQ算法量化权重
        
        学生任务:
            1. 获取权重矩阵 W (out_features, in_features)
            2. 计算Hessian逆矩阵 H_inv
            3. 逐列量化并更新
            4. 返回量化后的权重
        
        算法伪代码:
            ```
            H_inv = compute_hessian_inv()
            Q = zeros_like(W)  # 量化后的权重
            E = zeros_like(W)  # 累积误差
            
            for i in range(n_columns):
                # 量化当前列
                q_i = quantize(w_i)
                Q[:, i] = q_i
                
                # 计算误差
                e_i = w_i - q_i
                
                # 更新剩余列
                for j in range(i+1, n_columns):
                    W[:, j] -= (H_inv[i, j] / H_inv[i, i]) * e_i
            ```
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现GPTQ量化
        
        print(f"开始GPTQ量化，分组大小={self.group_size}")
        
        # 获取权重
        if isinstance(self.layer, nn.Linear):
            W = self.layer.weight.data.clone().to(device)  # (out_features, in_features)
        elif isinstance(self.layer, nn.Conv2d):
            # Conv: (out_channels, in_channels, k, k) -> reshape to (out, in*k*k)
            W = self.layer.weight.data.clone()
            out_channels = W.shape[0]
            W = W.reshape(out_channels, -1).to(device)
        else:
            raise ValueError(f"Unsupported layer type: {type(self.layer)}")
        
        out_dim, in_dim = W.shape
        
        # 计算Hessian逆
        print("计算Hessian逆矩阵...")
        H_inv = self.hessian_computer.compute_hessian_inv(device)
        
        if H_inv.shape[0] != in_dim:
            raise RuntimeError(
                f"Hessian dimension mismatch: H_inv={H_inv.shape[0]}, in_dim={in_dim}"
            )
        
        # 初始化量化后的权重
        Q = torch.zeros_like(W)
        
        # 计算每组的量化参数
        num_groups = math.ceil(in_dim / self.group_size)
        scales = []
        zero_points = []
        
        print("逐组量化...")
        for group_idx in tqdm(range(num_groups)):
            start_col = group_idx * self.group_size
            end_col = min((group_idx + 1) * self.group_size, in_dim)
            
            # 获取当前组
            W_group = W[:, start_col:end_col]
            
            # 计算量化参数（逐输出通道）
            if self.config.symmetric:
                max_val = W_group.abs().max(dim=1, keepdim=True)[0]
                qmax = 2 ** (self.config.n_bits - 1) - 1
                scale = max_val / qmax
                scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                zero_point = torch.zeros_like(scale, dtype=torch.int32)
            else:
                min_val = W_group.min(dim=1, keepdim=True)[0]
                max_val = W_group.max(dim=1, keepdim=True)[0]
                quant_range = self.config.quant_max - self.config.quant_min
                scale = (max_val - min_val) / quant_range
                scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                zero_point = self.config.quant_min - torch.round(min_val / scale)
                zero_point = torch.clamp(
                    zero_point,
                    self.config.quant_min,
                    self.config.quant_max
                ).to(torch.int32)
            
            scales.append(scale)
            zero_points.append(zero_point)
            
            # 逐列量化（GPTQ核心）
            for local_col in range(end_col - start_col):
                global_col = start_col + local_col
                
                # 当前列
                w_col = W[:, global_col].clone()
                
                # 量化
                w_quant = quantize_tensor(
                    w_col.unsqueeze(1),
                    scale,
                    zero_point,
                    self.config.quant_min,
                    self.config.quant_max,
                )
                w_dequant = dequantize_tensor(w_quant, scale, zero_point).squeeze(1)
                
                Q[:, global_col] = w_dequant
                
                # 计算误差
                error = w_col - w_dequant
                
                # 更新后续列（Hessian补偿）
                if global_col < in_dim - 1:
                    # 使用Hessian逆矩阵更新
                    h_inv_ii = H_inv[global_col, global_col]
                    if h_inv_ii.abs() > 1e-8:
                        # 更新剩余列
                        for j in range(global_col + 1, in_dim):
                            W[:, j] -= (H_inv[global_col, j] / h_inv_ii) * error
        
        # 存储量化参数
        self.scale = torch.cat(scales, dim=1) if len(scales) > 1 else scales[0]
        self.zero_point = torch.cat(zero_points, dim=1) if len(zero_points) > 1 else zero_points[0]
        
        print("GPTQ量化完成！")
        return Q
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def apply_quantization(self, device: str = 'cpu') -> None:
        """
        应用量化到层
        
        学生任务:
            1. 调用quantize_weight
            2. 更新层的权重
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 应用量化
        
        Q = self.quantize_weight(device)
        
        # 更新权重
        if isinstance(self.layer, nn.Linear):
            self.layer.weight.data = Q
        elif isinstance(self.layer, nn.Conv2d):
            # Reshape回卷积形状
            original_shape = self.layer.weight.data.shape
            self.layer.weight.data = Q.reshape(original_shape)
        
        # ==================== YOUR CODE HERE (结束) ====================


# ============================================================================
# 任务3: 实现模型级GPTQ量化
# ============================================================================

class GPTQModelQuantizer:
    """
    对整个模型应用GPTQ
    
    学生任务:
        1. 遍历模型的所有层
        2. 为每层收集输入并应用GPTQ
        3. 支持跳过某些层
        4. 实现顺序量化策略
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        group_size: int = 128,
        skip_layers: Optional[List[str]] = None,
    ):
        self.model = model
        self.config = config
        self.group_size = group_size
        self.skip_layers = skip_layers or []
        
        self.quantizers = {}
    
    def prepare(self) -> None:
        """
        准备量化：为每层创建量化器和hook
        
        学生任务:
            1. 遍历模型找到Linear和Conv层
            2. 创建GPTQ量化器
            3. 注册hook收集输入
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 准备量化
        
        def should_quantize(name: str) -> bool:
            """判断层是否需要量化"""
            for skip_pattern in self.skip_layers:
                if skip_pattern in name:
                    return False
            return True
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if not should_quantize(name):
                    print(f"跳过层: {name}")
                    continue
                
                # 创建量化器
                quantizer = GPTQQuantizer(module, self.config, self.group_size)
                self.quantizers[name] = quantizer
                
                # 注册hook收集输入
                def make_hook(quant):
                    def hook(module, input, output):
                        inp = input[0].detach()
                        quant.add_batch(inp)
                    return hook
                
                module.register_forward_hook(make_hook(quantizer))
        
        print(f"准备量化 {len(self.quantizers)} 层")
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def quantize(self, calibration_loader, num_batches: int = 100, device: str = 'cpu') -> None:
        """
        执行GPTQ量化
        
        学生任务:
            1. 运行校准数据收集输入
            2. 逐层应用GPTQ
            3. 清理hook
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 执行量化
        
        print(f"收集校准数据（{num_batches} batches）...")
        self.model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(calibration_loader, total=num_batches)):
                if i >= num_batches:
                    break
                
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                
                inputs = inputs.to(device)
                
                if isinstance(inputs, dict):
                    self.model(**inputs)
                else:
                    self.model(inputs)
        
        print("\n逐层应用GPTQ...")
        for name, quantizer in self.quantizers.items():
            print(f"\n量化层: {name}")
            quantizer.apply_quantization(device)
        
        print("\nGPTQ量化完成！")
        
        # ==================== YOUR CODE HERE (结束) ====================


# ============================================================================
# 测试代码
# ============================================================================

def test_gptq():
    """测试GPTQ功能"""
    print("=" * 80)
    print("测试GPTQ模块")
    print("=" * 80)
    
    # 测试1: Hessian计算
    print("\n测试1: Hessian计算")
    print("-" * 80)
    
    layer = nn.Linear(128, 64)
    hessian_computer = HessianComputer(layer)
    
    # 添加模拟数据
    for _ in range(10):
        inp = torch.randn(32, 128)
        hessian_computer.add_batch(inp)
    
    H = hessian_computer.compute_hessian()
    print(f"Hessian形状: {H.shape}")
    print(f"Hessian对称性检查: {torch.allclose(H, H.t(), atol=1e-5)}")
    
    H_inv = hessian_computer.compute_hessian_inv()
    print(f"Hessian逆形状: {H_inv.shape}")
    
    # 验证 H * H_inv ≈ I
    identity = torch.mm(H, H_inv)
    error = (identity - torch.eye(128)).abs().max().item()
    print(f"H * H_inv误差: {error:.6f}")
    
    # 测试2: GPTQ量化
    print("\n测试2: GPTQ量化")
    print("-" * 80)
    
    layer = nn.Linear(256, 128)
    config = QuantizationConfig(n_bits=4, symmetric=True)
    gptq = GPTQQuantizer(layer, config, group_size=64)
    
    # 收集数据
    for _ in range(20):
        inp = torch.randn(16, 256)
        gptq.add_batch(inp)
    
    # 量化
    original_weight = layer.weight.data.clone()
    Q = gptq.quantize_weight()
    
    # 分析误差
    error = (original_weight - Q).abs()
    print(f"量化误差统计:")
    print(f"  平均: {error.mean():.6f}")
    print(f"  最大: {error.max():.6f}")
    print(f"  中位数: {error.median():.6f}")
    
    # 计算SNR
    signal_power = (original_weight ** 2).mean()
    noise_power = (error ** 2).mean()
    snr = 10 * torch.log10(signal_power / noise_power)
    print(f"  SNR: {snr:.2f} dB")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    test_gptq()
