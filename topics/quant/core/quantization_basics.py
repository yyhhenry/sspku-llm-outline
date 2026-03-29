"""
量化基础模块 - Quantization Basics

本模块包含量化的核心实现，学生需要完成以下内容：
1. 对称量化 (Symmetric Quantization)
2. 非对称量化 (Asymmetric Quantization)
3. 逐张量量化 (Per-Tensor Quantization)
4. 逐通道量化 (Per-Channel Quantization)
5. 量化参数计算 (Scale & Zero-point)
6. 量化/反量化操作

作者: AI Research Lab
难度: ⭐⭐ (基础但重要)
预计完成时间: 3-4小时
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union
import warnings


class QuantizationConfig:
    """量化配置类"""
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        channel_axis: int = 0,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
    ):
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        
        # 计算量化范围
        if quant_min is None or quant_max is None:
            if symmetric:
                # 对称量化: [-2^(n-1), 2^(n-1)-1]
                self.quant_min = -(2 ** (n_bits - 1))
                self.quant_max = 2 ** (n_bits - 1) - 1
            else:
                # 非对称量化: [0, 2^n-1]
                self.quant_min = 0
                self.quant_max = 2 ** n_bits - 1
        else:
            self.quant_min = quant_min
            self.quant_max = quant_max
    
    def __repr__(self):
        return (f"QuantizationConfig(n_bits={self.n_bits}, symmetric={self.symmetric}, "
                f"per_channel={self.per_channel}, range=[{self.quant_min}, {self.quant_max}])")


# ============================================================================
# 任务1: 实现量化参数计算
# ============================================================================

def calculate_qparams_symmetric(
    tensor: torch.Tensor,
    n_bits: int = 8,
    per_channel: bool = False,
    channel_axis: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算对称量化的参数 (scale, zero_point)
    
    对称量化公式:
        scale = max(|tensor_min|, |tensor_max|) / (2^(n_bits-1) - 1)
        zero_point = 0
    
    学生任务:
        1. 根据per_channel参数，确定在哪个维度上计算min/max
        2. 计算scale，使得量化后的值域充分利用量化范围
        3. 对称量化的zero_point始终为0
        4. 处理tensor全为0的边界情况
    
    参数:
        tensor: 输入张量 (需要量化的权重或激活)
        n_bits: 量化位宽
        per_channel: 是否逐通道量化
        channel_axis: 通道所在的轴
    
    返回:
        scale: 量化缩放因子 (形状取决于per_channel)
        zero_point: 量化零点 (对称量化时为0)
    
    提示:
        - 使用torch.abs()计算绝对值
        - per_channel时，使用keepdim=True保持维度
        - 避免除以0的情况
    """
    # ==================== YOUR CODE HERE (开始) ====================
    # TODO: 实现对称量化参数计算
    
    # 步骤1: 计算张量的最大绝对值
    # 如果per_channel=True，需要在除channel_axis外的维度上计算
    # 如果per_channel=False，在整个张量上计算
    
    if per_channel:
        # 获取除channel_axis外的所有维度
        reduce_dims = list(range(tensor.ndim))
        reduce_dims.pop(channel_axis)
        
        # 在这些维度上计算最大绝对值
        max_val = torch.amax(torch.abs(tensor), dim=reduce_dims, keepdim=True)
    else:
        # 逐张量：整个张量的最大绝对值
        max_val = torch.abs(tensor).max()
    
    # 步骤2: 计算量化范围
    qmax = 2 ** (n_bits - 1) - 1  # 对称量化: 127 for 8-bit
    
    # 步骤3: 计算scale
    # scale = max_val / qmax
    # 注意: 当max_val为0时，设置scale为1以避免除以0
    scale = max_val / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    
    # 步骤4: zero_point对于对称量化始终为0
    zero_point = torch.zeros_like(scale, dtype=torch.int32)
    
    # ==================== YOUR CODE HERE (结束) ====================
    
    return scale, zero_point


def calculate_qparams_asymmetric(
    tensor: torch.Tensor,
    n_bits: int = 8,
    per_channel: bool = False,
    channel_axis: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算非对称量化的参数 (scale, zero_point)
    
    非对称量化公式:
        scale = (tensor_max - tensor_min) / (quant_max - quant_min)
        zero_point = quant_min - round(tensor_min / scale)
    
    学生任务:
        1. 计算张量的min和max值
        2. 根据公式计算scale和zero_point
        3. zero_point需要clamp到量化范围内
        4. 处理边界情况（如tensor_min == tensor_max）
    
    参数:
        tensor: 输入张量
        n_bits: 量化位宽
        per_channel: 是否逐通道量化
        channel_axis: 通道所在的轴
    
    返回:
        scale: 量化缩放因子
        zero_point: 量化零点
    
    提示:
        - 非对称量化可以更好地利用量化范围
        - zero_point的类型应为整数
        - 使用torch.clamp确保zero_point在有效范围内
    """
    # ==================== YOUR CODE HERE (开始) ====================
    # TODO: 实现非对称量化参数计算
    
    # 步骤1: 计算min和max
    if per_channel:
        reduce_dims = list(range(tensor.ndim))
        reduce_dims.pop(channel_axis)
        
        min_val = torch.amin(tensor, dim=reduce_dims, keepdim=True)
        max_val = torch.amax(tensor, dim=reduce_dims, keepdim=True)
    else:
        min_val = tensor.min()
        max_val = tensor.max()
    
    # 步骤2: 计算量化范围
    quant_min = 0
    quant_max = 2 ** n_bits - 1  # 255 for 8-bit
    
    # 步骤3: 计算scale
    # 注意: 当min_val == max_val时，设置合理的scale
    scale = (max_val - min_val) / (quant_max - quant_min)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    
    # 步骤4: 计算zero_point
    # zero_point = quant_min - round(min_val / scale)
    zero_point = quant_min - torch.round(min_val / scale)
    zero_point = torch.clamp(zero_point, quant_min, quant_max).to(torch.int32)
    
    # ==================== YOUR CODE HERE (结束) ====================
    
    return scale, zero_point


# ============================================================================
# 任务2: 实现量化和反量化操作
# ============================================================================

def quantize_tensor(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype = torch.int8,
) -> torch.Tensor:
    """
    将浮点张量量化为整数张量
    
    量化公式:
        quantized = clamp(round(tensor / scale) + zero_point, quant_min, quant_max)
    
    学生任务:
        1. 将tensor除以scale
        2. 加上zero_point
        3. 四舍五入到最近的整数
        4. clamp到[quant_min, quant_max]范围
        5. 转换为目标dtype
    
    参数:
        tensor: 输入浮点张量
        scale: 量化缩放因子
        zero_point: 量化零点
        quant_min: 量化最小值
        quant_max: 量化最大值
        dtype: 量化后的数据类型
    
    返回:
        quantized: 量化后的整数张量
    
    提示:
        - torch.round()用于四舍五入
        - torch.clamp()用于限制范围
        - 注意处理broadcasting
    """
    # ==================== YOUR CODE HERE (开始) ====================
    # TODO: 实现量化操作
    
    # 步骤1: 缩放并偏移
    quantized = tensor / scale + zero_point
    
    # 步骤2: 四舍五入
    quantized = torch.round(quantized)
    
    # 步骤3: Clamp到有效范围
    quantized = torch.clamp(quantized, quant_min, quant_max)
    
    # 步骤4: 转换数据类型
    quantized = quantized.to(dtype)
    
    # ==================== YOUR CODE HERE (结束) ====================
    
    return quantized


def dequantize_tensor(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    将量化的整数张量反量化为浮点张量
    
    反量化公式:
        dequantized = (quantized - zero_point) * scale
    
    学生任务:
        1. 减去zero_point
        2. 乘以scale
        3. 转换为目标浮点类型
    
    参数:
        quantized: 量化后的整数张量
        scale: 量化缩放因子
        zero_point: 量化零点
        dtype: 反量化后的数据类型
    
    返回:
        dequantized: 反量化后的浮点张量
    
    提示:
        - 确保计算在浮点域进行
        - 注意zero_point可能是整数类型
    """
    # ==================== YOUR CODE HERE (开始) ====================
    # TODO: 实现反量化操作
    
    # 步骤1: 转换为浮点类型
    dequantized = quantized.to(dtype)
    
    # 步骤2: 减去zero_point
    dequantized = dequantized - zero_point.to(dtype)
    
    # 步骤3: 乘以scale
    dequantized = dequantized * scale
    
    # ==================== YOUR CODE HERE (结束) ====================
    
    return dequantized


# ============================================================================
# 任务3: 实现完整的量化器类
# ============================================================================

class BasicQuantizer:
    """
    基础量化器类，封装量化/反量化操作
    
    学生任务:
        1. 在__init__中存储量化配置
        2. 实现calibrate方法来计算量化参数
        3. 实现quantize和dequantize方法
        4. 实现get_compression_ratio来计算压缩率
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.scale = None
        self.zero_point = None
        self.is_calibrated = False
    
    def calibrate(self, tensor: torch.Tensor) -> None:
        """
        校准量化器：计算并存储量化参数
        
        学生任务:
            1. 根据config选择对称或非对称量化
            2. 调用相应的参数计算函数
            3. 存储scale和zero_point
            4. 设置is_calibrated标志
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现校准逻辑
        
        if self.config.symmetric:
            self.scale, self.zero_point = calculate_qparams_symmetric(
                tensor,
                n_bits=self.config.n_bits,
                per_channel=self.config.per_channel,
                channel_axis=self.config.channel_axis,
            )
        else:
            self.scale, self.zero_point = calculate_qparams_asymmetric(
                tensor,
                n_bits=self.config.n_bits,
                per_channel=self.config.per_channel,
                channel_axis=self.config.channel_axis,
            )
        
        self.is_calibrated = True
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        量化张量
        
        学生任务:
            1. 检查是否已校准
            2. 调用quantize_tensor函数
            3. 返回量化结果
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现量化操作
        
        if not self.is_calibrated:
            raise RuntimeError("Quantizer not calibrated. Call calibrate() first.")
        
        # 确定dtype
        if self.config.n_bits == 8:
            dtype = torch.int8 if self.config.symmetric else torch.uint8
        else:
            dtype = torch.int32  # 通用类型
        
        quantized = quantize_tensor(
            tensor,
            self.scale,
            self.zero_point,
            self.config.quant_min,
            self.config.quant_max,
            dtype=dtype,
        )
        
        return quantized
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def dequantize(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        反量化张量
        
        学生任务:
            1. 检查是否已校准
            2. 调用dequantize_tensor函数
            3. 返回反量化结果
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现反量化操作
        
        if not self.is_calibrated:
            raise RuntimeError("Quantizer not calibrated. Call calibrate() first.")
        
        dequantized = dequantize_tensor(
            quantized,
            self.scale,
            self.zero_point,
            dtype=torch.float32,
        )
        
        return dequantized
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def quantize_dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        量化后立即反量化（用于模拟量化误差）
        
        这在QAT中非常有用
        """
        quantized = self.quantize(tensor)
        dequantized = self.dequantize(quantized)
        return dequantized
    
    def get_compression_ratio(self) -> float:
        """
        计算压缩率
        
        学生任务:
            1. 计算原始FP32需要的位数 (32 bits)
            2. 计算量化后需要的位数 (n_bits)
            3. 返回压缩率
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现压缩率计算
        
        original_bits = 32  # FP32
        quantized_bits = self.config.n_bits
        compression_ratio = original_bits / quantized_bits
        
        return compression_ratio
        
        # ==================== YOUR CODE HERE (结束) ====================


# ============================================================================
# 任务4: 实现量化误差分析工具
# ============================================================================

def compute_quantization_error(
    original: torch.Tensor,
    quantized: torch.Tensor,
    dequantized: torch.Tensor,
) -> dict:
    """
    计算量化误差的各种指标
    
    学生任务:
        1. 计算MSE (均方误差)
        2. 计算MAE (平均绝对误差)
        3. 计算SQNR (信号量化噪声比)
        4. 计算余弦相似度
        5. 计算最大误差
    
    参数:
        original: 原始浮点张量
        quantized: 量化后的整数张量
        dequantized: 反量化后的浮点张量
    
    返回:
        metrics: 包含各种误差指标的字典
    
    提示:
        - MSE = mean((original - dequantized)^2)
        - SQNR = 10 * log10(var(original) / var(original - dequantized))
        - 余弦相似度 = dot(x, y) / (||x|| * ||y||)
    """
    # ==================== YOUR CODE HERE (开始) ====================
    # TODO: 实现误差分析
    
    metrics = {}
    
    # 1. MSE (Mean Squared Error)
    mse = torch.mean((original - dequantized) ** 2).item()
    metrics['mse'] = mse
    
    # 2. MAE (Mean Absolute Error)
    mae = torch.mean(torch.abs(original - dequantized)).item()
    metrics['mae'] = mae
    
    # 3. RMSE (Root Mean Squared Error)
    rmse = torch.sqrt(torch.tensor(mse)).item()
    metrics['rmse'] = rmse
    
    # 4. SQNR (Signal to Quantization Noise Ratio)
    signal_power = torch.var(original).item()
    noise_power = torch.var(original - dequantized).item()
    if noise_power > 0:
        sqnr = 10 * np.log10(signal_power / noise_power)
    else:
        sqnr = float('inf')
    metrics['sqnr_db'] = sqnr
    
    # 5. 余弦相似度
    original_flat = original.flatten()
    dequantized_flat = dequantized.flatten()
    cosine_sim = torch.nn.functional.cosine_similarity(
        original_flat.unsqueeze(0),
        dequantized_flat.unsqueeze(0),
    ).item()
    metrics['cosine_similarity'] = cosine_sim
    
    # 6. 最大误差
    max_error = torch.max(torch.abs(original - dequantized)).item()
    metrics['max_error'] = max_error
    
    # 7. 相对误差
    relative_error = (mae / (torch.abs(original).mean().item() + 1e-8)) * 100
    metrics['relative_error_percent'] = relative_error
    
    # ==================== YOUR CODE HERE (结束) ====================
    
    return metrics


# ============================================================================
# 测试和示例代码
# ============================================================================

def test_quantization_basics():
    """
    测试量化基础功能
    """
    print("=" * 80)
    print("测试量化基础模块")
    print("=" * 80)
    
    # 创建测试张量
    torch.manual_seed(42)
    weight = torch.randn(64, 128)  # 模拟一个线性层的权重
    
    print(f"\n原始权重统计:")
    print(f"  形状: {weight.shape}")
    print(f"  均值: {weight.mean():.6f}")
    print(f"  标准差: {weight.std():.6f}")
    print(f"  范围: [{weight.min():.6f}, {weight.max():.6f}]")
    
    # 测试1: 对称量化（逐张量）
    print("\n" + "=" * 80)
    print("测试1: 对称量化（逐张量，8-bit）")
    print("=" * 80)
    config = QuantizationConfig(n_bits=8, symmetric=True, per_channel=False)
    quantizer = BasicQuantizer(config)
    quantizer.calibrate(weight)
    
    print(f"量化参数:")
    print(f"  Scale: {quantizer.scale.item():.8f}")
    print(f"  Zero-point: {quantizer.zero_point.item()}")
    
    quantized = quantizer.quantize(weight)
    dequantized = quantizer.dequantize(quantized)
    
    print(f"\n量化后统计:")
    print(f"  数据类型: {quantized.dtype}")
    print(f"  范围: [{quantized.min().item()}, {quantized.max().item()}]")
    print(f"  压缩率: {quantizer.get_compression_ratio():.2f}x")
    
    errors = compute_quantization_error(weight, quantized, dequantized)
    print(f"\n误差指标:")
    for key, value in errors.items():
        print(f"  {key}: {value:.6f}")
    
    # 测试2: 非对称量化（逐张量）
    print("\n" + "=" * 80)
    print("测试2: 非对称量化（逐张量，8-bit）")
    print("=" * 80)
    config = QuantizationConfig(n_bits=8, symmetric=False, per_channel=False)
    quantizer = BasicQuantizer(config)
    quantizer.calibrate(weight)
    
    print(f"量化参数:")
    print(f"  Scale: {quantizer.scale.item():.8f}")
    print(f"  Zero-point: {quantizer.zero_point.item()}")
    
    quantized = quantizer.quantize(weight)
    dequantized = quantizer.dequantize(quantized)
    
    errors = compute_quantization_error(weight, quantized, dequantized)
    print(f"\n误差指标:")
    for key, value in errors.items():
        print(f"  {key}: {value:.6f}")
    
    # 测试3: 逐通道量化
    print("\n" + "=" * 80)
    print("测试3: 对称量化（逐通道，8-bit）")
    print("=" * 80)
    config = QuantizationConfig(n_bits=8, symmetric=True, per_channel=True, channel_axis=0)
    quantizer = BasicQuantizer(config)
    quantizer.calibrate(weight)
    
    print(f"量化参数:")
    print(f"  Scale形状: {quantizer.scale.shape}")
    print(f"  Scale统计: mean={quantizer.scale.mean():.8f}, std={quantizer.scale.std():.8f}")
    print(f"  Zero-point: {quantizer.zero_point.unique().tolist()}")
    
    quantized = quantizer.quantize(weight)
    dequantized = quantizer.dequantize(quantized)
    
    errors = compute_quantization_error(weight, quantized, dequantized)
    print(f"\n误差指标:")
    for key, value in errors.items():
        print(f"  {key}: {value:.6f}")
    
    # 测试4: 不同位宽比较
    print("\n" + "=" * 80)
    print("测试4: 不同位宽比较")
    print("=" * 80)
    for n_bits in [2, 4, 8, 16]:
        config = QuantizationConfig(n_bits=n_bits, symmetric=True, per_channel=False)
        quantizer = BasicQuantizer(config)
        quantizer.calibrate(weight)
        
        quantized = quantizer.quantize(weight)
        dequantized = quantizer.dequantize(quantized)
        
        errors = compute_quantization_error(weight, quantized, dequantized)
        
        print(f"\n{n_bits}-bit量化:")
        print(f"  压缩率: {quantizer.get_compression_ratio():.2f}x")
        print(f"  SQNR: {errors['sqnr_db']:.2f} dB")
        print(f"  余弦相似度: {errors['cosine_similarity']:.6f}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    test_quantization_basics()
