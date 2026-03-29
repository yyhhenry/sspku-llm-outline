"""
QAT - 量化感知训练模块 (Quantization-Aware Training)

本模块实现QAT技术，包括：
1. Fake Quantization（伪量化）
2. Straight-Through Estimator (STE) 梯度
3. 量化感知微调
4. BatchNorm折叠
5. 学习率调度策略

作者: AI Research Lab
难度: ⭐⭐⭐⭐ (较难)
预计完成时间: 5-6小时
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import copy

from .quantization_basics import QuantizationConfig, BasicQuantizer


# ============================================================================
# 任务1: 实现Fake Quantization with STE
# ============================================================================

class FakeQuantize(nn.Module):
    """
    伪量化层：在前向传播中模拟量化，在反向传播中使用STE
    
    学生任务:
        1. 实现forward中的量化-反量化操作
        2. 实现STE梯度（使用@staticmethod）
        3. 支持动态范围更新
        4. 实现observer模式收集统计信息
    
    关键概念:
        - Fake Quantization: 在训练时模拟量化效果
        - STE: 梯度直通估计器，让梯度可以反向传播
    """
    
    def __init__(
        self,
        config: QuantizationConfig,
        observer_enabled: bool = True,
        fake_quant_enabled: bool = True,
    ):
        super().__init__()
        self.config = config
        self.observer_enabled = observer_enabled
        self.fake_quant_enabled = fake_quant_enabled
        
        # 注册为buffer（不参与梯度更新，但会保存在state_dict）
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0, dtype=torch.int32))
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        
        self.num_batches_tracked = 0
    
    def update_stats(self, tensor: torch.Tensor) -> None:
        """
        更新统计信息（observer模式）
        
        学生任务:
            1. 计算当前batch的min/max
            2. 使用移动平均更新min_val和max_val
            3. 根据新的min/max重新计算scale和zero_point
        
        提示:
            - 使用EMA: new_val = 0.9 * old_val + 0.1 * current_val
            - 或使用running min/max
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现统计更新
        
        if not self.observer_enabled:
            return
        
        with torch.no_grad():
            # 计算当前batch的min/max
            if self.config.per_channel:
                reduce_dims = list(range(tensor.ndim))
                reduce_dims.pop(self.config.channel_axis)
                current_min = torch.amin(tensor, dim=reduce_dims, keepdim=True)
                current_max = torch.amax(tensor, dim=reduce_dims, keepdim=True)
            else:
                current_min = tensor.min()
                current_max = tensor.max()
            
            # 初始化或使用EMA更新
            if self.num_batches_tracked == 0:
                self.min_val = current_min
                self.max_val = current_max
            else:
                # 使用running min/max (更激进的策略)
                self.min_val = torch.minimum(self.min_val, current_min)
                self.max_val = torch.maximum(self.max_val, current_max)
            
            # 重新计算量化参数
            if self.config.symmetric:
                abs_max = torch.maximum(torch.abs(self.min_val), torch.abs(self.max_val))
                qmax = 2 ** (self.config.n_bits - 1) - 1
                self.scale = abs_max / qmax
                self.scale = torch.where(self.scale > 0, self.scale, torch.ones_like(self.scale))
                self.zero_point = torch.zeros_like(self.scale, dtype=torch.int32)
            else:
                quant_range = self.config.quant_max - self.config.quant_min
                self.scale = (self.max_val - self.min_val) / quant_range
                self.scale = torch.where(self.scale > 0, self.scale, torch.ones_like(self.scale))
                self.zero_point = self.config.quant_min - torch.round(self.min_val / self.scale)
                self.zero_point = torch.clamp(
                    self.zero_point, 
                    self.config.quant_min, 
                    self.config.quant_max
                ).to(torch.int32)
            
            self.num_batches_tracked += 1
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：伪量化
        
        学生任务:
            1. 如果observer_enabled，更新统计信息
            2. 如果fake_quant_enabled，应用量化-反量化
            3. 使用自定义autograd函数实现STE
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现前向传播
        
        if self.observer_enabled:
            self.update_stats(x)
        
        if self.fake_quant_enabled:
            # 使用STE进行伪量化
            x = FakeQuantizeFunction.apply(
                x,
                self.scale,
                self.zero_point,
                self.config.quant_min,
                self.config.quant_max,
            )
        
        return x
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def extra_repr(self) -> str:
        return (f'n_bits={self.config.n_bits}, symmetric={self.config.symmetric}, '
                f'observer={self.observer_enabled}, fake_quant={self.fake_quant_enabled}')


class FakeQuantizeFunction(torch.autograd.Function):
    """
    自定义autograd函数实现STE (Straight-Through Estimator)
    
    学生任务:
        1. 在forward中实现量化-反量化
        2. 在backward中实现STE梯度（直通）
        3. 处理超出量化范围的梯度截断
    
    关键概念:
        STE允许梯度"穿过"不可微的量化操作
    """
    
    @staticmethod
    def forward(ctx, x, scale, zero_point, quant_min, quant_max):
        """
        前向传播：量化-反量化
        
        学生任务:
            1. 将x量化为整数
            2. 立即反量化回浮点
            3. 保存mask用于梯度裁剪
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现伪量化前向传播
        
        # 量化
        x_scaled = x / scale + zero_point
        x_quantized = torch.clamp(torch.round(x_scaled), quant_min, quant_max)
        
        # 反量化
        x_dequantized = (x_quantized - zero_point) * scale
        
        # 保存mask：标记哪些值被clamp了（用于梯度截断）
        mask = (x_scaled >= quant_min) & (x_scaled <= quant_max)
        ctx.save_for_backward(mask)
        
        return x_dequantized
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：STE梯度
        
        学生任务:
            1. 对于在量化范围内的值，梯度直通
            2. 对于超出范围的值，梯度置0（可选）
            3. 返回与forward输入对应的梯度
        
        提示:
            - STE: dy/dx ≈ 1（直通）
            - 或者使用mask进行选择性传播
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现STE梯度
        
        mask, = ctx.saved_tensors
        
        # STE: 梯度直通，但对超出范围的值截断梯度
        grad_input = grad_output.clone()
        grad_input = torch.where(mask, grad_input, torch.zeros_like(grad_input))
        
        # 返回与forward参数对应的梯度
        # (x, scale, zero_point, quant_min, quant_max)
        return grad_input, None, None, None, None
        
        # ==================== YOUR CODE HERE (结束) ====================


# ============================================================================
# 任务2: 实现量化感知层
# ============================================================================

class QuantizedLinear(nn.Module):
    """
    量化感知的全连接层
    
    学生任务:
        1. 为权重和激活添加FakeQuantize
        2. 实现forward传播
        3. 支持freeze量化参数
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_qconfig: Optional[QuantizationConfig] = None,
        activation_qconfig: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        
        # 原始层
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # 权重量化（默认8-bit对称逐通道）
        if weight_qconfig is None:
            weight_qconfig = QuantizationConfig(
                n_bits=8, symmetric=True, per_channel=True, channel_axis=0
            )
        self.weight_fake_quant = FakeQuantize(weight_qconfig)
        
        # 激活量化（默认8-bit非对称逐张量）
        if activation_qconfig is None:
            activation_qconfig = QuantizationConfig(
                n_bits=8, symmetric=False, per_channel=False
            )
        self.activation_fake_quant = FakeQuantize(activation_qconfig)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        学生任务:
            1. 对权重应用伪量化
            2. 执行线性运算
            3. 对输出应用伪量化
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现量化感知的前向传播
        
        # 量化权重
        quantized_weight = self.weight_fake_quant(self.linear.weight)
        
        # 使用量化权重进行计算
        output = F.linear(x, quantized_weight, self.linear.bias)
        
        # 量化激活
        output = self.activation_fake_quant(output)
        
        return output
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def freeze_quantization(self):
        """冻结量化参数（用于微调后期）"""
        self.weight_fake_quant.observer_enabled = False
        self.activation_fake_quant.observer_enabled = False


class QuantizedConv2d(nn.Module):
    """
    量化感知的卷积层
    
    学生任务:
        类似QuantizedLinear，实现量化感知的卷积
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        weight_qconfig: Optional[QuantizationConfig] = None,
        activation_qconfig: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=bias
        )
        
        if weight_qconfig is None:
            weight_qconfig = QuantizationConfig(
                n_bits=8, symmetric=True, per_channel=True, channel_axis=0
            )
        self.weight_fake_quant = FakeQuantize(weight_qconfig)
        
        if activation_qconfig is None:
            activation_qconfig = QuantizationConfig(
                n_bits=8, symmetric=False, per_channel=False
            )
        self.activation_fake_quant = FakeQuantize(activation_qconfig)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现量化感知的卷积
        
        quantized_weight = self.weight_fake_quant(self.conv.weight)
        output = F.conv2d(
            x, quantized_weight, self.conv.bias,
            stride=self.conv.stride, padding=self.conv.padding
        )
        output = self.activation_fake_quant(output)
        
        return output
        
        # ==================== YOUR CODE HERE (结束) ====================


# ============================================================================
# 任务3: 实现BatchNorm折叠
# ============================================================================

def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    将BatchNorm折叠到卷积层中
    
    学生任务:
        1. 计算折叠后的权重和偏置
        2. 创建新的Conv层
        3. 返回融合后的层
    
    公式:
        w_fused = w_conv * (gamma / sqrt(var + eps))
        b_fused = beta + (b_conv - mu) * (gamma / sqrt(var + eps))
    
    其中:
        gamma, beta: BN的缩放和偏移参数
        mu, var: BN的运行均值和方差
    """
    # ==================== YOUR CODE HERE (开始) ====================
    # TODO: 实现BN折叠
    
    # 获取BN参数
    gamma = bn.weight
    beta = bn.bias
    mu = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    
    # 计算折叠系数
    std = torch.sqrt(var + eps)
    factor = gamma / std
    
    # 创建新的Conv层
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,  # 确保有bias
    )
    
    # 折叠权重
    # w_fused = w_conv * factor (broadcast到所有维度)
    fused_conv.weight.data = conv.weight * factor.reshape(-1, 1, 1, 1)
    
    # 折叠偏置
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.out_channels, device=conv.weight.device)
    
    fused_conv.bias.data = beta + (b_conv - mu) * factor
    
    return fused_conv
    
    # ==================== YOUR CODE HERE (结束) ====================


def fuse_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    """
    将BatchNorm折叠到Linear层中
    
    学生任务:
        实现Linear + BN的折叠（类似Conv）
    """
    # ==================== YOUR CODE HERE (开始) ====================
    # TODO: 实现Linear+BN折叠
    
    gamma = bn.weight
    beta = bn.bias
    mu = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    
    std = torch.sqrt(var + eps)
    factor = gamma / std
    
    fused_linear = nn.Linear(
        linear.in_features,
        linear.out_features,
        bias=True,
    )
    
    fused_linear.weight.data = linear.weight * factor.reshape(-1, 1)
    
    if linear.bias is not None:
        b_linear = linear.bias
    else:
        b_linear = torch.zeros(linear.out_features, device=linear.weight.device)
    
    fused_linear.bias.data = beta + (b_linear - mu) * factor
    
    return fused_linear
    
    # ==================== YOUR CODE HERE (结束) ====================


# ============================================================================
# 任务4: 实现QAT训练器
# ============================================================================

class QATTrainer:
    """
    量化感知训练器
    
    学生任务:
        1. 实现训练循环
        2. 实现分阶段训练策略
        3. 实现量化参数的冻结策略
        4. 支持BN折叠
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.freeze_bn_delay = 0  # 多少个epoch后冻结BN统计
        self.freeze_quant_delay = float('inf')  # 多少个epoch后冻结量化参数
    
    def set_freeze_delays(self, freeze_bn_delay: int, freeze_quant_delay: int):
        """设置冻结策略"""
        self.freeze_bn_delay = freeze_bn_delay
        self.freeze_quant_delay = freeze_quant_delay
    
    def train_epoch(self, train_loader, epoch: int) -> float:
        """
        训练一个epoch
        
        学生任务:
            1. 实现标准训练循环
            2. 根据epoch决定是否冻结BN
            3. 根据epoch决定是否冻结量化参数
            4. 返回平均损失
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现QAT训练循环
        
        self.model.train()
        
        # 冻结BN统计
        if epoch >= self.freeze_bn_delay:
            for module in self.model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.eval()
        
        # 冻结量化参数
        if epoch >= self.freeze_quant_delay:
            for module in self.model.modules():
                if isinstance(module, FakeQuantize):
                    module.observer_enabled = False
        
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch
            else:
                inputs = batch
                targets = None
            
            inputs = inputs.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)
            
            # 前向传播
            outputs = self.model(inputs)
            
            if targets is not None:
                loss = self.criterion(outputs, targets)
            else:
                loss = self.criterion(outputs)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def evaluate(self, val_loader) -> float:
        """
        评估模型
        
        学生任务:
            实现验证循环
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现评估
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch
                    targets = None
                
                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                
                if targets is not None:
                    loss = self.criterion(outputs, targets)
                    total_loss += loss.item()
                    num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
        
        # ==================== YOUR CODE HERE (结束) ====================


# ============================================================================
# 测试代码
# ============================================================================

def test_qat():
    """测试QAT功能"""
    print("=" * 80)
    print("测试QAT模块")
    print("=" * 80)
    
    # 测试1: FakeQuantize
    print("\n测试1: FakeQuantize")
    print("-" * 80)
    
    config = QuantizationConfig(n_bits=8, symmetric=True)
    fake_quant = FakeQuantize(config)
    
    x = torch.randn(32, 128, requires_grad=True)
    y = fake_quant(x)
    
    print(f"输入范围: [{x.min():.4f}, {x.max():.4f}]")
    print(f"输出范围: [{y.min():.4f}, {y.max():.4f}]")
    print(f"Scale: {fake_quant.scale.item():.6f}")
    
    # 测试梯度
    loss = y.sum()
    loss.backward()
    print(f"梯度存在: {x.grad is not None}")
    print(f"梯度范围: [{x.grad.min():.4f}, {x.grad.max():.4f}]")
    
    # 测试2: QuantizedLinear
    print("\n测试2: QuantizedLinear")
    print("-" * 80)
    
    q_linear = QuantizedLinear(128, 64)
    x = torch.randn(32, 128)
    y = q_linear(x)
    
    print(f"输出形状: {y.shape}")
    print(f"权重已量化: {q_linear.weight_fake_quant.num_batches_tracked > 0}")
    print(f"激活已量化: {q_linear.activation_fake_quant.num_batches_tracked > 0}")
    
    # 测试3: BN折叠
    print("\n测试3: BatchNorm折叠")
    print("-" * 80)
    
    conv = nn.Conv2d(3, 64, 3, padding=1, bias=True)
    bn = nn.BatchNorm2d(64)
    
    # 模拟训练后的BN
    bn.running_mean = torch.randn(64)
    bn.running_var = torch.rand(64) + 0.1
    
    fused_conv = fuse_conv_bn(conv, bn)
    
    # 验证输出一致性
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y1 = bn(conv(x))
        y2 = fused_conv(x)
    
    diff = (y1 - y2).abs().max().item()
    print(f"折叠前后最大差异: {diff:.6f}")
    print(f"折叠成功: {diff < 1e-5}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    test_qat()
