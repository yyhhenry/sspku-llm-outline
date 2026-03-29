"""
PTQ - 训练后静态量化模块 (Post-Training Quantization)

本模块实现静态量化技术，包括：
1. MinMax校准
2. Percentile校准
3. MSE校准
4. Entropy校准（KL散度）
5. 逐层敏感度分析

作者: AI Research Lab
难度: ⭐⭐⭐ (中等)
预计完成时间: 4-5小时
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from tqdm import tqdm
import copy

from .quantization_basics import (
    QuantizationConfig,
    BasicQuantizer,
    quantize_tensor,
    dequantize_tensor,
    compute_quantization_error,
)


# ============================================================================
# 任务1: 实现不同的校准方法
# ============================================================================

class CalibrationMethod:
    """校准方法基类"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.num_batches = 0
        self.activations = []
    
    def collect_stats(self, tensor: torch.Tensor) -> None:
        """收集激活值统计信息"""
        self.activations.append(tensor.detach().cpu())
        self.num_batches += 1
    
    def compute_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算量化参数（需要子类实现）"""
        raise NotImplementedError


class MinMaxCalibration(CalibrationMethod):
    """
    MinMax校准：使用观察到的最小值和最大值
    
    学生任务:
        1. 在collect_stats中收集每个batch的min/max
        2. 在compute_qparams中使用全局min/max计算量化参数
        3. 考虑per_channel的情况
    
    优点: 简单，保留所有数值
    缺点: 对离群值敏感
    """
    
    def __init__(self, config: QuantizationConfig):
        super().__init__(config)
        self.running_min = None
        self.running_max = None
    
    def collect_stats(self, tensor: torch.Tensor) -> None:
        """
        收集统计信息
        
        学生任务:
            1. 计算当前batch的min和max
            2. 更新running_min和running_max
            3. 处理per_channel的情况
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现MinMax统计收集
        
        tensor = tensor.detach()
        
        if self.config.per_channel:
            # 逐通道：在除channel_axis外的维度上计算
            reduce_dims = list(range(tensor.ndim))
            reduce_dims.pop(self.config.channel_axis)
            
            current_min = torch.amin(tensor, dim=reduce_dims, keepdim=True)
            current_max = torch.amax(tensor, dim=reduce_dims, keepdim=True)
        else:
            # 逐张量：全局min/max
            current_min = tensor.min()
            current_max = tensor.max()
        
        # 更新running统计
        if self.running_min is None:
            self.running_min = current_min
            self.running_max = current_max
        else:
            self.running_min = torch.minimum(self.running_min, current_min)
            self.running_max = torch.maximum(self.running_max, current_max)
        
        self.num_batches += 1
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def compute_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算量化参数
        
        学生任务:
            1. 使用running_min和running_max计算scale和zero_point
            2. 根据对称/非对称量化选择不同的公式
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现量化参数计算
        
        if self.running_min is None:
            raise RuntimeError("No statistics collected. Call collect_stats first.")
        
        min_val = self.running_min
        max_val = self.running_max
        
        if self.config.symmetric:
            # 对称量化
            abs_max = torch.maximum(torch.abs(min_val), torch.abs(max_val))
            qmax = 2 ** (self.config.n_bits - 1) - 1
            scale = abs_max / qmax
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            zero_point = torch.zeros_like(scale, dtype=torch.int32)
        else:
            # 非对称量化
            quant_min = self.config.quant_min
            quant_max = self.config.quant_max
            
            scale = (max_val - min_val) / (quant_max - quant_min)
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            zero_point = quant_min - torch.round(min_val / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max).to(torch.int32)
        
        return scale, zero_point
        
        # ==================== YOUR CODE HERE (结束) ====================


class PercentileCalibration(CalibrationMethod):
    """
    Percentile校准：使用百分位数来忽略离群值
    
    学生任务:
        1. 收集所有激活值
        2. 计算指定百分位数（如99.9%）作为min/max
        3. 用于计算量化参数
    
    优点: 对离群值鲁棒
    缺点: 可能截断有效数据
    """
    
    def __init__(self, config: QuantizationConfig, percentile: float = 99.99):
        super().__init__(config)
        self.percentile = percentile
    
    def compute_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用百分位数计算量化参数
        
        学生任务:
            1. 将收集的所有激活值拼接
            2. 计算lower和upper百分位数
            3. 使用这些值计算量化参数
        
        提示:
            - 使用torch.quantile或torch.kthvalue
            - lower_percentile = (100 - percentile) / 2
            - upper_percentile = percentile + (100 - percentile) / 2
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现Percentile校准
        
        if len(self.activations) == 0:
            raise RuntimeError("No statistics collected.")
        
        # 拼接所有激活值
        all_activations = torch.cat([act.flatten() for act in self.activations])
        
        # 计算百分位数
        lower_percentile = (100 - self.percentile) / 2
        upper_percentile = self.percentile + (100 - self.percentile) / 2
        
        min_val = torch.quantile(all_activations, lower_percentile / 100)
        max_val = torch.quantile(all_activations, upper_percentile / 100)
        
        # 确保维度正确
        if self.config.per_channel:
            # 对于per_channel，需要重新计算
            # 这里简化处理，实际应该逐通道计算百分位数
            pass
        
        # 计算量化参数
        if self.config.symmetric:
            abs_max = max(abs(min_val.item()), abs(max_val.item()))
            qmax = 2 ** (self.config.n_bits - 1) - 1
            scale = torch.tensor(abs_max / qmax)
            zero_point = torch.tensor(0, dtype=torch.int32)
        else:
            quant_min = self.config.quant_min
            quant_max = self.config.quant_max
            
            scale = (max_val - min_val) / (quant_max - quant_min)
            zero_point = quant_min - torch.round(min_val / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max).to(torch.int32)
        
        return scale, zero_point
        
        # ==================== YOUR CODE HERE (结束) ====================


class MSECalibration(CalibrationMethod):
    """
    MSE校准：搜索使量化误差最小的scale和zero_point
    
    学生任务:
        1. 在一个范围内搜索最优的量化参数
        2. 对每组参数，计算量化后的MSE
        3. 选择MSE最小的参数
    
    优点: 最小化量化误差
    缺点: 计算开销大
    """
    
    def __init__(self, config: QuantizationConfig, num_candidates: int = 100):
        super().__init__(config)
        self.num_candidates = num_candidates
    
    def compute_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        通过MSE搜索最优量化参数
        
        学生任务:
            1. 首先用MinMax方法获得初始范围
            2. 在这个范围附近搜索候选参数
            3. 对每个候选参数计算MSE
            4. 返回MSE最小的参数
        
        提示:
            - 可以在[0.9*scale, 1.1*scale]范围内搜索
            - 使用linspace生成候选值
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现MSE校准
        
        if len(self.activations) == 0:
            raise RuntimeError("No statistics collected.")
        
        # 拼接所有激活值
        all_activations = torch.cat([act.flatten() for act in self.activations])
        
        # 使用MinMax作为初始估计
        min_val = all_activations.min()
        max_val = all_activations.max()
        
        best_mse = float('inf')
        best_scale = None
        best_zero_point = None
        
        # 生成候选scale值
        if self.config.symmetric:
            abs_max = max(abs(min_val.item()), abs(max_val.item()))
            qmax = 2 ** (self.config.n_bits - 1) - 1
            base_scale = abs_max / qmax
            
            # 在base_scale附近搜索
            scale_candidates = torch.linspace(
                base_scale * 0.8, base_scale * 1.2, self.num_candidates
            )
            
            for scale in scale_candidates:
                zero_point = torch.tensor(0, dtype=torch.int32)
                
                # 量化并反量化
                quantized = quantize_tensor(
                    all_activations, scale, zero_point,
                    self.config.quant_min, self.config.quant_max
                )
                dequantized = dequantize_tensor(quantized, scale, zero_point)
                
                # 计算MSE
                mse = torch.mean((all_activations - dequantized) ** 2).item()
                
                if mse < best_mse:
                    best_mse = mse
                    best_scale = scale
                    best_zero_point = zero_point
        else:
            # 非对称量化的搜索逻辑
            quant_min = self.config.quant_min
            quant_max = self.config.quant_max
            
            base_scale = (max_val - min_val) / (quant_max - quant_min)
            scale_candidates = torch.linspace(
                base_scale * 0.8, base_scale * 1.2, self.num_candidates
            )
            
            for scale in scale_candidates:
                zero_point = quant_min - torch.round(min_val / scale)
                zero_point = torch.clamp(zero_point, quant_min, quant_max).to(torch.int32)
                
                quantized = quantize_tensor(
                    all_activations, scale, zero_point,
                    quant_min, quant_max
                )
                dequantized = dequantize_tensor(quantized, scale, zero_point)
                
                mse = torch.mean((all_activations - dequantized) ** 2).item()
                
                if mse < best_mse:
                    best_mse = mse
                    best_scale = scale
                    best_zero_point = zero_point
        
        return best_scale, best_zero_point
        
        # ==================== YOUR CODE HERE (结束) ====================


# ============================================================================
# 任务2: 实现PTQ量化器
# ============================================================================

class PTQQuantizer:
    """
    训练后量化器
    
    学生任务:
        1. 对模型的每一层应用量化
        2. 使用校准数据集确定量化参数
        3. 支持跳过某些层（如第一层和最后一层）
        4. 实现敏感度分析
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        calibration_method: str = 'minmax',
    ):
        self.model = model
        self.config = config
        self.calibration_method = calibration_method
        
        # 存储每层的量化器
        self.quantizers = {}
        self.activation_calibrators = {}
        
    def prepare_calibration(self) -> None:
        """
        准备校准：为每层创建校准器
        
        学生任务:
            1. 遍历模型的所有层
            2. 为权重和激活创建量化器/校准器
            3. 注册forward hook来收集激活值
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现校准准备
        
        def get_calibrator(config):
            """根据配置创建校准器"""
            if self.calibration_method == 'minmax':
                return MinMaxCalibration(config)
            elif self.calibration_method == 'percentile':
                return PercentileCalibration(config)
            elif self.calibration_method == 'mse':
                return MSECalibration(config)
            else:
                raise ValueError(f"Unknown calibration method: {self.calibration_method}")
        
        # 为每个Linear和Conv层准备量化器
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # 权重量化器（逐通道）
                weight_config = QuantizationConfig(
                    n_bits=self.config.n_bits,
                    symmetric=True,
                    per_channel=True,
                    channel_axis=0,  # output channel
                )
                weight_quantizer = BasicQuantizer(weight_config)
                
                # 使用权重进行校准
                weight_quantizer.calibrate(module.weight.data)
                self.quantizers[f"{name}.weight"] = weight_quantizer
                
                # 激活值校准器（逐张量）
                act_config = QuantizationConfig(
                    n_bits=self.config.n_bits,
                    symmetric=False,
                    per_channel=False,
                )
                act_calibrator = get_calibrator(act_config)
                self.activation_calibrators[name] = act_calibrator
                
                # 注册hook收集激活值
                def make_hook(calibrator):
                    def hook(module, input, output):
                        calibrator.collect_stats(output)
                    return hook
                
                module.register_forward_hook(make_hook(act_calibrator))
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def calibrate(self, calibration_loader, num_batches: int = 100) -> None:
        """
        使用校准数据集进行校准
        
        学生任务:
            1. 将模型设置为eval模式
            2. 运行前向传播收集统计信息
            3. 计算所有层的量化参数
        
        参数:
            calibration_loader: 校准数据加载器
            num_batches: 使用多少个batch进行校准
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现校准过程
        
        self.model.eval()
        
        print(f"开始校准，使用{num_batches}个batch...")
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(calibration_loader, total=num_batches)):
                if i >= num_batches:
                    break
                
                # 前向传播（hooks会自动收集激活值）
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                
                if isinstance(inputs, dict):
                    self.model(**inputs)
                else:
                    self.model(inputs)
        
        # 计算所有激活值的量化参数
        print("计算量化参数...")
        for name, calibrator in self.activation_calibrators.items():
            scale, zero_point = calibrator.compute_qparams()
            
            # 创建量化器并存储
            quantizer = BasicQuantizer(calibrator.config)
            quantizer.scale = scale
            quantizer.zero_point = zero_point
            quantizer.is_calibrated = True
            
            self.quantizers[f"{name}.output"] = quantizer
        
        print("校准完成！")
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def quantize_model(self) -> nn.Module:
        """
        量化整个模型
        
        学生任务:
            1. 创建模型的副本
            2. 将每层的权重量化
            3. 返回量化后的模型
        
        注意: 这里实现的是"量化-反量化"模拟，用于评估量化误差
        """
        # ==================== YOUR CODE HERE (开始) ====================
        # TODO: 实现模型量化
        
        quantized_model = copy.deepcopy(self.model)
        
        for name, module in quantized_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight_key = f"{name}.weight"
                if weight_key in self.quantizers:
                    quantizer = self.quantizers[weight_key]
                    
                    # 量化权重
                    quantized_weight = quantizer.quantize(module.weight.data)
                    dequantized_weight = quantizer.dequantize(quantized_weight)
                    
                    # 更新权重
                    module.weight.data = dequantized_weight
        
        return quantized_model
        
        # ==================== YOUR CODE HERE (结束) ====================


# ============================================================================
# 任务3: 实现敏感度分析
# ============================================================================

def layer_sensitivity_analysis(
    model: nn.Module,
    calibration_loader,
    eval_func: Callable,
    config: QuantizationConfig,
) -> Dict[str, float]:
    """
    分析每层对量化的敏感度
    
    学生任务:
        1. 逐层量化模型
        2. 对每层量化后评估性能下降
        3. 返回每层的敏感度分数
    
    参数:
        model: 原始模型
        calibration_loader: 校准数据
        eval_func: 评估函数，返回accuracy/loss等指标
        config: 量化配置
    
    返回:
        sensitivity: 每层的敏感度分数（性能下降百分比）
    
    提示:
        - 敏感度高的层应该使用更高精度
        - 可以用于混合精度量化
    """
    # ==================== YOUR CODE HERE (开始) ====================
    # TODO: 实现敏感度分析
    
    sensitivity = {}
    
    # 获取baseline性能
    print("评估原始模型...")
    baseline_metric = eval_func(model, calibration_loader)
    print(f"Baseline metric: {baseline_metric:.4f}")
    
    # 逐层量化并评估
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            print(f"\n分析层: {name}")
            
            # 保存原始权重
            original_weight = module.weight.data.clone()
            
            # 量化这一层
            quantizer = BasicQuantizer(config)
            quantizer.calibrate(module.weight.data)
            quantized = quantizer.quantize(module.weight.data)
            dequantized = quantizer.dequantize(quantized)
            
            # 替换权重
            module.weight.data = dequantized
            
            # 评估性能
            quantized_metric = eval_func(model, calibration_loader)
            
            # 计算性能下降
            metric_drop = baseline_metric - quantized_metric
            sensitivity[name] = metric_drop
            
            print(f"  量化后metric: {quantized_metric:.4f}")
            print(f"  性能下降: {metric_drop:.4f}")
            
            # 恢复原始权重
            module.weight.data = original_weight
    
    return sensitivity
    
    # ==================== YOUR CODE HERE (结束) ====================


# ============================================================================
# 测试代码
# ============================================================================

def test_ptq():
    """测试PTQ功能"""
    print("=" * 80)
    print("测试PTQ模块")
    print("=" * 80)
    
    # 创建一个简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(256, 10)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    
    # 创建模拟校准数据
    calibration_data = [torch.randn(32, 128) for _ in range(10)]
    
    # 测试MinMax校准
    print("\n测试1: MinMax校准")
    print("-" * 80)
    config = QuantizationConfig(n_bits=8, symmetric=True)
    ptq = PTQQuantizer(model, config, calibration_method='minmax')
    ptq.prepare_calibration()
    ptq.calibrate(calibration_data, num_batches=10)
    
    print(f"量化器数量: {len(ptq.quantizers)}")
    
    # 量化模型
    quantized_model = ptq.quantize_model()
    
    # 比较输出
    test_input = torch.randn(1, 128)
    with torch.no_grad():
        original_output = model(test_input)
        quantized_output = quantized_model(test_input)
    
    error = torch.mean((original_output - quantized_output) ** 2).item()
    print(f"输出MSE: {error:.6f}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    test_ptq()
