"""
混合精度搜索模块 - Mixed Precision Search

本模块实现自动混合精度搜索，用于找到最优的逐层量化配置，包括：
1. 敏感度分析
2. 基于进化算法的搜索
3. 基于强化学习的搜索（可选）
4. 帕累托前沿优化
5. 硬件约束建模

作者: AI Research Lab
难度: ⭐⭐⭐⭐⭐ (高级)
预计完成时间: 6-8小时
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from tqdm import tqdm
import copy
import random

from .quantization_basics import QuantizationConfig, BasicQuantizer


class LayerSensitivity:
    """层敏感度分析器"""
    
    def __init__(self, model: nn.Module, eval_func: Callable):
        self.model = model
        self.eval_func = eval_func
        self.sensitivity_scores = {}
    
    def analyze(self, calibration_data, bits_options: List[int] = [4, 8]) -> Dict[str, float]:
        """
        分析每层对不同位宽的敏感度
        
        学生任务:
            1. 获取baseline性能
            2. 逐层量化并评估
            3. 计算每层的敏感度分数
        
        返回:
            layer_name -> sensitivity_score
        """
        # ==================== YOUR CODE HERE (开始) ====================
        
        baseline_metric = self.eval_func(self.model, calibration_data)
        print(f"Baseline metric: {baseline_metric:.4f}")
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # 保存原始权重
                original_weight = module.weight.data.clone()
                
                # 测试不同位宽
                sensitivities = []
                for n_bits in bits_options:
                    config = QuantizationConfig(n_bits=n_bits, symmetric=True, per_channel=True, channel_axis=0)
                    quantizer = BasicQuantizer(config)
                    quantizer.calibrate(module.weight.data)
                    
                    quantized = quantizer.quantize(module.weight.data)
                    dequantized = quantizer.dequantize(quantized)
                    
                    module.weight.data = dequantized
                    metric = self.eval_func(self.model, calibration_data)
                    
                    sensitivity = baseline_metric - metric
                    sensitivities.append(sensitivity)
                    
                    # 恢复权重
                    module.weight.data = original_weight
                
                # 取最大敏感度
                self.sensitivity_scores[name] = max(sensitivities)
                print(f"{name}: {self.sensitivity_scores[name]:.4f}")
        
        return self.sensitivity_scores
        
        # ==================== YOUR CODE HERE (结束) ====================


class MixedPrecisionConfig:
    """混合精度配置"""
    
    def __init__(self, layer_bits: Dict[str, int]):
        self.layer_bits = layer_bits
    
    def get_average_bits(self) -> float:
        """计算平均位宽"""
        return np.mean(list(self.layer_bits.values()))
    
    def get_model_size_mb(self, param_counts: Dict[str, int]) -> float:
        """计算模型大小（MB）"""
        total_bits = sum(param_counts[name] * bits for name, bits in self.layer_bits.items())
        return total_bits / 8 / 1e6
    
    def __repr__(self):
        return f"MixedPrecisionConfig(avg_bits={self.get_average_bits():.2f})"


class EvolutionarySearch:
    """
    进化算法搜索混合精度配置
    
    学生任务:
        1. 实现遗传算法的选择、交叉、变异
        2. 定义适应度函数（accuracy vs size）
        3. 找到帕累托最优解
    """
    
    def __init__(
        self,
        model: nn.Module,
        eval_func: Callable,
        layer_names: List[str],
        bits_options: List[int] = [2, 4, 6, 8],
        population_size: int = 20,
        num_generations: int = 50,
    ):
        self.model = model
        self.eval_func = eval_func
        self.layer_names = layer_names
        self.bits_options = bits_options
        self.population_size = population_size
        self.num_generations = num_generations
        
        # 获取每层参数数量
        self.param_counts = {}
        for name, module in model.named_modules():
            if name in layer_names:
                self.param_counts[name] = module.weight.numel()
    
    def create_random_config(self) -> MixedPrecisionConfig:
        """创建随机配置"""
        layer_bits = {name: random.choice(self.bits_options) for name in self.layer_names}
        return MixedPrecisionConfig(layer_bits)
    
    def evaluate_config(self, config: MixedPrecisionConfig, calibration_data) -> Tuple[float, float]:
        """
        评估一个配置
        
        学生任务:
            1. 应用配置到模型
            2. 评估accuracy
            3. 计算model size
            4. 返回(accuracy, size)
        
        返回:
            (accuracy, model_size_mb)
        """
        # ==================== YOUR CODE HERE (开始) ====================
        
        # 保存原始权重
        original_weights = {}
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                original_weights[name] = module.weight.data.clone()
        
        # 应用混合精度量化
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                n_bits = config.layer_bits[name]
                quant_config = QuantizationConfig(n_bits=n_bits, symmetric=True, per_channel=True, channel_axis=0)
                quantizer = BasicQuantizer(quant_config)
                quantizer.calibrate(module.weight.data)
                
                quantized = quantizer.quantize(module.weight.data)
                dequantized = quantizer.dequantize(quantized)
                module.weight.data = dequantized
        
        # 评估
        accuracy = self.eval_func(self.model, calibration_data)
        model_size = config.get_model_size_mb(self.param_counts)
        
        # 恢复权重
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                module.weight.data = original_weights[name]
        
        return accuracy, model_size
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def crossover(self, parent1: MixedPrecisionConfig, parent2: MixedPrecisionConfig) -> MixedPrecisionConfig:
        """
        交叉操作
        
        学生任务:
            实现单点交叉或均匀交叉
        """
        # ==================== YOUR CODE HERE (开始) ====================
        
        child_bits = {}
        for name in self.layer_names:
            # 均匀交叉：随机选择父母之一的基因
            if random.random() < 0.5:
                child_bits[name] = parent1.layer_bits[name]
            else:
                child_bits[name] = parent2.layer_bits[name]
        
        return MixedPrecisionConfig(child_bits)
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def mutate(self, config: MixedPrecisionConfig, mutation_rate: float = 0.1) -> MixedPrecisionConfig:
        """
        变异操作
        
        学生任务:
            以一定概率改变某些层的位宽
        """
        # ==================== YOUR CODE HERE (开始) ====================
        
        mutated_bits = config.layer_bits.copy()
        for name in self.layer_names:
            if random.random() < mutation_rate:
                mutated_bits[name] = random.choice(self.bits_options)
        
        return MixedPrecisionConfig(mutated_bits)
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def search(self, calibration_data) -> List[Tuple[MixedPrecisionConfig, float, float]]:
        """
        执行进化搜索
        
        学生任务:
            1. 初始化种群
            2. 迭代进化（选择、交叉、变异）
            3. 返回帕累托前沿
        
        返回:
            List of (config, accuracy, size)
        """
        # ==================== YOUR CODE HERE (开始) ====================
        
        print(f"开始进化搜索：{self.num_generations}代，种群大小={self.population_size}")
        
        # 初始化种群
        population = [self.create_random_config() for _ in range(self.population_size)]
        
        # 评估初始种群
        population_fitness = []
        for config in tqdm(population, desc="评估初始种群"):
            acc, size = self.evaluate_config(config, calibration_data)
            population_fitness.append((config, acc, size))
        
        # 进化
        for generation in range(self.num_generations):
            # 选择：基于accuracy排序
            population_fitness.sort(key=lambda x: x[1], reverse=True)
            
            # 保留top50%
            elite_size = self.population_size // 2
            elites = [item[0] for item in population_fitness[:elite_size]]
            
            # 生成新一代
            new_population = elites.copy()
            
            while len(new_population) < self.population_size:
                # 选择两个父母
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)
                
                # 交叉
                child = self.crossover(parent1, parent2)
                
                # 变异
                child = self.mutate(child, mutation_rate=0.1)
                
                new_population.append(child)
            
            # 评估新一代
            new_fitness = []
            for config in new_population:
                acc, size = self.evaluate_config(config, calibration_data)
                new_fitness.append((config, acc, size))
            
            population_fitness = new_fitness
            
            # 打印最佳配置
            best = max(population_fitness, key=lambda x: x[1])
            print(f"Generation {generation+1}: Best Acc={best[1]:.4f}, Size={best[2]:.2f}MB, Avg Bits={best[0].get_average_bits():.2f}")
        
        # 返回帕累托前沿
        pareto_front = self._compute_pareto_front(population_fitness)
        return pareto_front
        
        # ==================== YOUR CODE HERE (结束) ====================
    
    def _compute_pareto_front(self, population_fitness: List[Tuple]) -> List[Tuple]:
        """
        计算帕累托前沿
        
        学生任务:
            找到所有非支配解（accuracy越高越好，size越小越好）
        """
        # ==================== YOUR CODE HERE (开始) ====================
        
        pareto = []
        for i, (config_i, acc_i, size_i) in enumerate(population_fitness):
            dominated = False
            for j, (config_j, acc_j, size_j) in enumerate(population_fitness):
                if i != j:
                    # j支配i: j的accuracy更高且size更小（或相等）
                    if acc_j >= acc_i and size_j <= size_i and (acc_j > acc_i or size_j < size_i):
                        dominated = True
                        break
            
            if not dominated:
                pareto.append((config_i, acc_i, size_i))
        
        return pareto
        
        # ==================== YOUR CODE HERE (结束) ====================


def test_mixed_precision():
    """测试混合精度搜索"""
    print("=" * 80)
    print("测试混合精度搜索")
    print("=" * 80)
    
    # 创建简单模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    model = TestModel()
    
    # 简单评估函数
    def eval_func(model, data):
        model.eval()
        with torch.no_grad():
            outputs = model(data[0])
            return torch.randn(1).item() + 0.9  # 模拟accuracy
    
    # 模拟数据
    calibration_data = [torch.randn(16, 128)]
    
    # 进化搜索
    layer_names = ['fc1', 'fc2', 'fc3']
    searcher = EvolutionarySearch(
        model,
        eval_func,
        layer_names,
        bits_options=[4, 8],
        population_size=10,
        num_generations=5,
    )
    
    pareto_front = searcher.search(calibration_data)
    
    print(f"\n找到{len(pareto_front)}个帕累托最优解:")
    for config, acc, size in pareto_front:
        print(f"  Acc={acc:.4f}, Size={size:.2f}MB, Avg Bits={config.get_average_bits():.2f}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_mixed_precision()
