"""
性能基准测试模块

提供量化模型的完整性能评估
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Callable, Optional
from tqdm import tqdm
import psutil
import os


class QuantizationBenchmark:
    """量化基准测试器"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.results = {}
    
    def measure_inference_time(
        self,
        model: nn.Module,
        input_shape: tuple,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """
        测量推理时间
        
        返回:
            {'mean': float, 'std': float, 'median': float}
        """
        model.eval()
        model = model.to(self.device)
        
        # 准备输入
        dummy_input = torch.randn(input_shape, device=self.device)
        
        # 预热
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)
        
        # 同步（如果使用GPU）
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # 测量
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(dummy_input)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'median_ms': np.median(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
        }
    
    def measure_memory_usage(self, model: nn.Module) -> Dict[str, float]:
        """
        测量模型内存占用
        
        返回:
            {'model_size_mb': float, 'peak_memory_mb': float}
        """
        # 模型参数大小
        param_size = 0
        for param in model.parameters():
            param_size += param.numel() * param.element_size()
        
        model_size_mb = param_size / (1024 ** 2)
        
        # 峰值内存（仅GPU）
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            model = model.to(self.device)
            dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_memory_mb = 0
        
        return {
            'model_size_mb': model_size_mb,
            'peak_memory_mb': peak_memory_mb,
        }
    
    def measure_accuracy(
        self,
        model: nn.Module,
        dataloader,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """
        测量模型准确率
        
        返回:
            {'accuracy': float, 'loss': float}
        """
        model.eval()
        model = model.to(self.device)
        
        correct = 0
        total = 0
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估"):
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch
                    targets = None
                
                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                
                outputs = model(inputs)
                
                if targets is not None:
                    # 计算准确率
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                    # 计算损失
                    if criterion is not None:
                        loss = criterion(outputs, targets)
                        total_loss += loss.item()
                        num_batches += 1
        
        accuracy = 100 * correct / total if total > 0 else 0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
        }
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        input_shape: tuple,
        dataloader=None,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        比较多个模型
        
        参数:
            models: {name: model}
            input_shape: 输入形状
            dataloader: 数据加载器（用于准确率评估）
            criterion: 损失函数
        
        返回:
            {model_name: {metric_name: value}}
        """
        results = {}
        
        for name, model in models.items():
            print(f"\n评估模型: {name}")
            print("=" * 60)
            
            result = {}
            
            # 推理时间
            print("测量推理时间...")
            time_stats = self.measure_inference_time(model, input_shape)
            result.update(time_stats)
            
            # 内存占用
            print("测量内存占用...")
            memory_stats = self.measure_memory_usage(model)
            result.update(memory_stats)
            
            # 准确率（如果提供了数据）
            if dataloader is not None:
                print("测量准确率...")
                acc_stats = self.measure_accuracy(model, dataloader, criterion)
                result.update(acc_stats)
            
            results[name] = result
            
            # 打印结果
            print(f"\n结果:")
            for metric, value in result.items():
                print(f"  {metric}: {value:.4f}")
        
        return results
    
    def print_summary_table(self, results: Dict[str, Dict[str, float]]):
        """打印结果摘要表"""
        print("\n" + "=" * 100)
        print("基准测试结果摘要")
        print("=" * 100)
        
        # 表头
        metrics = list(next(iter(results.values())).keys())
        header = f"{'模型':<20}"
        for metric in metrics:
            header += f"{metric:<20}"
        print(header)
        print("-" * 100)
        
        # 数据行
        for name, values in results.items():
            row = f"{name:<20}"
            for metric in metrics:
                value = values.get(metric, 0)
                row += f"{value:<20.4f}"
            print(row)
        
        print("=" * 100)


def run_comprehensive_benchmark():
    """运行综合基准测试"""
    print("=" * 80)
    print("大模型量化综合基准测试")
    print("=" * 80)
    
    # 创建测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 56 * 56, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 10),
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # 创建模型
    fp32_model = TestModel()
    
    # 这里应该加载实际的量化模型
    # 为演示目的，我们使用相同的模型
    models = {
        'FP32': fp32_model,
        '8-bit量化': fp32_model,  # 实际应该是量化模型
        '4-bit量化': fp32_model,  # 实际应该是量化模型
    }
    
    # 运行基准测试
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    benchmark = QuantizationBenchmark(device=device)
    
    results = benchmark.compare_models(
        models,
        input_shape=(1, 3, 224, 224),
    )
    
    # 打印摘要
    benchmark.print_summary_table(results)
    
    return results


if __name__ == "__main__":
    run_comprehensive_benchmark()
