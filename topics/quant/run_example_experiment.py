#!/usr/bin/env python3
"""
完整实验示例 - 在MNIST上演示量化全流程

本脚本展示如何：
1. 训练一个基础模型
2. 应用PTQ
3. 应用QAT
4. 比较结果

运行: python run_example_experiment.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from core.quantization_basics import QuantizationConfig, BasicQuantizer, compute_quantization_error
from core.ptq_static import PTQQuantizer
from core.qat_training import QuantizedLinear, QATTrainer
from visualization import plot_quantization_comparison, plot_training_curves, create_comparison_table
from benchmark import QuantizationBenchmark

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# ============================================================================
# 1. 定义模型
# ============================================================================

class SimpleNet(nn.Module):
    """简单的MNIST分类器"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================================
# 2. 准备数据
# ============================================================================

def get_data_loaders(batch_size=128):
    """获取MNIST数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 校准数据（使用训练集的一个子集）
    calibration_dataset = torch.utils.data.Subset(train_dataset, range(1000))
    calibration_loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, calibration_loader


# ============================================================================
# 3. 训练和评估函数
# ============================================================================

def train_model(model, train_loader, epochs=5, lr=0.001):
    """训练模型"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc:.2f}%')
    
    return model


def evaluate_model(model, test_loader):
    """评估模型"""
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


# ============================================================================
# 4. 主实验流程
# ============================================================================

def main():
    print("=" * 80)
    print("大模型量化完整实验示例")
    print("=" * 80)
    
    # 准备数据
    print("\n1. 准备数据...")
    train_loader, test_loader, calibration_loader = get_data_loaders()
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    print(f"校准集: {len(calibration_loader.dataset)} 样本")
    
    # 训练基础模型
    print("\n2. 训练FP32基础模型...")
    fp32_model = SimpleNet()
    fp32_model = train_model(fp32_model, train_loader, epochs=3)
    
    # 评估FP32模型
    fp32_acc = evaluate_model(fp32_model, test_loader)
    print(f"\nFP32模型准确率: {fp32_acc:.2f}%")
    
    # 保存模型
    torch.save(fp32_model.state_dict(), 'fp32_model.pth')
    
    # ========================================================================
    # PTQ实验
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. PTQ (训练后量化) 实验")
    print("=" * 80)
    
    # 加载模型
    ptq_model = SimpleNet()
    ptq_model.load_state_dict(torch.load('fp32_model.pth'))
    
    # 应用PTQ
    config = QuantizationConfig(n_bits=8, symmetric=True, per_channel=True)
    ptq_quantizer = PTQQuantizer(ptq_model, config, calibration_method='minmax')
    ptq_quantizer.prepare_calibration()
    
    print("校准PTQ模型...")
    ptq_quantizer.calibrate(calibration_loader, num_batches=10)
    
    ptq_model_quantized = ptq_quantizer.quantize_model()
    
    # 评估PTQ模型
    ptq_acc = evaluate_model(ptq_model_quantized, test_loader)
    print(f"\nPTQ模型准确率: {ptq_acc:.2f}%")
    print(f"精度下降: {fp32_acc - ptq_acc:.2f}%")
    
    # ========================================================================
    # 权重分析
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. 量化效果分析")
    print("=" * 80)
    
    # 分析第一层权重
    original_weight = fp32_model.fc1.weight.data
    quantized_weight = ptq_model_quantized.fc1.weight.data
    
    errors = compute_quantization_error(original_weight, 
                                       quantized_weight, 
                                       quantized_weight)
    
    print(f"\n第一层权重量化误差:")
    for metric, value in errors.items():
        print(f"  {metric}: {value:.6f}")
    
    # ========================================================================
    # Benchmark
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. 性能基准测试")
    print("=" * 80)
    
    benchmark = QuantizationBenchmark(device=device)
    
    models = {
        'FP32': fp32_model,
        'PTQ-8bit': ptq_model_quantized,
    }
    
    results = benchmark.compare_models(
        models,
        input_shape=(1, 1, 28, 28),
        dataloader=test_loader,
        criterion=nn.CrossEntropyLoss(),
    )
    
    benchmark.print_summary_table(results)
    
    # ========================================================================
    # 保存结果
    # ========================================================================
    print("\n" + "=" * 80)
    print("6. 保存实验结果")
    print("=" * 80)
    
    # 创建结果表格
    results_df = create_comparison_table(results)
    print("\n实验结果汇总:")
    print(results_df.to_string(index=False))
    
    # 保存到CSV
    results_df.to_csv('experiment_results.csv', index=False)
    print("\n结果已保存到 experiment_results.csv")
    
    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)
    print(f"\n总结:")
    print(f"  FP32准确率: {fp32_acc:.2f}%")
    print(f"  PTQ准确率: {ptq_acc:.2f}%")
    print(f"  精度损失: {fp32_acc - ptq_acc:.2f}%")
    print(f"  模型压缩: ~4x (FP32 -> INT8)")


if __name__ == "__main__":
    main()
