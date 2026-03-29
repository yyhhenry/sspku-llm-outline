"""
可视化工具模块

提供量化实验的各种可视化功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd


def plot_quantization_comparison(
    original: torch.Tensor,
    quantized_dict: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
):
    """
    比较原始权重和不同量化方法的结果
    
    参数:
        original: 原始权重
        quantized_dict: {方法名: 量化后权重}
        save_path: 保存路径
    """
    num_methods = len(quantized_dict)
    fig, axes = plt.subplots(2, (num_methods + 2) // 2, figsize=(16, 8))
    axes = axes.flatten()
    
    # 原始分布
    axes[0].hist(original.flatten().numpy(), bins=100, alpha=0.7, color='blue')
    axes[0].set_title('原始权重', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('值')
    axes[0].set_ylabel('频数')
    axes[0].grid(True, alpha=0.3)
    
    # 各种量化方法
    for idx, (name, quantized) in enumerate(quantized_dict.items(), 1):
        axes[idx].hist(quantized.flatten().numpy(), bins=100, alpha=0.7)
        axes[idx].set_title(name, fontsize=14)
        axes[idx].set_xlabel('值')
        axes[idx].set_ylabel('频数')
        axes[idx].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(len(quantized_dict) + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_heatmap(
    errors: np.ndarray,
    layer_names: List[str],
    method_names: List[str],
    metric_name: str = 'MSE',
    save_path: Optional[str] = None,
):
    """
    绘制误差热力图
    
    参数:
        errors: (num_layers, num_methods) 误差矩阵
        layer_names: 层名称列表
        method_names: 方法名称列表
        metric_name: 指标名称
        save_path: 保存路径
    """
    plt.figure(figsize=(12, max(6, len(layer_names) * 0.5)))
    
    sns.heatmap(
        errors,
        annot=True,
        fmt='.4f',
        cmap='YlOrRd',
        xticklabels=method_names,
        yticklabels=layer_names,
        cbar_kws={'label': metric_name},
    )
    
    plt.title(f'逐层{metric_name}热力图', fontsize=16, fontweight='bold')
    plt.xlabel('量化方法', fontsize=12)
    plt.ylabel('层名称', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_pareto_front(
    results: List[Tuple[float, float, str]],
    xlabel: str = '模型大小 (MB)',
    ylabel: str = '准确率 (%)',
    save_path: Optional[str] = None,
):
    """
    绘制帕累托前沿
    
    参数:
        results: List of (size, accuracy, label)
        xlabel: x轴标签
        ylabel: y轴标签
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    sizes, accs, labels = zip(*results)
    
    plt.scatter(sizes, accs, s=100, alpha=0.6, c=range(len(results)), cmap='viridis')
    
    for i, label in enumerate(labels):
        plt.annotate(
            label,
            (sizes[i], accs[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8,
        )
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title('帕累托前沿：精度 vs 模型大小', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_bits_distribution(
    layer_bits: Dict[str, int],
    save_path: Optional[str] = None,
):
    """
    绘制混合精度配置的位宽分布
    
    参数:
        layer_bits: {layer_name: n_bits}
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 柱状图
    layers = list(layer_bits.keys())
    bits = list(layer_bits.values())
    colors = plt.cm.viridis(np.array(bits) / max(bits))
    
    ax1.bar(range(len(layers)), bits, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=45, ha='right')
    ax1.set_ylabel('位宽 (bits)', fontsize=12)
    ax1.set_title('逐层位宽配置', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 饼图
    bit_counts = {}
    for bit in bits:
        bit_counts[bit] = bit_counts.get(bit, 0) + 1
    
    ax2.pie(
        bit_counts.values(),
        labels=[f'{b}-bit' for b in bit_counts.keys()],
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.Set3(range(len(bit_counts))),
    )
    ax2.set_title('位宽分布', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    save_path: Optional[str] = None,
):
    """
    绘制训练曲线
    
    参数:
        train_losses: 训练损失
        val_losses: 验证损失
        train_accs: 训练准确率
        val_accs: 验证准确率
        save_path: 保存路径
    """
    if train_accs is not None and val_accs is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        ax2 = None
    
    # 损失曲线
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('损失', fontsize=12)
    ax1.set_title('训练/验证损失', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    if ax2 is not None:
        ax2.plot(epochs, train_accs, 'b-', label='训练准确率', linewidth=2)
        ax2.plot(epochs, val_accs, 'r-', label='验证准确率', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('准确率 (%)', fontsize=12)
        ax2.set_title('训练/验证准确率', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_comparison_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'model_size_mb', 'inference_time_ms'],
) -> pd.DataFrame:
    """
    创建方法比较表
    
    参数:
        results: {method_name: {metric_name: value}}
        metrics: 要显示的指标
    
    返回:
        DataFrame
    """
    data = []
    for method, values in results.items():
        row = {'方法': method}
        for metric in metrics:
            if metric in values:
                row[metric] = values[metric]
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # 测试可视化功能
    print("测试可视化工具")
    
    # 测试1: 量化比较
    original = torch.randn(1000, 1000)
    quantized_sym = torch.randn(1000, 1000) * 0.95
    quantized_asym = torch.randn(1000, 1000) * 0.92
    
    plot_quantization_comparison(
        original,
        {'对称量化': quantized_sym, '非对称量化': quantized_asym},
    )
    
    # 测试2: 帕累托前沿
    pareto_data = [
        (10.5, 92.3, '4-bit'),
        (15.2, 94.1, '6-bit'),
        (20.8, 95.2, '8-bit'),
        (42.0, 95.5, 'FP32'),
    ]
    plot_pareto_front(pareto_data)
    
    print("可视化测试完成！")
