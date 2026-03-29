# 大模型量化深度实验教程

## 🎯 实验目标

本实验旨在让学生从底层深入理解并实现多种量化算法，涵盖从基础量化到前沿技术的完整知识体系。

## 📚 实验内容

### 1. 量化基础（Quantization Basics）

- **对称量化 vs 非对称量化**
- **逐张量量化 vs 逐通道量化**
- **量化参数计算**（scale, zero_point）
- **反量化与误差分析**

### 2. PTQ - 训练后量化（Post-Training Quantization）

- **动态量化**：运行时计算量化参数
- **静态量化**：基于校准数据集确定参数
- **MinMax校准 vs Percentile校准 vs MSE校准**
- **逐层敏感度分析**

### 3. QAT - 量化感知训练（Quantization-Aware Training）

- **Fake Quantization**：在训练中模拟量化
- **STE（Straight-Through Estimator）**：梯度传播
- **量化感知微调策略**
- **BN折叠与权重吸收**

### 4. GPTQ - 高级权重量化

- **基于Hessian的逐层量化**
- **最优脑量化（OBQ）原理**
- **分组量化策略**
- **稀疏性与量化结合**

### 5. KV-Cache 量化

- **Transformer架构优化**
- **动态范围追踪**
- **内存占用优化**
- **推理加速**

### 6. 混合精度搜索

- **敏感度分析**
- **进化算法/强化学习搜索**
- **硬件约束建模**
- **帕累托前沿优化**

### 7. 高级挑战

- **量化感知蒸馏**
- **量化+剪枝联合优化**
- **自定义CUDA算子**
- **端到端部署优化**

## 🛠️ 环境要求

### 基础环境

```bash
Python >= 3.8
PyTorch >= 2.0.0
transformers >= 4.30.0
numpy >= 1.21.0
matplotlib >= 3.5.0
tqdm >= 4.65.0
scikit-learn >= 1.0.0
```

### 可选环境（用于高级实验）

```bash
CUDA >= 11.7（用于CUDA kernel实验）
bitsandbytes >= 0.41.0
accelerate >= 0.20.0
datasets >= 2.12.0
tensorboard >= 2.13.0
```

### 硬件建议

- **CPU实验**：4核+，16GB内存
- **GPU实验**：NVIDIA GPU with 8GB+ VRAM（推荐）
- **完整GPTQ实验**：16GB+ VRAM

## 📦 安装

```bash
# 克隆仓库
cd /mnt/user-ssd/wangguoan/research/tutorials/quant

# 安装依赖
pip install -r requirements.txt

# （可选）安装CUDA扩展
cd cuda_kernels && python setup.py install
```

## 🚀 快速开始

### 方式1: Jupyter Notebook（推荐新手）

```bash
jupyter notebook comprehensive_quantization_lab.ipynb
```

### 方式2: Python脚本（推荐进阶）

```bash
# 运行PTQ实验
python experiments/run_ptq_experiment.py --model bert-base-uncased --dataset sst2

# 运行QAT实验
python experiments/run_qat_experiment.py --model gpt2 --epochs 3

# 运行GPTQ实验
python experiments/run_gptq_experiment.py --model facebook/opt-125m --bits 4

# 运行完整benchmark
python benchmark.py --all
```

## 📊 实验结构

```
quant/
├── README.md                              # 本文件
├── requirements.txt                        # 依赖列表
├── setup.py                               # 包安装脚本
│
├── comprehensive_quantization_lab.ipynb   # 🌟 主实验Notebook
│
├── core/                                   # 核心量化库（学生需要完成）
│   ├── __init__.py
│   ├── quantization_basics.py             # 基础量化实现
│   ├── ptq_static.py                      # PTQ静态量化
│   ├── qat_training.py                    # QAT训练
│   ├── gptq_quantizer.py                  # GPTQ算法
│   ├── kv_cache_quant.py                  # KV-cache量化
│   └── mixed_precision_search.py          # 混合精度搜索
│
├── models/                                 # 模型定义
│   ├── __init__.py
│   ├── quantized_layers.py                # 量化层实现
│   ├── quantized_bert.py                  # 量化BERT
│   └── quantized_gpt.py                   # 量化GPT
│
├── experiments/                            # 独立实验脚本
│   ├── run_ptq_experiment.py
│   ├── run_qat_experiment.py
│   ├── run_gptq_experiment.py
│   └── run_mixed_precision.py
│
├── utils/                                  # 工具函数
│   ├── __init__.py
│   ├── calibration.py                     # 校准工具
│   ├── data_loader.py                     # 数据加载
│   └── metrics.py                         # 评估指标
│
├── visualization.py                        # 可视化工具
├── benchmark.py                            # 性能测试
│
├── cuda_kernels/                           # 🔥 CUDA扩展（高级挑战）
│   ├── setup.py
│   ├── int8_gemm.cu
│   └── quantize_kernel.cu
│
└── tests/                                  # 单元测试
    ├── test_quantization_basics.py
    ├── test_ptq.py
    └── test_qat.py
```

## 📝 学生任务

### 必做任务（60%）

1. ✅ 实现对称和非对称量化函数
2. ✅ 实现逐通道和逐张量量化
3. ✅ 实现PTQ的三种校准方法
4. ✅ 实现Fake Quantization和STE梯度
5. ✅ 在BERT上测试PTQ和QAT效果

### 进阶任务（30%）

6. ✅ 实现GPTQ的核心算法
7. ✅ 实现KV-cache量化
8. ✅ 进行敏感度分析和混合精度搜索
9. ✅ 完成误差分析和可视化

### 挑战任务（10%）

10. 🔥 实现量化感知蒸馏
11. 🔥 编写INT8矩阵乘法CUDA kernel
12. 🔥 在真实硬件上部署量化模型
13. 🔥 复现最新量化论文（AWQ/SmoothQuant）

## 📈 评分标准

- **代码正确性**（40%）：量化算法实现正确，通过单元测试
- **实验完整性**（30%）：完成必做+进阶任务，结果可复现
- **分析深度**（20%）：误差分析、可视化、benchmark详细
- **创新性**（10%）：挑战任务、算法改进、新方法探索

## 🎓 学习路径

### Week 1-2: 量化基础

- 学习量化数学原理
- 实现基础量化函数
- 测试简单模型（MLP）

### Week 3-4: PTQ与QAT

- 实现PTQ流程
- 实现QAT训练
- 在BERT/GPT-2上实验

### Week 5-6: 高级算法

- 理解GPTQ原理
- 实现KV-cache优化
- 混合精度搜索

### Week 7-8: 挑战与部署

- 选择挑战任务
- 端到端优化
- 撰写实验报告

## 📖 参考资料

### 论文

- [GPTQ: Accurate Post-Training Quantization for GPT](https://arxiv.org/abs/2210.17323)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers](https://arxiv.org/abs/2208.07339)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization](https://arxiv.org/abs/2211.10438)

### 代码库

- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [TensorRT](https://github.com/NVIDIA/TensorRT)

### 教程

- [PyTorch Quantization Tutorial](https://pytorch.org/docs/stable/quantization.html)
- [Neural Network Distiller](https://intellabs.github.io/distiller/)

## 🤝 贡献指南

欢迎提交Issues和Pull Requests改进本教程！

## 📄 许可证

MIT License

## 👨‍🏫 维护者

AI Research Lab - 量化技术教学组

---

**注意**: 本实验是学习工具，部分实现未做完整工程优化。生产环境请使用成熟框架（TensorRT, ONNX Runtime等）。
