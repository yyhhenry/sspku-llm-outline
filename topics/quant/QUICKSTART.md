# 大模型量化实验 - 快速开始指南

## 🚀 5分钟快速体验

### 1. 测试基础量化功能

```bash
cd /mnt/user-ssd/wangguoan/research/tutorials/quant
python core/quantization_basics.py
```

**预期输出**:
- 对称vs非对称量化的比较
- 不同位宽的SQNR对比
- 量化误差分析

### 2. 测试PTQ（训练后量化）

```bash
python core/ptq_static.py
```

**学习目标**:
- 理解MinMax、Percentile、MSE三种校准方法
- 观察不同校准方法对量化精度的影响

### 3. 测试QAT（量化感知训练）

```bash
python core/qat_training.py
```

**学习目标**:
- 理解Fake Quantization和STE
- 观察BatchNorm折叠的效果

### 4. 测试GPTQ高级量化

```bash
python core/gptq_quantizer.py
```

**学习目标**:
- 理解Hessian矩阵在量化中的作用
- 学习逐列量化和误差补偿

### 5. 运行完整实验（Jupyter Notebook）

```bash
jupyter notebook comprehensive_quantization_lab.ipynb
```

**实验内容**:
1. Part 1: 量化基础实验 (1-2小时)
2. Part 2: PTQ实验 (2-3小时)
3. Part 3: QAT实验 (2-3小时)
4. Part 4: GPTQ实验 (2-3小时)
5. Part 5: 综合评估与优化 (1-2小时)

---

## 📖 详细教学流程

### Week 1: 量化基础

#### Day 1-2: 理论学习
- 阅读README.md中的量化理论部分
- 理解对称/非对称量化的区别
- 学习量化参数（scale, zero_point）的计算

#### Day 3-4: 代码实现
- 完成`core/quantization_basics.py`中的TODO
- 实现`calculate_qparams_symmetric()`
- 实现`calculate_qparams_asymmetric()`
- 实现`quantize_tensor()`和`dequantize_tensor()`

#### Day 5: 实验与分析
- 运行测试脚本，观察不同配置的效果
- 绘制量化前后的分布对比图
- 分析不同位宽对SQNR的影响

**作业**:
1. 完成量化基础模块的所有TODO
2. 绘制2-bit到16-bit的SQNR曲线
3. 分析逐通道量化相比逐张量的优势

---

### Week 2: PTQ训练后量化

#### Day 1-2: 理论学习
- 理解PTQ的工作流程
- 学习三种校准方法的原理
- 理解敏感度分析的作用

#### Day 3-4: 代码实现
- 完成`core/ptq_static.py`中的TODO
- 实现MinMaxCalibration
- 实现PercentileCalibration
- 实现MSECalibration

#### Day 5: 实验与分析
- 在简单模型上测试PTQ
- 比较三种校准方法的效果
- 进行逐层敏感度分析

**作业**:
1. 实现PTQ的核心校准逻辑
2. 在MNIST数据集上测试PTQ效果
3. 分析哪些层对量化更敏感

---

### Week 3: QAT量化感知训练

#### Day 1-2: 理论学习
- 理解Fake Quantization的原理
- 学习STE（Straight-Through Estimator）
- 理解BatchNorm折叠

#### Day 3-4: 代码实现
- 完成`core/qat_training.py`中的TODO
- 实现FakeQuantizeFunction的forward和backward
- 实现QuantizedLinear和QuantizedConv2d
- 实现BatchNorm折叠函数

#### Day 5: 实验与分析
- 训练一个简单的QAT模型
- 比较QAT vs PTQ的效果
- 观察训练过程中量化参数的变化

**作业**:
1. 实现完整的QAT训练流程
2. 在CIFAR-10上训练QAT模型
3. 对比QAT和PTQ的精度差异

---

### Week 4: GPTQ高级量化

#### Day 1-2: 理论学习
- 阅读GPTQ论文
- 理解Hessian矩阵的作用
- 学习逐列量化算法

#### Day 3-4: 代码实现
- 完成`core/gptq_quantizer.py`中的TODO
- 实现Hessian矩阵计算
- 实现GPTQ核心量化算法
- 实现分组量化策略

#### Day 5: 实验与分析
- 在Transformer模型上测试GPTQ
- 比较GPTQ vs PTQ vs QAT
- 分析不同group size的影响

**作业**:
1. 实现GPTQ的完整流程
2. 在小型GPT模型上测试GPTQ
3. 分析GPTQ的计算复杂度

---

### Week 5-6: 高级主题与综合项目

#### 高级主题
1. **KV-Cache量化**: 优化Transformer推理
2. **混合精度搜索**: 使用进化算法找最优配置
3. **量化感知蒸馏**: 结合知识蒸馏和量化
4. **自定义CUDA Kernel**: 编写INT8矩阵乘法

#### 综合项目选题
1. **项目A**: 在BERT上实现完整的量化流程（PTQ+QAT+GPTQ）
2. **项目B**: 设计自动混合精度搜索系统
3. **项目C**: 复现AWQ或SmoothQuant论文
4. **项目D**: 在真实硬件上部署量化模型

---

## 🎯 评分标准详解

### 基础部分（60分）

#### 量化基础实现（15分）
- [ ] 对称量化正确实现 (5分)
- [ ] 非对称量化正确实现 (5分)
- [ ] 逐通道量化正确实现 (5分)

#### PTQ实现（15分）
- [ ] MinMax校准实现 (5分)
- [ ] Percentile校准实现 (5分)
- [ ] MSE校准实现 (5分)

#### QAT实现（15分）
- [ ] FakeQuantize正确实现 (7分)
- [ ] STE梯度正确 (5分)
- [ ] BN折叠正确 (3分)

#### 实验完整性（15分）
- [ ] 所有模块通过测试 (5分)
- [ ] 实验结果可复现 (5分)
- [ ] 代码规范、注释清晰 (5分)

### 进阶部分（30分）

#### GPTQ实现（10分）
- [ ] Hessian计算正确 (4分)
- [ ] 逐列量化正确 (4分)
- [ ] 误差补偿正确 (2分)

#### 混合精度搜索（10分）
- [ ] 敏感度分析实现 (4分)
- [ ] 进化算法实现 (4分)
- [ ] 帕累托前沿计算 (2分)

#### 分析与可视化（10分）
- [ ] 误差分析详细 (4分)
- [ ] 可视化清晰美观 (3分)
- [ ] Benchmark完整 (3分)

### 挑战部分（10分）

- [ ] 实现KV-Cache量化 (3分)
- [ ] 实现量化感知蒸馏 (3分)
- [ ] 编写CUDA kernel (4分)
- [ ] 复现最新论文 (额外加分)

---

## 💡 常见问题解答

### Q1: 量化后精度下降很多怎么办？

**A**: 尝试以下方法：
1. 使用QAT代替PTQ
2. 使用逐通道量化
3. 跳过敏感层（如第一层和最后一层）
4. 增加校准数据量
5. 尝试更好的校准方法（MSE vs MinMax）

### Q2: GPTQ计算很慢怎么优化？

**A**:
1. 减少校准数据量（100-1000个样本足够）
2. 增大group size（128或256）
3. 使用GPU加速
4. 使用混合精度（FP16进行Hessian计算）

### Q3: 如何在真实模型上应用量化？

**A**: 建议流程：
```python
# 1. 加载预训练模型
model = transformers.AutoModel.from_pretrained('bert-base-uncased')

# 2. PTQ快速测试
from core.ptq_static import PTQQuantizer
config = QuantizationConfig(n_bits=8)
ptq = PTQQuantizer(model, config)
ptq.prepare_calibration()
ptq.calibrate(calibration_loader, num_batches=100)
quantized_model = ptq.quantize_model()

# 3. 评估精度损失
evaluate(quantized_model, test_loader)

# 4. 如果精度不够，使用QAT微调
# ...
```

### Q4: 不同位宽如何选择？

**A**:
- **8-bit**: 通用选择，精度损失小，加速明显
- **4-bit**: 大模型推荐，配合GPTQ效果好
- **2-3-bit**: 需要QAT或特殊技术，精度损失较大
- **混合精度**: 敏感层用8-bit，其他层用4-bit

### Q5: 如何验证量化实现的正确性？

**A**: 使用以下检查：
```python
# 1. 检查量化-反量化的误差
error = (original - dequantized).abs().max()
assert error < threshold

# 2. 检查量化范围
assert quantized.min() >= quant_min
assert quantized.max() <= quant_max

# 3. 检查scale和zero_point的合理性
assert scale > 0
assert quant_min <= zero_point <= quant_max

# 4. 检查SQNR
sqnr = compute_sqnr(original, dequantized)
assert sqnr > expected_sqnr
```

---

## 📚 推荐学习资源

### 论文
1. **GPTQ** - Accurate Post-Training Quantization for GPT
2. **LLM.int8()** - 8-bit Matrix Multiplication for Transformers
3. **AWQ** - Activation-aware Weight Quantization
4. **SmoothQuant** - Accurate and Efficient Post-Training Quantization
5. **QLoRA** - Efficient Finetuning of Quantized LLMs

### 博客文章
- [Lei Mao's Blog - Neural Network Quantization](https://leimao.github.io/article/Neural-Networks-Quantization/)
- [Hugging Face - Quantization](https://huggingface.co/docs/optimum/concept_guides/quantization)
- [PyTorch Quantization Tutorial](https://pytorch.org/docs/stable/quantization.html)

### 开源项目
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [llm.int8()](https://github.com/TimDettmers/bitsandbytes)

### 视频教程
- MIT 6.5940: TinyML and Efficient Deep Learning
- Stanford CS231n: Efficient Methods and Hardware for Deep Learning

---

## 🔧 故障排除

### 环境问题

**问题**: 导入模块失败
```bash
ModuleNotFoundError: No module named 'core'
```

**解决**:
```bash
export PYTHONPATH=/mnt/user-ssd/wangguoan/research/tutorials/quant:$PYTHONPATH
# 或在Python中
import sys
sys.path.insert(0, '/mnt/user-ssd/wangguoan/research/tutorials/quant')
```

**问题**: CUDA out of memory

**解决**:
```python
# 减少batch size
# 或使用CPU
device = 'cpu'
# 或清理GPU缓存
torch.cuda.empty_cache()
```

### 数值问题

**问题**: 量化后全是NaN

**解决**:
```python
# 检查scale是否为0
assert (scale > 0).all()
# 添加小的epsilon
scale = torch.where(scale > 0, scale, torch.ones_like(scale) * 1e-6)
```

**问题**: 梯度爆炸或消失

**解决**:
```python
# 使用梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# 调整学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

---

## 📞 获取帮助

遇到问题？
1. 查看README.md和本快速开始指南
2. 查看代码中的注释和docstring
3. 运行单元测试检查实现
4. 查看GitHub Issues（如果有）
5. 联系课程助教或在论坛提问

**祝你学习愉快！🎉**
