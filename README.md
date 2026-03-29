# LLM 知识大纲

## 前置知识

> 需要自行查阅资料了解，主要是数学、编程和神经网络的基本功，作为后续学习的支撑。

### 数学基础

- 线性代数：向量、矩阵、特征值/特征向量、线性变换
- 微积分：导数、积分、多元微积分、梯度
- 概率与统计：分布、期望、方差、协方差、假设检验、贝叶斯推断

### Python 与数据科学

- Python 基础语法、面向对象编程
- 数据科学库：NumPy、Pandas、Matplotlib
- 数据预处理：缺失值处理、归一化、特征工程
- 机器学习库：Scikit-learn（回归、分类、聚类、降维）

### 神经网络

- 基本结构：层、权重、偏置、激活函数
- 训练与优化：反向传播、损失函数、优化器（SGD、Adam）
- 过拟合与正则化：Dropout、L1/L2、早停

### 传统自然语言处理（NLP）

- 文本预处理：分词、词干化、词形还原、停用词
- 特征表示：BoW、TF-IDF、n-grams
- 词向量：Word2Vec、GloVe、FastText
- 序列模型：RNN、LSTM、GRU

---

## LLM 训练推理

> 聚焦于如何构建和优化最先进的 LLM。

### 模型架构

- Transformer 基础
- Tokenization（分词与编码）
- 注意力机制（Self-Attention）
- 文本生成策略（Beam Search, Sampling Params: Greedy, Temperature, Top-P, Top-K）

### 数据准备

- 大规模语料清洗、去重、过滤、分桶
- 大规模数据的 Dataloader 工作流

### 预训练（Pretrain）

- 分布式训练：数据并行、流水线并行、张量并行
- 优化技巧：学习率调度、混合精度训练、梯度裁剪
- 框架：Megatron-LM

### 后训练（Post-train）

- 指令微调数据（SFT）
- 合成数据生成与质量过滤

### 强化学习（RL）

- PPO、GRPO（基于强化学习的对齐）
- 奖励模型（Reward Model）与 Rubric 奖励

### 推理优化（Inference）

- KV Cache
- Paged Attention
- 推理框架：vLLM, SGLang

### 模型评估（Evaluation）

- 自动化基准测试（MMLU 等）
- 人类评估（Arena、人工打分）
- 模型评估（Judge LLM、Reward Model）

### 微调（Fine-tuning）

- 全参数微调
- 参数高效微调（LoRA、QLoRA）

### 量化（Quantization）

- FP32/BF16 → FP8/INT8/INT4
- 工具：GGUF、llama.cpp、GPTQ、AWQ

### MoE 模型

- EP 设计
- 多卡部署推理

### 多模态模型

- ViT，图像二维位置编码
- Vocoder，音频编码（VQ，Semantic tokens，Acoustic tokens）

---

## LLM 应用

> 关注如何构建、优化和部署基于 LLM 的应用。

### 运行 LLM

- API 调用（OpenAI、Anthropic、Hugging Face）
- 本地运行（llama.cpp、Ollama、LM Studio）
- Prompt Engineering（Zero-shot、Few-shot、Chain-of-Thought、ReAct）

### 安全与可靠性

- 幻觉
- Prompt 注入
- 安全机制绕过

### 向量存储与检索增强（RAG）

- 文档加载与切分
- 嵌入模型（Embedding Models）
- 向量数据库（FAISS、Milvus、Chroma）
- RAG 框架（LangChain、LlamaIndex）

### 智能体（Agents）

- 工具调用（Tool Calling）
- 应用：Dify
