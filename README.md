# LLM 知识大纲

## 前置知识

> 需要自行查阅资料了解，作为后续学习的支撑。

学习本大纲前，建议具备以下基础：掌握线性代数（矩阵运算、特征值分解）和概率统计（分布、贝叶斯推断）等数学知识；熟悉 Python 编程及 NumPy、PyTorch 等常用库；理解神经网络的基本原理（前向/反向传播、优化器、正则化）；了解传统 NLP 的基本概念（分词、TF-IDF、Word2Vec、RNN/LSTM）。这些内容可通过各类在线课程或教材自学，无需精通，有基本认知即可进入后续 LLM 相关章节的学习。

---

## LLM 训练推理

> 聚焦于如何构建和优化最先进的 LLM。

### 模型架构

- Transformer 基础
  - Decoder-only架构（如 GPT 系列）
  - 自注意力机制（Self-Attention）
  - MLP（Multi-Layer Perceptron）
- Tokenization（分词与编码）
  - BPE（Byte Pair Encoding）
  - Embedding（词嵌入）
  - 输出采样机制（Softmax、Temperature、Top-P）
- 注意力机制
  - MHA（Multi-Head Attention）
  - GQA（Grouped Query Attention）
  - SWA（Sliding Window Attention）
- MTP（Multi-Token Prediction）
- MoE（Mixture of Experts）
  - Expert Parallelism 设计
  - 多卡部署与推理
- 多模态架构
  - ViT 与图像编码
  - DiT（Diffusion Transformer）
  - Vocoder 与音频编码（VQ、Semantic/Acoustic tokens）

### 数据准备

- 大规模语料清洗、去重、过滤、分桶
- 大规模数据的 Dataloader 工作流

### 预训练（Pretrain）

- 分布式训练：数据并行(DP)、流水线并行(PP)、张量并行(TP)
- 优化技巧：学习率调度、混合精度训练、梯度裁剪
- 优化器：Adam、Muon
- 框架：Megatron-LM

### 后训练（Post-train）

- 指令微调数据（SFT）
- 合成数据生成与质量过滤
- 微调方法
  - 全参数微调
  - 参数高效微调（LoRA、QLoRA）

### 强化学习（RL）

- PPO、GRPO
- 奖励模型（Reward Model）与 Rubric 奖励

### 推理优化（Inference）

- KV Cache
- Paged Attention
- 算子：`flash_attn`、`flashinfer`、`triton`
- 推理框架：SGLang、vLLM
- 量化
  - 精度：BF16 → FP8/INT8/INT4
  - 工具：GGUF、llama.cpp、GPTQ、AWQ

### 模型评估（Evaluation）

- 自动化基准测试
  - 选择题 MMLU 等
  - 数学题 GSM8K 等
  - 编程题 HumanEval 等
  - 测试特定数据的PPL
- Arena 评测平台，权威榜单如Artificial Analysis

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
