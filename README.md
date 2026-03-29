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

> 截至 2026 年上半年，LLM 应用的核心范式已从"检索增强生成"全面转向 **AI Agent（智能体）**。对绝大多数人来说，这一章是影响最大的部分——无论你是开发者还是普通用户，Agent 正在重新定义人与 AI 的交互方式。以 **Claude Code**（Anthropic 出品的 Agentic 编程工具，GitHub 84k+ star）和 **OpenClaw**（开源个人 AI 助手，可通过 WhatsApp/Telegram/Discord 等聊天应用与你交互）为代表，Agent 已不再是概念验证，而是每天真实运行、替代大量人类工作的生产力工具。RAG、向量数据库等技术并未消失，但它们已降级为 Agent 系统内部的一个子模块，而非应用层的主角。

### Agent 核心概念

- 什么是 Agent：具备规划、工具调用、记忆、自主决策能力的 LLM 应用
- Agent Loop（智能体循环）：感知 → 规划 → 行动 → 观察 → 反思
- 与传统 Chatbot 的区别：从"你问我答"到"你说目标，我来执行"

### 工具调用（Tool Use / Function Calling）

- Tool Calling 协议（OpenAI Function Calling、Anthropic Tool Use）
- 工具类型
  - 代码执行（Bash、Python 沙盒）
  - 文件读写（Read、Write、Edit、Glob、Grep）
  - 网络请求（WebFetch、WebSearch）
  - 浏览器控制（表单填写、数据抓取）
- 权限控制与安全边界
  - 权限模式（default、acceptEdits、plan、bypassPermissions）
  - 操作审批机制（PreToolUse → 执行 → PostToolUse）

### MCP（Model Context Protocol）

- MCP 协议概述：标准化 LLM 与外部工具/数据源的连接方式
- MCP Server 类型（stdio、HTTP、SSE、WebSocket）
- MCP 工具的发现与调用（`mcp__<server>__<tool>` 命名规范）
- 社区生态：MCP Connectors、第三方 MCP Server

### Skills（技能系统）

- Skill 的定义：可复用的提示词 + 工作流 + 工具组合
- Skill 文件格式（YAML frontmatter + Markdown 指令）
- 技能市场与社区共享（如 OpenClaw 的 ClawHub）
- 自生成技能：Agent 能够在运行中为自己编写新 Skill

### Subagent（子智能体）

- Subagent 的设计思想：任务分治、上下文隔离、专业化分工
- 内置 Subagent（如 Claude Code 的 Explore、Plan）
- 自定义 Subagent
  - 定义方式（Markdown + YAML frontmatter）
  - 关键配置：tools、model、permissionMode、memory、hooks
  - 作用域（用户级、项目级、插件级）
- Subagent 调度：自动委派 vs 显式调用（@-mention、`--agent`）
- 前台/后台运行与并行研究模式

### Agent Teams（多智能体协作）

- 从单 Agent 到多 Agent 编排
- 任务分配与进度跟踪（TaskCreate、TaskCompleted）
- Teammate 协同：各 Agent 拥有独立上下文，异步协作
- 典型模式：Coordinator → Worker + Reviewer

### Hooks（生命周期钩子）

- Hook 的作用：在 Agent 生命周期的关键节点注入自定义逻辑
- 关键 Hook 事件
  - SessionStart / SessionEnd
  - PreToolUse / PostToolUse（工具执行前后拦截）
  - SubagentStart / SubagentStop
  - Stop（控制 Agent 是否结束）
- Hook 类型：Command、HTTP、Prompt（LLM 评估）、Agent（子智能体验证）
- 应用场景：阻止危险操作、自动运行测试、审计日志

### 记忆系统（Memory）

- 短期记忆：对话上下文窗口、自动压缩（Auto-Compaction）
- 长期记忆
  - 指令文件（如 CLAUDE.md、.claude/rules/）
  - 持久化记忆目录（agent-memory/）
  - 跨会话学习与知识积累
- 记忆的作用域：用户级（user）、项目级（project）、本地级（local）

### 插件与集成（Plugins & Integrations）

- 插件系统：通过插件分发 Skills、Subagents、Hooks
- 通信渠道集成（WhatsApp、Telegram、Discord、Slack、iMessage）
- 外部服务集成（Gmail、GitHub、Calendar、Obsidian、Spotify 等）
- 定时任务与主动触发（Cron Jobs、Heartbeat、Webhook）

### 代表性产品

- Claude Code（Anthropic）
  - 终端/IDE/桌面/Web 多端 Agentic 编程工具
  - Subagent + Hooks + MCP + Skills 完整体系
  - Agent Teams 多智能体并行开发
  - GitHub Actions / GitLab CI/CD 集成
- OpenClaw（开源社区）
  - 个人 AI 助手，运行在本地机器上，数据私有
  - 通过聊天应用（WhatsApp/Telegram 等）交互
  - 自扩展能力：Agent 可自行编写和安装新 Skill
  - ClawHub 社区技能市场
  - 多模态集成（浏览器控制、语音、图像生成）

### 安全与可靠性

- 幻觉（Hallucination）与事实性验证
- Prompt 注入防御
- Agent 权限控制与沙盒隔离
- 工具调用审计与操作回滚
