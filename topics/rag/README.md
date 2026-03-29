# RAG技术知识梳理

## RAG技术诞生的背景

大语言模型（LLM）在广泛应用中，暴露出事实性幻觉、知识存在截止时效及垂直领域认知匮乏等核心瓶颈，必须有一种方法能在推理阶段，给大模型引入满足时效性、深入本领域的知识。而大模型的 zero-shot 能力本就可以从输入中获得知识，问题就转化为构建知识库并与用户请求相匹配。而嵌入模型的语义表征与向量数据库的高效检索能力，为此提供了关键支撑，由此催生了 RAG（Retrieval-Augmented Generation，检索增强生成）技术。

### 大模型幻觉：核心应用痛点

大模型幻觉（LLM Hallucination）中，与 RAG 技术强相关的核心是事实性幻觉（Factuality Hallucination）——即模型输出内容语法逻辑连贯，但结论与公开可追溯的客观事实直接相悖，如虚构学术文献、错误关联专业概念等，这也是 RAG 需重点解决的痛点。其技术根源集中于模型机制与数据缺陷两大维度，直接催生 RAG 的应用需求：其一，LLM 以概率驱动自回归生成，优先保证局部语义通顺，易忽视全局事实准确性，对低频专业知识更易出现表征偏差；其二，训练数据存在刚性局限，以主流模型 Gemini 3 Pro 为例，其知识边界截止至 2025 年 3 月，无法覆盖后续新信息，且专业领域数据分布极不均衡，在医疗罕见病诊疗、化工特种工艺等垂直场景，标注数据稀缺导致模型缺乏知识锚点。这类问题在关键领域风险极高，如医疗 AI 误判罕见病诊疗方案、金融模型错读新规，而 RAG 通过接入实时专业知识库构建事实约束，成为破解该困境的核心技术。

### 嵌入模型：语义理解的技术基石

嵌入模型（Embedding Model）的核心能力是将文本、图像等非结构化数据映射为低维稠密的语义向量，该向量通过捕捉语言的语法结构、语义关联与上下文依赖，实现对信息的数学化表征——向量空间中通过余弦相似度计算得出的距离越近，代表内容的语义相关性越强。这种特性使其能打破传统关键词匹配的局限，例如“人工智能发展趋势”与“AI 技术演进方向”虽用词不同，但向量相似度极高，可精准建立语义关联，这一过程正是 RAG 系统中“知识片段与用户查询实现精准匹配”的技术核心。例如，一个常用的嵌入模型就是用于中文的 quentinz/bge-large-zh-v1.5，可以使用 Ollama 等软件轻松在本地大批量调用。

值得关注的是，当前嵌入技术已演进至指令驱动阶段，即 Instruct Embedding，其核心是通过用户自定义指令引导模型生成针对性向量，典型的例子是 Qwen3-Embedding 系列，提供 0.6B 至 8B 全尺寸模型选择，可避免通常粗略简短的 query 与 knowledge 文本嵌入产生比较大的差异，相比传统嵌入模型更能贴合 RAG 系统的检索需求，显著提升特定任务下的知识匹配精度。

### 向量数据库：高效检索的存储载体

传统关系型数据库的结构化存储逻辑，完全无法适配高维向量的相似度计算需求，而向量数据库的核心价值就在于通过近似最近邻（ANN）算法实现高效检索。常见的 ANN 算法包括 IVF（倒排文件）、HNSW（分层导航小世界）等，它们与全量匹配的核心区别在于：全量匹配需遍历所有向量计算相似度，数据量达百万级时延迟会飙升至秒级甚至分钟级，而 ANN 通过“以精度微损换速度”的策略，能将检索延迟压缩至毫秒级，这恰好契合 RAG 场景——用户提问后需快速匹配海量知识向量的实时响应需求。

以最简单的 IVF 算法为例，其原理类似快递分拣：先用聚类算法将所有向量分成多个“簇”（如 1000 个），每个簇有一个中心向量；用户查询时，先计算查询向量与各簇中心的相似度，仅在最邻近的几个簇内进行精细匹配，相当于直接排除 99% 以上的无关向量，大幅降低计算成本。这种“先粗筛再精查”的逻辑，完美解决了 RAG 系统中“海量知识向量快速定位”的核心痛点。

很多 RAG 系统会接入 Faiss 作为向量检索工具，但需要注意，FAISS 并非向量数据库，而是 Facebook 开源的向量检索库，它封装了 IVF 等多种 ANN 算法，是构建向量数据库的核心技术部件。企业级场景中常用的向量数据库如 Milvus，它不仅能存储向量及关联的非结构化数据，还支持分布式部署、增量更新与高可用配置，可承载 RAG 系统的亿级知识向量存储与检索需求，为“查询-匹配-增强”全流程提供稳定的底层支撑。

## RAG技术核心知识

### RAG技术解决的核心问题

RAG 通过“外接知识”模式，精准弥补了大模型的四大核心缺陷：

- 解决知识时效性问题：外接知识库可实时更新，无需重新训练模型即可获取最新信息（如 2024 年税收政策、突发新闻等）；
- 降低大模型幻觉风险：所有回答均基于检索到的真实文档，可提供来源追溯，避免模型“一本正经地胡说八道”；
- 补充专业领域知识：可接入医疗文献、法律条文等垂直领域数据库，让通用大模型具备专业服务能力；
- 保障数据隐私安全：知识库可部署在本地或私有服务器，企业内部规章、医疗记录等敏感数据无需上传至云端大模型，避免数据泄露；
- 降低应用成本：相比模型微调所需的海量数据与高额算力，更新知识库的成本几乎可忽略不计。

### RAG的核心原理

RAG 技术本质是为大模型配置“外接知识库”，通过“检索-增强-生成”三步流程，让模型基于真实数据源生成答案，而非单纯依赖内部训练记忆。完整流程可分为离线预处理与在线交互两个阶段：

1. 离线预处理：构建可检索知识库。
2. 在线交互：实现精准生成。

离线预处理包括：

- 文档分块：将长文档（如 PDF、手册）按语义完整性拆分为短片段（通常 1000 字符左右），常用滑动窗口策略避免信息割裂；
- 向量化转换：通过嵌入模型将每个文本片段转换为向量；
- 向量存储：将向量及对应文本存入向量数据库，完成“知识索引”构建。

在线交互包括：

- 检索阶段：用户提问后，系统先将问题转换为向量，在向量数据库中检索相似度最高的 Top N 文本片段，并通过重排序去除重复信息；
- 增强阶段：将检索到的相关片段与用户问题整合为结构化提示（Prompt），格式如“基于以下资料 {检索信息}，回答：{用户问题}”；
- 生成阶段：将增强提示输入大模型，模型基于给定资料生成答案，确保输出的准确性与可追溯性。

## 简单实践

本小节提供一种门槛最低、无需复杂部署的 RAG 快速体验方案，通过 Open WebUI 的可视化界面与 OpenRouter 的免费模型，可在几分钟内完成从环境搭建到知识库问答的全流程，核心优势是无需手动配置嵌入模型、向量数据库等底层组件，适合快速验证 RAG 效果。具体步骤如下：

1. pip 安装并启动 Open WebUI：打开终端执行安装命令 `pip install open-webui`（需提前确保 Python 3.11 及以上环境），安装完成后输入 `open-webui serve` 启动服务。启动成功后，浏览器访问 http://localhost:8080 即可进入可视化界面，系统会自动下载适配的嵌入模型，体验可能不一定最好，但适合入门选择。

2. 连接 OpenRouter 并选择免费模型：OpenRouter 是统一的 AI 模型接口平台，可通过单一 API 访问数百个模型，且提供多款免费模型供体验。首先访问 OpenRouter 官网（https://openrouter.ai）完成注册，在个人中心创建 API 密钥并保存；随后在 Open WebUI 界面中，进入“设置-模型-添加模型”，选择“OpenAI 兼容 API”，填写 API 地址为 https://openrouter.ai/api/v1，粘贴获取的 API 密钥。

3. 创建知识库并实现问答：在 Open WebUI 主界面找到“工作空间 - 知识库”功能入口，创建知识库并上传文档，支持 PDF、TXT、Word 等常见格式的文本文件；上传完成后，系统会自动完成文档分块、向量化（基于内置的嵌入模型）与索引构建，无需手动干预。更简单的做法是，在聊天界面直接拖拽上传 PDF 文档，输入与知识库相关的问题，系统会自动从知识库中检索相关片段并结合所选模型生成答案，可在回答下方查看引用的知识库来源，直观验证 RAG 的事实增强效果。

注：该方案的核心价值是“零代码、低配置”快速体验 RAG 流程，Open WebUI 已封装嵌入模型调用、向量检索等底层逻辑，OpenRouter 的免费模型则降低了大模型调用成本，适合新手快速理解 RAG 的“检索-增强-生成”全链路。

## 手动实现

```bash
curl -L -o Qwen3Tokenizer.local.json "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer.json"
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
uv run rag_lab.py --no-interactive
```

使用 Python 代码来编写简单的 RAG 系统，从 PDF 加载数据，用 tokenizer 来更均匀地进行前面说的滑动窗口切分；随后调用 OpenRouter 上的 Qwen Embedding 8B 作为嵌入模型，然后使用 lab-1806-vec-db 作为向量数据库（建议读者用 Chroma 重新编写），把每一个 chunk 嵌入为向量表示然后存储。

然后在查询阶段，把用户输入与给定的嵌入提示词拼接得到查询向量，找到最接近的 top-k，加入文本大模型的上下文并得到有效的回答。

```
~/src/sspku-llm-outline/topics/rag$ uv run rag_lab.py --no-interactive
Found 1 files for collection 'rag-lab'
Files: ['assets/MiMo-V2-Flash.pdf']
Split file assets/MiMo-V2-Flash.pdf into 29 chunks
Total chunks to process: 29
[1/29] Processed chunk 9 from assets/MiMo-V2-Flash.pdf
[2/29] Processed chunk 2 from assets/MiMo-V2-Flash.pdf
[3/29] Processed chunk 25 from assets/MiMo-V2-Flash.pdf
[4/29] Processed chunk 22 from assets/MiMo-V2-Flash.pdf
[5/29] Processed chunk 26 from assets/MiMo-V2-Flash.pdf
[6/29] Processed chunk 20 from assets/MiMo-V2-Flash.pdf
[7/29] Processed chunk 15 from assets/MiMo-V2-Flash.pdf
[8/29] Processed chunk 27 from assets/MiMo-V2-Flash.pdf
[9/29] Processed chunk 18 from assets/MiMo-V2-Flash.pdf
[10/29] Processed chunk 3 from assets/MiMo-V2-Flash.pdf
[11/29] Processed chunk 7 from assets/MiMo-V2-Flash.pdf
[12/29] Processed chunk 5 from assets/MiMo-V2-Flash.pdf
[13/29] Processed chunk 0 from assets/MiMo-V2-Flash.pdf
[14/29] Processed chunk 24 from assets/MiMo-V2-Flash.pdf
[15/29] Processed chunk 8 from assets/MiMo-V2-Flash.pdf
[16/29] Processed chunk 4 from assets/MiMo-V2-Flash.pdf
[17/29] Processed chunk 14 from assets/MiMo-V2-Flash.pdf
[18/29] Processed chunk 17 from assets/MiMo-V2-Flash.pdf
[19/29] Processed chunk 21 from assets/MiMo-V2-Flash.pdf
[20/29] Processed chunk 19 from assets/MiMo-V2-Flash.pdf
[21/29] Processed chunk 28 from assets/MiMo-V2-Flash.pdf
[22/29] Processed chunk 6 from assets/MiMo-V2-Flash.pdf
[23/29] Processed chunk 12 from assets/MiMo-V2-Flash.pdf
[24/29] Processed chunk 11 from assets/MiMo-V2-Flash.pdf
[25/29] Processed chunk 23 from assets/MiMo-V2-Flash.pdf
[26/29] Processed chunk 16 from assets/MiMo-V2-Flash.pdf
[27/29] Processed chunk 10 from assets/MiMo-V2-Flash.pdf
[28/29] Processed chunk 13 from assets/MiMo-V2-Flash.pdf
[29/29] Processed chunk 1 from assets/MiMo-V2-Flash.pdf
Collection setup complete.
Running sample query: 介绍 MiMo-V2-Flash 使用 MTP 的情况
Answer:
Retrieved doc | Source: assets/MiMo-V2-Flash.pdf (Index: 0, Similarity: 0.5801138281822205) | 'MiMo-V2-Flash Technical Report.... 10\n3.2 Hyper-Parameters .'
Retrieved doc | Source: assets/MiMo-V2-Flash.pdf (Index: 7, Similarity: 0.5707711577415466) | '., 2025). In such scenarios, M...set to 0.001 during Stage 1'
Retrieved doc | Source: assets/MiMo-V2-Flash.pdf (Index: 1, Similarity: 0.5657748579978943) | ' . 9\n2.3.2 Lightweight MTP Des... workflows (Google DeepMind'
Retrieved doc | Source: assets/MiMo-V2-Flash.pdf (Index: 3, Similarity: 0.5298640131950378) | '2-Thinking on most reasoning b...ry Positional Embedding (Ro'
Retrieved doc | Source: assets/MiMo-V2-Flash.pdf (Index: 18, Similarity: 0.5221940279006958) | '(𝑦=4(1−0.58𝑥0.58)) with an𝑅2of...odels. ArXivpreprint , abs/'
根据提供的技术报告，MiMo-V2-Flash 中使用的 MTP（Multi-Token Prediction，多词元预测）主要体现在以下几个方面：

**1. 轻量化 MTP 设计 (Lightweight MTP Design)**
为了防止 MTP 成为推理时的瓶颈，MiMo-V2-Flash 采用了一种轻量化的 MTP 结构：
*   **架构选择**：使用小的**密集前馈网络 (dense FFN)** 而非 MoE（混合专家）架构，以限制参数量。
*   **注意力机制**：采用**滑动窗口注意力 (SWA)** 而非全局注意力 (GA)，以减少 KV 缓存和注意力计算成本。
*   **参数规模**：每个 MTP 块的参数量仅为 **0.33B**。
*   **训练阶段**：
    *   在**预训练阶段**，仅挂载单个 MTP 头，以避免额外的训练开销。
    *   在**后训练阶段**，将该头复制 $K$ 次以形成 $K$ 步 MTP 模块，并联合训练所有头部以进行多步预测。每个头部接收主模型的隐藏状态和词元嵌入作为输入，提供更丰富的预测信息。

**2. 预训练中的应用**
*   **数据与目标**：在预训练的 27 万亿词元中，MiMo-V2-Flash 使用了原生的 32k 上下文长度，并采用了 **MTP 目标**。
*   **超参数配置**：在预训练期间，模型仅使用**单个 MTP 层**。
*   **高效扩展**：MTP 增强了注意力和前馈网络 (FFN) 的计算效率，显著降低了整体延迟。

**3. 推理加速 (Speculative Decoding)**
*   **复用机制**：MiMo-V2-Flash 复用预训练好的 MTP 模块作为**推测解码 (Speculative Decoding)** 的**草稿模型 (Draft Model)**。
*   **性能提升**：通过这种方式，模型在推理时实现了显著的加速：
    *   **接受长度**：最高可达 **3.6**。
    *   **解码速度**：在使用三层 MTP 时，实现了 **2.6 倍** 的解码速度提升（具体数值随批处理大小变化，最高可达 2.7 倍）。

**4. 开源情况**
小米不仅开源了模型权重，还开源了**三层 MTP 权重**，以促进开放研究和社区协作。

总结来说，MiMo-V2-Flash 通过轻量化的 MTP 设计，在保证模型性能的同时，大幅提升了预训练效率和推理速度。
```
