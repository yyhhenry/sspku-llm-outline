# Tokenizer 详解

## 知识

### 工具

目前的 tokenizer 方案基本集中于 hf tokenizers 和 tiktoken 两种，而且它们都使用 Rust 编写核心算法，并导出 Python 包。

- HuggingFace 的 Tokenizers
  - 源码：https://github.com/huggingface/tokenizers
  - 几乎所有的 hf 开源模型都会附带 hf 格式的 tokenizer，是开源社区的事实标准。
  - 提供多种可扩展的配置项，推理时的编解码性能较好。
  - 附带了构建 tokenizer 的脚本，但性能并不强。
- OpenAI 的 TikToken
  - 源码：https://github.com/openai/tiktoken
  - 所有 OpenAI 的模型都会使用 tiktoken 作为 tokenizer。
  - 扩展配置相对较少，采用 OpenAI 统一的设计风格。
  - 推理时的性能理论上比 hf 更好，但是两者都足够快，不会成为瓶颈。
  - 仅仅附带一个 toy 构建脚本，基本不考虑性能。

### 设计

忽略一些历史残留的产品，所有新模型几乎都使用 B-BPE，也就是 Byte Pair Encoding 的 ByteLevel 实现。

### 基本思路

- 构建阶段
  - 准备一个用于构建 tokenizer 的数据集，按照 utf-8 编码做成字节流 `T`。
  - 初始状态下，设置一个大小为 256 的初始 vocab，代表字节流中任意的单字节。
  - 这时候我们已经可以把字节流转换为 vocab 中的一项了，但是没有任何压缩信息的效果。
  - 接下来循环进行合并字节对的操作。
    - 每次选择出现频率最高的字节对 `(a, b)`。
    - 这里的字节可能来自原始的 256 个字符，也可能是已经合并过的字节对。
    - 计算方法是 `T` 中出现最多的连续两个字节组成的字节对，将其合并为一个新的 token。
    - 设定 vocab 中的下一项 `c` 为 `a` 和 `b` 代表的原始字节的合并。
    - 更新原始的字节流 `T`，将所有连续出现的字节对 `(a, b)` 替换为新的 token，也就是 `c`。
    - 重复上述步骤，直到达到预设的词汇表大小或满足其他停止条件为止。
  - 这时候我们得到了一个词表，其中每个 token 代表一系列原始 256 个 token，也就是一些字节，并且记录下来了 `(a, b) -> c` 的映射关系。
- 解码阶段
  - 拿到一个 token 序列后，根据构建阶段记录的合并信息，直接把原始的 bytes 拼接就可以了。
- 编码阶段
  - 拿到 bytes，按照词表中的编号依次去匹配 `(a, b) -> c` 中的 `(a, b)` 这样的 pair。
  - 如果发现编号 `c` 的 token 对应 `(a, b)` 在 bytes 中存在，就将每个 `(a, b)` 替换为 `c`。
  - 直到整个词表全部扫描完为止。

### 优化

- 预分词（提升性能，控制效果）
  - 将文本按照指定的正则表达式，提前拆分为子单元。
  - 通常对于英语来说，会把一个空格和后面的单词划分到一个 pre-tokenized 单元中。
  - 对于构建阶段，合并操作和 pair 统计将不会跨过子单元边界进行。
    - 这导致很多子单元是重复的，例如英语单词，意味着可以做一次去重，然后做成一个 `(word, freq)` 的新数据结构，在统计词频的时候只需要算一次，结果乘以 `freq` 即可。
    - 这提供了并行计算的可能性，每次统计高频 pair 可以多个线程处理不同的子单元。
  - 对于编码阶段同理。
    - 先用较快的 regex 把字符串切分，随后并行做每个部分即可。
  - 对于任意阶段的合并操作。
    - 每个 word 很短，直接原地 `O(n)` 操作即可，无需额外优化。
  - 对于效果的控制。
    - 很多模型会用预分词控制词表的组成。
    - 例如 Qwen 的词表规定数字必须单独作为一个 token，这或许对小参数模型正确处理数学问题提供了帮助。
    - 其他一些常见模型，如 gpt-4o 的词表 o200k，则规定每 3 个数字作为一个 token。例如一个年月日信息 `2025年 11月 6日` 会被分词为 `["202", "5", "年", "11", "月", "6", "日"]`。
    - 其他常见规则还包括：单个符号可以和单词预分词为一个 word，一些符号后面跟着一些换行也可以是一个 word；而对于代码场景的驼峰表达，o200k 规定一个大写字母与后面的小写字母组合成一个 word，连续的大写字母是一个 word。例如 `HashMap` 会被分词为 `["Hash", "Map"]`，而 `HASH_MAP` 则是 `["HASH", "_MAP"]`。
- 构建优化
  - 使用一个 map 维护 `pair -> 包含 pair 的 word 列表`，这样在合并 pair 时可以快速定位需要更新的 word，然后动态更新这个 map，避免扫描不含指定 pair 的 word，显著提速。
  - 使用一个优先级队列维护待合并的 pair，每次直接取出最高优先级的 pair 进行合并操作，避免比较所有 pair 的优先级来找出最高优先级的 pair。
- 编码算法优化
  - 对于每个 word，长度很短，只和词表中的少量 token 相关。
  - 可以考虑如下循环。
    - 扫描这个 word，得到当前涉及到的每一对 pair `(a, b)`。
    - 找出存在的 pair 里面编号 `c` 最小的 `(a, b)`。
    - 执行 `(a, b) -> c` 的合并操作。
  - 这样就不用考虑无关的词表了，性能会大幅提升。

### 实际应用

- 特殊 token
  - 在实际模型部署中，除了常规的词表构建，还需要考虑一些特殊 token 的设计。这些 token 通常不参与 BPE 合并过程，但在模型训练和推理中扮演重要角色。
  - 例如 `<|endoftext|>` 就是很多模型，特别是基于 Qwen 的模型，都会用到的特殊 token，用于代替传统的 `<pad>` 和 `<eos>` token 的功能。
  - 这些 token 通常用尖括号和竖线包裹，与正常内容不同，不会走到分词阶段，会在预分词的时候就被 regex 直接匹配出来，并直接合成单个 token 且切开前后的 word，它本身不允许进一步合并。
- 提示词模板
  - 现代大模型广泛采用 prompt 工程来控制模型行为，tokenizer 的设计也需要配合这一机制。
  - 例如现在常用的格式有 `<|im_start|>` 和 `<|im_end|>`，它们用于明确区分不同角色的输入和输出，以及提供系统提示词，使得模型能够更好地理解和处理多轮对话信息；此外 `<think></think>` 用于表示模型的内部思考过程，帮助模型在生成回复前进行逻辑推理和信息整合。
  - 在输入传递给模型之前，会插入一个未完成的、只有 `"<|im_start|>assistant\n"` 的结尾，模型就会知道接下来需要以助手的身份进行回复。
  - token 是模型理解文本的最原始信息，所以 tokenizer 的词表设计至关重要。模型训练期间不能更换词表，因此现在大家通常会使用一些确保稳定可用的词表，避免整个训练开销被浪费。

```text
<|im_start|>system
这里是系统提示词<|im_end|>
<|im_start|>user
用户说了一些内容

可能有多行<|im_end|>
<|im_start|>assistant
<think>
模型可能会进行思考
</think>
这里是模型返回的正文<|im_end|>
<|im_start|>user
用户第二轮输入<|im_end|>
<|im_start|>assistant
模型非思考下输出的普通内容<|im_end|>
```

## 实验

没有固定的实现代码，更多是理解实现和一些探索。

如果求助 LLM 仍然并不能完成实验，请与作者取得联系。

### 准备

- 手动下载 https://huggingface.co/Qwen/Qwen3-0.6B
- 使用 HuggingFace 的 tokenizers 库单独加载 Qwen3 的 tokenizer。
- 提取并分析预分词使用的 Regex，借助大模型理解其中规则和逻辑，以更好分析分词行为。
- 理解 json 文件中 `vocabs`、`merges` 字段代表的含义，可能需要阅读 tokenizers 库代码。

### 正式实验

- 准备中英的真实文本，使用 `tokenizer_config.json` 中的 template 构造为对话格式，看看用到了什么特殊 token，并分析特殊 token 阶段、预分词阶段、编码阶段等不同阶段的分词行为。
- 用一些不同编程语言的代码数据测试 tokenizer 在代码场景下的行为，测试相同文本对应的 token 长度，可将上面的 Qwen3 词表与 DeepSeek、GLM、OpenAI 使用的词表对比。这时你需要额外用到 tiktoken。

### 进阶

以下三个任务建议只画出设计思路，不必实际实现。编辑 Tokenizer 并不是一项特别核心的工作，实际编写这些任务可能较为耗时，且收益并不大。

- 尝试使用 tiktoken 提供的 toy 代码理解 B-BPE 构造过程，使用 hf tokenizers 在中英数据集上进行真实的 B-BPE 词表构建，并对比不同预分词策略和数据策略的效果，尝试理解为什么 Qwen3 要设计那样的预分词策略。
- 尝试设计一个能够对生成词表进行后处理的脚本，输入输出都是合法的 `tokenizer.json`，可以根据用户输入的参数去除指定语言或者超过指定长度的 token，然后重新调整总词表大小，得到一个裁剪后的词表。例如去掉 `0xfffd` 乱码和其他字符组成的词，限制新词表大小为 128k。
- 尝试思考一下有没有可能 token 中会有不完整的 utf-8 被错误合并。阅读 utf-8 的规则，设计一个新的构建流程，使得每个 token 要么是一个合法的 utf-8 字符串，要么是单个 utf-8 字符的一个前缀。

### 参考结果

```python
from tokenizers import Encoding, Tokenizer

tokenizer_path = "你的路径/Qwen/Qwen3-0.6B/tokenizer.json"

tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)

text = "你好，世界！"
encoding: Encoding = tokenizer.encode(text)
decoded_text = tokenizer.decode(encoding.ids)

tokens_as_str = [tokenizer.decode([token_id]) for token_id in encoding.ids]

print(f"{text=}")
print(f"{encoding.ids=}")
print(f"{decoded_text=}")
print(f"{tokens_as_str=}")

# text='你好，世界！'
# encoding.ids=[108386, 3837, 99489, 6313]
# decoded_text='你好，世界！'
# tokens_as_str=['你好', '，', '世界', '！']
```

预分词策略：可选的空格，一个可选符号加一些字母或汉字；标点只能与标点合并或后接换行；代码标识符不分驼峰；单个数字为 token。具体可以让大模型分析。

```jsonc
{
  // ...
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {
        "type": "Split",
        "pattern": {
          "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        },
        "behavior": "Isolated",
        "invert": false,
      },
      {
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": false,
        "use_regex": false,
      },
    ],
  },
  // ...
}
```

可能的对比结果：

- 预分词方面，各个模型几乎都是从 GPT-2 的结构中修改而来。数字每一位都断开几乎只有 Qwen，其余模型均使用 3 位数字一个 token 的方式节约长度。
- Qwen 好几代模型的 vocab 和 merges 一样，special 不一样，总大小都是 151k 左右；其余模型多为 128k 左右；gpt-4o 的 o200k 顾名思义是 200k 词表。
- Qwen 在代码方面非常节约 token；o200k 词表中的中文占比明显小于其他几个中国开源模型，因此中文内容在 4o 中所需 token 数较多。
