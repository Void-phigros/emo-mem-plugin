# Emotional Memory Compressor · 情感记忆压缩器

> 一个 **WorkBuddy 技能（AI 插件）** + 可独立使用的 Python 库。
> 把长对话压缩成「三层结构化记忆」，在有限的上下文窗口下实现长期、一致、自然的对话记忆。

[English Summary](#english-summary) · [特性](#特性-features) · [安装](#安装-install) · [快速开始](#快速开始-quick-start) · [架构](#架构-architecture) · [测试](#测试-tests) · [许可证](#许可证-license)

---

## 它能解决什么问题

大模型对话受上下文窗口限制，长聊会出现「忘事」「人格漂移」「语气前后矛盾」。
本插件把**情感/语气**当作高压缩率的记忆索引 + 生成控制信号：

- 用**事实骨架**防止事实漂移与幻觉
- 用**情感/语气层**保持人格与语气一致（高压缩率）
- 用**原文锚点**做纠错与防过度概括
- 用**混合检索**（关键词+人物+情感+时间+重要性+未解决项）召回相关记忆
- 用**情感状态变量**控制生成语气（先事实、后情感修饰）

核心公式：**有限窗口 + 外部情感/事实混合记忆 + 检索注入 + 生成控制**

---

## 特性 / Features

- ✅ **三层记忆结构**：事实骨架 / 情感语气 / 原文锚点
- ✅ **混合检索**：多维加权，不只靠语义相似，避免错误归因
- ✅ **生成控制信号**：用内部情感状态变量调节语气，而非只写「我很共情」
- ✅ **生命周期钩子**：`pre_llm_call` / `post_llm_call` / `on_session_end` / `run_decay` 自动召回、压缩、遗忘
- ✅ **零硬依赖**：纯 Python 标准库即可运行；LLM 不可用时自动回退到规则式压缩（保证可测试）
- ✅ **遗忘机制**：低强度日常记忆衰减/合并，高重要/未解决项永不衰减
- ✅ **完整测试**：压缩保真度(8)、检索准确率(7)、注入正确性(8)共 23 项测试

---

## 安装 / Install

### 方式一：作为 WorkBuddy 技能（AI 插件）
下载仓库根目录的 **`emotional-memory-compressor.skill`**，在 WorkBuddy 中安装即可作为技能调用。
技能会自动在触发「压缩对话记忆 / 长期对话 / 防止人格漂移 / 混合检索记忆」等场景时生效。

### 方式二：作为 Python 库（自行集成）
```bash
git clone https://github.com/<your-name>/emotional-memory-compressor.git
cd emotional-memory-compressor/emotional-memory-compressor
pip install -r requirements.txt   # 仅 openai 为可选依赖
python run_all.py                  # 一键跑完示例 + 全部测试
```

---

## 快速开始 / Quick Start

```python
from compressor import compress_conversation
from memory_store import MemoryStore
from retriever import retrieve_memory
from injector import inject_and_control, fact_check

# 1) 压缩对话为结构化记忆（LLM 不可用时自动回退规则式）
history = [
    {"role": "user", "content": "项目截止日期快到了，我很焦虑，怕做不完。"},
    {"role": "assistant", "content": "我理解你的压力。我们来看看能做什么。"},
    {"role": "user", "content": "同事A说他会帮我处理数据，我保证下周三前交报告。"},
    {"role": "user", "content": "演示格式还没确定，待定。谢谢，感觉好多了。"},
]
memory, inject_text = compress_conversation(history, use_llm=False)

# 2) 存储
store = MemoryStore("./my_memory_store")
store.add(memory)

# 3) 混合检索
results = retrieve_memory(store, query_text="截止日期 焦虑", people=["同事A"], max_results=3)

# 4) 注入 + 生成控制
inject_text, directive = inject_and_control(results)
# inject_text 注入上下文；directive 注入 system prompt 控制语气

# 5) 生成后事实校验
warnings = fact_check(generated_text, directive)
```

更完整的生命周期编排见 `scripts/hooks.py`（`HookManager`）。

---

## 架构 / Architecture

```
对话 ──▶ 触发压缩 ──▶ 存储 ──▶ 检索 ──▶ 注入 ──▶ 生成 ──▶ 校验 ──▶ 遗忘
            │                                    ▲
        compress_conversation()            inject_and_control()
        schema.py (数据类)                  fact_check()
        memory_store.py (分层聚合)          hooks.py (生命周期编排)
        retriever.py (混合检索)
```

### 三层记忆结构

| 层级 | 内容 | 特点 |
|---|---|---|
| 事实骨架层 | 谁/何时/何事、承诺、未解决项、因果 | 短、硬、可核查 |
| 情感/语气层 | 情感标签+强度+对象、语气、关系变化 | 高压缩率，保持人格一致 |
| 原文锚点层 | 关键承诺/高情绪/歧义原句（≤3条） | 少量，用于纠错 |

### 文件结构

```
emotional-memory-compressor/
├── SKILL.md                 # 技能清单（frontmatter 即插件元数据）
├── scripts/
│   ├── schema.py            # 结构化记忆 Schema + 数据类
│   ├── compressor.py        # 压缩核心 compress_conversation()
│   ├── memory_store.py      # 文件型记忆库 + 分层聚合
│   ├── retriever.py         # 混合检索器 HybridRetriever
│   ├── injector.py          # 注入器 + 生成控制信号 + 事实校验
│   └── hooks.py             # 生命周期钩子
├── references/
│   ├── schema.md            # 记忆 Schema 字段详解
│   └── compression_prompt.md# 压缩 Prompt 设计要点
├── tests/                   # 23 项测试
└── run_all.py/.sh/.bat      # 一键示例 + 测试
```

---

## 测试 / Tests

```bash
cd emotional-memory-compressor
python run_all.py
# 或单独运行
cd scripts && PYTHONPATH=. python3 ../tests/test_compression.py
```

CI 在 Python 3.10–3.13 上自动跑全部示例与测试（见 `.github/workflows/ci.yml`）。

---

## 风险与边界

1. **情感标签有歧义**：同样「焦虑」原因可能不同，必须绑定对象与事实骨架
2. **有损压缩**：情感概括适合人格/语气一致性，不适合精确事实召回
3. **错误归因**：只靠情感检索会错误归因，必须用混合检索
4. **检索错误累积**：多轮对话中需保留原文锚点 + 事实校验
5. **隐私与操纵风险**：处理情感记忆请确保知情同意与权限声明

详见 `references/compression_prompt.md`。

---

## 许可证 / License

[MIT](LICENSE) © emotional-memory-compressor contributors

---

## English Summary

A WorkBuddy **skill (AI plugin)** and standalone Python library that compresses long
conversations into a three-layer structured memory (fact skeleton + emotion/tone + original
anchors). It enables long-term, consistent, natural dialogue under a limited context window
via hybrid retrieval and a generation-control signal driven by an internal emotion state.

- **Zero hard dependencies** (pure stdlib; `openai` optional, with rule-based fallback).
- **23 tests** covering compression fidelity, retrieval accuracy, and injection correctness.
- Install as a WorkBuddy `.skill` plugin, or use the `scripts/` modules directly in your own code.
- Licensed under MIT.
