---
name: emotional-memory-compressor
description: 情感/语气压缩记忆系统。当需要实现长期连续对话、管理长对话记忆、做情感记忆压缩与检索、构建带情感状态的 Agent 记忆层、或处理对话上下文窗口耗尽问题时使用。触发场景包括"压缩对话记忆""情感记忆""长期对话""防止人格漂移""混合检索记忆""对话记忆管理"等。基于三层记忆结构（事实骨架 + 情感语气 + 原文锚点）实现高压缩率记忆索引。
---

# Emotional Memory Compressor

## Overview

把"情感/语气"当作高压缩率的记忆索引和生成控制信号：用少量情感与语气摘要概括长段对话，用事实骨架防止事实漂移，用混合检索召回相关记忆，用情感状态变量控制生成语气，从而在有限上下文窗口下实现更长期、更一致、更自然的对话。

核心公式：**有限窗口 + 外部情感/事实混合记忆 + 检索注入 + 生成控制**。

## 核心思想

### 三层记忆结构

| 层级 | 内容 | 特点 |
|---|---|---|
| 事实骨架层 | 谁/何时/何事、承诺、未解决项、因果关系 | 短、硬、可核查 |
| 情感/语气层 | 情感标签+强度+对象、语气特征、关系状态变化 | 高压缩率，保持人格一致 |
| 原文锚点层 | 关键承诺原句、高情绪事件原句、歧义事实原句（≤3条） | 少量，用于纠错 |

### 四个核心原则

1. **混合检索，不只靠语义相似**：组合主题/人物/情感/时间/重要性/未解决项多维度
2. **先事实骨架，后情感修饰**：情感层只影响"怎么说"，不编造"发生了什么"
3. **分层聚合**：日摘要→周情感章节→月事实骨架，延缓检索稀释
4. **压缩与遗忘**：高情感/高重要性/未解决项优先保留，低强度日常记忆衰减

## Quick Start

### 1. 压缩对话为结构化记忆

```python
from compressor import compress_conversation

history = [
    {"role": "user", "content": "项目截止日期快到了，我很焦虑，怕做不完。"},
    {"role": "assistant", "content": "我理解你的压力。我们来看看能做什么。"},
    {"role": "user", "content": "同事A说他会帮我处理数据，我保证下周三前交报告。"},
    {"role": "assistant", "content": "好的，那下周三是关键节点。还有未解决的吗？"},
    {"role": "user", "content": "演示格式还没确定，待定。谢谢，感觉好多了。"},
]

# 压缩为结构化记忆（LLM 不可用时自动回退到规则式压缩）
memory, inject_text = compress_conversation(history, use_llm=False)
```

### 2. 存储记忆

```python
from memory_store import MemoryStore

store = MemoryStore("./my_memory_store")
store.add(memory)
print(store.stats())  # {'total_memories': 1, 'with_unresolved_items': 1, ...}
```

### 3. 混合检索

```python
from retriever import retrieve_memory

# 关键词 + 情感 + 人物组合检索
results = retrieve_memory(store, query_text="截止日期 焦虑", people=["同事A"], max_results=3)
for r in results:
    print(f"[{r.score:.3f}] {r.memory.emotion.primary} | {r.memory.fact_skeleton.events}")
```

### 4. 注入与生成控制

```python
from injector import inject_and_control, fact_check

inject_text, directive = inject_and_control(results)
# inject_text 注入到上下文（提供事实骨架）
# directive 注入到 system prompt（控制语气）

# 生成后做事实校验
warnings = fact_check(generated_text, control_signal)
```

### 5. 生命周期钩子（自动召回与压缩）

```python
from hooks import HookManager

hooks = HookManager(store, compression_threshold=20)

# LLM 调用前：自动召回 + 注入
pre = hooks.pre_llm_call("session-1", "用户消息...")

# LLM 调用后：记录 + 校验 + 检查压缩阈值
post = hooks.post_llm_call("session-1", "助手回复...")

# 会话结束：压缩剩余对话
hooks.on_session_end("session-1")  # 同步
hooks.on_session_end_async("session-1")  # 异步（不阻塞）

# 遗忘策略
hooks.run_decay(threshold=0.3)
```

## 文件结构

```
emotional-memory-compressor/
├── SKILL.md                          # 本文件
├── scripts/
│   ├── schema.py                     # 结构化记忆 Schema + 数据类
│   ├── compressor.py                 # 压缩核心 compress_conversation()
│   ├── memory_store.py               # 基于文件的记忆库 + 分层聚合
│   ├── retriever.py                  # 混合检索器 HybridRetriever
│   ├── injector.py                   # 注入器 + 生成控制信号 + 事实校验
│   └── hooks.py                      # 生命周期钩子（自动召回/压缩/遗忘）
├── tests/
│   ├── test_compression.py           # 压缩保真度测试（8项）
│   ├── test_retrieval.py             # 检索准确率测试（7项）
│   └── test_injection.py            # 注入正确性测试（8项）
└── references/
    ├── schema.md                     # 记忆 Schema 字段详解
    └── compression_prompt.md         # 压缩 Prompt 要点与最佳实践
```

## 运行测试

```bash
cd scripts
PYTHONPATH=. python3 ../tests/test_compression.py
PYTHONPATH=. python3 ../tests/test_retrieval.py
PYTHONPATH=. python3 ../tests/test_injection.py
```

或直接运行各脚本查看示例输出：

```bash
python3 scripts/schema.py        # 查看数据类
python3 scripts/compressor.py   # 压缩示例
python3 scripts/retriever.py    # 检索示例
python3 scripts/injector.py     # 注入示例
python3 scripts/hooks.py        # 钩子生命周期示例
```

## 模块依赖关系

```
schema.py（基础数据类）
  ↑
compressor.py（压缩核心）
  ↑
memory_store.py（存储）
  ↑
retriever.py（检索）
  ↑
injector.py（注入+控制）
  ↑
hooks.py（生命周期编排）
```

每个模块可独立使用，也可通过 hooks.py 编排完整生命周期：
**对话 → 触发压缩 → 存储 → 检索 → 注入 → 生成 → 校验 → 遗忘**。

## 实施路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| 第一步：纯函数内核 | `compress_conversation()` + `retrieve_memory()` | ✅ 已实现 |
| 第二步：Tool 形态原型 | 对话→压缩→存储→检索→注入闭环 | ✅ 已实现 |
| 第三步：Agent 编排 | 触发策略 + 混合检索 + 生成控制 + 遗忘 | ✅ 已实现 |
| 第四步：插件化 | API 插件或运行时钩子插件 | ✅ hooks.py |
| 第五步：长期评估 | 150~300天压力测试 | 待用户执行 |

## 预期效果

| 方案 | 预计可聊天数 | 说明 |
|---|---|---|
| 纯窗口（基线） | ~150天 | 100万上下文，~33token/句 |
| 混合压缩（窗口内） | 300~750天 | 压缩倍数 k=2~5 |
| 固定窗口+外部记忆+分层聚合 | 1~3年 | 现实工程预期 |

## Resources

### scripts/
- `schema.py` — Memory/FactSkeleton/Emotion/ToneProfile 等数据类 + JSON Schema + 校验 + 检索键生成
- `compressor.py` — `compress_conversation()` 压缩核心，含 Prompt 构建、LLM 调用、解析校验、规则式回退、注入摘要渲染
- `memory_store.py` — `MemoryStore` 基于文件的记忆库，支持增删改查、遗忘衰减、日/周/月分层聚合
- `retriever.py` — `HybridRetriever` 多维度加权检索（关键词+人物+情感+时间+重要性+未解决项）
- `injector.py` — `Injector` 注入器 + `GenerationControlSignal` 生成控制信号 + `fact_check()` 事实校验
- `hooks.py` — `HookManager` 生命周期钩子：pre_llm_call / post_llm_call / on_session_end / run_decay / run_aggregation

### references/
- `schema.md` — 记忆数据结构字段详解、检索键说明、分层聚合规则
- `compression_prompt.md` — 压缩 Prompt 设计要点、生成控制原则、压缩倍数估算、风险与边界

### tests/
- `test_compression.py` — 压缩保真度：事实骨架完整性、情感检测、锚点限制、重要性评分、注入格式
- `test_retrieval.py` — 检索准确率：关键词检索、情感过滤、情感消歧、人物过滤、未解决项过滤、Top-k限制
- `test_injection.py` — 注入正确性：注入非空、控制信号提取、指令格式、事实校验（否认检测、错误解决检测）、事实先于情感
