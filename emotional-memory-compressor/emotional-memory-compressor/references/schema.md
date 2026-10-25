# 结构化记忆 Schema 参考

本文档详细说明 `emotional-memory-compressor` 的记忆数据结构。

## 核心理念

一条结构化记忆 = **事实骨架** + **情感/语气** + **原文锚点** + **元信息**。

> 情感可以概括长段经历，但只能作为高压缩索引，不能替代事实记忆。

## 字段说明

### 顶层字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `memory_id` | string | 唯一标识，自动生成 |
| `time_range` | string | 时间范围（ISO8601 或可读格式） |
| `fact_skeleton` | object | 第一层：事实骨架 |
| `emotion` | object | 第二层：情感/语气 |
| `tone_profile` | object | 第二层：语气特征 |
| `relationship_state` | object | 第二层：关系状态变化 |
| `anchors` | array | 第三层：原文锚点 |
| `importance` | float(0~1) | 重要性权重 |
| `decay_policy` | enum | 衰减策略：normal / slow / never |
| `source_turns` | array[int] | 来源对话轮次编号 |

### fact_skeleton（事实骨架层）

短、硬、可核查。**防止事实漂移和幻觉的核心。**

```json
{
  "people": ["涉及的人物"],
  "events": ["发生了什么"],
  "commitments": ["承诺、约定"],
  "unresolved_items": ["未解决事项"],
  "causal_links": ["因果关系"]
}
```

### emotion（情感/语气层）

高压缩率索引。**必须绑定对象和强度。**

```json
{
  "primary": "焦虑 | null",
  "intensity": 0.0,
  "target": "情感对象 | null",
  "secondary": ["次要情感"]
}
```

**情感标签词表**：焦虑、被认可、防御、失落、期待、愤怒、喜悦、信任、怀疑、愧疚、释然、兴奋、疲惫、挫败

### tone_profile（语气特征）

决定"怎么说"而非"说什么"。

```json
{
  "style": "需要被认可 | 避免 | 直接 | 委婉 | ...",
  "defensiveness": "高 | 低 | null",
  "need": "被认可 | 安慰 | ...",
  "avoidance": "不要触碰的话题 | null"
}
```

### relationship_state（关系状态）

```json
{
  "trust_delta": 0.0,
  "tension_delta": 0.0,
  "notes": "备注 | null"
}
```

- `trust_delta`：信任变化，-1~1，正值表示信任上升
- `tension_delta`：张力变化，-1~1，正值表示张力上升

### anchors（原文锚点层）

**最多 3 条**。用于纠错和防止过度概括。

```json
[
  {"quote": "原句", "reason": "保留原因"}
]
```

保留优先级：
1. 关键承诺原句
2. 高情绪强度事件原句
3. 容易产生歧义的事实原句

### decay_policy（衰减策略）

| 值 | 适用场景 |
|---|---|
| `normal` | 低强度日常记忆，可衰减/合并 |
| `slow` | 中等重要性记忆，延缓衰减 |
| `never` | 高重要性 / 有未解决项，永不衰减 |

## 检索键

每条记忆会自动生成检索键，用于混合检索：

| 检索键 | 来源 | 用途 |
|---|---|---|
| `people` | fact_skeleton.people | 人物过滤 |
| `emotion_primary` | emotion.primary | 情感过滤 |
| `emotion_intensity` | emotion.intensity | 强度过滤 |
| `emotion_target` | emotion.target | 对象过滤 |
| `time_range` | time_range | 时间衰减 |
| `importance` | importance | 重要性排序 |
| `has_unresolved` | unresolved_items 非空 | 未解决项过滤 |
| `topics` | events + causal_links | 关键词匹配 |

## 分层聚合

按"日摘要—周情感章节—月事实骨架"分层聚合：

| 层级 | 聚合规则 | 衰减策略 |
|---|---|---|
| 日摘要 | 事实并集 + 情感取最强 + delta累加 | slow |
| 周情感章节 | 日摘要合并 + 情感演变轨迹 | slow |
| 月事实骨架 | 周章节合并 + 事实沉淀 | slow |

目标：延缓检索稀释（文档量增加时准确率下降）。
