#!/usr/bin/env python3
"""
结构化记忆 Schema 定义

基于"情感/语气压缩记忆"方案：把情感/语气当作高压缩率记忆索引，
用事实骨架防止事实漂移，用原文锚点纠错。

三层记忆结构：
  1. 事实骨架层  —— 短、硬、可核查（谁/何时/何事、承诺、未解决项）
  2. 情感/语气层  —— 高压缩率（情感标签、强度、对象、语气、关系变化）
  3. 原文锚点层  —— 少量高价值原句，用于纠错

使用 Memory 数据类管理单条记忆，MemoryStore 管理记忆集合。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------------------
# 默认情感标签词表（可扩展）
# ---------------------------------------------------------------------------
EMOTION_VOCAB = [
    "焦虑", "被认可", "防御", "失落", "期待", "愤怒", "喜悦",
    "信任", "怀疑", "愧疚", "释然", "兴奋", "疲惫", "挫败",
]

# 语气风格词表
TONE_STYLES = [
    "需要被认可", "避免冲突", "直接", "委婉", "防御", "开放",
    "试探", "坚定", "安抚", "质问",
]

# 衰减策略
DECAY_POLICIES = ["normal", "slow", "never"]


# ---------------------------------------------------------------------------
# JSON Schema（用于校验 LLM 输出）
# ---------------------------------------------------------------------------
MEMORY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "memory_id",
        "time_range",
        "fact_skeleton",
        "emotion",
        "tone_profile",
        "importance",
        "decay_policy",
        "source_turns",
    ],
    "properties": {
        "memory_id": {"type": "string"},
        "time_range": {"type": "string"},
        "fact_skeleton": {
            "type": "object",
            "properties": {
                "people": {"type": "array", "items": {"type": "string"}},
                "events": {"type": "array", "items": {"type": "string"}},
                "commitments": {"type": "array", "items": {"type": "string"}},
                "unresolved_items": {"type": "array", "items": {"type": "string"}},
                "causal_links": {"type": "array", "items": {"type": "string"}},
            },
        },
        "emotion": {
            "type": "object",
            "properties": {
                "primary": {"type": ["string", "null"]},
                "intensity": {"type": "number", "minimum": 0, "maximum": 1},
                "target": {"type": ["string", "null"]},
                "secondary": {"type": "array", "items": {"type": "string"}},
            },
        },
        "tone_profile": {
            "type": "object",
            "properties": {
                "style": {"type": ["string", "null"]},
                "defensiveness": {"type": ["string", "null"]},
                "need": {"type": ["string", "null"]},
                "avoidance": {"type": ["string", "null"]},
            },
        },
        "relationship_state": {
            "type": "object",
            "properties": {
                "trust_delta": {"type": "number"},
                "tension_delta": {"type": "number"},
                "notes": {"type": ["string", "null"]},
            },
        },
        "anchors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "importance": {"type": "number", "minimum": 0, "maximum": 1},
        "decay_policy": {"type": "string", "enum": DECAY_POLICIES},
        "source_turns": {"type": "array", "items": {"type": "integer"}},
    },
}


# ---------------------------------------------------------------------------
# Python 数据类（便于在代码中操作）
# ---------------------------------------------------------------------------
@dataclass
class FactSkeleton:
    """第一层：事实骨架 —— 短、硬、可核查"""

    people: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    commitments: list[str] = field(default_factory=list)
    unresolved_items: list[str] = field(default_factory=list)
    causal_links: list[str] = field(default_factory=list)


@dataclass
class Emotion:
    """第二层：情感/语气 —— 高压缩率索引"""

    primary: str | None = None
    intensity: float = 0.0  # 0.0 ~ 1.0
    target: str | None = None
    secondary: list[str] = field(default_factory=list)


@dataclass
class ToneProfile:
    """语气特征：决定"怎么说"而非"说什么" """

    style: str | None = None
    defensiveness: str | None = None
    need: str | None = None
    avoidance: str | None = None


@dataclass
class RelationshipState:
    """关系状态变化"""

    trust_delta: float = 0.0
    tension_delta: float = 0.0
    notes: str | None = None


@dataclass
class Anchor:
    """第三层：原文锚点 —— 少量高价值原句，用于纠错"""

    quote: str = ""
    reason: str = ""


@dataclass
class Memory:
    """
    单条结构化记忆，对应压缩输出的 JSON。

    一条 Memory = 事实骨架 + 情感/语气 + 原文锚点 + 元信息
    """

    memory_id: str = ""
    time_range: str = ""
    fact_skeleton: FactSkeleton = field(default_factory=FactSkeleton)
    emotion: Emotion = field(default_factory=Emotion)
    tone_profile: ToneProfile = field(default_factory=ToneProfile)
    relationship_state: RelationshipState = field(default_factory=RelationshipState)
    anchors: list[Anchor] = field(default_factory=list)
    importance: float = 0.0  # 0.0 ~ 1.0
    decay_policy: str = "normal"  # normal | slow | never
    source_turns: list[int] = field(default_factory=list)

    # ----- 序列化 -----
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        fs = data.get("fact_skeleton", {}) or {}
        em = data.get("emotion", {}) or {}
        tp = data.get("tone_profile", {}) or {}
        rs = data.get("relationship_state", {}) or {}
        anchors_data = data.get("anchors", []) or []
        return cls(
            memory_id=data.get("memory_id", ""),
            time_range=data.get("time_range", ""),
            fact_skeleton=FactSkeleton(
                people=fs.get("people", []),
                events=fs.get("events", []),
                commitments=fs.get("commitments", []),
                unresolved_items=fs.get("unresolved_items", []),
                causal_links=fs.get("causal_links", []),
            ),
            emotion=Emotion(
                primary=em.get("primary"),
                intensity=float(em.get("intensity", 0.0)),
                target=em.get("target"),
                secondary=em.get("secondary", []),
            ),
            tone_profile=ToneProfile(
                style=tp.get("style"),
                defensiveness=tp.get("defensiveness"),
                need=tp.get("need"),
                avoidance=tp.get("avoidance"),
            ),
            relationship_state=RelationshipState(
                trust_delta=float(rs.get("trust_delta", 0.0)),
                tension_delta=float(rs.get("tension_delta", 0.0)),
                notes=rs.get("notes"),
            ),
            anchors=[Anchor(quote=a.get("quote", ""), reason=a.get("reason", "")) for a in anchors_data],
            importance=float(data.get("importance", 0.0)),
            decay_policy=data.get("decay_policy", "normal"),
            source_turns=list(data.get("source_turns", [])),
        )


# ---------------------------------------------------------------------------
# 校验工具
# ---------------------------------------------------------------------------
def validate_memory_dict(data: dict[str, Any]) -> list[str]:
    """
    轻量校验：返回错误消息列表，空列表表示通过。

    不依赖 jsonschema 库（沙箱可能未安装），手动检查关键字段。
    """
    errors: list[str] = []

    if not data.get("memory_id"):
        errors.append("memory_id 不能为空")
    if not data.get("time_range"):
        errors.append("time_range 不能为空")

    em = data.get("emotion", {})
    if em and em.get("primary"):
        # 情感标签必须带对象和强度
        if em.get("target") is None:
            errors.append("情感标签 primary 必须绑定 target（对象）")
        intensity = em.get("intensity")
        if intensity is None:
            errors.append("情感标签 primary 必须有 intensity（强度 0~1）")

    anchors = data.get("anchors", [])
    if len(anchors) > 3:
        errors.append(f"原文锚点最多 3 条，当前 {len(anchors)} 条")

    dp = data.get("decay_policy", "normal")
    if dp not in DECAY_POLICIES:
        errors.append(f"decay_policy 必须是 {DECAY_POLICIES} 之一，当前为 {dp}")

    imp = data.get("importance", 0.0)
    if not (0 <= imp <= 1):
        errors.append(f"importance 必须在 0~1 之间，当前为 {imp}")

    return errors


def get_retrieval_keys(memory: Memory) -> dict[str, Any]:
    """
    生成检索键：主题/关键词、人物、情感标签、时间、重要性、未解决项状态。

    这是混合检索的基础 —— 避免只靠语义相似。
    """
    em = memory.emotion
    keys: dict[str, Any] = {
        "people": list(memory.fact_skeleton.people),
        "emotion_primary": em.primary,
        "emotion_intensity": em.intensity,
        "emotion_target": em.target,
        "emotion_secondary": list(em.secondary),
        "time_range": memory.time_range,
        "importance": memory.importance,
        "has_unresolved": len(memory.fact_skeleton.unresolved_items) > 0,
        "commitments": list(memory.fact_skeleton.commitments),
        "unresolved_items": list(memory.fact_skeleton.unresolved_items),
        # 主题/关键词从 events 和 causal_links 中提取
        "topics": list(memory.fact_skeleton.events) + list(memory.fact_skeleton.causal_links),
    }
    return keys
