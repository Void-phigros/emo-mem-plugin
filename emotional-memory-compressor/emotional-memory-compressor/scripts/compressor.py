#!/usr/bin/env python3
"""
压缩核心 —— compress_conversation()

把"近期对话"压缩为一条结构化记忆（事实骨架 + 情感/语气 + 原文锚点），
并生成一段用于注入上下文的短摘要。

设计要点（来自方案包 4.4 节）：
  1. 只输出 JSON，不输出无关解释
  2. 区分事实骨架与情感语气
  3. 不编造未出现的事实
  4. 保留承诺、未解决项、关键人物关系
  5. 情感标签必须带对象和强度
  6. 原文锚点最多 3 条
  7. 无法确定的信息写 null 或空数组

调用方式：
  - 真实模式：call_llm() 实际调用大模型
  - 离线/测试模式：当 LLM 不可用时回退到规则式压缩（保证可测试）
"""

from __future__ import annotations

import json
import re
import hashlib
import time
from typing import Any

from schema import (
    Memory,
    validate_memory_dict,
    get_retrieval_keys,
    EMOTION_VOCAB,
)


# ---------------------------------------------------------------------------
# 压缩 Prompt 构建
# ---------------------------------------------------------------------------
COMPRESSION_SYSTEM_PROMPT = """你是一个情感/语气压缩记忆系统。你的任务是把一段对话压缩成一条结构化记忆。

必须遵守的规则：
1. 只输出一个 JSON 对象，不输出任何解释、注释或 markdown 代码块标记。
2. 严格区分"事实骨架"（发生了什么）与"情感语气"（怎么说）。情感层只影响语气，不编造事实。
3. 不编造对话中未出现的事实。无法确定的字段写 null 或空数组 []。
4. 必须保留：承诺、约定、未解决项、关键人物关系与因果。
5. 情感标签必须带 target（对象）和 intensity（强度 0~1）。同一"焦虑"可能原因不同，必须绑定对象和事实。
6. 原文锚点最多 3 条，只保留：关键承诺原句、高情绪强度事件原句、容易产生歧义的事实原句。
7. importance 取 0~1，高情感强度/高重要性/未解决项应取较高值。
8. decay_policy：高重要性或未解决项用 slow 或 never，日常记忆用 normal。
9. source_turns 填写该记忆来源的对话轮次编号。
10. relationship_state.trust_delta / tension_delta 取 -1~1，表示相对变化。"""

COMPRESSION_OUTPUT_TEMPLATE = """\
输出 JSON 必须遵循此结构：
{
  "memory_id": "由时间戳生成的唯一ID",
  "time_range": "ISO8601 或可读时间范围",
  "fact_skeleton": {
    "people": ["涉及的人物"],
    "events": ["发生了什么"],
    "commitments": ["承诺、约定"],
    "unresolved_items": ["未解决事项"],
    "causal_links": ["因果关系"]
  },
  "emotion": {
    "primary": "主要情感标签 或 null",
    "intensity": 0.0,
    "target": "情感对象 或 null",
    "secondary": ["次要情感"]
  },
  "tone_profile": {
    "style": "语气风格 或 null",
    "defensiveness": "防御程度 或 null",
    "need": "用户需要 或 null",
    "avoidance": "用户回避 或 null"
  },
  "relationship_state": {
    "trust_delta": 0.0,
    "tension_delta": 0.0,
    "notes": "备注 或 null"
  },
  "anchors": [
    {"quote": "原句", "reason": "保留原因"}
  ],
  "importance": 0.0,
  "decay_policy": "normal",
  "source_turns": [0]
}"""


def build_compression_prompt(
    history: list[dict[str, str]],
    existing_memory: Memory | None = None,
    n_turns: int = 20,
) -> list[dict[str, str]]:
    """
    构建压缩 Prompt。

    Args:
        history: 近期对话，每个元素 {"role": "user"/"assistant", "content": "..."}
        existing_memory: 已有长期记忆摘要（可选，用于避免重复压缩已有信息）
        n_turns: 压缩轮次阈值
    Returns:
        OpenAI 风格的 messages 列表
    """
    # 截取最近 n_turns 轮
    recent = history[-n_turns:] if len(history) > n_turns else history

    dialogue_text = _format_dialogue(recent)

    existing_context = ""
    if existing_memory is not None:
        existing_context = (
            "\n\n【已有长期记忆摘要（避免重复，关注新增信息）】\n"
            f"事实：{existing_memory.fact_skeleton.events}\n"
            f"未解决项：{existing_memory.fact_skeleton.unresolved_items}\n"
            f"情感：{existing_memory.emotion.primary} -> {existing_memory.emotion.target}\n"
        )

    user_prompt = (
        f"请把以下对话压缩成一条结构化记忆。\n"
        f"对话共 {len(recent)} 轮。\n\n"
        f"【对话内容】\n{dialogue_text}"
        f"{existing_context}\n\n"
        f"{COMPRESSION_OUTPUT_TEMPLATE}"
    )

    return [
        {"role": "system", "content": COMPRESSION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _format_dialogue(history: list[dict[str, str]]) -> str:
    """把对话列表格式化为可读文本。"""
    lines = []
    for i, msg in enumerate(history):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        label = "用户" if role in ("user", "human") else "助手"
        lines.append(f"[{i}] {label}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM 调用（可替换）
# ---------------------------------------------------------------------------
def call_llm(messages: list[dict[str, str]], model: str = "gpt-4o-mini") -> str:
    """
    调用大模型。实际部署时替换为真实的 API 调用。

    在沙箱/离线环境下会抛出 RuntimeError，触发回退到规则式压缩。
    """
    # 尝试使用 openai SDK
    try:
        import openai  # noqa: F401
        client = openai.OpenAI()  # type: ignore[attr-defined]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""
    except Exception:
        raise RuntimeError("LLM 不可用，请使用 fallback_compress 或配置 API key")


# ---------------------------------------------------------------------------
# 解析与校验
# ---------------------------------------------------------------------------
def parse_and_validate(raw: str) -> dict[str, Any]:
    """
    从 LLM 原始输出中提取并校验 JSON。

    处理：去除 markdown 代码块标记、解析 JSON、运行校验。
    """
    cleaned = _strip_code_fences(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM 输出不是合法 JSON: {e}\n原始输出:\n{cleaned[:500]}")

    errors = validate_memory_dict(data)
    if errors:
        raise ValueError(f"记忆校验失败: {'; '.join(errors)}")

    # 确保 memory_id 存在
    if not data.get("memory_id"):
        data["memory_id"] = _gen_memory_id()

    # 确保嵌套对象存在
    data.setdefault("relationship_state", {})
    data["relationship_state"].setdefault("trust_delta", 0.0)
    data["relationship_state"].setdefault("tension_delta", 0.0)
    data.setdefault("anchors", [])

    return data


def _strip_code_fences(text: str) -> str:
    """去除 ```json ... ``` 代码块标记。"""
    text = text.strip()
    # 匹配开头的 ```json 或 ```
    if text.startswith("```"):
        # 去掉第一行
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        # 去掉结尾的 ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _gen_memory_id() -> str:
    """生成唯一 memory_id（加入随机数避免同毫秒碰撞）。"""
    import random
    ts = str(int(time.time() * 1000))
    nonce = str(random.randint(0, 999999))
    h = hashlib.md5((ts + nonce).encode()).hexdigest()[:8]
    return f"mem_{ts}_{h}"


# ---------------------------------------------------------------------------
# 注入摘要渲染
# ---------------------------------------------------------------------------
def render_inject_text(memory: Memory) -> str:
    """
    把结构化记忆渲染为一段短自然语言摘要，用于注入上下文窗口。

    遵循"先事实骨架，后情感修饰"原则。
    """
    parts: list[str] = []

    # 事实骨架（先说"发生了什么"）
    fs = memory.fact_skeleton
    if fs.events:
        parts.append("【事实】" + "；".join(fs.events))
    if fs.commitments:
        parts.append("【承诺】" + "；".join(fs.commitments))
    if fs.unresolved_items:
        parts.append("【未解决】" + "；".join(fs.unresolved_items))

    # 情感/语气（后说"怎么说"）
    em = memory.emotion
    if em.primary:
        target_str = f"（对象：{em.target}）" if em.target else ""
        parts.append(f"【情感】{em.primary}{target_str}，强度 {em.intensity:.1f}")

    tp = memory.tone_profile
    if tp.style:
        need_str = f"，需要{tp.need}" if tp.need else ""
        parts.append(f"【语气】{tp.style}{need_str}")

    # 关系状态
    rs = memory.relationship_state
    if rs.trust_delta != 0 or rs.tension_delta != 0:
        deltas = []
        if rs.trust_delta != 0:
            deltas.append(f"信任{'↑' if rs.trust_delta > 0 else '↓'}{abs(rs.trust_delta):.1f}")
        if rs.tension_delta != 0:
            deltas.append(f"张力{'↑' if rs.tension_delta > 0 else '↓'}{abs(rs.tension_delta):.1f}")
        parts.append("【关系】" + "，".join(deltas))

    if not parts:
        return ""

    header = f"[记忆 {memory.memory_id[:12]} | {memory.time_range}]"
    return header + " " + " | ".join(parts)


# ---------------------------------------------------------------------------
# 核心函数：compress_conversation
# ---------------------------------------------------------------------------
def compress_conversation(
    history: list[dict[str, str]],
    existing_memory: Memory | None = None,
    n_turns: int = 20,
    use_llm: bool = True,
) -> tuple[Memory, str]:
    """
    压缩对话为结构化记忆。

    Args:
        history: 近期对话 [{"role": "user"/"assistant", "content": "..."}]
        existing_memory: 已有长期记忆（可选）
        n_turns: 压缩轮次阈值，默认 20 轮
        use_llm: 是否尝试调用 LLM；失败时回退到规则式压缩
    Returns:
        (Memory, inject_text) —— 结构化记忆对象 + 注入摘要文本
    """
    if not history:
        raise ValueError("history 不能为空")

    if use_llm:
        try:
            messages = build_compression_prompt(history, existing_memory, n_turns)
            raw = call_llm(messages)
            data = parse_and_validate(raw)
            memory = Memory.from_dict(data)
            inject_text = render_inject_text(memory)
            return memory, inject_text
        except RuntimeError:
            # LLM 不可用，回退
            pass
        except (ValueError, KeyError) as e:
            # 解析失败，回退
            print(f"[compressor] LLM 输出解析失败，回退规则式压缩: {e}")

    # 回退：规则式压缩（离线可测试）
    memory = fallback_compress(history, existing_memory, n_turns)
    inject_text = render_inject_text(memory)
    return memory, inject_text


# ---------------------------------------------------------------------------
# 回退：规则式压缩（无需 LLM，用于离线测试）
# ---------------------------------------------------------------------------
def fallback_compress(
    history: list[dict[str, str]],
    existing_memory: Memory | None = None,
    n_turns: int = 20,
) -> Memory:
    """
    规则式压缩：当 LLM 不可用时的回退方案。

    通过关键词匹配和启发式规则生成结构化记忆，保证可测试性。
    不是生产级质量，但能验证管线完整性。
    """
    from schema import FactSkeleton, Emotion, ToneProfile, RelationshipState, Anchor

    recent = history[-n_turns:]
    full_text = " ".join(m.get("content", "") for m in recent)

    # 情感检测：关键词匹配
    emotion_map = {
        "焦虑": ["焦虑", "担心", "怕", "紧张", "压力", "deadline", "截止"],
        "愤怒": ["愤怒", "气", "烦", "讨厌", "不满", "抱怨"],
        "期待": ["期待", "希望", "盼", "等", "想要"],
        "失落": ["失落", "失望", "沮丧", "难过", "可惜"],
        "被认可": ["谢谢", "感谢", "做得好", "赞", "认可", "棒"],
        "信任": ["信任", "相信", "靠谱", "放心"],
        "防御": ["但是", "不是", "没有", "我不是说", "别误会"],
        "疲惫": ["累", "疲惫", "倦", "撑不住", "熬"],
    }

    detected_emotion = None
    detected_intensity = 0.0
    for label, keywords in emotion_map.items():
        hits = sum(1 for kw in keywords if kw in full_text)
        if hits > 0:
            score = min(hits / 5.0, 1.0)
            if score > detected_intensity:
                detected_emotion = label
                detected_intensity = score

    # 承诺检测
    commitment_keywords = ["我会", "我保证", "承诺", "约定", "下周", "明天", "之后", "一定"]
    commitments = [
        sent.strip()
        for sent in re.split(r"[。！？\n]", full_text)
        if any(kw in sent for kw in commitment_keywords) and len(sent.strip()) > 3
    ][:5]

    # 未解决项检测
    unresolved_keywords = ["还没", "待定", "未解决", "怎么办", "怎么处理", "不确定", "还需要"]
    unresolved = [
        sent.strip()
        for sent in re.split(r"[。！？\n]", full_text)
        if any(kw in sent for kw in unresolved_keywords) and len(sent.strip()) > 3
    ][:5]

    # 人物检测（简单启发式）
    people: list[str] = []
    person_patterns = [
        r"同事([A-Z])",
        r"同事(\w)",
        r"(我|你|他|她)",
    ]
    for pat in person_patterns:
        for m in re.finditer(pat, full_text):
            p = m.group(0)
            if p not in people and len(p) <= 4:
                people.append(p)

    # 事件提取（取用户消息的前若干句）
    user_msgs = [m.get("content", "") for m in recent if m.get("role") in ("user", "human")]
    events = []
    for msg in user_msgs[:5]:
        first_sent = re.split(r"[。！？\n]", msg)[0].strip()
        if len(first_sent) > 5:
            events.append(first_sent[:80])
    events = events[:5]

    # 原文锚点：取最长的用户句
    anchors: list[Anchor] = []
    all_sents = [(i, s.strip()) for i, s in enumerate(re.split(r"[。！？\n]", full_text)) if len(s.strip()) > 10]
    all_sents.sort(key=lambda x: len(x[1]), reverse=True)
    for idx, sent in all_sents[:3]:
        anchors.append(Anchor(quote=sent[:100], reason="长句/高信息密度"))

    # 重要性：基于情感强度 + 是否有承诺/未解决项
    importance = min(
        detected_intensity * 0.5
        + (0.3 if commitments else 0)
        + (0.3 if unresolved else 0)
        + (0.1 if anchors else 0),
        1.0,
    )

    # 衰减策略
    if importance > 0.7 or unresolved:
        decay_policy = "never"
    elif importance > 0.4:
        decay_policy = "slow"
    else:
        decay_policy = "normal"

    # 语气
    style = None
    if detected_emotion == "防御":
        style = "防御"
    elif detected_emotion == "被认可":
        style = "需要被认可"
    elif detected_emotion in ("愤怒",):
        style = "直接"

    memory = Memory(
        memory_id=_gen_memory_id(),
        time_range=f"turns 0-{len(recent) - 1}",
        fact_skeleton=FactSkeleton(
            people=people[:5],
            events=events,
            commitments=commitments,
            unresolved_items=unresolved,
            causal_links=[],
        ),
        emotion=Emotion(
            primary=detected_emotion,
            intensity=round(detected_intensity, 2),
            target="用户本人" if detected_emotion else None,
            secondary=[],
        ),
        tone_profile=ToneProfile(
            style=style,
            defensiveness="高" if detected_emotion == "防御" else None,
            need="被认可" if detected_emotion == "被认可" else None,
            avoidance=None,
        ),
        relationship_state=RelationshipState(
            trust_delta=0.1 if detected_emotion == "被认可" else 0.0,
            tension_delta=0.2 if detected_emotion in ("愤怒", "防御") else 0.0,
            notes=None,
        ),
        anchors=anchors,
        importance=round(importance, 2),
        decay_policy=decay_policy,
        source_turns=list(range(len(recent))),
    )

    return memory


# ---------------------------------------------------------------------------
# CLI 入口（可直接运行测试）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 示例对话
    sample_history = [
        {"role": "user", "content": "我这个项目的截止日期快到了，我很焦虑，怕做不完。"},
        {"role": "assistant", "content": "我理解你的压力。我们来看看还能做什么。"},
        {"role": "user", "content": "同事A说他会帮我处理数据部分，我保证下周三前交报告。"},
        {"role": "assistant", "content": "好的，那下周三是关键节点。还有未解决的部分吗？"},
        {"role": "user", "content": "还没确定演示的格式，待定。谢谢你的帮助，感觉好多了。"},
    ]

    memory, inject = compress_conversation(sample_history, use_llm=False)
    print("=== 结构化记忆 ===")
    print(json.dumps(memory.to_dict(), ensure_ascii=False, indent=2))
    print("\n=== 注入摘要 ===")
    print(inject)
    print("\n=== 检索键 ===")
    print(json.dumps(get_retrieval_keys(memory), ensure_ascii=False, indent=2))
