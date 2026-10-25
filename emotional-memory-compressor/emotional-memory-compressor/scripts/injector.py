#!/usr/bin/env python3
"""
注入器与生成控制 —— inject_and_control()

核心原则（方案包 2.3 节）：
  先事实骨架，后情感修饰。
  情感层只影响"怎么说"，不直接编造"发生了什么"。
  对关键事实使用原文锚点或事实骨架校验。

职责：
  1. 把检索到的记忆渲染为注入文本，拼接到 LLM 上下文
  2. 提取"生成控制信号"（情感状态变量），用于调节生成语气
  3. 事实校验：生成后检查输出是否与事实骨架/原文锚点冲突
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from schema import Memory
from compressor import render_inject_text
from retriever import RetrievalResult


# ---------------------------------------------------------------------------
# 生成控制信号
# ---------------------------------------------------------------------------
@dataclass
class GenerationControlSignal:
    """
    后台情感状态变量，用于调节生成语气。

    这些变量不直接写入生成文本，而是作为"控制信号"影响
    共情方式、措辞和防御程度。
    """

    primary_emotion: str | None = None
    emotion_intensity: float = 0.0
    emotion_target: str | None = None
    tone_style: str | None = None
    defensiveness: str | None = None
    user_need: str | None = None
    avoidance: str | None = None
    trust_delta: float = 0.0
    tension_delta: float = 0.0
    # 汇总的事实要点（用于"先事实骨架"）
    fact_points: list[str] = None  # type: ignore[assignment]
    commitments: list[str] = None  # type: ignore[assignment]
    unresolved_items: list[str] = None  # type: ignore[assignment]
    # 原文锚点（用于校验）
    anchors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.fact_points is None:
            self.fact_points = []
        if self.commitments is None:
            self.commitments = []
        if self.unresolved_items is None:
            self.unresolved_items = []
        if self.anchors is None:
            self.anchors = []

    def to_system_directive(self) -> str:
        """
        把控制信号转换为生成指令（注入 system prompt）。

        注意：这里只给"怎么说"的指引，不给"发生了什么"的事实。
        事实由 inject_text 单独注入。
        """
        parts: list[str] = []

        # 情感语气指引（"怎么说"）
        if self.primary_emotion:
            target_str = f"（指向{self.emotion_target}）" if self.emotion_target else ""
            intensity_label = (
                "强烈" if self.emotion_intensity > 0.7
                else "中等" if self.emotion_intensity > 0.3
                else "轻微"
            )
            parts.append(
                f"用户当前情感：{self.primary_emotion}{target_str}，强度{intensity_label}"
            )

        if self.tone_style:
            parts.append(f"语气风格：{self.tone_style}")

        if self.defensiveness:
            parts.append(f"防御程度：{self.defensiveness}（注意降低对抗感）")

        if self.user_need:
            parts.append(f"用户需要：{self.user_need}")

        if self.avoidance:
            parts.append(f"用户回避：{self.avoidance}（不要触碰）")

        # 关系状态
        if self.trust_delta != 0 or self.tension_delta != 0:
            parts.append(
                f"关系变化：信任{'↑' if self.trust_delta > 0 else '↓'}"
                f"{abs(self.trust_delta):.1f}，"
                f"张力{'↑' if self.tension_delta > 0 else '↓'}"
                f"{abs(self.tension_delta):.1f}"
            )

        if not parts:
            return ""

        return "【生成语气控制】" + "；".join(parts) + "。情感层只影响语气，不要编造事实。"


# ---------------------------------------------------------------------------
# 注入器
# ---------------------------------------------------------------------------
class Injector:
    """
    把检索到的记忆注入 LLM 上下文，并提取生成控制信号。

    注入格式遵循"先事实骨架，后情感修饰"：
      [注入记忆] 事实骨架 → 承诺 → 未解决项 → 情感 → 语气 → 关系 → 锚点
      [生成控制] 语气指引
    """

    def inject(
        self,
        results: list[RetrievalResult],
        max_memories: int = 3,
        include_anchors: bool = True,
    ) -> tuple[str, GenerationControlSignal]:
        """
        注入检索结果到上下文。

        Args:
            results: 检索结果列表
            max_memories: 最多注入多少条记忆
            include_anchors: 是否包含原文锚点
        Returns:
            (inject_text, control_signal)
        """
        top = results[:max_memories]
        if not top:
            return "", GenerationControlSignal()

        # 渲染每条记忆的注入文本
        memory_texts: list[str] = []
        for r in top:
            text = render_inject_text(r.memory)
            if text:
                memory_texts.append(text)

        inject_text = ""
        if memory_texts:
            inject_text = (
                "【长期记忆注入】以下是相关历史记忆，"
                "请基于这些事实骨架回答，情感信息用于调节语气：\n"
                + "\n".join(memory_texts)
            )

        # 提取生成控制信号
        control_signal = self._extract_control_signal(top, include_anchors)

        return inject_text, control_signal

    def _extract_control_signal(
        self, results: list[RetrievalResult], include_anchors: bool
    ) -> GenerationControlSignal:
        """从检索结果中提取情感状态变量。"""
        # 取得分最高的记忆作为主要情感信号
        if not results:
            return GenerationControlSignal()

        primary = results[0].memory
        em = primary.emotion
        tp = primary.tone_profile
        rs = primary.relationship_state

        # 汇总事实要点
        fact_points: list[str] = []
        commitments: list[str] = []
        unresolved_items: list[str] = []
        anchors: list[str] = []

        for r in results:
            m = r.memory
            fact_points.extend(m.fact_skeleton.events)
            commitments.extend(m.fact_skeleton.commitments)
            unresolved_items.extend(m.fact_skeleton.unresolved_items)
            if include_anchors:
                anchors.extend(a.quote for a in m.anchors)

        return GenerationControlSignal(
            primary_emotion=em.primary,
            emotion_intensity=em.intensity,
            emotion_target=em.target,
            tone_style=tp.style,
            defensiveness=tp.defensiveness,
            user_need=tp.need,
            avoidance=tp.avoidance,
            trust_delta=rs.trust_delta,
            tension_delta=rs.tension_delta,
            fact_points=fact_points[:5],
            commitments=list(dict.fromkeys(commitments))[:3],
            unresolved_items=list(dict.fromkeys(unresolved_items))[:3],
            anchors=anchors[:3],
        )


# ---------------------------------------------------------------------------
# 事实校验
# ---------------------------------------------------------------------------
def fact_check(
    generated_text: str,
    control_signal: GenerationControlSignal,
) -> list[str]:
    """
    事实校验：检查生成文本是否与事实骨架/原文锚点冲突。

    返回冲突警告列表，空列表表示通过。

    校验维度：
      1. 承诺是否被否认或遗忘
      2. 未解决项是否被错误地"解决"
      3. 原文锚点中的关键事实是否被曲解
    """
    warnings: list[str] = []
    text_lower = generated_text.lower()

    # 1. 承诺校验：如果生成文本中出现了与承诺矛盾的表达
    denial_patterns = ["没有承诺", "没说过", "不存在", "否认", "不是约定"]
    if control_signal.commitments:
        for pattern in denial_patterns:
            if pattern in text_lower:
                warnings.append(
                    f"⚠ 生成文本包含否认表达'{pattern}'，但记忆中有承诺："
                    f"{control_signal.commitments[:2]}"
                )
                break

    # 2. 未解决项校验：不要把未解决项说成已解决
    resolved_patterns = ["已经解决", "已完成", "搞定了", "没问题了", "已确认"]
    if control_signal.unresolved_items:
        for pattern in resolved_patterns:
            if pattern in text_lower:
                warnings.append(
                    f"⚠ 生成文本包含'已解决'表达'{pattern}'，但存在未解决项："
                    f"{control_signal.unresolved_items[:2]}"
                )
                break

    # 3. 原文锚点校验：检查关键事实是否被曲解
    if control_signal.anchors:
        # 简单检查：锚点中的关键数字/人名是否在生成文本中出现
        for anchor in control_signal.anchors[:2]:
            # 提取锚点中的数字
            import re
            numbers = re.findall(r"\d+", anchor)
            if numbers:
                missing = [n for n in numbers if n not in text_lower]
                if missing:
                    warnings.append(
                        f"⚠ 锚点中的数字 {missing} 未在生成文本中出现，"
                        f"可能遗漏关键事实。锚点：{anchor[:50]}"
                    )

    return warnings


# ---------------------------------------------------------------------------
# 完整注入流程
# ---------------------------------------------------------------------------
def inject_and_control(
    results: list[RetrievalResult],
    max_memories: int = 3,
    include_anchors: bool = True,
) -> tuple[str, str]:
    """
    一站式注入：返回 (inject_text, system_directive)。

    inject_text 注入到上下文中（提供事实骨架），
    system_directive 注入到 system prompt（控制语气）。
    """
    injector = Injector()
    inject_text, signal = injector.inject(results, max_memories, include_anchors)
    directive = signal.to_system_directive()
    return inject_text, directive


if __name__ == "__main__":
    import tempfile
    from compressor import compress_conversation
    from retriever import retrieve_memory
    from memory_store import MemoryStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(tmpdir)
        conv = [
            {"role": "user", "content": "项目截止日期快到了，我很焦虑，怕做不完。"},
            {"role": "assistant", "content": "我理解你的压力。"},
            {"role": "user", "content": "同事A说他会帮我处理数据，我保证下周三前交报告。"},
            {"role": "assistant", "content": "下周三是关键节点。还有未解决的吗？"},
            {"role": "user", "content": "演示格式还没确定，待定。谢谢，感觉好多了。"},
        ]
        mem, _ = compress_conversation(conv, use_llm=False)
        store.add(mem)

        results = retrieve_memory(store, query_text="截止日期 焦虑 报告", max_results=3)
        inject_text, directive = inject_and_control(results)

        print("=== 注入文本 ===")
        print(inject_text)
        print("\n=== 生成控制指令 ===")
        print(directive)

        # 模拟生成后的校验
        print("\n=== 事实校验（模拟生成文本）===")
        bad_text = "这个项目没有承诺过任何截止日期，已经解决了。"
        warnings = fact_check(bad_text, Injector().inject(results)[1])
        for w in warnings:
            print(w)
