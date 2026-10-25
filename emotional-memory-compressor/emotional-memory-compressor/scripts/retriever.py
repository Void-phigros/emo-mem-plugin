#!/usr/bin/env python3
"""
混合检索器 —— retrieve_memory()

不依赖纯语义相似，而是组合多维度检索键（方案包 2.2 节）：
  - 主题/关键词
  - 人物与关系
  - 情感标签与强度
  - 时间衰减
  - 重要性权重
  - 未解决项状态

目标：避免"同样愤怒"被错误归因到不同事件，
      避免只按语义相似召回表面相关但实际无关的记忆。
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import Any

from schema import Memory
from memory_store import MemoryStore


# ---------------------------------------------------------------------------
# 检索查询
# ---------------------------------------------------------------------------
@dataclass
class RetrievalQuery:
    """检索查询：组合多个维度的过滤条件。"""

    keywords: list[str] | None = None         # 主题/关键词
    people: list[str] | None = None           # 人物
    emotion: str | None = None                # 情感标签
    min_intensity: float = 0.0                # 最低情感强度
    emotion_target: str | None = None         # 情感对象
    time_range: str | None = None             # 时间范围
    only_unresolved: bool = False             # 只检索有未解决项的
    only_commitments: bool = False            # 只检索有承诺的
    min_importance: float = 0.0               # 最低重要性
    max_results: int = 5                      # Top-k


# ---------------------------------------------------------------------------
# 检索结果
# ---------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    """单条检索结果。"""

    memory: Memory
    score: float
    match_reasons: list[str]  # 命中了哪些维度，便于解释


# ---------------------------------------------------------------------------
# 混合检索器
# ---------------------------------------------------------------------------
class HybridRetriever:
    """
    混合检索器：在 MemoryStore 上做多维度加权检索。

    评分 = w_keyword * keyword_score
         + w_person  * person_score
         + w_emotion * emotion_score
         + w_time    * time_score
         + w_importance * importance_score
         + w_unresolved * unresolved_score
    """

    def __init__(
        self,
        store: MemoryStore,
        w_keyword: float = 0.3,
        w_person: float = 0.2,
        w_emotion: float = 0.2,
        w_time: float = 0.1,
        w_importance: float = 0.1,
        w_unresolved: float = 0.1,
        time_decay_half_life: float = 30.0,  # 天
    ):
        self.store = store
        self.weights = {
            "keyword": w_keyword,
            "person": w_person,
            "emotion": w_emotion,
            "time": w_time,
            "importance": w_importance,
            "unresolved": w_unresolved,
        }
        self.time_decay_half_life = time_decay_half_life

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """
        执行混合检索。

        流程：
          1. 从索引中按硬过滤条件初筛
          2. 对候选记忆计算多维度加权得分
          3. 按得分排序返回 Top-k
        """
        # 初筛：用索引做硬过滤
        candidate_ids = self._filter_candidates(query)
        if not candidate_ids:
            return []

        # 打分
        results: list[RetrievalResult] = []
        for mid in candidate_ids:
            memory = self.store.get(mid)
            if memory is None:
                continue
            score, reasons = self._score(memory, query)
            results.append(RetrievalResult(memory=memory, score=score, match_reasons=reasons))

        # 排序
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: query.max_results]

    # ----- 初筛 -----
    def _filter_candidates(self, query: RetrievalQuery) -> list[str]:
        """用索引做硬过滤，缩小候选集。"""
        candidates: list[str] = []

        for mid, keys in self.store._index.items():
            ok = True

            if query.only_unresolved and not keys.get("has_unresolved"):
                ok = False

            if query.only_commitments and not keys.get("commitments"):
                ok = False

            if query.people:
                mem_people = set(keys.get("people", []))
                if not any(p in mem_people for p in query.people):
                    ok = False

            if query.emotion and keys.get("emotion_primary") != query.emotion:
                ok = False

            if query.min_intensity > 0:
                if keys.get("emotion_intensity", 0) < query.min_intensity:
                    ok = False

            if query.min_importance > 0:
                if keys.get("importance", 0) < query.min_importance:
                    ok = False

            if ok:
                candidates.append(mid)

        return candidates

    # ----- 打分 -----
    def _score(
        self, memory: Memory, query: RetrievalQuery
    ) -> tuple[float, list[str]]:
        """计算单条记忆的加权得分和命中原因。"""
        score = 0.0
        reasons: list[str] = []

        # 1. 关键词匹配
        kw_score = self._keyword_score(memory, query.keywords or [])
        if kw_score > 0:
            score += self.weights["keyword"] * kw_score
            reasons.append(f"关键词匹配({kw_score:.2f})")

        # 2. 人物匹配
        person_score = self._person_score(memory, query.people or [])
        if person_score > 0:
            score += self.weights["person"] * person_score
            reasons.append(f"人物匹配({person_score:.2f})")

        # 3. 情感匹配
        emotion_score = self._emotion_score(memory, query)
        if emotion_score > 0:
            score += self.weights["emotion"] * emotion_score
            reasons.append(f"情感匹配({emotion_score:.2f})")

        # 4. 时间衰减（越近越高）
        time_score = self._time_score(memory)
        if time_score > 0:
            score += self.weights["time"] * time_score
            reasons.append(f"时间近度({time_score:.2f})")

        # 5. 重要性
        imp_score = memory.importance
        score += self.weights["importance"] * imp_score
        if imp_score > 0.5:
            reasons.append(f"高重要性({imp_score:.2f})")

        # 6. 未解决项加分
        if memory.fact_skeleton.unresolved_items:
            score += self.weights["unresolved"]
            reasons.append("有未解决项")

        return score, reasons

    def _keyword_score(self, memory: Memory, keywords: list[str]) -> float:
        """关键词匹配得分：命中数 / 关键词数。"""
        if not keywords:
            return 0.0
        text = " ".join(
            memory.fact_skeleton.events
            + memory.fact_skeleton.causal_links
            + memory.fact_skeleton.commitments
            + memory.fact_skeleton.unresolved_items
            + [a.quote for a in memory.anchors]
        ).lower()
        hits = sum(1 for kw in keywords if kw.lower() in text)
        return hits / len(keywords)

    def _person_score(self, memory: Memory, people: list[str]) -> float:
        """人物匹配得分。"""
        if not people:
            return 0.0
        mem_people = set(memory.fact_skeleton.people)
        hits = sum(1 for p in people if p in mem_people)
        return hits / len(people)

    def _emotion_score(self, memory: Memory, query: RetrievalQuery) -> float:
        """情感匹配得分：标签匹配 + 强度。"""
        em = memory.emotion
        score = 0.0
        if query.emotion and em.primary == query.emotion:
            score += 0.6
            # 强度越高越相关
            score += 0.4 * em.intensity
        elif query.emotion_target and em.target == query.emotion_target:
            score += 0.3
        elif not query.emotion and not query.emotion_target:
            # 无情感查询条件时，高情感强度的记忆略加分
            score = em.intensity * 0.3
        return score

    def _time_score(self, memory: Memory) -> float:
        """
        时间衰减得分：越近的记忆得分越高。

        使用指数衰减：score = exp(-age / half_life)
        由于 time_range 是字符串，尝试提取数字作为"天数"。
        """
        time_str = memory.time_range or ""
        # 尝试提取天数
        nums = re.findall(r"\d+", time_str)
        if not nums:
            return 0.5  # 无法解析时间，给中间分

        # 取最大的数字作为"距今天数"的近似
        age = max(int(n) for n in nums)
        return math.exp(-age / self.time_decay_half_life)


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------
def retrieve_memory(
    store: MemoryStore,
    query_text: str = "",
    people: list[str] | None = None,
    emotion: str | None = None,
    max_results: int = 5,
) -> list[RetrievalResult]:
    """
    便捷检索接口：从自然语言查询文本中提取关键词进行混合检索。

    Args:
        store: 记忆库
        query_text: 查询文本（会分词作为关键词）
        people: 人物过滤
        emotion: 情感过滤
        max_results: Top-k
    Returns:
        检索结果列表
    """
    # 简单分词：按空格和标点分割
    keywords = [
        w.strip().lower()
        for w in re.split(r"[]\s，。、！？；：""''（）[{}]+", query_text)
        if len(w.strip()) > 1
    ] if query_text else []

    retriever = HybridRetriever(store)
    query = RetrievalQuery(
        keywords=keywords or None,
        people=people,
        emotion=emotion,
        max_results=max_results,
    )
    return retriever.retrieve(query)


if __name__ == "__main__":
    import tempfile
    from compressor import compress_conversation

    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(tmpdir)

        # 存入多条记忆
        conversations = [
            [
                {"role": "user", "content": "项目截止日期快到了，我很焦虑，怕做不完。"},
                {"role": "assistant", "content": "我理解你的压力。"},
                {"role": "user", "content": "同事A说他会帮我处理数据，我保证下周三前交报告。"},
            ],
            [
                {"role": "user", "content": "今天的会议很顺利，领导认可了我的方案，我很开心。"},
                {"role": "assistant", "content": "太好了！恭喜你。"},
                {"role": "user", "content": "不过演示格式还没确定，待定。"},
            ],
            [
                {"role": "user", "content": "我和同事B吵架了，他很不满，我也很生气。"},
                {"role": "assistant", "content": "发生什么了？"},
                {"role": "user", "content": "他说我数据有问题，但其实是他的错。"},
            ],
        ]

        for conv in conversations:
            mem, _ = compress_conversation(conv, use_llm=False)
            store.add(mem)

        print("库中记忆数:", store.stats()["total_memories"])

        # 检索测试
        print("\n--- 检索：关键词'截止日期焦虑' ---")
        results = retrieve_memory(store, query_text="截止日期 焦虑", max_results=3)
        for r in results:
            print(f"  [{r.score:.3f}] {r.memory.emotion.primary} | {r.memory.fact_skeleton.events[:1]} | {r.match_reasons}")

        print("\n--- 检索：情感='愤怒' ---")
        results = retrieve_memory(store, emotion="愤怒", max_results=3)
        for r in results:
            print(f"  [{r.score:.3f}] {r.memory.emotion.primary}({r.memory.emotion.intensity}) | {r.match_reasons}")

        print("\n--- 检索：人物='同事A' ---")
        results = retrieve_memory(store, people=["同事A"], max_results=3)
        for r in results:
            print(f"  [{r.score:.3f}] people={r.memory.fact_skeleton.people} | {r.match_reasons}")
