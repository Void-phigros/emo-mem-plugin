#!/usr/bin/env python3
"""
记忆存储 —— 基于文件的结构化记忆库

设计：
  - 每条记忆存为单独的 JSON 文件，便于增量更新和审计
  - 维护一个索引文件 index.json，加速检索
  - 支持分层聚合：日摘要 → 周情感章节 → 月事实骨架

不依赖向量数据库，使用结构化字段过滤 + 轻量关键词匹配实现混合检索。
（向量检索可作为可选增强，见 retriever.py）
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from schema import Memory, get_retrieval_keys


# ---------------------------------------------------------------------------
# 记忆存储
# ---------------------------------------------------------------------------
class MemoryStore:
    """
    基于文件系统的结构化记忆库。

    目录结构：
      <store_dir>/
        memories/          # 每条记忆一个 JSON 文件
          <memory_id>.json
        index.json         # 检索索引（memory_id -> 检索键）
        aggregates/        # 分层聚合结果
          daily/           # 日摘要
          weekly/          # 周情感章节
          monthly/         # 月事实骨架
    """

    def __init__(self, store_dir: str | os.PathLike):
        self.store_dir = Path(store_dir)
        self.memories_dir = self.store_dir / "memories"
        self.index_path = self.store_dir / "index.json"
        self.aggregates_dir = self.store_dir / "aggregates"

        self.memories_dir.mkdir(parents=True, exist_ok=True)
        self.aggregates_dir.mkdir(parents=True, exist_ok=True)
        (self.aggregates_dir / "daily").mkdir(exist_ok=True)
        (self.aggregates_dir / "weekly").mkdir(exist_ok=True)
        (self.aggregates_dir / "monthly").mkdir(exist_ok=True)

        self._index: dict[str, dict[str, Any]] = self._load_index()

    # ----- 索引 -----
    def _load_index(self) -> dict[str, dict[str, Any]]:
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_index(self) -> None:
        self.index_path.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ----- 增删改查 -----
    def add(self, memory: Memory) -> None:
        """添加一条记忆并更新索引。"""
        path = self.memories_dir / f"{memory.memory_id}.json"
        path.write_text(
            json.dumps(memory.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._index[memory.memory_id] = get_retrieval_keys(memory)
        self._save_index()

    def get(self, memory_id: str) -> Memory | None:
        """按 ID 读取一条记忆。"""
        path = self.memories_dir / f"{memory_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Memory.from_dict(data)

    def delete(self, memory_id: str) -> bool:
        """删除一条记忆。"""
        path = self.memories_dir / f"{memory_id}.json"
        if path.exists():
            path.unlink()
        if memory_id in self._index:
            del self._index[memory_id]
            self._save_index()
            return True
        return False

    def list_all_ids(self) -> list[str]:
        """返回所有记忆 ID。"""
        return list(self._index.keys())

    def list_all(self) -> list[Memory]:
        """加载所有记忆（慎用于大库）。"""
        return [self.get(mid) for mid in self._index if (m := self.get(mid)) is not None]

    # ----- 遗忘/合并 -----
    def apply_decay(self, threshold: float = 0.3) -> list[str]:
        """
        应用时间衰减：删除 importance 低于阈值且 decay_policy=normal 的记忆。

        策略（方案包 2.4 节）：
          - 高情感强度、高重要性、未解决项优先保留
          - 低强度日常记忆按时间衰减或合并
        Returns:
            被删除的 memory_id 列表
        """
        deleted: list[str] = []
        for mid, keys in list(self._index.items()):
            if keys.get("decay_policy", "normal") == "normal":
                imp = keys.get("importance", 0.0)
                if imp < threshold and not keys.get("has_unresolved"):
                    self.delete(mid)
                    deleted.append(mid)
        return deleted

    # ----- 分层聚合 -----
    def aggregate_daily(self, date_str: str, memories: list[Memory]) -> dict[str, Any]:
        """
        日摘要聚合：合并同一天的多条记忆。

        合并规则：
          - 事实骨架的各字段取并集
          - 情感取强度最高的 primary
          - 关系状态 delta 累加
          - 原文锚点合并后取前 3 条
          - importance 取最大值
        """
        agg = self._aggregate_memories(memories, f"daily_{date_str}", date_str)
        path = self.aggregates_dir / "daily" / f"{date_str}.json"
        path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
        return agg

    def aggregate_weekly(self, week_str: str, memories: list[Memory]) -> dict[str, Any]:
        """周情感章节聚合：侧重情感演变。"""
        agg = self._aggregate_memories(memories, f"weekly_{week_str}", week_str)
        # 周聚合额外记录情感演变轨迹
        emotion_timeline = [
            {"time": m.time_range, "emotion": m.emotion.primary, "intensity": m.emotion.intensity}
            for m in memories
            if m.emotion.primary
        ]
        agg["emotion_timeline"] = emotion_timeline
        path = self.aggregates_dir / "weekly" / f"{week_str}.json"
        path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
        return agg

    def aggregate_monthly(self, month_str: str, memories: list[Memory]) -> dict[str, Any]:
        """月事实骨架聚合：侧重事实沉淀，防止事实漂移。"""
        agg = self._aggregate_memories(memories, f"monthly_{month_str}", month_str)
        path = self.aggregates_dir / "monthly" / f"{month_str}.json"
        path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
        return agg

    def _aggregate_memories(
        self, memories: list[Memory], agg_id: str, time_range: str
    ) -> dict[str, Any]:
        """合并多条记忆为一条聚合记忆（dict 形式）。"""
        all_people: set[str] = set()
        all_events: list[str] = []
        all_commitments: list[str] = []
        all_unresolved: list[str] = []
        all_causal: list[str] = []
        all_anchors: list[dict[str, str]] = []
        max_emotion = None
        max_intensity = 0.0
        trust_sum = 0.0
        tension_sum = 0.0
        max_importance = 0.0

        for m in memories:
            all_people.update(m.fact_skeleton.people)
            all_events.extend(m.fact_skeleton.events)
            all_commitments.extend(m.fact_skeleton.commitments)
            all_unresolved.extend(m.fact_skeleton.unresolved_items)
            all_causal.extend(m.fact_skeleton.causal_links)
            all_anchors.extend({"quote": a.quote, "reason": a.reason} for a in m.anchors)
            if m.emotion.primary and m.emotion.intensity > max_intensity:
                max_emotion = m.emotion.primary
                max_intensity = m.emotion.intensity
            trust_sum += m.relationship_state.trust_delta
            tension_sum += m.relationship_state.tension_delta
            max_importance = max(max_importance, m.importance)

        # 去重
        all_commitments = list(dict.fromkeys(all_commitments))
        all_unresolved = list(dict.fromkeys(all_unresolved))
        all_anchors = all_anchors[:3]

        return {
            "memory_id": agg_id,
            "time_range": time_range,
            "fact_skeleton": {
                "people": sorted(all_people),
                "events": all_events[:10],
                "commitments": all_commitments[:5],
                "unresolved_items": all_unresolved[:5],
                "causal_links": all_causal[:5],
            },
            "emotion": {
                "primary": max_emotion,
                "intensity": round(max_intensity, 2),
                "target": "聚合",
                "secondary": [],
            },
            "relationship_state": {
                "trust_delta": round(trust_sum, 2),
                "tension_delta": round(tension_sum, 2),
                "notes": "聚合记忆",
            },
            "anchors": all_anchors,
            "importance": round(max_importance, 2),
            "decay_policy": "slow",
            "source_count": len(memories),
        }

    # ----- 统计 -----
    def stats(self) -> dict[str, Any]:
        """返回记忆库统计信息。"""
        total = len(self._index)
        with_unresolved = sum(1 for v in self._index.values() if v.get("has_unresolved"))
        never_decay = sum(
            1 for v in self._index.values() if v.get("decay_policy") == "never"
        )
        avg_importance = (
            sum(v.get("importance", 0) for v in self._index.values()) / total
            if total
            else 0
        )
        return {
            "total_memories": total,
            "with_unresolved_items": with_unresolved,
            "never_decay": never_decay,
            "avg_importance": round(avg_importance, 3),
        }


if __name__ == "__main__":
    import tempfile
    from compressor import compress_conversation

    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(tmpdir)
        sample = [
            {"role": "user", "content": "我这个项目截止日期快到了，我很焦虑，怕做不完。"},
            {"role": "assistant", "content": "我理解你的压力。我们来看看还能做什么。"},
            {"role": "user", "content": "同事A说他会帮我处理数据，我保证下周三前交报告。"},
            {"role": "assistant", "content": "好的，下周三是关键节点。还有未解决的吗？"},
            {"role": "user", "content": "演示格式还没确定，待定。谢谢，感觉好多了。"},
        ]
        memory, inject = compress_conversation(sample, use_llm=False)
        store.add(memory)
        print("存储后统计:", store.stats())
        print("注入摘要:", inject)
        retrieved = store.get(memory.memory_id)
        print("取回的 memory_id:", retrieved.memory_id if retrieved else None)
