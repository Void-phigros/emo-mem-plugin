#!/usr/bin/env python3
"""
生命周期钩子 —— 自动召回与自动捕获压缩

方案包 5.2 节第四层：
  - pre_llm_call：LLM 调用前自动召回相关记忆并注入
  - on_session_end：对话结束后自动捕获并异步压缩
  - on_threshold_reached：达到压缩轮次阈值时触发压缩

支持在运行时中实现自动化的记忆生命周期管理。
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from schema import Memory
from compressor import compress_conversation
from memory_store import MemoryStore
from retriever import HybridRetriever, RetrievalQuery, retrieve_memory
from injector import inject_and_control, fact_check, GenerationControlSignal


# ---------------------------------------------------------------------------
# 会话上下文
# ---------------------------------------------------------------------------
@dataclass
class SessionContext:
    """
    单次对话会话的上下文状态。

    跟踪：对话历史、未压缩的轮次、压缩阈值、当前情感状态。
    """

    session_id: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    uncompressed_turns: int = 0         # 自上次压缩以来的轮次
    compression_threshold: int = 20    # 每 N 轮压缩一次
    current_emotion: str | None = None # 当前情感状态变量
    current_tone: str | None = None    # 当前语气状态
    last_injected: str = ""             # 最近注入的记忆文本
    last_control_signal: GenerationControlSignal | None = None

    def add_turn(self, role: str, content: str) -> None:
        """添加一轮对话。"""
        self.history.append({"role": role, "content": content})
        self.uncompressed_turns += 1


# ---------------------------------------------------------------------------
# 钩子管理器
# ---------------------------------------------------------------------------
class HookManager:
    """
    管理记忆生命周期的钩子。

    钩子流程：
      1. 用户消息进来 → pre_llm_call → 召回记忆 + 注入
      2. 助手回复出去 → 记入历史
      3. 未压缩轮次达到阈值 → on_threshold_reached → 压缩 + 存储
      4. 会话结束 → on_session_end → 异步压缩剩余对话
    """

    def __init__(
        self,
        store: MemoryStore,
        compression_threshold: int = 20,
        max_inject_memories: int = 3,
    ):
        self.store = store
        self.compression_threshold = compression_threshold
        self.max_inject_memories = max_inject_memories
        self.sessions: dict[str, SessionContext] = {}

    def get_or_create_session(self, session_id: str) -> SessionContext:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionContext(
                session_id=session_id,
                compression_threshold=self.compression_threshold,
            )
        return self.sessions[session_id]

    # ----- pre_llm_call：召回 + 注入 -----
    def pre_llm_call(
        self,
        session_id: str,
        user_message: str,
        people: list[str] | None = None,
        emotion: str | None = None,
    ) -> dict[str, Any]:
        """
        LLM 调用前钩子：自动召回相关记忆并注入。

        Returns:
            包含 inject_text, system_directive, retrieved_count 的字典
        """
        session = self.get_or_create_session(session_id)
        session.add_turn("user", user_message)

        # 自动检索
        results = retrieve_memory(
            self.store,
            query_text=user_message,
            people=people,
            emotion=emotion,
            max_results=5,
        )

        # 注入
        inject_text, directive = inject_and_control(
            results, max_memories=self.max_inject_memories
        )

        session.last_injected = inject_text
        session.current_emotion = (
            results[0].memory.emotion.primary if results and results[0].memory.emotion.primary
            else session.current_emotion
        )

        return {
            "inject_text": inject_text,
            "system_directive": directive,
            "retrieved_count": len(results),
            "top_emotion": session.current_emotion,
        }

    # ----- post_llm_call：记录回复 + 检查阈值 -----
    def post_llm_call(
        self,
        session_id: str,
        assistant_message: str,
        enable_fact_check: bool = True,
    ) -> dict[str, Any]:
        """
        LLM 调用后钩子：记录回复，检查是否需要压缩。

        Returns:
            包含 fact_check_warnings, compressed 的字典
        """
        session = self.get_or_create_session(session_id)
        session.add_turn("assistant", assistant_message)

        result: dict[str, Any] = {"fact_check_warnings": [], "compressed": False}

        # 事实校验
        if enable_fact_check and session.last_control_signal is not None:
            warnings = fact_check(assistant_message, session.last_control_signal)
            result["fact_check_warnings"] = warnings

        # 检查是否达到压缩阈值
        if session.uncompressed_turns >= session.compression_threshold:
            self._trigger_compression(session)
            result["compressed"] = True

        return result

    # ----- on_session_end：异步压缩剩余对话 -----
    def on_session_end(self, session_id: str) -> Memory | None:
        """
        会话结束钩子：压缩剩余未压缩的对话。

        同步执行（可改为异步线程）。
        """
        session = self.sessions.get(session_id)
        if session is None:
            return None

        if session.uncompressed_turns == 0:
            return None

        memory, _ = self._do_compress(session)
        self.store.add(memory)

        # 清理会话
        session.uncompressed_turns = 0

        return memory

    def on_session_end_async(self, session_id: str) -> threading.Thread:
        """
        异步压缩：在后台线程中执行压缩，不阻塞会话结束。

        适用于 on_session_end 不能阻塞的场景。
        """
        thread = threading.Thread(
            target=self._async_compress_wrapper,
            args=(session_id,),
            daemon=True,
        )
        thread.start()
        return thread

    def _async_compress_wrapper(self, session_id: str) -> None:
        """异步压缩的包装器。"""
        try:
            self.on_session_end(session_id)
        except Exception as e:
            print(f"[hooks] 异步压缩失败 session={session_id}: {e}")

    # ----- 内部：触发压缩 -----
    def _trigger_compression(self, session: SessionContext) -> Memory | None:
        """达到阈值时触发压缩。"""
        if session.uncompressed_turns == 0:
            return None
        memory, _ = self._do_compress(session)
        self.store.add(memory)
        session.uncompressed_turns = 0
        return memory

    def _do_compress(self, session: SessionContext) -> tuple[Memory, str]:
        """执行压缩。"""
        # 取未压缩的部分
        uncompressed = session.history[-session.uncompressed_turns:] if session.uncompressed_turns <= len(session.history) else session.history

        memory, inject_text = compress_conversation(
            uncompressed,
            use_llm=False,  # 离线模式用规则式
            n_turns=session.compression_threshold,
        )
        return memory, inject_text

    # ----- 遗忘策略 -----
    def run_decay(self, threshold: float = 0.3) -> list[str]:
        """
        执行遗忘策略：删除低重要性日常记忆。

        策略（方案包 2.4 节）：
          - 高情感强度、高重要性、未解决项优先保留
          - 低强度日常记忆按时间衰减或合并
        """
        return self.store.apply_decay(threshold)

    # ----- 分层聚合 -----
    def run_aggregation(
        self, date_str: str, memories: list[Memory] | None = None
    ) -> dict[str, Any]:
        """
        执行分层聚合：日摘要 → 周情感章节 → 月事实骨架。

        按方案包 2.4 节："日摘要—周情感章节—月事实骨架"分层聚合。
        """
        if memories is None:
            memories = self.store.list_all()

        if not memories:
            return {}

        return self.store.aggregate_daily(date_str, memories)


# ---------------------------------------------------------------------------
# 便捷函数：完整对话循环
# ---------------------------------------------------------------------------
def run_conversation_turn(
    hooks: HookManager,
    session_id: str,
    user_message: str,
    assistant_response: str,
) -> dict[str, Any]:
    """
    执行一轮完整的对话循环：

    用户消息 → [pre_llm_call: 召回+注入] → LLM → [post_llm_call: 校验+压缩检查]

    这是一个便利函数，展示完整的钩子流程。
    实际使用时 LLM 调用由外部完成。
    """
    pre = hooks.pre_llm_call(session_id, user_message)
    post = hooks.post_llm_call(session_id, assistant_response)

    return {
        "pre_llm": pre,
        "post_llm": post,
        "session_uncompressed_turns": hooks.sessions[session_id].uncompressed_turns,
    }


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(tmpdir)
        hooks = HookManager(store, compression_threshold=3)  # 测试用小阈值

        session = "test-session-1"

        # 模拟多轮对话
        turns = [
            ("项目截止日期快到了，我很焦虑，怕做不完。", "我理解你的压力，我们来看看能做什么。"),
            ("同事A说他会帮我处理数据，我保证下周三前交报告。", "好的，下周三是关键节点。"),
            ("演示格式还没确定，待定。", "我们可以先准备内容，格式后续再定。"),
        ]

        for user_msg, assistant_msg in turns:
            result = run_conversation_turn(hooks, session, user_msg, assistant_msg)
            print(
                f"用户: {user_msg[:30]}... | "
                f"召回{result['pre_llm']['retrieved_count']}条 | "
                f"已压缩: {result['post_llm']['compressed']}"
            )

        print("\n库中记忆数:", store.stats()["total_memories"])

        # 结束会话，压缩剩余
        final_mem = hooks.on_session_end(session)
        if final_mem:
            print("会话结束压缩记忆:", final_mem.memory_id[:20])
        print("最终记忆数:", store.stats()["total_memories"])
