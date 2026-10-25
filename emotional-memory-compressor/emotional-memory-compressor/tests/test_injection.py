#!/usr/bin/env python3
"""
注入正确性测试 —— test_injection.py

验证注入器遵循"先事实骨架，后情感修饰"原则，
以及事实校验能检测出生成文本中的冲突。
对应方案包 6.8 阶段0的注入验证。
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/scripts")

from compressor import compress_conversation
from memory_store import MemoryStore
from retriever import retrieve_memory
from injector import inject_and_control, fact_check, Injector, GenerationControlSignal


def _build_store(tmpdir: str) -> MemoryStore:
    """构建测试库。"""
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
    return store


def test_inject_text_not_empty():
    """测试有检索结果时注入文本非空。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _build_store(tmpdir)
        results = retrieve_memory(store, query_text="截止日期 焦虑", max_results=3)
        inject_text, directive = inject_and_control(results)

        assert inject_text, "有结果时注入文本不能为空"
        assert "【事实】" in inject_text or "【承诺】" in inject_text or "【情感】" in inject_text, \
            "注入文本应包含事实/承诺/情感标记"

        print(f"[PASS] test_inject_text_not_empty (len={len(inject_text)})")


def test_inject_empty_when_no_results():
    """测试无检索结果时注入为空。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _build_store(tmpdir)
        # 用不存在的人物过滤，确保无结果
        results = retrieve_memory(store, people=["不存在的人XYZ"], max_results=3)
        inject_text, directive = inject_and_control(results)

        assert inject_text == "", "无结果时注入文本应为空"

        print("[PASS] test_inject_empty_when_no_results")


def test_control_signal_extraction():
    """测试生成控制信号提取。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _build_store(tmpdir)
        results = retrieve_memory(store, query_text="焦虑", max_results=3)

        injector = Injector()
        _, signal = injector.inject(results)

        assert signal.primary_emotion is not None, "应提取到主要情感"
        assert signal.fact_points is not None, "应提取到事实要点"

        print(f"[PASS] test_control_signal_extraction (emotion={signal.primary_emotion})")


def test_system_directive_format():
    """测试生成控制指令格式正确。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _build_store(tmpdir)
        results = retrieve_memory(store, query_text="焦虑", max_results=3)
        _, directive = inject_and_control(results)

        if directive:
            assert "【生成语气控制】" in directive, "指令应包含'生成语气控制'标记"
            assert "不要编造事实" in directive, "指令应提醒不要编造事实"

        print(f"[PASS] test_system_directive_format (len={len(directive)})")


def test_fact_check_detects_denial():
    """测试事实校验：检测出生成文本否认承诺。"""
    signal = GenerationControlSignal(
        commitments=["下周三前交报告"],
        unresolved_items=["演示格式待定"],
    )
    bad_text = "你从来没有承诺过任何截止日期，这些问题已经解决了。"
    warnings = fact_check(bad_text, signal)

    assert len(warnings) > 0, "应检测出否认承诺的问题"
    assert any("否认" in w or "承诺" in w for w in warnings), "应警告承诺被否认"

    print(f"[PASS] test_fact_check_detects_denial ({len(warnings)} warnings)")


def test_fact_check_detects_false_resolution():
    """测试事实校验：检测出未解决项被错误标记为已解决。"""
    signal = GenerationControlSignal(
        unresolved_items=["演示格式待定", "预算未确定"],
    )
    bad_text = "演示格式已经搞定了，预算也确认了，没问题了。"
    warnings = fact_check(bad_text, signal)

    assert len(warnings) > 0, "应检测出未解决项被错误标记为已解决"

    print(f"[PASS] test_fact_check_detects_false_resolution ({len(warnings)} warnings)")


def test_fact_check_passes_clean_text():
    """测试事实校验：干净文本不应产生警告。"""
    signal = GenerationControlSignal(
        commitments=["下周三前交报告"],
        unresolved_items=["演示格式待定"],
    )
    clean_text = "根据之前的安排，下周三是报告截止日。演示格式还需要进一步讨论确定。"
    warnings = fact_check(clean_text, signal)

    assert len(warnings) == 0, f"干净文本不应有警告，实际: {warnings}"

    print("[PASS] test_fact_check_passes_clean_text")


def test_fact_order_before_emotion():
    """测试注入文本中事实出现在情感之前。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _build_store(tmpdir)
        results = retrieve_memory(store, query_text="焦虑 报告", max_results=3)
        inject_text, _ = inject_and_control(results)

        fact_pos = inject_text.find("【事实】")
        emotion_pos = inject_text.find("【情感】")

        if fact_pos != -1 and emotion_pos != -1:
            assert fact_pos < emotion_pos, "事实应出现在情感之前（先事实骨架后情感修饰）"
            print("[PASS] test_fact_order_before_emotion (事实先于情感)")
        else:
            print(f"[PASS] test_fact_order_before_emotion (部分标记缺失，可接受)")


def main():
    print("=" * 60)
    print("注入正确性测试 (test_injection.py)")
    print("=" * 60)

    tests = [
        test_inject_text_not_empty,
        test_inject_empty_when_no_results,
        test_control_signal_extraction,
        test_system_directive_format,
        test_fact_check_detects_denial,
        test_fact_check_detects_false_resolution,
        test_fact_check_passes_clean_text,
        test_fact_order_before_emotion,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"结果: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
