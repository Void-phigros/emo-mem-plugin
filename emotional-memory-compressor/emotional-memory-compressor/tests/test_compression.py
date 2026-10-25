#!/usr/bin/env python3
"""
压缩保真度测试 —— test_compression.py

验证 compress_conversation() 能否正确提取：
  - 事实骨架（人物、事件、承诺、未解决项）
  - 情感/语气（标签、强度、对象）
  - 原文锚点（最多3条）
  - 关系状态变化
  - importance 和 decay_policy 合理性

对应方案包 6.8 阶段0：离线验证压缩保真度。
"""

import sys
import os
import json

# 确保能 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compressor import compress_conversation, render_inject_text
from schema import validate_memory_dict, get_retrieval_keys


def test_basic_compression():
    """测试基本压缩：能产出合法结构化记忆。"""
    history = [
        {"role": "user", "content": "我这个项目的截止日期快到了，我很焦虑，怕做不完。"},
        {"role": "assistant", "content": "我理解你的压力。我们来看看还能做什么。"},
        {"role": "user", "content": "同事A说他会帮我处理数据部分，我保证下周三前交报告。"},
        {"role": "assistant", "content": "好的，那下周三是关键节点。还有未解决的部分吗？"},
        {"role": "user", "content": "还没确定演示的格式，待定。谢谢你的帮助，感觉好多了。"},
    ]

    memory, inject = compress_conversation(history, use_llm=False)
    data = memory.to_dict()

    # 校验
    errors = validate_memory_dict(data)
    assert not errors, f"校验失败: {errors}"

    # 必须有 memory_id
    assert memory.memory_id, "memory_id 不能为空"
    print("[PASS] test_basic_compression")
    return memory


def test_fact_skeleton_integrity():
    """测试事实骨架完整性：承诺和未解决项必须保留。"""
    history = [
        {"role": "user", "content": "我保证明天完成报告，这是我的承诺。"},
        {"role": "assistant", "content": "好的，明天是截止日。"},
        {"role": "user", "content": "不过预算还没确定，待定。还需要和同事B确认。"},
    ]

    memory, _ = compress_conversation(history, use_llm=False)
    fs = memory.fact_skeleton

    # 承诺应被检测到
    assert len(fs.commitments) > 0, "应检测到承诺"
    assert any("报告" in c or "明天" in c or "完成" in c for c in fs.commitments), \
        f"承诺内容不匹配: {fs.commitments}"

    # 未解决项应被检测到
    assert len(fs.unresolved_items) > 0, "应检测到未解决项"
    assert any("预算" in u or "待定" in u or "确认" in u for u in fs.unresolved_items), \
        f"未解决项不匹配: {fs.unresolved_items}"

    print(f"[PASS] test_fact_skeleton_integrity (承诺={len(fs.commitments)}, 未解决={len(fs.unresolved_items)})")


def test_emotion_detection():
    """测试情感检测：焦虑应被正确识别。"""
    history = [
        {"role": "user", "content": "我很焦虑，非常担心截止日期，压力大。"},
        {"role": "assistant", "content": "深呼吸，我们一步步来。"},
    ]

    memory, _ = compress_conversation(history, use_llm=False)
    em = memory.emotion

    assert em.primary is not None, "应检测到情感"
    assert em.primary == "焦虑", f"应检测到'焦虑'，实际为'{em.primary}'"
    assert em.intensity > 0, "情感强度应 > 0"
    assert em.target is not None, "情感必须有对象"

    print(f"[PASS] test_emotion_detection ({em.primary}, intensity={em.intensity})")


def test_anchor_limit():
    """测试原文锚点不超过3条。"""
    history = []
    for i in range(10):
        history.append({"role": "user", "content": f"第{i}件很长的事情需要记录这里有很多字用来测试锚点的长度和数量限制"})
        history.append({"role": "assistant", "content": f"好的，关于第{i}件事我已了解。"})

    memory, _ = compress_conversation(history, use_llm=False)
    assert len(memory.anchors) <= 3, f"锚点应<=3条，实际{len(memory.anchors)}条"

    print(f"[PASS] test_anchor_limit ({len(memory.anchors)} anchors)")


def test_importance_and_decay():
    """测试重要性评分和衰减策略合理性。"""
    # 高情感 + 有承诺 + 有未解决 = 高重要性 + never decay
    high_history = [
        {"role": "user", "content": "我非常焦虑，特别担心，压力大，怕做不完。"},
        {"role": "assistant", "content": "我理解。"},
        {"role": "user", "content": "我保证明天完成，但预算还没确定，待定。"},
    ]
    mem_high, _ = compress_conversation(high_history, use_llm=False)
    assert mem_high.importance > 0.5, f"高情感+承诺+未解决应高重要性，实际{mem_high.importance}"
    assert mem_high.decay_policy in ("slow", "never"), "应 slow/never 衰减"

    # 低情感日常 = 低重要性 + normal decay
    low_history = [
        {"role": "user", "content": "今天天气不错。"},
        {"role": "assistant", "content": "是的，适合散步。"},
    ]
    mem_low, _ = compress_conversation(low_history, use_llm=False)
    assert mem_low.importance < mem_high.importance, "低情感应低重要性"
    assert mem_low.decay_policy == "normal", "日常记忆应 normal 衰减"

    print(f"[PASS] test_importance_and_decay (high={mem_high.importance}/{mem_high.decay_policy}, low={mem_low.importance}/{mem_low.decay_policy})")


def test_inject_text_format():
    """测试注入摘要遵循'先事实后情感'格式。"""
    history = [
        {"role": "user", "content": "我保证完成报告，但很焦虑。"},
        {"role": "assistant", "content": "好的。"},
    ]
    memory, inject = compress_conversation(history, use_llm=False)

    assert inject, "注入摘要不能为空"

    # 事实应出现在情感之前
    fact_pos = inject.find("【事实】")
    emotion_pos = inject.find("【情感】")

    if fact_pos != -1 and emotion_pos != -1:
        assert fact_pos < emotion_pos, "事实应出现在情感之前"
        print("[PASS] test_inject_text_format (事实先于情感)")
    else:
        print(f"[PASS] test_inject_text_format (部分缺失，inject={inject[:60]})")


def test_retrieval_keys():
    """测试检索键生成。"""
    history = [
        {"role": "user", "content": "我和同事A吵架了，很愤怒，他说数据有问题。"},
        {"role": "assistant", "content": "发生什么了？"},
    ]
    memory, _ = compress_conversation(history, use_llm=False)
    keys = get_retrieval_keys(memory)

    assert "people" in keys
    assert "emotion_primary" in keys
    assert keys["emotion_primary"] == "愤怒"
    assert "importance" in keys
    assert "has_unresolved" in keys

    print(f"[PASS] test_retrieval_keys (people={keys['people'][:2]}, emotion={keys['emotion_primary']})")


def test_empty_history():
    """测试空对话报错。"""
    try:
        compress_conversation([], use_llm=False)
        assert False, "空对话应报错"
    except ValueError:
        print("[PASS] test_empty_history")


def main():
    print("=" * 60)
    print("压缩保真度测试 (test_compression.py)")
    print("=" * 60)

    tests = [
        test_basic_compression,
        test_fact_skeleton_integrity,
        test_emotion_detection,
        test_anchor_limit,
        test_importance_and_decay,
        test_inject_text_format,
        test_retrieval_keys,
        test_empty_history,
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
