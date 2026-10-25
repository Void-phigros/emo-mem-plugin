#!/usr/bin/env python3
"""一条命令跑完全部示例和测试。不修改任何业务模块。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"

DEMOS = [
    "compressor.py",
    "memory_store.py",
    "retriever.py",
    "injector.py",
    "hooks.py",
]
TEST_FILES = [
    "test_compression.py",
    "test_retrieval.py",
    "test_injection.py",
]


def run(label: str, script: Path, cwd: Path, extra_env: dict[str, str] | None = None) -> int:
    print("\n" + "=" * 60, flush=True)
    print(label, flush=True)
    print("=" * 60, flush=True)
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run([sys.executable, str(script)], cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"[失败] {label} 退出码 {result.returncode}", flush=True)
    else:
        print(f"[完成] {label}", flush=True)
    return result.returncode


def main() -> int:
    if not SCRIPTS.is_dir():
        print("找不到 scripts 目录。请把本文件放在含 SKILL.md 的那一层。", flush=True)
        return 1

    failed = 0
    for name in DEMOS:
        failed += run(f"示例 {name}", SCRIPTS / name, SCRIPTS) != 0
    for name in TEST_FILES:
        failed += run(f"测试 {name}", TESTS / name, SCRIPTS, {"PYTHONPATH": str(SCRIPTS)}) != 0

    print("\n" + "=" * 60, flush=True)
    if failed:
        print(f"结束：{failed} 项失败。", flush=True)
        return 1
    print("结束：示例和测试全部跑完，没有失败。", flush=True)
    print("schema.py、两份 md、以及 .skill 不用运行。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
