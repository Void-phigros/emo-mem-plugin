#!/bin/sh
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
  python3 run_all.py
else
  python run_all.py
fi
status=$?
echo
echo "按回车关闭窗口"
read -r _
exit "$status"
