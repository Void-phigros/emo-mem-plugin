@echo off
cd /d "%~dp0"
where py >nul 2>nul && (
  py -3 run_all.py
) || (
  python run_all.py
)
echo.
pause
