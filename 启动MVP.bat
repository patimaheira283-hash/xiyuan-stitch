@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo 未找到项目虚拟环境，请先运行：python -m venv .venv --system-site-packages
  echo 然后运行：.venv\Scripts\python.exe -m pip install -e ".[gui]"
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m xiyuan_mvp.gui
if errorlevel 1 pause

