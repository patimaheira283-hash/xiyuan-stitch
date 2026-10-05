@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\Setup.ps1" -Neural
  if errorlevel 1 goto failed
)
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=4
".venv\Scripts\python.exe" -m xiyuan_mvp.gui
if errorlevel 1 goto failed
exit /b 0
:failed
echo 启动失败，请查看以上信息和 docs 中的安装说明。
pause
exit /b 1
