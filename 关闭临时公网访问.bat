@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m gpt_fusion_mvp.share stop
pause
