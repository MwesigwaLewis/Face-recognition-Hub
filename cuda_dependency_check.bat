@echo off
cd /d "%~dp0"

".venv\Scripts\python.exe" cuda_dependency_check.py

pause