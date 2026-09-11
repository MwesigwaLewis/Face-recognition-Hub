@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo ERROR: .venv not found.
 echo Put this folder inside: C:\Users\Lewis K\Desktop\Project X RTSP
 pause
 exit /b 1
)
".venv\Scripts\python.exe" cuda_dll_diagnostic.py
