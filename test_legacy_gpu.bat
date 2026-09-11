@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo ==============================================================
echo LEWISCRYPT HUB - LEGACY CUDA RUNTIME TEST
echo ==============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Run this from the Lewiscrypt HUB project folder.
  echo The project .venv was not found.
  pause
  exit /b 1
)

.venv\Scripts\python.exe test_legacy_gpu.py
set "RC=%errorlevel%"

echo.
echo ==============================================================
if "%RC%"=="0" (
  echo PASS - GPU inference works through the legacy CUDA runtime.
) else (
  echo FAIL - GPU runtime test failed. The output above is the diagnosis.
)
echo ==============================================================
pause
exit /b %RC%
