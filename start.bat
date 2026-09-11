@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM Lewiscrypt HUB — universal Windows launcher.
REM First run may require Internet access to install missing Python packages.
REM Later runs use the existing .venv and do not reinstall packages.

set "PYTHONEXE=python"
where python >nul 2>&1
if errorlevel 1 (
    echo LEWIS > Python was not found. Please install 64-bit Python 3.11, 3.12, or 3.13.
    pause
    exit /b 1
)

python lewis_setup.py
if errorlevel 1 (
    echo.
    echo Lewis could not start.
    pause
)
exit /b %errorlevel%
