@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
py -3.12 "%~dp0windows_cover_studio.py" install
set "rc=%errorlevel%"
if not "%rc%"=="0" (
    echo.
    echo [ERROR] Install failed with exit code %rc%.
)
echo.
pause
exit /b %rc%
