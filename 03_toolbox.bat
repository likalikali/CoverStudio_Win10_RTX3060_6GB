@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
py -3.12 "%~dp0windows_cover_studio.py" toolbox
set "rc=%errorlevel%"
echo.
pause
exit /b %rc%
