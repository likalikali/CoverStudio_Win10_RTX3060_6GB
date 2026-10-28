@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_ENDPOINT=https://huggingface.co"
set "HF_HUB_DISABLE_XET=1"
set "HF_HUB_DOWNLOAD_TIMEOUT=120"
set "HF_HUB_ETAG_TIMEOUT=30"
py -3.12 "%~dp0windows_cover_studio.py" start --mode gpu
set "rc=%errorlevel%"
if not "%rc%"=="0" (
    echo.
    echo [ERROR] Server exited with code %rc%. Run 03_toolbox.bat for diagnostics.
)
echo.
pause
exit /b %rc%
