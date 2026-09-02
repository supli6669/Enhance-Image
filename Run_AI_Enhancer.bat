@echo off
title Wink Studio - AI Portrait & Image Restoration
cd /d "%~dp0"

echo ===================================================
echo    ✨ WINK STUDIO - AI PORTRAIT RESTORATION ✨
echo ===================================================
echo.
echo [1/2] Starting AI Engine & Streamlit Dashboard...

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless false
) else (
    python -m streamlit run app.py --server.port 8501 --server.headless false
)

pause
