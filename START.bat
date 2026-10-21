@echo off
rem Launches the poly-qwen-image server. Extra arguments are passed to "qwen-image serve".
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found at .venv. See README.md for setup.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
qwen-image serve --port 7860 %*
pause
