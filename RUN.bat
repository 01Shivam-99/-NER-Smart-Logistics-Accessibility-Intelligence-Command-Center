@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo NER Smart Logistics - Live GIS Command Center
echo ==================================================
echo.

chcp 65001 >nul
set PYTHONIOENCODING=utf-8

where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PY=python"
    ) else (
        echo Python was not found.
        echo Install Python 3.11 or newer from https://www.python.org/downloads/
        echo Make sure "Add Python to PATH" is enabled.
        pause
        exit /b 1
    )
)

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "..\.venv\Scripts\activate.bat" (
    call "..\.venv\Scripts\activate.bat"
) else (
    echo [1/3] Creating virtual environment...
    %PY% -m venv .venv
    if errorlevel 1 goto :error
    call ".venv\Scripts\activate.bat"
)

echo [2/3] Installing/updating required packages...
python -m pip install --upgrade pip --progress-bar off --quiet
if errorlevel 1 goto :error
python -m pip install -r requirements.txt --progress-bar off
if errorlevel 1 goto :error

echo [3/3] Starting Streamlit...
echo.
echo Browser: http://localhost:8501
start "NER Smart Logistics" http://localhost:8501
python -m streamlit run app.py --server.address localhost --server.port 8501
exit /b 0

:error
echo.
echo Something went wrong while setting up the application.
echo Check the messages above and make sure you have an internet connection.
pause
exit /b 1
