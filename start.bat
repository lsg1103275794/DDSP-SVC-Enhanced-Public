@echo off
REM ===================================
REM DDSP-SVC-Enhanced Quick Start Script
REM ===================================

echo.
echo Starting DDSP-SVC-Enhanced...
echo.

REM Check if virtual environment exists
if not exist ".venv" if not exist "venv" (
    echo Warning: Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
if exist ".venv" (
    call .venv\Scripts\activate.bat
) else if exist "venv" (
    call venv\Scripts\activate.bat
)

echo Virtual environment activated
echo.

REM Start API backend in new window
echo Starting API backend...
start "DDSP-SVC API" cmd /k "cd api && uvicorn main:app --reload --host 0.0.0.0 --port 8000"

REM Wait for API to start
timeout /t 3 /nobreak >nul

REM Start frontend in new window
echo Starting web frontend...
start "DDSP-SVC Web" cmd /k "cd web && npm run dev"

echo.
echo DDSP-SVC-Enhanced is running!
echo.
echo API Backend:  http://localhost:8000
echo API Docs:     http://localhost:8000/docs
echo Web Frontend: http://localhost:5173
echo.
echo Close the command windows to stop services
echo.
pause
