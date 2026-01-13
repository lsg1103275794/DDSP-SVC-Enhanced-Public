@echo off
REM ===================================
REM DDSP-SVC-Enhanced Setup Script
REM ===================================

echo.
echo ======================================
echo   DDSP-SVC-Enhanced Setup
echo ======================================
echo.

REM Check Python
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python %PYTHON_VERSION% detected
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist ".venv" (
    echo Virtual environment already exists. Skipping...
) else (
    python -m venv .venv
    echo Virtual environment created
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
echo.

REM Install PyTorch
echo.
echo Installing PyTorch...
echo Please select your CUDA version:
echo   1) CUDA 11.8
echo   2) CUDA 12.1
echo   3) CPU only
echo.
set /p CUDA_CHOICE="Enter choice (1-3): "

if "%CUDA_CHOICE%"=="1" (
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if "%CUDA_CHOICE%"=="2" (
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else if "%CUDA_CHOICE%"=="3" (
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
) else (
    echo Invalid choice
    pause
    exit /b 1
)
echo PyTorch installed
echo.

REM Install core dependencies
echo Installing core dependencies...
pip install -r requirements.txt
echo Core dependencies installed
echo.

REM Install API dependencies
echo Installing API dependencies...
pip install -r requirements-api.txt
echo API dependencies installed
echo.

REM Install frontend dependencies
echo Installing frontend dependencies...
where npm >nul 2>&1
if errorlevel 1 (
    echo Warning: npm not found. Skipping frontend setup.
    echo Please install Node.js and run: cd web ^&^& npm install
) else (
    cd web
    call npm install
    cd ..
    echo Frontend dependencies installed
)
echo.

REM Create directory structure
echo Creating directory structure...
if not exist "data\train\audio" mkdir data\train\audio
if not exist "data\val\audio" mkdir data\val\audio
if not exist "storage\uploads" mkdir storage\uploads
if not exist "storage\processed" mkdir storage\processed
if not exist "storage\outputs" mkdir storage\outputs
echo Directories created
echo.

REM Download pretrained models reminder
echo.
echo ======================================
echo   IMPORTANT: Download Pretrained Models
echo ======================================
echo.
echo 1. ContentVec encoder:
echo    https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr
echo    Place in: pretrain\contentvec\
echo.
echo 2. NSF-HiFiGAN vocoder:
echo    https://github.com/openvpi/vocoders/releases
echo    Place in: pretrain\nsf_hifigan\
echo.
echo 3. RMVPE pitch extractor:
echo    https://github.com/yxlllc/RMVPE/releases
echo    Place in: pretrain\rmvpe\
echo.

echo.
echo ======================================
echo   Setup Complete!
echo ======================================
echo.
echo Next steps:
echo   1. Download pretrained models (see above)
echo   2. Prepare your dataset in data\train\audio\
echo   3. Run: start.bat
echo.
pause
