@echo off
echo ========================================
echo Creating Virtual Environments for Windows
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found: 
python --version

echo.
echo ========================================
echo Creating Backend Virtual Environment
echo ========================================

echo Creating backend virtual environment...
cd backend
if exist venv (
    echo Backend venv already exists. Removing old one...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create backend virtual environment
    pause
    exit /b 1
)

echo Activating backend virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate backend virtual environment
    pause
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing backend requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install backend requirements
    pause
    exit /b 1
)

echo Backend virtual environment created successfully!
cd ..

echo.
echo ========================================
echo Creating Enhanced Backend Virtual Environment
echo ========================================

echo Creating enhanced backend virtual environment...
cd enhanced_backend
if exist venv (
    echo Enhanced backend venv already exists. Removing old one...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create enhanced backend virtual environment
    pause
    exit /b 1
)

echo Activating enhanced backend virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate enhanced backend virtual environment
    pause
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing enhanced backend requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install enhanced backend requirements
    pause
    exit /b 1
)

echo Enhanced backend virtual environment created successfully!
cd ..

echo.
echo ========================================
echo Virtual Environments Created Successfully!
echo ========================================
echo.
echo Backend venv: backend\venv\
echo Enhanced Backend venv: enhanced_backend\venv\
echo.
echo To activate and use:
echo   Backend: cd backend && venv\Scripts\activate.bat
echo   Enhanced Backend: cd enhanced_backend && venv\Scripts\activate.bat
echo.
pause 