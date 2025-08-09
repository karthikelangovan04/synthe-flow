@echo off
echo ========================================
echo Running Synthetic Data Platform Backend
echo ========================================
echo.

echo Checking if virtual environments exist...

if not exist "backend\venv" (
    echo ERROR: Backend virtual environment not found!
    echo Please run create_venv_windows.bat first
    pause
    exit /b 1
)

if not exist "enhanced_backend\venv" (
    echo ERROR: Enhanced backend virtual environment not found!
    echo Please run create_venv_windows.bat first
    pause
    exit /b 1
)

echo Virtual environments found. Starting services...
echo.

echo ========================================
echo Starting Backend Services
echo ========================================

echo Starting Original Backend on port 8002...
start "Backend - Port 8002" cmd /k "cd backend && venv\Scripts\activate.bat && cd sdv_service && echo Backend starting on http://localhost:8002 && python -m uvicorn main:app --host 127.0.0.1 --port 8002"

echo Waiting for backend to start...
timeout /t 3 /nobreak >nul

echo Starting Enhanced Backend on port 8003...
start "Enhanced Backend - Port 8003" cmd /k "cd enhanced_backend && venv\Scripts\activate.bat && cd enhanced_sdv_service && echo Enhanced Backend starting on http://localhost:8003 && python -m uvicorn main:app --host 127.0.0.1 --port 8003"

echo Waiting for enhanced backend to start...
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo Services Started Successfully!
echo ========================================
echo.
echo Backend Services:
echo   - Original Backend: http://localhost:8002
echo   - Enhanced Backend: http://localhost:8003
echo.
echo API Endpoints:
echo   - Health Check: http://localhost:8002/api/sdv/health
echo   - Enhanced Health: http://localhost:8003/api/sdv/health
echo.
echo To stop services, close the command windows or press Ctrl+C in each window
echo.
echo Press any key to exit this window...
pause >nul 