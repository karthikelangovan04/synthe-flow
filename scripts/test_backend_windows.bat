@echo off
echo ========================================
echo Testing Backend Services on Windows
echo ========================================
echo.

echo Testing Backend Services...
echo.

echo ========================================
echo Testing Original Backend (Port 8002)
echo ========================================

echo Testing health endpoint...
curl -s http://localhost:8002/api/sdv/health
if errorlevel 1 (
    echo ERROR: Original Backend is not responding on port 8002
    echo Please make sure the backend is running
) else (
    echo SUCCESS: Original Backend is running on port 8002
)

echo.
echo ========================================
echo Testing Enhanced Backend (Port 8003)
echo ========================================

echo Testing health endpoint...
curl -s http://localhost:8003/api/sdv/health
if errorlevel 1 (
    echo ERROR: Enhanced Backend is not responding on port 8003
    echo Please make sure the enhanced backend is running
) else (
    echo SUCCESS: Enhanced Backend is running on port 8003
)

echo.
echo ========================================
echo Testing Complete
echo ========================================
echo.
echo If both services show SUCCESS, your backends are running correctly!
echo.
echo You can now:
echo   1. Start the frontend with: start_frontend.bat
echo   2. Or start everything with: start_windows.bat
echo.
pause 