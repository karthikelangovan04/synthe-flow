@echo off
echo ========================================
echo Starting Frontend Only
echo ========================================
echo.

echo Checking if node_modules exists...
if not exist "node_modules" (
    echo ERROR: node_modules not found!
    echo Please run setup_frontend_windows.bat first
    pause
    exit /b 1
)

echo Checking if package.json exists...
if not exist "package.json" (
    echo ERROR: package.json not found!
    echo Please make sure you're in the correct directory
    pause
    exit /b 1
)

echo.
echo Starting frontend development server...
echo Frontend will be available at: http://localhost:3000
echo.
echo Press Ctrl+C to stop the server
echo.

npm run dev

pause 