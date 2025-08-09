@echo off
echo ========================================
echo Frontend Setup for Windows
echo ========================================
echo.

echo Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

echo Node.js found: 
node --version

echo.
echo Checking essential frontend files...

if not exist "package.json" (
    echo ERROR: package.json not found!
    echo Please make sure you're in the correct directory
    pause
    exit /b 1
)

if not exist "src\App.tsx" (
    echo ERROR: src\App.tsx not found!
    echo Frontend source files are missing
    pause
    exit /b 1
)

if not exist "index.html" (
    echo ERROR: index.html not found!
    echo Frontend entry point is missing
    pause
    exit /b 1
)

echo ✅ Essential files found
echo.

echo Installing frontend dependencies...
npm install

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Frontend Setup Complete!
echo ========================================
echo.
echo To start the frontend:
echo   npm run dev
echo.
echo The frontend will be available at:
echo   http://localhost:3000
echo.
echo Make sure your backend is running on:
echo   http://localhost:8002 (Original Backend)
echo   http://localhost:8003 (Enhanced Backend)
echo.
pause 