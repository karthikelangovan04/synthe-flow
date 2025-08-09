@echo off
echo ========================================
echo Frontend Structure Check for Windows
echo ========================================
echo.

echo Checking essential frontend files...
echo.

if exist "src\App.tsx" (
    echo ✅ src\App.tsx - Found
) else (
    echo ❌ src\App.tsx - Missing
)

if exist "src\main.tsx" (
    echo ✅ src\main.tsx - Found
) else (
    echo ❌ src\main.tsx - Missing
)

if exist "src\index.css" (
    echo ✅ src\index.css - Found
) else (
    echo ❌ src\index.css - Missing
)

if exist "index.html" (
    echo ✅ index.html - Found
) else (
    echo ❌ index.html - Missing
)

if exist "package.json" (
    echo ✅ package.json - Found
) else (
    echo ❌ package.json - Missing
)

if exist "vite.config.ts" (
    echo ✅ vite.config.ts - Found
) else (
    echo ❌ vite.config.ts - Missing
)

if exist "tsconfig.json" (
    echo ✅ tsconfig.json - Found
) else (
    echo ❌ tsconfig.json - Missing
)

if exist "tailwind.config.ts" (
    echo ✅ tailwind.config.ts - Found
) else (
    echo ❌ tailwind.config.ts - Missing
)

echo.
echo Checking src directory structure...
echo.

if exist "src\components" (
    echo ✅ src\components - Found
) else (
    echo ❌ src\components - Missing
)

if exist "src\pages" (
    echo ✅ src\pages - Found
) else (
    echo ❌ src\pages - Missing
)

if exist "src\hooks" (
    echo ✅ src\hooks - Found
) else (
    echo ❌ src\hooks - Missing
)

if exist "src\lib" (
    echo ✅ src\lib - Found
) else (
    echo ❌ src\lib - Missing
)

if exist "src\integrations" (
    echo ✅ src\integrations - Found
) else (
    echo ❌ src\integrations - Missing
)

echo.
echo Checking Node.js and npm...
echo.

node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js - Not installed or not in PATH
) else (
    echo ✅ Node.js - Found
    node --version
)

npm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ npm - Not installed or not in PATH
) else (
    echo ✅ npm - Found
    npm --version
)

echo.
echo Checking if node_modules exists...
if exist "node_modules" (
    echo ✅ node_modules - Found (dependencies installed)
) else (
    echo ❌ node_modules - Missing (run: npm install)
)

echo.
echo ========================================
echo Frontend Check Complete
echo ========================================
pause 