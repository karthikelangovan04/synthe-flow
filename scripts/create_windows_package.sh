#!/bin/bash

# Windows Package Creation Script
echo "📦 Creating Windows Package for Synthetic Data Platform"
echo "========================================================"

# Set package name with timestamp
PACKAGE_NAME="synthetic_data_platform_windows_$(date +%Y%m%d_%H%M%S)"
PACKAGE_DIR="${PACKAGE_NAME}"

echo "📁 Creating package directory: $PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

echo "📋 Copying application files..."

# Copy main application structure
cp -r src/ "$PACKAGE_DIR/"
cp -r backend/ "$PACKAGE_DIR/"
cp -r enhanced_backend/ "$PACKAGE_DIR/"
cp -r supabase/ "$PACKAGE_DIR/"
cp -r public/ "$PACKAGE_DIR/"

# Copy configuration files
cp package.json "$PACKAGE_DIR/"
cp package-lock.json "$PACKAGE_DIR/"
cp tsconfig*.json "$PACKAGE_DIR/"
cp vite.config.ts "$PACKAGE_DIR/"
cp tailwind.config.ts "$PACKAGE_DIR/"
cp postcss.config.js "$PACKAGE_DIR/"
cp eslint.config.js "$PACKAGE_DIR/"
cp components.json "$PACKAGE_DIR/"

# Copy environment files
cp env.example "$PACKAGE_DIR/"
cp env.development "$PACKAGE_DIR/"
cp env.production "$PACKAGE_DIR/"

# Copy startup scripts
cp start_*.sh "$PACKAGE_DIR/"
cp setup_env.sh "$PACKAGE_DIR/"

# Copy documentation
cp README.md "$PACKAGE_DIR/"
cp ENVIRONMENT_SETUP.md "$PACKAGE_DIR/"
cp SECURITY_CHECKLIST.md "$PACKAGE_DIR/"
cp QUICK_START_GUIDE.md "$PACKAGE_DIR/"
cp ENHANCED_SDV_DOCUMENTATION.md "$PACKAGE_DIR/"

# Copy CSV files from main path
echo "📊 Copying CSV files..."
find . -name "*.csv" -not -path "./backend/venv/*" -not -path "./enhanced_backend/venv/*" -not -path "./backend/sdv_service/uploads/*" -not -path "./backend/uploads/*" -exec cp {} "$PACKAGE_DIR/" \;

# Copy JSON files
find . -name "*.json" -not -path "./backend/venv/*" -not -path "./enhanced_backend/venv/*" -not -path "./node_modules/*" -exec cp {} "$PACKAGE_DIR/" \;

# Copy Python files
find . -name "*.py" -not -path "./backend/venv/*" -not -path "./enhanced_backend/venv/*" -exec cp {} "$PACKAGE_DIR/" \;

# Copy test files
cp test_*.py "$PACKAGE_DIR/" 2>/dev/null || true
cp test_env.html "$PACKAGE_DIR/" 2>/dev/null || true

# Create Windows-specific files
echo "🪟 Creating Windows-specific files..."

# Create Windows batch file for setup
cat > "$PACKAGE_DIR/setup_windows.bat" << 'EOF'
@echo off
echo ========================================
echo Synthetic Data Platform - Windows Setup
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

echo.
echo Creating virtual environments...

echo Creating backend virtual environment...
cd backend
python -m venv venv
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
cd ..

echo Creating enhanced backend virtual environment...
cd enhanced_backend
python -m venv venv
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
cd ..

echo.
echo Installing frontend dependencies...
npm install

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To start the application:
echo 1. Run: start_windows.bat
echo 2. Or run individual scripts:
echo    - start_frontend.bat
echo    - start_backend.bat
echo    - start_enhanced_backend.bat
echo.
pause
EOF

# Create Windows startup script
cat > "$PACKAGE_DIR/start_windows.bat" << 'EOF'
@echo off
echo ========================================
echo Starting Synthetic Data Platform
echo ========================================
echo.

echo Loading environment variables...
if exist env.development (
    for /f "tokens=1,* delims==" %%a in (env.development) do (
        if not "%%a"=="" if not "%%a:~0,1%"=="#" set %%a=%%b
    )
) else (
    echo WARNING: env.development not found, using default values
)

echo.
echo Starting Backend Services...

echo Starting Original Backend...
start "Backend" cmd /k "cd backend && venv\Scripts\activate.bat && cd sdv_service && python -m uvicorn main:app --host 127.0.0.1 --port 8002"

echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo Starting Frontend...
start "Frontend" cmd /k "npm run dev"

echo.
echo ========================================
echo Services started!
echo ========================================
echo Frontend: http://localhost:3000
echo Backend: http://localhost:8002
echo API Health: http://localhost:8002/api/sdv/health
echo.
echo Press any key to exit this window...
pause >nul
EOF

# Create individual service startup scripts
cat > "$PACKAGE_DIR/start_frontend.bat" << 'EOF'
@echo off
echo Starting Frontend...
npm run dev
pause
EOF

cat > "$PACKAGE_DIR/start_backend.bat" << 'EOF'
@echo off
echo Starting Backend...
cd backend
venv\Scripts\activate.bat
cd sdv_service
python -m uvicorn main:app --host 127.0.0.1 --port 8002
pause
EOF

cat > "$PACKAGE_DIR/start_enhanced_backend.bat" << 'EOF'
@echo off
echo Starting Enhanced Backend...
cd enhanced_backend
venv\Scripts\activate.bat
cd enhanced_sdv_service
python -m uvicorn main:app --host 127.0.0.1 --port 8003
pause
EOF

# Create PowerShell version for better compatibility
cat > "$PACKAGE_DIR/setup_windows.ps1" << 'EOF'
Write-Host "========================================" -ForegroundColor Green
Write-Host "Synthetic Data Platform - Windows Setup" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Node.js is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Node.js from https://nodejs.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Creating virtual environments..." -ForegroundColor Yellow

# Backend venv
Write-Host "Creating backend virtual environment..." -ForegroundColor Cyan
Set-Location backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
Set-Location ..

# Enhanced backend venv
Write-Host "Creating enhanced backend virtual environment..." -ForegroundColor Cyan
Set-Location enhanced_backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
Set-Location ..

Write-Host ""
Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
npm install

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application:" -ForegroundColor Cyan
Write-Host "1. Run: .\start_windows.bat" -ForegroundColor White
Write-Host "2. Or run individual scripts:" -ForegroundColor White
Write-Host "   - .\start_frontend.bat" -ForegroundColor White
Write-Host "   - .\start_backend.bat" -ForegroundColor White
Write-Host "   - .\start_enhanced_backend.bat" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to continue"
EOF

# Create README for Windows
cat > "$PACKAGE_DIR/README_WINDOWS.md" << 'EOF'
# Synthetic Data Platform - Windows Setup

## Prerequisites

1. **Python 3.8+**: Download from https://python.org
2. **Node.js**: Download from https://nodejs.org
3. **Git** (optional): Download from https://git-scm.com

## Quick Setup

### Option 1: Automatic Setup (Recommended)
```cmd
setup_windows.bat
```

### Option 2: PowerShell Setup
```powershell
.\setup_windows.ps1
```

## Manual Setup

### 1. Install Python Dependencies
```cmd
# Backend
cd backend
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
cd ..

# Enhanced Backend
cd enhanced_backend
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
cd ..
```

### 2. Install Frontend Dependencies
```cmd
npm install
```

## Starting the Application

### Option 1: Start All Services
```cmd
start_windows.bat
```

### Option 2: Start Individual Services
```cmd
# Frontend
start_frontend.bat

# Backend
start_backend.bat

# Enhanced Backend
start_enhanced_backend.bat
```

## Access Points

- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8002
- **Enhanced Backend**: http://localhost:8003
- **API Health**: http://localhost:8002/api/sdv/health

## Environment Configuration

1. Copy `env.example` to `env.development`
2. Update with your Supabase credentials
3. Update API URLs if needed

## Troubleshooting

### Port Already in Use
```cmd
# Find processes using ports
netstat -ano | findstr :8002
netstat -ano | findstr :3000

# Kill process by PID
taskkill /PID <PID> /F
```

### Virtual Environment Issues
```cmd
# Recreate virtual environment
rmdir /s backend\venv
rmdir /s enhanced_backend\venv
setup_windows.bat
```

### Node.js Issues
```cmd
# Clear npm cache
npm cache clean --force
npm install
```

## Development

- Frontend code: `src/`
- Backend code: `backend/sdv_service/`
- Enhanced backend: `enhanced_backend/enhanced_sdv_service/`
- Environment files: `env.*`

## Support

For issues, check:
1. `ENVIRONMENT_SETUP.md`
2. `SECURITY_CHECKLIST.md`
3. `README.md`
EOF

# Create .gitignore for Windows package
cat > "$PACKAGE_DIR/.gitignore" << 'EOF'
# Virtual environments
venv/
env/
ENV/
enhanced_backend/venv/
backend/venv/

# Python cache files
__pycache__/
*.py[cod]
*$py.class
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/

# Temporary files
*.tmp
*.temp

# Node modules (if any)
node_modules/

# Build outputs
dist/
build/

# Local configuration files
.env.local
.env.development.local
.env.test.local
.env.production.local

# Environment files (protect secrets)
env.development
env.production

# Upload directories
backend/sdv_service/uploads/
backend/uploads/

# Generated files and archives
*.zip
enterprise_synthetic_data_platform.zip

# Test and demo files
test_env.html
*.ipynb

# Temporary files
requirements_updated.txt
Enhanced_Sdv_Requiements
EOF

# Remove venv directories from the package
echo "🧹 Cleaning up virtual environments..."
rm -rf "$PACKAGE_DIR/backend/venv"
rm -rf "$PACKAGE_DIR/enhanced_backend/venv"
rm -rf "$PACKAGE_DIR/node_modules" 2>/dev/null || true

# Create zip file
echo "📦 Creating zip file..."
zip -r "${PACKAGE_NAME}.zip" "$PACKAGE_DIR" -x "*.DS_Store" "*/__pycache__/*" "*/node_modules/*"

# Clean up temporary directory
rm -rf "$PACKAGE_DIR"

echo ""
echo "🎉 Windows Package Created Successfully!"
echo "========================================"
echo "📦 Package: ${PACKAGE_NAME}.zip"
echo "📁 Size: $(du -h "${PACKAGE_NAME}.zip" | cut -f1)"
echo ""
echo "📋 Package Contents:"
echo "   ✅ Application source code"
echo "   ✅ Configuration files"
echo "   ✅ Environment templates"
echo "   ✅ CSV data files"
echo "   ✅ Windows setup scripts"
echo "   ✅ Documentation"
echo "   ❌ Virtual environments (excluded)"
echo "   ❌ Node modules (excluded)"
echo ""
echo "🚀 Next Steps:"
echo "   1. Transfer ${PACKAGE_NAME}.zip to Windows machine"
echo "   2. Extract the zip file"
echo "   3. Run setup_windows.bat"
echo "   4. Start development with start_windows.bat"
echo ""
echo "📖 For detailed instructions, see README_WINDOWS.md in the package" 