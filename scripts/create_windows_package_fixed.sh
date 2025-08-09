#!/bin/bash

# Windows Package Creation Script - Fixed Version
echo "📦 Creating Fixed Windows Package for Synthetic Data Platform"
echo "============================================================="

# Set package name with timestamp
PACKAGE_NAME="synthetic_data_platform_windows_fixed_$(date +%Y%m%d_%H%M%S)"
PACKAGE_DIR="${PACKAGE_NAME}"

echo "📁 Creating package directory: $PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

echo "📋 Copying application files..."

# Copy main application structure (excluding venv directories)
echo "📁 Copying src directory..."
cp -r src/ "$PACKAGE_DIR/"

echo "📁 Copying backend (excluding venv)..."
mkdir -p "$PACKAGE_DIR/backend"
cp -r backend/sdv_service/ "$PACKAGE_DIR/backend/"
cp backend/requirements.txt "$PACKAGE_DIR/backend/"
cp backend/start_sdv_service.sh "$PACKAGE_DIR/backend/"

echo "📁 Copying enhanced_backend (excluding venv)..."
mkdir -p "$PACKAGE_DIR/enhanced_backend"
cp -r enhanced_backend/enhanced_sdv_service/ "$PACKAGE_DIR/enhanced_backend/"
cp enhanced_backend/requirements.txt "$PACKAGE_DIR/enhanced_backend/"
cp enhanced_backend/start_enhanced_service.sh "$PACKAGE_DIR/enhanced_backend/"

echo "📁 Copying supabase directory..."
cp -r supabase/ "$PACKAGE_DIR/"

echo "📁 Copying public directory..."
cp -r public/ "$PACKAGE_DIR/"

# Copy configuration files
echo "⚙️ Copying configuration files..."
cp package.json "$PACKAGE_DIR/"
cp package-lock.json "$PACKAGE_DIR/"
cp tsconfig*.json "$PACKAGE_DIR/"
cp vite.config.ts "$PACKAGE_DIR/"
cp tailwind.config.ts "$PACKAGE_DIR/"
cp postcss.config.js "$PACKAGE_DIR/"
cp eslint.config.js "$PACKAGE_DIR/"
cp components.json "$PACKAGE_DIR/"

# Copy HTML entry point
echo "📄 Copying index.html..."
cp index.html "$PACKAGE_DIR/"

# Copy environment files
echo "🔧 Copying environment files..."
cp env.example "$PACKAGE_DIR/"
cp env.development "$PACKAGE_DIR/"
cp env.production "$PACKAGE_DIR/"

# Copy startup scripts
echo "🚀 Copying startup scripts..."
cp start_*.sh "$PACKAGE_DIR/"
cp setup_env.sh "$PACKAGE_DIR/"

# Copy Windows-specific helper scripts
echo "🪟 Copying Windows helper scripts..."
cp create_venv_windows.bat "$PACKAGE_DIR/"
cp run_backend_windows.bat "$PACKAGE_DIR/"
cp test_backend_windows.bat "$PACKAGE_DIR/"
cp check_frontend_windows.bat "$PACKAGE_DIR/"
cp setup_frontend_windows.bat "$PACKAGE_DIR/"
cp start_frontend_only.bat "$PACKAGE_DIR/"
cp WINDOWS_DEPLOYMENT_GUIDE.md "$PACKAGE_DIR/"
cp FRONTEND_INTEGRATION_GUIDE.md "$PACKAGE_DIR/"

# Copy documentation
echo "📚 Copying documentation..."
cp README.md "$PACKAGE_DIR/"
cp ENVIRONMENT_SETUP.md "$PACKAGE_DIR/"
cp SECURITY_CHECKLIST.md "$PACKAGE_DIR/"
cp QUICK_START_GUIDE.md "$PACKAGE_DIR/"
cp ENHANCED_SDV_DOCUMENTATION.md "$PACKAGE_DIR/"
cp LLM_INTEGRATION_README.md "$PACKAGE_DIR/"
cp SDV_INTEGRATION_README.md "$PACKAGE_DIR/"
cp DATA_SOURCE_CONNECTORS_README.md "$PACKAGE_DIR/"

# Copy CSV files from main path ONLY (excluding upload directories and src)
echo "📊 Copying CSV files..."
find . -maxdepth 1 -name "*.csv" -exec cp {} "$PACKAGE_DIR/" \;

# Copy JSON files from main path ONLY (excluding src)
echo "📄 Copying JSON files..."
find . -maxdepth 1 -name "*.json" -exec cp {} "$PACKAGE_DIR/" \;

# Copy Python files from main path ONLY (excluding src)
echo "🐍 Copying Python files..."
find . -maxdepth 1 -name "*.py" -exec cp {} "$PACKAGE_DIR/" \;

# Copy test files
echo "🧪 Copying test files..."
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
    echo Make sure to check "Add Python to PATH" during installation
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
python -m pip install --upgrade pip
pip install -r requirements.txt
cd ..

echo Creating enhanced backend virtual environment...
cd enhanced_backend
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
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

echo Starting Enhanced Backend...
start "Enhanced Backend" cmd /k "cd enhanced_backend && venv\Scripts\activate.bat && cd enhanced_sdv_service && python -m uvicorn main:app --host 127.0.0.1 --port 8003"

echo Waiting for enhanced backend to start...
timeout /t 5 /nobreak >nul

echo Starting Frontend...
start "Frontend" cmd /k "npm run dev"

echo.
echo ========================================
echo Services started!
echo ========================================
echo Frontend: http://localhost:3000
echo Backend: http://localhost:8002
echo Enhanced Backend: http://localhost:8003
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
enhanced_backend/enhanced_sdv_service/uploads/
enhanced_backend/uploads/

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

# Create zip file
echo "📦 Creating zip file..."
zip -r "${PACKAGE_NAME}.zip" "$PACKAGE_DIR" -x "*.DS_Store" "*/__pycache__/*"

# Clean up temporary directory
rm -rf "$PACKAGE_DIR"

echo ""
echo "🎉 Fixed Windows Package Created Successfully!"
echo "=============================================="
echo "📦 Package: ${PACKAGE_NAME}.zip"
echo "📁 Size: $(du -h "${PACKAGE_NAME}.zip" | cut -f1)"
echo ""
echo "📋 Package Contents:"
echo "   ✅ Application source code (with proper src/ structure)"
echo "   ✅ Configuration files"
echo "   ✅ Environment templates"
echo "   ✅ CSV data files"
echo "   ✅ Windows setup scripts"
echo "   ✅ Documentation"
echo "   ❌ Virtual environments (excluded - will be created on Windows)"
echo "   ❌ Node modules (excluded - will be installed on Windows)"
echo "   ❌ Upload directories (excluded - will be created automatically)"
echo ""
echo "🚀 Next Steps:"
echo "   1. Transfer ${PACKAGE_NAME}.zip to Windows machine"
echo "   2. Extract the zip file"
echo "   3. Run setup_windows.bat"
echo "   4. Start development with start_windows.bat"
echo ""
echo "📖 For detailed instructions, see FRONTEND_INTEGRATION_GUIDE.md in the package"
echo ""
echo "💡 Tips for Windows deployment:"
echo "   - Make sure Python is added to PATH during installation"
echo "   - Run setup scripts as Administrator if you encounter permission issues"
echo "   - Check Windows Defender/Firewall settings if services don't start" 