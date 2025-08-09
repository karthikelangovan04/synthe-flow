# Windows Deployment Guide - Synthetic Data Platform

## Prerequisites

Before starting, ensure you have the following installed on your Windows machine:

1. **Python 3.8+**: Download from https://python.org
   - ⚠️ **IMPORTANT**: Check "Add Python to PATH" during installation
   - Verify installation: `python --version`

2. **Node.js**: Download from https://nodejs.org
   - Verify installation: `node --version`

3. **Git** (optional): Download from https://git-scm.com

## Quick Start (Recommended)

### Step 1: Extract the Package
1. Extract `synthetic_data_platform_windows_YYYYMMDD_HHMMSS.zip` to a directory
2. Open Command Prompt in the extracted directory

### Step 2: Run Automatic Setup
```cmd
setup_windows.bat
```

This will:
- Check Python and Node.js installations
- Create virtual environments for both backends
- Install all Python dependencies
- Install Node.js dependencies

### Step 3: Start the Application
```cmd
start_windows.bat
```

## Manual Setup (Step by Step)

### Step 1: Create Virtual Environments

#### Option A: Using the Dedicated Script
```cmd
create_venv_windows.bat
```

#### Option B: Manual Creation

**Backend Virtual Environment:**
```cmd
cd backend
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
cd ..
```

**Enhanced Backend Virtual Environment:**
```cmd
cd enhanced_backend
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
cd ..
```

### Step 2: Install Frontend Dependencies
```cmd
npm install
```

### Step 3: Run Backend Services

#### Option A: Using the Dedicated Script
```cmd
run_backend_windows.bat
```

#### Option B: Manual Start

**Start Original Backend:**
```cmd
cd backend
venv\Scripts\activate.bat
cd sdv_service
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

**Start Enhanced Backend:**
```cmd
cd enhanced_backend
venv\Scripts\activate.bat
cd enhanced_sdv_service
python -m uvicorn main:app --host 127.0.0.1 --port 8003
```

### Step 4: Start Frontend
```cmd
npm run dev
```

## Service Management

### Available Scripts

| Script | Purpose |
|--------|---------|
| `setup_windows.bat` | Complete setup (venv + dependencies) |
| `create_venv_windows.bat` | Create virtual environments only |
| `run_backend_windows.bat` | Start both backend services |
| `start_windows.bat` | Start all services (frontend + backends) |
| `start_frontend.bat` | Start frontend only |
| `start_backend.bat` | Start original backend only |
| `start_enhanced_backend.bat` | Start enhanced backend only |
| `test_backend_windows.bat` | Test if backends are running |

### Individual Service Commands

**Frontend:**
```cmd
npm run dev
```

**Original Backend:**
```cmd
cd backend && venv\Scripts\activate.bat && cd sdv_service && python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

**Enhanced Backend:**
```cmd
cd enhanced_backend && venv\Scripts\activate.bat && cd enhanced_sdv_service && python -m uvicorn main:app --host 127.0.0.1 --port 8003
```

## Access Points

Once all services are running:

- **Frontend Application**: http://localhost:3000
- **Original Backend API**: http://localhost:8002
- **Enhanced Backend API**: http://localhost:8003
- **Backend Health Check**: http://localhost:8002/api/sdv/health
- **Enhanced Backend Health**: http://localhost:8003/api/sdv/health

## Troubleshooting

### Common Issues

#### 1. Python Not Found
```
ERROR: Python is not installed or not in PATH
```
**Solution:**
- Reinstall Python with "Add Python to PATH" checked
- Or manually add Python to PATH environment variable

#### 2. Virtual Environment Creation Fails
```
ERROR: Failed to create virtual environment
```
**Solution:**
- Run Command Prompt as Administrator
- Check if antivirus is blocking the operation
- Ensure sufficient disk space

#### 3. Port Already in Use
```
ERROR: [Errno 10048] Only one usage of each socket address
```
**Solution:**
```cmd
# Find processes using the port
netstat -ano | findstr :8002
netstat -ano | findstr :8003
netstat -ano | findstr :3000

# Kill the process
taskkill /PID <PID> /F
```

#### 4. Dependencies Installation Fails
```
ERROR: Failed to install requirements
```
**Solution:**
- Update pip: `python -m pip install --upgrade pip`
- Try installing with `--user` flag
- Check internet connection
- Run as Administrator

#### 5. Services Not Starting
```
ERROR: Backend is not responding
```
**Solution:**
- Check if virtual environments exist
- Verify Python dependencies are installed
- Check firewall settings
- Look for error messages in the service windows

### Testing Services

Use the test script to verify services are running:
```cmd
test_backend_windows.bat
```

### Recreating Virtual Environments

If you need to recreate virtual environments:
```cmd
# Remove existing venv
rmdir /s backend\venv
rmdir /s enhanced_backend\venv

# Recreate
create_venv_windows.bat
```

## Development Workflow

### Daily Development
1. Start backends: `run_backend_windows.bat`
2. Start frontend: `start_frontend.bat`
3. Make changes to code
4. Services will auto-reload

### Stopping Services
- Close the command windows for each service
- Or press `Ctrl+C` in each service window

### Restarting Services
- Stop services (close windows)
- Run `run_backend_windows.bat` again
- Run `start_frontend.bat` again

## Environment Configuration

### Setting Up Environment Variables
1. Copy `env.example` to `env.development`
2. Update with your Supabase credentials:
   ```
   VITE_SUPABASE_URL=your_supabase_url
   VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
   ```

### Backend Configuration
- Backend config: `backend/sdv_service/main.py`
- Enhanced backend config: `enhanced_backend/enhanced_sdv_service/main.py`

## File Structure

```
synthetic_data_platform_windows_YYYYMMDD_HHMMSS/
├── src/                          # Frontend React application
├── backend/                      # Original backend
│   ├── venv/                     # Virtual environment (created during setup)
│   ├── sdv_service/              # Backend service code
│   └── requirements.txt          # Python dependencies
├── enhanced_backend/             # Enhanced backend
│   ├── venv/                     # Virtual environment (created during setup)
│   ├── enhanced_sdv_service/     # Enhanced backend service code
│   └── requirements.txt          # Python dependencies
├── supabase/                     # Database configuration
├── public/                       # Static assets
├── *.bat                         # Windows scripts
├── package.json                  # Node.js dependencies
└── README_WINDOWS.md            # This guide
```

## Support

If you encounter issues:
1. Check this guide first
2. Review `ENVIRONMENT_SETUP.md`
3. Check `SECURITY_CHECKLIST.md`
4. Verify all prerequisites are installed correctly
5. Run services as Administrator if needed

## Performance Tips

- Use SSD storage for better performance
- Close unnecessary applications
- Consider increasing Python memory limits if needed
- Monitor system resources during heavy operations 