# Frontend Integration Guide for Windows

## Overview

This guide helps you integrate the frontend with your existing running backend on Windows.

## Prerequisites

- ✅ **Backend is already running** (ports 8002 and 8003)
- ✅ **Node.js installed** (version 16+ recommended)
- ✅ **npm installed** (comes with Node.js)

## Quick Start

### Step 1: Check Your Current Setup
```cmd
check_frontend_windows.bat
```

This will verify:
- Essential frontend files exist
- Node.js is installed
- Directory structure is correct

### Step 2: Setup Frontend
```cmd
setup_frontend_windows.bat
```

This will:
- Check Node.js installation
- Verify essential files
- Install npm dependencies

### Step 3: Start Frontend
```cmd
start_frontend_only.bat
```

This will start the frontend development server.

## Manual Setup

### Step 1: Verify Essential Files

Make sure you have these files in your directory:

```
your-project/
├── src/
│   ├── App.tsx
│   ├── main.tsx
│   ├── index.css
│   ├── components/
│   ├── pages/
│   ├── hooks/
│   ├── lib/
│   └── integrations/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
└── tailwind.config.ts
```

### Step 2: Install Dependencies
```cmd
npm install
```

### Step 3: Start Development Server
```cmd
npm run dev
```

## Troubleshooting

### Issue: "src\App.tsx not found"
**Solution:** The frontend files are missing. You need to:
1. Extract the complete Windows package
2. Or copy the `src/` directory from the original project

### Issue: "package.json not found"
**Solution:** You're not in the correct directory. Navigate to the project root.

### Issue: "node_modules not found"
**Solution:** Run `npm install` to install dependencies.

### Issue: "Port 3000 already in use"
**Solution:**
```cmd
# Find the process using port 3000
netstat -ano | findstr :3000

# Kill the process
taskkill /PID <PID> /F
```

### Issue: Frontend can't connect to backend
**Solution:** 
1. Verify backend is running on ports 8002 and 8003
2. Check firewall settings
3. Ensure backend URLs are correct in frontend config

## File Structure Requirements

### Essential Files
- `src/App.tsx` - Main application component
- `src/main.tsx` - Application entry point
- `src/index.css` - Global styles
- `index.html` - HTML template
- `package.json` - Dependencies and scripts
- `vite.config.ts` - Vite configuration
- `tsconfig.json` - TypeScript configuration

### Required Directories
- `src/components/` - React components
- `src/pages/` - Page components
- `src/hooks/` - Custom React hooks
- `src/lib/` - Utility functions
- `src/integrations/` - External integrations

## Environment Configuration

### Backend URLs
The frontend expects these backend URLs:
- Original Backend: `http://localhost:8002`
- Enhanced Backend: `http://localhost:8003`

### Environment Variables
Create `env.development` file:
```
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

## Development Workflow

### Daily Development
1. Start backend services (if not already running)
2. Run `start_frontend_only.bat`
3. Make changes to frontend code
4. Changes will auto-reload

### Stopping Frontend
- Press `Ctrl+C` in the terminal
- Or close the terminal window

## Access Points

Once everything is running:
- **Frontend**: http://localhost:3000
- **Original Backend**: http://localhost:8002
- **Enhanced Backend**: http://localhost:8003

## Common Commands

```cmd
# Check frontend structure
check_frontend_windows.bat

# Setup frontend (install dependencies)
setup_frontend_windows.bat

# Start frontend only
start_frontend_only.bat

# Manual npm commands
npm install          # Install dependencies
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
```

## Support

If you encounter issues:
1. Run `check_frontend_windows.bat` to diagnose problems
2. Check that all essential files exist
3. Verify Node.js installation
4. Ensure backend is running
5. Check firewall and port settings 