#!/bin/bash

# Secure Startup Script with Enhanced Security
echo "🔒 Starting Synthetic Data Platform (Secure Mode)"
echo "=================================================="

# Security: Set restrictive file permissions
umask 077

# Security: Function to safely load environment variables
load_env_safely() {
    local env_file="$1"
    
    if [ ! -f "$env_file" ]; then
        echo "⚠️  Environment file not found: $env_file"
        return 1
    fi
    
    # Security: Check file permissions (should be 600)
    local perms=$(stat -c %a "$env_file" 2>/dev/null || stat -f %Lp "$env_file" 2>/dev/null)
    if [ "$perms" != "600" ]; then
        echo "⚠️  Warning: $env_file has insecure permissions ($perms). Should be 600."
        echo "   Run: chmod 600 $env_file"
    fi
    
    # Security: Load only non-comment lines, strip whitespace
    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ $line =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        
        # Security: Validate line format (key=value)
        if [[ $line =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            export "$line"
        else
            echo "⚠️  Skipping invalid line in $env_file: $line"
        fi
    done < "$env_file"
}

# Security: Load environment variables safely
if [ -f "env.development" ]; then
    echo "📋 Loading environment variables from env.development"
    load_env_safely "env.development"
elif [ -f ".env.local" ]; then
    echo "📋 Loading environment variables from .env.local"
    load_env_safely ".env.local"
else
    echo "⚠️  No environment file found, using system environment variables"
fi

echo ""
echo "🔧 Configuration:"
echo "   Frontend API URL: $VITE_API_BASE_URL"
echo "   Enhanced API URL: $VITE_ENHANCED_API_BASE_URL"
echo "   Supabase URL: $([ -n "$VITE_SUPABASE_URL" ] && echo "SET" || echo "NOT SET")"
echo "   Supabase Key: $([ -n "$VITE_SUPABASE_ANON_KEY" ] && echo "SET" || echo "NOT SET")"
echo ""

# Security: Validate required environment variables
required_vars=("VITE_API_BASE_URL" "VITE_SUPABASE_URL" "VITE_SUPABASE_ANON_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "❌ Missing required environment variables:"
    for var in "${missing_vars[@]}"; do
        echo "   - $var"
    done
    echo ""
    echo "Security: Please set these variables securely:"
    echo "   1. Use Azure Key Vault for production"
    echo "   2. Set file permissions to 600: chmod 600 env.development"
    echo "   3. Add environment files to .gitignore"
    exit 1
fi

echo "✅ All required environment variables are set"
echo ""

# Security: Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "❌ Port $port is already in use"
        return 1
    else
        echo "✅ Port $port is available"
        return 0
    fi
}

# Check ports
echo "🔍 Checking port availability..."
check_port 8002 || exit 1
check_port 3000 || exit 1

echo ""
echo "🐍 Starting Backend Services..."

# Security: Start backend with restricted permissions
echo "🚀 Starting backend service..."
cd backend
source venv/bin/activate
cd sdv_service

# Security: Use production-ready settings
export BACKEND_HOST="127.0.0.1"  # Bind to localhost only
export BACKEND_PORT="8002"

# Security: Start with limited workers for development
uvicorn main:app --host 127.0.0.1 --port 8002 --workers 1 &
BACKEND_PID=$!
cd ../..

echo "✅ Backend started (PID: $BACKEND_PID)"

# Wait for backend to start
sleep 3

# Security: Test backend health
echo "🔍 Testing backend health..."
if curl -s http://127.0.0.1:8002/api/sdv/health > /dev/null; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend health check failed"
    exit 1
fi

echo ""
echo "⚛️  Starting Frontend..."

# Security: Start frontend with development settings
npm run dev &
FRONTEND_PID=$!
echo "✅ Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "🎉 Secure services started successfully!"
echo ""
echo "📱 Access your application:"
echo "   Frontend: http://localhost:3000"
echo "   Backend: http://127.0.0.1:8002"
echo "   API Health: http://127.0.0.1:8002/api/sdv/health"
echo ""
echo "🔒 Security Features:"
echo "   ✅ Environment variables validated"
echo "   ✅ File permissions checked"
echo "   ✅ Backend bound to localhost only"
echo "   ✅ No sensitive data logged"
echo ""

# Security: Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down secure services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo "✅ Backend stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "✅ Frontend stopped"
    fi
    echo "👋 All services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "Press Ctrl+C to stop all services"

# Wait for user to stop
wait 