#!/bin/bash

# Minimal Startup Script - Starts only essential services
echo "🚀 Starting Synthetic Data Platform (Minimal Mode)"
echo "=================================================="

# Load environment variables
if [ -f "env.development" ]; then
    echo "📋 Loading environment variables from env.development"
    export $(cat env.development | grep -v '^#' | xargs)
else
    echo "⚠️  env.development not found, using defaults"
    export VITE_API_BASE_URL="http://localhost:8002"
    export VITE_ENHANCED_API_BASE_URL="http://localhost:8003"
fi

echo ""
echo "🔧 Configuration:"
echo "   Frontend API URL: $VITE_API_BASE_URL"
echo "   Enhanced API URL: $VITE_ENHANCED_API_BASE_URL"
echo ""

# Function to check if port is available
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
echo "🐍 Starting Original Backend (Port 8002)..."
cd backend
source venv/bin/activate
cd sdv_service

# Start backend with uvicorn directly to avoid import issues
echo "🚀 Starting backend service with uvicorn..."
uvicorn main:app --host 0.0.0.0 --port 8002 &
BACKEND_PID=$!
cd ../..

echo "✅ Original Backend started (PID: $BACKEND_PID)"

# Wait a moment for backend to start
sleep 3

echo ""
echo "⚛️  Starting Frontend (Port 3000)..."
npm run dev &
FRONTEND_PID=$!
echo "✅ Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "🎉 Essential services started successfully!"
echo ""
echo "📱 Access your application:"
echo "   Frontend: http://localhost:3000"
echo "   Original Backend: http://localhost:8002"
echo "   API Health: http://localhost:8002/api/sdv/health"
echo ""
echo "⚠️  Note: Enhanced backend is disabled due to NumPy compatibility issues"
echo "   The basic synthetic data generation will still work."
echo ""

# Test backend health
sleep 5
echo "🔍 Testing backend health..."
if curl -s http://localhost:8002/api/sdv/health > /dev/null; then
    echo "✅ Original Backend is healthy"
else
    echo "❌ Original Backend health check failed"
fi

echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo "✅ Original Backend stopped"
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

# Wait for user to stop
wait 