#!/bin/bash

# Simple Startup Script - Starts services individually
echo "🚀 Starting Synthetic Data Platform (Simple Mode)"
echo "================================================="

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
check_port 8003 || exit 1
check_port 3000 || exit 1

echo ""
echo "🐍 Starting Original Backend (Port 8002)..."
cd backend
source venv/bin/activate

# Install/update numpy to compatible version
echo "📦 Updating NumPy to compatible version..."
pip install "numpy<2.0.0" --force-reinstall

cd sdv_service
echo "🚀 Starting backend service..."
python main.py &
BACKEND_PID=$!
cd ../..

echo "✅ Original Backend started (PID: $BACKEND_PID)"

# Wait a moment for backend to start
sleep 3

echo ""
echo "🧠 Starting Enhanced Backend (Port 8003)..."
cd enhanced_backend
source venv/bin/activate

# Install/update numpy to compatible version
echo "📦 Updating NumPy to compatible version..."
pip install "numpy<2.0.0" --force-reinstall

cd enhanced_sdv_service
echo "🚀 Starting enhanced backend service..."
python main.py &
ENHANCED_BACKEND_PID=$!
cd ../..

echo "✅ Enhanced Backend started (PID: $ENHANCED_BACKEND_PID)"

# Wait a moment for enhanced backend to start
sleep 3

echo ""
echo "⚛️  Starting Frontend (Port 3000)..."
npm run dev &
FRONTEND_PID=$!
echo "✅ Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "🎉 All services started successfully!"
echo ""
echo "📱 Access your application:"
echo "   Frontend: http://localhost:3000"
echo "   Original Backend: http://localhost:8002"
echo "   Enhanced Backend: http://localhost:8003"
echo "   API Docs: http://localhost:8003/docs"
echo ""
echo "🔍 Testing services..."
echo ""

# Test backend health
sleep 5
echo "Testing Original Backend..."
if curl -s http://localhost:8002/api/sdv/health > /dev/null; then
    echo "✅ Original Backend is healthy"
else
    echo "❌ Original Backend health check failed"
fi

echo "Testing Enhanced Backend..."
if curl -s http://localhost:8003/api/enhanced/health > /dev/null; then
    echo "✅ Enhanced Backend is healthy"
else
    echo "❌ Enhanced Backend health check failed"
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
    if [ ! -z "$ENHANCED_BACKEND_PID" ]; then
        kill $ENHANCED_BACKEND_PID 2>/dev/null
        echo "✅ Enhanced Backend stopped"
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