#!/bin/bash

# Development Startup Script
# This script starts all services with proper environment configuration

echo "🚀 Starting Synthetic Data Platform Development Environment"
echo "=========================================================="

# Load environment variables
if [ -f "env.development" ]; then
    echo "📋 Loading environment variables from env.development"
    export $(cat env.development | grep -v '^#' | xargs)
else
    echo "⚠️  env.development not found, using defaults"
    export VITE_API_BASE_URL="http://localhost:8002"
    export VITE_ENHANCED_API_BASE_URL="http://localhost:8003"
fi

# Display configuration
echo ""
echo "🔧 Configuration:"
echo "   Frontend API URL: $VITE_API_BASE_URL"
echo "   Enhanced API URL: $VITE_ENHANCED_API_BASE_URL"
echo "   Supabase URL: $VITE_SUPABASE_URL"
echo ""

# Function to start backend service
start_backend() {
    echo "🐍 Starting Original Backend (Port 8002)..."
    cd backend
    source venv/bin/activate
    cd sdv_service
    python main.py &
    BACKEND_PID=$!
    cd ../..
    echo "✅ Original Backend started (PID: $BACKEND_PID)"
}

# Function to start enhanced backend service
start_enhanced_backend() {
    echo "🧠 Starting Enhanced Backend (Port 8003)..."
    cd enhanced_backend
    source venv/bin/activate
    cd enhanced_sdv_service
    python main.py &
    ENHANCED_BACKEND_PID=$!
    cd ../..
    echo "✅ Enhanced Backend started (PID: $ENHANCED_BACKEND_PID)"
}

# Function to start frontend
start_frontend() {
    echo "⚛️  Starting Frontend (Port 3000)..."
    npm run dev &
    FRONTEND_PID=$!
    echo "✅ Frontend started (PID: $FRONTEND_PID)"
}

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

# Start services
start_backend
sleep 2

start_enhanced_backend
sleep 2

start_frontend

echo ""
echo "🎉 All services started successfully!"
echo ""
echo "📱 Access your application:"
echo "   Frontend: http://localhost:3000"
echo "   Original Backend: http://localhost:8002"
echo "   Enhanced Backend: http://localhost:8003"
echo "   API Docs: http://localhost:8003/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user to stop
wait 