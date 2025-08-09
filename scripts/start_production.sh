#!/bin/bash

# Production Startup Script for Azure Deployment
echo "🚀 Starting Synthetic Data Platform (Production Mode)"
echo "====================================================="

# Load environment variables from production file if it exists
if [ -f "env.production" ]; then
    echo "📋 Loading environment variables from env.production"
    export $(cat env.production | grep -v '^#' | xargs)
elif [ -f ".env.production" ]; then
    echo "📋 Loading environment variables from .env.production"
    export $(cat .env.production | grep -v '^#' | xargs)
else
    echo "⚠️  No production environment file found"
    echo "   Using environment variables from Azure DevOps/Azure App Service"
    echo "   Make sure these are set:"
    echo "   - VITE_API_BASE_URL"
    echo "   - VITE_ENHANCED_API_BASE_URL"
    echo "   - VITE_SUPABASE_URL"
    echo "   - VITE_SUPABASE_ANON_KEY"
fi

echo ""
echo "🔧 Configuration:"
echo "   Frontend API URL: $VITE_API_BASE_URL"
echo "   Enhanced API URL: $VITE_ENHANCED_API_BASE_URL"
echo "   Supabase URL: $VITE_SUPABASE_URL"
echo "   Environment: $NODE_ENV"
echo ""

# Validate required environment variables
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
    echo "Please set these variables in your Azure App Service configuration"
    exit 1
fi

echo "✅ All required environment variables are set"
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

# Check ports for local deployment (if not using Azure App Service)
if [ "$DEPLOYMENT_TYPE" != "azure" ]; then
    echo "🔍 Checking port availability..."
    check_port 8002 || exit 1
    check_port 3000 || exit 1
fi

echo ""
echo "🐍 Starting Backend Services..."

# Start backend if running locally
if [ "$DEPLOYMENT_TYPE" != "azure" ]; then
    echo "🚀 Starting backend service..."
    cd backend
    source venv/bin/activate
    cd sdv_service
    
    # Use production settings
    export BACKEND_HOST="0.0.0.0"
    export BACKEND_PORT="8002"
    
    uvicorn main:app --host 0.0.0.0 --port 8002 --workers 4 &
    BACKEND_PID=$!
    cd ../..
    echo "✅ Backend started (PID: $BACKEND_PID)"
    
    # Wait for backend to start
    sleep 5
    
    # Test backend health
    echo "🔍 Testing backend health..."
    if curl -s http://localhost:8002/api/sdv/health > /dev/null; then
        echo "✅ Backend is healthy"
    else
        echo "❌ Backend health check failed"
        exit 1
    fi
fi

echo ""
echo "⚛️  Starting Frontend..."

# Build frontend for production
echo "🔨 Building frontend for production..."
npm run build

if [ $? -eq 0 ]; then
    echo "✅ Frontend built successfully"
else
    echo "❌ Frontend build failed"
    exit 1
fi

# Start frontend server
if [ "$DEPLOYMENT_TYPE" != "azure" ]; then
    echo "🚀 Starting frontend server..."
    npm run preview &
    FRONTEND_PID=$!
    echo "✅ Frontend started (PID: $FRONTEND_PID)"
else
    echo "✅ Frontend ready for Azure App Service deployment"
fi

echo ""
echo "🎉 Production services started successfully!"
echo ""
echo "📱 Application Status:"
echo "   Frontend: Ready for deployment"
echo "   Backend: $([ ! -z "$BACKEND_PID" ] && echo "Running (PID: $BACKEND_PID)" || echo "Deployed to Azure")"
echo "   Environment: Production"
echo ""

if [ "$DEPLOYMENT_TYPE" != "azure" ]; then
    echo "🔍 Health Checks:"
    echo "   Backend: http://localhost:8002/api/sdv/health"
    echo "   Frontend: http://localhost:4173 (if running locally)"
    echo ""
    echo "Press Ctrl+C to stop all services"
    
    # Function to cleanup on exit
    cleanup() {
        echo ""
        echo "🛑 Shutting down production services..."
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
    
    # Wait for user to stop
    wait
else
    echo "🚀 Ready for Azure deployment!"
    echo "   Deploy the built frontend to Azure Static Web Apps"
    echo "   Deploy the backend to Azure App Service"
fi 