#!/bin/bash

# Activate enhanced virtual environment
source venv/bin/activate

# Install requirements if needed
#pip install -r requirements.txt

# Start the Enhanced SDV service
cd enhanced_sdv_service
echo "Starting Enhanced SDV Service on http://localhost:8003"
echo "API Documentation available at http://localhost:8003/docs"
echo "Enhanced endpoints:"
echo "  - /api/enhanced/generate - Generate synthetic data with neural models"
echo "  - /api/enhanced/upload/file - Upload files for enhanced processing"
echo "  - /api/enhanced/status/{session_id} - Check generation status"
echo "  - /api/enhanced/health - Health check"
echo "  - /api/enhanced/capabilities - Service capabilities"
uvicorn main:app --host 0.0.0.0 --port 8003 --reload 