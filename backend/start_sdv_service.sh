#!/bin/bash

# Activate SDV virtual environment
source /Users/karthike/GenAi/Synthetic_data/sdv/venv/bin/activate

# Install requirements if needed
pip install -r requirements.txt

# Start the SDV service
cd sdv_service
python main.py 