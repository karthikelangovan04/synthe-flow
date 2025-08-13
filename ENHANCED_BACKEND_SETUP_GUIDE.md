# 🚀 Enhanced Backend Setup & Execution Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Virtual Environment Setup](#virtual-environment-setup)
3. [Enhanced Backend Installation](#enhanced-backend-installation)
4. [Starting the Enhanced Backend Service](#starting-the-enhanced-backend-service)
5. [Using HR Datasets](#using-hr-datasets)
6. [Testing Multi-Relational Tables](#testing-multi-relational-tables)
7. [Output Analysis](#output-analysis)
8. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### System Requirements
- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: At least 2GB free space

### Required Software
- Python 3.8+
- pip (Python package installer)
- Git (for cloning the repository)

---

## 🐍 Virtual Environment Setup

### Step 1: Navigate to Enhanced Backend Directory
```bash
cd "/Users/karthike/Desktop/Vibe Coding/synthe-flow/enhanced_backend"
```

### Step 2: Activate Virtual Environment
```bash
# For macOS/Linux
source venv/bin/activate

# For Windows
venv\Scripts\activate
```

### Step 3: Verify Activation
```bash
# Check if virtual environment is active
which python
# Should show: /Users/karthike/Desktop/Vibe Coding/synthe-flow/enhanced_backend/venv/bin/python

# Check Python version
python --version
# Should show: Python 3.11.x
```

**Expected Output:**
```bash
(venv) karthike@Karthiks-MacBook-Pro enhanced_backend %
```

---

## 📦 Enhanced Backend Installation

### Step 1: Install Dependencies
```bash
# Make sure virtual environment is activated
pip install -r requirements.txt
```

### Step 2: Verify Key Dependencies
```bash
python -c "import fastapi, sdv, torch, pandas, numpy; print('✅ All dependencies installed successfully')"
```

**Expected Output:**
```bash
✅ All dependencies installed successfully
```

### Step 3: Install Missing Dependencies (if needed)
```bash
# If any dependencies are missing, install them individually
pip install sdv fastapi uvicorn pandas numpy scikit-learn torch
```

---

## 🚀 Starting the Enhanced Backend Service

### Method 1: Using the Start Script
```bash
# Navigate to enhanced backend directory
cd enhanced_backend

# Make script executable (macOS/Linux only)
chmod +x start_enhanced_service.sh

# Run the start script
./start_enhanced_service.sh
```

### Method 2: Manual Startup
```bash
# Navigate to enhanced backend directory
cd enhanced_backend

# Activate virtual environment
source venv/bin/activate

# Navigate to service directory
cd enhanced_sdv_service

# Start the service
python -m uvicorn main:app --host 0.0.0.0 --port 8003 --reload
```

### Step 3: Verify Service is Running
```bash
# Test health endpoint
curl http://localhost:8003/api/enhanced/health
```

**Expected Output:**
```json
{
  "status": "healthy",
  "service": "Enhanced SDV Enterprise Service",
  "version": "3.0.0",
  "timestamp": "2025-08-12T20:59:25"
}
```

---

## 📊 Using HR Datasets

### Available HR Datasets
The enhanced backend supports these **11 HR tables** with complex relationships:

| Table | Records | Description |
|-------|---------|-------------|
| `complex_hr_employees.csv` | 500 | Employee master data |
| `complex_hr_departments.csv` | 20 | Department information |
| `complex_hr_positions.csv` | 50 | Job positions |
| `complex_hr_salaries.csv` | 500 | Salary information |
| `complex_hr_performance.csv` | 500 | Performance reviews |
| `complex_hr_projects.csv` | 50 | Project data |
| `complex_hr_project_assignments.csv` | 273 | Employee-project assignments |
| `complex_hr_skills.csv` | 100 | Skills catalog |
| `complex_hr_employee_skills.csv` | 750 | Employee-skill mappings |
| `complex_hr_training.csv` | 30 | Training programs |
| `complex_hr_training_enrollments.csv` | 400 | Training enrollments |

### Complex Relationships
- **Employees ↔ Departments** (Many-to-One)
- **Employees ↔ Positions** (Many-to-One)
- **Employees ↔ Managers** (Self-referencing hierarchy)
- **Employees ↔ Projects** (Many-to-Many via assignments)
- **Employees ↔ Skills** (Many-to-Many via employee_skills)
- **Employees ↔ Training** (Many-to-Many via enrollments)

---

## 🧪 Testing Multi-Relational Tables

### Method 1: Using the Test Script
```bash
# Navigate to project root
cd "/Users/karthike/Desktop/Vibe Coding/synthe-flow"

# Run the test script
python test_enhanced_backend.py
```

**Expected Output:**
```bash
🧪 Testing Enhanced Backend with Complex HR Dataset
============================================================

1️⃣ Testing Health Check...
✅ Health Check: healthy
   Service: Enhanced SDV Enterprise Service
   Version: 3.0.0

2️⃣ Testing Capabilities...
✅ Capabilities Check:
   Max Tables: 100
   Max Relationships: 500
   Max Data Volume: 10M+ records
   Supported Formats: JSON, CSV, Excel, SQL, Parquet

3️⃣ Uploading Complex HR Dataset...
✅ Uploaded: complex_hr_employees.csv
✅ Uploaded: complex_hr_departments.csv
✅ Uploaded: complex_hr_positions.csv
✅ Uploaded: complex_hr_salaries.csv
✅ Uploaded: complex_hr_projects.csv
✅ Uploaded: complex_hr_project_assignments.csv

4️⃣ Testing Synthetic Data Generation...
🚀 Sending generation request...
✅ Generation Successful!
   Session ID: 26299d6f-790b-4399-84ab-d121d0a3a25f
   Status: processing
   Generation Time: 0.02 seconds

🎉 All tests passed! Enhanced backend is working correctly.
```

### Method 2: Manual API Testing
```bash
# Test capabilities
curl http://localhost:8003/api/enhanced/capabilities

# Upload a file
curl -X POST -F "file=@complex_hr_employees.csv" http://localhost:8003/api/enhanced/upload/file

# Generate synthetic data
curl -X POST -H "Content-Type: application/json" \
  -d '{"tables": [{"name": "employees", "columns": []}]}' \
  http://localhost:8003/api/enhanced/generate
```

---

## 📈 Output Analysis

### Output File Locations
```
📂 Enhanced Backend Output:
├── /enhanced_backend/enhanced_sdv_service/exports/
│   ├── {session_id}_employees_{timestamp}.json
│   ├── {session_id}_departments_{timestamp}.json
│   ├── {session_id}_positions_{timestamp}.json
│   ├── {session_id}_salaries_{timestamp}.json
│   ├── {session_id}_projects_{timestamp}.json
│   └── {session_id}_project_assignments_{timestamp}.json
```

### Analyzing Relational Integrity
```bash
# Run the relational integrity checker
python check_relational_integrity.py
```

**Expected Output:**
```bash
🚀 Starting Relational Integrity Analysis
============================================================
🔍 Loading session: 61afddf0-0800-4447-a3e8-307ed1bc4c68
✅ Loaded employees: 500 rows, 8 columns
✅ Loaded departments: 20 rows, 4 columns
✅ Loaded positions: 50 rows, 6 columns
✅ Loaded salaries: 500 rows, 6 columns
✅ Loaded projects: 50 rows, 9 columns
✅ Loaded project_assignments: 273 rows, 7 columns

🔑 Checking Primary Keys...
✅ departments: Primary key unique (20/20)
✅ positions: Primary key unique (50/50)
✅ projects: Primary key unique (50/50)

🔗 Checking Foreign Keys...
✅ employees.department_id -> departments.department_id: No orphaned keys
✅ employees.position_id -> positions.position_id: No orphaned keys

🔢 Checking Relationship Cardinality...
✅ Employee-Department: 20 departments have employees
✅ Employee-Position: 50 positions have employees
✅ Project Assignments: 500 employees, 50 projects

🎉 All relational integrity checks passed!
```

---

## 🔧 API Endpoints

### Core Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/enhanced/health` | GET | Service health check |
| `/api/enhanced/capabilities` | GET | Service capabilities |
| `/api/enhanced/upload/file` | POST | Upload data files |
| `/api/enhanced/generate` | POST | Generate synthetic data |
| `/api/enhanced/status/{session_id}` | GET | Check generation status |

### Example API Usage
```bash
# Health check
curl http://localhost:8003/api/enhanced/health

# Get capabilities
curl http://localhost:8003/api/enhanced/capabilities

# Upload HR dataset
curl -X POST -F "file=@complex_hr_employees.csv" \
  http://localhost:8003/api/enhanced/upload/file

# Generate synthetic data
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "tables": [
      {
        "name": "employees",
        "description": "HR Employees data",
        "columns": [
          {"name": "employee_id", "data_type": "int64", "is_primary_key": true},
          {"name": "first_name", "data_type": "object"},
          {"name": "last_name", "data_type": "object"},
          {"name": "email", "data_type": "object"},
          {"name": "department_id", "data_type": "int64"},
          {"name": "position_id", "data_type": "int64"},
          {"name": "manager_id", "data_type": "int64"},
          {"name": "hire_date", "data_type": "object"}
        ]
      }
    ],
    "relationships": [
      {
        "source_table": "employees",
        "source_column": "department_id",
        "target_table": "departments",
        "target_column": "department_id",
        "relationship_type": "many-to-one"
      }
    ]
  }' \
  http://localhost:8003/api/enhanced/generate
```

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Issue 1: Virtual Environment Not Activated
**Symptoms:**
```bash
ModuleNotFoundError: No module named 'fastapi'
```

**Solution:**
```bash
# Navigate to enhanced backend directory
cd enhanced_backend

# Activate virtual environment
source venv/bin/activate

# Verify activation
which python
# Should show path ending with /venv/bin/python
```

#### Issue 2: Port Already in Use
**Symptoms:**
```bash
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find process using port 8003
lsof -i :8003

# Kill the process
kill -9 <PID>

# Or use a different port
python -m uvicorn main:app --host 0.0.0.0 --port 8004 --reload
```

#### Issue 3: Missing Dependencies
**Symptoms:**
```bash
ModuleNotFoundError: No module named 'sdv'
```

**Solution:**
```bash
# Activate virtual environment
source venv/bin/activate

# Install missing dependencies
pip install sdv fastapi uvicorn pandas numpy scikit-learn torch

# Or reinstall all requirements
pip install -r requirements.txt
```

#### Issue 4: Service Won't Start
**Symptoms:**
```bash
ERROR: Error loading ASGI app. Could not import module "main".
```

**Solution:**
```bash
# Make sure you're in the correct directory
cd enhanced_backend/enhanced_sdv_service

# Check if main.py exists
ls -la main.py

# Start from the correct directory
python -m uvicorn main:app --host 0.0.0.0 --port 8003 --reload
```

#### Issue 5: File Upload Errors
**Symptoms:**
```bash
FileNotFoundError: [Errno 2] No such file or directory
```

**Solution:**
```bash
# Make sure HR dataset files exist
ls -la complex_hr_*.csv

# If files don't exist, generate them
python complex_hr_dataset.py
```

---

## 📚 Additional Resources

### Documentation Files
- `README.md` - Main project documentation
- `docs/ENHANCED_SDV_DOCUMENTATION.md` - Detailed SDV documentation
- `docs/FRONTEND_INTEGRATION_GUIDE.md` - Frontend integration guide
- `docs/ENVIRONMENT_SETUP.md` - Environment setup guide

### Scripts and Tools
- `test_enhanced_backend.py` - Test the enhanced backend
- `check_relational_integrity.py` - Analyze output data integrity
- `complex_hr_dataset.py` - Generate HR test datasets
- `start_enhanced_service.sh` - Service startup script

### Log Files
- Service logs are displayed in the terminal when running
- Check the terminal output for detailed error messages
- Use `--log-level debug` for more verbose logging

---

## 🎯 Quick Start Summary

```bash
# 1. Navigate to enhanced backend
cd enhanced_backend

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install dependencies (if needed)
pip install -r requirements.txt

# 4. Start the service
cd enhanced_sdv_service
python -m uvicorn main:app --host 0.0.0.0 --port 8003 --reload

# 5. Test the service
curl http://localhost:8003/api/enhanced/health

# 6. Run full test with HR datasets
cd /Users/karthike/Desktop/Vibe\ Coding/synthe-flow
python test_enhanced_backend.py

# 7. Analyze output integrity
python check_relational_integrity.py
```

---

## 📞 Support

If you encounter any issues not covered in this guide:

1. Check the troubleshooting section above
2. Review the service logs in the terminal
3. Verify all prerequisites are met
4. Ensure the virtual environment is properly activated
5. Check that all dependencies are installed correctly

The enhanced backend is designed to handle complex multi-relational datasets and should work seamlessly with the provided HR datasets once properly configured. 