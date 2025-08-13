# ⚡ Enhanced Backend Quick Start

## 🚀 One-Minute Setup

```bash
# 1. Navigate to enhanced backend
cd enhanced_backend

# 2. Activate virtual environment
source venv/bin/activate

# 3. Start the service
cd enhanced_sdv_service
python -m uvicorn main:app --host 0.0.0.0 --port 8003 --reload
```

## ✅ Verify It's Working

```bash
# Test health endpoint
curl http://localhost:8003/api/enhanced/health

# Expected output:
# {"status": "healthy", "service": "Enhanced SDV Enterprise Service", "version": "3.0.0"}
```

## 🧪 Test with HR Datasets

```bash
# From project root directory
cd "/Users/karthike/Desktop/Vibe Coding/synthe-flow"
python test_enhanced_backend.py
```

## 📊 Analyze Output

```bash
# Check relational integrity
python check_relational_integrity.py
```

## 📁 Output Location

```
enhanced_backend/enhanced_sdv_service/exports/
├── {session_id}_employees_{timestamp}.json
├── {session_id}_departments_{timestamp}.json
├── {session_id}_positions_{timestamp}.json
├── {session_id}_salaries_{timestamp}.json
├── {session_id}_projects_{timestamp}.json
└── {session_id}_project_assignments_{timestamp}.json
```

## 🔧 Common Commands

| Command | Description |
|---------|-------------|
| `source venv/bin/activate` | Activate virtual environment |
| `pip install -r requirements.txt` | Install dependencies |
| `python -m uvicorn main:app --host 0.0.0.0 --port 8003 --reload` | Start service |
| `curl http://localhost:8003/api/enhanced/health` | Health check |
| `python test_enhanced_backend.py` | Run full test |
| `python check_relational_integrity.py` | Analyze output |

## 🆘 Quick Fixes

**Virtual environment not activated:**
```bash
cd enhanced_backend && source venv/bin/activate
```

**Port in use:**
```bash
lsof -i :8003 && kill -9 <PID>
```

**Missing dependencies:**
```bash
pip install sdv fastapi uvicorn pandas numpy scikit-learn torch
```

**Service won't start:**
```bash
cd enhanced_backend/enhanced_sdv_service && python -m uvicorn main:app --host 0.0.0.0 --port 8003 --reload
```

---

📖 **Full Documentation:** `ENHANCED_BACKEND_SETUP_GUIDE.md` 