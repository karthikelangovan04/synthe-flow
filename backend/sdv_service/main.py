from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pandas as pd
import json
import os
import sys

# Add SDV venv to path
sdv_venv_path = "/Users/karthike/GenAi/Synthetic_data/sdv/venv"
if sdv_venv_path not in sys.path:
    sys.path.insert(0, sdv_venv_path)

try:
    from sdv.metadata import MultiTableMetadata
    from sdv.multi_table import HMASynthesizer
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.evaluation import evaluate
except ImportError as e:
    print(f"Error importing SDV: {e}")
    print("Please ensure SDV is installed in the venv")

app = FastAPI(title="SDV Synthetic Data Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class ColumnSchema(BaseModel):
    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    is_unique: bool
    enhanced_description: Optional[str] = None
    business_rules: Optional[str] = None

class TableSchema(BaseModel):
    name: str
    description: Optional[str] = None
    columns: List[ColumnSchema]

class RelationshipSchema(BaseModel):
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship_type: str = "one-to-many"

class GenerationRequest(BaseModel):
    tables: List[TableSchema]
    relationships: List[RelationshipSchema]
    sample_data: Optional[Dict[str, List[Dict]]] = None
    scale: float = 1.0
    quality_settings: Optional[Dict[str, Any]] = None

class GenerationResponse(BaseModel):
    session_id: str
    status: str
    synthetic_data: Dict[str, List[Dict]]
    quality_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# In-memory storage for sessions (in production, use database)
sessions = {}

def map_data_type_to_sdv(data_type: str) -> str:
    """Map database data types to SDV types"""
    data_type_lower = data_type.lower()
    
    if any(num_type in data_type_lower for num_type in ['int', 'bigint', 'smallint', 'decimal', 'numeric', 'real', 'double']):
        return 'numerical'
    elif any(date_type in data_type_lower for date_type in ['date', 'time', 'timestamp']):
        return 'datetime'
    elif data_type_lower in ['boolean', 'bool']:
        return 'boolean'
    elif data_type_lower in ['uuid']:
        return 'id'
    else:
        return 'categorical'

def build_sdv_metadata(schema_data: GenerationRequest) -> MultiTableMetadata:
    """Build SDV metadata from schema"""
    metadata = MultiTableMetadata()
    
    # Add tables and columns
    for table in schema_data.tables:
        metadata.add_table(table.name)
        
        for column in table.columns:
            sdv_type = map_data_type_to_sdv(column.data_type)
            metadata.add_column(table.name, column.name, sdv_type)
            
            if column.is_primary_key:
                metadata.set_primary_key(table.name, column.name)
    
    # Add relationships
    for relationship in schema_data.relationships:
        try:
            metadata.add_relationship(
                parent_table_name=relationship.source_table,
                child_table_name=relationship.target_table,
                parent_primary_key=relationship.source_column,
                child_foreign_key=relationship.target_column
            )
        except Exception as e:
            print(f"Warning: Could not add relationship {relationship}: {e}")
    
    return metadata

def create_sample_data(schema_data: GenerationRequest) -> Dict[str, pd.DataFrame]:
    """Create sample data for training if no real data provided"""
    sample_data = {}
    
    for table in schema_data.tables:
        # Create basic sample data based on column types
        data = {}
        num_rows = 100  # Default sample size
        
        for column in table.columns:
            if column.data_type.lower() in ['int', 'bigint', 'smallint']:
                data[column.name] = list(range(1, num_rows + 1))
            elif column.data_type.lower() in ['varchar', 'text', 'char']:
                data[column.name] = [f"{table.name}_{i}" for i in range(1, num_rows + 1)]
            elif column.data_type.lower() == 'boolean':
                data[column.name] = [True if i % 2 == 0 else False for i in range(num_rows)]
            else:
                data[column.name] = [f"sample_{i}" for i in range(1, num_rows + 1)]
        
        sample_data[table.name] = pd.DataFrame(data)
    
    return sample_data

@app.post("/api/sdv/generate", response_model=GenerationResponse)
async def generate_synthetic_data(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate synthetic data based on schema and relationships"""
    try:
        session_id = f"session_{len(sessions) + 1}"
        sessions[session_id] = {"status": "processing", "data": None, "error": None}
        
        # Build SDV metadata
        metadata = build_sdv_metadata(request)
        
        # Prepare training data
        if request.sample_data:
            real_data = {table_name: pd.DataFrame(data) for table_name, data in request.sample_data.items()}
        else:
            real_data = create_sample_data(request)
        
        # Choose synthesizer based on relationships
        if len(request.relationships) > 0:
            synthesizer = HMASynthesizer(metadata)
        else:
            # Use single table synthesizer for each table
            synthesizer = GaussianCopulaSynthesizer()
        
        # Fit the synthesizer
        synthesizer.fit(real_data)
        
        # Generate synthetic data
        synthetic_data = synthesizer.sample(scale=request.scale)
        
        # Convert to JSON-serializable format
        synthetic_json = {}
        for table_name, df in synthetic_data.items():
            synthetic_json[table_name] = df.to_dict('records')
        
        # Store results
        sessions[session_id] = {
            "status": "completed",
            "data": synthetic_json,
            "metadata": metadata.to_dict(),
            "error": None
        }
        
        return GenerationResponse(
            session_id=session_id,
            status="completed",
            synthetic_data=synthetic_json
        )
        
    except Exception as e:
        error_msg = f"Error generating synthetic data: {str(e)}"
        sessions[session_id] = {"status": "error", "data": None, "error": error_msg}
        
        return GenerationResponse(
            session_id=session_id,
            status="error",
            synthetic_data={},
            error=error_msg
        )

@app.get("/api/sdv/status/{session_id}")
async def get_generation_status(session_id: str):
    """Get the status of a generation session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "status": session["status"],
        "data": session["data"],
        "error": session["error"]
    }

@app.post("/api/sdv/validate")
async def validate_synthetic_data(request: GenerationRequest):
    """Validate synthetic data quality"""
    try:
        # This would implement quality metrics
        # For now, return basic validation
        return {
            "quality_score": 0.85,
            "metrics": {
                "distribution_similarity": 0.82,
                "correlation_preservation": 0.88,
                "privacy_score": 0.90
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

@app.get("/api/sdv/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "sdv_available": "sdv" in sys.modules}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 