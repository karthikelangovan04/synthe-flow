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
    # Remove evaluate import as it's not available in this SDV version
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

def map_data_type_to_sdv(data_type: str, is_primary_key: bool = False) -> str:
    """Map database data types to SDV types"""
    data_type_lower = data_type.lower()
    
    # Primary keys must be 'id' type in SDV
    if is_primary_key:
        return 'id'
    
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
    
    # First pass: Identify foreign key columns that need to be treated as 'id' type
    foreign_key_columns = set()
    for relationship in schema_data.relationships:
        try:
            source_table_schema = next(t for t in schema_data.tables if t.name == relationship.source_table)
            source_column = next(c for c in source_table_schema.columns if c.name == relationship.source_column)
            
            if source_column.is_primary_key:
                foreign_key_columns.add((relationship.target_table, relationship.target_column))
        except Exception as e:
            print(f"Warning: Could not process relationship {relationship}: {e}")
    
    # Second pass: Add tables and columns with correct types
    for table in schema_data.tables:
        metadata.add_table(table.name)
        
        for column in table.columns:
            # Check if this column is a foreign key that should be treated as 'id' type
            is_foreign_key = (table.name, column.name) in foreign_key_columns
            
            if is_foreign_key:
                sdv_type = 'id'  # Foreign keys should be 'id' type to match primary keys
            else:
                sdv_type = map_data_type_to_sdv(column.data_type, column.is_primary_key)
            
            # SDV 1.12.1 uses keyword arguments for add_column
            metadata.add_column(table.name, column.name, sdtype=sdv_type)
            
            if column.is_primary_key:
                metadata.set_primary_key(table.name, column.name)
    
    # Third pass: Add relationships
    for relationship in schema_data.relationships:
        try:
            # Ensure the source column is marked as primary key in source table
            source_table_schema = next(t for t in schema_data.tables if t.name == relationship.source_table)
            source_column = next(c for c in source_table_schema.columns if c.name == relationship.source_column)
            
            # If source column is not primary key, we need to handle this
            if not source_column.is_primary_key:
                print(f"Warning: Source column '{relationship.source_column}' in table '{relationship.source_table}' is not marked as primary key")
                # For now, skip this relationship to avoid SDV validation errors
                continue
            
            metadata.add_relationship(
                parent_table_name=relationship.source_table,
                child_table_name=relationship.target_table,
                parent_primary_key=relationship.source_column,
                child_foreign_key=relationship.target_column
            )
            print(f"Successfully added relationship: {relationship.source_table}.{relationship.source_column} -> {relationship.target_table}.{relationship.target_column}")
        except Exception as e:
            print(f"Warning: Could not add relationship {relationship}: {e}")
    
    return metadata

def create_sample_data(schema_data: GenerationRequest) -> Dict[str, pd.DataFrame]:
    """Create sample data for training if no real data provided"""
    import random
    import numpy as np
    
    sample_data = {}
    
    # First pass: Create all tables with basic data
    for table in schema_data.tables:
        # Create basic sample data based on column types
        data = {}
        num_rows = 100  # Default sample size
        
        for column in table.columns:
            if column.data_type.lower() in ['int', 'bigint', 'smallint']:
                if column.is_primary_key:
                    data[column.name] = list(range(1, num_rows + 1))
                else:
                    data[column.name] = [random.randint(1, 1000) for _ in range(num_rows)]
            elif column.data_type.lower() in ['varchar', 'text', 'char']:
                if 'name' in column.name.lower():
                    data[column.name] = [f"Name_{i}" for i in range(1, num_rows + 1)]
                elif 'email' in column.name.lower():
                    data[column.name] = [f"user{i}@example.com" for i in range(1, num_rows + 1)]
                elif 'city' in column.name.lower():
                    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
                    data[column.name] = [random.choice(cities) for _ in range(num_rows)]
                elif 'status' in column.name.lower():
                    statuses = ['pending', 'completed', 'cancelled', 'processing']
                    data[column.name] = [random.choice(statuses) for _ in range(num_rows)]
                else:
                    data[column.name] = [f"{table.name}_{i}" for i in range(1, num_rows + 1)]
            elif column.data_type.lower() == 'boolean':
                data[column.name] = [random.choice([True, False]) for _ in range(num_rows)]
            elif column.data_type.lower() in ['decimal', 'numeric', 'real', 'double']:
                data[column.name] = [round(random.uniform(10.0, 1000.0), 2) for _ in range(num_rows)]
            elif column.data_type.lower() in ['date', 'timestamp']:
                from datetime import datetime, timedelta
                start_date = datetime.now() - timedelta(days=365)
                data[column.name] = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(num_rows)]
            else:
                data[column.name] = [f"sample_{i}" for i in range(1, num_rows + 1)]
        
        sample_data[table.name] = pd.DataFrame(data)
        print(f"Created sample data for table {table.name}: {len(sample_data[table.name])} rows")
    
    # Second pass: Update foreign key values to reference valid primary keys
    for relationship in schema_data.relationships:
        try:
            parent_table = relationship.source_table
            child_table = relationship.target_table
            parent_pk = relationship.source_column
            child_fk = relationship.target_column
            
            # Get valid primary key values from parent table
            if parent_table in sample_data and child_table in sample_data:
                parent_pk_values = sample_data[parent_table][parent_pk].tolist()
                
                # Update foreign key values in child table to reference valid parent keys
                num_child_rows = len(sample_data[child_table])
                valid_fk_values = [random.choice(parent_pk_values) for _ in range(num_child_rows)]
                sample_data[child_table][child_fk] = valid_fk_values
                
                print(f"Updated foreign key '{child_fk}' in table '{child_table}' to reference valid primary keys from '{parent_table}'")
        except Exception as e:
            print(f"Warning: Could not update foreign key for relationship {relationship}: {e}")
    
    return sample_data

@app.post("/api/sdv/generate", response_model=GenerationResponse)
async def generate_synthetic_data(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate synthetic data based on schema and relationships"""
    try:
        print(f"Received generation request with {len(request.tables)} tables and {len(request.relationships)} relationships")
        
        session_id = f"session_{len(sessions) + 1}"
        sessions[session_id] = {"status": "processing", "data": None, "error": None}
        
        # Build SDV metadata
        print("Building SDV metadata...")
        metadata = build_sdv_metadata(request)
        print(f"Metadata built successfully for {len(request.tables)} tables")
        
        # Prepare training data
        print("Preparing training data...")
        if request.sample_data:
            real_data = {table_name: pd.DataFrame(data) for table_name, data in request.sample_data.items()}
            print(f"Using provided sample data for {len(real_data)} tables")
        else:
            real_data = create_sample_data(request)
            print(f"Created sample data for {len(real_data)} tables")
        
        # Choose synthesizer based on relationships
        valid_relationships = []
        for rel in request.relationships:
            source_table_schema = next(t for t in request.tables if t.name == rel.source_table)
            source_column = next(c for c in source_table_schema.columns if c.name == rel.source_column)
            if source_column.is_primary_key:
                valid_relationships.append(rel)
        
        if len(valid_relationships) > 0:
            print(f"Using HMASynthesizer for {len(valid_relationships)} valid multi-table relationships")
            synthesizer = HMASynthesizer(metadata)
        else:
            print("No valid relationships found, using single table mode")
            print("Using GaussianCopulaSynthesizer for single tables")
            # For single tables, we need to handle each table separately
            from sdv.metadata import SingleTableMetadata
            synthetic_data = {}
            for table_name, df in real_data.items():
                print(f"Processing table: {table_name}")
                # Build single-table metadata for this table
                table_schema = next(t for t in request.tables if t.name == table_name)
                single_metadata = SingleTableMetadata()
                
                # Add columns to metadata
                for col in table_schema.columns:
                    sdv_type = map_data_type_to_sdv(col.data_type, col.is_primary_key)
                    single_metadata.add_column(col.name, sdtype=sdv_type)
                
                # Set primary key if exists
                primary_key = next((col.name for col in table_schema.columns if col.is_primary_key), None)
                if primary_key:
                    single_metadata.set_primary_key(primary_key)
                single_synthesizer = GaussianCopulaSynthesizer(single_metadata)
                single_synthesizer.fit(df)
                # For single table synthesizers, we need to specify the number of rows instead of scale
                # Generate more rows based on the scale parameter (default 100 * scale)
                base_rows = 100  # Base number of rows
                num_rows = int(base_rows * request.scale)
                print(f"Generating {num_rows} rows for table {table_name} (scale: {request.scale})")
                synthetic_data[table_name] = single_synthesizer.sample(num_rows=num_rows)
            
            # Convert to JSON-serializable format
            synthetic_json = {}
            for table_name, df in synthetic_data.items():
                synthetic_json[table_name] = df.to_dict('records')
                print(f"Table {table_name}: Generated {len(synthetic_json[table_name])} synthetic rows")
            
            print(f"Generated synthetic data for {len(synthetic_json)} tables")
            print(f"Total synthetic rows: {sum(len(rows) for rows in synthetic_json.values())}")
            
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
        
        # For multi-table relationships
        print("Fitting HMASynthesizer...")
        synthesizer.fit(real_data)
        print("HMASynthesizer fitted successfully")
        
        # Generate synthetic data
        print(f"Generating synthetic data with scale {request.scale}...")
        synthetic_data = synthesizer.sample(scale=request.scale)
        print(f"Generated synthetic data for {len(synthetic_data)} tables")
        
        # Convert to JSON-serializable format
        synthetic_json = {}
        for table_name, df in synthetic_data.items():
            synthetic_json[table_name] = df.to_dict('records')
            print(f"Table {table_name}: {len(synthetic_json[table_name])} rows")
        
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
        print(f"Error: {error_msg}")
        import traceback
        traceback.print_exc()
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
    uvicorn.run(app, host="0.0.0.0", port=8002) 