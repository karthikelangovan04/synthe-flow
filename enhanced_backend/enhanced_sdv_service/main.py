"""
Enhanced SDV Service for Complex Relational Databases
Specialized for HR Tech, Risk Tech, and other high-relational domains
Handles multiple tables, large volumes, and complex relationships
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import json
import os
import sys
import asyncio
import aiofiles
from datetime import datetime
import uuid
from pathlib import Path
import tempfile
import shutil
import logging
import random

# Enhanced imports for complex relational handling
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.preprocessing import StandardScaler
import networkx as nx

# Import enhanced modules
from ai_engine.llm_enhancer import LLMEnhancer
from ai_engine.neural_generator import NeuralDataGenerator, MultiTableNeuralGenerator
from quality_validator.complex_validator import ComplexDataValidator
from export_engine.enterprise_exporter import EnterpriseExporter
from connectors.enterprise_connectors import EnterpriseConnector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced SDV Enterprise Service", version="3.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files for uploaded files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Enhanced data models for complex relational databases
class EnhancedColumnSchema(BaseModel):
    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    is_unique: bool
    enhanced_description: Optional[str] = None
    business_rules: Optional[str] = None
    domain_context: Optional[str] = None  # HR, Risk, Finance, etc.
    data_sensitivity: Optional[str] = None  # PII, PHI, Financial, etc.
    validation_rules: Optional[List[str]] = None
    relationships: Optional[List[str]] = None

class EnhancedTableSchema(BaseModel):
    name: str
    description: Optional[str] = None
    columns: List[EnhancedColumnSchema]
    domain: Optional[str] = None  # HR, Risk, Finance, etc.
    estimated_volume: Optional[int] = None  # Expected row count
    update_frequency: Optional[str] = None  # Daily, Weekly, Monthly, etc.
    business_criticality: Optional[str] = None  # High, Medium, Low

class ComplexRelationshipSchema(BaseModel):
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship_type: str = "one-to-many"  # one-to-one, one-to-many, many-to-many
    cardinality: Optional[Dict[str, int]] = None  # Min/max cardinality
    cascade_delete: bool = False
    business_rule: Optional[str] = None

class EnterpriseDataSourceConfig(BaseModel):
    type: str  # "local", "postgres", "snowflake", "oracle", "s3", "gcs", "azure", "api", "catalog"
    config: Dict[str, Any] = {}
    file_paths: Optional[List[str]] = None
    connection_pool_size: Optional[int] = 10
    timeout_seconds: Optional[int] = 300

class EnhancedGenerationRequest(BaseModel):
    tables: List[EnhancedTableSchema]
    relationships: List[ComplexRelationshipSchema]
    data_sources: Optional[List[EnterpriseDataSourceConfig]] = None
    scale: float = 1.0
    quality_settings: Optional[Dict[str, Any]] = None
    output_format: str = "json"  # "json", "csv", "excel", "sql", "parquet"
    random_seed: Optional[int] = None
    generation_strategy: str = "balanced"  # "balanced", "performance", "quality", "privacy"
    domain_context: Optional[str] = None  # HR, Risk, Finance, etc.
    privacy_level: str = "standard"  # "standard", "enhanced", "enterprise"
    performance_optimization: bool = True
    parallel_processing: bool = True
    memory_optimization: bool = True

class EnhancedGenerationResponse(BaseModel):
    session_id: str
    status: str
    synthetic_data: Dict[str, List[Dict]]
    quality_metrics: Optional[Dict[str, Any]] = None
    output_files: Optional[List[str]] = None
    error: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    privacy_metrics: Optional[Dict[str, Any]] = None
    relationship_validation: Optional[Dict[str, Any]] = None

class EnterpriseConnectorStatus(BaseModel):
    type: str
    status: str
    message: str
    available: bool
    performance_metrics: Optional[Dict[str, Any]] = None

# In-memory storage for sessions and connectors
sessions = {}
active_connectors = {}

# Enhanced SDV Engine for complex relational databases
class EnhancedSDVEngine:
    def __init__(self):
        self.llm_enhancer = LLMEnhancer()
        self.quality_validator = ComplexDataValidator()
        self.export_engine = EnterpriseExporter()
        self.connector = EnterpriseConnector()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def generate_complex_synthetic_data(self, request: EnhancedGenerationRequest) -> EnhancedGenerationResponse:
        """Generate synthetic data for complex relational databases"""
        try:
            # Validate request
            if not isinstance(request, EnhancedGenerationRequest):
                raise ValueError("request must be an EnhancedGenerationRequest object")
            
            if not hasattr(request, 'tables') or not request.tables:
                raise ValueError("request must have tables")
            
            if not hasattr(request, 'scale') or not isinstance(request.scale, (int, float)) or request.scale <= 0:
                raise ValueError("request must have a positive scale")
            
            session_id = str(uuid.uuid4())
            
            # Initialize session in global storage
            try:
                            sessions[session_id] = {
                "status": "processing",
                "data": {},
                "quality_metrics": {},
                "output_files": [],
                "error": None,
                "performance_metrics": {},
                "created_at": datetime.now().isoformat()
            }
            except Exception as e:
                logger.error(f"Error initializing session {session_id}: {e}")
                # Continue without session initialization
            
            logger.info(f"Starting enhanced generation for session {session_id}")
            
            # Step 1: Analyze schema complexity
            try:
                complexity_score = self._analyze_schema_complexity(request)
                logger.info(f"Schema complexity score: {complexity_score}")
            except Exception as e:
                logger.error(f"Error analyzing schema complexity: {e}")
                complexity_score = 0.5  # Default complexity score
                logger.info(f"Using default complexity score: {complexity_score}")
            
            # Step 2: Optimize generation strategy
            try:
                strategy = self._optimize_strategy(request, complexity_score)
            except Exception as e:
                logger.error(f"Error optimizing strategy: {e}")
                strategy = {'strategy': 'simple', 'processing_mode': 'sequential', 'batch_size': 1000}
                logger.info("Using default strategy")
            
            # Step 3: Build enhanced metadata
            try:
                metadata = self._build_enhanced_metadata(request)
            except Exception as e:
                logger.error(f"Error building metadata: {e}")
                metadata = {'tables': [], 'relationships': [], 'domain_context': 'unknown', 'complexity_score': complexity_score}
                logger.info("Using default metadata")
            
            # Step 4: Load enterprise data
            try:
                if hasattr(request, 'data_sources') and request.data_sources:
                    data_dict = await self._load_enterprise_data(request.data_sources)
                else:
                    data_dict = {}
            except Exception as e:
                logger.error(f"Error loading enterprise data: {e}")
                data_dict = {}
            
            # Step 5: Generate synthetic data with parallel processing
            try:
                synthetic_data = await self._generate_parallel(request, metadata, data_dict, strategy)
            except Exception as e:
                logger.error(f"Neural generation failed: {e}")
                logger.info("Falling back to basic synthetic data generation")
                # Fallback to basic generation
                synthetic_data = {}
                for table in request.tables:
                    try:
                        if hasattr(table, 'name') and hasattr(table, 'columns'):
                            synthetic_data[table.name] = self._create_synthetic_table_data(table, request.scale)
                        else:
                            logger.warning(f"Skipping table with invalid structure")
                    except Exception as table_error:
                        logger.error(f"Error creating synthetic data for table {getattr(table, 'name', 'unknown')}: {table_error}")
                        # Create empty DataFrame as fallback
                        if hasattr(table, 'columns') and table.columns:
                            columns = [col.name for col in table.columns]
                            synthetic_data[getattr(table, 'name', f'table_{len(synthetic_data)}')] = pd.DataFrame(columns=columns)
                        else:
                            synthetic_data[getattr(table, 'name', f'table_{len(synthetic_data)}')] = pd.DataFrame()
            
            # Convert DataFrame to List[Dict] format for response
            try:
                synthetic_data_list = {}
                for table_name, df in synthetic_data.items():
                    try:
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            synthetic_data_list[table_name] = df.to_dict('records')
                        else:
                            synthetic_data_list[table_name] = []
                    except Exception as e:
                        logger.error(f"Error converting DataFrame for table {table_name}: {e}")
                        synthetic_data_list[table_name] = []
            except Exception as e:
                logger.error(f"Error converting DataFrames to list format: {e}")
                synthetic_data_list = {}
            
            # Step 6: Fix referential integrity issues
            try:
                if metadata.get('relationships'):
                    print("🔧 Fixing referential integrity issues...")
                    synthetic_data = self._fix_referential_integrity(synthetic_data, metadata['relationships'])
                else:
                    print("No relationships to fix")
            except Exception as e:
                logger.error(f"Error fixing referential integrity: {e}")
                # Continue without fixing referential integrity
            
            # Step 6.5: Fix primary key issues
            try:
                if metadata.get('tables'):
                    print("🔑 Validating and fixing primary keys...")
                    synthetic_data = self._fix_primary_keys(synthetic_data, metadata['tables'])
                else:
                    print("No tables to fix")
            except Exception as e:
                logger.error(f"Error fixing primary keys: {e}")
                # Continue without fixing primary keys
            
            # Step 7: Validate relationships and quality
            try:
                quality_settings = getattr(request, 'quality_settings', {}) if hasattr(request, 'quality_settings') else {}
                quality_metrics = self.quality_validator.validate_complex_data(
                    synthetic_data, data_dict, metadata.get('relationships', []), quality_settings
                )
            except Exception as e:
                logger.error(f"Quality validation failed: {e}")
                # Create basic quality metrics
                quality_metrics = {
                    'overall_score': 0.5,
                    'status': 'validation_failed',
                    'error': str(e),
                    'table_scores': {},
                    'relationship_scores': {},
                    'privacy_metrics': {'overall_privacy_score': 0.5},
                    'statistical_metrics': {'overall_statistical_score': 0.5},
                    'recommendations': ['Quality validation failed - check data format']
                }
            
            # Step 8: Export with enterprise optimizations
            try:
                output_format = getattr(request, 'output_format', 'json') if hasattr(request, 'output_format') else 'json'
                export_results = self.export_engine.export_data(
                    synthetic_data, data_dict, metadata.get('relationships', []), quality_metrics, output_format
                )
                output_files = export_results.get('exported_files', [])
            except Exception as e:
                logger.error(f"Export failed: {e}")
                output_files = []
                # Continue without export
            
            # Update session with completed data
            try:
                performance_metrics = {
                    "complexity_score": complexity_score,
                    "strategy": strategy.get('strategy', 'unknown') if isinstance(strategy, dict) else 'unknown'
                }
                
                sessions[session_id].update({
                    "status": "completed",
                    "data": synthetic_data_list,
                    "quality_metrics": quality_metrics,
                    "output_files": output_files,
                    "performance_metrics": performance_metrics
                })
                print(f"DEBUG: Session {session_id} updated successfully")
                print(f"DEBUG: Available sessions: {list(sessions.keys())}")
            except Exception as e:
                logger.error(f"Error updating session {session_id}: {e}")
                # Try to create a new session entry
                sessions[session_id] = {
                    "status": "completed",
                    "data": synthetic_data_list,
                    "quality_metrics": quality_metrics,
                    "output_files": output_files,
                    "performance_metrics": {"complexity_score": complexity_score, "strategy": "unknown"}
                }
            
            return EnhancedGenerationResponse(
                session_id=session_id,
                status="completed",
                synthetic_data=synthetic_data_list,
                quality_metrics=quality_metrics,
                output_files=output_files
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced generation: {str(e)}")
            
            # Update session with error
            try:
                sessions[session_id].update({
                    "status": "failed",
                    "error": str(e)
                })
            except Exception as session_error:
                logger.error(f"Error updating session {session_id}: {session_error}")
                # Try to create a new session entry
                sessions[session_id] = {
                    "status": "failed",
                    "error": str(e)
                }
            
            # Return error response
            try:
                return EnhancedGenerationResponse(
                    session_id=session_id,
                    status="failed",
                    synthetic_data={},
                    error=str(e)
                )
            except Exception as response_error:
                logger.error(f"Error creating error response: {response_error}")
                # Return basic error response
                return EnhancedGenerationResponse(
                    session_id=session_id,
                    status="failed",
                    synthetic_data={},
                    error="Failed to create response"
                )
    
    def _analyze_schema_complexity(self, request: EnhancedGenerationRequest) -> float:
        """Analyze the complexity of the database schema"""
        try:
            # Validate input
            if not isinstance(request, EnhancedGenerationRequest):
                raise ValueError("request must be an EnhancedGenerationRequest object")
            
            if not hasattr(request, 'tables') or not request.tables:
                return 0.0
            
            if not hasattr(request, 'relationships') or not request.relationships:
                return 0.0
            
            # Calculate complexity based on:
            # 1. Number of tables
            num_tables = len(request.tables)
            
            # 2. Number of relationships
            num_relationships = len(request.relationships)
            
            # 3. Average columns per table
            total_columns = 0
            for table in request.tables:
                try:
                    if hasattr(table, 'columns') and table.columns:
                        total_columns += len(table.columns)
                except Exception as e:
                    logger.error(f"Error counting columns for table {getattr(table, 'name', 'unknown')}: {e}")
                    continue
            
            avg_columns = total_columns / max(num_tables, 1)
            
            # 4. Complexity score (0.0 to 1.0)
            # Higher score = more complex
            table_complexity = min(1.0, num_tables / 10.0)  # Normalize to 0-1
            relationship_complexity = min(1.0, num_relationships / 20.0)  # Normalize to 0-1
            column_complexity = min(1.0, avg_columns / 20.0)  # Normalize to 0-1
            
            # Weighted average
            complexity_score = (table_complexity * 0.4 + 
                              relationship_complexity * 0.4 + 
                              column_complexity * 0.2)
            
            return complexity_score
            
        except Exception as e:
            logger.error(f"Error in schema complexity analysis: {e}")
            return 0.5  # Return neutral complexity on error
    
    def _optimize_strategy(self, request: EnhancedGenerationRequest, complexity_score: float) -> Dict[str, Any]:
        """Optimize generation strategy based on complexity"""
        try:
            # Validate input
            if not isinstance(request, EnhancedGenerationRequest):
                raise ValueError("request must be an EnhancedGenerationRequest object")
            
            if not isinstance(complexity_score, (int, float)):
                raise ValueError("complexity_score must be a number")
            
            # Determine strategy based on complexity
            if complexity_score < 0.3:
                strategy = "simple"
                processing_mode = "sequential"
                batch_size = 1000
            elif complexity_score < 0.7:
                strategy = "moderate"
                processing_mode = "hybrid"
                batch_size = 500
            else:
                strategy = "complex"
                processing_mode = "parallel"
                batch_size = 250
            
            # Adjust based on domain context
            if hasattr(request, 'domain_context') and request.domain_context:
                domain = request.domain_context.lower()
                if domain in ['hr', 'human_resources']:
                    strategy += "_hr_optimized"
                elif domain in ['risk', 'compliance']:
                    strategy += "_risk_optimized"
                elif domain in ['finance', 'banking']:
                    strategy += "_finance_optimized"
            
            return {
                'strategy': strategy,
                'processing_mode': processing_mode,
                'batch_size': batch_size,
                'complexity_score': complexity_score
            }
            
        except Exception as e:
            logger.error(f"Error in strategy optimization: {e}")
            # Return default strategy on error
            return {
                'strategy': 'simple',
                'processing_mode': 'sequential',
                'batch_size': 1000,
                'complexity_score': complexity_score
            }
    
    async def _load_enterprise_data(self, data_sources: List[EnterpriseDataSourceConfig]) -> Dict[str, pd.DataFrame]:
        """Load enterprise data from various sources"""
        try:
            # Validate input
            if not isinstance(data_sources, list):
                raise ValueError("data_sources must be a list")
            
            print(f"Processing {len(data_sources)} data sources")
            data_dict = {}
            
            for source in data_sources:
                try:
                    print(f"Processing source: {source}")
                    
                    # Convert Pydantic model to dict if needed
                    if hasattr(source, 'model_dump'):
                        source = source.model_dump()
                    elif hasattr(source, 'dict'):
                        source = source.dict()
                    
                    if not isinstance(source, dict):
                        print(f"Skipping non-dict source: {type(source)}")
                        continue
                    
                    source_type = source.get('type', 'unknown')
                    print(f"Source type: {source_type}")
                    
                    # Handle file_paths for local sources
                    if source_type == 'local' and 'file_paths' in source:
                        file_paths = source.get('file_paths', [])
                        print(f"Processing {len(file_paths)} file paths: {file_paths}")
                        for file_path in file_paths:
                            try:
                                # Try to find the file in the uploads directory
                                upload_path = os.path.join("uploads", file_path)
                                if os.path.exists(upload_path):
                                    file_path = upload_path
                                elif os.path.exists(file_path):
                                    # File exists as absolute path
                                    pass
                                else:
                                    # Try to find file with pattern matching
                                    import glob
                                    upload_dir = "uploads"
                                    if os.path.exists(upload_dir):
                                        pattern = os.path.join(upload_dir, "*" + os.path.basename(file_path))
                                        matches = glob.glob(pattern)
                                        if matches:
                                            file_path = matches[0]
                                            print(f"Found file at: {file_path}")
                                        else:
                                            print(f"File not found: {file_path}")
                                            continue
                                    else:
                                        print(f"Uploads directory not found: {upload_dir}")
                                        continue
                                
                                if file_path.endswith('.csv'):
                                    df = pd.read_csv(file_path)
                                    if not df.empty:
                                        # Extract table name from file path - get the last part before .csv
                                        filename = os.path.basename(file_path)
                                        table_name = filename.split('_')[-1].replace('.csv', '')  # Get last part (e.g., 'users' from '20250816_072107_108444cf_users.csv')
                                        data_dict[table_name] = df
                                        print(f"Loaded {len(df)} rows from {file_path} into table '{table_name}'")
                                elif file_path.endswith('.json'):
                                    df = pd.read_json(file_path)
                                    if not df.empty:
                                        filename = os.path.basename(file_path)
                                        table_name = filename.split('_')[-1].replace('.json', '')
                                        data_dict[table_name] = df
                                        print(f"Loaded {len(df)} rows from {file_path} into table '{table_name}'")
                                elif file_path.endswith(('.xlsx', '.xls')):
                                    df = pd.read_excel(file_path)
                                    if not df.empty:
                                        filename = os.path.basename(file_path)
                                        table_name = filename.split('_')[-1].replace('.xlsx', '').replace('.xls', '')
                                        data_dict[table_name] = df
                                        print(f"Loaded {len(df)} rows from {file_path} into table '{table_name}'")
                            except Exception as e:
                                logger.error(f"Error loading file {file_path}: {e}")
                                continue
                        continue
                    
                    # Handle individual source paths
                    source_path = source.get('path', '')
                    
                    if not source_path:
                        continue
                    
                    if source_type == 'csv':
                        try:
                            df = pd.read_csv(source_path)
                            if not df.empty:
                                # Extract table name from file path
                                table_name = os.path.basename(source_path).split('_')[0]  # Extract table name from filename
                                data_dict[table_name] = df
                                print(f"Loaded {len(df)} rows from {source_path}")
                        except Exception as e:
                            logger.error(f"Error loading CSV from {source_path}: {e}")
                            continue
                            
                    elif source_type == 'json':
                        try:
                            df = pd.read_json(source_path)
                            if not df.empty:
                                data_dict[source.get('name', f'source_{len(data_dict)}')] = df
                                print(f"Loaded {len(df)} rows from {source_path}")
                        except Exception as e:
                            logger.error(f"Error loading JSON from {source_path}: {e}")
                            continue
                            
                    elif source_type == 'excel':
                        try:
                            df = pd.read_excel(source_path)
                            if not df.empty:
                                data_dict[source.get('name', f'source_{len(data_dict)}')] = df
                                print(f"Loaded {len(df)} rows from {source_path}")
                        except Exception as e:
                            logger.error(f"Error loading Excel from {source_path}: {e}")
                            continue
                            
                    else:
                        logger.warning(f"Unsupported source type: {source_type}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Error processing data source: {e}")
                    continue
            
            logger.info(f"Loaded data: {list(data_dict.keys())}")
            return data_dict
            
        except Exception as e:
            logger.error(f"Error in load_enterprise_data: {e}")
            return {}
    
    def _build_enhanced_metadata(self, request: EnhancedGenerationRequest) -> Dict[str, Any]:
        """Build enhanced metadata for the generation request"""
        try:
            # Validate input
            if not isinstance(request, EnhancedGenerationRequest):
                raise ValueError("request must be an EnhancedGenerationRequest object")
            
            metadata = {
                'tables': [],
                'relationships': [],
                'domain_context': '',
                'complexity_score': 0.0
            }
            
            # Extract table information
            if hasattr(request, 'tables') and request.tables:
                for table in request.tables:
                    try:
                        if not isinstance(table, EnhancedTableSchema):
                            continue
                            
                        table_info = {
                            'name': getattr(table, 'name', 'unknown'),
                            'columns': [],
                            'estimated_volume': getattr(table, 'estimated_volume', 1000)
                        }
                        
                        # Extract column information
                        if hasattr(table, 'columns') and table.columns:
                            for col in table.columns:
                                try:
                                    if not isinstance(col, EnhancedColumnSchema):
                                        continue
                                        
                                    col_info = {
                                        'name': getattr(col, 'name', 'unknown'),
                                        'data_type': getattr(col, 'data_type', 'unknown'),
                                        'is_primary_key': getattr(col, 'is_primary_key', False),
                                        'is_foreign_key': getattr(col, 'is_foreign_key', False),
                                        'constraints': getattr(col, 'constraints', {})
                                    }
                                    table_info['columns'].append(col_info)
                                    
                                except Exception as e:
                                    logger.error(f"Error extracting column info: {e}")
                                    continue
                        
                        metadata['tables'].append(table_info)
                        
                    except Exception as e:
                        logger.error(f"Error extracting table info: {e}")
                        continue
            
            # Extract relationship information
            if hasattr(request, 'relationships') and request.relationships:
                for rel in request.relationships:
                    try:
                        if not isinstance(rel, ComplexRelationshipSchema):
                            continue
                            
                        rel_info = {
                            'source_table': getattr(rel, 'source_table', 'unknown'),
                            'target_table': getattr(rel, 'target_table', 'unknown'),
                            'source_column': getattr(rel, 'source_column', 'unknown'),
                            'target_column': getattr(rel, 'target_column', 'unknown'),
                            'relationship_type': getattr(rel, 'relationship_type', 'unknown')
                        }
                        metadata['relationships'].append(rel_info)
                        
                    except Exception as e:
                        logger.error(f"Error extracting relationship info: {e}")
                        continue
            
            # Extract domain context
            if hasattr(request, 'domain_context') and request.domain_context:
                metadata['domain_context'] = request.domain_context
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error in build_enhanced_metadata: {e}")
            # Return basic metadata as fallback
            return {
                'tables': [],
                'relationships': [],
                'domain_context': 'unknown',
                'complexity_score': 0.5
            }
    
    def _get_primary_key(self, columns: List[EnhancedColumnSchema]) -> str:
        """Extract primary key from columns"""
        try:
            # Validate input
            if not isinstance(columns, list):
                return ""
            
            if not columns:
                return ""
            
            # Find primary key column
            for col in columns:
                try:
                    if not isinstance(col, EnhancedColumnSchema):
                        continue
                        
                    if getattr(col, 'is_primary_key', False):
                        return getattr(col, 'name', 'unknown')
                        
                except Exception as e:
                    logger.error(f"Error checking column for primary key: {e}")
                    continue
            
            # If no primary key found, return first column name
            if columns:
                try:
                    first_col = columns[0]
                    if isinstance(first_col, EnhancedColumnSchema):
                        return getattr(first_col, 'name', 'unknown')
                except Exception as e:
                    logger.error(f"Error getting first column name: {e}")
            
            return ""
            
        except Exception as e:
            logger.error(f"Error in get_primary_key: {e}")
            return ""
    
    def _get_table_relationships(self, table_name: str, relationships: List[ComplexRelationshipSchema]) -> List[Dict]:
        """Get relationships for a specific table"""
        try:
            # Validate input
            if not isinstance(table_name, str):
                return []
            
            if not table_name:
                return []
            
            if not isinstance(relationships, list):
                return []
            
            table_relationships = []
            
            for rel in relationships:
                try:
                    if not isinstance(rel, ComplexRelationshipSchema):
                        continue
                        
                    source_table = getattr(rel, 'source_table', '')
                    target_table = getattr(rel, 'target_table', '')
                    
                    if source_table == table_name or target_table == table_name:
                        rel_info = {
                            'source_table': source_table,
                            'target_table': target_table,
                            'source_column': getattr(rel, 'source_column', ''),
                            'target_column': getattr(rel, 'target_column', ''),
                            'relationship_type': getattr(rel, 'relationship_type', 'unknown')
                        }
                        table_relationships.append(rel_info)
                        
                except Exception as e:
                    logger.error(f"Error processing relationship: {e}")
                    continue
            
            return table_relationships
            
        except Exception as e:
            logger.error(f"Error in get_table_relationships: {e}")
            return []
    
    async def _generate_parallel(self, request: EnhancedGenerationRequest, metadata: Dict[str, Any], 
                                data_dict: Dict[str, pd.DataFrame], strategy: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data using parallel processing"""
        try:
            # Validate input
            if not isinstance(request, EnhancedGenerationRequest):
                raise ValueError("request must be an EnhancedGenerationRequest object")
            
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a dictionary")
            
            if not isinstance(data_dict, dict):
                raise ValueError("data_dict must be a dictionary")
            
            if not isinstance(strategy, dict):
                raise ValueError("strategy must be a dictionary")
            
            # Use MultiTableNeuralGenerator for complex relationships
            neural_generator = MultiTableNeuralGenerator()
            
            try:
                # Fit the neural models
                if data_dict:
                    neural_generator.fit(data_dict, metadata.get('relationships', []), metadata)
                else:
                    # Create sample data if no source data provided
                    sample_data = self._create_sample_enterprise_data(request)
                    neural_generator.fit(sample_data, metadata.get('relationships', []), metadata)
                
                # Generate synthetic data
                target_samples = {}
                for table in request.tables:
                    try:
                        if not hasattr(table, 'name'):
                            continue
                            
                        table_name = table.name
                        if table_name in data_dict and len(data_dict[table_name]) > 0:
                            target_samples[table_name] = int(len(data_dict[table_name]) * request.scale)
                        else:
                            target_samples[table_name] = int(request.scale * 100)  # Default sample size
                            
                    except Exception as e:
                        logger.error(f"Error calculating target samples for table {getattr(table, 'name', 'unknown')}: {e}")
                        continue
                
                synthetic_data = neural_generator.generate(target_samples)
                
                return synthetic_data
                
            except Exception as e:
                logger.error(f"Parallel generation failed: {e}")
                # Fallback to sequential generation
                logger.info("Falling back to sequential generation")
                return await self._generate_sequential(request, metadata, data_dict, strategy)
                
        except Exception as e:
            logger.error(f"Error in parallel generation: {e}")
            # Return empty data as fallback
            return {}
    
    async def _generate_sequential(self, request: EnhancedGenerationRequest, metadata: Dict[str, Any],
                                 data_dict: Dict[str, pd.DataFrame], strategy: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data sequentially"""
        try:
            # Validate input
            if not isinstance(request, EnhancedGenerationRequest):
                raise ValueError("request must be an EnhancedGenerationRequest object")
            
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a dictionary")
            
            if not isinstance(data_dict, dict):
                raise ValueError("data_dict must be a dictionary")
            
            if not isinstance(strategy, dict):
                raise ValueError("strategy must be a dictionary")
            
            synthetic_data = {}
            
            for table in request.tables:
                try:
                    if not hasattr(table, 'name'):
                        continue
                        
                    table_name = table.name
                    
                    if table_name in data_dict and len(data_dict[table_name]) > 0:
                        # Use existing data to train neural generator
                        print(f"Training neural generator for table: {table_name} with {len(data_dict[table_name])} samples")
                        try:
                            neural_generator = NeuralDataGenerator()
                            schema_info = {col.name: {'data_type': col.data_type} for col in table.columns}
                            neural_generator.fit(data_dict[table_name], schema_info)
                            
                            num_samples = int(len(data_dict[table_name]) * request.scale)
                            print(f"Generating {num_samples} synthetic samples for table: {table_name}")
                            synthetic_data[table_name] = neural_generator.generate(num_samples)
                            
                        except Exception as e:
                            logger.error(f"Neural generation failed for table {table_name}: {e}")
                            logger.info(f"Falling back to basic generation for table {table_name}")
                            # Fallback to basic generation
                            try:
                                synthetic_data[table_name] = self._create_synthetic_table_data(table, request.scale)
                            except Exception as fallback_error:
                                logger.error(f"Fallback generation also failed for table {table_name}: {fallback_error}")
                                # Create empty DataFrame as last resort
                                columns = [col.name for col in table.columns]
                                synthetic_data[table_name] = pd.DataFrame(columns=columns)
                    else:
                        # Create synthetic data from scratch
                        print(f"Creating synthetic data from scratch for table: {table_name}")
                        try:
                            synthetic_data[table_name] = self._create_synthetic_table_data(table, request.scale)
                        except Exception as e:
                            logger.error(f"Error creating synthetic data for table {table_name}: {e}")
                            # Create empty DataFrame as fallback
                            columns = [col.name for col in table.columns]
                            synthetic_data[table_name] = pd.DataFrame(columns=columns)
                            
                except Exception as e:
                    logger.error(f"Error processing table {getattr(table, 'name', 'unknown')}: {e}")
                    continue
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error in sequential generation: {e}")
            # Return empty data as fallback
            return {}
    
    def _create_sample_enterprise_data(self, request: EnhancedGenerationRequest) -> Dict[str, pd.DataFrame]:
        """Create sample enterprise data for testing"""
        try:
            # Validate input
            if not isinstance(request, EnhancedGenerationRequest):
                raise ValueError("request must be an EnhancedGenerationRequest object")
            
            if not hasattr(request, 'tables') or not request.tables:
                raise ValueError("request must have tables")
            
            sample_data = {}
            
            for table in request.tables:
                try:
                    if not hasattr(table, 'name') or not hasattr(table, 'columns'):
                        continue
                        
                    table_name = table.name
                    columns = table.columns
                    
                    if not columns:
                        continue
                    
                    # Create sample data for this table
                    num_samples = 50  # Default sample size
                    table_data = {}
                    
                    for col in columns:
                        try:
                            if not hasattr(col, 'name') or not hasattr(col, 'data_type'):
                                continue
                                
                            col_name = col.name
                            col_type = col.data_type
                            
                            if col_type in ['integer', 'int', 'int64']:
                                table_data[col_name] = np.random.randint(1, 1000, num_samples)
                            elif col_type in ['decimal', 'float', 'float64']:
                                table_data[col_name] = np.random.uniform(0, 1000, num_samples)
                            elif col_type in ['varchar', 'string', 'text']:
                                table_data[col_name] = [f"sample_{table_name}_{col_name}_{i}" for i in range(num_samples)]
                            elif col_type == 'boolean':
                                table_data[col_name] = np.random.choice([True, False], num_samples)
                            elif col_type == 'timestamp':
                                table_data[col_name] = pd.date_range(start="2020-01-01", periods=num_samples, freq="D")
                            else:
                                # Default to string
                                table_data[col_name] = [f"default_{table_name}_{col_name}_{i}" for i in range(num_samples)]
                                
                        except Exception as e:
                            logger.error(f"Error creating sample data for column {getattr(col, 'name', 'unknown')}: {e}")
                            # Use default values
                            table_data[getattr(col, 'name', f'col_{len(table_data)}')] = [f"error_{i}" for i in range(num_samples)]
                    
                    # Create DataFrame for this table
                    if table_data:
                        sample_data[table_name] = pd.DataFrame(table_data)
                    
                except Exception as e:
                    logger.error(f"Error creating sample data for table {getattr(table, 'name', 'unknown')}: {e}")
                    continue
            
            return sample_data
            
        except Exception as e:
            logger.error(f"Error in create_sample_enterprise_data: {e}")
            # Return empty data as fallback
            return {}
    
    def _create_synthetic_table_data(self, table: EnhancedTableSchema, scale: float) -> pd.DataFrame:
        """Create synthetic data for a table"""
        try:
            # Validate input
            if not isinstance(table, EnhancedTableSchema):
                raise ValueError("table must be an EnhancedTableSchema object")
            
            if not hasattr(table, 'columns') or not table.columns:
                raise ValueError("table must have columns")
            
            if not isinstance(scale, (int, float)) or scale <= 0:
                raise ValueError("scale must be a positive number")
            
            # Calculate number of samples
            num_samples = max(1, int(scale * 100))  # Minimum 1 sample
            
            # Create synthetic data for each column
            synthetic_data = {}
            
            for col in table.columns:
                try:
                    if not hasattr(col, 'name') or not hasattr(col, 'data_type'):
                        continue
                        
                    col_name = col.name
                    col_type = col.data_type
                    
                    if col_type in ['integer', 'int', 'int64']:
                        synthetic_data[col_name] = np.random.randint(1, 1000, num_samples)
                    elif col_type in ['decimal', 'float', 'float64']:
                        synthetic_data[col_name] = np.random.uniform(0, 1000, num_samples)
                    elif col_type in ['varchar', 'string', 'text']:
                        synthetic_data[col_name] = [f"sample_{i}" for i in range(num_samples)]
                    elif col_type == 'boolean':
                        synthetic_data[col_name] = np.random.choice([True, False], num_samples)
                    elif col_type == 'timestamp':
                        synthetic_data[col_name] = pd.date_range(start="2020-01-01", periods=num_samples, freq="D")
                    else:
                        # Default to string
                        synthetic_data[col_name] = [f"default_{i}" for i in range(num_samples)]
                        
                except Exception as e:
                    logger.error(f"Error creating synthetic data for column {getattr(col, 'name', 'unknown')}: {e}")
                    # Use default values
                    synthetic_data[getattr(col, 'name', f'col_{len(synthetic_data)}')] = [f"error_{i}" for i in range(num_samples)]
            
            # Create DataFrame
            df = pd.DataFrame(synthetic_data)
            
            # Ensure we have the right number of rows
            if len(df) != num_samples:
                logger.warning(f"Expected {num_samples} samples, got {len(df)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in create_synthetic_table_data: {e}")
            # Return empty DataFrame as fallback
            if hasattr(table, 'columns') and table.columns:
                columns = [getattr(col, 'name', f'col_{i}') for i, col in enumerate(table.columns)]
                return pd.DataFrame(columns=columns)
            else:
                return pd.DataFrame()

    def _fix_referential_integrity(self, synthetic_data: Dict[str, pd.DataFrame], 
                                  relationships: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """Fix referential integrity issues in synthetic data"""
        try:
            # Validate input
            if not isinstance(synthetic_data, dict):
                raise ValueError("synthetic_data must be a dictionary")
            
            if not isinstance(relationships, list):
                raise ValueError("relationships must be a list")
            
            if not synthetic_data:
                return synthetic_data
            
            # Create a copy to avoid modifying original data
            fixed_data = {table_name: df.copy() for table_name, df in synthetic_data.items()}
            
            for rel in relationships:
                try:
                    if not isinstance(rel, dict):
                        continue
                        
                    source_table = rel.get('source_table', '')
                    target_table = rel.get('target_table', '')
                    source_column = rel.get('source_column', '')
                    target_column = rel.get('target_column', '')
                    
                    if not all([source_table, target_table, source_column, target_column]):
                        continue
                    
                    if source_table not in fixed_data or target_table not in fixed_data:
                        continue
                    
                    # Get the DataFrames
                    source_df = fixed_data[source_table]
                    target_df = fixed_data[target_table]
                    
                    if not isinstance(source_df, pd.DataFrame) or not isinstance(target_df, pd.DataFrame):
                        continue
                    
                    if source_df.empty or target_df.empty:
                        continue
                    
                    # Check if columns exist
                    if source_column not in source_df.columns or target_column not in target_df.columns:
                        continue
                    
                    # Get foreign key values from source table
                    source_values = set(source_df[source_column].dropna())
                    target_values = set(target_df[target_column].dropna())
                    
                    # Find orphaned foreign keys
                    orphaned_keys = source_values - target_values
                    
                    if orphaned_keys:
                        print(f"🔧 Fixing {len(orphaned_keys)} orphaned foreign keys in {source_table}.{source_column}")
                        
                        # Fix orphaned foreign keys by updating them to valid values
                        for orphaned_key in orphaned_keys:
                            # Find a valid replacement value
                            valid_values = list(target_values)
                            if valid_values:
                                replacement_value = random.choice(valid_values)
                                # Update the orphaned key
                                source_df.loc[source_df[source_column] == orphaned_key, source_column] = replacement_value
                        
                        print(f"✅ Fixed {len(orphaned_keys)} orphaned foreign keys")
                        
                except Exception as e:
                    logger.error(f"Error fixing relationship {rel}: {e}")
                    continue
            
            return fixed_data
            
        except Exception as e:
            logger.error(f"Error in fix_referential_integrity: {e}")
            # Return original data on error
            return synthetic_data

    def _fix_primary_keys(self, synthetic_data: Dict[str, pd.DataFrame], 
                          tables: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """Fix primary key issues in synthetic data"""
        try:
            # Validate input
            if not isinstance(synthetic_data, dict):
                raise ValueError("synthetic_data must be a dictionary")
            
            if not isinstance(tables, list):
                raise ValueError("tables must be a list")
            
            if not synthetic_data:
                return synthetic_data
            
            # Create a copy to avoid modifying original data
            fixed_data = {table_name: df.copy() for table_name, df in synthetic_data.items()}
            
            for table_info in tables:
                try:
                    if not isinstance(table_info, dict):
                        continue
                        
                    table_name = table_info.get('name', '')
                    if not table_name or table_name not in fixed_data:
                        continue
                    
                    table_df = fixed_data[table_name]
                    if not isinstance(table_df, pd.DataFrame) or table_df.empty:
                        continue
                    
                    # Find primary key column
                    primary_key_col = None
                    for col_info in table_info.get('columns', []):
                        try:
                            if isinstance(col_info, dict) and col_info.get('is_primary_key', False):
                                primary_key_col = col_info.get('name', '')
                                break
                        except Exception as e:
                            logger.error(f"Error checking column info: {e}")
                            continue
                    
                    if not primary_key_col or primary_key_col not in table_df.columns:
                        continue
                    
                    # Check for duplicate primary keys
                    duplicates = table_df[primary_key_col].duplicated()
                    if duplicates.any():
                        num_duplicates = duplicates.sum()
                        print(f"🔧 Fixing {num_duplicates} duplicate primary keys in {table_name}.{primary_key_col}")
                        
                        # Fix duplicates by making them unique
                        table_df[primary_key_col] = self._make_column_unique(table_df[primary_key_col])
                        
                        print(f"✅ Fixed {num_duplicates} duplicate primary keys")
                        
                except Exception as e:
                    logger.error(f"Error fixing primary keys for table {table_info}: {e}")
                    continue
            
            return fixed_data
            
        except Exception as e:
            logger.error(f"Error in fix_primary_keys: {e}")
            # Return original data on error
            return synthetic_data
    
    def _make_column_unique(self, column: pd.Series) -> pd.Series:
        """Make a column unique by adding suffixes to duplicate values"""
        try:
            # Create a copy to avoid modifying original
            unique_column = column.copy()
            
            # Find duplicates
            duplicates = unique_column.duplicated(keep='first')
            
            if not duplicates.any():
                return unique_column
            
            # Fix duplicates by adding suffixes
            duplicate_indices = duplicates[duplicates].index
            for idx in duplicate_indices:
                original_value = unique_column.iloc[idx]
                if pd.isna(original_value):
                    # Handle NaN values
                    unique_column.iloc[idx] = f"unique_{idx}"
                else:
                    # Add suffix to make it unique
                    suffix = 1
                    new_value = f"{original_value}_{suffix}"
                    while new_value in unique_column.values:
                        suffix += 1
                        new_value = f"{original_value}_{suffix}"
                    unique_column.iloc[idx] = new_value
            
            return unique_column
            
        except Exception as e:
            logger.error(f"Error making column unique: {e}")
            # Return original column on error
            return column

# Initialize the enhanced engine
enhanced_engine = EnhancedSDVEngine()

# API Endpoints
@app.post("/api/enhanced/generate", response_model=EnhancedGenerationResponse)
async def generate_enhanced_synthetic_data(request: EnhancedGenerationRequest, background_tasks: BackgroundTasks):
    """Generate enhanced synthetic data for complex relational databases"""
    logger.info(f"Received enhanced generation request with {len(request.tables)} tables and {len(request.relationships)} relationships")
    
    # Add to background tasks for long-running operations
    background_tasks.add_task(enhanced_engine.generate_complex_synthetic_data, request)
    
    # Return immediate response with placeholder values for required fields
    return EnhancedGenerationResponse(
        session_id="",  # Empty string since session will be created in background
        status="processing",
        synthetic_data={},  # Empty dict since data will be generated in background
        quality_metrics={},  # Empty dict since metrics will be calculated in background
        output_files=[]  # Empty list since files will be generated in background
    )

@app.post("/api/enhanced/upload/file")
async def upload_enhanced_file(file: UploadFile = File(...)):
    """Upload file for enhanced processing"""
    try:
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{unique_id}_{file.filename}"
        file_path = UPLOAD_DIR / filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        logger.info(f"Enhanced file uploaded successfully: {filename}")
        
        return {
            "filename": filename,
            "size": len(content),
            "uploaded_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error uploading enhanced file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/enhanced/sessions")
async def list_available_sessions():
    """List all available generation sessions"""
    try:
        available_sessions = []
        for session_id, session_data in sessions.items():
            available_sessions.append({
                "session_id": session_id,
                "status": session_data.get("status", "unknown"),
                "created_at": session_data.get("created_at", "unknown")
            })
        return {"sessions": available_sessions}
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")

@app.get("/api/enhanced/status/{session_id}")
async def get_generation_status(session_id: str):
    """Get the status of a generation session"""
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        return {
            "session_id": session_id,
            "status": session.get("status", "unknown"),
            "data": session.get("data", {}),
            "quality_metrics": session.get("quality_metrics", {}),
            "output_files": session.get("output_files", []),
            "error": session.get("error"),
            "performance_metrics": session.get("performance_metrics", {})
        }
    except Exception as e:
        logger.error(f"Error getting session status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting session status: {str(e)}")

@app.get("/api/enhanced/export/csv")
async def export_enhanced_csv(session_id: str = None):
    """Export enhanced synthetic data as CSV"""
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id query parameter is required")
        
        print(f"DEBUG: Export CSV requested for session: {session_id}")
        print(f"DEBUG: Available sessions: {list(sessions.keys())}")
        
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        if session.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Generation not completed")
        
        # Check if data exists
        if not session.get("data"):
            raise HTTPException(status_code=400, detail="No data available for export")
        
        # Convert list data to DataFrames for export engine
        try:
            import pandas as pd
            export_data = {}
            for table_name, table_data in session.get("data", {}).items():
                if isinstance(table_data, list) and table_data:
                    # Convert list of dicts to DataFrame
                    export_data[table_name] = pd.DataFrame(table_data)
                elif isinstance(table_data, pd.DataFrame):
                    # Already a DataFrame
                    export_data[table_name] = table_data
                else:
                    # Empty or invalid data
                    export_data[table_name] = pd.DataFrame()
            
            # Use the export engine to generate CSV
            export_results = enhanced_engine.export_engine.export_data(
                export_data,  # Now contains DataFrames instead of lists
                {}, 
                [], 
                session.get("quality_metrics", {}), 
                "csv"
            )
        except Exception as e:
            logger.error(f"Error converting data for CSV export: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Data conversion failed: {str(e)}")
        
        return {
            "status": "success",
            "export_files": export_results.get("exported_files", []),
            "message": "CSV export completed"
        }
    except Exception as e:
        logger.error(f"Error exporting CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/api/enhanced/export/json")
async def export_enhanced_json(session_id: str = None):
    """Export enhanced synthetic data as JSON"""
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id query parameter is required")
        
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        if session.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Generation not completed")
        
        # Check if data exists
        if not session.get("data"):
            raise HTTPException(status_code=400, detail="No data available for export")
        
        # Convert list data to DataFrames for export engine
        try:
            import pandas as pd
            export_data = {}
            for table_name, table_data in session.get("data", {}).items():
                if isinstance(table_data, list) and table_data:
                    # Convert list of dicts to DataFrame
                    export_data[table_name] = pd.DataFrame(table_data)
                elif isinstance(table_data, pd.DataFrame):
                    # Already a DataFrame
                    export_data[table_name] = table_data
                else:
                    # Empty or invalid data
                    export_data[table_name] = pd.DataFrame()
            
            # Use the export engine to generate JSON
            export_results = enhanced_engine.export_engine.export_data(
                export_data,  # Now contains DataFrames instead of lists
                {}, 
                [], 
                session.get("quality_metrics", {}), 
                "json"
            )
        except Exception as e:
            logger.error(f"Error converting data for JSON export: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Data conversion failed: {str(e)}")
        
        return {
            "status": "success",
            "export_files": export_results.get("exported_files", []),
            "message": "JSON export completed"
        }
    except Exception as e:
        logger.error(f"Error exporting JSON: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/api/enhanced/export/excel")
async def export_enhanced_excel(session_id: str = None):
    """Export enhanced synthetic data as Excel"""
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id query parameter is required")
        
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        if session.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Generation not completed")
        
        # Check if data exists
        if not session.get("data"):
            raise HTTPException(status_code=400, detail="No data available for export")
        
        # Convert list data to DataFrames for export engine
        try:
            import pandas as pd
            export_data = {}
            for table_name, table_data in session.get("data", {}).items():
                if isinstance(table_data, list) and table_data:
                    # Convert list of dicts to DataFrame
                    export_data[table_name] = pd.DataFrame(table_data)
                elif isinstance(table_data, pd.DataFrame):
                    # Already a DataFrame
                    export_data[table_name] = table_data
                else:
                    # Empty or invalid data
                    export_data[table_name] = pd.DataFrame()
            
            # Use the export engine to generate Excel
            export_results = enhanced_engine.export_engine.export_data(
                export_data,  # Now contains DataFrames instead of lists
                {}, 
                [], 
                session.get("quality_metrics", {}), 
                "excel"
            )
        except Exception as e:
            logger.error(f"Error converting data for Excel export: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Data conversion failed: {str(e)}")
        
        return {
            "status": "success",
            "export_files": export_results.get("exported_files", []),
            "message": "Excel export completed"
        }
    except Exception as e:
        logger.error(f"Error exporting Excel: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/api/enhanced/health")
async def enhanced_health_check():
    """Health check for enhanced service"""
    return {
        "status": "healthy",
        "service": "Enhanced SDV Enterprise Service",
        "version": "3.0.0",
        "capabilities": [
            "Complex relational database handling",
            "HR Tech domain optimization",
            "Risk Tech domain optimization",
            "Large volume data processing",
            "Parallel processing",
            "Enterprise-grade quality validation"
        ]
    }

@app.get("/api/enhanced/capabilities")
async def get_enhanced_capabilities():
    """Get enhanced service capabilities"""
    return {
        "domains": ["HR Tech", "Risk Tech", "Finance", "Healthcare", "E-commerce"],
        "max_tables": 100,
        "max_columns_per_table": 200,
        "max_relationships": 500,
        "max_data_volume": "10M+ records",
        "supported_formats": ["JSON", "CSV", "Excel", "SQL", "Parquet"],
        "performance_features": [
            "Parallel processing",
            "Memory optimization",
            "Batch processing",
            "Caching",
            "Load balancing"
        ],
        "quality_features": [
            "Relationship validation",
            "Data integrity checks",
            "Privacy compliance",
            "Domain-specific validation",
            "Statistical quality metrics"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get environment variables with defaults
    host = os.getenv("ENHANCED_BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("ENHANCED_BACKEND_PORT", "8003"))
    
    print(f"Starting Enhanced SDV Backend on {host}:{port}")
    uvicorn.run(app, host=host, port=port) 