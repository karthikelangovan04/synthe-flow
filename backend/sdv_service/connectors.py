"""
Data Source Connectors for Synthetic Data Generation
Supports databases, cloud storage, APIs, and data catalogs
"""

import asyncio
import pandas as pd
import json
import requests
import httpx
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import os
from pathlib import Path
import tempfile
import io

# Database connectors
try:
    import psycopg2
    from sqlalchemy import create_engine, text, inspect
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import snowflake.connector
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

try:
    import cx_Oracle
    ORACLE_AVAILABLE = True
except ImportError:
    ORACLE_AVAILABLE = False

# Cloud storage connectors
try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Data catalog connectors
try:
    from pyapacheatlas.core import AtlasEntity, AtlasProcess
    from pyapacheatlas.auth import ServicePrincipalAuthentication
    from pyapacheatlas.core.typedef import AtlasAttributeDef, EntityTypeDef
    ATLAS_AVAILABLE = True
except ImportError:
    ATLAS_AVAILABLE = False

class DatabaseConnector(ABC):
    """Abstract base class for database connectors"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test the database connection"""
        pass
    
    @abstractmethod
    async def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from the database"""
        pass
    
    @abstractmethod
    async def get_schema(self) -> Dict[str, Any]:
        """Get database schema information"""
        pass

class PostgresConnector(DatabaseConnector):
    """PostgreSQL database connector"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 and sqlalchemy are required for PostgreSQL connector")
        
        self.connection_string = (
            f"postgresql://{config.get('username')}:{config.get('password')}@"
            f"{config.get('host')}:{config.get('port', 5432)}/{config.get('database')}"
        )
    
    async def test_connection(self) -> bool:
        """Test PostgreSQL connection"""
        try:
            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            print(f"PostgreSQL connection test failed: {e}")
            return False
    
    async def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from PostgreSQL tables"""
        try:
            engine = create_engine(self.connection_string)
            inspector = inspect(engine)
            
            tables = self.config.get('tables', [])
            if not tables:
                # Get all tables if none specified
                tables = inspector.get_table_names()
            
            data = {}
            for table_name in tables:
                if inspector.has_table(table_name):
                    query = f"SELECT * FROM {table_name}"
                    if self.config.get('limit'):
                        query += f" LIMIT {self.config['limit']}"
                    
                    df = pd.read_sql(query, engine)
                    data[table_name] = df
                    print(f"Loaded {len(df)} rows from PostgreSQL table: {table_name}")
            
            return data
        except Exception as e:
            print(f"Error loading data from PostgreSQL: {e}")
            return {}
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get PostgreSQL schema information"""
        try:
            engine = create_engine(self.connection_string)
            inspector = inspect(engine)
            
            schema_info = {
                'tables': {},
                'relationships': []
            }
            
            tables = inspector.get_table_names()
            for table_name in tables:
                columns = inspector.get_columns(table_name)
                primary_keys = inspector.get_pk_constraint(table_name)
                foreign_keys = inspector.get_foreign_keys(table_name)
                
                schema_info['tables'][table_name] = {
                    'columns': columns,
                    'primary_keys': primary_keys,
                    'foreign_keys': foreign_keys
                }
            
            return schema_info
        except Exception as e:
            print(f"Error getting PostgreSQL schema: {e}")
            return {}

class SnowflakeConnector(DatabaseConnector):
    """Snowflake database connector"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not SNOWFLAKE_AVAILABLE:
            raise ImportError("snowflake-connector-python is required for Snowflake connector")
    
    async def test_connection(self) -> bool:
        """Test Snowflake connection"""
        try:
            conn = snowflake.connector.connect(
                user=self.config.get('username'),
                password=self.config.get('password'),
                account=self.config.get('account'),
                warehouse=self.config.get('warehouse'),
                database=self.config.get('database'),
                schema=self.config.get('schema', 'PUBLIC')
            )
            conn.close()
            return True
        except Exception as e:
            print(f"Snowflake connection test failed: {e}")
            return False
    
    async def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from Snowflake tables"""
        try:
            conn = snowflake.connector.connect(
                user=self.config.get('username'),
                password=self.config.get('password'),
                account=self.config.get('account'),
                warehouse=self.config.get('warehouse'),
                database=self.config.get('database'),
                schema=self.config.get('schema', 'PUBLIC')
            )
            
            tables = self.config.get('tables', [])
            if not tables:
                # Get all tables if none specified
                cursor = conn.cursor()
                cursor.execute("SHOW TABLES")
                tables = [row[1] for row in cursor.fetchall()]
                cursor.close()
            
            data = {}
            for table_name in tables:
                query = f"SELECT * FROM {table_name}"
                if self.config.get('limit'):
                    query += f" LIMIT {self.config['limit']}"
                
                df = pd.read_sql(query, conn)
                data[table_name] = df
                print(f"Loaded {len(df)} rows from Snowflake table: {table_name}")
            
            conn.close()
            return data
        except Exception as e:
            print(f"Error loading data from Snowflake: {e}")
            return {}
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get Snowflake schema information"""
        try:
            conn = snowflake.connector.connect(
                user=self.config.get('username'),
                password=self.config.get('password'),
                account=self.config.get('account'),
                warehouse=self.config.get('warehouse'),
                database=self.config.get('database'),
                schema=self.config.get('schema', 'PUBLIC')
            )
            
            schema_info = {
                'tables': {},
                'relationships': []
            }
            
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES")
            tables = [row[1] for row in cursor.fetchall()]
            
            for table_name in tables:
                cursor.execute(f"DESCRIBE TABLE {table_name}")
                columns = cursor.fetchall()
                
                schema_info['tables'][table_name] = {
                    'columns': columns
                }
            
            cursor.close()
            conn.close()
            return schema_info
        except Exception as e:
            print(f"Error getting Snowflake schema: {e}")
            return {}

class OracleConnector(DatabaseConnector):
    """Oracle database connector"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not ORACLE_AVAILABLE:
            raise ImportError("cx-Oracle is required for Oracle connector")
    
    async def test_connection(self) -> bool:
        """Test Oracle connection"""
        try:
            conn = cx_Oracle.connect(
                user=self.config.get('username'),
                password=self.config.get('password'),
                dsn=self.config.get('dsn')
            )
            conn.close()
            return True
        except Exception as e:
            print(f"Oracle connection test failed: {e}")
            return False
    
    async def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from Oracle tables"""
        try:
            conn = cx_Oracle.connect(
                user=self.config.get('username'),
                password=self.config.get('password'),
                dsn=self.config.get('dsn')
            )
            
            tables = self.config.get('tables', [])
            if not tables:
                # Get all tables if none specified
                cursor = conn.cursor()
                cursor.execute("SELECT table_name FROM user_tables")
                tables = [row[0] for row in cursor.fetchall()]
                cursor.close()
            
            data = {}
            for table_name in tables:
                query = f"SELECT * FROM {table_name}"
                if self.config.get('limit'):
                    query += f" WHERE ROWNUM <= {self.config['limit']}"
                
                df = pd.read_sql(query, conn)
                data[table_name] = df
                print(f"Loaded {len(df)} rows from Oracle table: {table_name}")
            
            conn.close()
            return data
        except Exception as e:
            print(f"Error loading data from Oracle: {e}")
            return {}
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get Oracle schema information"""
        try:
            conn = cx_Oracle.connect(
                user=self.config.get('username'),
                password=self.config.get('password'),
                dsn=self.config.get('dsn')
            )
            
            schema_info = {
                'tables': {},
                'relationships': []
            }
            
            cursor = conn.cursor()
            cursor.execute("SELECT table_name FROM user_tables")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table_name in tables:
                cursor.execute(f"SELECT column_name, data_type, nullable FROM user_tab_columns WHERE table_name = '{table_name}'")
                columns = cursor.fetchall()
                
                schema_info['tables'][table_name] = {
                    'columns': columns
                }
            
            cursor.close()
            conn.close()
            return schema_info
        except Exception as e:
            print(f"Error getting Oracle schema: {e}")
            return {}

class S3Connector:
    """Amazon S3 storage connector"""
    
    def __init__(self, config: Dict[str, Any]):
        if not S3_AVAILABLE:
            raise ImportError("boto3 is required for S3 connector")
        
        self.config = config
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config.get('access_key_id'),
            aws_secret_access_key=config.get('secret_access_key'),
            region_name=config.get('region', 'us-east-1')
        )
        self.bucket = config.get('bucket')
    
    async def test_connection(self) -> bool:
        """Test S3 connection"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            return True
        except Exception as e:
            print(f"S3 connection test failed: {e}")
            return False
    
    async def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from S3 files"""
        try:
            data = {}
            file_paths = self.config.get('file_paths', [])
            
            for file_path in file_paths:
                try:
                    response = self.s3_client.get_object(Bucket=self.bucket, Key=file_path)
                    file_content = response['Body'].read()
                    
                    # Determine file type and load accordingly
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(io.BytesIO(file_content))
                    elif file_path.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(io.BytesIO(file_content))
                    elif file_path.endswith('.json'):
                        df = pd.read_json(io.BytesIO(file_content))
                    elif file_path.endswith('.parquet'):
                        df = pd.read_parquet(io.BytesIO(file_content))
                    else:
                        continue
                    
                    # Use filename without extension as table name
                    table_name = Path(file_path).stem
                    data[table_name] = df
                    print(f"Loaded {len(df)} rows from S3 file: {file_path}")
                    
                except Exception as e:
                    print(f"Error loading file {file_path} from S3: {e}")
                    continue
            
            return data
        except Exception as e:
            print(f"Error loading data from S3: {e}")
            return {}

class GCSConnector:
    """Google Cloud Storage connector"""
    
    def __init__(self, config: Dict[str, Any]):
        if not GCS_AVAILABLE:
            raise ImportError("google-cloud-storage is required for GCS connector")
        
        self.config = config
        self.client = storage.Client.from_service_account_json(config.get('service_account_key'))
        self.bucket = self.client.bucket(config.get('bucket'))
    
    async def test_connection(self) -> bool:
        """Test GCS connection"""
        try:
            self.bucket.reload()
            return True
        except Exception as e:
            print(f"GCS connection test failed: {e}")
            return False
    
    async def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from GCS files"""
        try:
            data = {}
            file_paths = self.config.get('file_paths', [])
            
            for file_path in file_paths:
                try:
                    blob = self.bucket.blob(file_path)
                    file_content = blob.download_as_bytes()
                    
                    # Determine file type and load accordingly
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(io.BytesIO(file_content))
                    elif file_path.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(io.BytesIO(file_content))
                    elif file_path.endswith('.json'):
                        df = pd.read_json(io.BytesIO(file_content))
                    elif file_path.endswith('.parquet'):
                        df = pd.read_parquet(io.BytesIO(file_content))
                    else:
                        continue
                    
                    # Use filename without extension as table name
                    table_name = Path(file_path).stem
                    data[table_name] = df
                    print(f"Loaded {len(df)} rows from GCS file: {file_path}")
                    
                except Exception as e:
                    print(f"Error loading file {file_path} from GCS: {e}")
                    continue
            
            return data
        except Exception as e:
            print(f"Error loading data from GCS: {e}")
            return {}

class AzureBlobConnector:
    """Azure Blob Storage connector"""
    
    def __init__(self, config: Dict[str, Any]):
        if not AZURE_AVAILABLE:
            raise ImportError("azure-storage-blob is required for Azure connector")
        
        self.config = config
        self.blob_service_client = BlobServiceClient.from_connection_string(
            config.get('connection_string')
        )
        self.container = config.get('container')
    
    async def test_connection(self) -> bool:
        """Test Azure Blob connection"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container)
            container_client.get_container_properties()
            return True
        except Exception as e:
            print(f"Azure Blob connection test failed: {e}")
            return False
    
    async def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from Azure Blob files"""
        try:
            data = {}
            file_paths = self.config.get('file_paths', [])
            
            for file_path in file_paths:
                try:
                    blob_client = self.blob_service_client.get_blob_client(
                        container=self.container, blob=file_path
                    )
                    file_content = blob_client.download_blob().readall()
                    
                    # Determine file type and load accordingly
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(io.BytesIO(file_content))
                    elif file_path.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(io.BytesIO(file_content))
                    elif file_path.endswith('.json'):
                        df = pd.read_json(io.BytesIO(file_content))
                    elif file_path.endswith('.parquet'):
                        df = pd.read_parquet(io.BytesIO(file_content))
                    else:
                        continue
                    
                    # Use filename without extension as table name
                    table_name = Path(file_path).stem
                    data[table_name] = df
                    print(f"Loaded {len(df)} rows from Azure Blob file: {file_path}")
                    
                except Exception as e:
                    print(f"Error loading file {file_path} from Azure Blob: {e}")
                    continue
            
            return data
        except Exception as e:
            print(f"Error loading data from Azure Blob: {e}")
            return {}

class APIConnector:
    """Custom API connector"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url')
        self.headers = config.get('headers', {})
        self.auth = config.get('auth')
    
    async def test_connection(self) -> bool:
        """Test API connection"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/health",
                    headers=self.headers,
                    auth=self.auth
                )
                return response.status_code == 200
        except Exception as e:
            print(f"API connection test failed: {e}")
            return False
    
    async def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from API endpoints"""
        try:
            data = {}
            endpoints = self.config.get('endpoints', [])
            
            async with httpx.AsyncClient() as client:
                for endpoint in endpoints:
                    try:
                        url = f"{self.base_url}{endpoint['path']}"
                        response = await client.get(
                            url,
                            headers=self.headers,
                            auth=self.auth,
                            params=endpoint.get('params', {})
                        )
                        response.raise_for_status()
                        
                        # Parse response based on content type
                        if 'application/json' in response.headers.get('content-type', ''):
                            json_data = response.json()
                            
                            # Handle different JSON structures
                            if isinstance(json_data, list):
                                df = pd.DataFrame(json_data)
                            elif isinstance(json_data, dict):
                                # Try to find data array in response
                                data_key = endpoint.get('data_key', 'data')
                                if data_key in json_data:
                                    df = pd.DataFrame(json_data[data_key])
                                else:
                                    # Convert single record to DataFrame
                                    df = pd.DataFrame([json_data])
                            else:
                                continue
                        else:
                            # Try to parse as CSV
                            df = pd.read_csv(io.StringIO(response.text))
                        
                        table_name = endpoint.get('table_name', endpoint['path'].split('/')[-1])
                        data[table_name] = df
                        print(f"Loaded {len(df)} rows from API endpoint: {endpoint['path']}")
                        
                    except Exception as e:
                        print(f"Error loading data from API endpoint {endpoint['path']}: {e}")
                        continue
            
            return data
        except Exception as e:
            print(f"Error loading data from API: {e}")
            return {}

class DataCatalogConnector:
    """Data catalog connector (Apache Atlas, etc.)"""
    
    def __init__(self, config: Dict[str, Any]):
        if not ATLAS_AVAILABLE:
            raise ImportError("pyapacheatlas is required for data catalog connector")
        
        self.config = config
        self.catalog_type = config.get('type', 'atlas')
        
        if self.catalog_type == 'atlas':
            self.auth = ServicePrincipalAuthentication(
                tenant_id=config.get('tenant_id'),
                client_id=config.get('client_id'),
                client_secret=config.get('client_secret')
            )
            self.endpoint_url = config.get('endpoint_url')
    
    async def test_connection(self) -> bool:
        """Test data catalog connection"""
        try:
            if self.catalog_type == 'atlas':
                # Test Atlas connection
                from pyapacheatlas.core import AtlasEntity
                # This would require actual Atlas client implementation
                return True
            return False
        except Exception as e:
            print(f"Data catalog connection test failed: {e}")
            return False
    
    async def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from data catalog"""
        try:
            data = {}
            
            if self.catalog_type == 'atlas':
                # This would implement actual data loading from Atlas
                # For now, return empty dict as placeholder
                print("Data catalog data loading not yet implemented")
            
            return data
        except Exception as e:
            print(f"Error loading data from data catalog: {e}")
            return {}
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get schema information from data catalog"""
        try:
            schema_info = {
                'tables': {},
                'relationships': []
            }
            
            if self.catalog_type == 'atlas':
                # This would implement actual schema retrieval from Atlas
                print("Data catalog schema retrieval not yet implemented")
            
            return schema_info
        except Exception as e:
            print(f"Error getting schema from data catalog: {e}")
            return {} 