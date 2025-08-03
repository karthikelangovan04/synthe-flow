"""
Enterprise Connectors for Multiple Data Sources
"""

import pandas as pd
import asyncio
from typing import Dict, List, Any, Optional
import os
from pathlib import Path
import tempfile
import shutil

class EnterpriseConnector:
    """Enterprise-grade data connector for multiple sources"""
    
    def __init__(self):
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
    
    async def load_enterprise_data(self, data_source_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Load data from enterprise sources"""
        
        source_type = data_source_config.get('type', 'local')
        
        if source_type == 'local':
            return await self._load_local_data(data_source_config)
        elif source_type == 'postgres':
            return await self._load_postgres_data(data_source_config)
        elif source_type == 'snowflake':
            return await self._load_snowflake_data(data_source_config)
        elif source_type == 'oracle':
            return await self._load_oracle_data(data_source_config)
        elif source_type == 's3':
            return await self._load_s3_data(data_source_config)
        elif source_type == 'gcs':
            return await self._load_gcs_data(data_source_config)
        elif source_type == 'azure':
            return await self._load_azure_data(data_source_config)
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")
    
    async def _load_local_data(self, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Load data from local files"""
        data_dict = {}
        
        file_paths = config.get('file_paths', [])
        if not file_paths:
            # Look for uploaded files in the uploads directory
            for file_path in self.upload_dir.glob("*.csv"):
                file_paths.append(str(file_path))
        
        for file_path in file_paths:
            try:
                # Handle both absolute and relative paths
                if not os.path.isabs(file_path):
                    # If it's a relative path, look in uploads directory
                    full_path = self.upload_dir / file_path
                else:
                    full_path = Path(file_path)
                
                if not full_path.exists():
                    print(f"File not found: {full_path}")
                    continue
                
                if file_path.endswith('.csv'):
                    df = pd.read_csv(full_path)
                elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                    df = pd.read_excel(full_path)
                elif file_path.endswith('.json'):
                    df = pd.read_json(full_path)
                elif file_path.endswith('.parquet'):
                    df = pd.read_parquet(full_path)
                else:
                    print(f"Unsupported file format: {file_path}")
                    continue
                
                # Use filename as table name
                table_name = Path(file_path).stem
                data_dict[table_name] = df
                
                print(f"Loaded {len(df)} rows from {file_path}")
                
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
        
        return data_dict
    
    async def _load_postgres_data(self, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Load data from PostgreSQL"""
        # For MVP, return empty dict
        # In full implementation, would connect to PostgreSQL
        print("PostgreSQL connector not implemented in MVP")
        return {}
    
    async def _load_snowflake_data(self, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Load data from Snowflake"""
        # For MVP, return empty dict
        # In full implementation, would connect to Snowflake
        print("Snowflake connector not implemented in MVP")
        return {}
    
    async def _load_oracle_data(self, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Load data from Oracle"""
        # For MVP, return empty dict
        # In full implementation, would connect to Oracle
        print("Oracle connector not implemented in MVP")
        return {}
    
    async def _load_s3_data(self, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Load data from AWS S3"""
        # For MVP, return empty dict
        # In full implementation, would connect to S3
        print("S3 connector not implemented in MVP")
        return {}
    
    async def _load_gcs_data(self, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Load data from Google Cloud Storage"""
        # For MVP, return empty dict
        # In full implementation, would connect to GCS
        print("GCS connector not implemented in MVP")
        return {}
    
    async def _load_azure_data(self, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Load data from Azure Blob Storage"""
        # For MVP, return empty dict
        # In full implementation, would connect to Azure
        print("Azure connector not implemented in MVP")
        return {}
    
    def create_sample_enterprise_data(self, schema_info: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Create sample enterprise data for testing"""
        data_dict = {}
        
        for table_info in schema_info.get('tables', []):
            table_name = table_info['name']
            columns = table_info.get('columns', [])
            
            # Generate sample data based on schema
            sample_data = self._generate_sample_table_data(columns, 1000)
            data_dict[table_name] = pd.DataFrame(sample_data)
        
        return data_dict
    
    def _generate_sample_table_data(self, columns: List[Dict[str, Any]], num_rows: int) -> List[Dict[str, Any]]:
        """Generate sample data for a table"""
        import random
        import string
        from datetime import datetime, timedelta
        
        data = []
        
        for i in range(num_rows):
            row = {}
            
            for col in columns:
                col_name = col['name']
                data_type = col.get('data_type', 'varchar')
                
                if data_type == 'integer':
                    row[col_name] = random.randint(1, 10000)
                elif data_type == 'decimal':
                    row[col_name] = round(random.uniform(0, 1000), 2)
                elif data_type == 'varchar':
                    row[col_name] = f"sample_{i}_{col_name}"
                elif data_type == 'text':
                    row[col_name] = f"Sample text data for {col_name} row {i}"
                elif data_type == 'boolean':
                    row[col_name] = random.choice([True, False])
                elif data_type == 'timestamp':
                    start_date = datetime(2020, 1, 1)
                    end_date = datetime(2024, 12, 31)
                    random_date = start_date + timedelta(
                        days=random.randint(0, (end_date - start_date).days)
                    )
                    row[col_name] = random_date
                elif data_type == 'date':
                    start_date = datetime(2020, 1, 1).date()
                    end_date = datetime(2024, 12, 31).date()
                    random_date = start_date + timedelta(
                        days=random.randint(0, (end_date - start_date).days)
                    )
                    row[col_name] = random_date
                else:
                    row[col_name] = f"value_{i}"
            
            data.append(row)
        
        return data 