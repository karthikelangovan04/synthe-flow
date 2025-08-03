"""
Enterprise Export Engine for Multiple Output Formats
"""

import pandas as pd
import json
import csv
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import shutil
from datetime import datetime

class EnterpriseExporter:
    """Enterprise-grade export engine for synthetic data"""
    
    def __init__(self):
        self.supported_formats = ['json', 'csv', 'excel', 'sql', 'parquet', 'xml']
        self.export_dir = Path("exports")
        self.export_dir.mkdir(exist_ok=True)
    
    async def export_enterprise_data(self, synthetic_data: Dict[str, pd.DataFrame], 
                                   output_format: str, session_id: str) -> List[str]:
        """Export synthetic data in enterprise formats"""
        
        print(f"Exporting data in {output_format} format...")
        
        if output_format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {output_format}. Supported: {self.supported_formats}")
        
        export_files = []
        
        if output_format == 'json':
            export_files = await self._export_json(synthetic_data, session_id)
        elif output_format == 'csv':
            export_files = await self._export_csv(synthetic_data, session_id)
        elif output_format == 'excel':
            export_files = await self._export_excel(synthetic_data, session_id)
        elif output_format == 'sql':
            export_files = await self._export_sql(synthetic_data, session_id)
        elif output_format == 'parquet':
            export_files = await self._export_parquet(synthetic_data, session_id)
        elif output_format == 'xml':
            export_files = await self._export_xml(synthetic_data, session_id)
        
        print(f"Exported {len(export_files)} files successfully")
        return export_files
    
    async def _export_json(self, synthetic_data: Dict[str, pd.DataFrame], 
                          session_id: str) -> List[str]:
        """Export data as JSON"""
        export_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for table_name, df in synthetic_data.items():
            filename = f"{session_id}_{table_name}_{timestamp}.json"
            filepath = self.export_dir / filename
            
            # Convert DataFrame to JSON
            json_data = {
                'table_name': table_name,
                'record_count': len(df),
                'columns': list(df.columns),
                'data': df.to_dict('records'),
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'session_id': session_id,
                    'format': 'json'
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            export_files.append(str(filepath))
        
        return export_files
    
    async def _export_csv(self, synthetic_data: Dict[str, pd.DataFrame], 
                         session_id: str) -> List[str]:
        """Export data as CSV files"""
        export_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for table_name, df in synthetic_data.items():
            filename = f"{session_id}_{table_name}_{timestamp}.csv"
            filepath = self.export_dir / filename
            
            # Export DataFrame to CSV
            df.to_csv(filepath, index=False)
            export_files.append(str(filepath))
        
        return export_files
    
    async def _export_excel(self, synthetic_data: Dict[str, pd.DataFrame], 
                           session_id: str) -> List[str]:
        """Export data as Excel file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{session_id}_synthetic_data_{timestamp}.xlsx"
        filepath = self.export_dir / filename
        
        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for table_name, df in synthetic_data.items():
                # Write each table to a separate sheet
                df.to_excel(writer, sheet_name=table_name, index=False)
        
        return [str(filepath)]
    
    async def _export_sql(self, synthetic_data: Dict[str, pd.DataFrame], 
                         session_id: str) -> List[str]:
        """Export data as SQL files"""
        export_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for table_name, df in synthetic_data.items():
            filename = f"{session_id}_{table_name}_{timestamp}.sql"
            filepath = self.export_dir / filename
            
            # Generate SQL INSERT statements
            sql_content = self._generate_sql_inserts(table_name, df)
            
            with open(filepath, 'w') as f:
                f.write(sql_content)
            
            export_files.append(str(filepath))
        
        return export_files
    
    async def _export_parquet(self, synthetic_data: Dict[str, pd.DataFrame], 
                             session_id: str) -> List[str]:
        """Export data as Parquet files"""
        export_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for table_name, df in synthetic_data.items():
            filename = f"{session_id}_{table_name}_{timestamp}.parquet"
            filepath = self.export_dir / filename
            
            # Export DataFrame to Parquet
            df.to_parquet(filepath, index=False)
            export_files.append(str(filepath))
        
        return export_files
    
    async def _export_xml(self, synthetic_data: Dict[str, pd.DataFrame], 
                         session_id: str) -> List[str]:
        """Export data as XML files"""
        export_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for table_name, df in synthetic_data.items():
            filename = f"{session_id}_{table_name}_{timestamp}.xml"
            filepath = self.export_dir / filename
            
            # Generate XML content
            xml_content = self._generate_xml(table_name, df)
            
            with open(filepath, 'w') as f:
                f.write(xml_content)
            
            export_files.append(str(filepath))
        
        return export_files
    
    def _generate_sql_inserts(self, table_name: str, df: pd.DataFrame) -> str:
        """Generate SQL INSERT statements"""
        sql_lines = []
        
        # Add header comment
        sql_lines.append(f"-- Synthetic data for table: {table_name}")
        sql_lines.append(f"-- Generated at: {datetime.now().isoformat()}")
        sql_lines.append(f"-- Record count: {len(df)}")
        sql_lines.append("")
        
        # Generate INSERT statements
        for _, row in df.iterrows():
            columns = list(df.columns)
            values = []
            
            for col in columns:
                value = row[col]
                if pd.isna(value):
                    values.append("NULL")
                elif isinstance(value, (int, float)):
                    values.append(str(value))
                else:
                    # Escape single quotes and wrap in quotes
                    escaped_value = str(value).replace("'", "''")
                    values.append(f"'{escaped_value}'")
            
            insert_stmt = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(values)});"
            sql_lines.append(insert_stmt)
        
        return "\n".join(sql_lines)
    
    def _generate_xml(self, table_name: str, df: pd.DataFrame) -> str:
        """Generate XML content"""
        xml_lines = []
        
        # XML header
        xml_lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        xml_lines.append(f'<synthetic_data table="{table_name}" generated_at="{datetime.now().isoformat()}">')
        
        # Add records
        for _, row in df.iterrows():
            xml_lines.append("  <record>")
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    xml_lines.append(f'    <{col} xsi:nil="true" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"/>')
                else:
                    xml_lines.append(f'    <{col}>{str(value)}</{col}>')
            xml_lines.append("  </record>")
        
        xml_lines.append("</synthetic_data>")
        
        return "\n".join(xml_lines)
    
    def create_metadata_file(self, synthetic_data: Dict[str, pd.DataFrame], 
                           session_id: str, quality_metrics: Optional[Dict[str, Any]] = None) -> str:
        """Create metadata file for the export"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{session_id}_metadata_{timestamp}.json"
        filepath = self.export_dir / filename
        
        metadata = {
            'session_id': session_id,
            'exported_at': datetime.now().isoformat(),
            'tables': {},
            'quality_metrics': quality_metrics or {},
            'export_info': {
                'total_tables': len(synthetic_data),
                'total_records': sum(len(df) for df in synthetic_data.values()),
                'supported_formats': self.supported_formats
            }
        }
        
        # Add table-specific metadata
        for table_name, df in synthetic_data.items():
            metadata['tables'][table_name] = {
                'record_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return str(filepath) 