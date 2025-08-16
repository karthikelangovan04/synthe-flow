"""
Enhanced Enterprise Export Engine
Supports multiple formats with quality reporting and metadata
"""

import pandas as pd
import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import zipfile
import tempfile
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EnterpriseExporter:
    """Enhanced exporter for enterprise synthetic data"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'json', 'parquet', 'excel', 'sql', 'zip']
        self.export_dir = Path("exports")
        self.export_dir.mkdir(exist_ok=True)
    
    def export_data(self, synthetic_data: Dict[str, pd.DataFrame], 
                   original_data: Dict[str, pd.DataFrame],
                   relationships: List[Dict[str, Any]],
                   quality_metrics: Dict[str, Any],
                   export_format: str = 'json',
                   include_metadata: bool = True,
                   include_quality_report: bool = True) -> Dict[str, Any]:
        """Export synthetic data with enhanced features"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"export_{timestamp}"
        
        export_results = {
            'session_id': session_id,
            'timestamp': timestamp,
            'exported_files': [],
            'format': export_format,
            'metadata': {},
            'quality_summary': {}
        }
        
        try:
            if export_format == 'json':
                files = self._export_json(synthetic_data, session_id, include_metadata, quality_metrics)
            elif export_format == 'csv':
                files = self._export_csv(synthetic_data, session_id, include_metadata, quality_metrics)
            elif export_format == 'parquet':
                files = self._export_parquet(synthetic_data, session_id, include_metadata, quality_metrics)
            elif export_format == 'excel':
                files = self._export_excel(synthetic_data, session_id, include_metadata, quality_metrics)
            elif export_format == 'sql':
                files = self._export_sql(synthetic_data, relationships, session_id, include_metadata, quality_metrics)
            elif export_format == 'zip':
                files = self._export_zip(synthetic_data, relationships, session_id, include_metadata, quality_metrics)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            export_results['exported_files'] = files
            
            # Add metadata
            if include_metadata:
                export_results['metadata'] = self._generate_metadata(synthetic_data, relationships, quality_metrics)
            
            # Add quality summary
            if include_quality_report:
                export_results['quality_summary'] = self._generate_quality_summary(quality_metrics)
            
            logger.info(f"Successfully exported {len(files)} files in {export_format} format")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            export_results['error'] = str(e)
        
        return export_results
    
    def _export_json(self, synthetic_data: Dict[str, pd.DataFrame], session_id: str, 
                     include_metadata: bool, quality_metrics: Dict[str, Any]) -> List[str]:
        """Export data in JSON format with enhanced structure"""
        exported_files = []
        
        for table_name, df in synthetic_data.items():
            # Convert DataFrame to records
            records = df.to_dict('records')
            
            # Create enhanced JSON structure
            json_data = {
                'table_name': table_name,
                'row_count': len(records),
                'column_count': len(df.columns),
                'columns': list(df.columns),
                'data_types': df.dtypes.astype(str).to_dict(),
                'data': records
            }
            
            if include_metadata:
                json_data['metadata'] = {
                    'export_timestamp': datetime.now().isoformat(),
                    'session_id': session_id,
                    'quality_score': quality_metrics.get('overall_score', 0.0)
                }
            
            # Export to file
            filename = f"{session_id}_{table_name}.json"
            filepath = self.export_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            exported_files.append(str(filepath))
        
        return exported_files
    
    def _export_csv(self, synthetic_data: Dict[str, pd.DataFrame], session_id: str,
                    include_metadata: bool, quality_metrics: Dict[str, Any]) -> List[str]:
        """Export data in CSV format with metadata"""
        exported_files = []
        
        for table_name, df in synthetic_data.items():
            # Export main data
            filename = f"{session_id}_{table_name}.csv"
            filepath = self.export_dir / filename
            
            df.to_csv(filepath, index=False)
            exported_files.append(str(filepath))
            
            # Export metadata if requested
            if include_metadata:
                metadata_filename = f"{session_id}_{table_name}_metadata.csv"
                metadata_filepath = self.export_dir / metadata_filename
                
                metadata_df = pd.DataFrame([
                    {'attribute': 'table_name', 'value': table_name},
                    {'attribute': 'row_count', 'value': len(df)},
                    {'attribute': 'column_count', 'value': len(df.columns)},
                    {'attribute': 'export_timestamp', 'value': datetime.now().isoformat()},
                    {'attribute': 'session_id', 'value': session_id},
                    {'attribute': 'quality_score', 'value': quality_metrics.get('overall_score', 0.0)}
                ])
                
                metadata_df.to_csv(metadata_filepath, index=False)
                exported_files.append(str(metadata_filepath))
        
        return exported_files
    
    def _export_parquet(self, synthetic_data: Dict[str, pd.DataFrame], session_id: str,
                        include_metadata: bool, quality_metrics: Dict[str, Any]) -> List[str]:
        """Export data in Parquet format for efficient storage"""
        exported_files = []
        
        for table_name, df in synthetic_data.items():
            filename = f"{session_id}_{table_name}.parquet"
            filepath = self.export_dir / filename
            
            df.to_parquet(filepath, index=False)
            exported_files.append(str(filepath))
        
        return exported_files
    
    def _export_excel(self, synthetic_data: Dict[str, pd.DataFrame], session_id: str,
                      include_metadata: bool, quality_metrics: Dict[str, Any]) -> List[str]:
        """Export data in Excel format with multiple sheets"""
        exported_files = []
        
        for table_name, df in synthetic_data.items():
            filename = f"{session_id}_{table_name}.xlsx"
            filepath = self.export_dir / filename
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Data', index=False)
                
                # Metadata sheet
                if include_metadata:
                    metadata_data = {
                        'Attribute': ['Table Name', 'Row Count', 'Column Count', 'Export Timestamp', 'Session ID', 'Quality Score'],
                        'Value': [table_name, len(df), len(df.columns), datetime.now().isoformat(), session_id, quality_metrics.get('overall_score', 0.0)]
                    }
                    metadata_df = pd.DataFrame(metadata_data)
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                
                # Quality metrics sheet
                if quality_metrics:
                    self._export_quality_to_excel(writer, quality_metrics, table_name)
            
            exported_files.append(str(filepath))
        
        return exported_files
    
    def _export_sql(self, synthetic_data: Dict[str, pd.DataFrame], relationships: List[Dict[str, Any]],
                    session_id: str, include_metadata: bool, quality_metrics: Dict[str, Any]) -> List[str]:
        """Export data as SQL INSERT statements with schema"""
        exported_files = []
        
        for table_name, df in synthetic_data.items():
            filename = f"{session_id}_{table_name}.sql"
            filepath = self.export_dir / filename
            
            with open(filepath, 'w') as f:
                # Write header
                f.write(f"-- Synthetic Data Export for {table_name}\n")
                f.write(f"-- Generated: {datetime.now().isoformat()}\n")
                f.write(f"-- Session ID: {session_id}\n")
                f.write(f"-- Quality Score: {quality_metrics.get('overall_score', 0.0)}\n\n")
                
                # Write CREATE TABLE statement (simplified)
                columns = df.columns.tolist()
                f.write(f"CREATE TABLE IF NOT EXISTS {table_name} (\n")
                for i, col in enumerate(columns):
                    col_type = self._infer_sql_type(df[col])
                    f.write(f"    {col} {col_type}")
                    if i < len(columns) - 1:
                        f.write(",")
                    f.write("\n")
                f.write(");\n\n")
                
                # Write INSERT statements
                for _, row in df.iterrows():
                    values = []
                    for col in columns:
                        val = row[col]
                        if pd.isna(val):
                            values.append("NULL")
                        elif isinstance(val, str):
                            values.append(f"'{val.replace(chr(39), chr(39)+chr(39))}'")
                        else:
                            values.append(str(val))
                    
                    f.write(f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(values)});\n")
            
            exported_files.append(str(filepath))
        
        return exported_files
    
    def _export_zip(self, synthetic_data: Dict[str, pd.DataFrame], relationships: List[Dict[str, Any]],
                    session_id: str, include_metadata: bool, quality_metrics: Dict[str, Any]) -> List[str]:
        """Export all data in a ZIP archive with comprehensive structure"""
        zip_filename = f"{session_id}_complete_export.zip"
        zip_filepath = self.export_dir / zip_filename
        
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add data files
            for table_name, df in synthetic_data.items():
                # CSV data
                csv_filename = f"{table_name}.csv"
                csv_filepath = self.export_dir / csv_filename
                df.to_csv(csv_filepath, index=False)
                zipf.write(csv_filepath, csv_filename)
                
                # Clean up temp file
                csv_filepath.unlink()
            
            # Add metadata
            if include_metadata:
                metadata = self._generate_metadata(synthetic_data, relationships, quality_metrics)
                metadata_filename = "metadata.json"
                metadata_filepath = self.export_dir / metadata_filename
                
                with open(metadata_filepath, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                zipf.write(metadata_filepath, metadata_filename)
                metadata_filepath.unlink()
            
            # Add quality report
            if quality_metrics:
                quality_filename = "quality_report.json"
                quality_filepath = self.export_dir / quality_filename
                
                with open(quality_filepath, 'w') as f:
                    json.dump(quality_metrics, f, indent=2, default=str)
                
                zipf.write(quality_filepath, quality_filename)
                quality_filepath.unlink()
            
            # Add relationships
            if relationships:
                relationships_filename = "relationships.json"
                relationships_filepath = self.export_dir / relationships_filename
                
                with open(relationships_filepath, 'w') as f:
                    json.dump(relationships, f, indent=2, default=str)
                
                zipf.write(relationships_filepath, relationships_filename)
                relationships_filepath.unlink()
        
        return [str(zip_filepath)]
    
    def _export_quality_to_excel(self, writer: pd.ExcelWriter, quality_metrics: Dict[str, Any], table_name: str):
        """Export quality metrics to Excel sheet"""
        # Table scores
        if 'table_scores' in quality_metrics and table_name in quality_metrics['table_scores']:
            table_scores = quality_metrics['table_scores'][table_name]
            if isinstance(table_scores, dict):
                scores_data = []
                for metric, value in table_scores.items():
                    if isinstance(value, dict) and 'score' in value:
                        scores_data.append({'Metric': metric, 'Score': value['score']})
                    elif isinstance(value, (int, float)):
                        scores_data.append({'Metric': metric, 'Score': value})
                
                if scores_data:
                    scores_df = pd.DataFrame(scores_data)
                    scores_df.to_excel(writer, sheet_name='Quality_Scores', index=False)
    
    def _generate_metadata(self, synthetic_data: Dict[str, pd.DataFrame], 
                          relationships: List[Dict[str, Any]], 
                          quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata"""
        metadata = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'total_tables': len(synthetic_data),
                'total_rows': sum(len(df) for df in synthetic_data.values()),
                'total_columns': sum(len(df.columns) for df in synthetic_data.values())
            },
            'tables': {},
            'relationships': relationships,
            'quality_overview': {
                'overall_score': quality_metrics.get('overall_score', 0.0),
                'table_count': len(quality_metrics.get('table_scores', {})),
                'has_privacy_metrics': 'privacy_metrics' in quality_metrics,
                'has_statistical_metrics': 'statistical_metrics' in quality_metrics
            }
        }
        
        for table_name, df in synthetic_data.items():
            metadata['tables'][table_name] = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': {
                    col: {
                        'data_type': str(df[col].dtype),
                        'null_count': df[col].isnull().sum(),
                        'unique_count': df[col].nunique()
                    } for col in df.columns
                }
            }
        
        return metadata
    
    def _generate_quality_summary(self, quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality summary for export"""
        summary = {
            'overall_score': quality_metrics.get('overall_score', 0.0),
            'table_scores': {},
            'privacy_score': 0.0,
            'statistical_score': 0.0,
            'recommendations': []
        }
        
        # Extract table scores
        if 'table_scores' in quality_metrics:
            for table_name, scores in quality_metrics['table_scores'].items():
                if isinstance(scores, dict):
                    summary['table_scores'][table_name] = scores.get('overall_score', 0.0)
        
        # Extract privacy and statistical scores
        if 'privacy_metrics' in quality_metrics:
            summary['privacy_score'] = quality_metrics['privacy_metrics'].get('overall_privacy_score', 0.0)
        
        if 'statistical_metrics' in quality_metrics:
            summary['statistical_score'] = quality_metrics['statistical_metrics'].get('overall_statistical_score', 0.0)
        
        # Collect recommendations
        for metric_type in ['privacy_metrics', 'statistical_metrics']:
            if metric_type in quality_metrics:
                recommendations = quality_metrics[metric_type].get('recommendations', [])
                summary['recommendations'].extend(recommendations)
        
        return summary
    
    def _infer_sql_type(self, series: pd.Series) -> str:
        """Infer SQL data type from pandas series"""
        dtype = str(series.dtype)
        
        if 'int' in dtype:
            return 'INTEGER'
        elif 'float' in dtype:
            return 'REAL'
        elif 'bool' in dtype:
            return 'BOOLEAN'
        elif 'datetime' in dtype:
            return 'TIMESTAMP'
        else:
            return 'TEXT' 