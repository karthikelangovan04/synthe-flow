"""
Complex Quality Validator for Synthetic Data Assessment
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
import json

class ComplexDataValidator:
    """Advanced quality validation for synthetic data"""
    
    def __init__(self):
        self.quality_metrics = {}
        self.thresholds = {
            'statistical_similarity': 0.8,
            'distribution_similarity': 0.7,
            'correlation_preservation': 0.8,
            'relationship_integrity': 0.95,
            'privacy_score': 0.9
        }
    
    def validate_complex_data(self, synthetic_data: Dict[str, pd.DataFrame], 
                            original_data: Dict[str, pd.DataFrame],
                            relationships: List[Dict[str, Any]],
                            quality_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensive validation of synthetic data"""
        
        print("Starting comprehensive quality validation...")
        
        # Validate input data
        if not isinstance(synthetic_data, dict) or not isinstance(original_data, dict):
            print("Error: Invalid data format provided")
            return {
                'overall_score': 0.0,
                'error': 'Invalid data format',
                'table_scores': {},
                'relationship_scores': {},
                'privacy_metrics': {},
                'statistical_metrics': {},
                'recommendations': []
            }
        
        validation_results = {
            'overall_score': 0.0,
            'table_scores': {},
            'relationship_scores': {},
            'privacy_metrics': {},
            'statistical_metrics': {},
            'recommendations': []
        }
        
        # Validate each table
        for table_name in synthetic_data.keys():
            if table_name in original_data:
                try:
                    # Ensure we have DataFrames
                    syn_df = synthetic_data[table_name]
                    orig_df = original_data[table_name]
                    
                    if not isinstance(syn_df, pd.DataFrame) or not isinstance(orig_df, pd.DataFrame):
                        print(f"Warning: Skipping table {table_name} - not a DataFrame")
                        continue
                    
                    if syn_df.empty or orig_df.empty:
                        print(f"Warning: Skipping table {table_name} - empty DataFrame")
                        continue
                    
                    table_score = self._validate_table(syn_df, orig_df, table_name)
                    validation_results['table_scores'][table_name] = table_score
                except Exception as e:
                    print(f"Error validating table {table_name}: {e}")
                    validation_results['table_scores'][table_name] = {
                        'overall_score': 0.0,
                        'error': str(e)
                    }
        
        # Validate relationships
        if relationships:
            try:
                relationship_score = self._validate_relationships(
                    synthetic_data, relationships
                )
                validation_results['relationship_scores'] = relationship_score
            except Exception as e:
                print(f"Error in relationship validation: {e}")
                validation_results['relationship_scores'] = {}
        
        # Privacy assessment
        try:
            privacy_score = self._assess_privacy(synthetic_data, original_data)
            validation_results['privacy_metrics'] = privacy_score
        except Exception as e:
            print(f"Error in privacy assessment: {e}")
            validation_results['privacy_metrics'] = {'overall_privacy_score': 0.5, 'error': str(e)}
        
        # Statistical quality
        try:
            statistical_score = self._assess_statistical_quality(synthetic_data, original_data)
            validation_results['statistical_metrics'] = statistical_score
        except Exception as e:
            print(f"Error in statistical quality assessment: {e}")
            validation_results['statistical_metrics'] = {'overall_statistical_score': 0.5, 'error': str(e)}
        
        # Calculate overall score
        try:
            validation_results['overall_score'] = self._calculate_overall_score(validation_results)
        except Exception as e:
            print(f"Error calculating overall score: {e}")
            validation_results['overall_score'] = 0.0
        
        # Generate recommendations
        try:
            validation_results['recommendations'] = self._generate_recommendations(validation_results)
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            validation_results['recommendations'] = ["Error generating recommendations"]
        
        return validation_results
    
    def _validate_table(self, synthetic_df: pd.DataFrame, 
                       original_df: pd.DataFrame, 
                       table_name: str) -> Dict[str, Any]:
        """Validate individual table quality"""
        
        try:
            # Validate input DataFrames
            if not isinstance(synthetic_df, pd.DataFrame) or not isinstance(original_df, pd.DataFrame):
                return {
                    'overall_score': 0.0,
                    'error': 'Invalid DataFrame type',
                    'shape_similarity': 0.0,
                    'dtype_similarity': 0.0,
                    'distribution_similarity': 0.0,
                    'correlation_similarity': 0.0,
                    'range_similarity': 0.0
                }
            
            if synthetic_df.empty or original_df.empty:
                return {
                    'overall_score': 0.0,
                    'error': 'Empty DataFrame',
                    'shape_similarity': 0.0,
                    'dtype_similarity': 0.0,
                    'distribution_similarity': 0.0,
                    'correlation_similarity': 0.0,
                    'range_similarity': 0.0
                }
            
            print(f"Validating table: {table_name}")
            
            # Basic shape validation
            shape_similarity = self._validate_shape(synthetic_df, original_df)
            
            # Data type validation
            dtype_similarity = self._validate_data_types(synthetic_df, original_df)
            
            # Distribution validation
            distribution_similarity = self._validate_distributions(synthetic_df, original_df)
            
            # Correlation validation
            correlation_similarity = self._validate_correlations(synthetic_df, original_df)
            
            # Range validation
            range_similarity = self._validate_ranges(synthetic_df, original_df)
            
            # Calculate table score
            table_score = np.mean([
                shape_similarity,
                dtype_similarity,
                distribution_similarity,
                correlation_similarity,
                range_similarity
            ])
            
            return {
                'overall_score': table_score,
                'shape_similarity': shape_similarity,
                'dtype_similarity': dtype_similarity,
                'distribution_similarity': distribution_similarity,
                'correlation_similarity': correlation_similarity,
                'range_similarity': range_similarity
            }
            
        except Exception as e:
            print(f"Error in table validation for {table_name}: {e}")
            return {
                'overall_score': 0.0,
                'error': str(e),
                'shape_similarity': 0.0,
                'dtype_similarity': 0.0,
                'distribution_similarity': 0.0,
                'correlation_similarity': 0.0,
                'range_similarity': 0.0
            }
    
    def _validate_shape(self, synthetic_df: pd.DataFrame, 
                       original_df: pd.DataFrame) -> float:
        """Validate data shape similarity"""
        try:
            # Validate input DataFrames
            if not isinstance(synthetic_df, pd.DataFrame) or not isinstance(original_df, pd.DataFrame):
                return 0.0
            
            if synthetic_df.empty and original_df.empty:
                return 1.0  # Both empty is considered similar
            
            if synthetic_df.empty or original_df.empty:
                return 0.0  # One empty, one not is dissimilar
            
            # Check if column structure is preserved
            common_cols = set(synthetic_df.columns) & set(original_df.columns)
            total_cols = set(synthetic_df.columns) | set(original_df.columns)
            
            if len(total_cols) == 0:
                return 0.0
            
            # Column similarity
            col_similarity = len(common_cols) / len(total_cols)
            
            # Row count similarity (if both have data)
            if len(synthetic_df) > 0 and len(original_df) > 0:
                row_ratio = min(len(synthetic_df), len(original_df)) / max(len(synthetic_df), len(original_df))
            else:
                row_ratio = 0.0
            
            # Combine scores
            overall_similarity = (col_similarity + row_ratio) / 2
            
            return overall_similarity
            
        except Exception as e:
            print(f"Error in shape validation: {e}")
            return 0.0
    
    def _validate_data_types(self, synthetic_df: pd.DataFrame, 
                           original_df: pd.DataFrame) -> float:
        """Validate data type preservation"""
        try:
            # Validate input DataFrames
            if not isinstance(synthetic_df, pd.DataFrame) or not isinstance(original_df, pd.DataFrame):
                return 0.0
            
            if synthetic_df.empty or original_df.empty:
                return 0.0
            
            common_cols = set(synthetic_df.columns) & set(original_df.columns)
            
            if not common_cols:
                return 0.0
            
            type_scores = []
            
            for col in common_cols:
                try:
                    # Compare data types
                    syn_dtype = str(synthetic_df[col].dtype)
                    orig_dtype = str(original_df[col].dtype)
                    
                    # Check if types are compatible
                    if syn_dtype == orig_dtype:
                        type_scores.append(1.0)
                    elif self._are_types_compatible(syn_dtype, orig_dtype):
                        type_scores.append(0.8)  # Compatible but not identical
                    else:
                        type_scores.append(0.0)  # Incompatible types
                        
                except Exception as e:
                    print(f"Error in data type validation for column {col}: {e}")
                    type_scores.append(0.0)
            
            return np.mean(type_scores) if type_scores else 0.0
            
        except Exception as e:
            print(f"Error in data type validation: {e}")
            return 0.0
    
    def _are_types_compatible(self, syn_type: str, orig_type: str) -> bool:
        """Helper to check if synthetic and original types are compatible."""
        # Define compatible type pairs
        compatible_pairs = [
            ('int64', 'float64'),
            ('float64', 'int64'),
            ('object', 'string'),
            ('string', 'object'),
            ('category', 'string'),
            ('string', 'category')
        ]
        return (syn_type, orig_type) in compatible_pairs or (orig_type, syn_type) in compatible_pairs
    
    def _validate_distributions(self, synthetic_df: pd.DataFrame, 
                              original_df: pd.DataFrame) -> float:
        """Validate distribution similarity"""
        try:
            # Validate input DataFrames
            if not isinstance(synthetic_df, pd.DataFrame) or not isinstance(original_df, pd.DataFrame):
                return 0.0
            
            if synthetic_df.empty or original_df.empty:
                return 0.0
            
            common_cols = set(synthetic_df.columns) & set(original_df.columns)
            
            if not common_cols:
                return 0.0
            
            distribution_scores = []
            
            for col in common_cols:
                try:
                    if synthetic_df[col].dtype in ['int64', 'float64']:
                        # For numerical columns, use basic statistics
                        syn_mean = synthetic_df[col].mean()
                        orig_mean = original_df[col].mean()
                        syn_std = synthetic_df[col].std()
                        orig_std = original_df[col].std()
                        
                        # Calculate similarity based on mean and std
                        mean_diff = abs(syn_mean - orig_mean) / (abs(orig_mean) + 1e-8)
                        std_diff = abs(syn_std - orig_std) / (abs(orig_std) + 1e-8)
                        
                        # Convert to similarity scores (lower diff = higher similarity)
                        mean_similarity = max(0, 1 - mean_diff)
                        std_similarity = max(0, 1 - std_diff)
                        
                        # Average the similarities
                        col_similarity = (mean_similarity + std_similarity) / 2
                        distribution_scores.append(col_similarity)
                        
                    else:
                        # For categorical columns, use simple overlap
                        syn_unique = set(synthetic_df[col].dropna())
                        orig_unique = set(original_df[col].dropna())
                        
                        if orig_unique:
                            overlap = len(syn_unique & orig_unique) / len(orig_unique)
                            distribution_scores.append(overlap)
                        else:
                            distribution_scores.append(1.0)
                            
                except Exception as e:
                    print(f"Error in distribution validation for column {col}: {e}")
                    distribution_scores.append(0.0)
            
            return np.mean(distribution_scores) if distribution_scores else 0.0
            
        except Exception as e:
            print(f"Error in distribution validation: {e}")
            return 0.0
    
    def _validate_categorical_distribution(self, synthetic_series: pd.Series, 
                                         original_series: pd.Series) -> float:
        """Validate categorical distribution similarity"""
        # Get value counts
        synthetic_counts = synthetic_series.value_counts(normalize=True)
        original_counts = original_series.value_counts(normalize=True)
        
        # Find common categories
        common_categories = set(synthetic_counts.index) & set(original_counts.index)
        
        if not common_categories:
            return 0.0
        
        # Calculate distribution similarity
        similarities = []
        for cat in common_categories:
            syn_prob = synthetic_counts.get(cat, 0)
            orig_prob = original_counts.get(cat, 0)
            
            # Calculate similarity (1 - absolute difference)
            similarity = 1 - abs(syn_prob - orig_prob)
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _validate_numerical_distribution(self, synthetic_series: pd.Series, 
                                       original_series: pd.Series) -> float:
        """Validate numerical distribution similarity"""
        try:
            # Remove NaN values
            syn_clean = synthetic_series.dropna()
            orig_clean = original_series.dropna()
            
            if len(syn_clean) == 0 or len(orig_clean) == 0:
                return 0.0
            
            # Kolmogorov-Smirnov test
            ks_statistic, ks_pvalue = stats.ks_2samp(syn_clean, orig_clean)
            
            # Higher p-value means more similar distributions
            ks_score = min(ks_pvalue * 10, 1.0)  # Scale p-value
            
            # Compare basic statistics
            syn_mean, syn_std = syn_clean.mean(), syn_clean.std()
            orig_mean, orig_std = orig_clean.mean(), orig_clean.std()
            
            mean_similarity = 1 - abs(syn_mean - orig_mean) / (abs(orig_mean) + 1e-8)
            std_similarity = 1 - abs(syn_std - orig_std) / (abs(orig_std) + 1e-8)
            
            stat_score = (mean_similarity + std_similarity) / 2
            
            return (ks_score + stat_score) / 2
            
        except Exception as e:
            print(f"Error in numerical distribution validation: {e}")
            return 0.0
    
    def _validate_correlations(self, synthetic_df: pd.DataFrame, 
                              original_df: pd.DataFrame) -> float:
        """Validate correlation preservation between columns"""
        try:
            # Validate input DataFrames
            if not isinstance(synthetic_df, pd.DataFrame) or not isinstance(original_df, pd.DataFrame):
                return 0.0
            
            if synthetic_df.empty or original_df.empty:
                return 0.0
            
            # Filter numeric columns only - use ORIGINAL data types to avoid synthetic float32 confusion
            numeric_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return 1.0  # No numeric columns to correlate
            
            # Calculate correlations for both datasets
            try:
                original_corr = original_df[numeric_cols].corr()
                synthetic_corr = synthetic_df[numeric_cols].corr()
                
                # Handle NaN values
                original_corr = original_corr.fillna(0)
                synthetic_corr = synthetic_corr.fillna(0)
                
                # Calculate correlation similarity using Frobenius norm
                diff_matrix = original_corr - synthetic_corr
                # Fix deprecation warning by explicitly specifying axis
                frobenius_norm = np.sqrt(np.sum(diff_matrix.values**2, axis=0).sum())
                max_possible_norm = np.sqrt(len(numeric_cols)**2)
                
                if max_possible_norm == 0:
                    return 1.0
                    
                similarity = max(0, 1 - (frobenius_norm / max_possible_norm))
                return similarity
                
            except Exception as e:
                print(f"Error calculating correlations: {e}")
                return 0.5
            
        except Exception as e:
            print(f"Error in correlation validation: {e}")
            return 0.5  # Return neutral score on error
    
    def _validate_ranges(self, synthetic_df: pd.DataFrame, 
                       original_df: pd.DataFrame) -> float:
        """Validate value range preservation"""
        try:
            # Validate input DataFrames
            if not isinstance(synthetic_df, pd.DataFrame) or not isinstance(original_df, pd.DataFrame):
                return 0.0
            
            if synthetic_df.empty or original_df.empty:
                return 0.0
            
            common_cols = set(synthetic_df.columns) & set(original_df.columns)
            
            if not common_cols:
                return 0.0
            
            range_scores = []
            
            for col in common_cols:
                try:
                    if synthetic_df[col].dtype in ['int64', 'float64']:
                        # Numerical range validation
                        syn_min, syn_max = synthetic_df[col].min(), synthetic_df[col].max()
                        orig_min, orig_max = original_df[col].min(), original_df[col].max()
                        
                        # Check if synthetic range is reasonable
                        range_overlap = min(syn_max, orig_max) - max(syn_min, orig_min)
                        range_union = max(syn_max, orig_max) - min(syn_min, orig_min)
                        
                        if range_union > 0:
                            range_score = max(0, range_overlap / range_union)
                        else:
                            range_score = 1.0
                        
                        range_scores.append(range_score)
                        
                    elif synthetic_df[col].dtype in ['object', 'string']:
                        # Categorical range validation
                        syn_unique = set(synthetic_df[col].dropna())
                        orig_unique = set(original_df[col].dropna())
                        
                        if orig_unique:
                            coverage = len(syn_unique & orig_unique) / len(orig_unique)
                            range_scores.append(coverage)
                        else:
                            range_scores.append(1.0)
                            
                except Exception as e:
                    print(f"Error in range validation for column {col}: {e}")
                    range_scores.append(0.0)
            
            return np.mean(range_scores) if range_scores else 0.0
            
        except Exception as e:
            print(f"Error in range validation: {e}")
            return 0.0
    
    def _validate_relationships(self, synthetic_data: Dict[str, pd.DataFrame], 
                              relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate relationship integrity"""
        print("Validating relationship integrity...")
        
        try:
            # Validate input data
            if not isinstance(synthetic_data, dict) or not isinstance(relationships, list):
                print("Error: Invalid data format for relationship validation")
                return {}
            
            relationship_scores = {}
            
            for rel in relationships:
                try:
                    if not isinstance(rel, dict):
                        continue
                        
                    source_table = rel.get('source_table')
                    target_table = rel.get('target_table')
                    source_col = rel.get('source_column')
                    target_col = rel.get('target_column')
                    
                    if not all([source_table, target_table, source_col, target_col]):
                        continue
                    
                    if source_table in synthetic_data and target_table in synthetic_data:
                        try:
                            score = self._validate_single_relationship(
                                synthetic_data[source_table], synthetic_data[target_table],
                                source_col, target_col
                            )
                            relationship_scores[f"{source_table}.{source_col} -> {target_table}.{target_col}"] = score
                        except Exception as e:
                            print(f"Error validating relationship {source_table}.{source_col} -> {target_table}.{target_col}: {e}")
                            relationship_scores[f"{source_table}.{source_col} -> {target_table}.{target_col}"] = 0.0
                            
                except Exception as e:
                    print(f"Error processing relationship: {e}")
                    continue
            
            return relationship_scores
            
        except Exception as e:
            print(f"Error in relationship validation: {e}")
            return {}
    
    def _validate_single_relationship(self, source_df: pd.DataFrame, 
                                    target_df: pd.DataFrame,
                                    source_col: str, target_col: str) -> float:
        """Validate a single relationship"""
        try:
            # Validate input DataFrames
            if not isinstance(source_df, pd.DataFrame) or not isinstance(target_df, pd.DataFrame):
                return 0.0
            
            if source_df.empty or target_df.empty:
                return 0.0
            
            if source_col not in source_df.columns or target_col not in target_df.columns:
                return 0.0
            
            # Get foreign key values
            source_values = set(source_df[source_col].dropna())
            target_values = set(target_df[target_col].dropna())
            
            if not source_values or not target_values:
                return 0.0
            
            # Check referential integrity
            # All source values should exist in target
            orphaned_keys = source_values - target_values
            orphaned_ratio = len(orphaned_keys) / len(source_values) if source_values else 0
            
            # Score: 1.0 = perfect integrity, 0.0 = complete failure
            integrity_score = 1.0 - orphaned_ratio
            
            return max(0.0, integrity_score)
            
        except Exception as e:
            print(f"Error in single relationship validation: {e}")
            return 0.0
    
    def fix_referential_integrity(self, synthetic_data: Dict[str, pd.DataFrame], 
                                relationships: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """Fix referential integrity issues in synthetic data"""
        print("🔧 Fixing referential integrity issues...")
        
        fixed_data = {table: df.copy() for table, df in synthetic_data.items()}
        
        for rel in relationships:
            source_table = rel['source_table']
            target_table = rel['target_table']
            source_col = rel['source_column']
            target_col = rel['target_column']
            
            if source_table in fixed_data and target_table in fixed_data:
                fixed_data = self._fix_single_relationship(
                    fixed_data, source_table, target_table, source_col, target_col
                )
        
        return fixed_data
    
    def _fix_single_relationship(self, data: Dict[str, pd.DataFrame], 
                               source_table: str, target_table: str,
                               source_col: str, target_col: str) -> Dict[str, pd.DataFrame]:
        """Fix referential integrity for a single relationship"""
        try:
            source_df = data[source_table]
            target_df = data[target_table]
            
            if source_col not in source_df.columns or target_col not in target_df.columns:
                return data
            
            # Get valid target values
            valid_target_values = set(target_df[target_col].dropna())
            
            if len(valid_target_values) == 0:
                print(f"⚠️  No valid target values in {target_table}.{target_col}")
                return data
            
            # Find orphaned foreign keys
            source_values = source_df[source_col].dropna()
            orphaned_mask = ~source_values.isin(valid_target_values)
            orphaned_count = orphaned_mask.sum(axis=0)
            
            if orphaned_count > 0:
                print(f"🔧 Fixing {orphaned_count} orphaned foreign keys in {source_table}.{source_col}")
                
                # Replace orphaned values with valid ones
                valid_values_list = list(valid_target_values)
                orphaned_indices = source_values[orphaned_mask].index
                
                for idx in orphaned_indices:
                    # Choose a random valid value
                    new_value = np.random.choice(valid_values_list)
                    data[source_table].loc[idx, source_col] = new_value
                
                print(f"✅ Fixed {orphaned_count} orphaned foreign keys")
            
            return data
            
        except Exception as e:
            print(f"Error fixing relationship {source_table}.{source_col} -> {target_table}.{target_col}: {e}")
            return data
    
    def validate_and_fix_primary_keys(self, synthetic_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Validate and fix primary key uniqueness issues"""
        print("🔑 Validating and fixing primary keys...")
        
        fixed_data = {table: df.copy() for table, df in synthetic_data.items()}
        
        for table_name, df in fixed_data.items():
            # Try to identify primary key column
            pk_candidates = [col for col in df.columns if 'id' in col.lower() and col.endswith('_id')]
            
            for pk_col in pk_candidates:
                if pk_col in df.columns:
                    # Check for duplicates
                    duplicates = df[pk_col].duplicated()
                    duplicate_count = duplicates.sum(axis=0)
                    
                    if duplicate_count > 0:
                        print(f"🔧 Fixing {duplicate_count} duplicate primary keys in {table_name}.{pk_col}")
                        
                        # Generate new unique values for duplicates
                        max_id = df[pk_col].max()
                        duplicate_indices = df[duplicates].index
                        
                        for i, idx in enumerate(duplicate_indices):
                            new_id = max_id + i + 1
                            fixed_data[table_name].loc[idx, pk_col] = new_id
                        
                        print(f"✅ Fixed {duplicate_count} duplicate primary keys")
        
        return fixed_data
    
    def _assess_privacy(self, synthetic_data: Dict[str, pd.DataFrame], 
                        original_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Assess privacy preservation in synthetic data"""
        try:
            # Validate input data
            if not isinstance(synthetic_data, dict) or not isinstance(original_data, dict):
                return {
                    'overall_privacy_score': 0.5,
                    'error': 'Invalid data format'
                }
            
            privacy_scores = {}
            
            for table_name in synthetic_data.keys():
                if table_name in original_data:
                    try:
                        syn_df = synthetic_data[table_name]
                        orig_df = original_data[table_name]
                        
                        # Ensure we have valid DataFrames
                        if not isinstance(syn_df, pd.DataFrame) or not isinstance(orig_df, pd.DataFrame):
                            privacy_scores[table_name] = {'overall_privacy_score': 0.5, 'error': 'Invalid DataFrame'}
                            continue
                        
                        if syn_df.empty or orig_df.empty:
                            privacy_scores[table_name] = {'overall_privacy_score': 0.5, 'error': 'Empty DataFrame'}
                            continue
                        
                        # Calculate privacy metrics
                        privacy_score = self._calculate_table_privacy(syn_df, orig_df)
                        privacy_scores[table_name] = privacy_score
                        
                    except Exception as e:
                        print(f"Error in privacy assessment for table {table_name}: {e}")
                        privacy_scores[table_name] = {'overall_privacy_score': 0.5, 'error': str(e)}
            
            # Calculate overall privacy score
            if privacy_scores:
                overall_score = np.mean([
                    score.get('overall_privacy_score', 0.5) 
                    for score in privacy_scores.values()
                ])
            else:
                overall_score = 0.5
            
            return {
                'overall_privacy_score': overall_score,
                'table_privacy_scores': privacy_scores
            }
            
        except Exception as e:
            print(f"Error in privacy assessment: {e}")
            return {
                'overall_privacy_score': 0.5,
                'error': str(e)
            }
    
    def _calculate_table_privacy(self, syn_df: pd.DataFrame, orig_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate privacy score for a single table"""
        try:
            # Simple privacy score based on data uniqueness
            # Higher uniqueness = lower privacy risk
            privacy_scores = []
            
            for col in syn_df.columns:
                if col in orig_df.columns:
                    try:
                        # Calculate uniqueness ratio
                        orig_unique = orig_df[col].nunique()
                        syn_unique = syn_df[col].nunique()
                        orig_total = len(orig_df[col].dropna())
                        syn_total = len(syn_df[col].dropna())
                        
                        if orig_total > 0 and syn_total > 0:
                            orig_ratio = orig_unique / orig_total
                            syn_ratio = syn_unique / syn_total
                            
                            # Privacy score: higher uniqueness = better privacy
                            privacy_score = min(1.0, syn_ratio / max(orig_ratio, 0.1))
                            privacy_scores.append(privacy_score)
                        else:
                            privacy_scores.append(0.5)
                    except:
                        privacy_scores.append(0.5)
            
            if privacy_scores:
                overall_score = np.mean(privacy_scores)
            else:
                overall_score = 0.5
            
            return {
                'overall_privacy_score': overall_score,
                'column_privacy_scores': privacy_scores
            }
            
        except Exception as e:
            print(f"Error calculating table privacy: {e}")
            return {'overall_privacy_score': 0.5, 'error': str(e)}
    
    def _detect_pii(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Detect potential PII in synthetic data"""
        pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-.]?\d{4}[-.]?\d{4}[-.]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
        pii_found = {}
        total_cells = len(df) * len(df.columns)
        pii_cells = 0
        
        for col in df.columns:
            col_pii = []
            for pattern_name, pattern in pii_patterns.items():
                matches = df[col].astype(str).str.contains(pattern, regex=True, na=False)
                if matches.any():
                    col_pii.append(pattern_name)
                    pii_cells += matches.sum(axis=0)
            
            if col_pii:
                pii_found[col] = col_pii
        
        pii_score = 1.0 - (pii_cells / total_cells) if total_cells > 0 else 1.0
        
        return {
            'score': pii_score,
            'pii_types_found': pii_found,
            'pii_cells_count': pii_cells,
            'total_cells': total_cells
        }
    
    def _assess_k_anonymity(self, syn_df: pd.DataFrame, orig_df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Assess k-anonymity for synthetic data"""
        # Identify quasi-identifiers (columns that could be used for re-identification)
        quasi_identifiers = []
        for col in syn_df.columns:
            if syn_df[col].dtype in ['object', 'category'] or syn_df[col].nunique() < len(syn_df) * 0.1:
                quasi_identifiers.append(col)
        
        if not quasi_identifiers:
            return {'score': 1.0, 'k_value': float('inf'), 'quasi_identifiers': []}
        
        # Calculate k-anonymity
        try:
            # Group by quasi-identifiers and count occurrences
            grouped = syn_df[quasi_identifiers].groupby(quasi_identifiers).size()
            k_value = grouped.min() if len(grouped) > 0 else float('inf')
            
            # Score based on k value (higher k = better privacy)
            if k_value == float('inf'):
                score = 1.0
            elif k_value >= 10:
                score = 1.0
            elif k_value >= 5:
                score = 0.8
            elif k_value >= 3:
                score = 0.6
            else:
                score = 0.3
                
        except Exception as e:
            print(f"Error calculating k-anonymity for {table_name}: {e}")
            k_value = 1
            score = 0.3
        
        return {
            'score': score,
            'k_value': k_value,
            'quasi_identifiers': quasi_identifiers
        }
    
    def _assess_differential_privacy(self, syn_df: pd.DataFrame, orig_df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Assess differential privacy characteristics"""
        # This is a simplified assessment - in practice, you'd need formal DP mechanisms
        try:
            # Calculate statistical distance between original and synthetic distributions
            distances = []
            for col in syn_df.columns:
                if syn_df[col].dtype in ['int64', 'float64']:
                    # Kolmogorov-Smirnov test for distribution similarity
                    try:
                        from scipy.stats import ks_2samp
                        stat, p_value = ks_2samp(orig_df[col].dropna(), syn_df[col].dropna())
                        distances.append(1 - stat)  # Convert to similarity score
                    except:
                        distances.append(0.5)
            
            if distances:
                dp_score = np.mean(distances)
            else:
                dp_score = 0.5
                
        except Exception as e:
            print(f"Error assessing differential privacy for {table_name}: {e}")
            dp_score = 0.5
        
        return {
            'score': dp_score,
            'method': 'statistical_distance',
            'note': 'Simplified assessment - formal DP requires specific mechanisms'
        }
    
    def _assess_linkage_risk(self, syn_df: pd.DataFrame, orig_df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Assess risk of linking synthetic data back to original data"""
        try:
            # Check for exact matches that could enable linkage
            exact_matches = 0
            total_comparisons = 0
            
            for col in syn_df.columns:
                if col in orig_df.columns:
                    syn_values = set(syn_df[col].dropna())
                    orig_values = set(orig_df[col].dropna())
                    
                    # Count exact matches
                    matches = len(syn_values.intersection(orig_values))
                    exact_matches += matches
                    total_comparisons += len(syn_values)
            
            if total_comparisons > 0:
                linkage_risk_score = 1.0 - (exact_matches / total_comparisons)
            else:
                linkage_risk_score = 1.0
                
        except Exception as e:
            print(f"Error assessing linkage risk for {table_name}: {e}")
            linkage_risk_score = 0.5
        
        return {
            'score': linkage_risk_score,
            'exact_matches': exact_matches,
            'total_comparisons': total_comparisons
        }
    
    def _generate_privacy_recommendations(self, privacy_metrics: Dict[str, Any]) -> List[str]:
        """Generate privacy improvement recommendations"""
        recommendations = []
        
        overall_score = privacy_metrics.get('overall_privacy_score', 0)
        
        if overall_score < 0.7:
            recommendations.append("Consider implementing differential privacy mechanisms")
            recommendations.append("Increase k-anonymity by generalizing quasi-identifiers")
            recommendations.append("Add noise to numerical columns to prevent exact matching")
        
        if overall_score < 0.5:
            recommendations.append("Review and remove or heavily anonymize PII columns")
            recommendations.append("Consider using synthetic data generation techniques that preserve privacy by design")
            recommendations.append("Implement data perturbation techniques")
        
        # Specific recommendations based on individual metrics
        for table_name, pii_info in privacy_metrics.get('pii_detection', {}).items():
            if isinstance(pii_info, dict) and pii_info.get('score', 1.0) < 0.8:
                recommendations.append(f"Review PII detection in table {table_name}")
        
        return recommendations
    
    def _assess_statistical_quality(self, synthetic_data: Dict[str, pd.DataFrame], 
                                   original_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Enhanced statistical quality assessment with multiple metrics"""
        statistical_metrics = {
            'overall_statistical_score': 0.0,
            'distribution_similarity': {},
            'correlation_preservation': {},
            'statistical_distance': {},
            'outlier_analysis': {},
            'recommendations': []
        }
        
        for table_name in synthetic_data.keys():
            if table_name in original_data:
                try:
                    syn_df = synthetic_data[table_name]
                    orig_df = original_data[table_name]
                    
                    # Ensure we have valid DataFrames
                    if not isinstance(syn_df, pd.DataFrame) or not isinstance(orig_df, pd.DataFrame):
                        print(f"Warning: Skipping statistical assessment for table {table_name} - invalid DataFrame")
                        continue
                    
                    if syn_df.empty or orig_df.empty:
                        print(f"Warning: Skipping statistical assessment for table {table_name} - empty DataFrame")
                        continue
                    
                    # Distribution similarity
                    dist_similarity = self._assess_distribution_similarity(syn_df, orig_df, table_name)
                    statistical_metrics['distribution_similarity'][table_name] = dist_similarity
                    
                    # Correlation preservation
                    corr_preservation = self._assess_correlation_preservation(syn_df, orig_df, table_name)
                    statistical_metrics['correlation_preservation'][table_name] = corr_preservation
                    
                    # Statistical distance
                    stat_distance = self._assess_statistical_distance(syn_df, orig_df, table_name)
                    statistical_metrics['statistical_distance'][table_name] = stat_distance
                    
                    # Outlier analysis
                    outlier_analysis = self._assess_outlier_patterns(syn_df, orig_df, table_name)
                    statistical_metrics['outlier_analysis'][table_name] = outlier_analysis
                    
                except Exception as e:
                    print(f"Error in statistical assessment for table {table_name}: {e}")
                    # Add error entry for this table
                    statistical_metrics['distribution_similarity'][table_name] = {'score': 0.0, 'error': str(e)}
                    statistical_metrics['correlation_preservation'][table_name] = {'score': 0.0, 'error': str(e)}
                    statistical_metrics['statistical_distance'][table_name] = {'score': 0.0, 'error': str(e)}
                    statistical_metrics['outlier_analysis'][table_name] = {'score': 0.0, 'error': str(e)}
        
        # Calculate overall statistical score
        all_scores = []
        for metric_type in ['distribution_similarity', 'correlation_preservation', 'statistical_distance', 'outlier_analysis']:
            for table_scores in statistical_metrics[metric_type].values():
                if isinstance(table_scores, dict) and 'score' in table_scores:
                    all_scores.append(table_scores['score'])
                elif isinstance(table_scores, (int, float)):
                    all_scores.append(table_scores)
        
        if all_scores:
            statistical_metrics['overall_statistical_score'] = np.mean(all_scores)
        
        # Generate statistical recommendations
        statistical_metrics['recommendations'] = self._generate_statistical_recommendations(statistical_metrics)
        
        return statistical_metrics
    
    def _assess_distribution_similarity(self, syn_df: pd.DataFrame, orig_df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Assess distribution similarity between original and synthetic data"""
        try:
            # Validate input DataFrames
            if not isinstance(syn_df, pd.DataFrame) or not isinstance(orig_df, pd.DataFrame):
                return {'score': 0.5, 'method': 'error', 'error': 'Invalid DataFrame type'}
            
            if syn_df.empty or orig_df.empty:
                return {'score': 0.5, 'method': 'error', 'error': 'Empty DataFrame'}
            
            distribution_scores = {}
            
            for col in syn_df.columns:
                if col in orig_df.columns:
                    try:
                        if syn_df[col].dtype in ['int64', 'float64']:
                            # For numerical columns, use KS test
                            from scipy.stats import ks_2samp
                            stat, p_value = ks_2samp(orig_df[col].dropna(), syn_df[col].dropna())
                            similarity = 1 - stat  # Convert to similarity score
                            
                        elif syn_df[col].dtype in ['object', 'category']:
                            # For categorical columns, use chi-square test
                            from scipy.stats import chi2_contingency
                            contingency_table = pd.crosstab(orig_df[col], syn_df[col])
                            if contingency_table.size > 0:
                                try:
                                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                                    # Convert chi-square to similarity (lower chi-square = higher similarity)
                                    max_chi2 = contingency_table.size * 100  # Heuristic max value
                                    similarity = max(0, 1 - (chi2 / max_chi2))
                                except:
                                    similarity = 0.5
                            else:
                                similarity = 0.5
                        else:
                            similarity = 0.5
                            
                        distribution_scores[col] = similarity
                        
                    except Exception as e:
                        print(f"Error assessing distribution for column {col} in {table_name}: {e}")
                        distribution_scores[col] = 0.5
            
            if distribution_scores:
                overall_score = np.mean(list(distribution_scores.values()))
            else:
                overall_score = 0.5
            
            return {
                'score': overall_score,
                'column_scores': distribution_scores,
                'method': 'ks_test_chi2'
            }
            
        except Exception as e:
            print(f"Error in distribution assessment for {table_name}: {e}")
            return {'score': 0.5, 'method': 'error', 'error': str(e)}
    
    def _assess_correlation_preservation(self, syn_df: pd.DataFrame, orig_df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Assess how well correlations are preserved between original and synthetic data"""
        try:
            # Validate input DataFrames
            if not isinstance(syn_df, pd.DataFrame) or not isinstance(orig_df, pd.DataFrame):
                return {'score': 0.5, 'method': 'error', 'error': 'Invalid DataFrame type'}
            
            if syn_df.empty or orig_df.empty:
                return {'score': 0.5, 'method': 'error', 'error': 'Empty DataFrame'}
            
            # Get numeric columns only - use ORIGINAL data types to avoid synthetic float32 confusion
            numeric_cols = orig_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return {'score': 1.0, 'method': 'insufficient_numeric_columns'}
            
            # Ensure we have valid numeric data
            try:
                # Calculate correlation matrices
                orig_corr = orig_df[numeric_cols].corr()
                syn_corr = syn_df[numeric_cols].corr()
                
                # Handle NaN values
                orig_corr = orig_corr.fillna(0)
                syn_corr = syn_corr.fillna(0)
                
                # Calculate correlation preservation using multiple metrics
                # 1. Frobenius norm
                diff_matrix = orig_corr - syn_corr
                frobenius_norm = np.sqrt(np.sum(diff_matrix.values**2, axis=0).sum())
                max_possible_norm = np.sqrt(len(numeric_cols)**2)
                frobenius_score = max(0, 1 - (frobenius_norm / max_possible_norm)) if max_possible_norm > 0 else 1.0
                
                # 2. Pearson correlation between correlation vectors
                orig_corr_vector = orig_corr.values[np.triu_indices(len(numeric_cols), k=1)]
                syn_corr_vector = syn_corr.values[np.triu_indices(len(numeric_cols), k=1)]
                
                if len(orig_corr_vector) > 0:
                    try:
                        from scipy.stats import pearsonr
                        pearson_corr, _ = pearsonr(orig_corr_vector, syn_corr_vector)
                        pearson_score = max(0, pearson_corr)  # Ensure non-negative
                    except:
                        pearson_score = 0.5
                else:
                    pearson_score = 1.0
                
                # 3. Mean absolute difference
                mean_abs_diff = np.mean(np.abs(diff_matrix.values).flatten())
                mad_score = max(0, 1 - mean_abs_diff)
                
                # Combine scores
                overall_score = np.mean([frobenius_score, pearson_score, mad_score])
                
                return {
                    'score': overall_score,
                    'frobenius_score': frobenius_score,
                    'pearson_score': pearson_score,
                    'mad_score': mad_score,
                    'method': 'multi_metric_correlation'
                }
                
            except Exception as e:
                print(f"Error in correlation calculation for {table_name}: {e}")
                # Log additional debug info
                print(f"Debug: numeric_cols = {numeric_cols}")
                print(f"Debug: syn_df dtypes = {syn_df.dtypes}")
                print(f"Debug: orig_df dtypes = {orig_df.dtypes}")
                return {'score': 0.5, 'method': 'error', 'error': str(e)}
            
        except Exception as e:
            print(f"Error assessing correlation preservation for {table_name}: {e}")
            return {'score': 0.5, 'method': 'error', 'error': str(e)}
    
    def _assess_statistical_distance(self, syn_df: pd.DataFrame, orig_df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Assess statistical distance between original and synthetic data"""
        try:
            # Validate input DataFrames
            if not isinstance(syn_df, pd.DataFrame) or not isinstance(orig_df, pd.DataFrame):
                return {'score': 0.5, 'method': 'error', 'error': 'Invalid DataFrame type'}
            
            if syn_df.empty or orig_df.empty:
                return {'score': 0.5, 'method': 'error', 'error': 'Empty DataFrame'}
            
            distances = {}
            
            for col in syn_df.columns:
                if col in orig_df.columns:
                    try:
                        # Only process numeric columns for statistical distance
                        if syn_df[col].dtype in ['int64', 'float64'] and orig_df[col].dtype in ['int64', 'float64']:
                            # For numerical columns, calculate multiple distance measures
                            orig_vals = orig_df[col].dropna()
                            syn_vals = syn_df[col].dropna()
                            
                            if len(orig_vals) > 0 and len(syn_vals) > 0:
                                # Wasserstein distance (Earth Mover's Distance)
                                try:
                                    from scipy.stats import wasserstein_distance
                                    wass_dist = wasserstein_distance(orig_vals, syn_vals)
                                    # Normalize by range
                                    range_val = orig_vals.max() - orig_vals.min()
                                    wass_score = max(0, 1 - (wass_dist / range_val)) if range_val > 0 else 1.0
                                except:
                                    wass_score = 0.5
                                
                                # Jensen-Shannon divergence
                                try:
                                    from scipy.spatial.distance import jensenshannon
                                    # Convert boolean to numeric for histogram creation
                                    orig_vals_numeric = orig_vals.astype(float) if orig_vals.dtype == 'bool' else orig_vals
                                    syn_vals_numeric = syn_vals.astype(float) if syn_vals.dtype == 'bool' else syn_vals
                                    
                                    # Create histograms for comparison
                                    bins = np.linspace(min(orig_vals_numeric.min(), syn_vals_numeric.min()), 
                                                     max(orig_vals_numeric.max(), syn_vals_numeric.max()), 20)
                                    orig_hist, _ = np.histogram(orig_vals_numeric, bins=bins, density=True)
                                    syn_hist, _ = np.histogram(syn_vals_numeric, bins=bins, density=True)
                                    
                                    # Add small epsilon to avoid division by zero
                                    orig_hist = orig_hist + 1e-10
                                    syn_hist = syn_hist + 1e-10
                                    
                                    js_div = jensenshannon(orig_hist, syn_hist)
                                    js_score = max(0, 1 - js_div)
                                except:
                                    js_score = 0.5
                                
                                # Combine scores
                                distances[col] = np.mean([wass_score, js_score])
                            else:
                                distances[col] = 0.5
                        else:
                            # For categorical columns, use simple overlap
                            orig_unique = set(orig_df[col].dropna())
                            syn_unique = set(syn_df[col].dropna())
                            
                            if len(orig_unique) > 0:
                                overlap = len(orig_unique.intersection(syn_unique)) / len(orig_unique)
                                distances[col] = overlap
                            else:
                                distances[col] = 1.0
                                
                    except Exception as e:
                        print(f"Error calculating distance for column {col}: {e}")
                        distances[col] = 0.5
            
            if distances:
                overall_score = np.mean(list(distances.values()))
            else:
                overall_score = 0.5
            
            return {
                'score': overall_score,
                'column_distances': distances,
                'method': 'wasserstein_js_divergence'
            }
            
        except Exception as e:
            print(f"Error assessing statistical distance for {table_name}: {e}")
            return {'score': 0.5, 'method': 'error', 'error': str(e)}
    
    def _assess_outlier_patterns(self, syn_df: pd.DataFrame, orig_df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Assess outlier pattern preservation"""
        try:
            # Validate input DataFrames
            if not isinstance(syn_df, pd.DataFrame) or not isinstance(orig_df, pd.DataFrame):
                return {'score': 0.5, 'method': 'error', 'error': 'Invalid DataFrame type'}
            
            if syn_df.empty or orig_df.empty:
                return {'score': 0.5, 'method': 'error', 'error': 'Empty DataFrame'}
            
            outlier_scores = {}
            
            for col in syn_df.columns:
                if col in orig_df.columns:
                    # Skip boolean columns as they don't have outliers in traditional sense
                    if syn_df[col].dtype == 'bool' or orig_df[col].dtype == 'bool':
                        outlier_scores[col] = 1.0  # Perfect score for boolean columns
                        continue
                    
                    # Check if columns can be converted to numeric
                    try:
                        orig_col_numeric = pd.to_numeric(orig_df[col], errors='coerce')
                        syn_col_numeric = pd.to_numeric(syn_df[col], errors='coerce')
                        
                        # Skip if conversion failed (too many NaN values)
                        if orig_col_numeric.isna().sum() > len(orig_df[col]) * 0.5 or syn_col_numeric.isna().sum() > len(syn_df[col]) * 0.5:
                            outlier_scores[col] = 0.5  # Neutral score for non-numeric columns
                            continue
                        
                        # Only process numeric columns
                        if syn_df[col].dtype in ['int64', 'float64'] or orig_df[col].dtype in ['int64', 'float64']:
                            try:
                                # Ensure both columns are numeric
                                orig_col_numeric = orig_col_numeric.fillna(orig_col_numeric.mean())
                                syn_col_numeric = syn_col_numeric.fillna(syn_col_numeric.mean())
                                
                                # Calculate outlier percentages using IQR method
                                orig_q1 = orig_col_numeric.quantile(0.25)
                                orig_q3 = orig_col_numeric.quantile(0.75)
                                orig_iqr = orig_q3 - orig_q1
                                orig_lower = orig_q1 - 1.5 * orig_iqr
                                orig_upper = orig_q3 + 1.5 * orig_iqr
                                
                                syn_q1 = syn_col_numeric.quantile(0.25)
                                syn_q3 = syn_col_numeric.quantile(0.75)
                                syn_iqr = syn_q3 - syn_q1
                                syn_lower = syn_q1 - 1.5 * syn_iqr
                                syn_upper = syn_q3 + 1.5 * syn_iqr
                                
                                # Calculate outlier percentages using IQR method
                                orig_outliers = ((orig_col_numeric < orig_lower) | (orig_col_numeric > orig_upper)).mean()
                                syn_outliers = ((syn_col_numeric < syn_lower) | (syn_col_numeric > syn_upper)).mean()
                                
                                # Score based on similarity of outlier percentages
                                if orig_outliers > 0:
                                    outlier_similarity = 1 - abs(orig_outliers - syn_outliers) / orig_outliers
                                else:
                                    outlier_similarity = 1.0 if syn_outliers == 0 else 0.5
                                
                                outlier_scores[col] = max(0, outlier_similarity)
                                
                            except Exception as e:
                                print(f"Error assessing outliers for column {col}: {e}")
                                outlier_scores[col] = 0.5
                        else:
                            outlier_scores[col] = 0.5  # Neutral score for non-numeric columns
                    except Exception as e:
                        print(f"Error converting column {col} to numeric: {e}")
                        outlier_scores[col] = 0.5  # Neutral score for conversion failures
            
            if outlier_scores:
                overall_score = np.mean(list(outlier_scores.values()))
            else:
                overall_score = 0.5
            
            return {
                'score': overall_score,
                'column_outlier_scores': outlier_scores,
                'method': 'iqr_outlier_analysis'
            }
            
        except Exception as e:
            print(f"Error assessing outlier patterns for {table_name}: {e}")
            return {'score': 0.5, 'method': 'error', 'error': str(e)}
    
    def _generate_statistical_recommendations(self, statistical_metrics: Dict[str, Any]) -> List[str]:
        """Generate statistical improvement recommendations"""
        try:
            recommendations = []
            
            overall_score = statistical_metrics.get('overall_statistical_score', 0)
            
            if overall_score < 0.7:
                recommendations.append("Consider adjusting the neural network architecture for better distribution learning")
                recommendations.append("Increase training epochs or adjust learning rate for better convergence")
                recommendations.append("Review data preprocessing to ensure proper scaling and encoding")
            
            if overall_score < 0.5:
                recommendations.append("The synthetic data may not be preserving key statistical properties")
                recommendations.append("Consider using different generation algorithms (e.g., GANs, CTGAN)")
                recommendations.append("Review the training data quality and quantity")
            
            # Specific recommendations based on individual metrics
            if statistical_metrics.get('distribution_similarity'):
                for table_name, dist_info in statistical_metrics['distribution_similarity'].items():
                    if isinstance(dist_info, dict) and dist_info.get('score', 1.0) < 0.6:
                        recommendations.append(f"Distribution similarity is low for table {table_name} - consider data augmentation")
            
            if statistical_metrics.get('correlation_preservation'):
                for table_name, corr_info in statistical_metrics['correlation_preservation'].items():
                    if isinstance(corr_info, dict) and corr_info.get('score', 1.0) < 0.6:
                        recommendations.append(f"Correlation preservation is poor for table {table_name} - review relationship modeling")
            
            return recommendations if recommendations else ["No specific statistical recommendations at this time"]
            
        except Exception as e:
            print(f"Error generating statistical recommendations: {e}")
            return ["Error generating statistical recommendations"]
    
    def _calculate_overall_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        try:
            scores = []
            
            # Table scores
            if validation_results.get('table_scores'):
                table_scores = []
                for score_info in validation_results['table_scores'].values():
                    if isinstance(score_info, dict):
                        if 'overall_score' in score_info:
                            table_scores.append(score_info['overall_score'])
                        elif 'error' in score_info:
                            # Skip tables with errors
                            continue
                    elif isinstance(score_info, (int, float)):
                        table_scores.append(score_info)
                
                if table_scores:
                    scores.append(np.mean(table_scores))
            
            # Relationship scores
            if validation_results.get('relationship_scores'):
                rel_scores = []
                for score_info in validation_results['relationship_scores'].values():
                    if isinstance(score_info, (int, float)):
                        rel_scores.append(score_info)
                
                if rel_scores:
                    scores.append(np.mean(rel_scores))
            
            # Privacy scores
            if validation_results.get('privacy_metrics'):
                privacy_info = validation_results['privacy_metrics']
                if isinstance(privacy_info, dict) and 'overall_privacy_score' in privacy_info:
                    scores.append(privacy_info['overall_privacy_score'])
            
            # Statistical scores
            if validation_results.get('statistical_metrics'):
                stats_info = validation_results['statistical_metrics']
                if isinstance(stats_info, dict) and 'overall_statistical_score' in stats_info:
                    scores.append(stats_info['overall_statistical_score'])
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            print(f"Error calculating overall score: {e}")
            return 0.0
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on validation results"""
        try:
            recommendations = []
            
            # Overall score recommendations
            overall_score = validation_results.get('overall_score', 0.0)
            if overall_score < 0.7:
                recommendations.append("Consider improving data quality and validation processes")
            if overall_score < 0.5:
                recommendations.append("Significant improvements needed in data generation and validation")
            
            # Table-specific recommendations
            if validation_results.get('table_scores'):
                for table_name, table_info in validation_results['table_scores'].items():
                    if isinstance(table_info, dict):
                        if 'error' in table_info:
                            recommendations.append(f"Fix validation errors in table {table_name}")
                        elif 'overall_score' in table_info:
                            score = table_info['overall_score']
                            if score < 0.6:
                                recommendations.append(f"Improve data quality for table {table_name}")
            
            # Privacy recommendations
            if validation_results.get('privacy_metrics'):
                privacy_info = validation_results['privacy_metrics']
                if isinstance(privacy_info, dict):
                    privacy_score = privacy_info.get('overall_privacy_score', 0.0)
                    if privacy_score < 0.7:
                        recommendations.append("Enhance privacy protection measures")
            
            # Statistical recommendations
            if validation_results.get('statistical_metrics'):
                stats_info = validation_results['statistical_metrics']
                if isinstance(stats_info, dict):
                    stats_score = stats_info.get('overall_statistical_score', 0.0)
                    if stats_score < 0.7:
                        recommendations.append("Improve statistical quality preservation")
            
            return recommendations if recommendations else ["No specific recommendations at this time"]
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"] 