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
                table_score = self._validate_table(
                    synthetic_data[table_name], 
                    original_data[table_name],
                    table_name
                )
                validation_results['table_scores'][table_name] = table_score
        
        # Validate relationships
        if relationships:
            relationship_score = self._validate_relationships(
                synthetic_data, relationships
            )
            validation_results['relationship_scores'] = relationship_score
        
        # Privacy assessment
        privacy_score = self._assess_privacy(synthetic_data, original_data)
        validation_results['privacy_metrics'] = privacy_score
        
        # Statistical quality
        statistical_score = self._assess_statistical_quality(synthetic_data, original_data)
        validation_results['statistical_metrics'] = statistical_score
        
        # Calculate overall score
        validation_results['overall_score'] = self._calculate_overall_score(validation_results)
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_recommendations(validation_results)
        
        return validation_results
    
    def _validate_table(self, synthetic_df: pd.DataFrame, 
                       original_df: pd.DataFrame, 
                       table_name: str) -> Dict[str, Any]:
        """Validate individual table quality"""
        
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
    
    def _validate_shape(self, synthetic_df: pd.DataFrame, 
                       original_df: pd.DataFrame) -> float:
        """Validate data shape similarity"""
        # Check if column structure is preserved
        common_cols = set(synthetic_df.columns) & set(original_df.columns)
        total_cols = set(synthetic_df.columns) | set(original_df.columns)
        
        if len(total_cols) == 0:
            return 0.0
        
        column_similarity = len(common_cols) / len(total_cols)
        
        # Check row count ratio (should be reasonable)
        row_ratio = min(synthetic_df.shape[0], original_df.shape[0]) / max(synthetic_df.shape[0], original_df.shape[0])
        
        return (column_similarity + row_ratio) / 2
    
    def _validate_data_types(self, synthetic_df: pd.DataFrame, 
                           original_df: pd.DataFrame) -> float:
        """Validate data type preservation"""
        common_cols = set(synthetic_df.columns) & set(original_df.columns)
        
        if not common_cols:
            return 0.0
        
        dtype_matches = 0
        for col in common_cols:
            if synthetic_df[col].dtype == original_df[col].dtype:
                dtype_matches += 1
        
        return dtype_matches / len(common_cols)
    
    def _validate_distributions(self, synthetic_df: pd.DataFrame, 
                              original_df: pd.DataFrame) -> float:
        """Validate distribution similarity"""
        common_cols = set(synthetic_df.columns) & set(original_df.columns)
        
        if not common_cols:
            return 0.0
        
        distribution_scores = []
        
        for col in common_cols:
            try:
                # Handle different data types
                if synthetic_df[col].dtype in ['object', 'string']:
                    # Categorical distribution
                    score = self._validate_categorical_distribution(
                        synthetic_df[col], original_df[col]
                    )
                elif synthetic_df[col].dtype in ['int64', 'float64']:
                    # Numerical distribution
                    score = self._validate_numerical_distribution(
                        synthetic_df[col], original_df[col]
                    )
                else:
                    score = 0.5  # Default score for unknown types
                
                distribution_scores.append(score)
                
            except Exception as e:
                print(f"Error validating distribution for column {col}: {e}")
                distribution_scores.append(0.0)
        
        return np.mean(distribution_scores) if distribution_scores else 0.0
    
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
        """Validate correlation preservation"""
        common_cols = list(set(synthetic_df.columns) & set(original_df.columns))
        
        if len(common_cols) < 2:
            return 1.0  # No correlations to validate
        
        try:
            # Calculate correlation matrices
            synthetic_corr = synthetic_df[common_cols].corr()
            original_corr = original_df[common_cols].corr()
            
            # Compare correlation matrices
            correlation_diff = np.abs(synthetic_corr - original_corr)
            correlation_similarity = 1 - np.mean(correlation_diff.values)
            
            return max(0, correlation_similarity)
            
        except Exception as e:
            print(f"Error in correlation validation: {e}")
            return 0.0
    
    def _validate_ranges(self, synthetic_df: pd.DataFrame, 
                       original_df: pd.DataFrame) -> float:
        """Validate value range preservation"""
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
    
    def _validate_relationships(self, synthetic_data: Dict[str, pd.DataFrame], 
                              relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate relationship integrity"""
        print("Validating relationship integrity...")
        
        relationship_scores = {}
        
        for rel in relationships:
            source_table = rel['source_table']
            target_table = rel['target_table']
            source_col = rel['source_column']
            target_col = rel['target_column']
            
            if source_table in synthetic_data and target_table in synthetic_data:
                score = self._validate_single_relationship(
                    synthetic_data[source_table], synthetic_data[target_table],
                    source_col, target_col
                )
                relationship_scores[f"{source_table}.{source_col} -> {target_table}.{target_col}"] = score
        
        return relationship_scores
    
    def _validate_single_relationship(self, source_df: pd.DataFrame, 
                                    target_df: pd.DataFrame,
                                    source_col: str, target_col: str) -> float:
        """Validate a single relationship"""
        try:
            if source_col not in source_df.columns or target_col not in target_df.columns:
                return 0.0
            
            # Get foreign key values
            source_values = set(source_df[source_col].dropna())
            target_values = set(target_df[target_col].dropna())
            
            # Check referential integrity
            if len(source_values) == 0:
                return 1.0  # No data to validate
            
            # Calculate how many source values exist in target
            valid_references = len(source_values & target_values)
            integrity_score = valid_references / len(source_values)
            
            return integrity_score
            
        except Exception as e:
            print(f"Error validating relationship: {e}")
            return 0.0
    
    def _assess_privacy(self, synthetic_data: Dict[str, pd.DataFrame], 
                       original_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Assess privacy preservation"""
        print("Assessing privacy...")
        
        privacy_scores = {}
        
        for table_name in synthetic_data.keys():
            if table_name in original_data:
                # Basic privacy metrics
                uniqueness_score = self._assess_uniqueness(
                    synthetic_data[table_name], original_data[table_name]
                )
                
                privacy_scores[table_name] = {
                    'uniqueness_score': uniqueness_score,
                    'overall_privacy_score': uniqueness_score
                }
        
        return privacy_scores
    
    def _assess_uniqueness(self, synthetic_df: pd.DataFrame, 
                          original_df: pd.DataFrame) -> float:
        """Assess uniqueness preservation"""
        try:
            # Check if synthetic data has reasonable uniqueness
            syn_uniqueness = synthetic_df.nunique().sum() / (synthetic_df.shape[0] * synthetic_df.shape[1])
            orig_uniqueness = original_df.nunique().sum() / (original_df.shape[0] * original_df.shape[1])
            
            # Similar uniqueness is good for privacy
            uniqueness_similarity = 1 - abs(syn_uniqueness - orig_uniqueness)
            
            return max(0, uniqueness_similarity)
            
        except Exception as e:
            print(f"Error in uniqueness assessment: {e}")
            return 0.0
    
    def _assess_statistical_quality(self, synthetic_data: Dict[str, pd.DataFrame], 
                                  original_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Assess overall statistical quality"""
        print("Assessing statistical quality...")
        
        overall_scores = []
        
        for table_name in synthetic_data.keys():
            if table_name in original_data:
                table_score = self._validate_table(
                    synthetic_data[table_name], original_data[table_name], table_name
                )
                overall_scores.append(table_score['overall_score'])
        
        return {
            'average_table_score': np.mean(overall_scores) if overall_scores else 0.0,
            'min_table_score': np.min(overall_scores) if overall_scores else 0.0,
            'max_table_score': np.max(overall_scores) if overall_scores else 0.0
        }
    
    def _calculate_overall_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        scores = []
        
        # Table scores
        if validation_results['table_scores']:
            table_scores = [score['overall_score'] for score in validation_results['table_scores'].values()]
            scores.append(np.mean(table_scores))
        
        # Relationship scores
        if validation_results['relationship_scores']:
            rel_scores = list(validation_results['relationship_scores'].values())
            scores.append(np.mean(rel_scores))
        
        # Privacy scores
        if validation_results['privacy_metrics']:
            privacy_scores = [score['overall_privacy_score'] for score in validation_results['privacy_metrics'].values()]
            scores.append(np.mean(privacy_scores))
        
        # Statistical scores
        if validation_results['statistical_metrics']:
            scores.append(validation_results['statistical_metrics']['average_table_score'])
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        overall_score = validation_results['overall_score']
        
        if overall_score < 0.7:
            recommendations.append("Overall quality is below threshold. Consider retraining with more data or adjusting model parameters.")
        
        # Table-specific recommendations
        for table_name, score in validation_results['table_scores'].items():
            if score['overall_score'] < 0.8:
                recommendations.append(f"Table '{table_name}' quality needs improvement. Check data distributions and correlations.")
        
        # Relationship recommendations
        if validation_results['relationship_scores']:
            rel_scores = list(validation_results['relationship_scores'].values())
            if np.mean(rel_scores) < 0.9:
                recommendations.append("Relationship integrity needs improvement. Check foreign key constraints.")
        
        # Privacy recommendations
        if validation_results['privacy_metrics']:
            privacy_scores = [score['overall_privacy_score'] for score in validation_results['privacy_metrics'].values()]
            if np.mean(privacy_scores) < 0.8:
                recommendations.append("Privacy preservation needs improvement. Consider additional anonymization techniques.")
        
        return recommendations 