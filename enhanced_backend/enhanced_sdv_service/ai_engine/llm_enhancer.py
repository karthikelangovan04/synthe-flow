"""
LLM Enhancer for Schema Understanding and Business Rule Generation
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Any, Optional
import json

class SchemaEncoder(nn.Module):
    """Neural encoder for database schemas"""
    
    def __init__(self, hidden_size=256, num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Column type embeddings
        self.data_type_embedding = nn.Embedding(20, hidden_size)  # Common data types
        
        # Constraint embeddings
        self.constraint_embedding = nn.Embedding(10, hidden_size)  # PK, FK, unique, etc.
        
        # Transformer for schema understanding
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1
            ),
            num_layers=num_layers
        )
        
        # Output projections
        self.schema_embedding = nn.Linear(hidden_size, hidden_size)
        self.relationship_predictor = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, schema_data):
        """Encode schema information"""
        # Extract features from schema
        column_embeddings = self._encode_columns(schema_data['columns'])
        table_embeddings = self._encode_table_context(schema_data)
        
        # Combine embeddings
        combined = torch.cat([column_embeddings, table_embeddings], dim=1)
        
        # Transform through transformer
        transformed = self.transformer(combined.unsqueeze(0))
        
        # Generate outputs
        schema_embedding = self.schema_embedding(transformed.squeeze(0))
        relationship_features = self.relationship_predictor(transformed.squeeze(0))
        
        return {
            'schema_embedding': schema_embedding,
            'relationship_features': relationship_features
        }
    
    def _encode_columns(self, columns):
        """Encode column information"""
        embeddings = []
        for col in columns:
            # Data type embedding
            dtype_emb = self.data_type_embedding(self._get_data_type_id(col['data_type']))
            
            # Constraint embedding
            constraint_emb = self.constraint_embedding(self._get_constraint_id(col))
            
            # Combine
            col_emb = dtype_emb + constraint_emb
            embeddings.append(col_emb)
        
        return torch.stack(embeddings)
    
    def _encode_table_context(self, schema_data):
        """Encode table-level context"""
        # Simple table context encoding
        table_emb = torch.randn(len(schema_data['columns']), self.hidden_size)
        return table_emb
    
    def _get_data_type_id(self, data_type):
        """Map data type to ID"""
        type_map = {
            'integer': 0, 'varchar': 1, 'text': 2, 'boolean': 3,
            'timestamp': 4, 'decimal': 5, 'float': 6, 'date': 7
        }
        return type_map.get(data_type.lower(), 0)
    
    def _get_constraint_id(self, column):
        """Map constraints to ID"""
        if column.get('is_primary_key'):
            return 0
        elif column.get('is_unique'):
            return 1
        elif column.get('is_nullable') == False:
            return 2
        else:
            return 3

class LLMEnhancer:
    """Enhanced LLM integration for schema understanding"""
    
    def __init__(self):
        self.schema_encoder = SchemaEncoder()
        self.tokenizer = None
        self.model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize transformer models"""
        try:
            # Use a smaller model for MVP
            model_name = "microsoft/DialoGPT-small"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Add special tokens for schema understanding
            special_tokens = {
                'additional_special_tokens': [
                    '[TABLE]', '[/TABLE]', '[COLUMN]', '[/COLUMN]',
                    '[RELATIONSHIP]', '[/RELATIONSHIP]', '[CONSTRAINT]', '[/CONSTRAINT]'
                ]
            }
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            
        except Exception as e:
            print(f"Warning: Could not load transformer model: {e}")
            self.model = None
    
    def understand_schema(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """Understand and analyze database schema"""
        try:
            # Encode schema using neural encoder
            schema_encoding = self.schema_encoder(schema_data)
            
            # Generate business rules
            business_rules = self._generate_business_rules(schema_data)
            
            # Analyze relationships
            relationship_analysis = self._analyze_relationships(schema_data)
            
            return {
                'schema_embedding': schema_encoding['schema_embedding'].detach().numpy(),
                'business_rules': business_rules,
                'relationship_analysis': relationship_analysis,
                'complexity_score': self._calculate_complexity(schema_data)
            }
            
        except Exception as e:
            print(f"Error in schema understanding: {e}")
            return self._fallback_understanding(schema_data)
    
    def _generate_business_rules(self, schema_data: Dict[str, Any]) -> List[str]:
        """Generate business rules from schema"""
        rules = []
        
        for col in schema_data.get('columns', []):
            # Data type specific rules
            if col['data_type'] == 'email':
                rules.append(f"{col['name']} must be a valid email format")
            elif col['data_type'] == 'phone':
                rules.append(f"{col['name']} must be a valid phone number")
            elif col['data_type'] == 'date':
                rules.append(f"{col['name']} must be a valid date")
            
            # Constraint specific rules
            if col.get('is_primary_key'):
                rules.append(f"{col['name']} must be unique and non-null")
            elif col.get('is_unique'):
                rules.append(f"{col['name']} must be unique across all records")
            elif col.get('is_nullable') == False:
                rules.append(f"{col['name']} cannot be null")
        
        return rules
    
    def _analyze_relationships(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze table relationships"""
        relationships = schema_data.get('relationships', [])
        
        analysis = {
            'total_relationships': len(relationships),
            'relationship_types': {},
            'complexity_level': 'simple'
        }
        
        # Count relationship types
        for rel in relationships:
            rel_type = rel.get('relationship_type', 'unknown')
            analysis['relationship_types'][rel_type] = analysis['relationship_types'].get(rel_type, 0) + 1
        
        # Determine complexity
        if len(relationships) > 10:
            analysis['complexity_level'] = 'complex'
        elif len(relationships) > 5:
            analysis['complexity_level'] = 'moderate'
        
        return analysis
    
    def _calculate_complexity(self, schema_data: Dict[str, Any]) -> float:
        """Calculate schema complexity score (0-1)"""
        columns = len(schema_data.get('columns', []))
        relationships = len(schema_data.get('relationships', []))
        tables = schema_data.get('table_count', 1)
        
        # Simple complexity formula
        complexity = (
            (columns / 100) * 0.4 +
            (relationships / 50) * 0.4 +
            (tables / 20) * 0.2
        )
        
        return min(complexity, 1.0)
    
    def _fallback_understanding(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback when neural models fail"""
        return {
            'schema_embedding': np.random.randn(256),
            'business_rules': self._generate_business_rules(schema_data),
            'relationship_analysis': self._analyze_relationships(schema_data),
            'complexity_score': self._calculate_complexity(schema_data)
        }
    
    def enhance_column_description(self, column_data: Dict[str, Any]) -> str:
        """Generate enhanced column descriptions"""
        base_desc = column_data.get('description', '')
        
        # Add data type context
        dtype = column_data.get('data_type', '')
        if dtype == 'email':
            enhanced = f"Email address field for user communication. {base_desc}"
        elif dtype == 'phone':
            enhanced = f"Phone number field for contact information. {base_desc}"
        elif dtype == 'date':
            enhanced = f"Date field for temporal data. {base_desc}"
        elif dtype == 'decimal':
            enhanced = f"Numeric field for financial calculations. {base_desc}"
        else:
            enhanced = f"{dtype.title()} field. {base_desc}"
        
        return enhanced.strip()
    
    def suggest_relationships(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest potential relationships between tables"""
        suggestions = []
        
        for i, table1 in enumerate(tables):
            for j, table2 in enumerate(tables[i+1:], i+1):
                # Look for common column patterns
                common_patterns = self._find_common_patterns(table1, table2)
                
                for pattern in common_patterns:
                    suggestions.append({
                        'source_table': table1['name'],
                        'source_column': pattern['col1'],
                        'target_table': table2['name'],
                        'target_column': pattern['col2'],
                        'relationship_type': 'one-to-many',
                        'confidence': pattern['confidence']
                    })
        
        return suggestions
    
    def _find_common_patterns(self, table1: Dict[str, Any], table2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find common column patterns between tables"""
        patterns = []
        
        for col1 in table1.get('columns', []):
            for col2 in table2.get('columns', []):
                # Check for common naming patterns
                if self._is_related_column(col1, col2):
                    patterns.append({
                        'col1': col1['name'],
                        'col2': col2['name'],
                        'confidence': 0.8
                    })
        
        return patterns
    
    def _is_related_column(self, col1: Dict[str, Any], col2: Dict[str, Any]) -> bool:
        """Check if two columns are potentially related"""
        name1 = col1['name'].lower()
        name2 = col2['name'].lower()
        
        # Common patterns
        patterns = [
            ('id', 'id'),
            ('user_id', 'id'),
            ('customer_id', 'id'),
            ('order_id', 'id'),
            ('product_id', 'id')
        ]
        
        for pattern1, pattern2 in patterns:
            if (pattern1 in name1 and pattern2 in name2) or (pattern2 in name1 and pattern1 in name2):
                return True
        
        return False 