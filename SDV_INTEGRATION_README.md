# SDV Integration for Synthetic Data Generation

## Overview

This document describes the SDV (Synthetic Data Vault) integration that enables synthetic data generation based on your schema design. The integration includes relationship management and AI-powered data generation.

## 🚀 New Features

### 1. Drag-and-Drop Relationship Management

- **Visual Relationship Creation**: Drag columns between tables to create relationships
- **Relationship Types**: Support for One-to-One, One-to-Many, and Many-to-Many relationships
- **Relationship Validation**: Prevents invalid relationships and self-references
- **Visual Feedback**: Clear indication of existing relationships

### 2. SDV-Powered Synthetic Data Generation

- **Multi-table Support**: Generate data for related tables using HMASynthesizer
- **Single Table Support**: Generate data for individual tables using GaussianCopulaSynthesizer
- **Quality Metrics**: Statistical similarity, correlation preservation, and privacy scores
- **Export Options**: CSV, JSON, and SQL export formats

### 3. Enhanced Schema Designer

- **Three-Tab Interface**: Schema, Relationships, and Synthetic Data tabs
- **Real-time Updates**: Live relationship management and data generation
- **AI Integration**: Leverages existing AI assistant for data generation rules

## 🏗️ Architecture

### Backend Components

```
backend/
├── sdv_service/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── start_sdv_service.sh     # Startup script
└── requirements.txt         # Main requirements
```

### Frontend Components

```
src/components/schema-designer/
├── RelationshipCanvas.tsx   # Drag-and-drop relationship management
├── SyntheticDataPanel.tsx  # SDV data generation interface
└── Enhanced SchemaDesigner # Updated with new tabs
```

## 🚀 Getting Started

### 1. Start the SDV Service

```bash
# Navigate to the backend directory
cd backend

# Start the SDV service
./start_sdv_service.sh
```

The service will be available at `http://localhost:8001`

### 2. Use the Enhanced Schema Designer

1. **Design Your Schema**: Create tables and columns as usual
2. **Create Relationships**: 
   - Go to the "Relationships" tab
   - Drag columns between tables to create relationships
   - Choose relationship type (One-to-One, One-to-Many, Many-to-Many)
3. **Generate Synthetic Data**:
   - Go to the "Synthetic Data" tab
   - Configure generation settings
   - Click "Generate Data"
   - Export results in your preferred format

## 📊 Features in Detail

### Relationship Management

**Features:**
- Visual drag-and-drop interface
- Real-time relationship validation
- Support for all relationship types
- Relationship deletion with confirmation
- Visual relationship indicators

**Usage:**
1. Select the "Relationships" tab
2. Drag a column from one table to another
3. Choose the relationship type
4. Confirm the relationship creation

### Synthetic Data Generation

**Features:**
- Automatic SDV metadata generation from schema
- Multi-table relationship support
- Quality metrics and validation
- Multiple export formats
- Configurable generation parameters

**Configuration Options:**
- **Data Scale**: Multiplier for data volume (1.0 = same as sample)
- **Sample Size**: Number of rows to generate per table
- **Quality Threshold**: Minimum quality score (0.0 - 1.0)
- **Include Relationships**: Preserve referential integrity

**Quality Metrics:**
- **Distribution Similarity**: How well synthetic data matches original distributions
- **Correlation Preservation**: Maintains relationships between columns
- **Privacy Score**: Measures data privacy preservation

## 🔧 Technical Details

### SDV Service API

**Endpoints:**
- `POST /api/sdv/generate` - Generate synthetic data
- `GET /api/sdv/status/{session_id}` - Check generation status
- `POST /api/sdv/validate` - Validate data quality
- `GET /api/sdv/health` - Health check

**Request Format:**
```json
{
  "tables": [
    {
      "name": "customers",
      "description": "Customer information",
      "columns": [
        {
          "name": "id",
          "data_type": "integer",
          "is_nullable": false,
          "is_primary_key": true,
          "is_unique": true
        }
      ]
    }
  ],
  "relationships": [
    {
      "source_table": "customers",
      "source_column": "id",
      "target_table": "orders",
      "target_column": "customer_id",
      "relationship_type": "one-to-many"
    }
  ],
  "scale": 1.0,
  "quality_settings": {
    "threshold": 0.8,
    "include_relationships": true
  }
}
```

### Database Schema

**Relationships Table:**
```sql
CREATE TABLE public.relationships (
  id UUID PRIMARY KEY,
  source_table_id UUID REFERENCES table_metadata(id),
  source_column_id UUID REFERENCES column_metadata(id),
  target_table_id UUID REFERENCES table_metadata(id),
  target_column_id UUID REFERENCES column_metadata(id),
  relationship_type TEXT DEFAULT 'one-to-many',
  created_at TIMESTAMP DEFAULT now()
);
```

## 🎯 Use Cases

### 1. Development and Testing
- Generate realistic test data for development
- Maintain referential integrity across tables
- Create data that matches production patterns

### 2. Data Privacy
- Create synthetic datasets for analysis
- Preserve statistical properties while protecting privacy
- Generate data for machine learning training

### 3. Schema Validation
- Test schema design with realistic data
- Validate business rules and constraints
- Identify potential issues before production

## 🔍 Troubleshooting

### Common Issues

1. **SDV Service Not Starting**
   - Check if the virtual environment path is correct
   - Ensure all dependencies are installed
   - Check port 8001 is available

2. **Relationship Creation Fails**
   - Verify both tables and columns exist
   - Check for duplicate relationships
   - Ensure proper foreign key constraints

3. **Data Generation Errors**
   - Check schema validity
   - Verify relationship consistency
   - Review SDV service logs

### Debug Commands

```bash
# Check SDV service health
curl http://localhost:8001/api/sdv/health

# Test data generation
curl -X POST http://localhost:8001/api/sdv/generate \
  -H "Content-Type: application/json" \
  -d '{"tables":[],"relationships":[],"scale":1.0}'
```

## 🚀 Next Steps

### Planned Enhancements

1. **Advanced Quality Metrics**
   - Custom business rule validation
   - Domain-specific quality measures
   - Real-time quality monitoring

2. **Production Data Integration**
   - Direct database connections
   - Incremental data generation
   - Real-time data streaming

3. **AI-Enhanced Generation**
   - Smart parameter selection
   - Automatic quality optimization
   - Context-aware data generation

## 📚 Resources

- [SDV Documentation](https://docs.sdv.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Query Documentation](https://tanstack.com/query)

---

This integration provides a powerful foundation for synthetic data generation while maintaining the existing AI-powered schema design capabilities. 