# Data Source Connectors & File Upload System

This document describes the comprehensive data source connector system implemented in Synthe-Flow, enabling users to connect to various data sources and upload files for synthetic data generation.

## Overview

The enhanced system provides:
- **File Upload**: Drag-and-drop interface for local file uploads
- **Database Connectors**: PostgreSQL, Snowflake, Oracle
- **Cloud Storage**: Amazon S3, Google Cloud Storage, Azure Blob Storage
- **API Connectors**: Custom API endpoints
- **Data Catalogs**: Apache Atlas integration
- **Dynamic File Generation**: Multiple output formats (JSON, CSV, Excel, SQL)

## Backend Architecture

### Enhanced Backend Service (`backend/sdv_service/main.py`)

The backend now includes:

1. **File Upload Endpoint**
   - `POST /api/upload/file` - Upload files for processing
   - Supports CSV, Excel, JSON, Parquet formats
   - Automatic file validation and storage

2. **Connector Management**
   - `GET /api/connectors/available` - List available connectors
   - `POST /api/connectors/test` - Test connector connections

3. **Enhanced Data Generation**
   - `POST /api/sdv/generate` - Generate synthetic data with multiple sources
   - Support for multiple output formats
   - Dynamic file generation based on table count

### Connectors Module (`backend/sdv_service/connectors.py`)

Comprehensive connector implementations:

#### Database Connectors
- **PostgreSQL**: Full schema and data loading support
- **Snowflake**: Warehouse and database connectivity
- **Oracle**: Database and schema extraction

#### Cloud Storage Connectors
- **Amazon S3**: Bucket and file access
- **Google Cloud Storage**: GCS bucket operations
- **Azure Blob Storage**: Container and blob management

#### API Connectors
- **Custom APIs**: RESTful API integration
- **JSON/CSV Parsing**: Automatic response parsing
- **Authentication**: Header and auth support

#### Data Catalog Connectors
- **Apache Atlas**: Metadata and schema extraction
- **Extensible**: Framework for other catalog systems

## Frontend Components

### Enhanced SyntheticDataPanel

The main synthetic data generation interface now includes:

1. **Data Sources Tab**
   - File upload drag-and-drop zone
   - Connector selection grid
   - Active data source management

2. **Configuration Tab**
   - Generation settings
   - Output format selection
   - Quality thresholds

3. **Results Tab**
   - Generation status monitoring
   - Quality metrics display
   - Data preview

4. **Export Tab**
   - Multiple export formats
   - File download options

### DataSourceManager Component

A dedicated component for managing data sources:

- **File Upload Interface**: Drag-and-drop with progress tracking
- **Connector Configuration**: Dynamic forms for each connector type
- **Connection Testing**: Real-time connection validation
- **Source Management**: Add/remove data sources

## Supported Data Sources

### 1. Local File Upload
- **Formats**: CSV, Excel (.xlsx, .xls), JSON, Parquet
- **Features**: Drag-and-drop, multiple file selection
- **Processing**: Automatic table name extraction

### 2. PostgreSQL
- **Connection**: Host, port, database, credentials
- **Features**: Schema extraction, table selection
- **Limits**: Configurable row limits

### 3. Snowflake
- **Connection**: Account, warehouse, database, credentials
- **Features**: Multi-table support, warehouse management
- **Security**: Secure credential handling

### 4. Oracle
- **Connection**: DSN, credentials
- **Features**: Schema inspection, table metadata
- **Compatibility**: Oracle 11g+ support

### 5. Amazon S3
- **Connection**: Bucket, access keys, region
- **Features**: File listing, automatic format detection
- **Security**: IAM role and access key support

### 6. Google Cloud Storage
- **Connection**: Bucket, service account key
- **Features**: GCS bucket operations
- **Authentication**: Service account JSON

### 7. Azure Blob Storage
- **Connection**: Container, connection string
- **Features**: Blob operations, container management
- **Security**: SAS tokens and connection strings

### 8. Custom API
- **Connection**: Base URL, endpoints configuration
- **Features**: JSON/CSV response parsing
- **Authentication**: Headers and auth support

### 9. Data Catalog (Apache Atlas)
- **Connection**: Endpoint URL, service principal
- **Features**: Metadata extraction, schema discovery
- **Integration**: Atlas entity and process support

## Configuration Examples

### PostgreSQL Configuration
```json
{
  "type": "postgres",
  "config": {
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "username": "user",
    "password": "password",
    "tables": ["users", "orders"]
  }
}
```

### Snowflake Configuration
```json
{
  "type": "snowflake",
  "config": {
    "account": "your-account.snowflakecomputing.com",
    "warehouse": "COMPUTE_WH",
    "database": "MY_DB",
    "username": "user",
    "password": "password"
  }
}
```

### S3 Configuration
```json
{
  "type": "s3",
  "config": {
    "bucket": "my-data-bucket",
    "access_key_id": "AKIA...",
    "secret_access_key": "secret...",
    "region": "us-east-1",
    "file_paths": ["data/users.csv", "data/orders.xlsx"]
  }
}
```

### API Configuration
```json
{
  "type": "api",
  "config": {
    "base_url": "https://api.example.com",
    "endpoints": [
      {
        "path": "/users",
        "table_name": "users"
      },
      {
        "path": "/orders",
        "table_name": "orders"
      }
    ],
    "headers": {
      "Authorization": "Bearer token"
    }
  }
}
```

## Output Formats

The system supports multiple output formats:

### 1. JSON
- **Format**: Structured JSON with table names as keys
- **Use Case**: API consumption, data exchange
- **File**: `synthetic_data.json`

### 2. CSV
- **Format**: Separate CSV file for each table
- **Use Case**: Excel import, data analysis
- **Files**: `table1.csv`, `table2.csv`, etc.

### 3. Excel
- **Format**: Single Excel file with multiple sheets
- **Use Case**: Business reporting, data sharing
- **File**: `synthetic_data.xlsx`

### 4. SQL
- **Format**: SQL INSERT statements
- **Use Case**: Database population, data migration
- **File**: `synthetic_data.sql`

## Installation & Setup

### Backend Dependencies

Add to `backend/requirements.txt`:
```
# File upload and processing
python-multipart==0.0.6
aiofiles==23.2.1

# Database connectors
psycopg2-binary==2.9.9
snowflake-connector-python==3.6.0
cx-Oracle==8.3.0
sqlalchemy==2.0.23

# Cloud storage
boto3==1.34.0
google-cloud-storage==2.10.0
azure-storage-blob==12.19.0

# Data processing
openpyxl==3.1.2
xlrd==2.0.1

# API connectors
requests==2.31.0
httpx==0.25.2

# Data catalog integration
pyapacheatlas==0.17.0

# Configuration
python-dotenv==1.0.0
```

### Frontend Dependencies

Add to `package.json`:
```json
{
  "dependencies": {
    "react-dropzone": "^14.2.3"
  }
}
```

### Environment Variables

Create `.env` file for connector configurations:
```env
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mydb
POSTGRES_USER=user
POSTGRES_PASSWORD=password

# Snowflake
SNOWFLAKE_ACCOUNT=your-account
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DB=MY_DB
SNOWFLAKE_USER=user
SNOWFLAKE_PASSWORD=password

# S3
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1

# API
API_BASE_URL=https://api.example.com
API_TOKEN=your-token
```

## Usage Workflow

### 1. File Upload
1. Navigate to the Data Sources tab
2. Drag and drop files or click to select
3. Files are automatically uploaded and processed
4. Review uploaded files in the interface

### 2. Database Connection
1. Click on the desired database connector
2. Fill in connection details
3. Test the connection
4. Select tables for data extraction

### 3. Cloud Storage
1. Choose cloud storage connector
2. Configure credentials and bucket/container
3. Specify file paths to process
4. Test connection and load data

### 4. API Integration
1. Select API connector
2. Configure base URL and endpoints
3. Add authentication headers if needed
4. Test and validate connection

### 5. Data Generation
1. Configure generation settings
2. Select output format
3. Set quality thresholds
4. Generate synthetic data
5. Export in desired format

## Security Considerations

### Credential Management
- Passwords and keys are encrypted in transit
- No credentials stored in plain text
- Session-based authentication for connectors

### File Upload Security
- File type validation
- Size limits and virus scanning
- Secure file storage with unique names

### API Security
- HTTPS enforcement
- Token-based authentication
- Rate limiting and request validation

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Verify credentials and network connectivity
   - Check firewall and security group settings
   - Validate endpoint URLs and ports

2. **File Upload Errors**
   - Ensure file format is supported
   - Check file size limits
   - Verify upload directory permissions

3. **Data Loading Issues**
   - Confirm table names and schema
   - Check data format compatibility
   - Validate API response structure

### Debug Mode

Enable debug logging in the backend:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

Test backend connectivity:
```bash
curl http://localhost:8002/api/sdv/health
```

## Future Enhancements

### Planned Features
- **Real-time Data Streaming**: Live data source connections
- **Advanced Transformations**: Data preprocessing and cleaning
- **Scheduled Generation**: Automated synthetic data creation
- **Quality Monitoring**: Real-time quality metrics
- **Multi-tenant Support**: User-specific data sources

### Connector Extensions
- **MongoDB**: NoSQL database support
- **Redis**: Cache and session data
- **Kafka**: Stream processing integration
- **Elasticsearch**: Search and analytics data

### Integration Enhancements
- **Data Lineage**: Track data source relationships
- **Version Control**: Data source configuration history
- **Collaboration**: Shared data source management
- **Audit Logging**: Comprehensive activity tracking

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review connector-specific documentation
3. Test with sample data first
4. Enable debug logging for detailed error messages

## Contributing

To add new connectors:
1. Implement the connector interface
2. Add configuration forms
3. Include proper error handling
4. Add comprehensive tests
5. Update documentation 