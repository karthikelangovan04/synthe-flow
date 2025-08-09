# Supabase Configuration for Synthetic Data Platform

This directory contains a comprehensive Supabase configuration for the synthetic data platform, including all necessary migrations, functions, and security policies.

## Directory Structure

```
supabase_new/
├── config.toml                 # Supabase configuration
├── migrations/                 # Database migrations
│   ├── 20250101000000-initial-schema.sql
│   ├── 20250101000001-security-policies.sql
│   ├── 20250101000002-functions-triggers.sql
│   └── 20250101000003-seed-data.sql
├── functions/                  # Edge functions
│   └── llm-assistant/
│       └── index.ts
└── README.md                   # This file
```

## Database Schema

### Core Tables

1. **projects** - User projects for organizing schema designs
2. **profiles** - User profile information
3. **table_metadata** - Metadata for database tables in schema designs
4. **column_metadata** - Metadata for table columns in schema designs
5. **relationships** - Foreign key relationships between tables
6. **api_secrets** - External API keys and secrets (service role only)
7. **llm_conversations** - LLM conversation history for AI assistance
8. **synthetic_data_generations** - History of synthetic data generation jobs

### Key Features

- **Row Level Security (RLS)** - All tables have RLS enabled with appropriate policies
- **Automatic Timestamps** - All tables have `created_at` and `updated_at` fields with automatic updates
- **User Ownership** - All data is properly scoped to user ownership
- **API Key Management** - Secure storage of external API keys
- **Full-Text Search** - Indexes for efficient search functionality

## Migration Files

### 1. Initial Schema (20250101000000-initial-schema.sql)
- Creates all core tables with proper relationships
- Includes business rules and enhanced description fields
- Adds comprehensive table and column comments

### 2. Security Policies (20250101000001-security-policies.sql)
- Enables RLS on all tables
- Creates comprehensive security policies
- Ensures data isolation between users

### 3. Functions and Triggers (20250101000002-functions-triggers.sql)
- Utility functions for timestamp updates
- User management functions
- Project validation functions
- Automatic triggers for data integrity

### 4. Seed Data (20250101000003-seed-data.sql)
- Initial API secrets configuration
- Performance indexes
- Full-text search indexes

## Edge Functions

### LLM Assistant (`functions/llm-assistant/index.ts`)
- Handles AI-powered schema assistance
- Integrates with OpenAI API
- Validates project ownership
- Saves conversation history
- Includes comprehensive error handling

## Security Features

### Row Level Security Policies
- **Projects**: Users can only access their own projects
- **Profiles**: Public read, user-specific write
- **Metadata**: Scoped to project ownership
- **API Secrets**: Service role only access
- **Conversations**: Scoped to project ownership

### Data Validation
- Foreign key constraints
- Check constraints for enum values
- Unique constraints where appropriate
- NOT NULL constraints for required fields

## Setup Instructions

### 1. Initialize Supabase
```bash
# Navigate to the supabase_new directory
cd synthe-flow/supabase_new

# Initialize Supabase (if not already done)
supabase init

# Link to your Supabase project
supabase link --project-ref mlwqzkcwnmjeyobixwap
```

### 2. Apply Migrations
```bash
# Apply all migrations
supabase db push

# Or apply migrations individually
supabase migration up
```

### 3. Deploy Edge Functions
```bash
# Deploy the LLM assistant function
supabase functions deploy llm-assistant
```

### 4. Configure Environment Variables
Set the following environment variables in your Supabase dashboard:
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Your service role key

### 5. Update API Keys
Update the API keys in the `api_secrets` table:
```sql
UPDATE api_secrets 
SET key_value = 'your-actual-openai-api-key' 
WHERE key_name = 'OPENAI_API_KEY';
```

## API Usage

### LLM Assistant Function
```typescript
const response = await fetch('/functions/v1/llm-assistant', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${supabaseKey}`
  },
  body: JSON.stringify({
    messages: [
      { role: 'system', content: 'You are a database schema expert.' },
      { role: 'user', content: 'Help me design a user table.' }
    ],
    context: {
      projectId: 'project-uuid',
      tableId: 'table-uuid',
      columnId: 'column-uuid'
    },
    model: 'gpt-4',
    temperature: 0.7,
    maxTokens: 1000
  })
});
```

## Database Functions

### Utility Functions
- `update_updated_at_column()` - Updates timestamp on row updates
- `handle_new_user()` - Creates profile on user signup
- `validate_project_ownership(project_uuid)` - Validates project access
- `get_user_projects()` - Returns user's projects with metadata
- `get_project_schema(project_uuid)` - Returns complete project schema as JSON

## Performance Optimizations

### Indexes
- Foreign key indexes for all relationships
- Full-text search indexes for search functionality
- Composite indexes for common query patterns

### Query Optimization
- JSON aggregation for efficient schema retrieval
- Proper join strategies for hierarchical data
- Optimized RLS policies

## Monitoring and Maintenance

### Logging
- All edge functions include comprehensive logging
- Database operations are logged for debugging
- Error tracking for API failures

### Backup Strategy
- Regular database backups through Supabase
- Migration version control
- Rollback procedures documented

## Troubleshooting

### Common Issues

1. **RLS Policy Errors**
   - Ensure user is authenticated
   - Check project ownership
   - Verify policy syntax

2. **API Key Issues**
   - Verify key is active in `api_secrets` table
   - Check service role permissions
   - Validate API key format

3. **Migration Failures**
   - Check for existing data conflicts
   - Verify migration order
   - Review constraint violations

### Debug Commands
```bash
# Check migration status
supabase migration list

# View logs
supabase logs

# Reset database (development only)
supabase db reset
```

## Security Considerations

1. **API Key Storage**: All external API keys are stored securely in the database
2. **Access Control**: RLS ensures data isolation between users
3. **Input Validation**: All user inputs are validated and sanitized
4. **Error Handling**: Sensitive information is not exposed in error messages
5. **Audit Trail**: All operations are logged for security monitoring

## Future Enhancements

1. **Additional LLM Providers**: Support for Claude, Gemini, etc.
2. **Advanced Analytics**: Usage tracking and analytics
3. **Collaboration Features**: Multi-user project sharing
4. **Export Formats**: Additional data export options
5. **Real-time Features**: WebSocket support for live updates 