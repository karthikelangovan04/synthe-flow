# Supabase Migration Comparison

This document compares the existing Supabase setup with the new comprehensive setup.

## Issues Found in Current Setup

### 1. Missing API Secrets Table
**Problem**: The LLM assistant function references `api_secrets` table but it doesn't exist in migrations.

**Current Function Code**:
```typescript
const { data: apiKeyData, error: apiKeyError } = await supabaseClient
  .from('api_secrets')  // ❌ This table doesn't exist
  .select('key_value')
  .eq('key_name', 'OPENAI_API_KEY')
  .single()
```

**Solution**: Added `api_secrets` table in the new setup.

### 2. Inconsistent Migration Order
**Problem**: Migration files have inconsistent timestamps and ordering.

**Current Files**:
- `20250728155606-8bf4ecf6-dca6-4262-a05d-b2735f9eb0b6.sql` (Initial schema)
- `20250728155633-49c70e53-8a9c-4fdb-b727-eacaa864fbef.sql` (Function fix)
- `20250729020757-16bdc76d-69d4-401c-8e6b-aef7cd64deba.sql` (Profiles)
- `20250730022916-f492f0aa-9939-4bad-b8d0-7867ce0dae20.sql` (User ID fix)
- `20250115000000-add-business-rules.sql` (Business rules)
- `20250115000001-add-business-rules-columns.sql` (Business rules columns)

**Issues**:
- Business rules migrations have dates in 2025 (future)
- Inconsistent naming conventions
- Some migrations modify the same columns

**Solution**: Reorganized with logical timestamp ordering.

### 3. Missing Synthetic Data Generation Tracking
**Problem**: No way to track synthetic data generation jobs.

**Solution**: Added `synthetic_data_generations` table.

### 4. Incomplete Security Policies
**Problem**: Some tables may not have comprehensive RLS policies.

**Solution**: Comprehensive security policies for all tables.

### 5. Missing Performance Optimizations
**Problem**: No indexes for better query performance.

**Solution**: Added strategic indexes and full-text search.

## New Setup Improvements

### 1. Complete Schema Definition
```sql
-- New tables added:
- api_secrets (for API key management)
- synthetic_data_generations (for job tracking)
```

### 2. Enhanced Security
```sql
-- All tables have RLS enabled
-- Comprehensive policies for each table
-- Service role restrictions for sensitive data
```

### 3. Better Function Organization
```sql
-- Utility functions for common operations
-- Project validation functions
-- JSON aggregation for efficient data retrieval
```

### 4. Performance Optimizations
```sql
-- Strategic indexes on foreign keys
-- Full-text search indexes
-- Composite indexes for common queries
```

### 5. Comprehensive Documentation
- Detailed README with setup instructions
- API usage examples
- Troubleshooting guide
- Security considerations

## Migration Strategy

### Option 1: Fresh Start (Recommended)
1. Backup existing data
2. Apply new migrations to fresh database
3. Migrate data if needed

### Option 2: Incremental Migration
1. Create missing tables (`api_secrets`, `synthetic_data_generations`)
2. Add missing indexes
3. Update function security
4. Test thoroughly

### Option 3: Hybrid Approach
1. Use new setup for new features
2. Gradually migrate existing data
3. Maintain backward compatibility

## Data Migration Script

If you need to migrate existing data, here's a script:

```sql
-- Migrate existing projects (if any)
INSERT INTO public.projects_new (id, name, description, user_id, created_at, updated_at)
SELECT id, name, description, user_id, created_at, updated_at
FROM public.projects;

-- Migrate existing table metadata
INSERT INTO public.table_metadata_new (id, project_id, name, description, position_x, position_y, created_at, updated_at)
SELECT id, project_id, name, description, position_x, position_y, created_at, updated_at
FROM public.table_metadata;

-- Migrate existing column metadata
INSERT INTO public.column_metadata_new (id, table_id, name, data_type, is_nullable, is_primary_key, is_unique, default_value, max_length, pattern, sample_values, position, created_at, updated_at)
SELECT id, table_id, name, data_type, is_nullable, is_primary_key, is_unique, default_value, max_length, pattern, sample_values, position, created_at, updated_at
FROM public.column_metadata;

-- Migrate existing relationships
INSERT INTO public.relationships_new (id, source_table_id, source_column_id, target_table_id, target_column_id, relationship_type, created_at)
SELECT id, source_table_id, source_column_id, target_table_id, target_column_id, relationship_type, created_at
FROM public.relationships;
```

## Testing Checklist

### Database Schema
- [ ] All tables created successfully
- [ ] Foreign key constraints working
- [ ] RLS policies functioning
- [ ] Triggers updating timestamps
- [ ] Indexes improving performance

### Edge Functions
- [ ] LLM assistant function deployed
- [ ] API key retrieval working
- [ ] Error handling functioning
- [ ] Conversation saving working
- [ ] Security validation working

### Application Integration
- [ ] Frontend can create projects
- [ ] Schema designer working
- [ ] AI assistance functioning
- [ ] Data generation tracking working
- [ ] User authentication working

## Rollback Plan

If issues arise, you can rollback using:

```bash
# Reset to previous migration
supabase migration down

# Or reset entire database (development only)
supabase db reset
```

## Next Steps

1. **Review the new setup** - Examine all migration files
2. **Test in development** - Apply to a test environment first
3. **Plan migration strategy** - Choose the best approach for your data
4. **Execute migration** - Apply changes to production
5. **Monitor performance** - Watch for any issues
6. **Update documentation** - Keep team informed of changes 