-- Seed Data Migration
-- This migration adds initial data and sample records

-- Insert sample API secrets (these should be updated with real values)
INSERT INTO public.api_secrets (key_name, key_value, description, is_active) VALUES
  ('OPENAI_API_KEY', 'your-openai-api-key-here', 'OpenAI API key for LLM functionality', true),
  ('ANTHROPIC_API_KEY', 'your-anthropic-api-key-here', 'Anthropic API key for Claude integration', true)
ON CONFLICT (key_name) DO NOTHING;

-- Insert sample project for demonstration (will be created by first user)
-- Note: This is commented out as it requires a valid user_id
-- INSERT INTO public.projects (name, description, user_id) VALUES
--   ('Sample HR Database', 'A comprehensive HR database schema with employees, departments, and performance data', '00000000-0000-0000-0000-000000000000');

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON public.projects(user_id);
CREATE INDEX IF NOT EXISTS idx_table_metadata_project_id ON public.table_metadata(project_id);
CREATE INDEX IF NOT EXISTS idx_column_metadata_table_id ON public.column_metadata(table_id);
CREATE INDEX IF NOT EXISTS idx_relationships_source_table_id ON public.relationships(source_table_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target_table_id ON public.relationships(target_table_id);
CREATE INDEX IF NOT EXISTS idx_llm_conversations_project_id ON public.llm_conversations(project_id);
CREATE INDEX IF NOT EXISTS idx_synthetic_data_generations_project_id ON public.synthetic_data_generations(project_id);
CREATE INDEX IF NOT EXISTS idx_api_secrets_key_name ON public.api_secrets(key_name);

-- Create full-text search indexes for better search functionality
CREATE INDEX IF NOT EXISTS idx_projects_name_description_fts ON public.projects USING gin(to_tsvector('english', name || ' ' || COALESCE(description, '')));
CREATE INDEX IF NOT EXISTS idx_table_metadata_name_description_fts ON public.table_metadata USING gin(to_tsvector('english', name || ' ' || COALESCE(description, '') || ' ' || COALESCE(enhanced_description, '')));
CREATE INDEX IF NOT EXISTS idx_column_metadata_name_description_fts ON public.column_metadata USING gin(to_tsvector('english', name || ' ' || COALESCE(enhanced_description, '') || ' ' || COALESCE(business_rules, ''))); 