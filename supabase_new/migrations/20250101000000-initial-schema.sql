-- Initial Schema Migration
-- This migration creates the core database structure for the synthetic data platform

-- Create projects table to organize schema designs
CREATE TABLE public.projects (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE
);

-- Create profiles table for additional user information
CREATE TABLE public.profiles (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
  display_name TEXT,
  avatar_url TEXT,
  bio TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create tables metadata table
CREATE TABLE public.table_metadata (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id UUID REFERENCES public.projects(id) ON DELETE CASCADE NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  business_rules TEXT,
  enhanced_description TEXT,
  domain_context TEXT,
  position_x INTEGER DEFAULT 0,
  position_y INTEGER DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  UNIQUE(project_id, name)
);

-- Create columns metadata table
CREATE TABLE public.column_metadata (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  table_id UUID REFERENCES public.table_metadata(id) ON DELETE CASCADE NOT NULL,
  name TEXT NOT NULL,
  data_type TEXT NOT NULL,
  is_nullable BOOLEAN DEFAULT true,
  is_primary_key BOOLEAN DEFAULT false,
  is_unique BOOLEAN DEFAULT false,
  default_value TEXT,
  max_length INTEGER,
  pattern TEXT,
  sample_values TEXT[],
  business_rules TEXT,
  enhanced_description TEXT,
  data_generation_rules TEXT,
  validation_rules TEXT,
  position INTEGER DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  UNIQUE(table_id, name)
);

-- Create relationships table for foreign key relationships
CREATE TABLE public.relationships (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  source_table_id UUID REFERENCES public.table_metadata(id) ON DELETE CASCADE NOT NULL,
  source_column_id UUID REFERENCES public.column_metadata(id) ON DELETE CASCADE NOT NULL,
  target_table_id UUID REFERENCES public.table_metadata(id) ON DELETE CASCADE NOT NULL,
  target_column_id UUID REFERENCES public.column_metadata(id) ON DELETE CASCADE NOT NULL,
  relationship_type TEXT DEFAULT 'one-to-many',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  UNIQUE(source_column_id, target_column_id)
);

-- Create API secrets table for storing external API keys
CREATE TABLE public.api_secrets (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  key_name TEXT NOT NULL UNIQUE,
  key_value TEXT NOT NULL,
  description TEXT,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create a table for storing LLM conversation history
CREATE TABLE public.llm_conversations (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id UUID REFERENCES public.projects(id) ON DELETE CASCADE NOT NULL,
  table_id UUID REFERENCES public.table_metadata(id) ON DELETE CASCADE,
  column_id UUID REFERENCES public.column_metadata(id) ON DELETE CASCADE,
  conversation_type TEXT NOT NULL CHECK (conversation_type IN ('table_description', 'column_business_rules', 'data_generation')),
  messages JSONB NOT NULL DEFAULT '[]',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create synthetic data generation history table
CREATE TABLE public.synthetic_data_generations (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id UUID REFERENCES public.projects(id) ON DELETE CASCADE NOT NULL,
  table_id UUID REFERENCES public.table_metadata(id) ON DELETE CASCADE,
  generation_config JSONB NOT NULL DEFAULT '{}',
  row_count INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
  output_file_path TEXT,
  error_message TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Add comments to tables
COMMENT ON TABLE public.projects IS 'Projects for organizing schema designs';
COMMENT ON TABLE public.profiles IS 'User profile information';
COMMENT ON TABLE public.table_metadata IS 'Metadata for database tables in schema designs';
COMMENT ON TABLE public.column_metadata IS 'Metadata for table columns in schema designs';
COMMENT ON TABLE public.relationships IS 'Foreign key relationships between tables';
COMMENT ON TABLE public.api_secrets IS 'External API keys and secrets';
COMMENT ON TABLE public.llm_conversations IS 'LLM conversation history for AI assistance';
COMMENT ON TABLE public.synthetic_data_generations IS 'History of synthetic data generation jobs';

-- Add comments to important columns
COMMENT ON COLUMN public.column_metadata.business_rules IS 'AI-generated business rules for the column';
COMMENT ON COLUMN public.column_metadata.enhanced_description IS 'AI-enhanced description of the column';
COMMENT ON COLUMN public.column_metadata.data_generation_rules IS 'Rules for generating synthetic data for this column';
COMMENT ON COLUMN public.column_metadata.validation_rules IS 'Validation rules for the column data'; 