-- Functions and Triggers Migration
-- This migration creates utility functions and triggers for the application

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER 
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

-- Create function to handle new user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = ''
AS $$
BEGIN
  INSERT INTO public.profiles (user_id, display_name)
  VALUES (NEW.id, COALESCE(NEW.raw_user_meta_data ->> 'display_name', NEW.email));
  RETURN NEW;
END;
$$;

-- Create function to validate project ownership
CREATE OR REPLACE FUNCTION public.validate_project_ownership(project_uuid UUID)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = ''
AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 FROM public.projects 
    WHERE id = project_uuid 
    AND user_id = auth.uid()
  );
END;
$$;

-- Create function to get user's projects with metadata
CREATE OR REPLACE FUNCTION public.get_user_projects()
RETURNS TABLE (
  id UUID,
  name TEXT,
  description TEXT,
  table_count BIGINT,
  created_at TIMESTAMP WITH TIME ZONE,
  updated_at TIMESTAMP WITH TIME ZONE
)
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = ''
AS $$
BEGIN
  RETURN QUERY
  SELECT 
    p.id,
    p.name,
    p.description,
    COUNT(tm.id)::BIGINT as table_count,
    p.created_at,
    p.updated_at
  FROM public.projects p
  LEFT JOIN public.table_metadata tm ON p.id = tm.project_id
  WHERE p.user_id = auth.uid()
  GROUP BY p.id, p.name, p.description, p.created_at, p.updated_at
  ORDER BY p.updated_at DESC;
END;
$$;

-- Create function to get project schema with all metadata
CREATE OR REPLACE FUNCTION public.get_project_schema(project_uuid UUID)
RETURNS JSON
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = ''
AS $$
DECLARE
  result JSON;
BEGIN
  -- Check if user owns the project
  IF NOT public.validate_project_ownership(project_uuid) THEN
    RAISE EXCEPTION 'Access denied: Project not found or access denied';
  END IF;

  SELECT json_build_object(
    'project', json_build_object(
      'id', p.id,
      'name', p.name,
      'description', p.description,
      'created_at', p.created_at,
      'updated_at', p.updated_at
    ),
    'tables', COALESCE(tables_data, '[]'::json),
    'relationships', COALESCE(relationships_data, '[]'::json)
  ) INTO result
  FROM public.projects p
  LEFT JOIN (
    SELECT 
      tm.project_id,
      json_agg(
        json_build_object(
          'id', tm.id,
          'name', tm.name,
          'description', tm.description,
          'business_rules', tm.business_rules,
          'enhanced_description', tm.enhanced_description,
          'domain_context', tm.domain_context,
          'position_x', tm.position_x,
          'position_y', tm.position_y,
          'created_at', tm.created_at,
          'updated_at', tm.updated_at,
          'columns', COALESCE(columns_data, '[]'::json)
        )
      ) as tables_data
    FROM public.table_metadata tm
    LEFT JOIN (
      SELECT 
        cm.table_id,
        json_agg(
          json_build_object(
            'id', cm.id,
            'name', cm.name,
            'data_type', cm.data_type,
            'is_nullable', cm.is_nullable,
            'is_primary_key', cm.is_primary_key,
            'is_unique', cm.is_unique,
            'default_value', cm.default_value,
            'max_length', cm.max_length,
            'pattern', cm.pattern,
            'sample_values', cm.sample_values,
            'business_rules', cm.business_rules,
            'enhanced_description', cm.enhanced_description,
            'data_generation_rules', cm.data_generation_rules,
            'validation_rules', cm.validation_rules,
            'position', cm.position,
            'created_at', cm.created_at,
            'updated_at', cm.updated_at
          )
        ) as columns_data
      FROM public.column_metadata cm
      GROUP BY cm.table_id
    ) cols ON tm.id = cols.table_id
    GROUP BY tm.project_id
  ) tables ON p.id = tables.project_id
  LEFT JOIN (
    SELECT 
      r.source_table_id,
      json_agg(
        json_build_object(
          'id', r.id,
          'source_table_id', r.source_table_id,
          'source_column_id', r.source_column_id,
          'target_table_id', r.target_table_id,
          'target_column_id', r.target_column_id,
          'relationship_type', r.relationship_type,
          'created_at', r.created_at
        )
      ) as relationships_data
    FROM public.relationships r
    JOIN public.table_metadata tm ON r.source_table_id = tm.id
    WHERE tm.project_id = project_uuid
    GROUP BY r.source_table_id
  ) rels ON p.id = rels.source_table_id
  WHERE p.id = project_uuid;

  RETURN result;
END;
$$;

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_projects_updated_at
  BEFORE UPDATE ON public.projects
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_profiles_updated_at
  BEFORE UPDATE ON public.profiles
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_table_metadata_updated_at
  BEFORE UPDATE ON public.table_metadata
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_column_metadata_updated_at
  BEFORE UPDATE ON public.column_metadata
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_llm_conversations_updated_at
  BEFORE UPDATE ON public.llm_conversations
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_synthetic_data_generations_updated_at
  BEFORE UPDATE ON public.synthetic_data_generations
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_api_secrets_updated_at
  BEFORE UPDATE ON public.api_secrets
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

-- Create trigger to automatically create profile on signup
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user(); 