-- Create projects table to organize schema designs
CREATE TABLE public.projects (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE
);

-- Create tables metadata table
CREATE TABLE public.table_metadata (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id UUID REFERENCES public.projects(id) ON DELETE CASCADE NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
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

-- Enable Row Level Security
ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.table_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.column_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.relationships ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY "Users can view their own projects" ON public.projects
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own projects" ON public.projects
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own projects" ON public.projects
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own projects" ON public.projects
  FOR DELETE USING (auth.uid() = user_id);

-- Table metadata policies
CREATE POLICY "Users can view table metadata in their projects" ON public.table_metadata
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM public.projects 
      WHERE projects.id = table_metadata.project_id 
      AND projects.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can create table metadata in their projects" ON public.table_metadata
  FOR INSERT WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.projects 
      WHERE projects.id = table_metadata.project_id 
      AND projects.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can update table metadata in their projects" ON public.table_metadata
  FOR UPDATE USING (
    EXISTS (
      SELECT 1 FROM public.projects 
      WHERE projects.id = table_metadata.project_id 
      AND projects.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can delete table metadata in their projects" ON public.table_metadata
  FOR DELETE USING (
    EXISTS (
      SELECT 1 FROM public.projects 
      WHERE projects.id = table_metadata.project_id 
      AND projects.user_id = auth.uid()
    )
  );

-- Column metadata policies
CREATE POLICY "Users can view column metadata in their projects" ON public.column_metadata
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM public.table_metadata tm
      JOIN public.projects p ON p.id = tm.project_id
      WHERE tm.id = column_metadata.table_id 
      AND p.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can create column metadata in their projects" ON public.column_metadata
  FOR INSERT WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.table_metadata tm
      JOIN public.projects p ON p.id = tm.project_id
      WHERE tm.id = column_metadata.table_id 
      AND p.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can update column metadata in their projects" ON public.column_metadata
  FOR UPDATE USING (
    EXISTS (
      SELECT 1 FROM public.table_metadata tm
      JOIN public.projects p ON p.id = tm.project_id
      WHERE tm.id = column_metadata.table_id 
      AND p.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can delete column metadata in their projects" ON public.column_metadata
  FOR DELETE USING (
    EXISTS (
      SELECT 1 FROM public.table_metadata tm
      JOIN public.projects p ON p.id = tm.project_id
      WHERE tm.id = column_metadata.table_id 
      AND p.user_id = auth.uid()
    )
  );

-- Relationships policies
CREATE POLICY "Users can view relationships in their projects" ON public.relationships
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM public.table_metadata tm
      JOIN public.projects p ON p.id = tm.project_id
      WHERE tm.id = relationships.source_table_id 
      AND p.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can create relationships in their projects" ON public.relationships
  FOR INSERT WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.table_metadata tm
      JOIN public.projects p ON p.id = tm.project_id
      WHERE tm.id = relationships.source_table_id 
      AND p.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can update relationships in their projects" ON public.relationships
  FOR UPDATE USING (
    EXISTS (
      SELECT 1 FROM public.table_metadata tm
      JOIN public.projects p ON p.id = tm.project_id
      WHERE tm.id = relationships.source_table_id 
      AND p.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can delete relationships in their projects" ON public.relationships
  FOR DELETE USING (
    EXISTS (
      SELECT 1 FROM public.table_metadata tm
      JOIN public.projects p ON p.id = tm.project_id
      WHERE tm.id = relationships.source_table_id 
      AND p.user_id = auth.uid()
    )
  );

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_projects_updated_at
  BEFORE UPDATE ON public.projects
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