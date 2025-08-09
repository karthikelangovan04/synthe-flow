-- Security Policies Migration
-- This migration enables Row Level Security and creates policies for all tables

-- Enable Row Level Security on all tables
ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.table_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.column_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.api_secrets ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.llm_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.synthetic_data_generations ENABLE ROW LEVEL SECURITY;

-- Projects policies
CREATE POLICY "Users can view their own projects" ON public.projects
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own projects" ON public.projects
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own projects" ON public.projects
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own projects" ON public.projects
  FOR DELETE USING (auth.uid() = user_id);

-- Profiles policies
CREATE POLICY "Profiles are viewable by everyone" 
ON public.profiles 
FOR SELECT 
USING (true);

CREATE POLICY "Users can update their own profile" 
ON public.profiles 
FOR UPDATE 
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own profile" 
ON public.profiles 
FOR INSERT 
WITH CHECK (auth.uid() = user_id);

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

-- API secrets policies (restrict to service role only)
CREATE POLICY "Service role can manage API secrets" ON public.api_secrets
  FOR ALL USING (auth.role() = 'service_role');

-- LLM conversations policies
CREATE POLICY "Users can view conversations in their projects" ON public.llm_conversations
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM public.projects 
      WHERE projects.id = llm_conversations.project_id 
      AND projects.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can create conversations in their projects" ON public.llm_conversations
  FOR INSERT WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.projects 
      WHERE projects.id = llm_conversations.project_id 
      AND projects.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can update conversations in their projects" ON public.llm_conversations
  FOR UPDATE USING (
    EXISTS (
      SELECT 1 FROM public.projects 
      WHERE projects.id = llm_conversations.project_id 
      AND projects.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can delete conversations in their projects" ON public.llm_conversations
  FOR DELETE USING (
    EXISTS (
      SELECT 1 FROM public.projects 
      WHERE projects.id = llm_conversations.project_id 
      AND projects.user_id = auth.uid()
    )
  );

-- Synthetic data generations policies
CREATE POLICY "Users can view generations in their projects" ON public.synthetic_data_generations
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM public.projects 
      WHERE projects.id = synthetic_data_generations.project_id 
      AND projects.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can create generations in their projects" ON public.synthetic_data_generations
  FOR INSERT WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.projects 
      WHERE projects.id = synthetic_data_generations.project_id 
      AND projects.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can update generations in their projects" ON public.synthetic_data_generations
  FOR UPDATE USING (
    EXISTS (
      SELECT 1 FROM public.projects 
      WHERE projects.id = synthetic_data_generations.project_id 
      AND projects.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can delete generations in their projects" ON public.synthetic_data_generations
  FOR DELETE USING (
    EXISTS (
      SELECT 1 FROM public.projects 
      WHERE projects.id = synthetic_data_generations.project_id 
      AND projects.user_id = auth.uid()
    )
  ); 