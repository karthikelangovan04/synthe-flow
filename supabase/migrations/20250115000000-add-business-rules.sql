-- Add business rules and enhanced description fields to column_metadata
ALTER TABLE public.column_metadata 
ADD COLUMN business_rules TEXT,
ADD COLUMN enhanced_description TEXT,
ADD COLUMN data_generation_rules TEXT,
ADD COLUMN validation_rules TEXT;

-- Add business rules and enhanced description to table_metadata
ALTER TABLE public.table_metadata 
ADD COLUMN business_rules TEXT,
ADD COLUMN enhanced_description TEXT,
ADD COLUMN domain_context TEXT;

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

-- Enable RLS for llm_conversations
ALTER TABLE public.llm_conversations ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for llm_conversations
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

-- Create trigger for automatic timestamp updates
CREATE TRIGGER update_llm_conversations_updated_at
  BEFORE UPDATE ON public.llm_conversations
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column(); 