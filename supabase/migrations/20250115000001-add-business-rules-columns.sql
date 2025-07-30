-- Add business rules and enhanced description columns to column_metadata table
ALTER TABLE public.column_metadata 
ADD COLUMN IF NOT EXISTS business_rules TEXT,
ADD COLUMN IF NOT EXISTS enhanced_description TEXT,
ADD COLUMN IF NOT EXISTS data_generation_rules TEXT,
ADD COLUMN IF NOT EXISTS validation_rules TEXT;

-- Add comments to the new columns
COMMENT ON COLUMN public.column_metadata.business_rules IS 'AI-generated business rules for the column';
COMMENT ON COLUMN public.column_metadata.enhanced_description IS 'AI-enhanced description of the column';
COMMENT ON COLUMN public.column_metadata.data_generation_rules IS 'Rules for generating synthetic data for this column';
COMMENT ON COLUMN public.column_metadata.validation_rules IS 'Validation rules for the column data'; 