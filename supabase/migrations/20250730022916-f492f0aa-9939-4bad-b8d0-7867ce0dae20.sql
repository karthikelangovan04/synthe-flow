-- Make user_id NOT NULL in projects table since every project must belong to a user
-- First update any existing rows with null user_id (there shouldn't be any but just in case)
UPDATE projects SET user_id = (SELECT id FROM auth.users LIMIT 1) WHERE user_id IS NULL;

-- Now make the column NOT NULL
ALTER TABLE projects ALTER COLUMN user_id SET NOT NULL;