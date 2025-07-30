import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { supabase } from '@/integrations/supabase/client';
import { ProjectList } from '@/components/schema-designer/ProjectList';
import { SchemaCanvas } from '@/components/schema-designer/SchemaCanvas';
import { TableEditor } from '@/components/schema-designer/TableEditor';
import { SchemaImport } from '@/components/schema-designer/SchemaImport';
import { RelationshipCanvas } from '@/components/schema-designer/RelationshipCanvas';
import { SyntheticDataPanel } from '@/components/schema-designer/SyntheticDataPanel';
import { Button } from '@/components/ui/button';
import { Plus, Upload, Home, Link, Database } from 'lucide-react';
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useToast } from '@/hooks/use-toast';

export default function SchemaDesigner() {
  const navigate = useNavigate();
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [selectedTableId, setSelectedTableId] = useState<string | null>(null);
  const [showNewProject, setShowNewProject] = useState(false);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const { toast } = useToast();

  const { data: projects, refetch: refetchProjects } = useQuery({
    queryKey: ['projects'],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('projects')
        .select('*')
        .order('updated_at', { ascending: false });
      
      if (error) throw error;
      return data;
    },
  });

  const { data: tables, refetch: refetchTables } = useQuery({
    queryKey: ['tables', selectedProjectId],
    queryFn: async () => {
      if (!selectedProjectId) return [];
      
      console.log('Fetching tables for project:', selectedProjectId);
      
      const { data, error } = await supabase
        .from('table_metadata')
        .select(`
          *,
          column_metadata(*)
        `)
        .eq('project_id', selectedProjectId)
        .order('name');
      
      if (error) {
        console.error('Error fetching tables:', error);
        throw error;
      }
      
      console.log('Fetched tables:', data);
      return data;
    },
    enabled: !!selectedProjectId,
  });

  const { data: relationships, refetch: refetchRelationships } = useQuery({
    queryKey: ['relationships', selectedProjectId],
    queryFn: async () => {
      if (!selectedProjectId) return [];
      
      const { data, error } = await supabase
        .from('relationships')
        .select(`
          *,
          source_table:table_metadata!relationships_source_table_id_fkey(name),
          source_column:column_metadata!relationships_source_column_id_fkey(name),
          target_table:table_metadata!relationships_target_table_id_fkey(name),
          target_column:column_metadata!relationships_target_column_id_fkey(name)
        `)
        .eq('source_table_id', selectedProjectId);
      
      if (error) {
        console.error('Error fetching relationships:', error);
        throw error;
      }
      
      return data || [];
    },
    enabled: !!selectedProjectId,
  });

  const handleProjectSelect = (projectId: string) => {
    setSelectedProjectId(projectId);
    setSelectedTableId(null);
  };

  const handleTableSelect = (tableId: string) => {
    setSelectedTableId(tableId);
  };

  const handleProjectCreated = () => {
    refetchProjects();
    setShowNewProject(false);
  };

  const handleTableCreated = () => {
    refetchTables();
  };

  const handleSchemaImported = async (tables: any[]) => {
    if (!selectedProjectId) {
      toast({ title: 'Error', description: 'No project selected', variant: 'destructive' });
      return;
    }
    
    try {
      let importedCount = 0;
      let skippedCount = 0;
      
      for (const table of tables) {
        // Check if table already exists
        const { data: existingTable } = await supabase
          .from('table_metadata')
          .select('id, name')
          .eq('project_id', selectedProjectId)
          .eq('name', table.name)
          .single();
        
        let tableId: string;
        
        if (existingTable) {
          // Table already exists, use existing table ID
          tableId = existingTable.id;
          skippedCount++;
          console.log(`Table "${table.name}" already exists, skipping table creation`);
        } else {
          // Insert new table
          const { data: tableData, error: tableError } = await supabase
            .from('table_metadata')
            .insert({
              name: table.name,
              description: table.description || '',
              project_id: selectedProjectId,
            })
            .select('id')
            .single();
          
          if (tableError) {
            console.error('Error inserting table:', tableError);
            throw new Error(`Failed to create table "${table.name}": ${tableError.message}`);
          }
          
          tableId = tableData.id;
          importedCount++;
        }
        
        // Insert columns (check for existing columns first)
        for (const [idx, column] of (table.columns || []).entries()) {
          // Check if column already exists
          const { data: existingColumn } = await supabase
            .from('column_metadata')
            .select('id')
            .eq('table_id', tableId)
            .eq('name', column.name)
            .single();
          
          if (!existingColumn) {
            // Insert new column
            const { error: colError } = await supabase
              .from('column_metadata')
              .insert({
                name: column.name,
                data_type: column.dataType,
                is_nullable: column.isNullable ?? true,
                is_primary_key: column.isPrimaryKey ?? false,
                is_unique: column.isUnique ?? false,
                table_id: tableId,
                position: idx,
                pattern: null,
                max_length: null,
                default_value: null,
                sample_values: null,
              });
            
            if (colError) {
              console.error('Error inserting column:', colError);
              throw new Error(`Failed to create column "${column.name}" in table "${table.name}": ${colError.message}`);
            }
          }
        }
      }
      
      let message = `Successfully imported ${importedCount} new table(s)`;
      if (skippedCount > 0) {
        message += `, skipped ${skippedCount} existing table(s)`;
      }
      
      toast({ title: 'Success', description: message });
      setShowImportDialog(false);
      
      // Wait a moment then refetch to ensure data is available
      setTimeout(() => {
        console.log('Refetching tables after import...');
        refetchTables();
      }, 500);
    } catch (error: any) {
      console.error('Import error:', error);
      toast({ 
        title: 'Error', 
        description: error.message || 'Failed to import schema', 
        variant: 'destructive' 
      });
    }
  };

  return (
    <div className="h-screen flex flex-col">
      <header className="border-b bg-background px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => navigate('/')}
              className="flex items-center gap-2"
            >
              <Home className="h-4 w-4" />
              Home
            </Button>
            <div>
              <h1 className="text-2xl font-bold">Schema Designer</h1>
              <p className="text-muted-foreground">Design and manage your database schemas</p>
            </div>
          </div>
          <div className="flex gap-2">
            <Dialog open={showImportDialog} onOpenChange={setShowImportDialog}>
              <DialogTrigger asChild>
                <Button variant="outline">
                  <Upload className="h-4 w-4 mr-2" />
                  Import Schema
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
                <DialogHeader>
                  <DialogTitle>Import Schema</DialogTitle>
                </DialogHeader>
                <SchemaImport
                  projectId={selectedProjectId || ''}
                  onSchemaImported={handleSchemaImported}
                />
              </DialogContent>
            </Dialog>
            <Button onClick={() => setShowNewProject(true)}>
              <Plus className="h-4 w-4 mr-2" />
              New Project
            </Button>
          </div>
        </div>
      </header>

      <ResizablePanelGroup direction="horizontal" className="flex-1">
        <ResizablePanel defaultSize={25} minSize={20} maxSize={40}>
          <ProjectList
            projects={projects || []}
            selectedProjectId={selectedProjectId}
            onProjectSelect={handleProjectSelect}
            showNewProject={showNewProject}
            onProjectCreated={handleProjectCreated}
            onCancelNewProject={() => setShowNewProject(false)}
          />
        </ResizablePanel>

        <ResizableHandle withHandle />

        <ResizablePanel defaultSize={50} minSize={40}>
          <Tabs defaultValue="schema" className="h-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="schema">Schema</TabsTrigger>
              <TabsTrigger value="relationships">Relationships</TabsTrigger>
              <TabsTrigger value="synthetic">Synthetic Data</TabsTrigger>
            </TabsList>
            
            <TabsContent value="schema" className="h-full">
              <SchemaCanvas
                tables={tables || []}
                selectedTableId={selectedTableId}
                onTableSelect={handleTableSelect}
                onTableCreated={handleTableCreated}
                projectId={selectedProjectId}
              />
            </TabsContent>
            
            <TabsContent value="relationships" className="h-full">
              <RelationshipCanvas
                tables={tables || []}
                projectId={selectedProjectId}
                onRelationshipCreated={refetchRelationships}
              />
            </TabsContent>
            
            <TabsContent value="synthetic" className="h-full">
              <SyntheticDataPanel
                tables={tables || []}
                relationships={relationships || []}
                projectId={selectedProjectId}
              />
            </TabsContent>
          </Tabs>
        </ResizablePanel>

        <ResizableHandle withHandle />

        <ResizablePanel defaultSize={25} minSize={20} maxSize={40}>
          <TableEditor
            tableId={selectedTableId}
            onTableUpdated={handleTableCreated}
          />
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}