import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { supabase } from '@/integrations/supabase/client';
import { ProjectList } from '@/components/schema-designer/ProjectList';
import { SchemaCanvas } from '@/components/schema-designer/SchemaCanvas';
import { TableEditor } from '@/components/schema-designer/TableEditor';
import { SchemaImport } from '@/components/schema-designer/SchemaImport';
import { Button } from '@/components/ui/button';
import { Plus, Upload } from 'lucide-react';
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';

export default function SchemaDesigner() {
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [selectedTableId, setSelectedTableId] = useState<string | null>(null);
  const [showNewProject, setShowNewProject] = useState(false);
  const [showImportDialog, setShowImportDialog] = useState(false);

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
      
      const { data, error } = await supabase
        .from('table_metadata')
        .select(`
          *,
          column_metadata(*)
        `)
        .eq('project_id', selectedProjectId)
        .order('name');
      
      if (error) throw error;
      return data;
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

  const handleSchemaImported = (tables: any[]) => {
    // Handle the imported schema - this would create tables in the database
    console.log('Imported schema:', tables);
    setShowImportDialog(false);
    refetchTables();
  };

  return (
    <div className="h-screen flex flex-col">
      <header className="border-b bg-background px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Schema Designer</h1>
            <p className="text-muted-foreground">Design and manage your database schemas</p>
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
          <SchemaCanvas
            tables={tables || []}
            selectedTableId={selectedTableId}
            onTableSelect={handleTableSelect}
            onTableCreated={handleTableCreated}
            projectId={selectedProjectId}
          />
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