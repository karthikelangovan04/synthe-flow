import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Plus, Table, Database, Key, Link2 } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';

interface TableData {
  id: string;
  name: string;
  description: string | null;
  position_x: number;
  position_y: number;
  column_metadata: Array<{
    id: string;
    name: string;
    data_type: string;
    is_nullable: boolean;
    is_primary_key: boolean;
    is_unique: boolean;
  }>;
}

interface SchemaCanvasProps {
  tables: TableData[];
  selectedTableId: string | null;
  onTableSelect: (tableId: string) => void;
  onTableCreated: () => void;
  projectId: string | null;
}

export function SchemaCanvas({
  tables,
  selectedTableId,
  onTableSelect,
  onTableCreated,
  projectId,
}: SchemaCanvasProps) {
  const [newTableName, setNewTableName] = useState('');
  const [isCreatingTable, setIsCreatingTable] = useState(false);
  const [showNewTableDialog, setShowNewTableDialog] = useState(false);
  const { toast } = useToast();

  const handleCreateTable = async () => {
    if (!projectId || !newTableName.trim()) {
      toast({
        title: "Error",
        description: "Project and table name are required",
        variant: "destructive",
      });
      return;
    }

    setIsCreatingTable(true);
    try {
      const { error } = await supabase
        .from('table_metadata')
        .insert({
          project_id: projectId,
          name: newTableName.trim(),
          position_x: Math.floor(Math.random() * 300) + 50,
          position_y: Math.floor(Math.random() * 200) + 50,
        });

      if (error) throw error;

      toast({
        title: "Success",
        description: "Table created successfully",
      });

      setNewTableName('');
      setShowNewTableDialog(false);
      onTableCreated();
    } catch (error) {
      console.error('Error creating table:', error);
      toast({
        title: "Error",
        description: "Failed to create table",
        variant: "destructive",
      });
    } finally {
      setIsCreatingTable(false);
    }
  };

  const getDataTypeIcon = (dataType: string) => {
    if (dataType.includes('text') || dataType.includes('varchar')) return '📝';
    if (dataType.includes('int') || dataType.includes('numeric')) return '🔢';
    if (dataType.includes('bool')) return '✓';
    if (dataType.includes('date') || dataType.includes('time')) return '📅';
    if (dataType.includes('uuid')) return '🔑';
    return '📄';
  };

  return (
    <div className="h-full flex flex-col">
      <div className="border-b p-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Database className="h-5 w-5" />
          <h2 className="text-lg font-semibold">Schema Canvas</h2>
          <Badge variant="secondary">{tables.length} tables</Badge>
        </div>
        
        {projectId && (
          <Dialog open={showNewTableDialog} onOpenChange={setShowNewTableDialog}>
            <DialogTrigger asChild>
              <Button size="sm">
                <Plus className="h-4 w-4 mr-2" />
                Add Table
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create New Table</DialogTitle>
              </DialogHeader>
              <div className="space-y-4">
                <Input
                  placeholder="Table name"
                  value={newTableName}
                  onChange={(e) => setNewTableName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleCreateTable()}
                />
                <div className="flex gap-2">
                  <Button
                    onClick={handleCreateTable}
                    disabled={isCreatingTable}
                    className="flex-1"
                  >
                    {isCreatingTable ? 'Creating...' : 'Create Table'}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setShowNewTableDialog(false)}
                    disabled={isCreatingTable}
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>
        )}
      </div>

      <div className="flex-1 overflow-auto p-4">
        {!projectId ? (
          <div className="h-full flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <Database className="h-16 w-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg">Select a project to start designing</p>
              <p className="text-sm">Choose a project from the left panel to view its schema</p>
            </div>
          </div>
        ) : tables.length === 0 ? (
          <div className="h-full flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <Table className="h-16 w-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg">No tables in this project</p>
              <p className="text-sm mb-4">Create your first table to get started</p>
              <Button onClick={() => setShowNewTableDialog(true)}>
                <Plus className="h-4 w-4 mr-2" />
                Create Table
              </Button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {tables.map((table) => (
              <Card
                key={table.id}
                className={`cursor-pointer transition-all hover:shadow-md ${
                  selectedTableId === table.id ? 'ring-2 ring-primary' : ''
                }`}
                onClick={() => onTableSelect(table.id)}
              >
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Table className="h-4 w-4" />
                    {table.name}
                  </CardTitle>
                  {table.description && (
                    <p className="text-xs text-muted-foreground">{table.description}</p>
                  )}
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {table.column_metadata?.slice(0, 5).map((column) => (
                      <div
                        key={column.id}
                        className="flex items-center justify-between text-xs"
                      >
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                          <span className="text-sm">{getDataTypeIcon(column.data_type)}</span>
                          <span className="truncate font-medium">{column.name}</span>
                          {column.is_primary_key && (
                            <Key className="h-3 w-3 text-yellow-500" />
                          )}
                          {column.is_unique && (
                            <Badge variant="outline" className="text-xs px-1 py-0">
                              U
                            </Badge>
                          )}
                        </div>
                        <span className="text-muted-foreground text-xs">
                          {column.data_type}
                        </span>
                      </div>
                    ))}
                    {table.column_metadata && table.column_metadata.length > 5 && (
                      <p className="text-xs text-muted-foreground text-center pt-1">
                        +{table.column_metadata.length - 5} more columns
                      </p>
                    )}
                    {(!table.column_metadata || table.column_metadata.length === 0) && (
                      <p className="text-xs text-muted-foreground text-center py-2">
                        No columns defined
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}