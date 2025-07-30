import { useState, useRef, useEffect } from 'react';
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Plus, Link, Trash2, ArrowRight } from 'lucide-react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';

interface Relationship {
  id: string;
  source_table_id: string;
  source_column_id: string;
  target_table_id: string;
  target_column_id: string;
  relationship_type: string;
  source_table_name: string;
  source_column_name: string;
  target_table_name: string;
  target_column_name: string;
}

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

interface RelationshipCanvasProps {
  tables: TableData[];
  projectId: string | null;
  onRelationshipCreated: () => void;
}

export function RelationshipCanvas({
  tables,
  projectId,
  onRelationshipCreated,
}: RelationshipCanvasProps) {
  const [relationships, setRelationships] = useState<Relationship[]>([]);
  const [draggedColumn, setDraggedColumn] = useState<{
    tableId: string;
    columnId: string;
    columnName: string;
    tableName: string;
  } | null>(null);
  const [showNewRelationship, setShowNewRelationship] = useState(false);
  const [newRelationship, setNewRelationship] = useState({
    source_table_id: '',
    source_column_id: '',
    target_table_id: '',
    target_column_id: '',
    relationship_type: 'one-to-many',
  });
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Fetch existing relationships
  const { data: existingRelationships } = useQuery({
    queryKey: ['relationships', projectId],
    queryFn: async () => {
      if (!projectId) return [];

      const { data, error } = await supabase
        .from('relationships')
        .select(`
          *,
          source_table:table_metadata!relationships_source_table_id_fkey(name),
          source_column:column_metadata!relationships_source_column_id_fkey(name),
          target_table:table_metadata!relationships_target_table_id_fkey(name),
          target_column:column_metadata!relationships_target_column_id_fkey(name)
        `)
        .eq('source_table_id', projectId);

      if (error) throw error;
      return data || [];
    },
    enabled: !!projectId,
  });

  useEffect(() => {
    if (existingRelationships) {
      const mappedRelationships = existingRelationships.map(rel => ({
        id: rel.id,
        source_table_id: rel.source_table_id,
        source_column_id: rel.source_column_id,
        target_table_id: rel.target_table_id,
        target_column_id: rel.target_column_id,
        relationship_type: rel.relationship_type,
        source_table_name: rel.source_table?.name || '',
        source_column_name: rel.source_column?.name || '',
        target_table_name: rel.target_table?.name || '',
        target_column_name: rel.target_column?.name || '',
      }));
      setRelationships(mappedRelationships);
    }
  }, [existingRelationships]);

  const createRelationshipMutation = useMutation({
    mutationFn: async (relationship: typeof newRelationship) => {
      const { data, error } = await supabase
        .from('relationships')
        .insert(relationship)
        .select()
        .single();

      if (error) throw error;
      return data;
    },
    onSuccess: () => {
      toast({
        title: 'Success',
        description: 'Relationship created successfully',
      });
      setShowNewRelationship(false);
      setNewRelationship({
        source_table_id: '',
        source_column_id: '',
        target_table_id: '',
        target_column_id: '',
        relationship_type: 'one-to-many',
      });
      onRelationshipCreated();
    },
    onError: (error: any) => {
      toast({
        title: 'Error',
        description: error.message || 'Failed to create relationship',
        variant: 'destructive',
      });
    },
  });

  const deleteRelationshipMutation = useMutation({
    mutationFn: async (relationshipId: string) => {
      const { error } = await supabase
        .from('relationships')
        .delete()
        .eq('id', relationshipId);

      if (error) throw error;
    },
    onSuccess: () => {
      toast({
        title: 'Success',
        description: 'Relationship deleted successfully',
      });
      onRelationshipCreated();
    },
    onError: (error: any) => {
      toast({
        title: 'Error',
        description: error.message || 'Failed to delete relationship',
        variant: 'destructive',
      });
    },
  });

  const handleDragStart = (
    tableId: string,
    columnId: string,
    columnName: string,
    tableName: string
  ) => {
    setDraggedColumn({ tableId, columnId, columnName, tableName });
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (
    targetTableId: string,
    targetColumnId: string,
    targetColumnName: string,
    targetTableName: string
  ) => {
    if (!draggedColumn) return;

    // Don't allow self-referencing relationships
    if (draggedColumn.tableId === targetTableId && draggedColumn.columnId === targetColumnId) {
      return;
    }

    setNewRelationship({
      source_table_id: draggedColumn.tableId,
      source_column_id: draggedColumn.columnId,
      target_table_id: targetTableId,
      target_column_id: targetColumnId,
      relationship_type: 'one-to-many',
    });
    setShowNewRelationship(true);
    setDraggedColumn(null);
  };

  const handleCreateRelationship = () => {
    createRelationshipMutation.mutate(newRelationship);
  };

  const getTableById = (tableId: string) => {
    return tables.find(table => table.id === tableId);
  };

  const getColumnById = (tableId: string, columnId: string) => {
    const table = getTableById(tableId);
    return table?.column_metadata.find(col => col.id === columnId);
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b">
        <div>
          <h2 className="text-lg font-semibold">Table Relationships</h2>
          <p className="text-sm text-muted-foreground">
            Drag and drop columns to create relationships
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowNewRelationship(true)}
        >
          <Plus className="h-4 w-4 mr-2" />
          Add Relationship
        </Button>
      </div>

      <div className="flex-1 overflow-auto p-4">
        {/* Tables Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          {tables.map((table) => (
            <Card key={table.id} className="relative">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Link className="h-4 w-4" />
                  {table.name}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-1">
                  {table.column_metadata?.map((column) => (
                    <div
                      key={column.id}
                      className="flex items-center justify-between p-2 border rounded text-xs hover:bg-muted/50 cursor-grab active:cursor-grabbing"
                      draggable
                      onDragStart={() =>
                        handleDragStart(table.id, column.id, column.name, table.name)
                      }
                      onDragOver={handleDragOver}
                      onDrop={() =>
                        handleDrop(table.id, column.id, column.name, table.name)
                      }
                    >
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{column.name}</span>
                        {column.is_primary_key && (
                          <Badge variant="outline" className="text-xs px-1 py-0">
                            PK
                          </Badge>
                        )}
                      </div>
                      <span className="text-muted-foreground text-xs">
                        {column.data_type}
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Existing Relationships */}
        {relationships.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-md font-semibold">Existing Relationships</h3>
            <div className="space-y-2">
              {relationships.map((relationship) => (
                <Card key={relationship.id} className="p-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">
                        {relationship.source_table_name}.{relationship.source_column_name}
                      </span>
                      <ArrowRight className="h-4 w-4 text-muted-foreground" />
                      <span className="font-medium">
                        {relationship.target_table_name}.{relationship.target_column_name}
                      </span>
                      <Badge variant="outline" className="text-xs">
                        {relationship.relationship_type}
                      </Badge>
                    </div>
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-6 w-6 p-0 text-destructive hover:text-destructive"
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>Delete Relationship</AlertDialogTitle>
                          <AlertDialogDescription>
                            Are you sure you want to delete this relationship? This action cannot be undone.
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>Cancel</AlertDialogCancel>
                          <AlertDialogAction
                            onClick={() => deleteRelationshipMutation.mutate(relationship.id)}
                            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                          >
                            Delete
                          </AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* New Relationship Dialog */}
      {showNewRelationship && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-background p-6 rounded-lg shadow-lg max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-4">Create Relationship</h3>
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium">Source</label>
                <p className="text-sm text-muted-foreground">
                  {getTableById(newRelationship.source_table_id)?.name}.
                  {getColumnById(newRelationship.source_table_id, newRelationship.source_column_id)?.name}
                </p>
              </div>
              <div>
                <label className="text-sm font-medium">Target</label>
                <p className="text-sm text-muted-foreground">
                  {getTableById(newRelationship.target_table_id)?.name}.
                  {getColumnById(newRelationship.target_table_id, newRelationship.target_column_id)?.name}
                </p>
              </div>
              <div>
                <label className="text-sm font-medium">Relationship Type</label>
                <select
                  value={newRelationship.relationship_type}
                  onChange={(e) =>
                    setNewRelationship({
                      ...newRelationship,
                      relationship_type: e.target.value,
                    })
                  }
                  className="w-full mt-1 p-2 border rounded"
                >
                  <option value="one-to-one">One-to-One</option>
                  <option value="one-to-many">One-to-Many</option>
                  <option value="many-to-many">Many-to-Many</option>
                </select>
              </div>
            </div>
            <div className="flex gap-2 mt-6">
              <Button
                variant="outline"
                onClick={() => setShowNewRelationship(false)}
                className="flex-1"
              >
                Cancel
              </Button>
              <Button
                onClick={handleCreateRelationship}
                disabled={createRelationshipMutation.isPending}
                className="flex-1"
              >
                {createRelationshipMutation.isPending ? 'Creating...' : 'Create'}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 