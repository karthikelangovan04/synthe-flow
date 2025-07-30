import { useState, useEffect } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Plus, Link, Trash2, ArrowRight, Database } from 'lucide-react';

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
  const { toast } = useToast();

  const { data: existingRelationships } = useQuery({
    queryKey: ['relationships', projectId],
    queryFn: async () => {
      if (!projectId) return [];
      const { data: projectTables } = await supabase
        .from('table_metadata')
        .select('id')
        .eq('project_id', projectId);
      
      if (!projectTables || projectTables.length === 0) return [];
      
      const tableIds = projectTables.map(t => t.id);
      const { data } = await supabase
        .from('relationships')
        .select(`
          *,
          source_table:table_metadata!relationships_source_table_id_fkey(name),
          source_column:column_metadata!relationships_source_column_id_fkey(name),
          target_table:table_metadata!relationships_target_table_id_fkey(name),
          target_column:column_metadata!relationships_target_column_id_fkey(name)
        `)
        .in('source_table_id', tableIds)
        .in('target_table_id', tableIds);

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

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b bg-background">
        <div>
          <h2 className="text-lg font-semibold">Table Relationships</h2>
          <p className="text-sm text-muted-foreground">
            View and manage table relationships
          </p>
        </div>
        <Button variant="outline" size="sm">
          <Plus className="h-4 w-4 mr-2" />
          Add Relationship
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-6">
          <div>
            <h3 className="text-md font-semibold mb-3">
              Tables & Columns ({tables.length} tables)
            </h3>
            {tables.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {tables.map((table) => (
                  <Card key={table.id} className="relative">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm flex items-center gap-2">
                        <Link className="h-4 w-4" />
                        {table.name}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-1 max-h-48 overflow-y-auto">
                        {table.column_metadata && table.column_metadata.length > 0 ? (
                          table.column_metadata.map((column) => (
                            <div
                              key={column.id}
                              className="flex items-center justify-between p-2 border rounded text-xs"
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
                          ))
                        ) : (
                          <div className="text-center py-4 text-muted-foreground text-xs">
                            No columns found
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Database className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>No tables found</p>
                <p className="text-sm">Create tables first to manage relationships</p>
              </div>
            )}
          </div>

          <div>
            <h3 className="text-md font-semibold mb-3">
              Existing Relationships ({relationships.length})
            </h3>
            {relationships.length > 0 ? (
              <div className="space-y-2 max-h-64 overflow-y-auto border rounded-lg p-3">
                {relationships.map((relationship) => (
                  <Card key={relationship.id} className="p-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-medium text-sm">
                          {relationship.source_table_name}.{relationship.source_column_name}
                        </span>
                        <ArrowRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                        <span className="font-medium text-sm">
                          {relationship.target_table_name}.{relationship.target_column_name}
                        </span>
                        <Badge variant="outline" className="text-xs flex-shrink-0">
                          {relationship.relationship_type}
                        </Badge>
                      </div>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 w-6 p-0 text-destructive hover:text-destructive flex-shrink-0"
                        onClick={() => deleteRelationshipMutation.mutate(relationship.id)}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Link className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>No relationships created yet</p>
                <p className="text-sm">Relationships will appear here when created</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
} 