import { useState, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Label } from '@/components/ui/label';
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
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [selectedSourceTable, setSelectedSourceTable] = useState<string>('');
  const [selectedSourceColumn, setSelectedSourceColumn] = useState<string>('');
  const [selectedTargetTable, setSelectedTargetTable] = useState<string>('');
  const [selectedTargetColumn, setSelectedTargetColumn] = useState<string>('');
  const [selectedRelationshipType, setSelectedRelationshipType] = useState<string>('one-to-many');
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Clear the cache when component mounts
  useEffect(() => {
    if (projectId) {
      queryClient.removeQueries({ queryKey: ['relationships-enriched', projectId] });
    }
  }, [projectId, queryClient]);

  // Reset form when dialog opens
  useEffect(() => {
    if (showAddDialog) {
      resetForm();
    }
  }, [showAddDialog]);

  const { data: existingRelationships } = useQuery({
    queryKey: ['relationships-enriched', projectId],
    queryFn: async () => {
      if (!projectId) return [];
      const { data: projectTables } = await supabase
        .from('table_metadata')
        .select('id')
        .eq('project_id', projectId);
      
      if (!projectTables || projectTables.length === 0) return [];
      
      const tableIds = projectTables.map(t => t.id);
      
      // First get all relationships for this project
      const { data: relationshipsData, error } = await supabase
        .from('relationships')
        .select('*')
        .in('source_table_id', tableIds)
        .in('target_table_id', tableIds);

      if (error) {
        console.error('Error fetching relationships:', error);
        return [];
      }

      if (!relationshipsData || relationshipsData.length === 0) return [];

      // Then get the table and column names for each relationship
      const enrichedRelationships = await Promise.all(
        relationshipsData.map(async (rel) => {
          // Get source table name
          const { data: sourceTable } = await supabase
            .from('table_metadata')
            .select('name')
            .eq('id', rel.source_table_id)
            .single();

          // Get source column name
          const { data: sourceColumn } = await supabase
            .from('column_metadata')
            .select('name')
            .eq('id', rel.source_column_id)
            .single();

          // Get target table name
          const { data: targetTable } = await supabase
            .from('table_metadata')
            .select('name')
            .eq('id', rel.target_table_id)
            .single();

          // Get target column name
          const { data: targetColumn } = await supabase
            .from('column_metadata')
            .select('name')
            .eq('id', rel.target_column_id)
            .single();

          return {
            ...rel,
            source_table: { name: sourceTable?.name || '' },
            source_column: { name: sourceColumn?.name || '' },
            target_table: { name: targetTable?.name || '' },
            target_column: { name: targetColumn?.name || '' },
          };
        })
      );

      return enrichedRelationships;
    },
    enabled: !!projectId,
    staleTime: 0, // Always consider data stale
    gcTime: 0, // Don't cache the data
  });

  // Transform the query data to match our interface
  const relationships = existingRelationships?.map(rel => ({
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
  })) || [];



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
      queryClient.invalidateQueries({ queryKey: ['relationships-enriched', projectId] });
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

  const createRelationshipMutation = useMutation({
    mutationFn: async () => {
      if (!selectedSourceTable || !selectedSourceColumn || !selectedTargetTable || !selectedTargetColumn) {
        throw new Error('Please select all required fields');
      }

      // Validation: Source and target tables should not be the same
      if (selectedSourceTable === selectedTargetTable) {
        throw new Error('Source and target tables cannot be the same');
      }

      // Validation: Check for existing relationships to prevent duplicates
      const { data: existingRelationships, error: checkError } = await supabase
        .from('relationships')
        .select('*')
        .or(`and(source_table_id.eq.${selectedSourceTable},source_column_id.eq.${selectedSourceColumn},target_table_id.eq.${selectedTargetTable},target_column_id.eq.${selectedTargetColumn}),and(source_table_id.eq.${selectedTargetTable},source_column_id.eq.${selectedTargetColumn},target_table_id.eq.${selectedSourceTable},target_column_id.eq.${selectedSourceColumn})`);

      if (checkError) {
        console.error('Error checking existing relationships:', checkError);
        throw new Error('Failed to validate relationship');
      }

      if (existingRelationships && existingRelationships.length > 0) {
        throw new Error('This relationship already exists');
      }

      const { error } = await supabase
        .from('relationships')
        .insert({
          source_table_id: selectedSourceTable,
          source_column_id: selectedSourceColumn,
          target_table_id: selectedTargetTable,
          target_column_id: selectedTargetColumn,
          relationship_type: selectedRelationshipType,
        });

      if (error) throw error;
    },
    onSuccess: () => {
      toast({
        title: 'Success',
        description: 'Relationship created successfully',
      });
      setShowAddDialog(false);
      resetForm();
      queryClient.invalidateQueries({ queryKey: ['relationships-enriched', projectId] });
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

  const resetForm = () => {
    setSelectedSourceTable('');
    setSelectedSourceColumn('');
    setSelectedTargetTable('');
    setSelectedTargetColumn('');
    setSelectedRelationshipType('one-to-many');
  };

  // Reset form when dialog is opened or closed
  const handleDialogChange = (open: boolean) => {
    setShowAddDialog(open);
    if (open) {
      // Reset form when opening to ensure fresh state
      resetForm();
    }
  };

  const handleCreateRelationship = () => {
    // Client-side validation
    if (!selectedSourceTable || !selectedSourceColumn || !selectedTargetTable || !selectedTargetColumn) {
      toast({
        title: "Error",
        description: "Please select all required fields",
        variant: "destructive",
      });
      return;
    }

    // Validation: Source and target tables should not be the same
    if (selectedSourceTable === selectedTargetTable) {
      toast({
        title: "Error",
        description: "Source and target tables cannot be the same",
        variant: "destructive",
      });
      return;
    }

    // Validation: Source and target columns should not be the same
    if (selectedSourceTable === selectedTargetTable && selectedSourceColumn === selectedTargetColumn) {
      toast({
        title: "Error",
        description: "Cannot create relationship between the same column",
        variant: "destructive",
      });
      return;
    }

    createRelationshipMutation.mutate();
  };

  const getSourceColumns = () => {
    const table = tables.find(t => t.id === selectedSourceTable);
    return table?.column_metadata || [];
  };

  const getTargetColumns = () => {
    const table = tables.find(t => t.id === selectedTargetTable);
    return table?.column_metadata || [];
  };

  // Helper functions for validation
  const isFormValid = () => {
    return selectedSourceTable && selectedSourceColumn && selectedTargetTable && selectedTargetColumn;
  };

  const hasSameTableError = () => {
    return selectedSourceTable && selectedTargetTable && selectedSourceTable === selectedTargetTable;
  };

  const hasSameColumnError = () => {
    return selectedSourceTable && selectedTargetTable && selectedSourceColumn && selectedTargetColumn &&
           selectedSourceTable === selectedTargetTable && selectedSourceColumn === selectedTargetColumn;
  };

  const getSourceTableName = () => {
    const table = tables.find(t => t.id === selectedSourceTable);
    return table?.name || '';
  };

  const getTargetTableName = () => {
    const table = tables.find(t => t.id === selectedTargetTable);
    return table?.name || '';
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b bg-background">
        <div>
          <h2 className="text-lg font-semibold">Table Relationships</h2>
          <p className="text-sm text-muted-foreground">
            View and manage table relationships
          </p>
        </div>
        <Dialog open={showAddDialog} onOpenChange={handleDialogChange}>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm">
              <Plus className="h-4 w-4 mr-2" />
              Add Relationship
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Add New Relationship</DialogTitle>
              <DialogDescription>
                Create a relationship between two tables by selecting the source and target columns.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="source-table" className="text-right">
                  Source Table
                </Label>
                <Select value={selectedSourceTable} onValueChange={setSelectedSourceTable}>
                  <SelectTrigger className={`col-span-3 ${hasSameTableError() ? 'border-destructive' : ''}`}>
                    <SelectValue placeholder="Select source table" />
                  </SelectTrigger>
                  <SelectContent>
                    {tables.map((table) => (
                      <SelectItem key={table.id} value={table.id}>
                        {table.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {hasSameTableError() && (
                <div className="text-sm text-destructive text-center">
                  Source and target tables cannot be the same
                </div>
              )}
              
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="source-column" className="text-right">
                  Source Column
                </Label>
                <Select 
                  value={selectedSourceColumn} 
                  onValueChange={setSelectedSourceColumn}
                  disabled={!selectedSourceTable}
                >
                  <SelectTrigger className="col-span-3">
                    <SelectValue placeholder="Select source column" />
                  </SelectTrigger>
                  <SelectContent>
                    {getSourceColumns().map((column) => (
                      <SelectItem key={column.id} value={column.id}>
                        {column.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="target-table" className="text-right">
                  Target Table
                </Label>
                <Select value={selectedTargetTable} onValueChange={setSelectedTargetTable}>
                  <SelectTrigger className={`col-span-3 ${hasSameTableError() ? 'border-destructive' : ''}`}>
                    <SelectValue placeholder="Select target table" />
                  </SelectTrigger>
                  <SelectContent>
                    {tables.map((table) => (
                      <SelectItem key={table.id} value={table.id}>
                        {table.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="target-column" className="text-right">
                  Target Column
                </Label>
                <Select 
                  value={selectedTargetColumn} 
                  onValueChange={setSelectedTargetColumn}
                  disabled={!selectedTargetTable}
                >
                  <SelectTrigger className="col-span-3">
                    <SelectValue placeholder="Select target column" />
                  </SelectTrigger>
                  <SelectContent>
                    {getTargetColumns().map((column) => (
                      <SelectItem key={column.id} value={column.id}>
                        {column.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="relationship-type" className="text-right">
                  Type
                </Label>
                <Select value={selectedRelationshipType} onValueChange={setSelectedRelationshipType}>
                  <SelectTrigger className="col-span-3">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="one-to-one">One-to-One</SelectItem>
                    <SelectItem value="one-to-many">One-to-Many</SelectItem>
                    <SelectItem value="many-to-one">Many-to-One</SelectItem>
                    <SelectItem value="many-to-many">Many-to-Many</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <DialogFooter>
              <Button 
                type="submit" 
                onClick={handleCreateRelationship}
                disabled={createRelationshipMutation.isPending || !isFormValid() || hasSameTableError() || hasSameColumnError()}
              >
                {createRelationshipMutation.isPending ? 'Creating...' : 'Create Relationship'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
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