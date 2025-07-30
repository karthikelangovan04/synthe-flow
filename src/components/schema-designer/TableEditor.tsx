import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { supabase } from '@/integrations/supabase/client';
import { Plus, Edit, Trash2, Save, X, Key, Database, MessageSquare, Sparkles } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { BusinessRulesChat } from './BusinessRulesChat';
import { GeneratedBusinessRules } from '@/lib/llm-service';

interface Column {
  id: string;
  name: string;
  data_type: string;
  is_nullable: boolean;
  is_primary_key: boolean;
  is_unique: boolean;
  default_value: string | null;
  max_length: number | null;
  pattern: string | null;
  sample_values: string[] | null;
  position: number;
  business_rules?: string | null;
  enhanced_description?: string | null;
  data_generation_rules?: string | null;
  validation_rules?: string | null;
}

interface TableData {
  id: string;
  name: string;
  description: string | null;
  business_rules?: string | null;
  enhanced_description?: string | null;
  domain_context?: string | null;
  column_metadata: Column[];
}

interface TableEditorProps {
  tableId: string | null;
  onTableUpdated: () => void;
}

const DATA_TYPES = [
  'text', 'varchar', 'char', 'integer', 'bigint', 'smallint',
  'decimal', 'numeric', 'real', 'double precision', 'boolean',
  'date', 'time', 'timestamp', 'timestamptz', 'uuid', 'json', 'jsonb'
];

export function TableEditor({ tableId, onTableUpdated }: TableEditorProps) {
  const [editingColumn, setEditingColumn] = useState<string | null>(null);
  const [newColumn, setNewColumn] = useState({
    name: '',
    data_type: 'text',
    is_nullable: true,
    is_primary_key: false,
    is_unique: false,
  });
  const [showNewColumn, setShowNewColumn] = useState(false);
  const [tableName, setTableName] = useState('');
  const [tableDescription, setTableDescription] = useState('');
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: table, isLoading } = useQuery({
    queryKey: ['table', tableId],
    queryFn: async () => {
      if (!tableId) return null;
      
      const { data, error } = await supabase
        .from('table_metadata')
        .select(`
          *,
          column_metadata(*)
        `)
        .eq('id', tableId)
        .single();
      
      if (error) throw error;
      return data as TableData;
    },
    enabled: !!tableId,
  });

  const updateTableMutation = useMutation({
    mutationFn: async ({ 
      name, 
      description, 
      business_rules, 
      enhanced_description 
    }: { 
      name: string; 
      description: string; 
      business_rules?: string; 
      enhanced_description?: string; 
    }) => {
      if (!tableId) throw new Error('No table selected');
      
      const { error } = await supabase
        .from('table_metadata')
        .update({ 
          name, 
          description, 
          business_rules, 
          enhanced_description 
        })
        .eq('id', tableId);
      
      if (error) throw error;
    },
    onSuccess: () => {
      toast({ title: "Success", description: "Table updated successfully" });
      queryClient.invalidateQueries({ queryKey: ['table', tableId] });
      onTableUpdated();
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to update table", variant: "destructive" });
    },
  });

  // Update local state when table data changes
  useEffect(() => {
    if (table) {
      setTableName(table.name);
      setTableDescription(table.description || '');
    }
  }, [table]);

  // Debounced update effect
  useEffect(() => {
    if (!table) return;
    
    const timeoutId = setTimeout(() => {
      if (tableName !== table.name || tableDescription !== (table.description || '')) {
        updateTableMutation.mutate({ name: tableName, description: tableDescription });
      }
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [tableName, tableDescription, table, updateTableMutation]);

  const createColumnMutation = useMutation({
    mutationFn: async (columnData: typeof newColumn) => {
      if (!tableId) throw new Error('No table selected');
      
      const { error } = await supabase
        .from('column_metadata')
        .insert({
          table_id: tableId,
          ...columnData,
          position: (table?.column_metadata?.length || 0) + 1,
        });
      
      if (error) throw error;
    },
    onSuccess: () => {
      toast({ title: "Success", description: "Column created successfully" });
      setNewColumn({
        name: '',
        data_type: 'text',
        is_nullable: true,
        is_primary_key: false,
        is_unique: false,
      });
      setShowNewColumn(false);
      queryClient.invalidateQueries({ queryKey: ['table', tableId] });
      onTableUpdated();
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to create column", variant: "destructive" });
    },
  });

  const updateColumnMutation = useMutation({
    mutationFn: async ({ id, ...columnData }: Partial<Column> & { id: string }) => {
      const { error } = await supabase
        .from('column_metadata')
        .update(columnData)
        .eq('id', id);
      
      if (error) throw error;
    },
    onSuccess: () => {
      toast({ title: "Success", description: "Column updated successfully" });
      setEditingColumn(null);
      queryClient.invalidateQueries({ queryKey: ['table', tableId] });
      onTableUpdated();
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to update column", variant: "destructive" });
    },
  });

  const deleteColumnMutation = useMutation({
    mutationFn: async (columnId: string) => {
      const { error } = await supabase
        .from('column_metadata')
        .delete()
        .eq('id', columnId);
      
      if (error) throw error;
    },
    onSuccess: () => {
      toast({ title: "Success", description: "Column deleted successfully" });
      queryClient.invalidateQueries({ queryKey: ['table', tableId] });
      onTableUpdated();
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to delete column", variant: "destructive" });
    },
  });

  if (!tableId) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground border-l">
        <div className="text-center">
          <Database className="h-16 w-16 mx-auto mb-4 opacity-50" />
          <p className="text-lg">No table selected</p>
          <p className="text-sm">Select a table from the canvas to edit its structure</p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground border-l">
        <p>Loading table details...</p>
      </div>
    );
  }

  if (!table) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground border-l">
        <p>Table not found</p>
      </div>
    );
  }

  const handleBusinessRulesGenerated = (rules: GeneratedBusinessRules) => {
    // Update the table with generated business rules
    if (table) {
      updateTableMutation.mutate({
        name: tableName,
        description: tableDescription,
        business_rules: rules.businessRules,
        enhanced_description: rules.enhancedDescription
      });
    }
  };

  return (
    <div className="h-full flex flex-col border-l">
      <Tabs defaultValue="structure" className="h-full flex flex-col">
        <div className="border-b px-4 pt-4">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="structure">Structure</TabsTrigger>
            <TabsTrigger value="business-rules">Business Rules</TabsTrigger>
            <TabsTrigger value="ai-assistant">AI Assistant</TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="structure" className="flex-1 flex flex-col">
          <div className="border-b p-4">
            <h2 className="text-lg font-semibold mb-4">Table Editor</h2>
            
            <div className="space-y-3">
              <div>
                <label className="text-sm font-medium">Table Name</label>
                <Input
                  value={tableName}
                  onChange={(e) => setTableName(e.target.value)}
                  className="mt-1"
                />
              </div>
              
              <div>
                <label className="text-sm font-medium">Description</label>
                <Textarea
                  value={tableDescription}
                  onChange={(e) => setTableDescription(e.target.value)}
                  className="mt-1 min-h-[60px]"
                  placeholder="Table description..."
                />
              </div>

              {table.enhanced_description && (
                <div>
                  <label className="text-sm font-medium flex items-center gap-2">
                    <Sparkles className="h-4 w-4" />
                    AI Enhanced Description
                  </label>
                  <Textarea
                    value={table.enhanced_description}
                    readOnly
                    className="mt-1 min-h-[60px] bg-muted/50"
                    placeholder="AI generated description will appear here..."
                  />
                </div>
              )}
            </div>
          </div>

          <div className="flex-1 overflow-auto">
            <div className="p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-medium">Columns</h3>
                <Button size="sm" onClick={() => setShowNewColumn(true)}>
                  <Plus className="h-4 w-4 mr-2" />
                  Add Column
                </Button>
              </div>

              {showNewColumn && (
                <Card className="mb-4">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">New Column</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <Input
                      placeholder="Column name"
                      value={newColumn.name}
                      onChange={(e) => setNewColumn({ ...newColumn, name: e.target.value })}
                    />
                    
                    <Select
                      value={newColumn.data_type}
                      onValueChange={(value) => setNewColumn({ ...newColumn, data_type: value })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {DATA_TYPES.map((type) => (
                          <SelectItem key={type} value={type}>
                            {type}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>

                    <div className="flex flex-wrap gap-4">
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="nullable"
                          checked={newColumn.is_nullable}
                          onCheckedChange={(checked) =>
                            setNewColumn({ ...newColumn, is_nullable: checked as boolean })
                          }
                        />
                        <label htmlFor="nullable" className="text-sm">Nullable</label>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="primary"
                          checked={newColumn.is_primary_key}
                          onCheckedChange={(checked) =>
                            setNewColumn({ ...newColumn, is_primary_key: checked as boolean })
                          }
                        />
                        <label htmlFor="primary" className="text-sm">Primary Key</label>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="unique"
                          checked={newColumn.is_unique}
                          onCheckedChange={(checked) =>
                            setNewColumn({ ...newColumn, is_unique: checked as boolean })
                          }
                        />
                        <label htmlFor="unique" className="text-sm">Unique</label>
                      </div>
                    </div>

                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        onClick={() => createColumnMutation.mutate(newColumn)}
                        disabled={createColumnMutation.isPending || !newColumn.name}
                        className="flex-1"
                      >
                        <Save className="h-4 w-4 mr-2" />
                        Save
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => setShowNewColumn(false)}
                        disabled={createColumnMutation.isPending}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}

              <div className="space-y-2">
                {table.column_metadata?.map((column) => (
                  <Card key={column.id}>
                    <CardContent className="p-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 flex-1">
                          <div className="flex items-center gap-1">
                            {column.is_primary_key && <Key className="h-3 w-3 text-yellow-500" />}
                            <span className="font-medium text-sm">{column.name}</span>
                          </div>
                          
                          <Badge variant="outline" className="text-xs">
                            {column.data_type}
                          </Badge>
                          
                          {column.is_unique && (
                            <Badge variant="secondary" className="text-xs">
                              Unique
                            </Badge>
                          )}
                          
                          {!column.is_nullable && (
                            <Badge variant="destructive" className="text-xs">
                              NOT NULL
                            </Badge>
                          )}

                          {column.business_rules && (
                            <Badge variant="default" className="text-xs bg-green-100 text-green-800">
                              <Sparkles className="h-3 w-3 mr-1" />
                              AI Rules
                            </Badge>
                          )}
                        </div>
                        
                        <div className="flex gap-1">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => setEditingColumn(column.id)}
                          >
                            <Edit className="h-3 w-3" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => deleteColumnMutation.mutate(column.id)}
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>

                      {column.enhanced_description && (
                        <div className="mt-2 p-2 bg-muted/50 rounded text-xs">
                          <div className="font-medium mb-1 flex items-center gap-1">
                            <Sparkles className="h-3 w-3" />
                            AI Description
                          </div>
                          <div className="text-muted-foreground">{column.enhanced_description}</div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="business-rules" className="flex-1 p-4">
          <div className="space-y-4">
            <div>
              <h3 className="text-lg font-semibold mb-2">Business Rules</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Define business rules and constraints for your table and columns.
              </p>
            </div>

            {table.business_rules && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Sparkles className="h-4 w-4" />
                    Table Business Rules
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Textarea
                    value={table.business_rules}
                    readOnly
                    className="min-h-[100px] bg-muted/50"
                  />
                </CardContent>
              </Card>
            )}

            <div className="space-y-3">
              {table.column_metadata?.map((column) => (
                <Card key={column.id}>
                  <CardHeader>
                    <CardTitle className="text-sm">{column.name}</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {column.business_rules && (
                      <div>
                        <label className="text-sm font-medium">Business Rules</label>
                        <Textarea
                          value={column.business_rules}
                          readOnly
                          className="min-h-[80px] bg-muted/50"
                        />
                      </div>
                    )}
                    {column.validation_rules && (
                      <div>
                        <label className="text-sm font-medium">Validation Rules</label>
                        <Textarea
                          value={column.validation_rules}
                          readOnly
                          className="min-h-[60px] bg-muted/50"
                        />
                      </div>
                    )}
                    {column.data_generation_rules && (
                      <div>
                        <label className="text-sm font-medium">Data Generation Rules</label>
                        <Textarea
                          value={column.data_generation_rules}
                          readOnly
                          className="min-h-[60px] bg-muted/50"
                        />
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="ai-assistant" className="flex-1 p-4">
          <BusinessRulesChat
            projectId={table.project_id || ''}
            tableId={table.id}
            columnId={null}
            tableName={table.name}
            onBusinessRulesGenerated={handleBusinessRulesGenerated}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}