import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Upload, 
  FileText, 
  Database, 
  Sparkles, 
  Loader2,
  CheckCircle,
  AlertCircle
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { llmService, BusinessRuleContext, GeneratedBusinessRules } from '@/lib/llm-service';

interface SchemaTable {
  name: string;
  description?: string;
  columns: SchemaColumn[];
}

interface SchemaColumn {
  name: string;
  dataType: string;
  isNullable?: boolean;
  isPrimaryKey?: boolean;
  isUnique?: boolean;
  description?: string;
  businessRules?: string;
  enhancedDescription?: string;
  dataGenerationRules?: string;
  validationRules?: string;
}

interface SchemaImportProps {
  projectId: string;
  onSchemaImported: (tables: SchemaTable[]) => void;
}

export function SchemaImport({ projectId, onSchemaImported }: SchemaImportProps) {
  const [importMethod, setImportMethod] = useState<'json' | 'csv' | 'manual'>('json');
  const [jsonInput, setJsonInput] = useState('');
  const [csvInput, setCsvInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isGeneratingRules, setIsGeneratingRules] = useState(false);
  const [parsedTables, setParsedTables] = useState<SchemaTable[]>([]);
  const [selectedTable, setSelectedTable] = useState<SchemaTable | null>(null);
  const { toast } = useToast();

  const parseJsonSchema = (jsonString: string): SchemaTable[] => {
    try {
      const data = JSON.parse(jsonString);
      
      // Handle different JSON formats
      if (Array.isArray(data)) {
        return data.map(table => ({
          name: table.name || table.table_name || '',
          description: table.description || '',
          columns: (table.columns || []).map((col: any) => ({
            name: col.name || col.column_name || '',
            dataType: col.dataType || col.data_type || col.type || 'text',
            isNullable: col.isNullable !== false,
            isPrimaryKey: col.isPrimaryKey || col.primary_key || false,
            isUnique: col.isUnique || col.unique || false,
            description: col.description || ''
          }))
        }));
      } else if (data.tables) {
        return data.tables.map((table: any) => ({
          name: table.name || '',
          description: table.description || '',
          columns: (table.columns || []).map((col: any) => ({
            name: col.name || '',
            dataType: col.dataType || col.type || 'text',
            isNullable: col.isNullable !== false,
            isPrimaryKey: col.isPrimaryKey || false,
            isUnique: col.isUnique || false,
            description: col.description || ''
          }))
        }));
      }
      
      throw new Error('Invalid JSON format');
    } catch (error) {
      throw new Error('Failed to parse JSON schema');
    }
  };

  const parseCsvSchema = (csvString: string): SchemaTable[] => {
    try {
      const lines = csvString.trim().split('\n');
      if (lines.length < 2) throw new Error('CSV must have at least header and one data row');
      
      const headers = lines[0].split(',').map(h => h.trim());
      const tables: { [key: string]: SchemaTable } = {};
      
      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => v.trim());
        if (values.length !== headers.length) continue;
        
        const row: { [key: string]: string } = {};
        headers.forEach((header, index) => {
          row[header] = values[index];
        });
        
        const tableName = row.table_name || row.tableName || 'default_table';
        const columnName = row.column_name || row.columnName || '';
        const dataType = row.data_type || row.dataType || row.type || 'text';
        
        if (!tables[tableName]) {
          tables[tableName] = {
            name: tableName,
            description: row.table_description || row.tableDescription || '',
            columns: []
          };
        }
        
        if (columnName) {
          tables[tableName].columns.push({
            name: columnName,
            dataType,
            isNullable: row.is_nullable !== 'false',
            isPrimaryKey: row.is_primary_key === 'true' || row.primaryKey === 'true',
            isUnique: row.is_unique === 'true' || row.unique === 'true',
            description: row.column_description || row.description || ''
          });
        }
      }
      
      return Object.values(tables);
    } catch (error) {
      throw new Error('Failed to parse CSV schema');
    }
  };

  const handleImport = async () => {
    setIsProcessing(true);
    
    try {
      let tables: SchemaTable[] = [];
      
      switch (importMethod) {
        case 'json':
          if (!jsonInput.trim()) throw new Error('Please enter JSON schema');
          tables = parseJsonSchema(jsonInput);
          break;
        case 'csv':
          if (!csvInput.trim()) throw new Error('Please enter CSV schema');
          tables = parseCsvSchema(csvInput);
          break;
        default:
          throw new Error('Invalid import method');
      }
      
      if (tables.length === 0) throw new Error('No tables found in schema');
      
      setParsedTables(tables);
      setSelectedTable(tables[0]);
      
      toast({
        title: "Success",
        description: `Successfully parsed ${tables.length} table(s)`
      });
      
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to import schema",
        variant: "destructive"
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const generateBusinessRulesForTable = async (table: SchemaTable) => {
    setIsGeneratingRules(true);
    
    try {
      const updatedTable = { ...table };
      
      // Generate business rules for each column
      for (const column of updatedTable.columns) {
        const context: BusinessRuleContext = {
          tableName: table.name,
          columnName: column.name,
          dataType: column.dataType,
          isNullable: column.isNullable,
          isPrimaryKey: column.isPrimaryKey,
          isUnique: column.isUnique,
          existingDescription: column.description,
          domainContext: 'General business domain'
        };
        
        const generatedRules = await llmService.generateBusinessRules(context);
        
        column.businessRules = generatedRules.businessRules;
        column.enhancedDescription = generatedRules.enhancedDescription;
        column.dataGenerationRules = generatedRules.dataGenerationRules;
        column.validationRules = generatedRules.validationRules;
      }
      
      // Update the parsed tables with generated rules
      setParsedTables(prev => 
        prev.map(t => t.name === table.name ? updatedTable : t)
      );
      
      toast({
        title: "Success",
        description: `Generated business rules for ${table.name}`
      });
      
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to generate business rules",
        variant: "destructive"
      });
    } finally {
      setIsGeneratingRules(false);
    }
  };

  const handleFinalImport = () => {
    if (parsedTables.length === 0) return;
    
    onSchemaImported(parsedTables);
    toast({
      title: "Success",
      description: "Schema imported successfully!"
    });
  };

  const sampleJson = `{
  "tables": [
    {
      "name": "users",
      "description": "User account information",
      "columns": [
        {
          "name": "id",
          "dataType": "uuid",
          "isPrimaryKey": true,
          "isNullable": false,
          "description": "Unique user identifier"
        },
        {
          "name": "email",
          "dataType": "varchar",
          "isNullable": false,
          "isUnique": true,
          "description": "User email address"
        },
        {
          "name": "created_at",
          "dataType": "timestamp",
          "isNullable": false,
          "description": "Account creation timestamp"
        }
      ]
    }
  ]
}`;

  const sampleCsv = `table_name,column_name,data_type,is_nullable,is_primary_key,is_unique,column_description
users,id,uuid,false,true,true,Unique user identifier
users,email,varchar,false,false,true,User email address
users,created_at,timestamp,false,false,false,Account creation timestamp
orders,id,uuid,false,true,true,Unique order identifier
orders,user_id,uuid,false,false,false,Reference to user
orders,amount,decimal,false,false,false,Order total amount`;

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="h-5 w-5" />
          Import Schema
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-6">
        <Tabs value={importMethod} onValueChange={(value) => setImportMethod(value as any)}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="json">JSON</TabsTrigger>
            <TabsTrigger value="csv">CSV</TabsTrigger>
            <TabsTrigger value="manual">Manual</TabsTrigger>
          </TabsList>
          
          <TabsContent value="json" className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">JSON Schema</label>
              <Textarea
                value={jsonInput}
                onChange={(e) => setJsonInput(e.target.value)}
                placeholder="Paste your JSON schema here..."
                className="min-h-[200px] font-mono text-sm"
              />
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setJsonInput(sampleJson)}
                >
                  Load Sample
                </Button>
                <Button
                  onClick={handleImport}
                  disabled={isProcessing || !jsonInput.trim()}
                >
                  {isProcessing ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <FileText className="h-4 w-4 mr-2" />
                  )}
                  Parse JSON
                </Button>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="csv" className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">CSV Schema</label>
              <Textarea
                value={csvInput}
                onChange={(e) => setCsvInput(e.target.value)}
                placeholder="Paste your CSV schema here..."
                className="min-h-[200px] font-mono text-sm"
              />
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setCsvInput(sampleCsv)}
                >
                  Load Sample
                </Button>
                <Button
                  onClick={handleImport}
                  disabled={isProcessing || !csvInput.trim()}
                >
                  {isProcessing ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <FileText className="h-4 w-4 mr-2" />
                  )}
                  Parse CSV
                </Button>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="manual" className="space-y-4">
            <div className="text-center py-8 text-muted-foreground">
              <Database className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>Manual schema creation will be available in the schema designer</p>
            </div>
          </TabsContent>
        </Tabs>

        {/* Parsed Tables Preview */}
        {parsedTables.length > 0 && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium">Parsed Tables ({parsedTables.length})</h3>
              <div className="flex gap-2">
                <Button
                  onClick={() => generateBusinessRulesForTable(selectedTable!)}
                  disabled={isGeneratingRules || !selectedTable}
                  variant="outline"
                >
                  {isGeneratingRules ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Sparkles className="h-4 w-4 mr-2" />
                  )}
                  Generate Business Rules
                </Button>
                <Button onClick={handleFinalImport}>
                  <CheckCircle className="h-4 w-4 mr-2" />
                  Import to Schema Designer
                </Button>
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Table List */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Tables</label>
                <div className="space-y-1">
                  {parsedTables.map((table) => (
                    <div
                      key={table.name}
                      className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                        selectedTable?.name === table.name
                          ? 'border-primary bg-primary/5'
                          : 'hover:bg-muted/50'
                      }`}
                      onClick={() => setSelectedTable(table)}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium">{table.name}</div>
                          <div className="text-sm text-muted-foreground">
                            {table.columns.length} columns
                          </div>
                        </div>
                        {table.columns.some(col => col.businessRules) && (
                          <Badge variant="secondary" className="text-xs">
                            <Sparkles className="h-3 w-3 mr-1" />
                            AI Enhanced
                          </Badge>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Table Details */}
              {selectedTable && (
                <div className="space-y-2">
                  <label className="text-sm font-medium">Columns in {selectedTable.name}</label>
                  <div className="space-y-2 max-h-[400px] overflow-y-auto">
                    {selectedTable.columns.map((column) => (
                      <div key={column.name} className="p-3 border rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <div className="font-medium">{column.name}</div>
                          <Badge variant="outline" className="text-xs">
                            {column.dataType}
                          </Badge>
                        </div>
                        {column.description && (
                          <div className="text-sm text-muted-foreground mb-2">
                            {column.description}
                          </div>
                        )}
                        {column.businessRules && (
                          <div className="text-xs bg-muted p-2 rounded">
                            <div className="font-medium mb-1">Business Rules:</div>
                            <div className="line-clamp-2">{column.businessRules}</div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
} 