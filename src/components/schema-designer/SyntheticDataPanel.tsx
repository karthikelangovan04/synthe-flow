import { useState, useCallback } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Database, 
  Upload, 
  Download, 
  Play, 
  CheckCircle, 
  AlertCircle,
  FileText,
  Settings,
  Cloud,
  Globe,
  BookOpen,
  Plus,
  Trash2,
  TestTube,
  FileUp,
  CloudSnow,
  Server
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { useDropzone } from 'react-dropzone';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import React from 'react'; // Added missing import for React

interface TableData {
  id: string;
  name: string;
  description: string | null;
  column_metadata: Array<{
    id: string;
    name: string;
    data_type: string;
    is_nullable: boolean;
    is_primary_key: boolean;
    is_unique: boolean;
    enhanced_description?: string | null;
  }>;
}

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

interface SyntheticDataPanelProps {
  tables: TableData[];
  relationships: Relationship[];
  projectId: string | null;
}

interface DataSourceConfig {
  type: string;
  config: Record<string, any>;
  file_paths?: string[];
}

interface GenerationConfig {
  scale: number;
  quality_threshold: number;
  include_relationships: boolean;
  sample_size: number;
  output_format: 'json' | 'csv' | 'excel' | 'sql';
  data_sources: DataSourceConfig[];
  random_seed?: number | null;  // For reproducible results
}

interface ConnectorInfo {
  type: string;
  name: string;
  description: string;
  icon: string;
  supported_formats: string[];
}

export function SyntheticDataPanel({
  tables,
  relationships,
  projectId,
}: SyntheticDataPanelProps) {
  console.log('=== SyntheticDataPanel Props ===');
  console.log('SyntheticDataPanel received relationships:', relationships);
  console.log('SyntheticDataPanel received tables:', tables);
  console.log('Table names in order:', tables.map(t => t.name));
  console.log('Relationships count:', relationships.length);
  console.log('Build timestamp:', new Date().toISOString());
  console.log('Cache busting:', Math.random());
  console.log('=== FORCE CACHE BUSTING ===');
  console.log('Current time:', Date.now());
  console.log('Random number:', Math.random());
  console.log('=== END CACHE BUSTING ===');
  
  // Immediate debugging for table ordering
  const usersTable = tables.find(t => t.name.toLowerCase().includes('user'));
  const postsTable = tables.find(t => t.name.toLowerCase().includes('post'));
  console.log('Found users table:', usersTable?.name);
  console.log('Found posts table:', postsTable?.name);
  const [config, setConfig] = useState<GenerationConfig>({
    scale: 1.0,
    quality_threshold: 0.8,
    include_relationships: true,
    sample_size: 1000,
    output_format: 'json',
    data_sources: [],
    random_seed: null,  // For reproducible results
  });
  const [generationStatus, setGenerationStatus] = useState<'idle' | 'generating' | 'completed' | 'error'>('idle');
  const [syntheticData, setSyntheticData] = useState<any>(null);
  const [qualityMetrics, setQualityMetrics] = useState<any>(null);
  const [uploadedFiles, setUploadedFiles] = useState<Array<{filename: string, original_name?: string, size: number, path: string}>>([]);
  const [showConnectorDialog, setShowConnectorDialog] = useState(false);
  const [selectedConnector, setSelectedConnector] = useState<ConnectorInfo | null>(null);
  const [connectorConfig, setConnectorConfig] = useState<Record<string, any>>({});
  const { toast } = useToast();

  // Fetch available connectors
  const { data: connectorsData } = useQuery({
    queryKey: ['connectors'],
    queryFn: async () => {
      const response = await fetch('http://localhost:8002/api/connectors/available');
      if (!response.ok) throw new Error('Failed to fetch connectors');
      return response.json();
    },
  });

  // File upload dropzone
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    console.log('🚀🚀🚀 UPDATED onDrop callback triggered 🚀🚀🚀');
    console.log('=== FORCE CACHE BUSTING in onDrop ===');
    console.log('Current time:', Date.now());
    console.log('Random number:', Math.random());
    console.log('Accepted files:', acceptedFiles.map(f => f.name));
    console.log('Current config.data_sources:', config.data_sources);
    
    // Check for duplicate files
    const existingFileNames = config.data_sources
      .filter(ds => ds.type === 'local')
      .map(ds => ds.config?.file_name)
      .filter(Boolean);
    
    console.log('Existing file names:', existingFileNames);
    
    // Filter out files that already exist
    const newFiles = acceptedFiles.filter(file => 
      !existingFileNames.includes(file.name)
    );
    
    if (newFiles.length === 0) {
      toast({
        title: "No new files to upload",
        description: "All selected files have already been uploaded.",
        variant: "destructive",
      });
      return;
    }
    
    if (newFiles.length !== acceptedFiles.length) {
      toast({
        title: "Some files skipped",
        description: `${acceptedFiles.length - newFiles.length} files were already uploaded and skipped.`,
        variant: "default",
      });
    }
    
    const uploadPromises = newFiles.map(async (file) => {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8002/api/upload/file', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Failed to upload ${file.name}`);
      }

      return response.json();
    });

    try {
      const results = await Promise.all(uploadPromises);
      setUploadedFiles(prev => [...prev, ...results]);
      
      // Create separate data sources for each file, mapped to tables
      console.log('=== DEBUG: File Upload Mapping ===');
      console.log('Tables received:', tables);
      console.log('Tables names:', tables.map(t => t.name));
      console.log('Upload results:', results);
      
            // NEW APPROACH: Dynamic upload based on table count
      console.log('=== NEW DYNAMIC UPLOAD APPROACH ===');
      console.log('Number of tables in schema:', tables.length);
      console.log('Tables:', tables.map(t => t.name));
      
      // Create a simple mapping: each file maps to a table in order
      const tableNames = tables.map(t => t.name);
      console.log('Table names for mapping:', tableNames);
      
      // Show a toast notification to confirm dynamic upload is working
      toast({
        title: "Dynamic Upload Active",
        description: `Mapping ${results.length} files to ${tableNames.length} tables: ${tableNames.join(', ')}`,
        variant: "default",
      });
      console.log('Number of results to map:', results.length);
      
      const newDataSources = results.map((result, index) => {
        // Calculate the total index based on existing files + current batch
        const existingFileCount = config.data_sources.filter(ds => ds.type === 'local').length;
        const totalIndex = existingFileCount + index;
        const targetTable = tableNames[totalIndex] || 'unknown';
        
        console.log(`=== File Mapping Debug for ${result.filename} ===`);
        console.log(`File index in batch: ${index}`);
        console.log(`Existing file count: ${existingFileCount}`);
        console.log(`Total index: ${totalIndex}`);
        console.log(`Available tables: [${tableNames.join(', ')}]`);
        console.log(`Table names array:`, tableNames);
        console.log(`Table names length:`, tableNames.length);
        console.log(`Total index ${totalIndex} accessing tableNames[${totalIndex}]:`, tableNames[totalIndex]);
        console.log(`Mapping file ${totalIndex} to table: ${targetTable}`);
        
        return {
          type: 'local',
          config: {
            table_name: targetTable,
            file_name: result.filename,
          },
          file_paths: [result.filename],
        };
      });
      
      console.log('New data sources created:', newDataSources);
      
      setConfig(prev => {
        console.log('=== DEBUG: setConfig callback ===');
        console.log('Previous config:', prev);
        console.log('Previous data sources:', prev.data_sources);
        
        // Get current local sources to avoid duplicates
        const currentLocalSources = prev.data_sources.filter(ds => ds.type === 'local');
        const existingFileNames = currentLocalSources.map(ds => ds.config?.file_name).filter(Boolean);
        
        console.log('Current local sources:', currentLocalSources);
        console.log('Existing file names:', existingFileNames);
        
        // Filter out any files that already exist
        const uniqueNewDataSources = newDataSources.filter(ds => 
          !existingFileNames.includes(ds.config.file_name)
        );
        
        console.log('Unique new data sources:', uniqueNewDataSources);
        
        const newConfig = {
          ...prev,
          data_sources: [...prev.data_sources, ...uniqueNewDataSources]
        };
        
        console.log('New config:', newConfig);
        return newConfig;
      });

      toast({
        title: 'Success',
        description: `Uploaded ${acceptedFiles.length} file(s) and mapped to ${tables.length} tables`,
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to upload files',
        variant: 'destructive',
      });
    }
  }, [toast, tables, config.data_sources]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'application/json': ['.json'],
    },
  });

  // Test connector connection
  const testConnectorMutation = useMutation({
    mutationFn: async (connectorConfig: DataSourceConfig) => {
      const response = await fetch('http://localhost:8002/api/connectors/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(connectorConfig),
      });
      if (!response.ok) throw new Error('Failed to test connector');
      return response.json();
    },
    onSuccess: (data) => {
      if (data.available) {
        toast({
          title: 'Success',
          description: `Connection to ${selectedConnector?.name} successful`,
        });
        // Add connector to data sources
        setConfig(prev => ({
          ...prev,
          data_sources: [...prev.data_sources, {
            type: selectedConnector!.type,
            config: connectorConfig,
          }]
        }));
        setShowConnectorDialog(false);
      } else {
        toast({
          title: 'Error',
          description: `Connection to ${selectedConnector?.name} failed: ${data.message}`,
          variant: 'destructive',
        });
      }
    },
    onError: (error) => {
      toast({
        title: 'Error',
        description: 'Failed to test connector connection',
        variant: 'destructive',
      });
    },
  });

  const generateDataMutation = useMutation({
    mutationFn: async () => {
      // Get current values to avoid stale closure issues
      const currentRelationships = relationships;
      const currentTables = tables;
      const currentConfig = config;
      
      console.log('=== Current Values in Mutation ===');
      console.log('Current relationships:', currentRelationships);
      console.log('Current relationships length:', currentRelationships.length);
      console.log('Current tables:', currentTables);
      console.log('Current config:', currentConfig);
      
      // Prepare schema data for SDV service
      const schemaData = {
        tables: currentTables.map(table => ({
          name: table.name,
          description: table.description || '',
          columns: table.column_metadata.map(col => ({
            name: col.name,
            data_type: col.data_type,
            is_nullable: col.is_nullable ?? true,
            is_primary_key: col.is_primary_key ?? false,
            is_unique: col.is_unique ?? false,
            enhanced_description: col.enhanced_description || '',
          })),
        })),
        relationships: (() => {
          console.log('=== Processing Relationships ===');
          console.log('Current relationships:', currentRelationships);
          console.log('Current relationships length:', currentRelationships.length);
          
          const filteredRelationships = currentRelationships.filter(rel => 
            rel.source_table_name && 
            rel.source_column_name && 
            rel.target_table_name && 
            rel.target_column_name
          );
          
          console.log('Filtered relationships:', filteredRelationships);
          console.log('Filtered relationships length:', filteredRelationships.length);
          
          let mappedRelationships = filteredRelationships.map(rel => ({
            source_table: rel.source_table_name,
            source_column: rel.source_column_name,
            target_table: rel.target_table_name,
            target_column: rel.target_column_name,
            relationship_type: rel.relationship_type,
          }));
          
          // If no relationships found but we have users and posts tables, create a basic relationship
          if (mappedRelationships.length === 0) {
            const usersTable = currentTables.find(t => t.name.toLowerCase().includes('user'));
            const postsTable = currentTables.find(t => t.name.toLowerCase().includes('post'));
            
            if (usersTable && postsTable) {
              const usersUserIdColumn = usersTable.column_metadata.find(c => c.name.toLowerCase().includes('user_id') || c.is_primary_key);
              const postsUserIdColumn = postsTable.column_metadata.find(c => c.name.toLowerCase().includes('user_id'));
              
              if (usersUserIdColumn && postsUserIdColumn) {
                console.log('Creating fallback relationship: users -> posts');
                mappedRelationships = [{
                  source_table: usersTable.name,
                  source_column: usersUserIdColumn.name,
                  target_table: postsTable.name,
                  target_column: postsUserIdColumn.name,
                  relationship_type: 'one-to-many',
                }];
              }
            }
          }
          
          console.log('Final mapped relationships:', mappedRelationships);
          return mappedRelationships;
        })(),
        data_sources: currentConfig.data_sources.length > 0 ? currentConfig.data_sources : undefined,
        scale: currentConfig.scale,
        output_format: currentConfig.output_format,
        random_seed: currentConfig.random_seed,  // For reproducible results
        quality_settings: {
          threshold: currentConfig.quality_threshold,
          include_relationships: currentConfig.include_relationships,
        },
      };

      console.log('=== DEBUG: Generation Request ===');
      console.log('Relationships received:', currentRelationships);
      console.log('Relationships length:', currentRelationships.length);
      console.log('Filtered relationships:', currentRelationships.filter(rel => 
        rel.source_table_name && 
        rel.source_column_name && 
        rel.target_table_name && 
        rel.target_column_name
      ));
      console.log('Data sources:', currentConfig.data_sources);
      console.log('Data sources length:', currentConfig.data_sources.length);
      console.log('Sending schema data to backend:', JSON.stringify(schemaData, null, 2));
      
      const response = await fetch('http://localhost:8002/api/sdv/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(schemaData),
      });

      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
        } catch (e) {
          errorData = { detail: `HTTP error! status: ${response.status}` };
        }
        
        // Handle different types of error responses
        let errorMessage = `HTTP error! status: ${response.status}`;
        
        if (errorData.detail) {
          errorMessage = errorData.detail;
        } else if (errorData.message) {
          errorMessage = errorData.message;
        } else if (errorData.error) {
          errorMessage = errorData.error;
        } else if (errorData.errors && Array.isArray(errorData.errors)) {
          errorMessage = errorData.errors.map((e: any) => e.message || e.msg || JSON.stringify(e)).join(', ');
        } else if (typeof errorData === 'string') {
          errorMessage = errorData;
        }
        
        console.error('Backend error response:', errorData);
        throw new Error(errorMessage);
      }

      return response.json();
    },
    onSuccess: (data) => {
      setSyntheticData(data.synthetic_data);
      setQualityMetrics(data.quality_metrics);
      setGenerationStatus('completed');
      toast({
        title: 'Success',
        description: 'Synthetic data generated successfully',
      });
    },
    onError: (error: any) => {
      setGenerationStatus('error');
      
      // Extract error message properly
      let errorMessage = 'Failed to generate synthetic data';
      
      if (error.message) {
        errorMessage = error.message;
      } else if (error.detail) {
        errorMessage = error.detail;
      } else if (typeof error === 'string') {
        errorMessage = error;
      } else if (error && typeof error === 'object') {
        // Try to extract meaningful error information
        if (error.errors && Array.isArray(error.errors)) {
          errorMessage = error.errors.map((e: any) => e.message || e.msg || JSON.stringify(e)).join(', ');
        } else if (error.error) {
          errorMessage = error.error;
        } else {
          errorMessage = JSON.stringify(error);
        }
      }
      
      console.error('Generation error:', error);
      
      toast({
        title: 'Error',
        description: errorMessage,
        variant: 'destructive',
      });
    },
  });

  const handleGenerateData = () => {
    setGenerationStatus('generating');
    generateDataMutation.mutate();
  };

  const handleTestConnector = () => {
    if (selectedConnector && connectorConfig) {
      testConnectorMutation.mutate({
        type: selectedConnector.type,
        config: connectorConfig,
      });
    }
  };

  const handleRemoveDataSource = (index: number) => {
    setConfig(prev => ({
      ...prev,
      data_sources: prev.data_sources.filter((_, i) => i !== index),
    }));
    
    // Also remove from uploaded files if it's a local file
    const dataSource = config.data_sources[index];
    if (dataSource?.type === 'local' && dataSource?.config?.file_name) {
      setUploadedFiles(prev => prev.filter(file => file.filename !== dataSource.config.file_name));
    }
  };

  const handleClearAllFiles = () => {
    setUploadedFiles([]);
    setConfig(prev => ({
      ...prev,
      data_sources: prev.data_sources.filter(ds => ds.type !== 'local')
    }));
  };

  const handleExportData = (format: 'csv' | 'json' | 'sql') => {
    if (!syntheticData) return;

    let content = '';
    let filename = 'synthetic_data';

    if (format === 'json') {
      content = JSON.stringify(syntheticData, null, 2);
      filename += '.json';
    } else if (format === 'csv') {
      const csvContent = Object.entries(syntheticData).map(([tableName, rows]: [string, any]) => {
        if (!Array.isArray(rows) || rows.length === 0) return '';
        
        const headers = Object.keys(rows[0]);
        const csvRows = [headers.join(',')];
        
        rows.forEach((row: any) => {
          const values = headers.map(header => `"${row[header]}"`);
          csvRows.push(values.join(','));
        });
        
        return `\n${tableName}\n${csvRows.join('\n')}`;
      }).join('\n');
      
      content = csvContent;
      filename += '.csv';
    } else if (format === 'sql') {
      const sqlStatements = Object.entries(syntheticData).map(([tableName, rows]: [string, any]) => {
        if (!Array.isArray(rows) || rows.length === 0) return '';
        
        const columns = Object.keys(rows[0]);
        const insertStatements = rows.map((row: any) => {
          const values = columns.map(col => `'${row[col]}'`);
          return `INSERT INTO ${tableName} (${columns.join(', ')}) VALUES (${values.join(', ')});`;
        });
        
        return insertStatements.join('\n');
      }).join('\n\n');
      
      content = sqlStatements;
      filename += '.sql';
    }

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getConnectorIcon = (iconName: string) => {
    const iconMap: Record<string, any> = {
      upload: Upload,
      database: Database,
      'cloud-snow': CloudSnow,
      cloud: Cloud,
      globe: Globe,
      'book-open': BookOpen,
      server: Server,
    };
    return iconMap[iconName] || Database;
  };

  const renderConnectorConfig = () => {
    if (!selectedConnector) return null;

    switch (selectedConnector.type) {
      case 'postgres':
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="host">Host</Label>
                <Input
                  id="host"
                  value={connectorConfig.host || ''}
                  onChange={(e) => setConnectorConfig(prev => ({ ...prev, host: e.target.value }))}
                  placeholder="localhost"
                />
              </div>
              <div>
                <Label htmlFor="port">Port</Label>
                <Input
                  id="port"
                  type="number"
                  value={connectorConfig.port || 5432}
                  onChange={(e) => setConnectorConfig(prev => ({ ...prev, port: parseInt(e.target.value) }))}
                />
              </div>
            </div>
            <div>
              <Label htmlFor="database">Database</Label>
              <Input
                id="database"
                value={connectorConfig.database || ''}
                onChange={(e) => setConnectorConfig(prev => ({ ...prev, database: e.target.value }))}
                placeholder="mydb"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="username">Username</Label>
                <Input
                  id="username"
                  value={connectorConfig.username || ''}
                  onChange={(e) => setConnectorConfig(prev => ({ ...prev, username: e.target.value }))}
                />
              </div>
              <div>
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  value={connectorConfig.password || ''}
                  onChange={(e) => setConnectorConfig(prev => ({ ...prev, password: e.target.value }))}
                />
              </div>
            </div>
          </div>
        );

      case 'snowflake':
        return (
          <div className="space-y-4">
            <div>
              <Label htmlFor="account">Account</Label>
              <Input
                id="account"
                value={connectorConfig.account || ''}
                onChange={(e) => setConnectorConfig(prev => ({ ...prev, account: e.target.value }))}
                placeholder="your-account.snowflakecomputing.com"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="warehouse">Warehouse</Label>
                <Input
                  id="warehouse"
                  value={connectorConfig.warehouse || ''}
                  onChange={(e) => setConnectorConfig(prev => ({ ...prev, warehouse: e.target.value }))}
                />
              </div>
              <div>
                <Label htmlFor="database">Database</Label>
                <Input
                  id="database"
                  value={connectorConfig.database || ''}
                  onChange={(e) => setConnectorConfig(prev => ({ ...prev, database: e.target.value }))}
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="username">Username</Label>
                <Input
                  id="username"
                  value={connectorConfig.username || ''}
                  onChange={(e) => setConnectorConfig(prev => ({ ...prev, username: e.target.value }))}
                />
              </div>
              <div>
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  value={connectorConfig.password || ''}
                  onChange={(e) => setConnectorConfig(prev => ({ ...prev, password: e.target.value }))}
                />
              </div>
            </div>
          </div>
        );

      case 's3':
        return (
          <div className="space-y-4">
            <div>
              <Label htmlFor="bucket">Bucket Name</Label>
              <Input
                id="bucket"
                value={connectorConfig.bucket || ''}
                onChange={(e) => setConnectorConfig(prev => ({ ...prev, bucket: e.target.value }))}
                placeholder="my-bucket"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="access_key">Access Key ID</Label>
                <Input
                  id="access_key"
                  value={connectorConfig.access_key_id || ''}
                  onChange={(e) => setConnectorConfig(prev => ({ ...prev, access_key_id: e.target.value }))}
                />
              </div>
              <div>
                <Label htmlFor="secret_key">Secret Access Key</Label>
                <Input
                  id="secret_key"
                  type="password"
                  value={connectorConfig.secret_access_key || ''}
                  onChange={(e) => setConnectorConfig(prev => ({ ...prev, secret_access_key: e.target.value }))}
                />
              </div>
            </div>
            <div>
              <Label htmlFor="region">Region</Label>
              <Input
                id="region"
                value={connectorConfig.region || 'us-east-1'}
                onChange={(e) => setConnectorConfig(prev => ({ ...prev, region: e.target.value }))}
              />
            </div>
          </div>
        );

      case 'api':
        return (
          <div className="space-y-4">
            <div>
              <Label htmlFor="base_url">Base URL</Label>
              <Input
                id="base_url"
                value={connectorConfig.base_url || ''}
                onChange={(e) => setConnectorConfig(prev => ({ ...prev, base_url: e.target.value }))}
                placeholder="https://api.example.com"
              />
            </div>
            <div>
              <Label htmlFor="endpoints">Endpoints (JSON)</Label>
              <Textarea
                id="endpoints"
                value={connectorConfig.endpoints || ''}
                onChange={(e) => setConnectorConfig(prev => ({ ...prev, endpoints: e.target.value }))}
                placeholder='[{"path": "/users", "table_name": "users"}]'
                rows={4}
              />
            </div>
          </div>
        );

      default:
        return (
          <div className="text-center text-muted-foreground">
            Configuration not available for this connector type.
          </div>
        );
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b">
        <div>
          <h2 className="text-lg font-semibold">Synthetic Data Generation</h2>
          <p className="text-sm text-muted-foreground">
            Generate synthetic data using SDV with multiple data sources
          </p>
        </div>
        <Button
          onClick={handleGenerateData}
          disabled={generationStatus === 'generating' || tables.length === 0}
          className="flex items-center gap-2"
        >
          <Play className="h-4 w-4" />
          Generate Data
        </Button>
      </div>

      <div className="flex-1 overflow-auto p-4">
        <Tabs defaultValue="sources" className="h-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="sources">Data Sources</TabsTrigger>
            <TabsTrigger value="config">Configuration</TabsTrigger>
            <TabsTrigger value="results">Results</TabsTrigger>
            <TabsTrigger value="export">Export</TabsTrigger>
          </TabsList>

          <TabsContent value="sources" className="space-y-4">
            {/* File Upload Section */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <Upload className="h-4 w-4" />
                  Local File Upload
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                    isDragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'
                  }`}
                >
                  <input {...getInputProps()} />
                  <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    {isDragActive
                      ? 'Drop files here...'
                      : 'Drag & drop files here, or click to select'}
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Supports CSV, Excel, JSON files
                  </p>
                  {tables.length > 0 && (
                    <div className="mt-2 p-2 bg-blue-50 rounded text-xs text-blue-700">
                      {(() => {
                        // NEW APPROACH: Simple guidance based on table order
                        const tableNames = tables.map(t => t.name);
                        
                        console.log('=== NEW UPLOAD GUIDANCE ===');
                        console.log('Tables for guidance:', tableNames);
                        console.log('Number of tables:', tableNames.length);
                        
                        // Add a visual indicator that dynamic upload is active
                        const dynamicIndicator = (
                          <div className="mb-2 p-2 bg-green-50 border border-green-200 rounded text-xs text-green-700">
                            🚀 <strong>Dynamic Upload Active:</strong> {tableNames.length} tables detected
                          </div>
                        );
                        
                        if (tableNames.length === 0) {
                          return <span className="text-orange-600">No tables defined in schema</span>;
                        } else if (tableNames.length === 1) {
                          return (
                            <>
                              {dynamicIndicator}
                              <strong>Upload:</strong> {tableNames[0]}.csv
                              <br />
                              <span className="text-blue-600">💡 Upload {tableNames[0]}.csv file</span>
                            </>
                          );
                        } else {
                          return (
                            <>
                              {dynamicIndicator}
                              <strong>Upload Order:</strong> {tableNames.map((name, index) => 
                                `${index + 1}. ${name}.csv`
                              ).join(', ')}
                              <br />
                              <span className="text-blue-600">💡 Upload {tableNames.length} different files: {tableNames.join(', ')}</span>
                              <br />
                              <span className="text-orange-600">⚠️ Make sure each file has a different name!</span>
                            </>
                          );
                        }
                      })()}
                    </div>
                  )}
                </div>
                
                {uploadedFiles.length > 0 && (
                  <div className="mt-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="text-sm font-medium">Uploaded Files:</h4>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleClearAllFiles}
                        className="text-xs"
                      >
                        Clear All
                      </Button>
                    </div>
                    {uploadedFiles.map((file, index) => {
                      // Find the corresponding data source to show mapping
                      const dataSource = config.data_sources.find(ds => 
                        ds.type === 'local' && ds.config?.file_name === file.filename
                      );
                      const mappedTable = dataSource?.config?.table_name || 'Unknown';
                      
                      console.log(`File ${file.filename} mapped to table: ${mappedTable}`);
                      console.log(`Available data sources:`, config.data_sources);
                      
                      return (
                        <div key={index} className="flex items-center justify-between p-2 border rounded">
                          <div className="flex-1">
                            <span className="text-sm font-medium">{file.original_name || file.filename}</span>
                            <div className="text-xs text-muted-foreground">
                              Mapped to: <span className="font-medium text-blue-600">{mappedTable}</span>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge variant="outline">{(file.size / 1024).toFixed(1)} KB</Badge>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                const dsIndex = config.data_sources.findIndex(ds => 
                                  ds.type === 'local' && ds.config?.file_name === file.filename
                                );
                                if (dsIndex !== -1) handleRemoveDataSource(dsIndex);
                              }}
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Data Source Connectors */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <Database className="h-4 w-4" />
                  Data Source Connectors
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {connectorsData?.connectors?.map((connector: ConnectorInfo) => (
                    <div
                      key={connector.type}
                      className="border rounded-lg p-4 hover:border-primary/50 transition-colors cursor-pointer"
                      onClick={() => {
                        setSelectedConnector(connector);
                        setConnectorConfig({});
                        setShowConnectorDialog(true);
                      }}
                    >
                      <div className="flex items-center gap-3">
                        {React.createElement(getConnectorIcon(connector.icon), { className: "h-5 w-5" })}
                        <div className="flex-1">
                          <h4 className="font-medium text-sm">{connector.name}</h4>
                          <p className="text-xs text-muted-foreground">{connector.description}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Active Data Sources */}
            {config.data_sources.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Active Data Sources</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {config.data_sources.map((source, index) => (
                      <div key={index} className="flex items-center justify-between p-3 border rounded">
                        <div className="flex items-center gap-3">
                          {React.createElement(getConnectorIcon(
                            connectorsData?.connectors?.find((c: ConnectorInfo) => c.type === source.type)?.icon || 'database'
                          ), { className: "h-4 w-4" })}
                          <div>
                            <p className="text-sm font-medium">
                              {connectorsData?.connectors?.find((c: ConnectorInfo) => c.type === source.type)?.name || source.type}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {source.type === 'local' && source.file_paths 
                                ? source.config.table_name 
                                  ? `Mapped to: ${source.config.table_name}`
                                  : `${source.file_paths.length} file(s)`
                                : 'Connected'
                              }
                            </p>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleRemoveDataSource(index)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="config" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <Settings className="h-4 w-4" />
                  Generation Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="scale">Data Scale</Label>
                    <Input
                      id="scale"
                      type="number"
                      step="0.1"
                      min="0.1"
                      max="10"
                      value={config.scale}
                      onChange={(e) => setConfig({ ...config, scale: parseFloat(e.target.value) })}
                    />
                    <p className="text-xs text-muted-foreground mt-1">
                      Multiplier for data volume (1.0 = same as sample)
                    </p>
                  </div>
                  <div>
                    <Label htmlFor="output_format">Output Format</Label>
                    <Select
                      value={config.output_format}
                      onValueChange={(value: 'json' | 'csv' | 'excel' | 'sql') => 
                        setConfig({ ...config, output_format: value })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="json">JSON</SelectItem>
                        <SelectItem value="csv">CSV</SelectItem>
                        <SelectItem value="excel">Excel</SelectItem>
                        <SelectItem value="sql">SQL</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div>
                  <Label htmlFor="quality_threshold">Quality Threshold</Label>
                  <Input
                    id="quality_threshold"
                    type="number"
                    step="0.1"
                    min="0"
                    max="1"
                    value={config.quality_threshold}
                    onChange={(e) => setConfig({ ...config, quality_threshold: parseFloat(e.target.value) })}
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Minimum quality score (0.0 - 1.0)
                  </p>
                </div>
                <div>
                  <Label htmlFor="random_seed">Random Seed (Optional)</Label>
                  <Input
                    id="random_seed"
                    type="number"
                    step="1"
                    min="1"
                    placeholder="Leave empty for random results"
                    value={config.random_seed || ''}
                    onChange={(e) => {
                      const value = e.target.value;
                      setConfig({ 
                        ...config, 
                        random_seed: value === '' ? null : parseInt(value) 
                      });
                    }}
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Set a seed for reproducible results (same input = same output)
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="include_relationships"
                    checked={config.include_relationships}
                    onCheckedChange={(checked) => setConfig({ ...config, include_relationships: checked })}
                  />
                  <Label htmlFor="include_relationships">Include Relationships</Label>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <Database className="h-4 w-4" />
                  Schema Summary
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Tables:</span>
                    <Badge variant="outline">{tables.length}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Total Columns:</span>
                    <Badge variant="outline">
                      {tables.reduce((sum, table) => sum + table.column_metadata.length, 0)}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Relationships:</span>
                    <Badge variant="outline">{relationships.length}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Data Sources:</span>
                    <Badge variant="outline">{config.data_sources.length}</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="results" className="space-y-4">
            {generationStatus === 'generating' && (
              <Card>
                <CardContent className="p-6">
                  <div className="text-center space-y-4">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
                    <p>Generating synthetic data...</p>
                    <Progress value={50} className="w-full" />
                  </div>
                </CardContent>
              </Card>
            )}

            {generationStatus === 'completed' && syntheticData && (
              <div className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      Generation Complete
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {Object.entries(syntheticData).map(([tableName, rows]: [string, any]) => (
                        <div key={tableName} className="flex justify-between items-center p-2 border rounded">
                          <span className="font-medium">{tableName}</span>
                          <Badge variant="outline">{Array.isArray(rows) ? rows.length : 0} rows</Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {qualityMetrics && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">Quality Metrics</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span>Overall Score:</span>
                          <Badge variant={qualityMetrics.quality_score > 0.8 ? 'default' : 'secondary'}>
                            {(qualityMetrics.quality_score * 100).toFixed(1)}%
                          </Badge>
                        </div>
                        <div className="flex justify-between">
                          <span>Distribution Similarity:</span>
                          <span>{(qualityMetrics.metrics?.distribution_similarity * 100 || 0).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Correlation Preservation:</span>
                          <span>{(qualityMetrics.metrics?.correlation_preservation * 100 || 0).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Privacy Score:</span>
                          <span>{(qualityMetrics.metrics?.privacy_score * 100 || 0).toFixed(1)}%</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            )}

            {generationStatus === 'error' && (
              <Card>
                <CardContent className="p-6">
                  <div className="text-center space-y-4">
                    <AlertCircle className="h-8 w-8 text-destructive mx-auto" />
                    <p>Failed to generate synthetic data</p>
                    <p className="text-sm text-muted-foreground">
                      Please check your configuration and try again
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="export" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <Download className="h-4 w-4" />
                  Export Options
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Button
                      variant="outline"
                      onClick={() => handleExportData('csv')}
                      disabled={!syntheticData}
                      className="flex items-center gap-2"
                    >
                      <FileText className="h-4 w-4" />
                      Export CSV
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => handleExportData('json')}
                      disabled={!syntheticData}
                      className="flex items-center gap-2"
                    >
                      <FileText className="h-4 w-4" />
                      Export JSON
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => handleExportData('sql')}
                      disabled={!syntheticData}
                      className="flex items-center gap-2"
                    >
                      <FileText className="h-4 w-4" />
                      Export SQL
                    </Button>
                  </div>
                  {!syntheticData && (
                    <p className="text-sm text-muted-foreground text-center">
                      Generate data first to enable export options
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Connector Configuration Dialog */}
      <Dialog open={showConnectorDialog} onOpenChange={setShowConnectorDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {selectedConnector && React.createElement(getConnectorIcon(selectedConnector.icon), { className: "h-5 w-5" })}
              Configure {selectedConnector?.name}
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            {renderConnectorConfig()}
            <div className="flex gap-2 pt-4">
              <Button
                onClick={handleTestConnector}
                disabled={testConnectorMutation.isPending}
                className="flex items-center gap-2"
              >
                <TestTube className="h-4 w-4" />
                Test Connection
              </Button>
              <Button
                variant="outline"
                onClick={() => setShowConnectorDialog(false)}
              >
                Cancel
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
} 