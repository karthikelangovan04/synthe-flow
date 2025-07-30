import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
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
  Settings
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

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

interface GenerationConfig {
  scale: number;
  quality_threshold: number;
  include_relationships: boolean;
  sample_size: number;
}

export function SyntheticDataPanel({
  tables,
  relationships,
  projectId,
}: SyntheticDataPanelProps) {
  const [config, setConfig] = useState<GenerationConfig>({
    scale: 1.0,
    quality_threshold: 0.8,
    include_relationships: true,
    sample_size: 1000,
  });
  const [generationStatus, setGenerationStatus] = useState<'idle' | 'generating' | 'completed' | 'error'>('idle');
  const [syntheticData, setSyntheticData] = useState<any>(null);
  const [qualityMetrics, setQualityMetrics] = useState<any>(null);
  const { toast } = useToast();

  const generateDataMutation = useMutation({
    mutationFn: async () => {
      // Prepare schema data for SDV service
      const schemaData = {
        tables: tables.map(table => ({
          name: table.name,
          description: table.description,
          columns: table.column_metadata.map(col => ({
            name: col.name,
            data_type: col.data_type,
            is_nullable: col.is_nullable,
            is_primary_key: col.is_primary_key,
            is_unique: col.is_unique,
            enhanced_description: col.enhanced_description,
          })),
        })),
        relationships: relationships.map(rel => ({
          source_table: rel.source_table_name,
          source_column: rel.source_column_name,
          target_table: rel.target_table_name,
          target_column: rel.target_column_name,
          relationship_type: rel.relationship_type,
        })),
        scale: config.scale,
        quality_settings: {
          threshold: config.quality_threshold,
          include_relationships: config.include_relationships,
        },
      };

      // Call SDV service
      const response = await fetch('http://localhost:8001/api/sdv/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(schemaData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
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
      toast({
        title: 'Error',
        description: error.message || 'Failed to generate synthetic data',
        variant: 'destructive',
      });
    },
  });

  const handleGenerateData = () => {
    setGenerationStatus('generating');
    generateDataMutation.mutate();
  };

  const handleExportData = (format: 'csv' | 'json' | 'sql') => {
    if (!syntheticData) return;

    let content = '';
    let filename = 'synthetic_data';

    if (format === 'json') {
      content = JSON.stringify(syntheticData, null, 2);
      filename += '.json';
    } else if (format === 'csv') {
      // Convert to CSV format
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
      // Generate SQL INSERT statements
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

    // Download file
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

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b">
        <div>
          <h2 className="text-lg font-semibold">Synthetic Data Generation</h2>
          <p className="text-sm text-muted-foreground">
            Generate synthetic data using SDV (Synthetic Data Vault)
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
        <Tabs defaultValue="config" className="h-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="config">Configuration</TabsTrigger>
            <TabsTrigger value="results">Results</TabsTrigger>
            <TabsTrigger value="export">Export</TabsTrigger>
          </TabsList>

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
                    <Label htmlFor="sample_size">Sample Size</Label>
                    <Input
                      id="sample_size"
                      type="number"
                      min="100"
                      max="10000"
                      value={config.sample_size}
                      onChange={(e) => setConfig({ ...config, sample_size: parseInt(e.target.value) })}
                    />
                    <p className="text-xs text-muted-foreground mt-1">
                      Number of rows to generate per table
                    </p>
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
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="include_relationships"
                    checked={config.include_relationships}
                    onChange={(e) => setConfig({ ...config, include_relationships: e.target.checked })}
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
    </div>
  );
} 