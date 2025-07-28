import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { supabase } from '@/integrations/supabase/client';
import { Plus, FolderOpen, Calendar } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface Project {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

interface ProjectListProps {
  projects: Project[];
  selectedProjectId: string | null;
  onProjectSelect: (projectId: string) => void;
  showNewProject: boolean;
  onProjectCreated: () => void;
  onCancelNewProject: () => void;
}

export function ProjectList({
  projects,
  selectedProjectId,
  onProjectSelect,
  showNewProject,
  onProjectCreated,
  onCancelNewProject,
}: ProjectListProps) {
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDescription, setNewProjectDescription] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const { toast } = useToast();

  const handleCreateProject = async () => {
    if (!newProjectName.trim()) {
      toast({
        title: "Error",
        description: "Project name is required",
        variant: "destructive",
      });
      return;
    }

    setIsCreating(true);
    try {
      const { error } = await supabase
        .from('projects')
        .insert({
          name: newProjectName.trim(),
          description: newProjectDescription.trim() || null,
        });

      if (error) throw error;

      toast({
        title: "Success",
        description: "Project created successfully",
      });

      setNewProjectName('');
      setNewProjectDescription('');
      onProjectCreated();
    } catch (error) {
      console.error('Error creating project:', error);
      toast({
        title: "Error",
        description: "Failed to create project",
        variant: "destructive",
      });
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <div className="h-full flex flex-col border-r bg-muted/30">
      <div className="p-4 border-b">
        <h2 className="text-lg font-semibold mb-2">Projects</h2>
        
        {showNewProject && (
          <Card className="mb-4">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">New Project</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Input
                placeholder="Project name"
                value={newProjectName}
                onChange={(e) => setNewProjectName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleCreateProject()}
              />
              <Textarea
                placeholder="Description (optional)"
                value={newProjectDescription}
                onChange={(e) => setNewProjectDescription(e.target.value)}
                className="min-h-[60px]"
              />
              <div className="flex gap-2">
                <Button
                  size="sm"
                  onClick={handleCreateProject}
                  disabled={isCreating}
                  className="flex-1"
                >
                  {isCreating ? 'Creating...' : 'Create'}
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={onCancelNewProject}
                  disabled={isCreating}
                >
                  Cancel
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-2">
        {projects.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <FolderOpen className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p className="text-sm">No projects yet</p>
            <p className="text-xs">Create your first project to get started</p>
          </div>
        ) : (
          projects.map((project) => (
            <Card
              key={project.id}
              className={`cursor-pointer transition-colors hover:bg-accent ${
                selectedProjectId === project.id ? 'bg-accent border-primary' : ''
              }`}
              onClick={() => onProjectSelect(project.id)}
            >
              <CardContent className="p-3">
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-sm truncate">{project.name}</h3>
                    {project.description && (
                      <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                        {project.description}
                      </p>
                    )}
                    <div className="flex items-center mt-2 text-xs text-muted-foreground">
                      <Calendar className="h-3 w-3 mr-1" />
                      {new Date(project.updated_at).toLocaleDateString()}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
}