import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Database, FileText, Code, Layers } from "lucide-react";

export const Architecture = () => {
  const components = [
    {
      title: "Meta Data",
      description: "Metadata for tables representing functional data elements",
      icon: Layers,
      deliverable: "Configuration files (JSON) containing the column metadata",
      color: "text-data-primary"
    },
    {
      title: "Test Data",
      description: "Set up test data for the tables",
      icon: Database,
      deliverable: "RDS (PostgreSQL) containing test data for key tables",
      color: "text-data-secondary"
    },
    {
      title: "SQL Scripts",
      description: "DDL & DML SQL scripts to create / modify the tables and data",
      icon: Code,
      deliverable: "Creation of SQL scripts manually",
      color: "text-data-accent"
    }
  ];

  return (
    <section className="py-24">
      <div className="container px-4">
        <div className="text-center mb-16">
          <Badge variant="outline" className="mb-4">System Architecture</Badge>
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Core Platform Components
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Built on a robust foundation of metadata management, test data infrastructure,
            and automated SQL generation for enterprise-grade synthetic data workflows.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto mb-16">
          {components.map((component, index) => (
            <Card key={index} className="text-center shadow-card-custom hover:shadow-data transition-all duration-300 hover:-translate-y-1">
              <CardHeader>
                <div className="mx-auto p-3 rounded-lg bg-gradient-surface w-fit mb-4">
                  <component.icon className={`w-8 h-8 ${component.color}`} />
                </div>
                <CardTitle className="text-xl">{component.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4 leading-relaxed">
                  {component.description}
                </p>
                <div className="p-3 rounded-lg bg-data-surface border border-data-primary/20">
                  <h4 className="font-semibold text-sm mb-2 text-data-primary">Key Deliverable</h4>
                  <p className="text-sm text-muted-foreground">
                    {component.deliverable}
                  </p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Data Flow Visualization */}
        <div className="max-w-4xl mx-auto">
          <Card className="bg-gradient-surface border-data-primary/20">
            <CardHeader>
              <CardTitle className="text-center flex items-center justify-center">
                <FileText className="w-5 h-5 mr-2 text-data-primary" />
                Data Generation Flow
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0 md:space-x-8">
                <div className="flex-1 text-center">
                  <div className="w-12 h-12 rounded-full bg-data-primary text-primary-foreground flex items-center justify-center mx-auto mb-3 font-bold">
                    1
                  </div>
                  <h3 className="font-semibold mb-2">Schema Input</h3>
                  <p className="text-sm text-muted-foreground">Upload metadata and define functional patterns</p>
                </div>
                
                <div className="hidden md:block w-8 h-px bg-data-primary/30" />
                
                <div className="flex-1 text-center">
                  <div className="w-12 h-12 rounded-full bg-data-secondary text-primary-foreground flex items-center justify-center mx-auto mb-3 font-bold">
                    2
                  </div>
                  <h3 className="font-semibold mb-2">Processing</h3>
                  <p className="text-sm text-muted-foreground">Apply transformations and maintain relationships</p>
                </div>
                
                <div className="hidden md:block w-8 h-px bg-data-primary/30" />
                
                <div className="flex-1 text-center">
                  <div className="w-12 h-12 rounded-full bg-data-accent text-primary-foreground flex items-center justify-center mx-auto mb-3 font-bold">
                    3
                  </div>
                  <h3 className="font-semibold mb-2">Output</h3>
                  <p className="text-sm text-muted-foreground">Generate high-quality synthetic data</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
};