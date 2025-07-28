import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { FileCode, Settings, Upload, Download, ArrowRight } from "lucide-react";

export const UseCases = () => {
  const useCases = [
    {
      id: "UC1",
      title: "Schema-Driven Synthetic Data",
      description: "Generate synthetic data by taking user inputs like functional definition/pattern, schema, fields, and sample data.",
      icon: FileCode,
      inputs: ["Functional attributes", "Metadata in JSON file", "Sample data in CSV"],
      deliverables: ["CSV file containing synthetic data", "UI screen with login page", "Metadata upload interface"],
      features: ["Automated schema recognition", "Pattern-based generation", "Relationship preservation"]
    },
    {
      id: "UC2", 
      title: "Customizable Data Generation",
      description: "Customize synthetic data during generation stages (Pre, During, Post) using predefined transformation processes and rules.",
      icon: Settings,
      inputs: ["Metadata in JSON file", "Data transformation policies", "Custom rules"],
      deliverables: ["UI for uploading metadata", "Transformation policy selection", "Customized synthetic data output"],
      features: ["Multi-stage customization", "Policy-driven transforms", "Rule-based constraints"]
    }
  ];

  return (
    <section className="py-24 bg-gradient-surface">
      <div className="container px-4">
        <div className="text-center mb-16">
          <Badge variant="outline" className="mb-4">Use Cases</Badge>
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Powerful Data Generation Workflows
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Two comprehensive use cases designed to handle enterprise-scale synthetic data generation
            with maximum flexibility and control.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {useCases.map((useCase) => (
            <Card key={useCase.id} className="shadow-card-custom hover:shadow-data transition-shadow duration-300">
              <CardHeader>
                <div className="flex items-center space-x-3 mb-3">
                  <div className="p-2 rounded-lg bg-data-primary/10">
                    <useCase.icon className="w-5 h-5 text-data-primary" />
                  </div>
                  <Badge variant="secondary">{useCase.id}</Badge>
                </div>
                <CardTitle className="text-xl">{useCase.title}</CardTitle>
                <CardDescription className="text-base leading-relaxed">
                  {useCase.description}
                </CardDescription>
              </CardHeader>
              
              <CardContent className="space-y-6">
                <div>
                  <h4 className="font-semibold text-sm text-data-primary mb-3 flex items-center">
                    <Upload className="w-4 h-4 mr-2" />
                    Inputs
                  </h4>
                  <ul className="space-y-2">
                    {useCase.inputs.map((input, index) => (
                      <li key={index} className="text-sm text-muted-foreground flex items-center">
                        <div className="w-1.5 h-1.5 rounded-full bg-data-primary mr-3" />
                        {input}
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4 className="font-semibold text-sm text-data-primary mb-3 flex items-center">
                    <Download className="w-4 h-4 mr-2" />
                    Key Deliverables
                  </h4>
                  <ul className="space-y-2">
                    {useCase.deliverables.map((deliverable, index) => (
                      <li key={index} className="text-sm text-muted-foreground flex items-center">
                        <div className="w-1.5 h-1.5 rounded-full bg-data-accent mr-3" />
                        {deliverable}
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4 className="font-semibold text-sm text-data-primary mb-3">Features</h4>
                  <div className="flex flex-wrap gap-2">
                    {useCase.features.map((feature, index) => (
                      <Badge key={index} variant="outline" className="text-xs">
                        {feature}
                      </Badge>
                    ))}
                  </div>
                </div>

                <Button variant="data-outline" className="w-full group">
                  Learn More About {useCase.id}
                  <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};