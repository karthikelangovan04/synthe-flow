import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowRight, Database, Shield, Zap, Layers } from "lucide-react";
import { Link } from "react-router-dom";
import heroImage from "@/assets/hero-data-platform.jpg";

export const Hero = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-hero" />
      <div 
        className="absolute inset-0 opacity-30"
        style={{
          backgroundImage: `url(${heroImage})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
        }}
      />
      
      {/* Content */}
      <div className="relative z-10 container text-center px-4">
        <Badge variant="secondary" className="mb-6 bg-data-surface border-data-primary/20">
          <Zap className="w-3 h-3 mr-1" />
          Now Supporting PostgreSQL & Advanced Schema Recognition
        </Badge>
        
        <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 bg-gradient-primary bg-clip-text text-transparent">
          Generate Synthetic Data
          <br />
          at Enterprise Scale
        </h1>
        
        <p className="text-lg md:text-xl text-muted-foreground mb-8 max-w-3xl mx-auto leading-relaxed">
          Maintain data-entity relationships, column-level profiles, and distribution patterns
          while automating synthetic data generation for QA, testing, and development environments.
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
          <Link to="/auth">
            <Button variant="data" size="lg" className="group">
              <Layers className="w-4 h-4 mr-2" />
              Get Started
              <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
            </Button>
          </Link>
          <Button variant="data-outline" size="lg">
            View Documentation
          </Button>
        </div>
        
        {/* Feature highlights */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
          <div className="flex items-center justify-center space-x-3 p-4 rounded-lg bg-card/50 backdrop-blur-sm border">
            <Database className="w-5 h-5 text-data-primary" />
            <span className="text-sm font-medium">Schema-Driven Generation</span>
          </div>
          <div className="flex items-center justify-center space-x-3 p-4 rounded-lg bg-card/50 backdrop-blur-sm border">
            <Shield className="w-5 h-5 text-data-primary" />
            <span className="text-sm font-medium">Data Security Compliance</span>
          </div>
          <div className="flex items-center justify-center space-x-3 p-4 rounded-lg bg-card/50 backdrop-blur-sm border">
            <Zap className="w-5 h-5 text-data-primary" />
            <span className="text-sm font-medium">Automated Workflows</span>
          </div>
        </div>
      </div>
    </section>
  );
};