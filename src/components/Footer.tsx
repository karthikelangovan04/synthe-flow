import { Database, Github, Twitter, Linkedin } from "lucide-react";

export const Footer = () => {
  return (
    <footer className="bg-card border-t">
      <div className="container px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Database className="h-6 w-6 text-data-primary" />
              <span className="text-lg font-bold">SyntheticGen</span>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Enterprise-grade synthetic data generation platform for maintaining data relationships
              and automating QA workflows at scale.
            </p>
            <div className="flex space-x-4">
              <Github className="h-5 w-5 text-muted-foreground hover:text-data-primary cursor-pointer transition-colors" />
              <Twitter className="h-5 w-5 text-muted-foreground hover:text-data-primary cursor-pointer transition-colors" />
              <Linkedin className="h-5 w-5 text-muted-foreground hover:text-data-primary cursor-pointer transition-colors" />
            </div>
          </div>
          
          <div>
            <h3 className="font-semibold mb-4">Platform</h3>
            <ul className="space-y-3 text-sm text-muted-foreground">
              <li className="hover:text-data-primary cursor-pointer transition-colors">Schema Recognition</li>
              <li className="hover:text-data-primary cursor-pointer transition-colors">Data Generation</li>
              <li className="hover:text-data-primary cursor-pointer transition-colors">Transformation Rules</li>
              <li className="hover:text-data-primary cursor-pointer transition-colors">API Access</li>
            </ul>
          </div>
          
          <div>
            <h3 className="font-semibold mb-4">Solutions</h3>
            <ul className="space-y-3 text-sm text-muted-foreground">
              <li className="hover:text-data-primary cursor-pointer transition-colors">QA & Testing</li>
              <li className="hover:text-data-primary cursor-pointer transition-colors">Development</li>
              <li className="hover:text-data-primary cursor-pointer transition-colors">Data Privacy</li>
              <li className="hover:text-data-primary cursor-pointer transition-colors">Compliance</li>
            </ul>
          </div>
          
          <div>
            <h3 className="font-semibold mb-4">Resources</h3>
            <ul className="space-y-3 text-sm text-muted-foreground">
              <li className="hover:text-data-primary cursor-pointer transition-colors">Documentation</li>
              <li className="hover:text-data-primary cursor-pointer transition-colors">API Reference</li>
              <li className="hover:text-data-primary cursor-pointer transition-colors">Support</li>
              <li className="hover:text-data-primary cursor-pointer transition-colors">Community</li>
            </ul>
          </div>
        </div>
        
        <div className="border-t mt-12 pt-8 flex flex-col md:flex-row justify-between items-center">
          <p className="text-sm text-muted-foreground">
            © 2024 SyntheticGen. All rights reserved.
          </p>
          <div className="flex space-x-6 text-sm text-muted-foreground mt-4 md:mt-0">
            <span className="hover:text-data-primary cursor-pointer transition-colors">Privacy Policy</span>
            <span className="hover:text-data-primary cursor-pointer transition-colors">Terms of Service</span>
            <span className="hover:text-data-primary cursor-pointer transition-colors">Security</span>
          </div>
        </div>
      </div>
    </footer>
  );
};