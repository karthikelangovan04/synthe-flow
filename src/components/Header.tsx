import { Button } from "@/components/ui/button";
import { Database, Menu, User, Layers } from "lucide-react";
import { Link, useLocation } from "react-router-dom";

export const Header = () => {
  const location = useLocation();
  
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        <div className="flex items-center space-x-2">
          <Link to="/" className="flex items-center space-x-2">
            <Database className="h-8 w-8 text-data-primary" />
            <span className="text-xl font-bold">SyntheticGen</span>
          </Link>
        </div>
        
        <nav className="hidden md:flex items-center space-x-6">
          <Link 
            to="/" 
            className={`text-sm font-medium hover:text-data-primary transition-colors ${
              location.pathname === '/' ? 'text-data-primary' : ''
            }`}
          >
            Home
          </Link>
          <Link 
            to="/schema-designer" 
            className={`text-sm font-medium hover:text-data-primary transition-colors flex items-center gap-2 ${
              location.pathname === '/schema-designer' ? 'text-data-primary' : ''
            }`}
          >
            <Layers className="h-4 w-4" />
            Schema Designer
          </Link>
          <a href="#" className="text-sm font-medium hover:text-data-primary transition-colors">
            Documentation
          </a>
          <a href="#" className="text-sm font-medium hover:text-data-primary transition-colors">
            Pricing
          </a>
        </nav>

        <div className="flex items-center space-x-4">
          <Button variant="data-ghost" size="sm">
            Sign In
          </Button>
          <Button variant="data" size="sm">
            Get Started
          </Button>
          <Button variant="ghost" size="icon" className="md:hidden">
            <Menu className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </header>
  );
};