import { Button } from "@/components/ui/button";
import { Database, Menu, User, Layers, LogOut } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { useAuth } from "@/hooks/useAuth";

export const Header = () => {
  const location = useLocation();
  const { user, signOut } = useAuth();
  
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
          {user ? (
            <>
              <span className="text-sm text-muted-foreground hidden md:inline">
                Welcome, {user.email}
              </span>
              <Button variant="data-ghost" size="sm" onClick={signOut}>
                <LogOut className="h-4 w-4 md:mr-2" />
                <span className="hidden md:inline">Sign Out</span>
              </Button>
            </>
          ) : (
            <>
              <Button variant="data-ghost" size="sm" asChild>
                <Link to="/auth">Sign In</Link>
              </Button>
              <Button variant="data" size="sm" asChild>
                <Link to="/auth">Get Started</Link>
              </Button>
            </>
          )}
          <Button variant="ghost" size="icon" className="md:hidden">
            <Menu className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </header>
  );
};