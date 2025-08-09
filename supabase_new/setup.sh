#!/bin/bash

# Supabase Setup Script for Synthetic Data Platform
# This script helps set up the new Supabase configuration

set -e  # Exit on any error

echo "🚀 Setting up Supabase for Synthetic Data Platform"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Supabase CLI is installed
check_supabase_cli() {
    print_status "Checking Supabase CLI installation..."
    
    if ! command -v supabase &> /dev/null; then
        print_error "Supabase CLI is not installed. Please install it first:"
        echo "  npm install -g supabase"
        echo "  or visit: https://supabase.com/docs/guides/cli"
        exit 1
    fi
    
    SUPABASE_VERSION=$(supabase --version)
    print_success "Supabase CLI found: $SUPABASE_VERSION"
}

# Initialize Supabase project
init_supabase() {
    print_status "Initializing Supabase project..."
    
    if [ ! -f "config.toml" ]; then
        print_error "config.toml not found. Please run this script from the supabase_new directory."
        exit 1
    fi
    
    # Check if already initialized
    if [ -d ".temp" ]; then
        print_warning "Supabase project already initialized. Skipping initialization."
        return
    fi
    
    supabase init
    print_success "Supabase project initialized"
}

# Link to existing project
link_project() {
    print_status "Linking to Supabase project..."
    
    PROJECT_ID="mlwqzkcwnmjeyobixwap"
    
    # Check if already linked
    if supabase status &> /dev/null; then
        print_warning "Project already linked. Skipping linking."
        return
    fi
    
    supabase link --project-ref $PROJECT_ID
    print_success "Linked to project: $PROJECT_ID"
}

# Apply migrations
apply_migrations() {
    print_status "Applying database migrations..."
    
    # Check if migrations directory exists
    if [ ! -d "migrations" ]; then
        print_error "Migrations directory not found!"
        exit 1
    fi
    
    # Count migration files
    MIGRATION_COUNT=$(ls migrations/*.sql 2>/dev/null | wc -l)
    print_status "Found $MIGRATION_COUNT migration files"
    
    # Apply migrations
    supabase db push
    
    print_success "Migrations applied successfully"
}

# Deploy edge functions
deploy_functions() {
    print_status "Deploying edge functions..."
    
    # Check if functions directory exists
    if [ ! -d "functions" ]; then
        print_warning "Functions directory not found. Skipping function deployment."
        return
    fi
    
    # Deploy each function
    for func_dir in functions/*/; do
        if [ -d "$func_dir" ]; then
            func_name=$(basename "$func_dir")
            print_status "Deploying function: $func_name"
            supabase functions deploy "$func_name"
            print_success "Function $func_name deployed"
        fi
    done
}

# Setup API secrets
setup_api_secrets() {
    print_status "Setting up API secrets..."
    
    echo ""
    echo "To complete the setup, you need to add your API keys to the database."
    echo "You can do this through the Supabase dashboard or using the following SQL:"
    echo ""
    echo "UPDATE api_secrets SET key_value = 'your-actual-openai-api-key' WHERE key_name = 'OPENAI_API_KEY';"
    echo ""
    
    read -p "Do you want to open the Supabase dashboard? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        supabase dashboard
    fi
}

# Verify setup
verify_setup() {
    print_status "Verifying setup..."
    
    # Check if we can connect to the database
    if supabase status &> /dev/null; then
        print_success "Supabase connection verified"
    else
        print_error "Failed to verify Supabase connection"
        exit 1
    fi
    
    # Check if functions are deployed
    FUNCTIONS=$(supabase functions list 2>/dev/null | grep -c "llm-assistant" || echo "0")
    if [ "$FUNCTIONS" -gt 0 ]; then
        print_success "Edge functions verified"
    else
        print_warning "Edge functions may not be deployed"
    fi
}

# Main execution
main() {
    echo ""
    print_status "Starting Supabase setup..."
    
    # Check prerequisites
    check_supabase_cli
    
    # Initialize and setup
    init_supabase
    link_project
    
    # Apply database changes
    apply_migrations
    
    # Deploy functions
    deploy_functions
    
    # Setup API secrets
    setup_api_secrets
    
    # Verify everything
    verify_setup
    
    echo ""
    print_success "🎉 Supabase setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Add your API keys to the api_secrets table"
    echo "2. Test the LLM assistant function"
    echo "3. Verify your application integration"
    echo ""
    echo "For more information, see README.md"
    echo ""
}

# Run main function
main "$@" 