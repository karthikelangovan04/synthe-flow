#!/bin/bash

echo "🔧 Synthetic Data Platform Environment Setup"
echo "============================================="
echo ""

# Check if env.development exists
if [ ! -f "env.development" ]; then
    echo "❌ env.development file not found!"
    exit 1
fi

echo "📋 Current environment configuration:"
echo ""

# Display current values
echo "Current VITE_SUPABASE_URL:"
grep "VITE_SUPABASE_URL" env.development
echo ""

echo "Current VITE_SUPABASE_ANON_KEY:"
grep "VITE_SUPABASE_ANON_KEY" env.development
echo ""

echo "🔍 Checking if Supabase credentials are configured..."
echo ""

# Check if using placeholder values
if grep -q "your-project-id.supabase.co" env.development; then
    echo "❌ VITE_SUPABASE_URL is still using placeholder value"
    echo "   Please update it with your actual Supabase project URL"
    echo ""
fi

if grep -q "your-actual-anon-key-here" env.development; then
    echo "❌ VITE_SUPABASE_ANON_KEY is still using placeholder value"
    echo "   Please update it with your actual Supabase anon key"
    echo ""
fi

echo "📝 To fix the sign-in issue, you need to:"
echo ""
echo "1. Go to https://supabase.com/dashboard"
echo "2. Select your project (or create a new one)"
echo "3. Go to Settings > API"
echo "4. Copy the 'Project URL' and 'anon public' key"
echo "5. Edit env.development and replace the placeholder values"
echo ""
echo "Example of correct values:"
echo "VITE_SUPABASE_URL=https://abcdefghijklmnop.supabase.co"
echo "VITE_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
echo ""

echo "💡 Quick setup commands:"
echo "nano env.development  # Edit the file"
echo "./start_simple.sh     # Start services after updating"
echo ""

echo "🔍 To test if everything is working:"
echo "1. Update the environment variables"
echo "2. Start services: ./start_simple.sh"
echo "3. Open: http://localhost:3000/test_env.html"
echo "4. Try signing in to the application" 