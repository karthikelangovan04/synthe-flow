#!/bin/bash

# Azure Frontend Deployment Script (Simplified)
# This script deploys the built frontend to Azure Web App Service
# Environment variables are configured directly in Azure Web App

echo "🚀 Deploying Frontend to Azure Web App Service"
echo "================================================"
echo "📋 NOTE: Environment variables are set in Azure Web App Configuration"
echo ""

# Configuration - UPDATE THESE VALUES
RESOURCE_GROUP="your-us-east-resource-group-name"
WEBAPP_NAME="your-us-east-webapp-name"
SUBSCRIPTION_ID="your-subscription-id"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI is not installed. Please install it first."
    exit 1
fi

# Check if user is logged in
if ! az account show &> /dev/null; then
    echo "🔐 Please login to Azure first:"
    echo "   az login"
    exit 1
fi

# Set subscription
echo "📋 Setting subscription to: $SUBSCRIPTION_ID"
az account set --subscription $SUBSCRIPTION_ID

# Check if dist folder exists
if [ ! -d "dist" ]; then
    echo "❌ dist folder not found. Please run 'npm run build' first."
    exit 1
fi

# Create zip file for deployment
echo "📦 Creating deployment package..."
cd dist
zip -r ../frontend-deployment.zip .
cd ..

# Deploy to Azure
echo "🚀 Deploying to Azure Web App: $WEBAPP_NAME"
az webapp deployment source config-zip \
    --resource-group $RESOURCE_GROUP \
    --name $WEBAPP_NAME \
    --src frontend-deployment.zip

if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo "🌐 Your frontend is now available at:"
    echo "   https://$WEBAPP_NAME.azurewebsites.net"
    echo ""
    echo "⚠️  IMPORTANT: Set these environment variables in Azure Web App:"
    echo "   VITE_API_BASE_URL = https://cts-vibeappea2402-3.azurewebsites.net"
    echo "   VITE_SUPABASE_URL = https://mlwqzkcwnmjeyobixwap.supabase.co"
    echo "   VITE_SUPABASE_ANON_KEY = [your-supabase-key]"
    echo ""
    echo "📍 Go to: Azure Portal → Your Web App → Configuration → Application settings"
else
    echo "❌ Deployment failed!"
    exit 1
fi

# Clean up
echo "🧹 Cleaning up deployment files..."
rm -f frontend-deployment.zip

echo "🎉 Deployment complete!"
echo "🔗 Backend: https://cts-vibeappea2402-3.azurewebsites.net"
echo "🔗 Frontend: https://$WEBAPP_NAME.azurewebsites.net" 