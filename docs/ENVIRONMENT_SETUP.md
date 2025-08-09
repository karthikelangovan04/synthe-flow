# Environment Configuration Guide

This guide explains how to configure the synthetic data platform for different environments (development, staging, production).

## 🚀 Quick Start

### 1. Development Environment

```bash
# Copy the example environment file
cp env.example .env.local

# Edit the environment variables
nano .env.local

# Start all services with environment configuration
./start_dev.sh
```

### 2. Production Environment

```bash
# Set production environment variables
export VITE_API_BASE_URL="https://your-backend.azurewebsites.net"
export VITE_ENHANCED_API_BASE_URL="https://your-enhanced-backend.azurewebsites.net"
export VITE_SUPABASE_URL="https://your-project.supabase.co"
export VITE_SUPABASE_ANON_KEY="your-production-anon-key"

# Start services
npm run build
python backend/sdv_service/main.py
python enhanced_backend/enhanced_sdv_service/main.py
```

## 📋 Environment Variables

### Frontend Variables (Vite)

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `VITE_API_BASE_URL` | Original backend API URL | `http://localhost:8002` | Yes |
| `VITE_ENHANCED_API_BASE_URL` | Enhanced backend API URL | `http://localhost:8003` | Yes |
| `VITE_SUPABASE_URL` | Supabase project URL | - | Yes |
| `VITE_SUPABASE_ANON_KEY` | Supabase anonymous key | - | Yes |

### Backend Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `BACKEND_HOST` | Original backend host | `0.0.0.0` | No |
| `BACKEND_PORT` | Original backend port | `8002` | No |
| `ENHANCED_BACKEND_HOST` | Enhanced backend host | `0.0.0.0` | No |
| `ENHANCED_BACKEND_PORT` | Enhanced backend port | `8003` | No |

### Database Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | - | Yes |
| `SUPABASE_URL` | Supabase project URL | - | Yes |
| `SUPABASE_ANON_KEY` | Supabase anonymous key | - | Yes |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key | - | Yes |
| `SUPABASE_ACCESS_TOKEN` | Supabase access token | - | No |

## 🔧 Configuration Files

### 1. `env.example`
Template file showing all available environment variables.

### 2. `env.development`
Development environment configuration with localhost URLs.

### 3. `.env.local` (recommended)
Local development environment file (not committed to git).

### 4. `.env.production`
Production environment configuration.

## 🌍 Environment-Specific Configurations

### Development
```env
VITE_API_BASE_URL=http://localhost:8002
VITE_ENHANCED_API_BASE_URL=http://localhost:8003
VITE_SUPABASE_URL=https://dev-project.supabase.co
VITE_SUPABASE_ANON_KEY=dev-anon-key
NODE_ENV=development
```

### Staging
```env
VITE_API_BASE_URL=https://staging-backend.azurewebsites.net
VITE_ENHANCED_API_BASE_URL=https://staging-enhanced-backend.azurewebsites.net
VITE_SUPABASE_URL=https://staging-project.supabase.co
VITE_SUPABASE_ANON_KEY=staging-anon-key
NODE_ENV=staging
```

### Production
```env
VITE_API_BASE_URL=https://prod-backend.azurewebsites.net
VITE_ENHANCED_API_BASE_URL=https://prod-enhanced-backend.azurewebsites.net
VITE_SUPABASE_URL=https://prod-project.supabase.co
VITE_SUPABASE_ANON_KEY=prod-anon-key
NODE_ENV=production
```

## 🚀 Startup Scripts

### Development Startup
```bash
./start_dev.sh
```
- Loads environment variables from `env.development`
- Starts all services (frontend, backends)
- Handles graceful shutdown

### Manual Startup
```bash
# Terminal 1: Original Backend
cd backend
source venv/bin/activate
cd sdv_service
python main.py

# Terminal 2: Enhanced Backend
cd enhanced_backend
source venv/bin/activate
cd enhanced_sdv_service
python main.py

# Terminal 3: Frontend
npm run dev
```

## 🔒 Security Considerations

### Environment Variables
- Never commit `.env.local` or production environment files
- Use Azure Key Vault or similar for production secrets
- Rotate API keys regularly

### Supabase Configuration
- Use different projects for dev/staging/production
- Configure Row Level Security (RLS) policies
- Set up proper authentication

## 🐛 Troubleshooting

### Common Issues

#### 1. Environment Variables Not Loading
```bash
# Check if variables are loaded
echo $VITE_API_BASE_URL

# Manually load from file
source env.development
```

#### 2. Backend Connection Issues
```bash
# Test backend health
curl http://localhost:8002/api/sdv/health
curl http://localhost:8003/api/enhanced/health
```

#### 3. Supabase Connection Issues
```bash
# Check Supabase configuration
echo $VITE_SUPABASE_URL
echo $VITE_SUPABASE_ANON_KEY
```

### Debug Mode
```bash
# Enable debug logging
export DEBUG=true
export VITE_DEBUG=true

# Start with debug output
./start_dev.sh
```

## 📊 Monitoring

### Health Checks
- Frontend: `http://localhost:3000`
- Original Backend: `http://localhost:8002/api/sdv/health`
- Enhanced Backend: `http://localhost:8003/api/enhanced/health`

### Logs
- Frontend: Browser console
- Backend: Terminal output
- Enhanced Backend: Terminal output

## 🔄 Migration from Hardcoded URLs

The application has been updated to use environment variables instead of hardcoded localhost URLs. All API calls now use the configuration utility:

```typescript
import { config } from '@/lib/config';

// Instead of hardcoded URLs
const response = await fetch('http://localhost:8002/api/sdv/generate', ...);

// Use configuration
const response = await fetch(config.endpoints.generateSyntheticData, ...);
```

## 📝 Next Steps

1. **Set up your Supabase project** and update the environment variables
2. **Configure your production environment** for Azure deployment
3. **Set up monitoring and logging** for production
4. **Configure CI/CD pipelines** with proper environment management 