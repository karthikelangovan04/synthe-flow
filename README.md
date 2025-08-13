# SyntheticGen - AI-Powered Synthetic Data Generation

A comprehensive web application for generating high-quality synthetic data using AI, built with modern web technologies and an intuitive user interface.

## 🚀 Features

### Core Pages & Functionality
- **Landing Page** - Modern hero section showcasing SyntheticGen's capabilities with feature highlights and use cases
- **Getting Started** - Step-by-step guide with code examples and quick start instructions
- **Schema Generator** - Interactive tool for creating, validating, and testing JSON schemas with real-time data generation
- **API Reference** - Complete documentation of the Python API with detailed examples and parameter descriptions
- **Examples** - Real-world schema templates for various industries including HR, E-commerce, Finance, and Healthcare

### Key Features
- **Multiple Data Domains** - Support for various data types including personal, financial, e-commerce, and healthcare data
- **JSON Schema Based** - Industry-standard JSON Schema format for data validation and generation
- **High Quality Output** - AI-powered generation ensuring realistic and coherent synthetic data
- **Interactive Schema Editor** - Real-time schema validation and data preview
- **Export Capabilities** - Download generated data in multiple formats
- **Responsive Design** - Modern UI that works seamlessly across desktop and mobile devices

### Design & User Experience
- **Modern Purple Theme** - Elegant gradient-based design with dark/light mode support
- **Intuitive Navigation** - Collapsible sidebar with easy access to all features
- **Interactive Components** - Real-time feedback and smooth animations
- **Accessibility** - Built with accessibility best practices using Radix UI components

## 🛠️ Technology Stack

### Frontend
- **React 18** - Modern React with hooks and functional components
- **TypeScript** - Type-safe development with excellent developer experience
- **Vite** - Fast build tool and development server
- **Tailwind CSS** - Utility-first CSS framework with custom design system
- **shadcn/ui** - High-quality, accessible UI components built on Radix UI

### Key Libraries
- **React Router** - Client-side routing for single-page application
- **React Hook Form** - Performant forms with easy validation
- **Lucide React** - Beautiful, customizable icons
- **Recharts** - Responsive chart library for data visualization
- **Sonner** - Modern toast notifications

### Backend & Infrastructure

#### Dual Backend Architecture
SyntheticGen features two sophisticated backend implementations for different use cases:

##### 1. SDV Framework Backend (`/backend`)
- **Open Source SDV Integration** - Built on the proven Synthetic Data Vault framework
- **Multi-table Relationships** - HMASynthesizer for complex relational data generation
- **FastAPI Service** - High-performance Python REST API with async capabilities
- **Relationship Management** - Drag-and-drop interface for defining table relationships
- **Quality Metrics** - Comprehensive validation and quality assessment
- **Multiple Export Formats** - CSV, JSON, SQL export capabilities
- **Azure Deployment Ready** - Production-ready deployment configuration

##### 2. Enhanced Custom Backend (`/enhanced_backend`)
- **Neural Network Architecture** - Custom-built from scratch without SDV dependencies
- **Conditional VAE** - Conditional Variational Autoencoders for high-quality data generation
- **Graph Neural Networks** - Advanced relationship modeling for complex data structures
- **Privacy-Preserving** - Built-in differential privacy and anonymization techniques
- **Enterprise Export Engine** - Support for JSON, CSV, Excel, SQL, Parquet, XML formats
- **Parallel Processing** - Optimized for large-scale data generation with multi-threading
- **Quality Validation** - Comprehensive data validation and integrity checking
- **Performance Optimization** - Advanced caching and memory management

#### Frontend Integration Platform
**Supabase** - Complete backend-as-a-service platform providing:
- **Database** - PostgreSQL database for data persistence
- **Authentication** - User management and authentication system
- **Real-time** - Live data synchronization and subscriptions
- **File Storage** - Secure file upload and management
- **Edge Functions** - Serverless functions for custom logic
- **API Integration** - RESTful and GraphQL APIs for external services

## 🏃‍♂️ Quick Start

### Prerequisites
- Node.js (v18 or higher)
- npm or yarn package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <YOUR_GIT_URL>
   cd <YOUR_PROJECT_NAME>
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

4. **Open your browser** - Navigate to `http://localhost:5173` to see the application

### Building for Production
```bash
npm run build
npm run preview
```

## 📚 Documentation

For detailed setup and usage instructions, see our comprehensive documentation:
- [Enhanced Backend Setup Guide](ENHANCED_BACKEND_SETUP_GUIDE.md)
- [Quick Start Guide](QUICK_START_ENHANCED_BACKEND.md)
- [Referential Integrity Improvements](REFERENTIAL_INTEGRITY_IMPROVEMENTS.md)

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
