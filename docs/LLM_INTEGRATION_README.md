# LLM-Powered Business Rules & Schema Import Features

## Overview

This document describes the new AI-powered features added to the Synthe-Flow schema designer that enable automatic generation of business rules, enhanced descriptions, and schema import capabilities.

## 🚀 New Features

### 1. AI-Powered Business Rules Generation

The schema designer now includes an AI assistant that can automatically generate:
- **Business Rules**: Domain-specific constraints and validation rules
- **Enhanced Descriptions**: Detailed explanations of table and column purposes
- **Data Generation Rules**: Guidelines for synthetic data creation
- **Validation Rules**: Data integrity and format validation requirements

### 2. Schema Import from JSON/CSV

Users can now import existing database schemas from:
- **JSON Format**: Structured schema definitions
- **CSV Format**: Tabular schema representations
- **Manual Entry**: Step-by-step schema creation

### 3. LLM Chat Interface

An interactive chat interface allows users to:
- Ask questions about business rules
- Request enhanced descriptions
- Generate data generation strategies
- Get validation recommendations

## 🏗️ Architecture

### Database Schema Updates

New fields added to existing tables:

```sql
-- Column metadata enhancements
ALTER TABLE public.column_metadata 
ADD COLUMN business_rules TEXT,
ADD COLUMN enhanced_description TEXT,
ADD COLUMN data_generation_rules TEXT,
ADD COLUMN validation_rules TEXT;

-- Table metadata enhancements
ALTER TABLE public.table_metadata 
ADD COLUMN business_rules TEXT,
ADD COLUMN enhanced_description TEXT,
ADD COLUMN domain_context TEXT;

-- LLM conversation history
CREATE TABLE public.llm_conversations (
  id UUID PRIMARY KEY,
  project_id UUID REFERENCES public.projects(id),
  table_id UUID REFERENCES public.table_metadata(id),
  column_id UUID REFERENCES public.column_metadata(id),
  conversation_type TEXT,
  messages JSONB,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);
```

### Component Structure

```
src/
├── lib/
│   └── llm-service.ts              # LLM integration service
├── components/schema-designer/
│   ├── BusinessRulesChat.tsx       # AI chat interface
│   ├── SchemaImport.tsx            # Schema import component
│   └── TableEditor.tsx             # Enhanced table editor
└── pages/
    └── SchemaDesigner.tsx          # Main schema designer page
```

## 🔧 Implementation Details

### LLM Service (`llm-service.ts`)

The LLM service provides:

```typescript
interface BusinessRuleContext {
  tableName: string;
  columnName?: string;
  dataType?: string;
  isNullable?: boolean;
  isPrimaryKey?: boolean;
  isUnique?: boolean;
  existingDescription?: string;
  domainContext?: string;
}

interface GeneratedBusinessRules {
  businessRules: string;
  enhancedDescription: string;
  dataGenerationRules?: string;
  validationRules?: string;
}
```

**Key Methods:**
- `generateBusinessRules(context)`: Generate comprehensive business rules
- `saveConversation()`: Store chat history
- `getConversationHistory()`: Retrieve previous conversations

### Business Rules Chat (`BusinessRulesChat.tsx`)

Features:
- Real-time chat interface with AI assistant
- Suggested prompts for common questions
- Automatic business rule generation
- Conversation history persistence
- Context-aware responses based on table/column metadata

### Schema Import (`SchemaImport.tsx`)

Supports multiple import formats:

**JSON Format:**
```json
{
  "tables": [
    {
      "name": "users",
      "description": "User account information",
      "columns": [
        {
          "name": "id",
          "dataType": "uuid",
          "isPrimaryKey": true,
          "description": "Unique user identifier"
        }
      ]
    }
  ]
}
```

**CSV Format:**
```csv
table_name,column_name,data_type,is_nullable,is_primary_key,column_description
users,id,uuid,false,true,Unique user identifier
users,email,varchar,false,false,User email address
```

### Enhanced Table Editor

The table editor now includes three tabs:

1. **Structure**: Traditional table and column editing
2. **Business Rules**: View generated business rules and validation
3. **AI Assistant**: Interactive chat interface for rule generation

## 🎯 Usage Examples

### 1. Import Existing Schema

1. Click "Import Schema" button in the header
2. Choose JSON or CSV format
3. Paste or upload your schema
4. Review parsed tables and columns
5. Generate business rules automatically
6. Import to schema designer

### 2. Generate Business Rules

1. Select a table in the schema designer
2. Navigate to "AI Assistant" tab
3. Ask questions like:
   - "What business rules should apply to this table?"
   - "Generate validation rules for email columns"
   - "How should synthetic data be generated?"
4. Review and save generated rules

### 3. Enhanced Column Management

1. View AI-enhanced descriptions in the structure tab
2. See business rules badges on columns with AI-generated rules
3. Access detailed rules in the business rules tab
4. Modify and customize generated rules as needed

## 🔌 LLM Integration

### Current Implementation

The current implementation uses a mock LLM service that provides realistic responses based on context. To integrate with real LLM providers:

1. **OpenAI Integration:**
```typescript
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function callLLM(messages: LLMMessage[]): Promise<string> {
  const response = await openai.chat.completions.create({
    model: "gpt-4",
    messages: messages,
    temperature: 0.7,
  });
  
  return response.choices[0].message.content || '';
}
```

2. **Anthropic Claude Integration:**
```typescript
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

async function callLLM(messages: LLMMessage[]): Promise<string> {
  const response = await anthropic.messages.create({
    model: "claude-3-sonnet-20240229",
    max_tokens: 1000,
    messages: messages,
  });
  
  return response.content[0].text;
}
```

### Environment Variables

Add to your `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## 🚀 Future Enhancements

### Planned Features

1. **Multi-Modal Support**: Import schemas from images/diagrams
2. **Advanced AI Models**: Integration with domain-specific models
3. **Rule Templates**: Pre-built business rule templates
4. **Collaborative Editing**: Real-time collaboration on business rules
5. **Version Control**: Track changes to business rules over time
6. **Export Capabilities**: Export business rules to documentation formats

### Performance Optimizations

1. **Caching**: Cache LLM responses for similar contexts
2. **Batch Processing**: Generate rules for multiple columns simultaneously
3. **Streaming**: Real-time streaming of LLM responses
4. **Offline Mode**: Work with cached rules when offline

## 🧪 Testing

### Unit Tests

```typescript
// Test business rule generation
describe('LLM Service', () => {
  it('should generate business rules for user table', async () => {
    const context: BusinessRuleContext = {
      tableName: 'users',
      columnName: 'email',
      dataType: 'varchar',
      isNullable: false,
      isUnique: true
    };
    
    const rules = await llmService.generateBusinessRules(context);
    expect(rules.businessRules).toContain('email');
    expect(rules.enhancedDescription).toBeTruthy();
  });
});
```

### Integration Tests

```typescript
// Test schema import
describe('Schema Import', () => {
  it('should parse JSON schema correctly', () => {
    const jsonSchema = '{"tables": [{"name": "users", "columns": []}]}';
    const tables = parseJsonSchema(jsonSchema);
    expect(tables).toHaveLength(1);
    expect(tables[0].name).toBe('users');
  });
});
```

## 📚 API Reference

### LLM Service Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `generateBusinessRules` | Generate business rules for table/column | `BusinessRuleContext` | `GeneratedBusinessRules` |
| `saveConversation` | Save chat conversation | `projectId`, `tableId`, `columnId`, `conversationType`, `messages` | `Promise<void>` |
| `getConversationHistory` | Get conversation history | `projectId`, `tableId?`, `columnId?` | `Promise<LLMMessage[]>` |

### Component Props

#### BusinessRulesChat
```typescript
interface BusinessRulesChatProps {
  projectId: string;
  tableId: string | null;
  columnId: string | null;
  tableName: string;
  columnName?: string;
  dataType?: string;
  isNullable?: boolean;
  isPrimaryKey?: boolean;
  isUnique?: boolean;
  existingDescription?: string;
  onBusinessRulesGenerated: (rules: GeneratedBusinessRules) => void;
}
```

#### SchemaImport
```typescript
interface SchemaImportProps {
  projectId: string;
  onSchemaImported: (tables: SchemaTable[]) => void;
}
```

## 🤝 Contributing

To contribute to the LLM integration features:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support with the LLM integration features:

1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information
4. Contact the development team

---

**Note**: This implementation provides a foundation for AI-powered schema design. The mock LLM service can be easily replaced with real LLM providers for production use. 