import { supabase } from '@/integrations/supabase/client';

export interface LLMMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface BusinessRuleContext {
  tableName: string;
  columnName?: string;
  dataType?: string;
  isNullable?: boolean;
  isPrimaryKey?: boolean;
  isUnique?: boolean;
  existingDescription?: string;
  domainContext?: string;
}

export interface GeneratedBusinessRules {
  businessRules: string;
  enhancedDescription: string;
  dataGenerationRules?: string;
  validationRules?: string;
}

class LLMService {
  private async getOpenAIKey(): Promise<string | null> {
    try {
      // Try to get from environment variables first (for development)
      if (import.meta.env.VITE_OPENAI_API_KEY) {
        return import.meta.env.VITE_OPENAI_API_KEY;
      }

      // If not in env, try to get from Supabase secrets table
      const { data, error } = await supabase
        .from('api_secrets')
        .select('key_value')
        .eq('key_name', 'OPENAI_API_KEY')
        .single();

      if (error) {
        console.warn('Could not fetch API key from Supabase:', error);
        return null;
      }

      return data?.key_value || null;
    } catch (error) {
      console.warn('Error fetching OpenAI API key:', error);
      return null;
    }
  }

  private async callOpenAI(messages: LLMMessage[]): Promise<string> {
    const apiKey = await this.getOpenAIKey();
    
    if (!apiKey) {
      // Fallback to mock implementation if no API key
      return this.callMockLLM(messages);
    }

    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: 'gpt-4',
          messages: messages,
          temperature: 0.7,
          max_tokens: 1000,
        }),
      });

      if (!response.ok) {
        throw new Error(`OpenAI API error: ${response.status}`);
      }

      const data = await response.json();
      return data.choices[0]?.message?.content || 'No response from AI';
    } catch (error) {
      console.error('OpenAI API call failed:', error);
      // Fallback to mock implementation
      return this.callMockLLM(messages);
    }
  }

  private async callMockLLM(messages: LLMMessage[]): Promise<string> {
    // Mock implementation for when API key is not available
    const systemPrompt = messages.find(m => m.role === 'system')?.content || '';
    const userMessage = messages.find(m => m.role === 'user')?.content || '';
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Mock response based on the context
    if (userMessage.includes('business rules')) {
      return this.generateMockBusinessRules(userMessage);
    } else if (userMessage.includes('description')) {
      return this.generateMockDescription(userMessage);
    }
    
    return "I understand you're asking about business rules and descriptions. Please provide more specific context about the table and columns you'd like me to help with.";
  }

  private generateMockBusinessRules(userMessage: string): string {
    if (userMessage.includes('user') || userMessage.includes('customer')) {
      return `Business Rules for User/Customer Data:
1. Email addresses must be unique across the system
2. Phone numbers should follow international format (E.164)
3. Date of birth must be in the past and user must be at least 13 years old
4. Password must meet security requirements (8+ chars, uppercase, lowercase, number, special char)
5. Account status can only be: 'active', 'inactive', 'suspended', 'pending_verification'
6. User IDs must be UUID format and globally unique
7. Last login timestamp should be updated on each successful authentication
8. Profile completion percentage should be calculated based on filled required fields`;
    }
    
    if (userMessage.includes('order') || userMessage.includes('transaction')) {
      return `Business Rules for Order/Transaction Data:
1. Order amounts must be positive numbers with 2 decimal places
2. Order status transitions: 'pending' → 'confirmed' → 'processing' → 'shipped' → 'delivered'
3. Order numbers must be unique and follow format: ORD-YYYYMMDD-XXXXX
4. Payment status must match order status (pending orders cannot have 'paid' status)
5. Shipping address must be validated before order confirmation
6. Order total must equal sum of line items plus tax and shipping
7. Cancelled orders cannot be modified
8. Refund amounts cannot exceed original order amount`;
    }
    
    return `General Business Rules:
1. All timestamps should be in UTC timezone
2. Soft deletes should be used instead of hard deletes for audit trails
3. Foreign key relationships must be maintained for data integrity
4. Unique constraints should be enforced at application and database level
5. Data validation should occur at multiple layers (UI, API, Database)
6. Audit fields (created_at, updated_at, created_by, updated_by) should be maintained
7. Status fields should have predefined enum values
8. Monetary amounts should use decimal type with appropriate precision`;
  }

  private generateMockDescription(userMessage: string): string {
    if (userMessage.includes('user') || userMessage.includes('customer')) {
      return `This table stores comprehensive user account information including authentication details, personal information, and account status. It serves as the central repository for user identity and profile data, supporting features like user registration, login, profile management, and account administration. The table maintains referential integrity with related entities like orders, preferences, and activity logs.`;
    }
    
    if (userMessage.includes('order') || userMessage.includes('transaction')) {
      return `This table manages the complete order lifecycle from creation to fulfillment. It tracks order details, customer information, payment status, shipping information, and order history. The table supports e-commerce operations including order processing, inventory management, shipping coordination, and customer service. It maintains relationships with product catalog, customer accounts, and payment processing systems.`;
    }
    
    return `This table serves as a core entity in the system, managing essential business data and relationships. It supports key business processes and maintains data integrity through proper constraints and relationships. The table is designed to scale with business growth while maintaining performance and data quality standards.`;
  }

  async generateBusinessRules(context: BusinessRuleContext): Promise<GeneratedBusinessRules> {
    const systemPrompt = `You are an expert database architect and business analyst. Your task is to generate comprehensive business rules and descriptions for database tables and columns.

Context:
- Table: ${context.tableName}
- Column: ${context.columnName || 'N/A'}
- Data Type: ${context.dataType || 'N/A'}
- Nullable: ${context.isNullable || 'N/A'}
- Primary Key: ${context.isPrimaryKey || 'N/A'}
- Unique: ${context.isUnique || 'N/A'}
- Domain Context: ${context.domainContext || 'General business domain'}

Please provide:
1. Business rules that should be enforced for this table/column
2. Enhanced description explaining the business purpose and usage
3. Data generation rules for synthetic data creation
4. Validation rules for data integrity

Be specific, practical, and consider real-world business scenarios.`;

    const userPrompt = `Generate business rules and enhanced description for ${context.columnName ? `column '${context.columnName}' in ` : ''}table '${context.tableName}'.`;

    const messages: LLMMessage[] = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt }
    ];

    const response = await this.callOpenAI(messages);

    return {
      businessRules: response,
      enhancedDescription: this.generateMockDescription(userPrompt),
      dataGenerationRules: this.generateDataGenerationRules(context),
      validationRules: this.generateValidationRules(context)
    };
  }

  private generateDataGenerationRules(context: BusinessRuleContext): string {
    if (context.columnName?.toLowerCase().includes('email')) {
      return "Generate realistic email addresses using faker.js with domains matching the business context";
    }
    if (context.columnName?.toLowerCase().includes('phone')) {
      return "Generate phone numbers in E.164 format with country codes based on location data";
    }
    if (context.dataType?.includes('date')) {
      return "Generate dates within reasonable business ranges (e.g., birth dates 18-80 years ago, order dates within last 2 years)";
    }
    if (context.dataType?.includes('decimal') || context.dataType?.includes('numeric')) {
      return "Generate realistic monetary amounts with appropriate ranges for the business context";
    }
    return "Use faker.js to generate realistic data matching the column's data type and business context";
  }

  private generateValidationRules(context: BusinessRuleContext): string {
    if (context.columnName?.toLowerCase().includes('email')) {
      return "Email format validation using regex pattern, check for valid domain structure";
    }
    if (context.columnName?.toLowerCase().includes('phone')) {
      return "Phone number validation for E.164 format, country code validation";
    }
    if (context.dataType?.includes('date')) {
      return "Date range validation, ensure dates are within business logic constraints";
    }
    return "Standard validation based on data type, null constraints, and unique constraints";
  }

  async saveConversation(
    projectId: string,
    tableId: string | null,
    columnId: string | null,
    conversationType: string,
    messages: LLMMessage[]
  ): Promise<void> {
    const { error } = await supabase
      .from('llm_conversations')
      .insert({
        project_id: projectId,
        table_id: tableId,
        column_id: columnId,
        conversation_type: conversationType,
        messages: messages
      });

    if (error) throw error;
  }

  async getConversationHistory(
    projectId: string,
    tableId?: string,
    columnId?: string
  ): Promise<LLMMessage[]> {
    const { data, error } = await supabase
      .from('llm_conversations')
      .select('messages')
      .eq('project_id', projectId)
      .eq('table_id', tableId || null)
      .eq('column_id', columnId || null)
      .order('created_at', { ascending: false })
      .limit(1)
      .single();

    if (error) return [];
    return data?.messages || [];
  }
}

export const llmService = new LLMService(); 