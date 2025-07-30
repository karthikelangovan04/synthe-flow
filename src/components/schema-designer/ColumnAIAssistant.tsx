import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Send, 
  Bot, 
  User, 
  Sparkles, 
  Save, 
  Loader2,
  MessageSquare,
  X
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { 
  llmService, 
  LLMMessage, 
  BusinessRuleContext, 
  GeneratedBusinessRules 
} from '@/lib/llm-service';
import { supabase } from '@/integrations/supabase/client';

interface ColumnAIAssistantProps {
  projectId: string;
  tableId: string;
  columnId: string;
  tableName: string;
  columnName: string;
  dataType: string;
  isNullable: boolean;
  isPrimaryKey: boolean;
  isUnique: boolean;
  existingDescription?: string | null;
  onClose: () => void;
  onColumnUpdated: () => void;
}

export function ColumnAIAssistant({
  projectId,
  tableId,
  columnId,
  tableName,
  columnName,
  dataType,
  isNullable,
  isPrimaryKey,
  isUnique,
  existingDescription,
  onClose,
  onColumnUpdated
}: ColumnAIAssistantProps) {
  const [messages, setMessages] = useState<LLMMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  // Load conversation history on mount
  useEffect(() => {
    loadConversationHistory();
  }, [projectId, tableId, columnId]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  const loadConversationHistory = async () => {
    try {
      const history = await llmService.getConversationHistory(projectId, tableId, columnId);
      if (history.length > 0) {
        setMessages(history);
      } else {
        // Initialize with a welcome message
        setMessages([{
          role: 'assistant',
          content: `Hello! I'm here to help you enhance the description and business rules for the '${columnName}' column in the '${tableName}' table. What would you like to know?`
        }]);
      }
    } catch (error) {
      console.error('Failed to load conversation history:', error);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: LLMMessage = {
      role: 'user',
      content: inputMessage
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Save conversation to database
      await llmService.saveConversation(
        projectId,
        tableId,
        columnId,
        'column_business_rules',
        [...messages, userMessage]
      );

      // Get AI response
      const context: BusinessRuleContext = {
        tableName,
        columnName,
        dataType,
        isNullable,
        isPrimaryKey,
        isUnique,
        existingDescription: existingDescription || undefined,
        domainContext: 'General business domain'
      };

      const response = await llmService.generateBusinessRules(context, projectId, tableId, columnId);
      
      const assistantMessage: LLMMessage = {
        role: 'assistant',
        content: `Here's what I can help you with for the '${columnName}' column:\n\n**Business Rules:**\n${response.businessRules}\n\n**Enhanced Description:**\n${response.enhancedDescription}\n\n**Data Generation Rules:**\n${response.dataGenerationRules}\n\n**Validation Rules:**\n${response.validationRules}`
      };

      const updatedMessages = [...messages, userMessage, assistantMessage];
      setMessages(updatedMessages);

      // Save updated conversation
      await llmService.saveConversation(
        projectId,
        tableId,
        columnId,
        'column_business_rules',
        updatedMessages
      );

    } catch (error) {
      console.error('Error in chat:', error);
      toast({
        title: "Error",
        description: "Failed to get AI response",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateBusinessRules = async () => {
    setIsGenerating(true);
    
    try {
      const context: BusinessRuleContext = {
        tableName,
        columnName,
        dataType,
        isNullable,
        isPrimaryKey,
        isUnique,
        existingDescription: existingDescription || undefined,
        domainContext: 'General business domain'
      };
      
      const generatedRules = await llmService.generateBusinessRules(context, projectId, tableId, columnId);
      
      const assistantMessage: LLMMessage = {
        role: 'assistant',
        content: `I've generated comprehensive business rules and descriptions for the '${columnName}' column:\n\n**Business Rules:**\n${generatedRules.businessRules}\n\n**Enhanced Description:**\n${generatedRules.enhancedDescription}\n\n**Data Generation Rules:**\n${generatedRules.dataGenerationRules}\n\n**Validation Rules:**\n${generatedRules.validationRules}`
      };

      const updatedMessages = [...messages, assistantMessage];
      setMessages(updatedMessages);

      // Save conversation
      await llmService.saveConversation(
        projectId,
        tableId,
        columnId,
        'column_business_rules',
        updatedMessages
      );
      
      toast({
        title: "Success",
        description: `Generated business rules for ${columnName}`
      });
      
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to generate business rules",
        variant: "destructive"
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSaveToColumn = async () => {
    setIsSaving(true);
    
    try {
      // Find the last AI response that contains business rules
      const lastAIMessage = messages
        .filter(m => m.role === 'assistant')
        .pop();

      if (lastAIMessage) {
        // Extract business rules and enhanced description from the message
        const content = lastAIMessage.content;
        
        // Simple parsing - you might want to improve this
        const businessRulesMatch = content.match(/\*\*Business Rules:\*\*\n([\s\S]*?)(?=\n\*\*|$)/);
        const enhancedDescriptionMatch = content.match(/\*\*Enhanced Description:\*\*\n([\s\S]*?)(?=\n\*\*|$)/);
        const dataGenerationRulesMatch = content.match(/\*\*Data Generation Rules:\*\*\n([\s\S]*?)(?=\n\*\*|$)/);
        const validationRulesMatch = content.match(/\*\*Validation Rules:\*\*\n([\s\S]*?)(?=\n\*\*|$)/);

        const businessRules = businessRulesMatch ? businessRulesMatch[1].trim() : null;
        const enhancedDescription = enhancedDescriptionMatch ? enhancedDescriptionMatch[1].trim() : null;
        const dataGenerationRules = dataGenerationRulesMatch ? dataGenerationRulesMatch[1].trim() : null;
        const validationRules = validationRulesMatch ? validationRulesMatch[1].trim() : null;

        // Update the column in Supabase
        const { error } = await supabase
          .from('column_metadata')
          .update({
            business_rules: businessRules,
            enhanced_description: enhancedDescription,
            data_generation_rules: dataGenerationRules,
            validation_rules: validationRules,
          })
          .eq('id', columnId);

        if (error) throw error;

        toast({
          title: "Success",
          description: "Business rules and descriptions saved to column!"
        });

        onColumnUpdated();
      } else {
        toast({
          title: "Warning",
          description: "No business rules to save. Generate some first!",
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error('Error saving to column:', error);
      toast({
        title: "Error",
        description: "Failed to save to column",
        variant: "destructive"
      });
    } finally {
      setIsSaving(false);
    }
  };

  const suggestedPrompts = [
    "What business rules should apply to this column?",
    "Generate a detailed description for this column",
    "What validation rules should be implemented?",
    "How should synthetic data be generated for this field?",
    "What are the data quality requirements?"
  ];

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            AI Assistant - {columnName}
          </CardTitle>
          <Button
            size="sm"
            variant="ghost"
            onClick={onClose}
            className="h-8 w-8 p-0"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="text-sm text-muted-foreground">
          Column: {columnName} ({dataType}) - {tableName}
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col gap-4">
        {/* Quick Actions */}
        <div className="flex gap-2">
          <Button
            size="sm"
            onClick={handleGenerateBusinessRules}
            disabled={isGenerating}
            className="flex-1"
          >
            {isGenerating ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4 mr-2" />
            )}
            Generate Rules
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={handleSaveToColumn}
            disabled={isSaving}
          >
            {isSaving ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            Save to Column
          </Button>
        </div>

        {/* Suggested Prompts */}
        <div className="flex flex-wrap gap-2">
          {suggestedPrompts.map((prompt, index) => (
            <Button
              key={index}
              size="sm"
              variant="outline"
              onClick={() => setInputMessage(prompt)}
              className="text-xs"
            >
              {prompt}
            </Button>
          ))}
        </div>

        {/* Conversation Area */}
        <ScrollArea ref={scrollAreaRef} className="flex-1 border rounded-md p-4 min-h-[300px]">
          <div className="space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex gap-3 ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {message.role === 'assistant' && (
                  <div className="flex-shrink-0 w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center">
                    <Bot className="h-4 w-4 text-primary" />
                  </div>
                )}
                <div
                  className={`max-w-[80%] p-3 rounded-lg ${
                    message.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted'
                  }`}
                >
                  <div className="whitespace-pre-wrap text-sm">{message.content}</div>
                </div>
                {message.role === 'user' && (
                  <div className="flex-shrink-0 w-8 h-8 bg-primary rounded-full flex items-center justify-center">
                    <User className="h-4 w-4 text-primary-foreground" />
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="flex gap-3 justify-start">
                <div className="flex-shrink-0 w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center">
                  <Bot className="h-4 w-4 text-primary" />
                </div>
                <div className="bg-muted p-3 rounded-lg">
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span className="text-sm">Thinking...</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Input Area */}
        <div className="flex gap-2">
          <Input
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="Ask about business rules, descriptions, or validation..."
            disabled={isLoading}
            className="flex-1"
          />
          <Button
            onClick={handleSendMessage}
            disabled={isLoading || !inputMessage.trim()}
            size="sm"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
} 