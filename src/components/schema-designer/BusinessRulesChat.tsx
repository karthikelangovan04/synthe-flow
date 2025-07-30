import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { 
  Send, 
  Bot, 
  User, 
  Sparkles, 
  Save, 
  Loader2,
  MessageSquare,
  FileText,
  Settings
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { 
  llmService, 
  LLMMessage, 
  BusinessRuleContext, 
  GeneratedBusinessRules 
} from '@/lib/llm-service';

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

export function BusinessRulesChat({
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
  onBusinessRulesGenerated
}: BusinessRulesChatProps) {
  const [messages, setMessages] = useState<LLMMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [conversationType, setConversationType] = useState<'table_description' | 'column_business_rules' | 'data_generation'>('column_business_rules');
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
      const history = await llmService.getConversationHistory(projectId, tableId || undefined, columnId || undefined);
      if (history.length > 0) {
        setMessages(history);
      } else {
        // Initialize with a welcome message
        setMessages([{
          role: 'assistant',
          content: `Hello! I'm here to help you generate business rules and descriptions for ${columnName ? `the '${columnName}' column in ` : ''}the '${tableName}' table. What would you like to know?`
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
      // In a real implementation, you would call the actual LLM API here
      // For now, we'll simulate a response
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const assistantMessage: LLMMessage = {
        role: 'assistant',
        content: `I understand you're asking about "${inputMessage}". Let me help you with that. Based on the context of ${columnName ? `column '${columnName}' in ` : ''}table '${tableName}', here are some relevant insights...`
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Save conversation to database
      await llmService.saveConversation(
        projectId,
        tableId,
        columnId,
        conversationType,
        [...messages, userMessage, assistantMessage]
      );

    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to send message. Please try again.",
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
        existingDescription,
        domainContext: 'General business domain'
      };

      const generatedRules = await llmService.generateBusinessRules(context);
      
      // Add the generated rules to the conversation
      const assistantMessage: LLMMessage = {
        role: 'assistant',
        content: `I've generated comprehensive business rules and descriptions for ${columnName ? `column '${columnName}' in ` : ''}table '${tableName}'. Here's what I found:

**Business Rules:**
${generatedRules.businessRules}

**Enhanced Description:**
${generatedRules.enhancedDescription}

**Data Generation Rules:**
${generatedRules.dataGenerationRules}

**Validation Rules:**
${generatedRules.validationRules}

Would you like me to save these to your schema?`
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Call the callback to update the parent component
      onBusinessRulesGenerated(generatedRules);

      toast({
        title: "Success",
        description: "Business rules generated successfully!"
      });

    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to generate business rules. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSaveToSchema = () => {
    // This will be handled by the parent component through the callback
    toast({
      title: "Success",
      description: "Business rules saved to schema!"
    });
  };

  const suggestedPrompts = [
    "What business rules should apply to this column?",
    "Generate a detailed description for this table/column",
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
            AI Business Rules Assistant
          </CardTitle>
          <Badge variant="secondary" className="text-xs">
            {columnName ? 'Column' : 'Table'} Context
          </Badge>
        </div>
        <div className="text-sm text-muted-foreground">
          {columnName ? `Column: ${columnName} (${dataType})` : `Table: ${tableName}`}
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
            onClick={handleSaveToSchema}
          >
            <Save className="h-4 w-4 mr-2" />
            Save to Schema
          </Button>
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
                  <Loader2 className="h-4 w-4 animate-spin" />
                </div>
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Suggested Prompts */}
        <div className="space-y-2">
          <div className="text-xs font-medium text-muted-foreground">Suggested prompts:</div>
          <div className="flex flex-wrap gap-1">
            {suggestedPrompts.map((prompt, index) => (
              <Button
                key={index}
                size="sm"
                variant="outline"
                className="text-xs h-auto py-1 px-2"
                onClick={() => setInputMessage(prompt)}
              >
                {prompt}
              </Button>
            ))}
          </div>
        </div>

        <Separator />

        {/* Input Area */}
        <div className="flex gap-2">
          <Textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Ask about business rules, descriptions, or data generation..."
            className="min-h-[60px] resize-none"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
              }
            }}
          />
          <Button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            size="sm"
            className="self-end"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
} 