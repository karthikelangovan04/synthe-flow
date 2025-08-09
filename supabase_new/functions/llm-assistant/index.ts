import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface LLMRequest {
  messages: Array<{
    role: 'system' | 'user' | 'assistant';
    content: string;
  }>;
  context?: {
    projectId?: string;
    tableId?: string;
    columnId?: string;
  };
  model?: string;
  temperature?: number;
  maxTokens?: number;
}

interface LLMResponse {
  success: boolean;
  response?: string;
  error?: string;
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    console.log('LLM Assistant Edge Function started')
    
    // Validate request method
    if (req.method !== 'POST') {
      throw new Error('Only POST requests are allowed')
    }

    // Create Supabase client with service role
    const supabaseUrl = Deno.env.get('SUPABASE_URL')
    const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')
    
    if (!supabaseUrl || !serviceRoleKey) {
      throw new Error('Missing Supabase configuration')
    }
    
    const supabaseClient = createClient(supabaseUrl, serviceRoleKey, {
      auth: {
        autoRefreshToken: false,
        persistSession: false
      }
    })

    // Parse and validate request body
    const requestBody: LLMRequest = await req.json()
    
    if (!requestBody.messages || !Array.isArray(requestBody.messages) || requestBody.messages.length === 0) {
      throw new Error('Invalid messages array in request')
    }

    console.log('Received request with messages:', requestBody.messages.length, 'context:', requestBody.context)

    // Get OpenAI API key from secrets table
    console.log('Fetching OpenAI API key from api_secrets table...')
    const { data: apiKeyData, error: apiKeyError } = await supabaseClient
      .from('api_secrets')
      .select('key_value')
      .eq('key_name', 'OPENAI_API_KEY')
      .eq('is_active', true)
      .single()

    if (apiKeyError) {
      console.error('Error fetching API key:', apiKeyError)
      throw new Error(`Failed to fetch OpenAI API key: ${apiKeyError.message}`)
    }

    if (!apiKeyData?.key_value) {
      console.error('No active API key found in database')
      throw new Error('OpenAI API key not found or inactive in database')
    }

    const openaiApiKey = apiKeyData.key_value
    console.log('OpenAI API key found, length:', openaiApiKey.length)

    // Validate project ownership if context is provided
    if (requestBody.context?.projectId) {
      const { data: projectData, error: projectError } = await supabaseClient
        .from('projects')
        .select('id')
        .eq('id', requestBody.context.projectId)
        .single()

      if (projectError || !projectData) {
        throw new Error('Project not found or access denied')
      }
    }

    // Prepare OpenAI request
    const openaiRequest = {
      model: requestBody.model || 'gpt-4',
      messages: requestBody.messages,
      temperature: requestBody.temperature || 0.7,
      max_tokens: requestBody.maxTokens || 1000,
    }

    console.log('Calling OpenAI API with model:', openaiRequest.model)
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${openaiApiKey}`,
      },
      body: JSON.stringify(openaiRequest),
    })

    console.log('OpenAI response status:', response.status)

    if (!response.ok) {
      const errorText = await response.text()
      console.error('OpenAI API error:', errorText)
      throw new Error(`OpenAI API error: ${response.status} - ${errorText}`)
    }

    const data = await response.json()
    const aiResponse = data.choices[0]?.message?.content || 'No response from AI'
    console.log('OpenAI response received, length:', aiResponse.length)

    // Save conversation to database if context is provided
    if (requestBody.context?.projectId) {
      console.log('Saving conversation to database...')
      const { error: saveError } = await supabaseClient
        .from('llm_conversations')
        .insert({
          project_id: requestBody.context.projectId,
          table_id: requestBody.context.tableId || null,
          column_id: requestBody.context.columnId || null,
          conversation_type: 'column_business_rules',
          messages: requestBody.messages
        })
      
      if (saveError) {
        console.error('Error saving conversation:', saveError)
        // Don't fail the request if saving conversation fails
      } else {
        console.log('Conversation saved successfully')
      }
    }

    const responseBody: LLMResponse = {
      success: true,
      response: aiResponse
    }

    return new Response(
      JSON.stringify(responseBody),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200 
      }
    )

  } catch (error) {
    console.error('LLM Assistant Edge Function error:', error)
    
    const errorResponse: LLMResponse = {
      success: false,
      error: error.message || 'Internal server error'
    }

    return new Response(
      JSON.stringify(errorResponse),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500 
      }
    )
  }
}) 