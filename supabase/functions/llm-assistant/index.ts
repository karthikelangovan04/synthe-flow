import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    console.log('Edge Function started')
    
    // Create Supabase client with service role
    const supabaseUrl = Deno.env.get('SUPABASE_URL')
    const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')
    
    console.log('Supabase URL:', supabaseUrl ? 'Set' : 'Missing')
    console.log('Service Role Key:', serviceRoleKey ? 'Set' : 'Missing')
    
    const supabaseClient = createClient(
      supabaseUrl ?? '',
      serviceRoleKey ?? '',
      {
        auth: {
          autoRefreshToken: false,
          persistSession: false
        }
      }
    )

    const { messages, context } = await req.json()
    console.log('Received request with messages:', messages.length, 'context:', context)

    // Get OpenAI API key from secrets table
    console.log('Attempting to fetch OpenAI API key from api_secrets table...')
    const { data: apiKeyData, error: apiKeyError } = await supabaseClient
      .from('api_secrets')
      .select('key_value')
      .eq('key_name', 'OPENAI_API_KEY')
      .single()

    console.log('API Key query result:', { data: apiKeyData, error: apiKeyError })

    if (apiKeyError) {
      console.error('Error fetching API key:', apiKeyError)
      throw new Error(`Failed to fetch OpenAI API key: ${apiKeyError.message}`)
    }

    if (!apiKeyData?.key_value) {
      console.error('No API key found in database')
      throw new Error('OpenAI API key not found in database')
    }

    const openaiApiKey = apiKeyData.key_value
    console.log('OpenAI API key found, length:', openaiApiKey.length)

    // Call OpenAI API
    console.log('Calling OpenAI API...')
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${openaiApiKey}`,
      },
      body: JSON.stringify({
        model: 'gpt-4',
        messages: messages,
        temperature: 0.7,
        max_tokens: 1000,
      }),
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
    if (context?.projectId && context?.tableId) {
      console.log('Saving conversation to database...')
      const { error: saveError } = await supabaseClient
        .from('llm_conversations')
        .insert({
          project_id: context.projectId,
          table_id: context.tableId,
          column_id: context.columnId || null,
          conversation_type: 'column_business_rules',
          messages: messages
        })
      
      if (saveError) {
        console.error('Error saving conversation:', saveError)
      } else {
        console.log('Conversation saved successfully')
      }
    }

    return new Response(
      JSON.stringify({ 
        success: true, 
        response: aiResponse 
      }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200 
      }
    )

  } catch (error) {
    console.error('Edge Function error:', error)
    return new Response(
      JSON.stringify({ 
        success: false, 
        error: error.message 
      }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500 
      }
    )
  }
}) 