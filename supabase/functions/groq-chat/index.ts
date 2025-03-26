import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';
import Groq from 'npm:groq-sdk'; // Use npm specifier for Deno

// Polyfill for TextEncoderStream if not available (Deno should have it)
if (!globalThis.TextEncoderStream) {
  const { TextEncoderStream, TextDecoderStream } = await import('npm:streams-text-encoding@0.9.0');
  globalThis.TextEncoderStream = TextEncoderStream;
  globalThis.TextDecoderStream = TextDecoderStream;
}

// Helper to format messages for Groq API
const formatMessages = (messages) => {
  return messages.map(msg => ({
    role: msg.sender === 'user' ? 'user' : 'assistant',
    content: msg.content,
  }));
};

serve(async (req) => {
  // --- CORS Headers ---
  // Handle OPTIONS request for CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', {
      headers: {
        'Access-Control-Allow-Origin': '*', // Adjust in production!
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
      },
    });
  }

  try {
    // --- API Key ---
    const groqApiKey = Deno.env.get('GROQ_API_KEY');
    if (!groqApiKey) {
      console.error('GROQ_API_KEY environment variable not set.');
      return new Response(JSON.stringify({ error: 'Missing API key configuration.' }), {
        status: 500,
        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
      });
    }

    const groq = new Groq({ apiKey: groqApiKey });

    // --- Request Body ---
    const { messages, model } = await req.json();

    if (!messages || !Array.isArray(messages)) {
      return new Response(JSON.stringify({ error: 'Missing or invalid "messages" in request body.' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
      });
    }
    if (!model) {
        return new Response(JSON.stringify({ error: 'Missing "model" in request body.' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
        });
      }

    const formattedMsgs = formatMessages(messages);

    // --- Groq API Call ---
    const stream = await groq.chat.completions.create({
      messages: formattedMsgs,
      model: model, // Use the model passed from the frontend
      temperature: 0.7, // Adjust as needed
      max_tokens: 1024, // Adjust as needed
      top_p: 1,
      stream: true,
    });

    // --- Streaming Response ---
    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();
    const encoder = new TextEncoder();

    (async () => {
      try {
        for await (const chunk of stream) {
          const content = chunk.choices[0]?.delta?.content || '';
          if (content) {
            await writer.write(encoder.encode(`data: ${JSON.stringify({ content })}\n\n`));
          }
        }
        // Signal end of stream (optional, depends on client handling)
        // await writer.write(encoder.encode('event: end\ndata: {}\n\n'));
      } catch (error) {
        console.error('Error reading Groq stream:', error);
        await writer.write(encoder.encode(`event: error\ndata: ${JSON.stringify({ error: 'Stream read error' })}\n\n`));
      } finally {
        await writer.close();
      }
    })();

    // Return the readable stream to the client
    return new Response(readable, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*', // Adjust in production!
      },
    });

  } catch (error) {
    console.error('Error processing request:', error);
    return new Response(JSON.stringify({ error: error.message || 'Internal Server Error' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
    });
  }
});
