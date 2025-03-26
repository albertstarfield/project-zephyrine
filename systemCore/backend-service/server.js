require('dotenv').config(); // Load .env variables
const express = require('express');
const cors = require('cors');
const Groq = require('groq-sdk');

const app = express();
const port = process.env.PORT || 3001;

// --- Middleware ---
// Enable CORS for requests from your frontend (adjust origin in production)
app.use(cors({ origin: '*' })); // Allow all origins for now
app.use(express.json()); // Parse JSON request bodies

// --- Groq Client ---
const groqApiKey = process.env.GROQ_API_KEY;
if (!groqApiKey) {
  console.error('FATAL ERROR: GROQ_API_KEY is not defined in .env');
  process.exit(1); // Exit if API key is missing
}
const groq = new Groq({ apiKey: groqApiKey });

// --- Helper ---
// Format messages for Groq API (sender -> role)
const formatMessages = (messages) => {
  return messages.map(msg => ({
    role: msg.sender === 'user' ? 'user' : 'assistant',
    content: msg.content,
  }));
};

// --- API Endpoint ---
app.post('/api/chat', async (req, res) => {
  const { messages, model } = req.body;

  // Basic validation
  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'Missing or invalid "messages" array in request body.' });
  }
  if (!model) {
    return res.status(400).json({ error: 'Missing "model" in request body.' });
  }

  try {
    const formattedMsgs = formatMessages(messages);

    // Set headers for Server-Sent Events (SSE)
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders(); // Send headers immediately

    console.log(`Streaming response for model: ${model}`);

    const stream = await groq.chat.completions.create({
      messages: formattedMsgs,
      model: model,
      temperature: 0.7,
      max_tokens: 1024,
      top_p: 1,
      stream: true,
    });

    // Stream data to the client
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';
      if (content) {
        // Format as SSE: data: {...}\n\n
        res.write(`data: ${JSON.stringify({ content })}\n\n`);
      }
    }

    // Signal end of stream (optional, client might handle closure)
    // res.write('event: end\ndata: {}\n\n');

    console.log('Stream finished.');
    res.end(); // End the response when the stream is finished

  } catch (error) {
    console.error('Error calling Groq API:', error);
    // If headers haven't been sent, send a JSON error
    if (!res.headersSent) {
      res.status(500).json({ error: 'Failed to get response from Groq API.' });
    } else {
      // If headers were sent, try to send an error event via SSE
      res.write(`event: error\ndata: ${JSON.stringify({ error: 'Groq API error' })}\n\n`);
      res.end(); // End the response
    }
  }
});

// --- Start Server ---
app.listen(port, () => {
  console.log(`Backend service listening on port ${port}`);
});
