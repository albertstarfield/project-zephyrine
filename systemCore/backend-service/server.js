require('dotenv').config(); // Load .env variables
const express = require('express');
const cors = require('cors');
const Groq = require('groq-sdk');
const http = require('http'); // Import http module
const { WebSocketServer } = require('ws'); // Import WebSocketServer

const app = express();
const port = process.env.PORT || 3001;
const server = http.createServer(app); // Create HTTP server from Express app
const wss = new WebSocketServer({ server }); // Create WebSocket server

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

// --- WebSocket Handling ---
wss.on('connection', (ws) => {
  console.log('Client connected via WebSocket');

  ws.on('message', async (message) => {
    console.log('Received message:', message.toString());
    let requestData;

    try {
      requestData = JSON.parse(message.toString());
    } catch (error) {
      console.error('Failed to parse incoming message:', error);
      ws.send(JSON.stringify({ type: 'error', payload: { error: 'Invalid message format. Expected JSON.' } }));
      return;
    }

    // Handle 'chat' message type
    if (requestData.type === 'chat') {
      const { messages, model } = requestData.payload;

      // Basic validation
      if (!messages || !Array.isArray(messages) || messages.length === 0) {
        ws.send(JSON.stringify({ type: 'error', payload: { error: 'Missing or invalid "messages" array in payload.' } }));
        return;
      }
      if (!model) {
        ws.send(JSON.stringify({ type: 'error', payload: { error: 'Missing "model" in payload.' } }));
        return;
      }

      try {
        const formattedMsgs = formatMessages(messages);
        console.log(`Streaming response via WebSocket for model: ${model}`);

        const stream = await groq.chat.completions.create({
          messages: formattedMsgs,
          model: model,
          temperature: 0.7,
          max_tokens: 1024,
          top_p: 1,
          stream: true,
        });

        // Stream data back to the client via WebSocket
        for await (const chunk of stream) {
          const content = chunk.choices[0]?.delta?.content || '';
          if (content) {
            ws.send(JSON.stringify({ type: 'chunk', payload: { content } }));
          }
        }

        // Signal end of stream
        ws.send(JSON.stringify({ type: 'end' }));
        console.log('Stream finished for WebSocket client.');

      } catch (error) {
        console.error('Error calling Groq API:', error);
        ws.send(JSON.stringify({ type: 'error', payload: { error: 'Failed to get response from Groq API.' } }));
      }
    } else {
      console.log(`Received unhandled message type: ${requestData.type}`);
      // Optionally send an error or ignore
      // ws.send(JSON.stringify({ type: 'error', payload: { error: `Unhandled message type: ${requestData.type}` } }));
    }
  });

  ws.on('close', () => {
    console.log('Client disconnected');
  });

  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
  });
});


// --- Start Server ---
// Start the HTTP server which includes WebSocket support
server.listen(port, () => {
  console.log(`Backend service with WebSocket listening on port ${port}`);
});
