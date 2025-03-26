require("dotenv").config(); // Load .env variables
const express = require("express");
const cors = require("cors");
const Groq = require("groq-sdk");
const http = require("http"); // Import http module
const { WebSocketServer } = require("ws"); // Import WebSocketServer
const { createClient } = require("@supabase/supabase-js"); // Import Supabase

const app = express();
const port = process.env.PORT || 3001;
const server = http.createServer(app); // Create HTTP server from Express app
const wss = new WebSocketServer({ server }); // Create WebSocket server

// --- Middleware ---
// Enable CORS for requests from your frontend (adjust origin in production)
app.use(cors({ origin: "*" })); // Allow all origins for now
app.use(express.json()); // Parse JSON request bodies

// --- Groq Client ---
const groqApiKey = process.env.GROQ_API_KEY;
if (!groqApiKey) {
  console.error("FATAL ERROR: GROQ_API_KEY is not defined in .env");
  process.exit(1); // Exit if API key is missing
}
const groq = new Groq({ apiKey: groqApiKey });

// --- Supabase Client ---
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY; // Use Service Role Key for backend operations

if (!supabaseUrl || !supabaseServiceKey) {
  console.error(
    "FATAL ERROR: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is not defined in .env"
  );
  process.exit(1);
}
// Initialize Supabase client with Service Role Key for admin-level access
const supabase = createClient(supabaseUrl, supabaseServiceKey);

// --- Helper ---
// Format messages for Groq API (sender -> role)
const formatMessages = (messages) => {
  return messages.map((msg) => ({
    role: msg.sender === "user" ? "user" : "assistant",
    content: msg.content,
  }));
};

// --- Title Generation Helper (Simplified - Takes Content Directly) ---
const TITLE_PROMPT =
  "write a concise three word summary of the title based on the conversation and context below. ONLY THREE WORDS, NO OTHER EXPLANATION OR I WILL KILL YOU";

// Add wsClient and firstUserMessageContent parameters
const generateAndSaveChatTitle = async (
  chatId,
  userId,
  wsClient,
  firstUserMessageContent
) => {
  if (!chatId || !userId) {
    console.error("Cannot generate title: Missing chatId or userId.");
    return;
  }
  // Check if content was provided
  if (!firstUserMessageContent) {
    console.warn(
      `Cannot generate title for chat ${chatId}: First user message content not provided.`
    );
    return;
  }

  try {
    // 1. Format for Groq (using only the provided first user message content)
    const groqMessages = [
      { role: "system", content: TITLE_PROMPT },
      // Provide the first message content directly in the user role for the prompt context
      { role: "user", content: firstUserMessageContent },
    ];

    // 2. Call Groq API
    console.log(
      `Generating title for chat ${chatId} based on first message...`
    );
    const completion = await groq.chat.completions.create({
      messages: groqMessages,
      model: "llama3-70b-8192", // Use a fast model
      temperature: 0.5,
      max_tokens: 20,
      stream: false,
    });

    let generatedTitle =
      completion.choices[0]?.message?.content?.trim() ||
      `Chat ${chatId.substring(0, 8)}`;
    // Basic cleanup
    generatedTitle = generatedTitle.replace(/["']/g, "");
    console.log(`Generated title for chat ${chatId}: "${generatedTitle}"`);

    // 3. Save title to Supabase 'chats' table
    const { error: upsertError } = await supabase.from("chats").upsert(
      { id: chatId, user_id: userId, title: generatedTitle },
      { onConflict: "id" } // If chat 'id' already exists, update title
    );

    if (upsertError) {
      console.error(
        `Error saving title to DB for chat ${chatId}:`,
        upsertError
      );
      // Log error but don't crash the main process
    } else {
      console.log(`Successfully saved title for chat ${chatId}.`);

      // Send confirmation back to the specific client that initiated this chat
      if (wsClient && wsClient.readyState === wsClient.OPEN) {
        wsClient.send(
          JSON.stringify({
            type: "title_updated",
            payload: { chatId: chatId, title: generatedTitle },
          })
        );
        console.log(`Sent title_updated confirmation for chat ${chatId}`);
      }
    }
  } catch (error) {
    console.error(
      `Unexpected error during title generation/saving for chat ${chatId}:`,
      error
    );
  }
};

// --- WebSocket Handling ---
wss.on("connection", (ws) => {
  let currentStream = null; // Keep track of the current stream for cancellation
  console.log("Client connected via WebSocket");

  ws.on("message", async (message) => {
    console.log("Received message:", message.toString());
    let requestData;

    try {
      requestData = JSON.parse(message.toString());
    } catch (error) {
      console.error("Failed to parse incoming message:", error);
      ws.send(
        JSON.stringify({
          type: "error",
          payload: { error: "Invalid message format. Expected JSON." },
        })
      );
      return;
    }

    // Handle 'chat' message type
    if (requestData.type === "chat") {
      // IMPORTANT: Expecting userId and potentially firstUserMessageContent
      const { messages, model, chatId, userId, firstUserMessageContent } =
        requestData.payload;

      // Basic validation
      if (!chatId) {
        ws.send(
          JSON.stringify({
            type: "error",
            payload: { error: 'Missing "chatId" in payload.' },
          })
        );
        return;
      }
      if (!userId) {
        ws.send(
          JSON.stringify({
            type: "error",
            payload: { error: 'Missing "userId" in payload.' },
          })
        );
        return;
      }
      if (!messages || !Array.isArray(messages) || messages.length === 0) {
        ws.send(
          JSON.stringify({
            type: "error",
            payload: {
              error: 'Missing or invalid "messages" array in payload.',
            },
          })
        );
        return;
      }
      if (!model) {
        ws.send(
          JSON.stringify({
            type: "error",
            payload: { error: 'Missing "model" in payload.' },
          })
        );
        return;
      }

      try {
        const formattedMsgs = formatMessages(messages);
        console.log(
          `Streaming response via WebSocket for model: ${model}, chatId: ${chatId}`
        );

        // Store the stream promise/controller if you need finer control over stopping
        currentStream = groq.chat.completions.create({
          messages: formattedMsgs,
          model: model,
          temperature: 0.7,
          max_tokens: 1024,
          top_p: 1,
          stream: true,
        });

        // Stream data back
        for await (const chunk of await currentStream) {
          // Check if stream was cancelled externally (e.g., by 'stop' message)
          if (!currentStream) break; // Exit loop if cancelled

          const content = chunk.choices[0]?.delta?.content || "";
          if (content) {
            // Check WebSocket state before sending
            if (ws.readyState === ws.OPEN) {
              ws.send(JSON.stringify({ type: "chunk", payload: { content } }));
            } else {
              console.log("WebSocket closed during streaming, stopping.");
              currentStream = null; // Stop processing if client disconnected
              break;
            }
          }
        }

        // Check if stream completed naturally (wasn't cancelled) and WS is still open
        if (currentStream && ws.readyState === ws.OPEN) {
          ws.send(JSON.stringify({ type: "end" }));
          console.log(
            `Stream finished for WebSocket client (chatId: ${chatId}).`
          );

          // --- Trigger Title Generation & Saving if firstUserMessageContent was provided in the initial request ---
          // The frontend sends 'firstUserMessageContent' only when the user sends their first message.
          // We retrieve it from the original requestData here.
          const userMessageContentForTitle =
            requestData.payload.firstUserMessageContent;

          if (userMessageContentForTitle) {
            console.log(
              `First user message detected for chat ${chatId}, triggering title generation.`
            );
            // Call the function, passing the content directly from the initial payload
            // Run in background (don't await)
            generateAndSaveChatTitle(
              chatId,
              userId,
              ws,
              userMessageContentForTitle
            );
          }
          // --- End Title Generation Trigger ---
        } else if (!currentStream) {
          console.log(`Stream cancelled for chatId: ${chatId}`);
        }
      } catch (error) {
        console.error("Error during Groq API call or streaming:", error);
        if (ws.readyState === ws.OPEN) {
          ws.send(
            JSON.stringify({
              type: "error",
              payload: { error: "Failed to get response from Groq API." },
            })
          );
        }
      } finally {
        currentStream = null; // Ensure stream tracker is cleared
      }

      // Handle 'stop' message type
    } else if (requestData.type === "stop") {
      console.log("Received stop request from client.");
      currentStream = null; // Signal to stop processing the current stream
      // Optionally send confirmation back, though stopping generation locally is usually enough
      // ws.send(JSON.stringify({ type: 'stopped' }));
    } else {
      console.log(`Received unhandled message type: ${requestData.type}`);
    }
  });

  ws.on("close", () => {
    console.log("Client disconnected");
    currentStream = null; // Clear stream reference on disconnect
  });

  ws.on("error", (error) => {
    console.error("WebSocket error:", error);
  });
});

// --- Start Server ---
// Start the HTTP server which includes WebSocket support
server.listen(port, () => {
  console.log(`Backend service with WebSocket listening on port ${port}`);
});
