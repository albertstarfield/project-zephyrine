// ExternalAnalyzer/backend-service/server.js
require('dotenv').config(); // Loads environment variables from .env file
const WebSocket = require('ws');
const http = require('http');
const { OpenAI } = require('openai'); // OpenAI SDK for LLM interaction
const sqlite3 = require('sqlite3').verbose(); // SQLite for database
const { v4: uuidv4 } = require('uuid'); // For generating unique IDs

// Configuration
const PORT = process.env.PORT || 3001;
const OPENAI_API_BASE_URL = process.env.OPENAI_API_BASE_URL || 'http://localhost:11434/v1'; // Default for local Ollama
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || 'ollama'; // API key, often a placeholder for local models
const DB_PATH = './project_zephyrine_chats.db'; // Path to SQLite database file

// LLM Parameters from environment variables, with defaults
const LLM_TEMPERATURE = parseFloat(process.env.LLM_TEMPERATURE) || 0.7;
const LLM_MAX_TOKENS = parseInt(process.env.LLM_MAX_TOKENS) || 2048;
const LLM_TOP_P = parseFloat(process.env.LLM_TOP_P) || 1.0;

// Initialize OpenAI client for OpenAI-compatible APIs
const openai = new OpenAI({
    baseURL: OPENAI_API_BASE_URL,
    apiKey: OPENAI_API_KEY,
});

// Database setup
const db = new sqlite3.Database(DB_PATH, (err) => {
    if (err) {
        console.error('Error opening database', err.message);
        // Consider exiting if DB connection fails, as it's critical
        process.exit(1); 
    } else {
        console.log('Connected to the SQLite database at', DB_PATH);
        // Create tables if they don't exist
        db.serialize(() => {
            // Chats table to store chat sessions
            db.run(`CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                user_id TEXT, 
                title TEXT,
                created_at DATETIME DEFAULT (datetime('now', 'localtime')),
                updated_at DATETIME DEFAULT (datetime('now', 'localtime'))
            )`, (err) => {
                if (err) console.error("Error creating chats table:", err.message);
            });

            // Messages table to store individual messages within chats
            // id is TEXT PRIMARY KEY, so we will generate UUIDs for it.
            db.run(`CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, 
                chat_id TEXT NOT NULL,
                user_id TEXT,
                sender TEXT NOT NULL, -- 'user' or 'assistant'
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )`, (err) => {
                if (err) console.error("Error creating messages table:", err.message);
            });

            // Optional: Users table (if you plan to implement user accounts)
            // db.run(\`CREATE TABLE IF NOT EXISTS users (
            //     id TEXT PRIMARY KEY,
            //     email TEXT UNIQUE,
            //     password_hash TEXT, // Store hashed passwords, not plain text
            //     created_at DATETIME DEFAULT (datetime('now', 'localtime'))
            // )\`, (err) => {
            //     if (err) console.error("Error creating users table:", err.message);
            // });
        });
    }
});

// Create HTTP server (mainly to host the WebSocket server)
const server = http.createServer((req, res) => {
    // Basic HTTP response for health checks or info
    if (req.url === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'ok', message: 'WebSocket server is healthy' }));
    } else {
        res.writeHead(200, { 'Content-Type': 'text/plain' });
        res.end('Project Zephyrine WebSocket server is running.');
    }
});

// Initialize WebSocket server
const wss = new WebSocket.Server({ server });
console.log(`WebSocket server setup on port ${PORT}`);

// For 'stop' functionality (conceptual, requires AbortController integration with LLM calls)
// const activeGenerations = new Map(); // Example: Map<chatId, AbortController>

// Handle WebSocket connections
wss.on('connection', (ws) => {
    console.log('Client connected via WebSocket');

    ws.on('close', (code, reason) => {
        console.log(`Client disconnected. Code: ${code}, Reason: ${reason || 'N/A'}`);
        // Clean up any resources associated with this client if necessary
        // e.g., if tracking activeGenerations per ws instance
    });

    ws.on('error', (error) => {
        console.error('WebSocket error for a client:', error);
        // ws.terminate() might be needed if the connection is in an unstable state
    });

    ws.on("message", async (message) => {
        const messageString = message.toString();
        // Log truncated message to avoid flooding console with very long messages
        console.log("Received raw message string (truncated):", messageString.substring(0, 250) + (messageString.length > 250 ? "..." : ""));
        
        let parsedMessage;
        let payload;

        try {
            parsedMessage = JSON.parse(messageString);
            payload = parsedMessage.payload || {}; // Ensure payload is at least an empty object

            console.log(`Processing WebSocket message: type = ${parsedMessage.type}, chatId = ${payload.chatId || 'N/A'}`);

            switch (parsedMessage.type) {
                case 'get_messages':
                    if (!payload.chatId) {
                        console.error("Invalid payload for get_messages: Missing chatId.", payload);
                        ws.send(JSON.stringify({ type: 'error', message: 'Invalid payload for get_messages: Missing chatId.' }));
                        return;
                    }
                    const { chatId: getChatId } = payload;
                    console.log(`Fetching messages for chat: ${getChatId}`);
                    try {
                        const messages = await new Promise((resolve, reject) => {
                            db.all("SELECT id, chat_id, user_id, sender, content, created_at FROM messages WHERE chat_id = ? ORDER BY created_at ASC", [getChatId], (err, rows) => {
                                if (err) {
                                    console.error(`Error fetching messages from SQLite for chat ${getChatId}:`, err.message);
                                    reject(err);
                                } else {
                                    // Map 'sender' to 'role' if frontend expects 'role'
                                    resolve(rows.map(r => ({...r, role: r.sender }))); 
                                }
                            });
                        });
                        ws.send(JSON.stringify({ type: 'chat_history', payload: { chatId: getChatId, messages: messages } }));
                        console.log(`Sending history for chat ${getChatId} (${messages.length} messages)`);
                    } catch (dbError) {
                        console.error(`Failed to fetch chat history for ${getChatId}:`, dbError.message);
                        ws.send(JSON.stringify({ type: 'chat_history_error', message: 'Failed to fetch chat history.', payload: { chatId: getChatId } }));
                    }
                    break;

                case 'chat':
                    if (!payload || !payload.messages || !payload.model || !payload.chatId || !payload.userId) {
                         console.error("Invalid payload for chat:", payload);
                         ws.send(JSON.stringify({ type: 'error', message: 'Invalid payload for chat: Missing required fields.' }));
                         return;
                    }
                    const { messages: clientMessages, model, chatId, userId, firstUserMessageContent } = payload;

                    if (!clientMessages || clientMessages.length === 0) {
                        ws.send(JSON.stringify({ type: 'error', message: 'Received empty messages for chat.' }));
                        return;
                    }
                    const userMessage = clientMessages[clientMessages.length - 1];

                    try {
                        // 1. Check if chat exists; if not, create it.
                        const chatExists = await new Promise((resolve, reject) => {
                            db.get("SELECT id FROM chats WHERE id = ?", [chatId], (err, row) => {
                                if (err) { console.error("Error checking if chat exists:", err.message); reject(err); } else { resolve(row); }
                            });
                        });

                        if (!chatExists) {
                            console.log(`Chat with ID ${chatId} does not exist. Creating new chat.`);
                            let chatTitle = `Chat (${model})`; // Default title
                            const latestUserMsgForTitle = firstUserMessageContent || userMessage.content;

                            // Generate title if it's effectively the first user message in a new chat
                            if (latestUserMsgForTitle && clientMessages.filter(m => (m.role === 'user' || m.sender === 'user')).length === 1) {
                                try {
                                    const titlePrompt = [{ role: "user", content: `Generate a very short, concise title (3-5 words maximum) for a new chat that begins with the following user message: "${latestUserMsgForTitle}"` }];
                                    const titleCompletion = await openai.chat.completions.create({
                                        messages: titlePrompt,
                                        model: model, // Use the model selected by the user, or a specific fast model for titles
                                        max_tokens: 25,
                                        temperature: 0.5,
                                    });
                                    const generatedTitle = titleCompletion.choices[0]?.message?.content?.trim();
                                    if (generatedTitle) {
                                        chatTitle = generatedTitle.replace(/["']/g, ''); // Remove quotes from title
                                    }
                                    console.log(`Generated title for new chat ${chatId}: "${chatTitle}"`);
                                } catch (titleError) {
                                    console.error("Error generating chat title with LLM:", titleError.message);
                                    // Fallback to a simpler title if LLM call fails
                                    chatTitle = `Chat: ${latestUserMsgForTitle.substring(0, 30)}...`;
                                }
                            } else if (userMessage.content) { 
                                // Fallback if not the first message or title generation failed
                                chatTitle = `Chat: ${userMessage.content.substring(0, 30)}...`;
                            }

                            await new Promise((resolve, reject) => {
                                db.run(
                                    "INSERT INTO chats (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, datetime('now', 'localtime'), datetime('now', 'localtime'))",
                                    [chatId, userId, chatTitle],
                                    function (err) {
                                        if (err) { console.error("Error inserting new chat into SQLite:", err.message); reject(err); }
                                        else {
                                            console.log(`New chat created with ID: ${chatId}, Title: "${chatTitle}"`);
                                            // Inform client about the new title (optional, frontend might handle this)
                                            ws.send(JSON.stringify({ type: 'title_updated', payload: { chatId: chatId, title: chatTitle } }));
                                            resolve(this.lastID);
                                        }
                                    }
                                );
                            });
                        }

                        // 2. Save the new user message to the database
                        if (!userMessage.content || typeof userMessage.content !== 'string') {
                            ws.send(JSON.stringify({ type: 'error', message: 'User message content is invalid.' }));
                            return;
                        }
                        const userMessageId = uuidv4(); // Generate unique ID for the user's message
                        await new Promise((resolve, reject) => {
                            db.run(
                                "INSERT INTO messages (id, chat_id, user_id, sender, content, created_at) VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))",
                                [userMessageId, chatId, userId, 'user', userMessage.content],
                                function(err) {
                                    if (err) { console.error(`Error saving user message to SQLite for chat ${chatId}:`, err.message); reject(err); }
                                    else {
                                        console.log(`User message (ID: ${userMessageId}) saved for chat ${chatId}: "${userMessage.content.substring(0, 50)}..."`);
                                        resolve(this.lastID);
                                    }
                                }
                            );
                        });

                        // 3. Call the LLM and stream the response
                        console.log(`Calling LLM (OpenAI-compatible) for chat ${chatId} with model ${model}`);
                        // const abortController = new AbortController(); // For 'stop' functionality
                        // activeGenerations.set(chatId, abortController);

                        const stream = await openai.chat.completions.create({
                            messages: clientMessages.map(m => ({ role: m.role || m.sender, content: m.content })), // Ensure correct role format
                            model: model,
                            stream: true,
                            temperature: LLM_TEMPERATURE,
                            max_tokens: LLM_MAX_TOKENS,
                            top_p: LLM_TOP_P,
                        }/*, { signal: abortController.signal }*/); // Pass signal if implementing stop

                        let assistantResponse = '';
                        for await (const chunk of stream) {
                            // if (abortController.signal.aborted) {
                            //     console.log(`Stream for chat ${chatId} aborted by client.`);
                            //     ws.send(JSON.stringify({ type: 'info', message: 'Generation stopped by user.'}));
                            //     break; 
                            // }
                            const content = chunk.choices[0]?.delta?.content || '';
                            if (content) {
                                assistantResponse += content;
                                ws.send(JSON.stringify({ type: 'chunk', payload: { content: content } }));
                            }
                        }
                        ws.send(JSON.stringify({ type: 'end' })); // Signal end of stream
                        console.log(`LLM stream ended for chat ${chatId}. Full response length: ${assistantResponse.length}`);
                        // activeGenerations.delete(chatId); // Clean up after stream ends or is aborted

                        // 4. Save the assistant's full response to the database
                        if (assistantResponse.trim()) {
                            const assistantMessageId = uuidv4(); // Generate unique ID for the assistant's message
                            await new Promise((resolve, reject) => {
                                db.run(
                                    "INSERT INTO messages (id, chat_id, user_id, sender, content, created_at) VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))",
                                    [assistantMessageId, chatId, userId, 'assistant', assistantResponse.trim()], // Storing assistant's response
                                    function(err) {
                                        if (err) { console.error("Error saving assistant message to SQLite:", err.message); reject(err); } // Don't necessarily need to break client flow
                                        else {
                                            console.log(`Assistant message (ID: ${assistantMessageId}) saved for chat ${chatId}: "${assistantResponse.substring(0,50)}..."`);
                                            resolve(this.lastID);
                                        }
                                    }
                                );
                            });
                            // Update chat's updated_at timestamp
                            db.run("UPDATE chats SET updated_at = datetime('now', 'localtime') WHERE id = ?", [chatId], err => {
                                if (err) console.error(`Error updating chat timestamp for ${chatId}:`, err.message);
                            });
                        }

                    } catch (chatProcessingError) {
                        console.error(`Error in 'chat' case for chatID ${chatId}:`, chatProcessingError.message, chatProcessingError.stack);
                        ws.send(JSON.stringify({ type: 'error', message: `Server error processing chat: ${chatProcessingError.message}` }));
                        // activeGenerations.delete(chatId); // Ensure cleanup on error
                    }
                    break;

                case 'edit_message':
                    if (!payload || !payload.messageId || !payload.newContent || !payload.chatId || !payload.history || !payload.userId || !payload.model) {
                        console.error("Invalid payload for edit_message:", payload);
                        ws.send(JSON.stringify({ type: 'error', message: "Invalid payload for edit_message: Missing required fields." }));
                        return;
                    }
                    const { messageId, newContent, chatId: editChatId, history: editHistory, userId: editUserId, model: editModel } = payload;
                    console.log(`Received edit_message for chat ${editChatId}, message ID ${messageId}`);
                    try {
                        // 1. Update the user's message content in the database
                        await new Promise((resolve, reject) => {
                            db.run(
                                "UPDATE messages SET content = ?, created_at = datetime('now', 'localtime') WHERE id = ? AND sender = 'user' AND chat_id = ?",
                                [newContent, messageId, editChatId],
                                function(err) {
                                    if (err) {
                                        console.error(`Error updating user message ${messageId} in SQLite:`, err.message);
                                        return reject(err);
                                    }
                                    if (this.changes === 0) {
                                        console.warn(`No message found to update for ID ${messageId}, or sender was not 'user', or chat ID mismatch.`);
                                        // Potentially reject or inform client if update failed critically
                                    } else {
                                        console.log(`User message ${messageId} updated in chat ${editChatId}.`);
                                    }
                                    resolve();
                                }
                            );
                        });

                        // 2. Prepare messages for LLM (history up to and including the edited message)
                        // Ensure the history provided by client is accurate and ends with the newly edited user message.
                        const messagesForLlm = editHistory.map(m => ({ role: m.role || m.sender, content: m.content }));
                        
                        // 3. Call LLM for new assistant response
                        console.log(`Calling LLM for edited message in chat ${editChatId} with model ${editModel}`);
                        const editStream = await openai.chat.completions.create({
                            messages: messagesForLlm,
                            model: editModel,
                            stream: true,
                            temperature: LLM_TEMPERATURE,
                            max_tokens: LLM_MAX_TOKENS,
                            top_p: LLM_TOP_P,
                        });

                        let newAssistantResponse = '';
                        for await (const chunk of editStream) {
                            const content = chunk.choices[0]?.delta?.content || '';
                            if (content) {
                                newAssistantResponse += content;
                                ws.send(JSON.stringify({ type: 'chunk', payload: { content: content } }));
                            }
                        }
                        ws.send(JSON.stringify({ type: 'end' }));
                        console.log(`LLM stream for edited response ended for chat ${editChatId}. Length: ${newAssistantResponse.length}`);

                        // 4. Save the new assistant's response
                        if (newAssistantResponse.trim()) {
                            const newAssistantMessageId = uuidv4(); // Generate ID for the new assistant message
                            await new Promise((resolve, reject) => {
                                db.run(
                                    "INSERT INTO messages (id, chat_id, user_id, sender, content, created_at) VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))",
                                    [newAssistantMessageId, editChatId, editUserId, 'assistant', newAssistantResponse.trim()],
                                    function(err) {
                                        if (err) { console.error("Error saving new assistant message for edited chat:", err.message); reject(err); }
                                        else {
                                            console.log(`New assistant message (ID: ${newAssistantMessageId}) saved for edited chat ${editChatId}.`);
                                            resolve(this.lastID);
                                        }
                                    }
                                );
                            });
                            db.run("UPDATE chats SET updated_at = datetime('now', 'localtime') WHERE id = ?", [editChatId]);
                        }
                        // Inform client that edit processing is complete (optional)
                        ws.send(JSON.stringify({ type: 'message_edit_complete', payload: { chatId: editChatId } }));


                    } catch (editError) {
                        console.error(`Error processing edit_message for chat ${editChatId}:`, editError.message, editError.stack);
                        ws.send(JSON.stringify({ type: 'error', message: `Server error processing message edit: ${editError.message}` }));
                    }
                    break;

                case 'stop':
                    // Placeholder for full 'stop' functionality.
                    // Requires AbortController to be passed to the LLM's fetch/stream call.
                    if (!payload || !payload.chatId) {
                        console.error("Invalid payload for stop:", payload);
                        ws.send(JSON.stringify({ type: 'error', message: "Invalid payload for stop: Missing chatId." }));
                        return;
                    }
                    const { chatId: stopChatId } = payload;
                    console.log(`Received stop request for chat ${stopChatId}. (Functionality placeholder)`);
                    // const controller = activeGenerations.get(stopChatId);
                    // if (controller) {
                    //     controller.abort();
                    //     console.log(`Aborted generation for chat ${stopChatId}`);
                    // } else {
                    //     console.log(`No active generation to stop for chat ${stopChatId}`);
                    // }
                    ws.send(JSON.stringify({ type: 'info', message: 'Stop functionality placeholder: Not fully implemented for active streams.' }));
                    break;

                default:
                    console.log(`Received unknown message type: ${parsedMessage.type}`);
                    ws.send(JSON.stringify({ type: 'error', message: `Unknown message type: ${parsedMessage.type}` }));
            }
        } catch (error) {
            // Catch errors from JSON.parse or other synchronous issues before the switch
            if (error instanceof SyntaxError && error.message.includes("JSON")) {
                console.error('Error parsing incoming WebSocket message as JSON. Raw string was:', messageString, error);
                ws.send(JSON.stringify({ type: 'error', message: 'Invalid message format received (not valid JSON).' }));
            } else {
                // General error handling for other types of errors within the message handler
                const typeForError = parsedMessage ? parsedMessage.type : 'unknown/unparsable';
                console.error(`Critical error processing WebSocket message (type: ${typeForError}). Raw string (truncated): ${messageString.substring(0,200)}. Error:`, error.message, error.stack);
                ws.send(JSON.stringify({ type: 'error', message: `An unexpected critical error occurred on the server: ${error.message}` }));
            }
        }
    });
}); // End of wss.on('connection')


// Graceful shutdown logic
const cleanup = (signal) => {
    console.log(`\nReceived ${signal}. Closing server and database...`);
    wss.clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.close(1001, "Server is shutting down."); // 1001: Going Away
        }
    });
    
    wss.close(() => {
        console.log('WebSocket server closed.');
        server.close(() => {
            console.log('HTTP server closed.');
            db.close((err) => {
                if (err) {
                    console.error('Error closing database', err.message);
                } else {
                    console.log('Database connection closed.');
                }
                console.log("Exiting process.");
                process.exit(0);
            });
        });
    });

    // Force exit if cleanup takes too long
    setTimeout(() => {
        console.error("Graceful shutdown timed out. Forcing exit.");
        process.exit(1);
    }, 5000); // 5 seconds timeout
};

process.on('SIGINT', () => cleanup('SIGINT')); // Ctrl+C
process.on('SIGTERM', () => cleanup('SIGTERM')); // kill command

// Start the HTTP server (which the WebSocket server is attached to)
server.listen(PORT, () => {
    console.log(`HTTP and WebSocket server is running on port ${PORT}`);
    console.log(`OpenAI API Base URL: ${OPENAI_API_BASE_URL}`);
});
