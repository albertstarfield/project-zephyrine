// ExternalAnalyzer/backend-service/server.js
require('dotenv').config();
const WebSocket = require('ws');
const http = require('http');
const { OpenAI } = require('openai');
const sqlite3 = require('sqlite3').verbose();
const { v4: uuidv4 } = require('uuid');

// Configuration
const PORT = process.env.PORT || 3001;
const OPENAI_API_BASE_URL = process.env.OPENAI_API_BASE_URL || 'http://localhost:11434/v1';
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || 'ollama';
const DB_PATH = './project_zephyrine_chats.db';

// LLM Parameters
const LLM_TEMPERATURE = parseFloat(process.env.LLM_TEMPERATURE) || 0.7;
const LLM_MAX_TOKENS = parseInt(process.env.LLM_MAX_TOKENS) || 2048;
const LLM_TOP_P = parseFloat(process.env.LLM_TOP_P) || 1.0;

// Initialize OpenAI client
const openai = new OpenAI({
    baseURL: OPENAI_API_BASE_URL,
    apiKey: OPENAI_API_KEY,
});

// Database setup
const db = new sqlite3.Database(DB_PATH, (err) => {
    if (err) {
        console.error('Error opening database', err.message);
        process.exit(1); 
    } else {
        console.log('Connected to the SQLite database at', DB_PATH);
        db.serialize(() => {
            db.run(`CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                user_id TEXT, 
                title TEXT,
                created_at DATETIME DEFAULT (datetime('now', 'localtime')),
                updated_at DATETIME DEFAULT (datetime('now', 'localtime'))
            )`, (err) => {
                if (err) console.error("Error creating chats table:", err.message);
            });
            db.run(`CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, 
                chat_id TEXT NOT NULL,
                user_id TEXT,
                sender TEXT NOT NULL, 
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )`, (err) => {
                if (err) console.error("Error creating messages table:", err.message);
            });
            // Optional users table
            // db.run(`CREATE TABLE IF NOT EXISTS users (...)`);
        });
    }
});

// HTTP server and WebSocket server
const server = http.createServer((req, res) => {
    if (req.url === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'ok', message: 'WebSocket server is healthy' }));
    } else {
        res.writeHead(200, { 'Content-Type': 'text/plain' });
        res.end('Project Zephyrine WebSocket server is running.');
    }
});
const wss = new WebSocket.Server({ server });
console.log(`WebSocket server setup on port ${PORT}`);

// Map to store clients by userId to send targeted updates
const clients = new Map();

wss.on('connection', (ws) => {
    console.log('Client connected via WebSocket');
    // For targeted updates, we might need to associate ws with a userId upon authentication
    // For now, broadcasts or client-initiated refreshes will handle updates.
    // A simple way to track for broadcast (though not ideal for multi-user per user_id specific updates):
    // const clientId = uuidv4(); 
    // clients.set(clientId, ws);


    ws.on('close', () => {
        console.log('Client disconnected');
        // clients.delete(clientId);
    });

    ws.on('error', (error) => {
        console.error('WebSocket error for a client:', error);
    });

    ws.on("message", async (message) => {
        const messageString = message.toString();
        console.log("Received raw message string (truncated):", messageString.substring(0, 250) + (messageString.length > 250 ? "..." : ""));
        
        let parsedMessage;
        let payload;

        try {
            parsedMessage = JSON.parse(messageString);
            payload = parsedMessage.payload || {};
            const userId = payload.userId; // Assuming userId is part of most payloads for user-specific actions

            console.log(`Processing WebSocket message: type = ${parsedMessage.type}, chatId = ${payload.chatId || 'N/A'}, userId = ${userId || 'N/A'}`);

            switch (parsedMessage.type) {
                case 'get_messages':
                    // ... (your existing get_messages logic)
                    if (!payload.chatId) {
                        ws.send(JSON.stringify({ type: 'chat_history_error', message: 'Chat ID missing for get_messages' }));
                        return;
                    }
                    db.all("SELECT id, chat_id, user_id, sender, content, created_at FROM messages WHERE chat_id = ? ORDER BY created_at ASC", 
                        [payload.chatId], 
                        (err, rows) => {
                            if (err) {
                                console.error(`Error fetching messages for chat ${payload.chatId}:`, err.message);
                                ws.send(JSON.stringify({ type: 'chat_history_error', payload: { chatId: payload.chatId, error: err.message } }));
                            } else {
                                ws.send(JSON.stringify({ type: 'chat_history', payload: { chatId: payload.chatId, messages: rows } }));
                            }
                        });
                    break;

                case 'get_chat_history_list':
                    if (!userId) {
                        ws.send(JSON.stringify({ type: 'error', message: 'User ID missing for get_chat_history_list' }));
                        return;
                    }
                    try {
                        db.all("SELECT id, title, updated_at FROM chats WHERE user_id = ? ORDER BY updated_at DESC", 
                            [userId], 
                            (err, rows) => {
                                if (err) {
                                    console.error("Error fetching chat history list for user", userId, ":", err.message);
                                    ws.send(JSON.stringify({ type: 'error', message: 'Failed to fetch chat history list from DB.' }));
                                } else {
                                    console.log(`Sending chat history list for user ${userId}: ${rows.length} chats`);
                                    ws.send(JSON.stringify({ type: 'chat_history_list', payload: { chats: rows } }));
                                }
                            }
                        );
                    } catch (dbError) {
                        console.error("Exception fetching chat history list:", dbError.message);
                        ws.send(JSON.stringify({ type: 'error', message: 'Server exception fetching chat history list.' }));
                    }
                    break;

                case 'chat':
                    // ... (your existing detailed 'chat' logic from previous steps)
                    // This should include creating a chat if it doesn't exist, saving user message,
                    // calling LLM, streaming response, and saving assistant message.
                    // Ensure it also sends a 'title_updated' or similar message that client can use
                    // to refresh chat history list.
                    // (Assuming the full 'chat' case from before is here)
                    const { messages: clientMessages, model, chatId: currentChatId, firstUserMessageContent } = payload;
                    if (!clientMessages || clientMessages.length === 0 || !currentChatId || !userId) {
                        ws.send(JSON.stringify({ type: 'error', message: 'Invalid payload for chat message.' }));
                        return;
                    }
                    const userMessageContent = clientMessages[clientMessages.length - 1].content;

                    try {
                        const chatExists = await new Promise((resolve, reject) => {
                            db.get("SELECT id FROM chats WHERE id = ? AND user_id = ?", [currentChatId, userId], (err, row) => {
                                if (err) { console.error("Error checking if chat exists:", err.message); reject(err); } else { resolve(row); }
                            });
                        });

                        if (!chatExists) {
                            let chatTitle = `Chat (${model})`;
                            const latestUserMsgForTitle = firstUserMessageContent || userMessageContent;
                            if (latestUserMsgForTitle && clientMessages.filter(m => (m.role === 'user' || m.sender === 'user')).length === 1) {
                                try {
                                    const titlePrompt = [{ role: "user", content: `Generate a very short, concise title (3-5 words maximum) for a new chat that begins with: "${latestUserMsgForTitle}"` }];
                                    const titleCompletion = await openai.chat.completions.create({
                                        messages: titlePrompt, model: model, max_tokens: 25, temperature: 0.5,
                                    });
                                    const generatedTitle = titleCompletion.choices[0]?.message?.content?.trim();
                                    if (generatedTitle) chatTitle = generatedTitle.replace(/["']/g, '');
                                } catch (titleError) {
                                    console.error("Error generating chat title with LLM:", titleError.message);
                                    chatTitle = `Chat: ${latestUserMsgForTitle.substring(0, 30)}...`;
                                }
                            } else if (userMessageContent) {
                                chatTitle = `Chat: ${userMessageContent.substring(0, 30)}...`;
                            }
                            await new Promise((resolve, reject) => {
                                db.run( "INSERT INTO chats (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, datetime('now', 'localtime'), datetime('now', 'localtime'))",
                                    [currentChatId, userId, chatTitle], function (err) {
                                        if (err) { console.error("Error inserting new chat:", err.message); reject(err); }
                                        else {
                                            console.log(`New chat ${currentChatId} created with title "${chatTitle}"`);
                                            ws.send(JSON.stringify({ type: 'title_updated', payload: { chatId: currentChatId, title: chatTitle } }));
                                            resolve(this.lastID);
                                        }
                                    }
                                );
                            });
                        }
                        const userMessageId = uuidv4();
                        await new Promise((resolve, reject) => {
                            db.run("INSERT INTO messages (id, chat_id, user_id, sender, content, created_at) VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))",
                                [userMessageId, currentChatId, userId, 'user', userMessageContent], function(err) {
                                if (err) { console.error("Error saving user message:", err.message); reject(err); } else { resolve(this.lastID); }
                            });
                        });

                        const llmStream = await openai.chat.completions.create({
                            messages: clientMessages.map(m => ({ role: m.role || m.sender, content: m.content })),
                            model: model, stream: true, temperature: LLM_TEMPERATURE, max_tokens: LLM_MAX_TOKENS, top_p: LLM_TOP_P,
                        });
                        let assistantResponse = '';
                        for await (const chunk of llmStream) {
                            const content = chunk.choices[0]?.delta?.content || '';
                            if (content) {
                                assistantResponse += content;
                                ws.send(JSON.stringify({ type: 'chunk', payload: { content: content } }));
                            }
                        }
                        ws.send(JSON.stringify({ type: 'end' }));
                        if (assistantResponse.trim()) {
                            const assistantMessageId = uuidv4();
                            await new Promise((resolve, reject) => {
                                db.run("INSERT INTO messages (id, chat_id, user_id, sender, content, created_at) VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))",
                                    [assistantMessageId, currentChatId, userId, 'assistant', assistantResponse.trim()], function(err) {
                                    if (err) { console.error("Error saving assistant message:", err.message); reject(err); } else { resolve(this.lastID); }
                                });
                            });
                            db.run("UPDATE chats SET updated_at = datetime('now', 'localtime') WHERE id = ?", [currentChatId]);
                        }
                    } catch (chatProcessingError) {
                        console.error(`Error in 'chat' case for chatID ${currentChatId}:`, chatProcessingError.message);
                        ws.send(JSON.stringify({ type: 'error', message: `Server error processing chat: ${chatProcessingError.message}` }));
                    }
                    break;

                // --- NEW CASE FOR RENAMING CHAT ---
                case 'rename_chat':
                    if (!payload.chatId || typeof payload.newTitle !== 'string' || !userId) {
                        ws.send(JSON.stringify({ type: 'rename_chat_error', payload: { chatId: payload.chatId, error: 'Invalid payload for rename_chat.' } }));
                        return;
                    }
                    try {
                        db.run("UPDATE chats SET title = ?, updated_at = datetime('now', 'localtime') WHERE id = ? AND user_id = ?",
                            [payload.newTitle, payload.chatId, userId],
                            function(err) {
                                if (err) {
                                    console.error(`Error renaming chat ${payload.chatId}:`, err.message);
                                    ws.send(JSON.stringify({ type: 'rename_chat_error', payload: { chatId: payload.chatId, error: err.message } }));
                                } else if (this.changes === 0) {
                                    console.warn(`Rename chat: Chat ${payload.chatId} not found or user ${userId} not authorized.`);
                                    ws.send(JSON.stringify({ type: 'rename_chat_error', payload: { chatId: payload.chatId, error: 'Chat not found or not authorized.' } }));
                                } else {
                                    console.log(`Chat ${payload.chatId} renamed to "${payload.newTitle}" for user ${userId}`);
                                    ws.send(JSON.stringify({ type: 'chat_renamed', payload: { chatId: payload.chatId, newTitle: payload.newTitle } }));
                                    // Optionally, send an updated chat list to this specific user/client
                                    // Or client can re-fetch chat list upon 'chat_renamed' success.
                                }
                            }
                        );
                    } catch (dbError) {
                        console.error("Exception during rename_chat:", dbError.message);
                        ws.send(JSON.stringify({ type: 'rename_chat_error', payload: { chatId: payload.chatId, error: 'Server exception during rename.' } }));
                    }
                    break;

                // --- NEW CASE FOR DELETING CHAT ---
                case 'delete_chat':
                    if (!payload.chatId || !userId) {
                        ws.send(JSON.stringify({ type: 'delete_chat_error', payload: { chatId: payload.chatId, error: 'Invalid payload for delete_chat.' } }));
                        return;
                    }
                    try {
                        // The ON DELETE CASCADE on messages.chat_id should handle deleting messages
                        db.run("DELETE FROM chats WHERE id = ? AND user_id = ?",
                            [payload.chatId, userId],
                            function(err) {
                                if (err) {
                                    console.error(`Error deleting chat ${payload.chatId}:`, err.message);
                                    ws.send(JSON.stringify({ type: 'delete_chat_error', payload: { chatId: payload.chatId, error: err.message } }));
                                } else if (this.changes === 0) {
                                    console.warn(`Delete chat: Chat ${payload.chatId} not found or user ${userId} not authorized.`);
                                    ws.send(JSON.stringify({ type: 'delete_chat_error', payload: { chatId: payload.chatId, error: 'Chat not found or not authorized.' } }));
                                } else {
                                    console.log(`Chat ${payload.chatId} deleted for user ${userId}`);
                                    ws.send(JSON.stringify({ type: 'chat_deleted', payload: { chatId: payload.chatId } }));
                                    // Optionally, send an updated chat list to this specific user/client
                                    // Or client can re-fetch chat list upon 'chat_deleted' success.
                                }
                            }
                        );
                    } catch (dbError) {
                        console.error("Exception during delete_chat:", dbError.message);
                        ws.send(JSON.stringify({ type: 'delete_chat_error', payload: { chatId: payload.chatId, error: 'Server exception during delete.' } }));
                    }
                    break;

                case 'edit_message':
                    // ... (your existing edit_message logic)
                     const { messageIdToUpdateInDB, newContent: editNewContent, chatId: editChatId, historyForRegen, model: editModel } = payload;
                    if (!messageIdToUpdateInDB || typeof editNewContent !== 'string' || !editChatId || !userId || !historyForRegen || !editModel) {
                         ws.send(JSON.stringify({ type: 'message_update_error', payload: { messageId: messageIdToUpdateInDB, error: 'Invalid payload for edit_message.' } }));
                         return;
                     }
                     try {
                         await new Promise((resolve, reject) => {
                             db.run("UPDATE messages SET content = ?, created_at = datetime('now', 'localtime') WHERE id = ? AND user_id = ? AND chat_id = ?",
                                 [editNewContent, messageIdToUpdateInDB, userId, editChatId], function (err) {
                                 if (err) { console.error("Error updating message in DB:", err.message); reject(err); }
                                 else if (this.changes === 0) { console.warn("No message updated in DB (not found or not owned)."); resolve(false); /* Indicate no update */ }
                                 else { console.log(`Message ${messageIdToUpdateInDB} updated in DB.`); resolve(true); /* Indicate success */ }
                             });
                         });
 
                         ws.send(JSON.stringify({ type: 'message_updated', payload: { id: messageIdToUpdateInDB, newContent: editNewContent } }));
 
                         const llmEditStream = await openai.chat.completions.create({
                             messages: historyForRegen.map(m => ({ role: m.sender, content: m.content })), // Assuming historyForRegen has sender/content
                             model: editModel, stream: true, temperature: LLM_TEMPERATURE, max_tokens: LLM_MAX_TOKENS, top_p: LLM_TOP_P,
                         });
                         let editAssistantResponse = '';
                         for await (const chunk of llmEditStream) {
                             const content = chunk.choices[0]?.delta?.content || '';
                             if (content) {
                                 editAssistantResponse += content;
                                 ws.send(JSON.stringify({ type: 'chunk', payload: { content: content } }));
                             }
                         }
                         ws.send(JSON.stringify({ type: 'end' }));
                         if (editAssistantResponse.trim()) {
                             const newAssistantMessageId = uuidv4();
                             await new Promise((resolve, reject) => {
                                 db.run("INSERT INTO messages (id, chat_id, user_id, sender, content, created_at) VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))",
                                     [newAssistantMessageId, editChatId, userId, 'assistant', editAssistantResponse.trim()], function(err) {
                                     if (err) { console.error("Error saving new assistant message after edit:", err.message); reject(err); } else { resolve(this.lastID); }
                                 });
                             });
                             db.run("UPDATE chats SET updated_at = datetime('now', 'localtime') WHERE id = ?", [editChatId]);
                         }
                     } catch (editProcessingError) {
                         console.error("Error processing edit_message:", editProcessingError.message);
                         ws.send(JSON.stringify({ type: 'message_update_error', payload: { messageId: messageIdToUpdateInDB, error: editProcessingError.message } }));
                     }
                     break;


                case 'stop':
                    // ... (your existing stop logic)
                    console.log(`Stop request received for chat: ${payload.chatId}. Backend stop logic placeholder.`);
                    // In a real scenario, you'd find the active LLM stream for this chatId/userId and cancel it.
                    ws.send(JSON.stringify({ type: 'stopped', payload: { message: "Generation stopped by backend (simulated)." } }));
                    break;

                default:
                    console.warn(`Unknown message type received: ${parsedMessage.type}`);
                    ws.send(JSON.stringify({ type: 'error', message: `Unknown message type: ${parsedMessage.type}` }));
            }
        } catch (error) {
            if (error instanceof SyntaxError && error.message.includes("JSON")) {
                console.error('Error parsing incoming WebSocket message as JSON. Raw string was:', messageString, error);
                ws.send(JSON.stringify({ type: 'error', message: 'Invalid message format received (not valid JSON).' }));
            } else {
                const typeForError = parsedMessage ? parsedMessage.type : 'unknown/unparsable';
                console.error(`Critical error processing WebSocket message (type: ${typeForError}). Raw string (truncated): ${messageString.substring(0,200)}. Error:`, error.message, error.stack);
                ws.send(JSON.stringify({ type: 'error', message: `An unexpected critical error occurred on the server: ${error.message}` }));
            }
        }
    });
});

// Graceful shutdown
const cleanup = (signal) => {
    console.log(`\nReceived ${signal}. Closing server and database...`);
    wss.clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.close(1001, "Server is shutting down.");
        }
    });
    wss.close(() => {
        console.log('WebSocket server closed.');
        server.close(() => {
            console.log('HTTP server closed.');
            db.close((err) => {
                if (err) console.error('Error closing database', err.message);
                else console.log('Database connection closed.');
                process.exit(0);
            });
        });
    });
    setTimeout(() => {
        console.error("Graceful shutdown timed out. Forcing exit.");
        process.exit(1);
    }, 5000);
};

process.on('SIGINT', () => cleanup('SIGINT'));
process.on('SIGTERM', () => cleanup('SIGTERM'));

server.listen(PORT, () => {
    console.log(`HTTP and WebSocket server is running on port ${PORT}`);
    console.log(`OpenAI API Base URL: ${OPENAI_API_BASE_URL}`);
});