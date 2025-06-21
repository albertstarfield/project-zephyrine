// In a real project, this would be in a new file, e.g., websocket_handler.go
// with "package main" at the top.

package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
	"io"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/sashabaranov/go-openai"
)

// --- WebSocket Message Structures ---

// WebSocketMessage is the generic wrapper for all incoming messages.
type WebSocketMessage struct {
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload"`
}

// GenericPayload is used to extract common fields like userId and chatId.
type GenericPayload struct {
	UserID string `json:"userId"`
	ChatID string `json:"chatId"`
}

// WSMessage is the structure for sending messages to the client.
type WSMessage struct {
	Type    string      `json:"type"`
	Payload interface{} `json:"payload,omitempty"`
}

// ChatPayload is the specific payload for 'chat' type messages.
type ChatPayload struct {
	Messages              []openai.ChatCompletionMessage `json:"messages"`
	Model                 string                         `json:"model"`
	ChatID                string                         `json:"chatId"`
	UserID                string                         `json:"userId"`
	FirstUserMessageContent string                        `json:"firstUserMessageContent"`
}

// EditMessagePayload defines the structure for the 'edit_message' event.
type EditMessagePayload struct {
    MessageIDToUpdateInDB string `json:"messageIdToUpdateInDB"`
    NewContent            string `json:"newContent"`
    ChatID                string `json:"chatId"`
    UserID                string `json:"userId"`
    HistoryForRegen       []openai.ChatCompletionMessage `json:"historyForRegen"`
    Model                 string `json:"model"`
}


// --- WebSocket Handler ---

// serveWs handles websocket requests from clients.
func (app *App) serveWs(w http.ResponseWriter, r *http.Request) {
	conn, err := app.WebsocketUpgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("WebSocket upgrade error:", err)
		return
	}
	defer conn.Close()

	log.Println("Client connected via WebSocket")

	// This loop handles incoming messages from a single client connection.
	for {
		messageType, message, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket read error: %v", err)
			} else {
				log.Println("Client disconnected gracefully.")
			}
			break
		}
		if messageType != websocket.TextMessage {
			log.Println("Received non-text message, ignoring.")
			continue
		}

		// Process the message in a separate goroutine to not block the read loop.
		// This is a simple approach. For high-concurrency, a worker pool might be better.
		go app.handleWebSocketMessage(conn, message)
	}
}

// handleWebSocketMessage decodes and routes a message to the correct handler function.
func (app *App) handleWebSocketMessage(conn *websocket.Conn, rawMessage []byte) {
	var msg WebSocketMessage
	if err := json.Unmarshal(rawMessage, &msg); err != nil {
		log.Printf("Error parsing WebSocket JSON: %v. Raw message: %s", err, string(rawMessage))
		sendWsError(conn, "Invalid message format (not valid JSON).")
		return
	}
    
    // It's useful to extract common payload fields early for logging
    var commonPayload GenericPayload
    _ = json.Unmarshal(msg.Payload, &commonPayload)
	log.Printf("Processing WebSocket message: type=%s, chatId=%s, userId=%s", msg.Type, commonPayload.ChatID, commonPayload.UserID)

	ctx := context.Background() // Create a new context for this operation

	// The big switch, similar to the Node.js version
	switch msg.Type {
	case "get_messages":
		app.wsGetMessages(ctx, conn, msg.Payload)
	case "get_chat_history_list":
		app.wsGetChatHistoryList(ctx, conn, msg.Payload)
	case "chat":
		app.wsChat(ctx, conn, msg.Payload)
    case "rename_chat":
        app.wsRenameChat(ctx, conn, msg.Payload)
    case "delete_chat":
        app.wsDeleteChat(ctx, conn, msg.Payload)
    case "edit_message":
        app.wsEditMessage(ctx, conn, msg.Payload)
    case "stop":
        // In a real implementation, you'd use a context with cancellation
        // to stop the OpenAI stream in the `wsChat` handler.
        log.Printf("Stop request received for chat: %s. Backend stop logic placeholder.", commonPayload.ChatID)
        sendWsMessage(conn, "stopped", map[string]string{"message": "Generation stopped by backend (simulated)."})
	default:
		log.Printf("Unknown message type received: %s", msg.Type)
		sendWsError(conn, fmt.Sprintf("Unknown message type: %s", msg.Type))
	}
}

// --- WebSocket Message Handlers (The individual `case` blocks) ---

func (app *App) wsGetMessages(ctx context.Context, conn *websocket.Conn, payload json.RawMessage) {
	var p GenericPayload
	if err := json.Unmarshal(payload, &p); err != nil || p.ChatID == "" {
		sendWsMessage(conn, "chat_history_error", map[string]string{"message": "Chat ID missing or invalid payload for get_messages"})
		return
	}

	rows, err := app.DB.QueryContext(ctx, "SELECT id, chat_id, user_id, sender, content, created_at FROM messages WHERE chat_id = ? ORDER BY created_at ASC", p.ChatID)
	if err != nil {
		log.Printf("Error fetching messages for chat %s: %v", p.ChatID, err)
		sendWsMessage(conn, "chat_history_error", map[string]string{"chatId": p.ChatID, "error": err.Error()})
		return
	}
	defer rows.Close()

	type Message struct {
		ID        string `json:"id"`
		ChatID    string `json:"chat_id"`
		UserID    string `json:"user_id"`
		Sender    string `json:"sender"`
		Content   string `json:"content"`
		CreatedAt string `json:"created_at"`
	}
	var messages []Message
	for rows.Next() {
		var m Message
		if err := rows.Scan(&m.ID, &m.ChatID, &m.UserID, &m.Sender, &m.Content, &m.CreatedAt); err != nil {
			log.Printf("Error scanning message row for chat %s: %v", p.ChatID, err)
			continue // Skip bad rows
		}
		messages = append(messages, m)
	}

	sendWsMessage(conn, "chat_history", map[string]interface{}{"chatId": p.ChatID, "messages": messages})
}

func (app *App) wsGetChatHistoryList(ctx context.Context, conn *websocket.Conn, payload json.RawMessage) {
	var p GenericPayload
	if err := json.Unmarshal(payload, &p); err != nil || p.UserID == "" {
		sendWsError(conn, "User ID missing or invalid payload for get_chat_history_list")
		return
	}

	rows, err := app.DB.QueryContext(ctx, "SELECT id, title, updated_at FROM chats WHERE user_id = ? ORDER BY updated_at DESC", p.UserID)
	if err != nil {
		log.Printf("Error fetching chat history list for user %s: %v", p.UserID, err)
		sendWsError(conn, "Failed to fetch chat history list from DB.")
		return
	}
	defer rows.Close()

	type ChatInfo struct {
		ID        string `json:"id"`
		Title     string `json:"title"`
		UpdatedAt string `json:"updated_at"`
	}
	var chats []ChatInfo
	for rows.Next() {
		var c ChatInfo
		if err := rows.Scan(&c.ID, &c.Title, &c.UpdatedAt); err != nil {
			log.Printf("Error scanning chat info row for user %s: %v", p.UserID, err)
			continue
		}
		chats = append(chats, c)
	}

	log.Printf("Sending chat history list for user %s: %d chats", p.UserID, len(chats))
	sendWsMessage(conn, "chat_history_list", map[string]interface{}{"chats": chats})
}

func (app *App) wsChat(ctx context.Context, conn *websocket.Conn, payload json.RawMessage) {
    var p ChatPayload
    if err := json.Unmarshal(payload, &p); err != nil {
        sendWsError(conn, "Invalid payload for chat message.")
        return
    }
	if len(p.Messages) == 0 || p.ChatID == "" || p.UserID == "" {
        sendWsError(conn, "Invalid payload content for chat message.")
		return
	}

    // 1. Check if chat exists, create if not
    var chatIDExists string
    err := app.DB.QueryRowContext(ctx, "SELECT id FROM chats WHERE id = ? AND user_id = ?", p.ChatID, p.UserID).Scan(&chatIDExists)

    if err == sql.ErrNoRows {
        // Chat doesn't exist, create it.
        chatTitle := fmt.Sprintf("Chat (%s)", p.Model)
        latestUserMsgForTitle := p.FirstUserMessageContent
        if latestUserMsgForTitle == "" {
            latestUserMsgForTitle = p.Messages[len(p.Messages)-1].Content
        }

        // Try to generate a title with the LLM
        titleReq := openai.ChatCompletionRequest{
            Model:     p.Model,
            Messages:  []openai.ChatCompletionMessage{
                {Role: openai.ChatMessageRoleUser, Content: fmt.Sprintf(`Generate a very short, concise title (3-5 words maximum) for a new chat that begins with: "%s"`, latestUserMsgForTitle)},
            },
            MaxTokens:   25,
            Temperature: 0.5,
        }
        titleResp, titleErr := app.OpenAIClient.CreateChatCompletion(ctx, titleReq)
        if titleErr == nil && len(titleResp.Choices) > 0 {
            generatedTitle := strings.Trim(titleResp.Choices[0].Message.Content, ` "`)
            if generatedTitle != "" {
                chatTitle = generatedTitle
            }
        } else {
             log.Printf("Error generating chat title with LLM: %v", titleErr)
             // Fallback title
             if len(latestUserMsgForTitle) > 30 {
                 chatTitle = fmt.Sprintf("Chat: %s...", latestUserMsgForTitle[:30])
             } else {
                chatTitle = fmt.Sprintf("Chat: %s", latestUserMsgForTitle)
             }
        }

        _, dbErr := app.DB.ExecContext(ctx, "INSERT INTO chats (id, user_id, title) VALUES (?, ?, ?)", p.ChatID, p.UserID, chatTitle)
        if dbErr != nil {
            log.Printf("Error inserting new chat %s: %v", p.ChatID, dbErr)
            sendWsError(conn, "Failed to create new chat in database.")
            return
        }
        log.Printf("New chat %s created with title '%s'", p.ChatID, chatTitle)
        sendWsMessage(conn, "title_updated", map[string]string{"chatId": p.ChatID, "title": chatTitle})

    } else if err != nil {
        log.Printf("Error checking if chat exists %s: %v", p.ChatID, err)
        sendWsError(conn, "Database error when checking for chat.")
        return
    }

    // 2. Save user message to DB
	// Note: The JS version handles multi-part messages (text/image). Go-openai also supports this.
	// For simplicity here, we'll assume content is a simple string, which it is in most cases.
    lastUserMessage := p.Messages[len(p.Messages)-1]
    userMessageId := uuid.New().String()
    _, err = app.DB.ExecContext(ctx, "INSERT INTO messages (id, chat_id, user_id, sender, content) VALUES (?, ?, ?, ?, ?)",
        userMessageId, p.ChatID, p.UserID, "user", lastUserMessage.Content)
    if err != nil {
        log.Printf("Error saving user message for chat %s: %v", p.ChatID, err)
        // Non-fatal, we can still proceed with the chat
    }

    // 3. Stream response from LLM
    streamReq := openai.ChatCompletionRequest{
        Model:       p.Model,
        Messages:    p.Messages,
        Stream:      true,
        Temperature: app.Config.LLMTemperature,
        MaxTokens:   app.Config.LLMTopCapTokens,
        TopP:        app.Config.LLMTopP,
    }

    stream, err := app.OpenAIClient.CreateChatCompletionStream(ctx, streamReq)
    if err != nil {
        log.Printf("Error creating LLM stream for chat %s: %v", p.ChatID, err)
        sendWsError(conn, "Failed to start chat with the language model.")
        return
    }
    defer stream.Close()

    var fullResponse strings.Builder
    for {
        response, err := stream.Recv()
        if errors.Is(err, io.EOF) {
            sendWsMessage(conn, "end", nil)
            break
        }
        if err != nil {
            log.Printf("LLM stream error for chat %s: %v", p.ChatID, err)
            sendWsError(conn, "An error occurred during streaming.")
            break
        }
        content := response.Choices[0].Delta.Content
        if content != "" {
            fullResponse.WriteString(content)
            sendWsMessage(conn, "chunk", map[string]string{"content": content})
        }
    }

    // 4. Save assistant response to DB and update chat timestamp
    assistantResponse := strings.TrimSpace(fullResponse.String())
    if assistantResponse != "" {
        assistantMessageId := uuid.New().String()
        _, err = app.DB.ExecContext(ctx, "INSERT INTO messages (id, chat_id, user_id, sender, content) VALUES (?, ?, ?, ?, ?)",
            assistantMessageId, p.ChatID, p.UserID, "assistant", assistantResponse)
        if err != nil {
            log.Printf("Error saving assistant message for chat %s: %v", p.ChatID, err)
        }

        _, err = app.DB.ExecContext(ctx, "UPDATE chats SET updated_at = datetime('now', 'localtime') WHERE id = ?", p.ChatID)
        if err != nil {
            log.Printf("Error updating chat timestamp for chat %s: %v", p.ChatID, err)
        }
    }
}

func (app *App) wsRenameChat(ctx context.Context, conn *websocket.Conn, payload json.RawMessage) {
    var p struct {
        ChatID   string `json:"chatId"`
        NewTitle string `json:"newTitle"`
        UserID   string `json:"userId"`
    }
    if err := json.Unmarshal(payload, &p); err != nil || p.ChatID == "" || p.UserID == "" {
        sendWsMessage(conn, "rename_chat_error", map[string]string{"chatId": p.ChatID, "error": "Invalid payload for rename_chat."})
        return
    }

    res, err := app.DB.ExecContext(ctx, "UPDATE chats SET title = ?, updated_at = datetime('now', 'localtime') WHERE id = ? AND user_id = ?",
        p.NewTitle, p.ChatID, p.UserID)
    if err != nil {
        log.Printf("Error renaming chat %s: %v", p.ChatID, err)
        sendWsMessage(conn, "rename_chat_error", map[string]string{"chatId": p.ChatID, "error": err.Error()})
        return
    }

    rowsAffected, _ := res.RowsAffected()
    if rowsAffected == 0 {
        log.Printf("Rename chat: Chat %s not found or user %s not authorized.", p.ChatID, p.UserID)
        sendWsMessage(conn, "rename_chat_error", map[string]string{"chatId": p.ChatID, "error": "Chat not found or not authorized."})
        return
    }
    
    log.Printf("Chat %s renamed to '%s' for user %s", p.ChatID, p.NewTitle, p.UserID)
    sendWsMessage(conn, "chat_renamed", map[string]string{"chatId": p.ChatID, "newTitle": p.NewTitle})
}

func (app *App) wsDeleteChat(ctx context.Context, conn *websocket.Conn, payload json.RawMessage) {
    var p GenericPayload
    if err := json.Unmarshal(payload, &p); err != nil || p.ChatID == "" || p.UserID == "" {
        sendWsMessage(conn, "delete_chat_error", map[string]string{"chatId": p.ChatID, "error": "Invalid payload for delete_chat."})
        return
    }

    res, err := app.DB.ExecContext(ctx, "DELETE FROM chats WHERE id = ? AND user_id = ?", p.ChatID, p.UserID)
    if err != nil {
        log.Printf("Error deleting chat %s: %v", p.ChatID, err)
        sendWsMessage(conn, "delete_chat_error", map[string]string{"chatId": p.ChatID, "error": err.Error()})
        return
    }

    rowsAffected, _ := res.RowsAffected()
    if rowsAffected == 0 {
        log.Printf("Delete chat: Chat %s not found or user %s not authorized.", p.ChatID, p.UserID)
        sendWsMessage(conn, "delete_chat_error", map[string]string{"chatId": p.ChatID, "error": "Chat not found or not authorized."})
        return
    }

    log.Printf("Chat %s deleted for user %s", p.ChatID, p.UserID)
    sendWsMessage(conn, "chat_deleted", map[string]string{"chatId": p.ChatID})
}


func (app *App) wsEditMessage(ctx context.Context, conn *websocket.Conn, payload json.RawMessage) {
    var p EditMessagePayload
    if err := json.Unmarshal(payload, &p); err != nil {
        sendWsError(conn, "Invalid payload for edit_message.")
        return
    }
    if p.MessageIDToUpdateInDB == "" || p.ChatID == "" || p.UserID == "" || len(p.HistoryForRegen) == 0 || p.Model == "" {
        sendWsError(conn, "Invalid payload content for edit_message.")
        return
    }
    
    // 1. Update the user's message in the database.
    res, err := app.DB.ExecContext(ctx, "UPDATE messages SET content = ?, created_at = datetime('now', 'localtime') WHERE id = ? AND user_id = ? AND chat_id = ?",
        p.NewContent, p.MessageIDToUpdateInDB, p.UserID, p.ChatID)
    if err != nil {
        log.Printf("Error updating message in DB: %v", err)
        sendWsMessage(conn, "message_update_error", map[string]string{"messageId": p.MessageIDToUpdateInDB, "error": err.Error()})
        return
    }
    rowsAffected, _ := res.RowsAffected()
    if rowsAffected == 0 {
        log.Printf("No message updated in DB (not found or not owned): %s", p.MessageIDToUpdateInDB)
        // We can still proceed, but log a warning.
    } else {
        log.Printf("Message %s updated in DB.", p.MessageIDToUpdateInDB)
    }

    sendWsMessage(conn, "message_updated", map[string]string{"id": p.MessageIDToUpdateInDB, "newContent": p.NewContent})
    
    // 2. The rest of the logic is identical to the main chat handler, just with the new history.
    // To avoid duplication, you could refactor this streaming logic into a shared function.
    // For this direct translation, we'll repeat it.
    
    streamReq := openai.ChatCompletionRequest{
        Model:       p.Model,
        Messages:    p.HistoryForRegen,
        Stream:      true,
        Temperature: app.Config.LLMTemperature,
        MaxTokens:   app.Config.LLMTopCapTokens,
        TopP:        app.Config.LLMTopP,
    }
    stream, err := app.OpenAIClient.CreateChatCompletionStream(ctx, streamReq)
    if err != nil {
        log.Printf("Error creating LLM stream for edit on chat %s: %v", p.ChatID, err)
        sendWsMessage(conn, "message_update_error", map[string]string{"messageId": p.MessageIDToUpdateInDB, "error": "Failed to start regeneration."})
        return
    }
    defer stream.Close()

    var fullResponse strings.Builder
    for {
        response, err := stream.Recv()
        if errors.Is(err, io.EOF) {
            sendWsMessage(conn, "end", nil)
            break
        }
        if err != nil {
            log.Printf("LLM stream error for edit on chat %s: %v", p.ChatID, err)
            sendWsMessage(conn, "message_update_error", map[string]string{"messageId": p.MessageIDToUpdateInDB, "error": "An error occurred during streaming."})
            break
        }
        content := response.Choices[0].Delta.Content
        if content != "" {
            fullResponse.WriteString(content)
            sendWsMessage(conn, "chunk", map[string]string{"content": content})
        }
    }

    // 3. Save the new assistant response.
    assistantResponse := strings.TrimSpace(fullResponse.String())
    if assistantResponse != "" {
        newAssistantMessageId := uuid.New().String()
        _, err = app.DB.ExecContext(ctx, "INSERT INTO messages (id, chat_id, user_id, sender, content) VALUES (?, ?, ?, ?, ?)",
            newAssistantMessageId, p.ChatID, p.UserID, "assistant", assistantResponse)
        if err != nil {
            log.Printf("Error saving new assistant message after edit for chat %s: %v", p.ChatID, err)
        }
        
        _, err = app.DB.ExecContext(ctx, "UPDATE chats SET updated_at = datetime('now', 'localtime') WHERE id = ?", p.ChatID)
        if err != nil {
            log.Printf("Error updating chat timestamp after edit for chat %s: %v", p.ChatID, err)
        }
    }
}


// --- WebSocket Helper Functions ---

func sendWsMessage(conn *websocket.Conn, msgType string, payload interface{}) {
	msg := WSMessage{Type: msgType, Payload: payload}
	// Use a write mutex for concurrent writes if you have multiple goroutines writing to the same conn
	// For this model (one read loop, one write goroutine per message), it's safe.
	if err := conn.WriteJSON(msg); err != nil {
		log.Printf("Error sending WebSocket message: %v", err)
	}
}

func sendWsError(conn *websocket.Conn, message string) {
	sendWsMessage(conn, "error", map[string]string{"message": message})
}