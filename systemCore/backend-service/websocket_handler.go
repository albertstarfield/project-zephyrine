// backend-service/websocket_handler.go

package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time" // We will now use this to generate timestamps

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/sashabaranov/go-openai"
)

// --- WebSocket Message Structures ---

type WebSocketMessage struct {
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload"`
}

type GenericPayload struct {
	UserID string `json:"userId"`
	ChatID string `json:"chatId"`
}

type WSMessage struct {
	Type    string      `json:"type"`
	Payload interface{} `json:"payload,omitempty"`
}

type ChatPayload struct {
	Messages            []openai.ChatCompletionMessage `json:"messages"`
	Model               string                         `json:"model"`
	ChatID              string                         `json:"chatId"`
	UserID              string                         `json:"userId"`
	OptimisticMessageID string                         `json:"optimisticMessageId"`
}

type EditMessagePayload struct {
	MessageIDToUpdateInDB string                         `json:"messageIdToUpdateInDB"`
	NewContent            string                         `json:"newContent"`
	ChatID                string                         `json:"chatId"`
	UserID                string                         `json:"userId"`
	HistoryForRegen       []openai.ChatCompletionMessage `json:"historyForRegen"`
	Model                 string                         `json:"model"`
}

// --- WebSocket Handler ---

func (app *App) serveWs(w http.ResponseWriter, r *http.Request) {
	conn, err := app.WebsocketUpgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("WebSocket upgrade error:", err)
		return
	}
	defer conn.Close()
	log.Println("Client connected via WebSocket")

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
		if messageType == websocket.TextMessage {
			app.handleWebSocketMessage(conn, message)
		}
	}
}

func (app *App) handleWebSocketMessage(conn *websocket.Conn, rawMessage []byte) {
	var msg WebSocketMessage
	if err := json.Unmarshal(rawMessage, &msg); err != nil {
		log.Printf("Error parsing WebSocket JSON: %v. Raw message: %s", err, string(rawMessage))
		sendWsError(conn, "Invalid message format.")
		return
	}

	ctx := context.Background()

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
		log.Printf("Stop request received. Stop logic placeholder.")
		sendWsMessage(conn, "stopped", map[string]string{"message": "Generation stopped."})
	default:
		log.Printf("Unknown message type received: %s", msg.Type)
		sendWsError(conn, fmt.Sprintf("Unknown message type: %s", msg.Type))
	}
}

// wsChat now orchestrates the entire transactional process for a single user turn.
func (app *App) wsChat(ctx context.Context, conn *websocket.Conn, payload json.RawMessage) {
	var p ChatPayload
	if err := json.Unmarshal(payload, &p); err != nil || p.ChatID == "" || p.UserID == "" || len(p.Messages) == 0 {
		sendWsError(conn, "Invalid payload for chat message.")
		return
	}

	// --- Cancellation Logic ---
	app.ongoingStreamsMux.Lock()
	if cancel, exists := app.ongoingStreams[p.ChatID]; exists {
		log.Printf("New message received for chat %s. Cancelling previous generation.", p.ChatID)
		cancel()
	}
	streamCtx, cancelFunc := context.WithCancel(context.Background())
	app.ongoingStreams[p.ChatID] = cancelFunc
	app.ongoingStreamsMux.Unlock()

	defer func() {
		app.ongoingStreamsMux.Lock()
		delete(app.ongoingStreams, p.ChatID)
		app.ongoingStreamsMux.Unlock()
		log.Printf("Cleaned up stream for chat %s.", p.ChatID)
	}()
	// --- End Cancellation Logic ---

	chatTitle, isNewChat, err := app.ensureChatExistsAndGetTitle(ctx, &p)
	if err != nil {
		log.Printf("CRITICAL ERROR: Failed to ensure chat exists for chat %s: %v", p.ChatID, err)
		sendWsError(conn, fmt.Sprintf("Failed to initialize chat session: %v", err))
		return
	}
	if isNewChat {
		sendWsMessage(conn, "title_updated", map[string]string{"chatId": p.ChatID, "title": chatTitle})
	}

	userMessage := p.Messages[len(p.Messages)-1]
	err = app.saveMessage(ctx, "msg_"+uuid.New().String(), p.ChatID, p.UserID, userMessage.Role, userMessage.Content, time.Now())
	if err != nil {
		log.Printf("CRITICAL ERROR: Failed to save user message for chat %s: %v", p.ChatID, err)
		sendWsError(conn, "Failed to save your message to the database.")
		return
	}
	sendWsMessage(conn, "message_status_update", map[string]string{"id": p.OptimisticMessageID, "status": "delivered"})

	stream, err := app.OpenAIClient.CreateChatCompletionStream(streamCtx, openai.ChatCompletionRequest{
		Model:       p.Model,
		Messages:    p.Messages,
		Stream:      true,
		Temperature: app.Config.LLMTemperature,
		MaxTokens:   app.Config.LLMTopCapTokens,
		TopP:        app.Config.LLMTopP,
	})

	if err != nil {
		if errors.Is(err, context.Canceled) {
			log.Printf("Stream for chat %s was canceled successfully before starting.", p.ChatID)
			return
		}
		log.Printf("Error creating LLM stream for chat %s: %v", p.ChatID, err)
		sendWsError(conn, "Failed to start chat with the language model.")
		return
	}
	defer stream.Close()

	var fullResponse strings.Builder
	for {
		response, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			// Add the ID to the "end" message
			sendWsMessage(conn, "end", map[string]string{
				"optimisticMessageId": p.OptimisticMessageID,
			})
			break
		}
		if err != nil {
			if errors.Is(err, context.Canceled) {
				log.Printf("Stream for chat %s was canceled mid-generation.", p.ChatID)
			} else {
				log.Printf("LLM stream error for chat %s: %v", p.ChatID, err)
				sendWsError(conn, "An error occurred during streaming.")
			}
			break
		}
		content := response.Choices[0].Delta.Content
		if content != "" {
			fullResponse.WriteString(content)
			// Add the ID to every "chunk" message
			sendWsMessage(conn, "chunk", map[string]string{
				"content":             content,
				"optimisticMessageId": p.OptimisticMessageID,
			})
		}
	}

	// --- START: Added Save Guard ---
	// If the stream context was canceled, do not save the (potentially partial) response.
	if streamCtx.Err() == context.Canceled {
		log.Printf("Not saving partial response for canceled stream in chat %s", p.ChatID)
		return
	}
	// --- END: Added Save Guard ---

	assistantResponse := strings.TrimSpace(fullResponse.String())
	if assistantResponse != "" {
		err = app.saveMessage(ctx, "msg_"+uuid.New().String(), p.ChatID, p.UserID, openai.ChatMessageRoleAssistant, assistantResponse, time.Now())
		if err != nil {
			log.Printf("CRITICAL ERROR: Failed to save assistant message for chat %s: %v", p.ChatID, err)
		}
	}
}

// ensureChatExistsAndGetTitle creates a chat if it doesn't exist and returns its title.
func (app *App) ensureChatExistsAndGetTitle(ctx context.Context, p *ChatPayload) (string, bool, error) {
	app.DBMutex.Lock()
	defer app.DBMutex.Unlock()

	var title string
	err := app.DB.QueryRowContext(ctx, "SELECT title FROM chats WHERE id = ?", p.ChatID).Scan(&title)
	if err == nil {
		return title, false, nil
	}
	if err != sql.ErrNoRows {
		return "", false, fmt.Errorf("unexpected error when checking for chat existence: %w", err)
	}

	log.Printf("[Chat %s] Does not exist. Creating now...", p.ChatID)
	newTitle := "New Chat"
	if len(p.Messages) > 0 {
		content := p.Messages[len(p.Messages)-1].Content
		if len(content) > 30 {
			content = content[:30]
		}
		newTitle = content + "..."
	}

	_, dbErr := app.DB.ExecContext(ctx, "INSERT INTO chats (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)", p.ChatID, p.UserID, newTitle, time.Now(), time.Now())
	if dbErr != nil {
		if strings.Contains(dbErr.Error(), "UNIQUE constraint failed") {
			log.Printf("Race condition avoided: Chat %s was created by another thread.", p.ChatID)
			var existingTitle string
			_ = app.DB.QueryRowContext(ctx, "SELECT title FROM chats WHERE id = ?", p.ChatID).Scan(&existingTitle)
			return existingTitle, false, nil
		}
		return "", true, fmt.Errorf("failed to insert new chat record: %w", dbErr)
	}

	log.Printf("New chat %s created with title '%s'", p.ChatID, newTitle)
	return newTitle, true, nil
}

// saveMessage saves a single message and updates the parent chat's timestamp.
func (app *App) saveMessage(ctx context.Context, id, chatID, userID, sender, content string, createdAt time.Time) error {
	app.DBMutex.Lock()
	defer app.DBMutex.Unlock()

	tx, err := app.DB.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	// FIX: Explicitly set created_at from Go
	_, err = tx.ExecContext(ctx, "INSERT INTO messages (id, chat_id, user_id, sender, content, created_at) VALUES (?, ?, ?, ?, ?, ?)", id, chatID, userID, sender, content, createdAt)
	if err != nil {
		return fmt.Errorf("failed to insert message: %w", err)
	}

	// FIX: Update the timestamp using the Go-generated time
	_, err = tx.ExecContext(ctx, "UPDATE chats SET updated_at = ? WHERE id = ?", createdAt, chatID)
	if err != nil {
		return fmt.Errorf("failed to update chat timestamp: %w", err)
	}

	return tx.Commit()
}

func (app *App) wsGetMessages(ctx context.Context, conn *websocket.Conn, payload json.RawMessage) {
	app.DBMutex.Lock()
	defer app.DBMutex.Unlock()
	var p GenericPayload
	if err := json.Unmarshal(payload, &p); err != nil || p.ChatID == "" {
		sendWsMessage(conn, "chat_history_error", map[string]string{"message": "Chat ID missing"})
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
		ID        string    `json:"id"`
		ChatID    string    `json:"chat_id"`
		UserID    string    `json:"user_id"`
		Sender    string    `json:"sender"`
		Content   string    `json:"content"`
		CreatedAt time.Time `json:"created_at"`
	}
	var messages []Message
	for rows.Next() {
		var m Message
		if err := rows.Scan(&m.ID, &m.ChatID, &m.UserID, &m.Sender, &m.Content, &m.CreatedAt); err != nil {
			log.Printf("Error scanning message row for chat %s: %v", p.ChatID, err)
			continue
		}
		messages = append(messages, m)
	}

	sendWsMessage(conn, "chat_history", map[string]interface{}{"chatId": p.ChatID, "messages": messages})
}

func (app *App) wsGetChatHistoryList(ctx context.Context, conn *websocket.Conn, payload json.RawMessage) {
	app.DBMutex.Lock()
	defer app.DBMutex.Unlock()
	var p GenericPayload
	if err := json.Unmarshal(payload, &p); err != nil || p.UserID == "" {
		sendWsError(conn, "User ID missing.")
		return
	}

	rows, err := app.DB.QueryContext(ctx, "SELECT id, title, updated_at FROM chats WHERE user_id = ? ORDER BY updated_at DESC", p.UserID)
	if err != nil {
		log.Printf("Error fetching chat history list for user %s: %v", p.UserID, err)
		sendWsError(conn, "Failed to fetch chat history list.")
		return
	}
	defer rows.Close()

	type ChatInfo struct {
		ID        string    `json:"id"`
		Title     string    `json:"title"`
		UpdatedAt time.Time `json:"updated_at"`
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

	sendWsMessage(conn, "chat_history_list", map[string]interface{}{"chats": chats})
}

func (app *App) wsRenameChat(ctx context.Context, conn *websocket.Conn, payload json.RawMessage) {
	app.DBMutex.Lock()
	defer app.DBMutex.Unlock()

	tx, err := app.DB.BeginTx(ctx, nil)
	if err != nil {
		sendWsError(conn, "Failed to start database operation for rename.")
		return
	}
	defer tx.Rollback()

	var p struct {
		ChatID   string `json:"chatId"`
		NewTitle string `json:"newTitle"`
		UserID   string `json:"userId"`
	}
	if err := json.Unmarshal(payload, &p); err != nil || p.ChatID == "" || p.UserID == "" {
		sendWsMessage(conn, "rename_chat_error", map[string]string{"chatId": p.ChatID, "error": "Invalid payload for rename_chat."})
		return
	}

	res, err := tx.ExecContext(ctx, "UPDATE chats SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
		p.NewTitle, time.Now(), p.ChatID, p.UserID)
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

	if err := tx.Commit(); err != nil {
		log.Printf("Error committing rename chat transaction %s: %v", p.ChatID, err)
		sendWsMessage(conn, "rename_chat_error", map[string]string{"chatId": p.ChatID, "error": "Database error during commit."})
		return
	}

	log.Printf("Chat %s renamed to '%s' for user %s", p.ChatID, p.NewTitle, p.UserID)
	sendWsMessage(conn, "chat_renamed", map[string]string{"chatId": p.ChatID, "newTitle": p.NewTitle})
}

func (app *App) wsDeleteChat(ctx context.Context, conn *websocket.Conn, payload json.RawMessage) {
	app.DBMutex.Lock()
	defer app.DBMutex.Unlock()

	tx, err := app.DB.BeginTx(ctx, nil)
	if err != nil {
		sendWsError(conn, "Failed to start database operation for delete.")
		return
	}
	defer tx.Rollback()

	var p GenericPayload
	if err := json.Unmarshal(payload, &p); err != nil || p.ChatID == "" || p.UserID == "" {
		sendWsMessage(conn, "delete_chat_error", map[string]string{"chatId": p.ChatID, "error": "Invalid payload for delete_chat."})
		return
	}

	res, err := tx.ExecContext(ctx, "DELETE FROM chats WHERE id = ? AND user_id = ?", p.ChatID, p.UserID)
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

	if err := tx.Commit(); err != nil {
		log.Printf("Error committing delete chat transaction %s: %v", p.ChatID, err)
		sendWsMessage(conn, "delete_chat_error", map[string]string{"chatId": p.ChatID, "error": "Database error during commit."})
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

	// 1. Update the user's message content and timestamp
	updateTime := time.Now()
	err := app.updateMessageContent(ctx, p.MessageIDToUpdateInDB, p.ChatID, p.UserID, p.NewContent, updateTime)
	if err != nil {
		log.Printf("Error updating message in DB: %v", err)
		sendWsMessage(conn, "message_update_error", map[string]string{"messageId": p.MessageIDToUpdateInDB, "error": err.Error()})
		return
	}
	sendWsMessage(conn, "message_updated", map[string]string{"id": p.MessageIDToUpdateInDB, "newContent": p.NewContent})

	// 2. LLM Call
	stream, err := app.OpenAIClient.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
		Model:       p.Model,
		Messages:    p.HistoryForRegen,
		Stream:      true,
		Temperature: app.Config.LLMTemperature,
		MaxTokens:   app.Config.LLMTopCapTokens,
		TopP:        app.Config.LLMTopP,
	})
	if err != nil {
		log.Printf("Error creating LLM stream for edit on chat %s: %v", p.ChatID, err)
		sendWsError(conn, "Failed to start regeneration.")
		return
	}
	defer stream.Close()

	var fullResponse strings.Builder
	for {
		response, err_recv := stream.Recv()
		if errors.Is(err_recv, io.EOF) {
			sendWsMessage(conn, "end", nil)
			break
		}
		if err_recv != nil {
			log.Printf("LLM stream error for edit on chat %s: %v", p.ChatID, err_recv)
			sendWsError(conn, "An error occurred during streaming.")
			break
		}
		content := response.Choices[0].Delta.Content
		if content != "" {
			fullResponse.WriteString(content)
			sendWsMessage(conn, "chunk", map[string]string{"content": content})
		}
	}

	// 3. Save the new assistant response
	assistantResponse := strings.TrimSpace(fullResponse.String())
	if assistantResponse != "" {
		err := app.saveMessage(ctx, "msg_"+uuid.New().String(), p.ChatID, p.UserID, openai.ChatMessageRoleAssistant, assistantResponse, time.Now())
		if err != nil {
			log.Printf("CRITICAL ERROR: Failed to save regenerated assistant message for chat %s: %v", p.ChatID, err)
		}
	}
}

// updateMessageContent is a new transactional helper for editing a message.
func (app *App) updateMessageContent(ctx context.Context, msgID, chatID, userID, newContent string, updateTime time.Time) error {
	app.DBMutex.Lock()
	defer app.DBMutex.Unlock()

	tx, err := app.DB.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction for update: %w", err)
	}
	defer tx.Rollback()

	_, err = tx.ExecContext(ctx, "UPDATE messages SET content = ?, created_at = ? WHERE id = ? AND user_id = ? AND chat_id = ?",
		newContent, updateTime, msgID, userID, chatID)
	if err != nil {
		return fmt.Errorf("failed to update message content: %w", err)
	}

	_, err = tx.ExecContext(ctx, "UPDATE chats SET updated_at = ? WHERE id = ?", updateTime, chatID)
	if err != nil {
		return fmt.Errorf("failed to update chat timestamp on edit: %w", err)
	}

	return tx.Commit()
}

func sendWsMessage(conn *websocket.Conn, msgType string, payload interface{}) {
	msg := WSMessage{Type: msgType, Payload: payload}
	if err := conn.WriteJSON(msg); err != nil {
		log.Printf("Error sending WebSocket message: %v", err)
	}
}

func sendWsError(conn *websocket.Conn, message string) {
	sendWsMessage(conn, "error", map[string]string{"message": message})
}