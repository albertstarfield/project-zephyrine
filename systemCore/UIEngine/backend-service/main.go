package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"bufio"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/joho/godotenv"
	_ "modernc.org/sqlite"
	"github.com/sashabaranov/go-openai"
)

// --- Configuration ---

// Config holds all configuration for the application, loaded from environment variables.
type Config struct {
	Port                string
	OpenAIAPIBaseURL    string
	LLMAPIKey           string
	LLMTemperature      float32
	LLMTopCapTokens     int
	LLMTopP             float32
	DBPath              string
	AllowedFileTypes    map[string]bool
	LLMAPIRoot          string
	ProactiveMessageCycleSeconds int // <-- ADD THIS
	LLMNotificationURL  string // <-- ADD THIS
}

// Define the structure for buffered messages
type BufferedMessage struct {
	ID        string
	ChatID    string
	UserID    string
	Sender    string
	Content   string
	CreatedAt time.Time
}

// loadConfig loads configuration from .env file and environment variables.
func loadConfig() (*Config, error) {
	godotenv.Load() // Load .env file if it exists, but don't error if it doesn't

	port := getEnv("PORT", "3001")
	apiBaseURL := getEnv("OPENAI_API_BASE_URL", "http://localhost:11434/v1")
	tempStr := getEnv("LLM_TEMPERATURE", "0.7")
	topCapStr := getEnv("LLM_TOPCAP_TOKENS", "2048")
	topPStr := getEnv("LLM_TOP_P", "1.0")
	llmNotificationURL := getEnv("LLM_NOTIFICATION_URL", "http://localhost:11434/v1/chat/notification") // <-- ADD THIS. Default to empty string to make it optional.

	temp, err := strconv.ParseFloat(tempStr, 32)
	if err != nil {
		return nil, fmt.Errorf("invalid LLM_TEMPERATURE: %w", err)
	}
	topCap, err := strconv.Atoi(topCapStr)
	if err != nil {
		return nil, fmt.Errorf("invalid LLM_TOPCAP_TOKENS: %w", err)
	}
	topP, err := strconv.ParseFloat(topPStr, 32)
	if err != nil {
		return nil, fmt.Errorf("invalid LLM_TOP_P: %w", err)
	}

	// Create a set-like map for efficient lookups
	allowedFileTypes := make(map[string]bool)
	for _, t := range []string{
		"text/plain", "application/jsonl", "application/json", "text/csv",
		"text/tab-separated-values", "application/x-parquet", "application/parquet",
		"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
		"application/vnd.ms-excel",
	} {
		allowedFileTypes[t] = true
	}

	return &Config{
		Port:                port,
		OpenAIAPIBaseURL:    apiBaseURL,
		LLMAPIKey:           getEnv("OPENAI_API_KEY", "ollama"),
		DBPath:              getEnv("DB_PATH", "./project_zephyrine_chats.db"),
		LLMTemperature:      float32(temp),
		LLMTopCapTokens:     topCap,
		LLMTopP:             float32(topP),
		AllowedFileTypes:    allowedFileTypes,
		LLMAPIRoot:          strings.TrimSuffix(apiBaseURL, "/v1"),
		LLMNotificationURL:  llmNotificationURL, // <-- ADD THIS
	}, nil
}

// getEnv is a helper to read an environment variable or return a default value.
func getEnv(key, fallback string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return fallback
}

// --- Global State & Application Struct ---

// App holds application-wide dependencies.
type App struct {
	Config            *Config
	DB                *sql.DB
	OpenAIClient      *openai.Client
	PrimedState       *PrimedReadyState
	WebsocketUpgrader websocket.Upgrader
	DBMutex           sync.Mutex

	// --- NEW: Buffer for batch writing ---
	MessageBuffer []BufferedMessage
	BufferMutex   sync.Mutex

	ongoingStreams    map[string]context.CancelFunc
	ongoingStreamsMux sync.Mutex

	// --- NEW: Connection Manager for Proactive Messages ---
	// Maps a userId to their active WebSocket connection.
	activeConnections map[string]*websocket.Conn
	connectionsMux    sync.RWMutex // RWMutex is efficient for many reads (lookups) and fewer writes (registrations)
}

// PrimedReadyState simulates the readiness check.
type PrimedReadyState struct {
	mu               sync.RWMutex
	IsReady          bool
	BenchmarkMS      float64
	InitialStartTime time.Time
}

type ProactiveMessagePayload struct {
	UserID  string `json:"userId"`
	ChatID  string `json:"chatId"`
	Message string `json:"message"`
}

// --- Database Setup ---

// initDB connects to the SQLite database and creates tables if they don't exist.
func initDB(dbPath string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, fmt.Errorf("error opening database: %w", err)
	}
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("error pinging database: %w", err)
	}

	log.Println("Connected to the SQLite database at", dbPath)

	// SQL statements to create tables.
	// We execute them sequentially.
	createStatements := []string{
		`CREATE TABLE IF NOT EXISTS chats (
			id TEXT PRIMARY KEY,
			user_id TEXT,
			title TEXT,
			created_at DATETIME DEFAULT (datetime('now', 'localtime')),
			updated_at DATETIME DEFAULT (datetime('now', 'localtime'))
		)`,
		`CREATE TABLE IF NOT EXISTS messages (
			id TEXT PRIMARY KEY,
			chat_id TEXT NOT NULL,
			user_id TEXT,
			sender TEXT NOT NULL,
			content TEXT NOT NULL,
			created_at DATETIME DEFAULT (datetime('now', 'localtime')),
			FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
		)`,
		`CREATE TABLE IF NOT EXISTS generated_images (
			id TEXT PRIMARY KEY,
			user_id TEXT NOT NULL,
			prompt TEXT NOT NULL,
			image_url TEXT NOT NULL,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE TABLE IF NOT EXISTS fine_tuning_files (
			id TEXT PRIMARY KEY,
			user_id TEXT NOT NULL,
			filename TEXT NOT NULL,
			filetype TEXT NOT NULL,
			status TEXT NOT NULL,
			uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			llm_file_id TEXT
		)`,
	}

	for _, stmt := range createStatements {
		if _, err := db.Exec(stmt); err != nil {
			return nil, fmt.Errorf("error executing create table statement: %w", err)
		}
	}
	log.Println("Database tables initialized successfully.")
	return db, nil
}

// --- NEW: Connection Management Functions ---

// registerConnection adds a user's connection to the manager.
func (app *App) registerConnection(userID string, conn *websocket.Conn) {
	app.connectionsMux.Lock()
	defer app.connectionsMux.Unlock()
	app.activeConnections[userID] = conn
	log.Printf("Connection registered for user: %s", userID)
}

// deregisterConnection removes a user's connection.
func (app *App) deregisterConnection(userID string) {
	app.connectionsMux.Lock()
	defer app.connectionsMux.Unlock()
	if _, ok := app.activeConnections[userID]; ok {
		delete(app.activeConnections, userID)
		log.Printf("Connection deregistered for user: %s", userID)
	}
}

func (app *App) handleProactiveNotification(w http.ResponseWriter, r *http.Request) {
	var payload ProactiveMessagePayload
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		respondWithError(w, http.StatusBadRequest, "Invalid request body")
		return
	}
	if payload.UserID == "" || payload.ChatID == "" || payload.Message == "" {
		respondWithError(w, http.StatusBadRequest, "userId, chatId, and message are required fields.")
		return
	}

	log.Printf("Received proactive message via HTTP POST for user %s in chat %s", payload.UserID, payload.ChatID)

	// Find the user's active WebSocket connection
	conn, found := app.findConnection(payload.UserID)
	if !found {
		log.Printf("Could not find active WebSocket connection for user %s. Message not sent.", payload.UserID)
		respondWithError(w, http.StatusNotFound, "User is not currently connected via WebSocket.")
		return
	}

	// The client will expect a specific payload structure.
	proactivePayload := map[string]string{
		"chatId":  payload.ChatID,
		"message": payload.Message,
	}

	// This is a simplified version of the WSMessage struct from websocket_handler.go
	// to avoid import cycle issues.
	type WsMsg struct {
		Type    string      `json:"type"`
		Payload interface{} `json:"payload,omitempty"`
	}

	msgToSend := WsMsg{Type: "proactive_thought", Payload: proactivePayload}
	
    // We must lock when writing to the connection from a different goroutine
    // than the one that owns the read loop.
	app.connectionsMux.Lock()
	err := conn.WriteJSON(msgToSend)
	app.connectionsMux.Unlock()

	if err != nil {
		log.Printf("Failed to send proactive message to user %s: %v", payload.UserID, err)
		// The connection might be broken. Clean it up.
		app.deregisterConnection(payload.UserID)
		respondWithError(w, http.StatusInternalServerError, "Failed to write message to client's WebSocket.")
		return
	}

	log.Printf("Successfully sent proactive message to user %s.", payload.UserID)
	respondWithJSON(w, http.StatusAccepted, map[string]string{"status": "message_queued_for_delivery"})
}

func (app *App) handleProactiveThoughtEvent(jsonData string) {
	var payload struct {
		UserID  string `json:"userId"`
		ChatID  string `json:"chatId"`
		Message string `json:"message"`
	}

	if err := json.Unmarshal([]byte(jsonData), &payload); err != nil {
		log.Printf("ERROR: Could not unmarshal proactive_thought data: %v. Data: %s", err, jsonData)
		return
	}

	if payload.UserID == "" || payload.ChatID == "" {
		log.Printf("WARNING: Received proactive_thought event with missing userId or chatId. Data: %s", jsonData)
		return
	}

	log.Printf("Received proactive thought from LLM for user %s", payload.UserID)

	// Find the user's active WebSocket connection
	conn, found := app.findConnection(payload.UserID)
	if !found {
		log.Printf("Could not find active WebSocket for user %s. Proactive message from LLM was not delivered.", payload.UserID)
		return
	}
	
	// Create the payload to send to the client
	wsPayload := map[string]string{
		"chatId":  payload.ChatID,
		"message": payload.Message,
	}

	// This function is in websocket_handler.go. We need to make sure it's accessible.
	// For now, let's assume `sendWsMessage` is a public function or we reimplement the write logic here.
	// Let's use the safer direct write method to avoid import cycle issues.
	msgToSend := WSMessage{Type: "proactive_thought", Payload: wsPayload}
	
	// We need to lock the connection when writing from a different goroutine
	app.connectionsMux.Lock()
	err := conn.WriteJSON(msgToSend)
	app.connectionsMux.Unlock()

	if err != nil {
		log.Printf("Failed to forward proactive LLM message to user %s: %v", payload.UserID, err)
		// Connection is likely broken, so we deregister the user.
		app.deregisterConnection(payload.UserID)
	} else {
		log.Printf("Successfully forwarded proactive LLM thought to user %s.", payload.UserID)
	}
}

// findConnection safely retrieves a connection by userID.
func (app *App) findConnection(userID string) (*websocket.Conn, bool) {
	app.connectionsMux.RLock()
	defer app.connectionsMux.RUnlock()
	conn, found := app.activeConnections[userID]
	return conn, found
}

func (app *App) listenForLLMNotifications(ctx context.Context) {
	log.Printf("Starting LLM notification listener for URL: %s", app.Config.LLMNotificationURL)

	for {
		select {
		case <-ctx.Done():
			log.Println("Shutting down LLM notification listener.")
			return
		default:
			// This structure allows for automatic reconnection on error
			req, err := http.NewRequestWithContext(ctx, "GET", app.Config.LLMNotificationURL, nil)
			if err != nil {
				log.Printf("ERROR: Failed to create LLM notification request: %v. Retrying in 15s.", err)
				time.Sleep(15 * time.Second)
				continue
			}

			client := &http.Client{}
			resp, err := client.Do(req)
			if err != nil {
				log.Printf("ERROR: Failed to connect to LLM notification stream: %v. Retrying in 15s.", err)
				time.Sleep(15 * time.Second)
				continue
			}

			if resp.StatusCode != http.StatusOK {
				log.Printf("ERROR: LLM notification stream returned non-200 status: %d. Retrying in 15s.", resp.StatusCode)
				resp.Body.Close()
				time.Sleep(15 * time.Second)
				continue
			}

			log.Println("Successfully connected to LLM notification stream.")
			
			// Process the stream
			processSSEStream(ctx, resp, app)
			
			// If processSSEStream returns, it means the connection was closed or an error occurred.
			// The loop will then automatically try to reconnect after a short delay.
			log.Println("LLM notification stream disconnected. Reconnecting in 5s...")
			time.Sleep(5 * time.Second)
		}
	}
}

func processSSEStream(ctx context.Context, resp *http.Response, app *App) {
	defer resp.Body.Close()
	scanner := bufio.NewScanner(resp.Body)
	var eventType, eventData string

	for scanner.Scan() {
		// Check if the parent context has been canceled (e.g., app shutdown)
		select {
		case <-ctx.Done():
			return
		default:
		}

		line := scanner.Text()
		if line == "" { // An empty line marks the end of an event
			if eventType == "proactive_thought" {
				app.handleProactiveThoughtEvent(eventData)
			}
			// Reset for the next event
			eventType, eventData = "", ""
			continue
		}

		if strings.HasPrefix(line, "event:") {
			eventType = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		} else if strings.HasPrefix(line, "data:") {
			eventData = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		}
	}

	if err := scanner.Err(); err != nil {
		// Don't log context canceled errors as they are expected on shutdown
		if !errors.Is(err, context.Canceled) {
			log.Printf("ERROR reading from LLM notification stream: %v", err)
		}
	}
}

// --- Main Application ---

func main() {
	// 1. Load Configuration
	cfg, err := loadConfig()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// 2. Initialize Database
	db, err := initDB(cfg.DBPath)
	if err != nil {
		log.Fatalf("Failed to initialize database: %v", err)
	}
	defer db.Close()

	// 3. Initialize OpenAI Client
	openaiConfig := openai.DefaultConfig(cfg.LLMAPIKey)
	openaiConfig.BaseURL = cfg.OpenAIAPIBaseURL
	openaiClient := openai.NewClientWithConfig(openaiConfig)

	// 4. Initialize App State
	app := &App{
		Config:       cfg,
		DB:           db,
		OpenAIClient: openaiClient,
		PrimedState: &PrimedReadyState{
			InitialStartTime: time.Now(),
		},
		WebsocketUpgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
			CheckOrigin: func(r *http.Request) bool {
				return true
			},
		},
		MessageBuffer: make([]BufferedMessage, 0, 100), // Pre-allocate capacity
		ongoingStreams: make(map[string]context.CancelFunc),

		// --- NEW: Initialize the Connection Manager ---
		activeConnections: make(map[string]*websocket.Conn),
	}

	// Create a context that will be canceled on shutdown
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	if cfg.LLMNotificationURL != "" {
		go app.listenForLLMNotifications(ctx)
	} else {
        log.Println("LLM_NOTIFICATION_URL not set. Skipping proactive LLM message listener.")
	}

	// Start the background message flusher
	go app.startMessageFlusher(ctx)

	// 5. Setup Router and Middleware
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-CSRF-Token"},
		ExposedHeaders:   []string{"Link"},
		AllowCredentials: true,
		MaxAge:           300,
	}))

	// 6. Define HTTP and WebSocket Routes
	// Note: We are setting the WebSocket handler on the root path, overwriting any previous handler.
	// This matches the original user configuration.
	r.Get("/health", app.handleHealth)
	r.Get("/primedready", app.handlePrimedReady)
	r.Get("/api/instrumentviewportdatastreamlowpriopreview", app.handleInstrumentProxy)

	r.HandleFunc("/ZephyCortexConfig", app.handleZephyCortexConfig)

	r.Route("/api/v1", func(r chi.Router) {
		r.Post("/images/generations", app.handleImageGeneration)
		r.Get("/images/history", app.handleImageHistory)
		r.Post("/files", app.handleFileUpload)
		r.Get("/files", app.handleFileHistory)
		// --- NEW: Endpoint for proactive notifications ---
		r.Post("/chat/notification", app.handleProactiveNotification)
	})

	

	// The WebSocket endpoint, bound to the root path as in the original file.
	r.Get("/", app.serveWs)

	// 7. Setup and Start Server
	srv := &http.Server{
		Addr:    ":" + cfg.Port,
		Handler: r,
	}

	// 8. Graceful Shutdown
	go func() {
		log.Printf("Server starting on port %s", cfg.Port)
		log.Printf("OpenAI API Base URL: %s", cfg.OpenAIAPIBaseURL)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("ListenAndServe error: %v", err)
		}
	}()

	// Wait for interrupt signal to gracefully shut down the server
	<-ctx.Done()
	log.Println("Shutting down server...")

	// Perform a final flush before shutting down
	log.Println("Performing final message flush to database...")
	if err := app.flushMessagesToDB(context.Background()); err != nil {
		log.Printf("CRITICAL: Final message flush failed: %v", err)
	} else {
		log.Println("Final message flush successful.")
	}

	// The context is used to inform the server it has 5 seconds to finish
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Fatal("Server forced to shutdown:", err)
	}

	log.Println("Server exiting.")
}

// startMessageFlusher runs in the background, flushing the message buffer periodically.
func (app *App) startMessageFlusher(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	log.Println("Background message flusher started.")

	for {
		select {
		case <-ticker.C:
			if err := app.flushMessagesToDB(ctx); err != nil {
				log.Printf("ERROR during periodic message flush: %v", err)
			}
		case <-ctx.Done():
			log.Println("Background message flusher stopping.")
			return
		}
	}
}

func (app *App) flushMessagesToDB(ctx context.Context) error {
	app.BufferMutex.Lock()
	if len(app.MessageBuffer) == 0 {
		app.BufferMutex.Unlock()
		return nil
	}

	messagesToFlush := make([]BufferedMessage, len(app.MessageBuffer))
	copy(messagesToFlush, app.MessageBuffer)
	app.MessageBuffer = app.MessageBuffer[:0]
	app.BufferMutex.Unlock()

	log.Printf("Flushing %d messages to the database...", len(messagesToFlush))

	return app.executeTransactionWithRetry(ctx, func(tx *sql.Tx) error {
		stmt, err := tx.PrepareContext(ctx, "INSERT INTO messages (id, chat_id, user_id, sender, content, created_at) VALUES (?, ?, ?, ?, ?, ?)")
		if err != nil {
			return fmt.Errorf("failed to prepare statement: %w", err)
		}
		defer stmt.Close()

		chatsToUpdate := make(map[string]bool)

		for _, msg := range messagesToFlush {
			_, err := stmt.ExecContext(ctx, msg.ID, msg.ChatID, msg.UserID, msg.Sender, msg.Content, msg.CreatedAt)
			if err != nil {
				return fmt.Errorf("failed to execute statement for message %s: %w", msg.ID, err)
			}
			chatsToUpdate[msg.ChatID] = true
		}

		updateStmt, err := tx.PrepareContext(ctx, "UPDATE chats SET updated_at = datetime('now', 'localtime') WHERE id = ?")
		if err != nil {
			return fmt.Errorf("failed to prepare update statement: %w", err)
		}
		defer updateStmt.Close()

		for chatID := range chatsToUpdate {
			_, err := updateStmt.ExecContext(ctx, chatID)
			if err != nil {
				return fmt.Errorf("failed to update timestamp for chat %s: %w", chatID, err)
			}
		}

		return nil
	})
}

// executeTransactionWithRetry handles potential "database is locked" errors from SQLite.
func (app *App) executeTransactionWithRetry(ctx context.Context, fn func(tx *sql.Tx) error) error {
	const maxRetries = 5
	const retryDelay = 50 * time.Millisecond
	var err error

	for i := 0; i < maxRetries; i++ {
		app.DBMutex.Lock()
		tx, startErr := app.DB.BeginTx(ctx, nil)
		if startErr != nil {
			app.DBMutex.Unlock()
			return fmt.Errorf("failed to begin transaction: %w", startErr)
		}

		err = fn(tx)

		if err == nil {
			if commitErr := tx.Commit(); commitErr == nil {
				app.DBMutex.Unlock()
				return nil
			} else {
				err = commitErr
			}
		}

		tx.Rollback()
		app.DBMutex.Unlock()

		if strings.Contains(err.Error(), "database is locked") || strings.Contains(err.Error(), "(5)") {
			log.Printf("Database locked, retrying transaction... (attempt %d/%d): %v", i+1, maxRetries, err)
			time.Sleep(retryDelay * time.Duration(i+1))
			continue
		}
		break
	}
	return fmt.Errorf("database operation failed after %d retries: %w", maxRetries, err)
}

// --- HTTP Handlers ---

func (app *App) handleHealth(w http.ResponseWriter, r *http.Request) {
	respondWithJSON(w, http.StatusOK, map[string]string{"status": "ok", "message": "HTTP and WebSocket server is healthy"})
}

func (app *App) handlePrimedReady(w http.ResponseWriter, r *http.Request) {
	const simulatedReadyTime = 60 * time.Second
	app.PrimedState.mu.Lock()
	defer app.PrimedState.mu.Unlock()

	elapsed := time.Since(app.PrimedState.InitialStartTime)

	if elapsed >= simulatedReadyTime {
		if !app.PrimedState.IsReady {
			app.PrimedState.IsReady = true
			app.PrimedState.BenchmarkMS = 1000 + rand.Float64()*2000
		}
		respondWithJSON(w, http.StatusOK, map[string]interface{}{
			"elp1_benchmark_ms":   app.PrimedState.BenchmarkMS,
			"primed_and_ready":    true,
			"status":              "Power-on Self Test complete. All systems nominal. Ready for engagement.",
		})
	} else {
		remainingTime := math.Ceil(simulatedReadyTime.Seconds() - elapsed.Seconds())
		respondWithJSON(w, http.StatusOK, map[string]interface{}{
			"elp1_benchmark_ms":   nil,
			"primed_and_ready":    false,
			"status":              fmt.Sprintf("Power-on Self Test in progress... T-minus %.0f seconds.", remainingTime),
		})
	}
}

func (app *App) handleZephyCortexConfig(w http.ResponseWriter, r *http.Request) {
	targetURL := app.Config.LLMAPIRoot + "/ZephyCortexConfig"
	log.Printf("Proxying config request for %s %s to: %s", r.Method, r.URL.Path, targetURL)

	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to create upstream request: %v", err), http.StatusInternalServerError)
		return
	}

	proxyReq.Header.Set("Content-Type", r.Header.Get("Content-Type"))
	proxyReq.Header.Set("Accept", r.Header.Get("Accept"))

	client := &http.Client{}
	resp, err := client.Do(proxyReq)
	if err != nil {
		log.Printf("Error during proxy config request: %v", err)
		http.Error(w, fmt.Sprintf("Backend proxy error: %v", err), http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	for key, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

func (app *App) handleInstrumentProxy(w http.ResponseWriter, r *http.Request) {
	targetURL := app.Config.LLMAPIRoot + "/instrumentviewportdatastreamlowpriopreview"
	log.Printf("Proxying SSE instrument data request to: %s", targetURL)

	req, err := http.NewRequestWithContext(r.Context(), "GET", targetURL, nil)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to create upstream request: %v", err), http.StatusInternalServerError)
		return
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Error establishing SSE proxy connection: %v", err)
		sseError(w, fmt.Sprintf("Backend proxy error establishing connection: %v", err))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		log.Printf("LLM API instrument data SSE upstream failed: %d - %s", resp.StatusCode, string(bodyBytes))
		sseError(w, fmt.Sprintf("Upstream service error: %d", resp.StatusCode))
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	_, err = io.Copy(w, resp.Body)
	if err != nil && !errors.Is(err, context.Canceled) {
		log.Printf("Error streaming SSE data to client: %v", err)
	}
	log.Println("SSE stream finished or client disconnected.")
}

// --- Image API Handlers ---

type ImageGenerationRequest struct {
	Prompt string `json:"prompt"`
	UserID string `json:"userId"`
}

func (app *App) handleImageGeneration(w http.ResponseWriter, r *http.Request) {
	var reqBody ImageGenerationRequest
	if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
		respondWithError(w, http.StatusBadRequest, "Invalid request body")
		return
	}
	if reqBody.Prompt == "" || reqBody.UserID == "" {
		respondWithError(w, http.StatusBadRequest, "Prompt and userId are required.")
		return
	}

	resp, err := app.OpenAIClient.CreateImage(
		r.Context(),
		openai.ImageRequest{
			Prompt:         reqBody.Prompt,
			Model:          openai.CreateImageModelDallE2,
			N:              1,
			Size:           openai.CreateImageSize1024x1024,
			ResponseFormat: openai.CreateImageResponseFormatURL,
		},
	)
	if err != nil {
		log.Printf("Error generating image from OpenAI/LLM: %v", err)
		respondWithError(w, http.StatusInternalServerError, "Failed to generate image")
		return
	}
	if len(resp.Data) == 0 {
		respondWithError(w, http.StatusInternalServerError, "API returned no image data")
		return
	}
	imageUrl := resp.Data[0].URL

	imageId := "img_" + uuid.New().String()
	_, err = app.DB.ExecContext(r.Context(), `
		INSERT INTO generated_images (id, user_id, prompt, image_url) VALUES (?, ?, ?, ?)
	`, imageId, reqBody.UserID, reqBody.Prompt, imageUrl)
	if err != nil {
		log.Printf("Database error saving generated image: %v", err)
		respondWithError(w, http.StatusInternalServerError, "Failed to save image details to database.")
		return
	}

	type ImageData struct {
		URL           string `json:"url"`
		RevisedPrompt string `json:"revised_prompt"`
	}
	type SuccessResponse struct {
		Success bool        `json:"success"`
		Created int64       `json:"created"`
		Data    []ImageData `json:"data"`
	}
	respondWithJSON(w, http.StatusOK, SuccessResponse{
		Success: true,
		Created: time.Now().Unix(),
		Data:    []ImageData{{URL: imageUrl, RevisedPrompt: reqBody.Prompt}},
	})
}

func (app *App) handleImageHistory(w http.ResponseWriter, r *http.Request) {
	userId := r.URL.Query().Get("userId")
	if userId == "" {
		respondWithError(w, http.StatusBadRequest, "User ID is required to fetch image history.")
		return
	}
	rows, err := app.DB.QueryContext(r.Context(), `
		SELECT id, prompt, image_url, created_at FROM generated_images WHERE user_id = ? ORDER BY created_at DESC
	`, userId)
	if err != nil {
		log.Printf("[GET /api/v1/images/history] DB Error: %v", err)
		respondWithError(w, http.StatusInternalServerError, "Failed to fetch image history from database.")
		return
	}
	defer rows.Close()

	type ImageHistoryItem struct {
		ID        string `json:"id"`
		Prompt    string `json:"prompt"`
		ImageURL  string `json:"image_url"`
		CreatedAt string `json:"created_at"`
	}
	var history []ImageHistoryItem
	for rows.Next() {
		var item ImageHistoryItem
		if err := rows.Scan(&item.ID, &item.Prompt, &item.ImageURL, &item.CreatedAt); err != nil {
			log.Printf("[GET /api/v1/images/history] DB Scan Error: %v", err)
			respondWithError(w, http.StatusInternalServerError, "Error processing image history.")
			return
		}
		history = append(history, item)
	}
	log.Printf("[GET /api/v1/images/history] Fetched %d image history entries for user %s.", len(history), userId)
	respondWithJSON(w, http.StatusOK, map[string]interface{}{"success": true, "data": history})
}

// --- Fine-Tuning File API Handlers ---

type FileUploadRequest struct {
	Filename  string `json:"filename"`
	Filetype  string `json:"filetype"`
	UserID    string `json:"userId"`
	LLMFileID string `json:"llmFileId"`
}

func (app *App) handleFileUpload(w http.ResponseWriter, r *http.Request) {
	var reqBody FileUploadRequest
	if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
		respondWithError(w, http.StatusBadRequest, "Invalid request body")
		return
	}
	if reqBody.Filename == "" || reqBody.Filetype == "" || reqBody.UserID == "" {
		respondWithError(w, http.StatusBadRequest, "Filename, filetype, and user ID are required.")
		return
	}
	if !app.Config.AllowedFileTypes[reqBody.Filetype] {
		msg := fmt.Sprintf("Unsupported file type: %s", reqBody.Filetype)
		respondWithError(w, http.StatusUnsupportedMediaType, msg)
		return
	}
	fileId := uuid.New().String()
	status := "uploaded"
	var llmFileID sql.NullString
	if reqBody.LLMFileID != "" {
		llmFileID.String = reqBody.LLMFileID
		llmFileID.Valid = true
	}
	_, err := app.DB.ExecContext(r.Context(), `
		INSERT INTO fine_tuning_files (id, user_id, filename, filetype, status, llm_file_id) VALUES (?, ?, ?, ?, ?, ?)
	`, fileId, reqBody.UserID, reqBody.Filename, reqBody.Filetype, status, llmFileID)
	if err != nil {
		log.Printf("[POST /api/v1/files] DB INSERT error: %v", err)
		respondWithError(w, http.StatusInternalServerError, "Failed to save file metadata to database.")
		return
	}
	log.Printf("[POST /api/v1/files] File metadata %s saved to DB with ID: %s.", reqBody.Filename, fileId)
	respondWithJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"message": "File metadata uploaded successfully.",
		"file": map[string]interface{}{
			"id":          fileId,
			"filename":    reqBody.Filename,
			"filetype":    reqBody.Filetype,
			"status":      status,
			"llm_file_id": llmFileID,
		},
	})
}

func (app *App) handleFileHistory(w http.ResponseWriter, r *http.Request) {
	userId := r.URL.Query().Get("userId")
	if userId == "" {
		respondWithError(w, http.StatusBadRequest, "User ID is required to fetch file history.")
		return
	}
	rows, err := app.DB.QueryContext(r.Context(), `
		SELECT id, filename, filetype, status, uploaded_at, llm_file_id FROM fine_tuning_files WHERE user_id = ? ORDER BY uploaded_at DESC
	`, userId)
	if err != nil {
		log.Printf("[GET /api/v1/files] DB Error: %v", err)
		respondWithError(w, http.StatusInternalServerError, "Failed to fetch file history from database.")
		return
	}
	defer rows.Close()

	type FileHistoryItem struct {
		ID         string         `json:"id"`
		Filename   string         `json:"filename"`
		Filetype   string         `json:"filetype"`
		Status     string         `json:"status"`
		UploadedAt string         `json:"uploaded_at"`
		LLMFileID  sql.NullString `json:"llm_file_id"`
	}
	var history []FileHistoryItem
	for rows.Next() {
		var item FileHistoryItem
		if err := rows.Scan(&item.ID, &item.Filename, &item.Filetype, &item.Status, &item.UploadedAt, &item.LLMFileID); err != nil {
			log.Printf("[GET /api/v1/files] DB Scan Error: %v", err)
			respondWithError(w, http.StatusInternalServerError, "Error processing file history.")
			return
		}
		history = append(history, item)
	}
	log.Printf("[GET /api/v1/files] Fetched %d fine-tuning file entries for user %s.", len(history), userId)
	respondWithJSON(w, http.StatusOK, map[string]interface{}{"success": true, "data": history})
}

// --- Helper Functions ---

func respondWithJSON(w http.ResponseWriter, code int, payload interface{}) {
	response, err := json.Marshal(payload)
	if err != nil {
		log.Printf("Error marshalling JSON: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("Internal Server Error"))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(response)
}

func respondWithError(w http.ResponseWriter, code int, message string) {
	respondWithJSON(w, code, map[string]string{"success": "false", "message": message})
}

func sseError(w http.ResponseWriter, message string) {
	fmt.Fprintf(w, "event: error\ndata: {\"message\": \"%s\"}\n\n", message)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}