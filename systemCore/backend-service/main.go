package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
}

// loadConfig loads configuration from .env file and environment variables.
func loadConfig() (*Config, error) {
	godotenv.Load() // Load .env file if it exists, but don't error if it doesn't

	port := getEnv("PORT", "3001")
	apiBaseURL := getEnv("OPENAI_API_BASE_URL", "http://localhost:11434/v1")
	tempStr := getEnv("LLM_TEMPERATURE", "0.7")
	topCapStr := getEnv("LLM_TOPCAP_TOKENS", "2048")
	topPStr := getEnv("LLM_TOP_P", "1.0")

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
	Config          *Config
	DB              *sql.DB
	OpenAIClient    *openai.Client
	PrimedState     *PrimedReadyState
	WebsocketUpgrader websocket.Upgrader
}

// PrimedReadyState simulates the readiness check.
type PrimedReadyState struct {
	mu               sync.RWMutex
	IsReady          bool
	BenchmarkMS      float64
	InitialStartTime time.Time
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
	// Note: The go-openai client needs a specific configuration for custom base URLs.
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
			// Allow all origins for simplicity, matching the Node.js `cors()` setup.
			CheckOrigin: func(r *http.Request) bool {
				return true
			},
		},
	}

	// 5. Setup Router and Middleware
	r := chi.NewRouter()
	r.Use(middleware.Logger)   // Log requests
	r.Use(middleware.Recoverer) // Recover from panics
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-CSRF-Token"},
		ExposedHeaders:   []string{"Link"},
		AllowCredentials: true,
		MaxAge:           300,
	}))

	// 6. Define HTTP and WebSocket Routes
	r.Get("/", app.handleRoot)
	r.Get("/health", app.handleHealth)
	r.Get("/primedready", app.handlePrimedReady)
	r.Get("/api/instrumentviewportdatastreamlowpriopreview", app.handleInstrumentProxy)

	r.Route("/api/v1", func(r chi.Router) {
		r.Post("/images/generations", app.handleImageGeneration)
		r.Get("/images/history", app.handleImageHistory)
		r.Post("/files", app.handleFileUpload)
		r.Get("/files", app.handleFileHistory)
	})

	// The WebSocket endpoint
	//r.Get("/ws", app.serveWs)
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
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down server...")

	// The context is used to inform the server it has 5 seconds to finish
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatal("Server forced to shutdown:", err)
	}

	log.Println("Server exiting.")
}

// --- HTTP Handlers ---

func (app *App) handleRoot(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain")
	w.Write([]byte("Project Zephyrine WebSocket server is running. HTTP endpoints are available."))
}

func (app *App) handleHealth(w http.ResponseWriter, r *http.Request) {
	respondWithJSON(w, http.StatusOK, map[string]string{"status": "ok", "message": "HTTP and WebSocket server is healthy"})
}

// handlePrimedReady simulates the LLM readiness check.
func (app *App) handlePrimedReady(w http.ResponseWriter, r *http.Request) {
	const simulatedReadyTime = 60 * time.Second
	app.PrimedState.mu.Lock()
	defer app.PrimedState.mu.Unlock()

	elapsed := time.Since(app.PrimedState.InitialStartTime)

	if elapsed >= simulatedReadyTime {
		if !app.PrimedState.IsReady {
			app.PrimedState.IsReady = true
			// Simulate a benchmark result between 1 and 3 seconds
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

// handleInstrumentProxy proxies the SSE stream from the LLM.
func (app *App) handleInstrumentProxy(w http.ResponseWriter, r *http.Request) {
	targetURL := app.Config.LLMAPIRoot + "/instrumentviewportdatastreamlowpriopreview"
	log.Printf("Proxying SSE instrument data request to: %s", targetURL)

	// Create a new request to the target URL, passing through the original context
	// so that if the client disconnects, the upstream request is cancelled.
	req, err := http.NewRequestWithContext(r.Context(), "GET", targetURL, nil)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to create upstream request: %v", err), http.StatusInternalServerError)
		return
	}

	// Make the request to the upstream service
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

	// Set SSE headers for the client
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK) // Flush the headers to the client

	// Stream the response body directly to the client
	// io.Copy is highly efficient for this.
	_, err = io.Copy(w, resp.Body)
	if err != nil && !errors.Is(err, context.Canceled) {
		// context.Canceled happens when the client disconnects, which is expected.
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

	// Call OpenAI API
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

	// Save to DB
	imageId := "img_" + uuid.New().String()
	_, err = app.DB.ExecContext(r.Context(), `
		INSERT INTO generated_images (id, user_id, prompt, image_url) VALUES (?, ?, ?, ?)
	`, imageId, reqBody.UserID, reqBody.Prompt, imageUrl)
	if err != nil {
		log.Printf("Database error saving generated image: %v", err)
		respondWithError(w, http.StatusInternalServerError, "Failed to save image details to database.")
		return
	}

	// Send success response
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
	LLMFileID string `json:"llmFileId"` // Optional
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

	// Use sql.NullString for optional fields like llmFileId
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
		ID        string         `json:"id"`
		Filename  string         `json:"filename"`
		Filetype  string         `json:"filetype"`
		Status    string         `json:"status"`
		UploadedAt string        `json:"uploaded_at"`
		LLMFileID sql.NullString `json:"llm_file_id"`
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

// respondWithJSON writes a JSON response.
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

// respondWithError writes a standard JSON error response.
func respondWithError(w http.ResponseWriter, code int, message string) {
	respondWithJSON(w, code, map[string]string{"success": "false", "message": message})
}

// sseError sends a server-sent event error message.
func sseError(w http.ResponseWriter, message string) {
	// The SSE format for a custom event named 'error'
	fmt.Fprintf(w, "event: error\ndata: {\"message\": \"%s\"}\n\n", message)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

