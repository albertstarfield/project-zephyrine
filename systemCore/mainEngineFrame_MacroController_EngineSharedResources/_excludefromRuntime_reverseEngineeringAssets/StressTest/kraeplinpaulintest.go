package main

// This test is to replicate Denial of Service aspect of Kraepelin and Paulin Test that's required as a baseline test of Candidates
// https://dealls.com/pengembangan-karir/contoh-tes-kraepelin#apa-itu-tes-kraepelin?
// https://glints.com/id/lowongan/tes-kraepelin/
// Stability testing of an Engine or mind pyschologically aspect ofc.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// --- Configuration ---
const (
	TargetURL       = "http://127.0.0.1:11434/v1/chat/completions" // Standard OpenAI/Ollama endpoint
	ModelName       = "Zephy-Direct0.6-async-51.0B"
	TotalRequests   = 1_000_000
	Concurrency     = 1 // Adjust this to burn your kernel faster. 500-2000 is usually the "sweet spot" for local DoS.
	RequestTimeout  = 5 * time.Second
)

// --- Stats Tracking ---
var (
	opsProcessed uint64
	opsSuccess   uint64
	opsFailed    uint64
	totalLatency int64 // in Microseconds (accumulated)
)

// Payload structure to match OpenAI API
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

type ChatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func main() {
	fmt.Printf("🚀 STARTING KRAEPELIN-PAULI STRESS TEST\n")
	fmt.Printf("🎯 Target: %s\n", TargetURL)
	fmt.Printf("🔥 Model: %s\n", ModelName)
	fmt.Printf("💣 Total Requests: %d | Concurrency: %d\n", TotalRequests, Concurrency)
	fmt.Println("----------------------------------------------------------------")

	// Create a custom transport to reuse connections aggressively (Keep-Alive)
	// This tests the engine's ability to handle throughput, not just the OS's ability to open ports.
	transport := &http.Transport{
		MaxIdleConns:        Concurrency,
		MaxIdleConnsPerHost: Concurrency,
		IdleConnTimeout:     90 * time.Second,
		DisableKeepAlives:   false,
	}
	client := &http.Client{
		Transport: transport,
		Timeout:   RequestTimeout,
	}

	// Semaphore to limit concurrency (burn control)
	sem := make(chan struct{}, Concurrency)
	var wg sync.WaitGroup

	startTime := time.Now()

	// The Bombardment Loop
	for i := 0; i < TotalRequests; i++ {
		wg.Add(1)
		sem <- struct{}{} // Acquire token

		// Generate random math problem
		a := rand.Intn(10000000)
		b := rand.Intn(10000000)
		// Switch operator to keep cache variance high (Kraepelin style)
		opIdx := rand.Intn(4)
		var op string
		var expected int
		
		// Simple integer math to verify quickly
		switch opIdx {
		case 0:
			op = "+"
			expected = a + b
		case 1:
			op = "-"
			expected = a - b
		case 2:
			op = "*"
			expected = a * b
		case 3:
			op = "/"
			if b == 0 { b = 1 } // Safety
			expected = a / b
		}

		prompt := fmt.Sprintf("Calculate %d %s %d", a, op, b)

		go func(p string, exp int) {
			defer wg.Done()
			defer func() { <-sem }() // Release token

			// Measure Time
			reqStart := time.Now()
			
			success := sendRequest(client, p)
			
			duration := time.Since(reqStart)

			// Atomic Stats Update
			atomic.AddUint64(&opsProcessed, 1)
			atomic.AddInt64(&totalLatency, int64(duration.Microseconds()))

			if success {
				atomic.AddUint64(&opsSuccess, 1)
			} else {
				atomic.AddUint64(&opsFailed, 1)
			}

			// Progress Report every 1000 requests
			current := atomic.LoadUint64(&opsProcessed)
			if current%1 == 0 {
				avgLat := float64(atomic.LoadInt64(&totalLatency)) / float64(current) / 1000.0 // ms
				fmt.Printf("\r[%d/%d] | Success: %d | Fail: %d | Avg Latency: %.4f ms", 
					current, TotalRequests, atomic.LoadUint64(&opsSuccess), atomic.LoadUint64(&opsFailed), avgLat)
			}
		}(prompt, expected)
	}

	wg.Wait()
	totalTime := time.Since(startTime)
	
	fmt.Println("\n----------------------------------------------------------------")
	fmt.Printf("🏁 TEST COMPLETE in %s\n", totalTime)
	fmt.Printf("✅ Success: %d\n", opsSuccess)
	fmt.Printf("❌ Failed:  %d\n", opsFailed)
	fmt.Printf("⚡ Throughput: %.2f req/sec\n", float64(TotalRequests)/totalTime.Seconds())
	fmt.Printf("⏱️ Avg Latency: %.4f ms\n", float64(totalLatency)/float64(TotalRequests)/1000.0)
}

func sendRequest(client *http.Client, prompt string) bool {
	reqBody := ChatRequest{
		Model:  ModelName,
		Stream: false, // Non-streaming for faster transactional throughput measurement
		Messages: []ChatMessage{
			{Role: "user", Content: prompt},
		},
	}

	jsonData, _ := json.Marshal(reqBody)

	req, err := http.NewRequest("POST", TargetURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return false
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		// This print will spam if your server dies, useful for knowing WHEN it crashed
		// fmt.Printf("Network Error: %v\n", err) 
		return false
	}
	defer resp.Body.Close()
	
	// We read the body to ensure the request actually completed fully
	_, _ = io.Copy(io.Discard, resp.Body)

	return resp.StatusCode == 200
}