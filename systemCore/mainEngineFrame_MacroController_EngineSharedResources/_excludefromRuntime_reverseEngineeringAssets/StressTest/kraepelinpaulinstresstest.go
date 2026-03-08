package main

// Fixed Kraepelin-Paulin Stress Test with proper latency measurement
// Measures actual server inference time, not just network latency

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
	TargetURL       = "http://127.0.0.1:11434/v1/chat/completions"
	ModelName       = "Zephy-Direct0.6-async-51.0B"
	TotalRequests   = 1000 // Start smaller to see degradation pattern
	Concurrency     = 1    // Sequential for now
	RequestTimeout  = 30 * time.Second
)

// --- Stats Tracking ---
var (
	opsProcessed uint64
	opsSuccess   uint64
	opsFailed    uint64
	totalLatency int64 // in Microseconds
	maxLatency   int64
	minLatency   int64 = 999_999_999
)

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
	fmt.Printf("🚀 STARTING KRAEPELIN-PAULI STRESS TEST (FIXED)\n")
	fmt.Printf("🎯 Target: %s\n", TargetURL)
	fmt.Printf("🔥 Model: %s\n", ModelName)
	fmt.Printf("💣 Total Requests: %d | Concurrency: %d\n", TotalRequests, Concurrency)
	fmt.Println("================================================================")

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

	sem := make(chan struct{}, Concurrency)
	var wg sync.WaitGroup

	startTime := time.Now()

	for i := 0; i < TotalRequests; i++ {
		wg.Add(1)
		sem <- struct{}{}

		a := rand.Intn(10000000)
		b := rand.Intn(10000000)
		opIdx := rand.Intn(4)
		var op string

		switch opIdx {
		case 0:
			op = "+"
		case 1:
			op = "-"
		case 2:
			op = "*"
		case 3:
			op = "/"
			if b == 0 {
				b = 1
			}
		}

		prompt := fmt.Sprintf("Calculate %d %s %d", a, op, b)

		go func(p string, reqNum int) {
			defer wg.Done()
			defer func() { <-sem }()

			latency := sendRequest(client, p)

			atomic.AddUint64(&opsProcessed, 1)

			if latency > 0 {
				atomic.AddUint64(&opsSuccess, 1)
				atomic.AddInt64(&totalLatency, int64(latency.Microseconds()))

				// Track min/max
				for {
					old := atomic.LoadInt64(&maxLatency)
					new := int64(latency.Microseconds())
					if new <= old || atomic.CompareAndSwapInt64(&maxLatency, old, new) {
						break
					}
				}
				for {
					old := atomic.LoadInt64(&minLatency)
					new := int64(latency.Microseconds())
					if new >= old || atomic.CompareAndSwapInt64(&minLatency, old, new) {
						break
					}
				}
			} else {
				atomic.AddUint64(&opsFailed, 1)
			}

			current := atomic.LoadUint64(&opsProcessed)
			if current%1 == 0 || current == 1 {
				avgLat := float64(atomic.LoadInt64(&totalLatency)) / float64(atomic.LoadUint64(&opsSuccess)) / 1000.0
				maxLat := float64(atomic.LoadInt64(&maxLatency)) / 1000.0
				minLat := float64(atomic.LoadInt64(&minLatency)) / 1000.0
				fmt.Printf("\r[%d/%d] ✅:%d ❌:%d | Avg:%.2fms Min:%.2fms Max:%.2fms", 
					current, TotalRequests, 
					atomic.LoadUint64(&opsSuccess), 
					atomic.LoadUint64(&opsFailed),
					avgLat, minLat, maxLat)
			}
		}(prompt, i)
	}

	wg.Wait()
	totalTime := time.Since(startTime)

	fmt.Println("\n================================================================")
	fmt.Printf("🏁 TEST COMPLETE in %s\n", totalTime)
	fmt.Printf("✅ Success: %d\n", opsSuccess)
	fmt.Printf("❌ Failed:  %d\n", opsFailed)
	fmt.Printf("⚡ Throughput: %.2f req/sec\n", float64(TotalRequests)/totalTime.Seconds())

	if opsSuccess > 0 {
		avgLat := float64(totalLatency) / float64(opsSuccess) / 1000.0
		fmt.Printf("⏱️  Avg Latency: %.2f ms\n", avgLat)
		fmt.Printf("📊 Min Latency: %.2f ms\n", float64(minLatency)/1000.0)
		fmt.Printf("📈 Max Latency: %.2f ms\n", float64(maxLatency)/1000.0)
	}
}

func sendRequest(client *http.Client, prompt string) time.Duration {
	reqBody := ChatRequest{
		Model:  ModelName,
		Stream: false,
		Messages: []ChatMessage{
			{Role: "user", Content: prompt},
		},
	}

	jsonData, _ := json.Marshal(reqBody)

	req, err := http.NewRequest("POST", TargetURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return -1
	}
	req.Header.Set("Content-Type", "application/json")

	// Measure from request send to response fully read
	reqStart := time.Now()

	resp, err := client.Do(req)
	if err != nil {
		fmt.Printf("\n❌ Network Error: %v\n", err)
		return -1
	}
	defer resp.Body.Close()

	// Read the entire response body (this includes server processing time)
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Printf("\n❌ Read Error: %v\n", err)
		return -1
	}

	duration := time.Since(reqStart)

	if resp.StatusCode != 200 {
		fmt.Printf("\n❌ HTTP Error: %d | Body: %s\n", resp.StatusCode, string(body))
		return -1
	}

	// Optional: Verify response is valid JSON
	var respData ChatResponse
	if err := json.Unmarshal(body, &respData); err != nil {
		fmt.Printf("\n❌ JSON Parse Error: %v\n", err)
		return -1
	}

	fmt.Printf("Response: %s\n", string(body))

	return duration
}