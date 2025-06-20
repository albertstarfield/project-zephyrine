// Filename: watchdog_thread1.go
//
// This program is the first-stage watchdog, written in Go. It is designed to be launched
// by launcher.py and is responsible for monitoring the main Python AI application.
//
// New Features:
// - Takes the command to run as arguments.
// - Performs a SHA256 integrity check on a specified critical file before launch.
// - Monitors the application's exit code to detect crashes from signals (e.g., segfault).
// - Continues to use a challenge-response handshake to detect logical freezes.
//
// Build: go build -o watchdog_thread1
//
// To Run (handled by launcher.py):
// ./watchdog_thread1 --integrity-check-file=./launcher.py -- python AdelaideAlbertCortex.py

package main

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"io"
	"io/ioutil"
	"log"
	"math/big"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const (
	challengeFile = "watchdog_challenge.txt"
	responseFile  = "watchdog_response.txt"
	watchdogTimeout = 15 * time.Second // Increased timeout for potentially slower AI startup
	heartbeatSleep  = 3 * time.Second
	magicPrime = 48611
)

var initialHash = ""

func main() {
	// The application command and its arguments will be passed directly after the watchdog's own flags.
	integrityCheckFile := flag.String("integrity-check-file", "", "Path to a critical file to perform SHA256 integrity check on at startup.")
	flag.Parse()

	appArgs := flag.Args()
	if len(appArgs) == 0 {
		// This program is now designed to be called with another program's command as arguments.
		// For standalone testing of the --app mode, you can run: go run . --app
		// This block allows `go run . --app` to work for testing the app logic.
		if len(os.Args) > 1 && os.Args[1] == "--app" {
			runApplication()
			return
		}
		log.Fatal("WATCHDOG: No application command provided. Usage: ./watchdog_thread1 --integrity-check-file=<file> <command> [args...]")
	}
	
	runWatchdog(*integrityCheckFile, appArgs)
}

// calculateSHA256 computes the SHA256 hash of a file.
func calculateSHA256(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

// runWatchdog starts and monitors the application process.
func runWatchdog(integrityFile string, appCommandAndArgs []string) {
	log.Println("WATCHDOG: Starting in monitor mode.")
	var appCmd *exec.Cmd
	
	// Perform initial integrity check
	if integrityFile != "" {
		var err error
		initialHash, err = calculateSHA256(integrityFile)
		if err != nil {
			log.Fatalf("WATCHDOG: CRITICAL - Could not calculate initial hash for '%s': %v. Halting.", integrityFile, err)
		}
		log.Printf("WATCHDOG: Initial integrity hash for '%s' is %s", integrityFile, initialHash)
	}

	startApp := func() {
		// Before every start, verify file integrity if enabled
		if integrityFile != "" {
			currentHash, err := calculateSHA256(integrityFile)
			if err != nil {
				log.Printf("WATCHDOG: CRITICAL - Could not calculate hash for '%s' on restart: %v. Halting.", integrityFile, err)
				// In a real mission, you might try restoring a backup of the file. Here we halt.
				os.Exit(1)
			}
			if currentHash != initialHash {
				log.Printf("WATCHDOG: CRITICAL - Integrity check failed! File '%s' has been modified or corrupted. Halting.", integrityFile)
				log.Printf("WATCHDOG: Expected hash: %s, Got: %s", initialHash, currentHash)
				os.Exit(1)
			}
			log.Println("WATCHDOG: Integrity check passed.")
		}

		log.Println("WATCHDOG: Attempting to start the application...")
		appCmd = exec.Command(appCommandAndArgs[0], appCommandAndArgs[1:]...)
		appCmd.Stdout = os.Stdout
		appCmd.Stderr = os.Stderr
		
		// Set Pdeathsig on Linux to have child killed if watchdog dies. Not portable.
		// #if defined(__linux__)
		// appCmd.SysProcAttr = &syscall.SysProcAttr{
		// 	Pdeathsig: syscall.SIGKILL,
		// }
		// #endif

		err := appCmd.Start()
		if err != nil {
			log.Fatalf("WATCHDOG: Failed to start application: %v", err)
		}
		log.Printf("WATCHDOG: Application started with PID: %d", appCmd.Process.Pid)

		// Goroutine to wait for the process to exit and detect crashes
		go func() {
			waitErr := appCmd.Wait()
			log.Printf("WATCHDOG: Monitored application (PID: %d) has exited.", appCmd.Process.Pid)
			if waitErr != nil {
				if exitErr, ok := waitErr.(*exec.ExitError); ok {
					if status, ok := exitErr.Sys().(syscall.WaitStatus); ok {
						// On Unix-like systems, exit codes > 128 often mean termination by a signal.
						// Signal = ExitCode - 128. SIGSEGV=11 -> 139, SIGABRT=6 -> 134.
						exitCode := status.ExitStatus()
						if exitCode > 128 {
							signal := exitCode - 128
							log.Printf("WATCHDOG: CRASH DETECTED! Application terminated by signal %d.", signal)
						} else {
							log.Printf("WATCHDOG: Application exited with code: %d.", exitCode)
						}
					}
				} else {
					log.Printf("WATCHDOG: Error waiting for application exit: %v", waitErr)
				}
			}
		}()
	}

	// Initial start
	startApp()

	for {
		// ... (The challenge-response logic remains the same) ...
		challenge, err := rand.Int(rand.Reader, big.NewInt(100000))
		if err != nil { log.Fatalf("WATCHDOG: Could not generate challenge: %v", err) }
		challengeInt := int(challenge.Int64())
		err = ioutil.WriteFile(challengeFile, []byte(strconv.Itoa(challengeInt)), 0644)
		if err != nil { log.Printf("WATCHDOG: Error writing challenge file: %v", err); continue }
		log.Printf("WATCHDOG: Wrote new challenge: %d", challengeInt)

		time.Sleep(watchdogTimeout)
		
		// Before checking the response, check if the process is still running.
		if appCmd.ProcessState != nil && appCmd.ProcessState.Exited() {
			log.Printf("WATCHDOG: Failure! Application process exited unexpectedly. Resetting application.")
			resetApplication(&appCmd, startApp)
			continue
		}

		content, err := ioutil.ReadFile(responseFile)
		if err != nil { log.Printf("WATCHDOG: Failure! Could not read response file: %v. Resetting application.", err); resetApplication(&appCmd, startApp); continue }
		responseInt, err := strconv.Atoi(strings.TrimSpace(string(content)))
		if err != nil { log.Printf("WATCHDOG: Failure! Could not parse response '%s': %v. Resetting application.", content, err); resetApplication(&appCmd, startApp); continue }
		expectedResponse := challengeInt + magicPrime
		if responseInt == expectedResponse {
			log.Printf("WATCHDOG: Success! Received correct response: %d", responseInt)
		} else {
			log.Printf("WATCHDOG: Failure! Expected response %d, but got %d. Resetting application.", expectedResponse, responseInt)
			resetApplication(&appCmd, startApp)
		}
	}
}

// resetApplication forcefully terminates and restarts the monitored process.
func resetApplication(cmd **exec.Cmd, startFunc func()) {
	// ... (This function remains exactly the same as before) ...
	log.Println("WATCHDOG: --- RESET SEQUENCE INITIATED ---")
	if *cmd != nil && (*cmd).Process != nil {
		log.Printf("WATCHDOG: Killing process with PID: %d", (*cmd).Process.Pid)
		// Use Kill for forceful termination (equivalent to SIGKILL)
		if err := (*cmd).Process.Kill(); err != nil {
			log.Printf("WATCHDOG: Failed to kill process %d: %v", (*cmd).Process.Pid, err)
		} else {
			log.Println("WATCHDOG: Process killed.")
		}
		(*cmd).Wait() // Clean up zombie process
	} else {
		log.Println("WATCHDOG: No process found to kill.")
	}
	startFunc()
	log.Println("WATCHDOG: --- RESET SEQUENCE COMPLETE ---")
}

// runApplication simulates the main Zephy AI application.
func runApplication() {
	// This function remains the same, but the interactive crash part is removed
	// as crashes are now detected by the watchdog's Wait() call.
	log.Println("APPLICATION: Starting in application mode.")
	for {
		content, err := ioutil.ReadFile(challengeFile)
		if err != nil { log.Printf("APPLICATION: Error reading challenge file: %v", err); time.Sleep(heartbeatSleep); continue }
		challengeInt, err := strconv.Atoi(strings.TrimSpace(string(content)))
		if err != nil { log.Printf("APPLICATION: Error parsing challenge '%s': %v", content, err); time.Sleep(heartbeatSleep); continue }
		log.Printf("APPLICATION: Read challenge: %d", challengeInt)
		responseInt := challengeInt + magicPrime
		err = ioutil.WriteFile(responseFile, []byte(strconv.Itoa(responseInt)), 0644)
		if err != nil { log.Printf("APPLICATION: Error writing response file: %v", err)
		} else { log.Printf("APPLICATION: Wrote response: %d", responseInt) }
		time.Sleep(heartbeatSleep)
	}
}
