// Filename: watchdog_thread1.go
//
// This program is the first-stage watchdog, rewritten to act as a system-wide
// process monitor. It is launched by launcher.py and is responsible for
// periodically checking if the main application processes are running.
//
// It does NOT launch or restart processes itself. It only monitors and logs.
//
// Build: go build -o watchdog_thread1
//
// To Run (as handled by launcher.py):
// ./watchdog_thread1 --targets="hypercorn,zephyrine-backend,node"
//
// You can also add --pid-file="path/to/some.pid" to monitor a specific PID.

package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"strconv"
	"strings"
	"time"

	// We need a third-party library to find processes, similar to Python's psutil.
	// The launcher will need to fetch this dependency.
	"github.com/shirou/gopsutil/v3/process"
)

func main() {
	// Define command-line flags to specify what to monitor.
	// --targets: A comma-separated list of process names to look for.
	// --pid-file: The path to a file containing a single Process ID to monitor.
	// --interval: How often to check, in seconds.
	targetNames := flag.String("targets", "", "Comma-separated list of process names to monitor (e.g., 'hypercorn,node').")
	pidFile := flag.String("pid-file", "", "Path to a file containing a PID to monitor.")
	intervalSec := flag.Int("interval", 15, "Monitoring interval in seconds.")
	flag.Parse()

	// Validate that we have something to monitor.
	if *targetNames == "" && *pidFile == "" {
		log.Fatal("WATCHDOG: Error - No targets specified. Use --targets=\"name1,name2\" or --pid-file=\"/path/to.pid\".")
	}

	// Split the comma-separated target names into a slice.
	var targets []string
	if *targetNames != "" {
		targets = strings.Split(*targetNames, ",")
		for i, t := range targets {
			targets[i] = strings.TrimSpace(t) // Clean up whitespace
		}
	}

	log.Printf("--- Go Watchdog (Thread 1) Activated ---")
	log.Printf("Monitoring Interval: %d seconds", *intervalSec)
	if len(targets) > 0 {
		log.Printf("Monitoring Process Names: %v", targets)
	}
	if *pidFile != "" {
		log.Printf("Monitoring PID File: %s", *pidFile)
	}
	log.Println("----------------------------------------")

	// Create a ticker that fires at the specified interval. This is more
	// efficient than a `for { time.Sleep() }` loop.
	ticker := time.NewTicker(time.Duration(*intervalSec) * time.Second)
	defer ticker.Stop()

	// Run the check immediately on startup, then wait for the ticker.
	runCheck(targets, *pidFile)

	// Main monitoring loop.
	for range ticker.C {
		runCheck(targets, *pidFile)
	}
}

// runCheck performs a single monitoring scan.
func runCheck(targetNames []string, pidFilePath string) {
	log.Println("WATCHDOG: Performing health check scan...")

	// --- 1. Get all running processes from the system ---
	pids, err := process.Pids()
	if err != nil {
		log.Printf("WATCHDOG: CRITICAL - Failed to list system processes: %v", err)
		return
	}

	// Create a map to easily look up running process names.
	// The key is the process name (e.g., "hypercorn"), the value is a boolean.
	runningProcesses := make(map[string]bool)
	for _, pid := range pids {
		proc, err := process.NewProcess(pid)
		if err != nil {
			// This can happen for transient or system-protected processes, it's usually safe to ignore.
			continue
		}
		name, err := proc.Name()
		if err != nil {
			continue
		}
		// On Windows, names often end with .exe, so we trim it for consistent matching.
		name = strings.TrimSuffix(name, ".exe")
		runningProcesses[name] = true
	}

	// --- 2. Check for each target process by name ---
	allOk := true
	for _, target := range targetNames {
		if _, found := runningProcesses[target]; found {
			log.Printf("WATCHDOG: [OK] Target process '%s' is RUNNING.", target)
		} else {
			log.Printf("WATCHDOG: [FAIL] Target process '%s' was NOT FOUND.", target)
			allOk = false
		}
	}

	// --- 3. Check for the process specified in the PID file ---
	if pidFilePath != "" {
		pidFrom_file, err := readPidFromFile(pidFilePath)
		if err != nil {
			log.Printf("WATCHDOG: [WARN] Could not read PID from file '%s': %v", pidFilePath, err)
			allOk = false
		} else {
			exists, err := process.PidExists(pidFrom_file)
			if err != nil {
				log.Printf("WATCHDOG: [WARN] Error checking existence of PID %d: %v", pidFrom_file, err)
				allOk = false
			} else if exists {
				log.Printf("WATCHDOG: [OK] Target PID %d from file is RUNNING.", pidFrom_file)
			} else {
				log.Printf("WATCHDOG: [FAIL] Target PID %d from file was NOT FOUND.", pidFrom_file)
				allOk = false
			}
		}
	}

	if allOk {
		log.Println("WATCHDOG: Health check passed. All targets are running.")
	} else {
		log.Println("WATCHDOG: Health check FAILED. One or more targets are down.")
	}
	log.Println("---") // Separator for the next check
}

// readPidFromFile reads a PID from a file.
func readPidFromFile(path string) (int32, error) {
	content, err := ioutil.ReadFile(path)
	if err != nil {
		return 0, err
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(content)))
	if err != nil {
		return 0, fmt.Errorf("could not parse PID: %w", err)
	}

	return int32(pid), nil
}