package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"zephymesh/internal/node"
)

func main() {
	// Configure logging
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	log.Println("--- Starting ZephyMesh Node (Go) ---")

	// Define command-line flags
	listenPort := flag.Int("port", 21000, "Port number for the libp2p host to listen on")
	identityPath := flag.String("identity", "zephymesh_identity.key", "Path to the node's private key file")
	// The API port is now handled automatically within the node, so the flag is removed.
	flag.Parse()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create and start the libp2p node
	hostNode, err := node.NewNode(ctx, *listenPort, *identityPath)
	if err != nil {
		log.Fatalf("FATAL: Failed to create libp2p node: %v", err)
	}
	defer hostNode.Close()

	log.Printf("[INFO] ✅ ZephyMesh Node Initialized. Peer ID: %s", hostNode.ID())
	for _, addr := range hostNode.Addrs() {
		log.Printf("[INFO]   Listening on: %s/p2p/%s", addr, hostNode.ID())
	}

	// Start the local HTTP API server, which will find its own port
	// and write it to the port info file for the Python launcher.
	go node.StartAPIAndNotify(hostNode, "../zephymesh_ports.json")

	// Graceful shutdown
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGINT, syscall.SIGTERM)
	<-ch

	log.Println("Shutdown signal received, closing node...")
}