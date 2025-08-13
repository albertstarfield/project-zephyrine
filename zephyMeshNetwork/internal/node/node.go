package node

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/libp2p/go-libp2p"
	dht "github.com/libp2p/go-libp2p-kad-dht"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/p2p/discovery/routing"
	"github.com/multiformats/go-multiaddr"
)

const (
	ManifestProtocolID     = "/zephymesh/manifest/1.0.0" // Kept for reference, but handler is removed
	FileProtocolID         = "/zephymesh/file_transfer/1.0.0"
	ManifestExchangeProtocolID = "/zephymesh/manifest_exchange/1.0.0" 
	FileProtocolID             = "/zephymesh/file_transfer/1.0.0"
	RelayDirCapBytes       = 1 * 1024 * 1024 * 1024 // 1 GB
	RelayDirPruneTarget    = 0.9                    // Prune to 90% of capacity
)

type AssetInfo struct {
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

type Libp2pNode struct {
	host.Host
	manifest      map[string]AssetInfo
	manifestLock  sync.RWMutex
	projectRoot   string
	relayDir      string
	relayDirLock  sync.Mutex
}

type PeerManifests map[peer.ID]map[string]AssetInfo

type Libp2pNode struct {
	host.Host
	manifest       map[string]AssetInfo
	manifestLock   sync.RWMutex
	
	// NEW: State to hold manifests from other peers
	peerManifests  PeerManifests
	peerManifestsLock sync.RWMutex

	projectRoot    string
	relayDir       string
	relayDirLock   sync.Mutex
}

type discoveryNotifee struct {
	node *Libp2pNode
}

type connectionNotifee struct {
	node *Libp2pNode
	ctx  context.Context
}

type QueryResponse struct {
	Found       bool          `json:"found"`
	Source      string        `json:"source"` // "local", "peer", or "none"
	LocalPath   string        `json:"localPath,omitempty"`
	PeerID      string        `json:"peerId,omitempty"`
	AssetInfo   *AssetInfo    `json:"assetInfo,omitempty"`
}

func (n *Libp2pNode) handleQueryAsset(w http.ResponseWriter, r *http.Request) {
	// Get the 'filepath' query parameter from the request URL.
	filePathKeys, ok := r.URL.Query()["filepath"]
	if !ok || len(filePathKeys[0]) < 1 {
		http.Error(w, `{"error": "Missing 'filepath' query parameter"}`, http.StatusBadRequest)
		return
	}
	requestedPath := filePathKeys[0]
	log.Printf("[API Query] Received query for asset: %s", requestedPath)

	// --- Step 1: Check own manifest first (it's the fastest) ---
	n.manifestLock.RLock()
	asset, existsInLocal := n.manifest[requestedPath]
	n.manifestLock.RUnlock()

	if existsInLocal {
		log.Printf("[API Query] Asset '%s' found in LOCAL manifest.", requestedPath)
		response := QueryResponse{
			Found:     true,
			Source:    "local",
			LocalPath: filepath.Join(n.projectRoot, requestedPath),
			AssetInfo: &asset,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		return
	}

	// --- Step 2: If not found locally, check all cached peer manifests ---
	n.peerManifestsLock.RLock()
	defer n.peerManifestsLock.RUnlock()

	for peerID, peerManifest := range n.peerManifests {
		if asset, existsInPeer := peerManifest[requestedPath]; existsInPeer {
			log.Printf("[API Query] Asset '%s' found in PEER manifest (Peer ID: %s)", requestedPath, peerID)
			response := QueryResponse{
				Found:     true,
				Source:    "peer",
				PeerID:    peerID.String(),
				AssetInfo: &asset,
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(response)
			return
		}
	}

	// --- Step 3: If not found anywhere, respond accordingly ---
	log.Printf("[API Query] Asset '%s' was NOT FOUND in local or any peer manifests.", requestedPath)
	response := QueryResponse{
		Found:  false,
		Source: "none",
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (n *Libp2pNode) handleManifestStream(s network.Stream) {
	remotePeerID := s.Conn().RemotePeer()
	log.Printf("[ManifestExchange] Received manifest request from peer %s", remotePeerID)
	defer s.Close()

	// We don't need to read anything; the request is implicit by opening the stream.
	// Just respond with our current manifest.

	n.manifestLock.RLock()
	manifestBytes, err := json.Marshal(n.manifest)
	n.manifestLock.RUnlock()

	if err != nil {
		log.Printf("[ManifestExchange] ERROR: Failed to marshal own manifest for peer %s: %v", remotePeerID, err)
		return
	}

	_, err = s.Write(manifestBytes)
	if err != nil {
		log.Printf("[ManifestExchange] ERROR: Failed to send manifest to peer %s: %v", remotePeerID, err)
		return
	}

	log.Printf("[ManifestExchange] Successfully sent manifest (%d assets) to peer %s", len(n.manifest), remotePeerID)
}

func (n *Libp2pNode) requestManifestFromPeer(ctx context.Context, pid peer.ID) {
	log.Printf("[ManifestExchange] Requesting manifest from peer %s...", pid)

	s, err := n.Host.NewStream(ctx, pid, ManifestExchangeProtocolID)
	if err != nil {
		log.Printf("[ManifestExchange] WARN: Could not open stream to peer %s: %v", pid, err)
		return
	}
	defer s.Close()

	// Read the response from the peer
	decoder := json.NewDecoder(s)
	var receivedManifest map[string]AssetInfo
	if err := decoder.Decode(&receivedManifest); err != nil {
		log.Printf("[ManifestExchange] ERROR: Failed to decode manifest from peer %s: %v", pid, err)
		return
	}

	// Store the received manifest
	n.peerManifestsLock.Lock()
	n.peerManifests[pid] = receivedManifest
	n.peerManifestsLock.Unlock()

	log.Printf("[ManifestExchange] ✅ Successfully received and stored manifest from %s (%d assets)", pid, len(receivedManifest))
}


func (dn *discoveryNotifee) HandlePeerFound(pi peer.AddrInfo) {
	// We connect to the peer. If the connection is successful, the
	// Connected event will be triggered, where we'll do the manifest exchange.
	log.Printf("[Discovery] Discovered new peer: %s", pi.ID)
	// Use a background context for the connection attempt.
	err := dn.node.Host.Connect(context.Background(), pi)
	if err != nil {
		log.Printf("[Discovery] Failed to connect to discovered peer %s: %v", pi.ID, err)
	}
}

func NewNode(ctx context.Context, listenPort int, identityPath string) (*Libp2pNode, error) {
	// Step 1: Load or generate the node's unique cryptographic identity.
	privKey, err := loadOrGenerateIdentity(identityPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load/generate identity: %w", err)
	}

	// Step 2: Create the libp2p host, which is the core of the P2P node.
	// It will listen on all available network interfaces on the specified TCP port.
	listenAddr, _ := multiaddr.NewMultiaddr(fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", listenPort))
	h, err := libp2p.New(
		libp2p.ListenAddrs(listenAddr),
		libp2p.Identity(privKey),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create libp2p host: %w", err)
	}

	// Step 3: Determine the project's root directory.
	// We assume the binary is run from the root, which the Python launcher ensures.
	pRoot, err := filepath.Abs(".")
	if err != nil {
		h.Close() // Clean up the host if we can't proceed.
		return nil, fmt.Errorf("could not determine project root directory: %w", err)
	}
	log.Printf("[DEBUG] Project Root detected as: %s", pRoot)

	// Step 4: Create the main Node object and initialize its state.
	node := &Libp2pNode{
		Host: h,
		// Initialize the map that will hold manifests from other peers.
		peerManifests: make(PeerManifests),
		projectRoot:   pRoot,
		relayDir:      filepath.Join(pRoot, "systemCore", "engineMain", "meshCommunicationRelay"),
	}

	// Step 5: Register the handlers for our custom P2P protocols.
	// This tells the node how to respond when a peer opens a stream.
	node.SetStreamHandler(FileProtocolID, node.handleFileStream)
	node.SetStreamHandler(ManifestExchangeProtocolID, node.handleManifestStream)

	// Step 6: Register a "Notifee" to automatically handle connection events.
	// This is the key to automating the manifest exchange.
	notifee := &connectionNotifee{node: node, ctx: ctx}
	node.Network().Notify(notifee)

	// Step 7: Ensure the directory for relayed files exists.
	if err := os.MkdirAll(node.relayDir, 0755); err != nil {
		h.Close()
		return nil, fmt.Errorf("could not create relay directory: %w", err)
	}

	// Step 8: Generate this node's own manifest by scanning local files on startup.
	if err := node.generateManifest(); err != nil {
		h.Close()
		// We treat this as a potentially critical error and log it, but you could
		// decide to let the node start with an empty manifest if preferred.
		// For now, let's log it as a warning.
		log.Printf("[WARN] Could not generate manifest on startup: %v", err)
	}

	// Step 9: Initialize the DHT for peer discovery and start the process in the background.
	go node.initDHT(ctx)

	// Step 10: Return the fully initialized node.
	return node, nil
}

func (n *Libp2pNode) initDHT(ctx context.Context) {
	log.Println("[INFO] Initializing Kademlia DHT for peer discovery...")
	kademliaDHT, err := dht.New(ctx, n.Host)
	if err != nil {
		log.Printf("[ERROR] Error creating DHT: %v", err)
		return
	}

	if err = kademliaDHT.Bootstrap(ctx); err != nil {
		log.Printf("[ERROR] Error bootstrapping DHT: %v", err)
		return
	}

	var wg sync.WaitGroup
	for _, peerAddr := range dht.DefaultBootstrapPeers {
		peerinfo, _ := peer.AddrInfoFromP2pAddr(peerAddr)
		wg.Add(1)
		go func() {
			defer wg.Done()
			log.Printf("[DEBUG] Connecting to bootstrap peer: %s", peerinfo.ID)
			if err := n.Host.Connect(ctx, *peerinfo); err != nil {
				log.Printf("[WARN] Connection to bootstrap peer %s failed: %s", peerinfo.ID, err)
			} else {
				log.Printf("[INFO] Connection established with bootstrap peer: %s", peerinfo.ID)
			}
		}()
	}
	wg.Wait()
	log.Println("[INFO] ✅ DHT Initialized and bootstrap process completed.")

	routingDiscovery := routing.NewRoutingDiscovery(kademliaDHT)
	// We advertise the file transfer protocol to let others know we can serve files.
	routingDiscovery.Advertise(ctx, FileProtocolID)
	log.Println("[INFO] Now advertising our services on the network.")
}

func (n *Libp2pNode) generateManifest() error {
	log.Println("[INFO] Regenerating asset manifest...")
	n.manifestLock.Lock()
	defer n.manifestLock.Unlock()

	assets := make(map[string]AssetInfo)

	dirsToScan := map[string]string{
		// All GGUF models and the HF cache are in the static model pool
		"staticmodelpool":                   staticModelPoolPath,

		// NEW: Explicitly add the Hugging Face cache directory to the scan.
		// Its contents are critical for non-GGUF models and configs.
		"huggingface_cache":                 filepath.Join(staticModelPoolPath, "huggingface_cache"),

		// The compiled libraries are also valuable assets to share
		"llama-cpp-python_build":            filepath.Join(n.projectRoot, "llama-cpp-python_build"),
		"stable-diffusion-cpp-python_build": filepath.Join(n.projectRoot, "stable-diffusion-cpp-python_build"),
		"pywhispercpp_build":                filepath.Join(n.projectRoot, "pywhispercpp_build"),

		// The relay directory is for DTN and may contain in-transit files
		"meshCommunicationRelay":            n.relayDir,
	}

	for key, dirPath := range dirsToScan {
		log.Printf("[DEBUG] Scanning directory '%s' at path '%s'", key, dirPath)
		if _, err := os.Stat(dirPath); os.IsNotExist(err) {
			log.Printf("[WARN] Asset directory not found, skipping: %s", dirPath)
			continue
		}

		filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
			if err == nil && !info.IsDir() {
				relPath, _ := filepath.Rel(n.projectRoot, path)
				checksum, _ := calculateSHA256(path)
				if checksum != "" {
					assets[filepath.ToSlash(relPath)] = AssetInfo{
						SHA256: checksum,
						Size:   info.Size(),
					}
				}
			}
			return nil
		})
	}

	n.manifest = assets
	log.Printf("[INFO] ✅ Manifest generated with %d total assets.", len(n.manifest))
	return nil
}

func (n *Libp2pNode) handleFileStream(s network.Stream) {
	remotePeerID := s.Conn().RemotePeer()
	log.Printf("[INFO] Received new file request from %s", remotePeerID)
	defer s.Close()

	reader := bufio.NewReader(s)
	// Expecting a JSON request like {"path": "staticmodelpool/model.gguf"}
	requestBytes, err := reader.ReadBytes('\n')
	if err != nil {
		log.Printf("[ERROR] Error reading file request from peer %s: %v", remotePeerID, err)
		s.Write([]byte(`{"status": "error", "message": "Bad request format"}\n`))
		return
	}

	var request struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(requestBytes, &request); err != nil {
		log.Printf("[ERROR] Error decoding file request JSON from peer %s: %v", remotePeerID, err)
		s.Write([]byte(`{"status": "error", "message": "Invalid JSON request"}\n`))
		return
	}

	requestedPath := filepath.ToSlash(filepath.Clean(request.Path))
	log.Printf("[DEBUG] Peer %s requested file path: '%s'", remotePeerID, requestedPath)

	// --- Security Check: Validate against manifest ---
	n.manifestLock.RLock()
	asset, exists := n.manifest[requestedPath]
	n.manifestLock.RUnlock()

	if !exists {
		log.Printf("[SECURITY] Peer %s requested an unlisted file: '%s'. Denying request.", remotePeerID, requestedPath)
		s.Write([]byte(`{"status": "error", "message": "File not found in manifest"}\n`))
		return
	}

	// File exists in manifest, proceed to serve
	absolutePath := filepath.Join(n.projectRoot, requestedPath)
	file, err := os.Open(absolutePath)
	if err != nil {
		log.Printf("[ERROR] Failed to open listed file '%s': %v", absolutePath, err)
		s.Write([]byte(`{"status": "error", "message": "Could not open file on host"}\n`))
		return
	}
	defer file.Close()

	// Send confirmation before streaming
	s.Write([]byte(fmt.Sprintf(`{"status": "sending", "size": %d, "sha256": "%s"}\n`, asset.Size, asset.SHA256)))

	// Stream the file content
	_, err = io.Copy(s, file)
	if err != nil {
		log.Printf("[ERROR] Error while streaming file '%s' to peer %s: %v", absolutePath, remotePeerID, err)
	} else {
		log.Printf("[INFO] ✅ Successfully streamed file '%s' to peer %s", requestedPath, remotePeerID)
	}
}

func StartAPIAndNotify(node *Libp2pNode, portInfoFilePath string) {
	mux := http.NewServeMux()

	// --- Register API v1 Endpoints ---
	
	// Endpoint to get the node's own P2P addresses
	mux.HandleFunc("/api/v1/addrs", func(w http.ResponseWriter, r *http.Request) {
		var addrs []string
		for _, addr := range node.Addrs() {
			addrs = append(addrs, fmt.Sprintf("%s/p2p/%s", addr, node.ID()))
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"peerId": node.ID().String(),
			"addrs":  addrs,
		})
	})
	
	// NEW: Endpoint for the Python launcher to query for assets
	mux.HandleFunc("/api/v1/query/asset", node.handleQueryAsset)

	// NEW: Endpoint for the Python launcher to trigger a manifest refresh
	mux.HandleFunc("/api/v1/manifest/refresh", node.handleRefreshManifest)

	// --- Start Server and Notify Launcher ---

	listener, port, err := listenOnAvailablePort(22616, 22700)
	if err != nil {
		log.Fatalf("FATAL: Could not find an open port for the API server: %v", err)
	}
	log.Printf("[INFO] ✅ API Server found open port: %d", port)

	portInfo := map[string]int{"api_port": port}
	portBytes, _ := json.Marshal(portInfo)
	if err := os.WriteFile(portInfoFilePath, portBytes, 0644); err != nil {
		log.Fatalf("FATAL: Could not write port info to file %s: %v", portInfoFilePath, err)
	}
	log.Printf("[INFO] Successfully wrote API port to %s", portInfoFilePath)

	log.Printf("[INFO] Starting API server on %s with registered endpoints:", listener.Addr().String())
	log.Println("  - /api/v1/addrs")
	log.Println("  - /api/v1/query/asset")
	log.Println("  - /api/v1/manifest/refresh (POST)") // Add new endpoint to log

	if err := http.Serve(listener, mux); err != nil {
		log.Fatalf("HTTP API server failed: %v", err)
	}
}

// broadcastManifest sends the node's current manifest to all connected peers.
func (n *Libp2pNode) broadcastManifest(ctx context.Context) {
	log.Println("[ManifestBroadcast] Broadcasting updated manifest to all connected peers...")
	
	n.manifestLock.RLock()
	manifestBytes, err := json.Marshal(n.manifest)
	n.manifestLock.RUnlock()

	if err != nil {
		log.Printf("[ManifestBroadcast] ERROR: Failed to marshal own manifest for broadcast: %v", err)
		return
	}

	// Get a list of all currently connected peers
	connectedPeers := n.Host.Network().Peers()
	
	if len(connectedPeers) == 0 {
		log.Println("[ManifestBroadcast] No peers connected, skipping broadcast.")
		return
	}

	successfulBroadcasts := 0
	var wg sync.WaitGroup
	// Send the manifest to each peer in a separate goroutine for parallelism.
	for _, pid := range connectedPeers {
		if pid == n.Host.ID() {
			continue // Don't send to ourselves
		}
		wg.Add(1)
		go func(peerID peer.ID) {
			defer wg.Done()
			log.Printf("[ManifestBroadcast] Sending manifest to peer %s...", peerID)
			s, err := n.Host.NewStream(ctx, peerID, ManifestExchangeProtocolID)
			if err != nil {
				log.Printf("[ManifestBroadcast] WARN: Could not open stream to peer %s for broadcast: %v", peerID, err)
				return
			}
			defer s.Close()

			_, err = s.Write(manifestBytes)
			if err != nil {
				log.Printf("[ManifestBroadcast] ERROR: Failed to send manifest during broadcast to peer %s: %v", peerID, err)
			} else {
				successfulBroadcasts++
			}
		}(pid)
	}

	wg.Wait()
	log.Printf("[ManifestBroadcast] Broadcast complete. Sent manifest to %d/%d peers.", successfulBroadcasts, len(connectedPeers))
}

// handleRefreshManifest is the HTTP handler for the launcher's request to rescan assets.
func (n *Libp2pNode) handleRefreshManifest(w http.ResponseWriter, r *http.Request) {
	// We only accept POST requests for this endpoint.
	if r.Method != http.MethodPost {
		http.Error(w, `{"error": "Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}
	
	log.Println("[API Refresh] Received request to refresh asset manifest.")

	// Call the existing manifest generation function.
	err := n.generateManifest()
	if err != nil {
		log.Printf("[API Refresh] ERROR: Failed to regenerate manifest: %v", err)
		http.Error(w, `{"error": "Failed to regenerate manifest"}`, http.StatusInternalServerError)
		return
	}

	// After a successful regeneration, broadcast the new manifest to peers.
	// We run this in a goroutine so it doesn't block the HTTP response.
	go n.broadcastManifest(context.Background())

	// Respond with success.
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "success",
		"message": "Manifest refresh initiated and broadcast scheduled.",
	})
	log.Println("[API Refresh] Manifest refresh complete. Broadcast initiated.")
}


// Connected is called by the network whenever a new connection to a peer is established.
func (cn *connectionNotifee) Connected(net network.Network, conn network.Conn) {
	remotePeerID := conn.RemotePeer()
	log.Printf("[Connection] Established new connection with peer: %s", remotePeerID)
	
	// Launch a goroutine to request the manifest so we don't block the notifier.
	go cn.node.requestManifestFromPeer(cn.ctx, remotePeerID)
}

// Disconnected is called when a connection is closed.
func (cn *connectionNotifee) Disconnected(net network.Network, conn network.Conn) {
	remotePeerID := conn.RemotePeer()
	log.Printf("[Connection] Disconnected from peer: %s", remotePeerID)
	
	// Remove the peer's manifest from our cache.
	cn.node.peerManifestsLock.Lock()
	delete(cn.node.peerManifests, remotePeerID)
	cn.node.peerManifestsLock.Unlock()
	log.Printf("[ManifestExchange] Cleared cached manifest for peer %s", remotePeerID)
}

// Listen and ListenClose are required by the interface but we don't need them.
func (cn *connectionNotifee) Listen(net network.Network, a multiaddr.Multiaddr) {}
func (cn *connectionNotifee) ListenClose(net network.Network, a multiaddr.Multiaddr) {}

// --- Helper Functions ---

func listenOnAvailablePort(startPort, endPort int) (net.Listener, int, error) {
	for port := startPort; port <= endPort; port++ {
		listenAddr := fmt.Sprintf("127.0.0.1:%d", port)
		log.Printf("[DEBUG] API trying to listen on %s...", listenAddr)
		listener, err := net.Listen("tcp", listenAddr)
		if err == nil {
			return listener, port, nil // Success
		}
		log.Printf("[WARN] Port %d is already in use, trying next...", port)
	}
	return nil, 0, fmt.Errorf("no available ports found in range %d-%d", startPort, endPort)
}

func loadOrGenerateIdentity(path string) (crypto.PrivKey, error) {
	if _, err := os.Stat(path); err == nil {
		keyBytes, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		// FIX: Check key size before unmarshalling
		if len(keyBytes) != 32 {
			log.Printf("[WARN] Identity file at '%s' has incorrect size (%d bytes, expected 32). Deleting and regenerating.", path, len(keyBytes))
			if err := os.Remove(path); err != nil {
				return nil, fmt.Errorf("failed to remove invalid identity file: %w", err)
			}
			// Fall through to generation
		} else {
			log.Printf("[INFO] Loaded existing node identity from %s", path)
			return crypto.UnmarshalSecp256k1PrivateKey(keyBytes)
		}
	}

	// Generation logic
	log.Printf("[INFO] Generating new node identity and saving to %s", path)
	priv, _, err := crypto.GenerateKeyPair(crypto.Secp256k1, -1)
	if err != nil {
		return nil, err
	}
	keyBytes, err := crypto.MarshalPrivateKey(priv)
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(path, keyBytes, 0600); err != nil {
		return nil, err
	}
	return priv, nil
}

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