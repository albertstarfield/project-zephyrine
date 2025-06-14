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

func NewNode(ctx context.Context, listenPort int, identityPath string) (*Libp2pNode, error) {
	privKey, err := loadOrGenerateIdentity(identityPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load/generate identity: %w", err)
	}

	listenAddr, _ := multiaddr.NewMultiaddr(fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", listenPort))

	h, err := libp2p.New(
		libp2p.ListenAddrs(listenAddr),
		libp2p.Identity(privKey),
	)
	if err != nil {
		return nil, err
	}

	// FIX: Determine project root from the current working directory,
	// which is set by the Python launcher.
	pRoot, err := filepath.Abs(".")
	if err != nil {
		return nil, fmt.Errorf("could not determine project root directory: %w", err)
	}
	log.Printf("[DEBUG] Project Root detected as: %s", pRoot)

	node := &Libp2pNode{
		Host:        h,
		projectRoot: pRoot,
		relayDir:    filepath.Join(pRoot, "systemCore", "engineMain", "meshCommunicationRelay"),
	}

	node.SetStreamHandler(FileProtocolID, node.handleFileStream)

	if err := os.MkdirAll(node.relayDir, 0755); err != nil {
		return nil, fmt.Errorf("could not create relay directory: %w", err)
	}

	if err := node.generateManifest(); err != nil {
		log.Printf("[WARN] Could not generate manifest on startup: %v", err)
	}

	go node.initDHT(ctx)

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
		"staticmodelpool":                   filepath.Join(n.projectRoot, "systemCore", "engineMain", "staticmodelpool"),
		"llama-cpp-python_build":            filepath.Join(n.projectRoot, "llama-cpp-python_build"),
		"stable-diffusion-cpp-python_build": filepath.Join(n.projectRoot, "stable-diffusion-cpp-python_build"),
		"pywhispercpp_build":                filepath.Join(n.projectRoot, "pywhispercpp_build"),
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
	mux.HandleFunc("/addrs", func(w http.ResponseWriter, r *http.Request) {
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

	log.Printf("[INFO] Starting API server on %s", listener.Addr().String())
	if err := http.Serve(listener, mux); err != nil {
		log.Fatalf("HTTP API server failed: %v", err)
	}
}

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