
# ZephyMesh: Decentralized Asset & Compute Network

ZephyMesh is the decentralized peer-to-peer (P2P) backbone for Project Zephyrine. It is designed to create a resilient, self-healing network that reduces reliance on centralized servers, accelerates setup times, and enables distributed computing capabilities, even in challenging network environments.

## Core Principles

1.  **Resilience over Reliability:** The network is designed with the assumption that connections are ephemeral and unreliable. It prioritizes eventual consistency and data delivery over maintaining constant, stable links.
2.  **Zero Configuration:** A user should be able to launch the application and automatically join the local mesh network without any manual setup (e.g., entering IP addresses).
3.  **Resource Aggregation:** The mesh should aggregate the resources of all participating nodes (storage, compute) and make them available to the network as a whole.
4.  **Security First:** Communication is encrypted, and nodes cannot access arbitrary files on a peer's system. All shareable assets are explicitly declared in a verifiable manifest.

## Architecture Overview

ZephyMesh is built using **[libp2p](https://libp2p.io/)**, the modular networking stack that powers IPFS and other next-generation decentralized systems. Each instance of Project Zephyrine runs an embedded ZephyMesh node, which performs several key functions:

-   **Peer Discovery:** Uses a Kademlia Distributed Hash Table (DHT) to discover and connect to other nodes, both on the local network (LAN) and potentially across the internet (WAN).
-   **Asset Manifesting:** Scans local directories to create a "manifest"—a verifiable list of all the assets (AI models, compiled libraries, cache files) it can share with the network.
-   **P2P Communication:** Establishes secure, multiplexed streams with other peers to exchange manifests and transfer data.
-   **Local API Server:** Provides a simple HTTP API for the main Python application to interact with the mesh (e.g., query for assets, request distributed inference).

 <!-- It would be great to add the Mermaid diagram we designed here -->

---

## Feature Roadmap & Status

This section details the planned capabilities of the ZephyMesh network.

### Priority 1: Core P2P and File Sharing (LAN Focus)

These features form the foundation of the asset distribution system.

| Functionality | Status | Description |
| :--- | :--- | :--- |
| **P2P Host & Discovery** | ✅ **Implemented** | The node can join the network using a DHT and find peers. |
| **Local API Server** | ✅ **Implemented** | The node can receive HTTP commands from the Python launcher. |
| **Asset Manifesting** | ✅ **Implemented** | The node scans `staticmodelpool`, `huggingface_cache`, and build directories to know what files it has. |
| **Manifest Sharing** | ✅ **Implemented** | Nodes automatically exchange manifests upon connecting. |
| **Query Asset API** | ✅ **Implemented** | `GET /api/v1/query/asset` allows the launcher to ask "who has this file?". |
| **Refresh Manifest API**| ✅ **Implemented** | `POST /api/v1/manifest/refresh` allows the launcher to signal that a new asset has been downloaded. |
| **P2P File Transfer** | 🚧 **Partially Implemented**| The **server** side (`handleFileStream`) is built and can serve files to peers. The **client** side (initiating a download) needs to be implemented in the Python launcher. |

### Priority 2: Crowd-Inference-Assist (Distributed Compute)

This layer transforms the mesh from a file-sharing network into a distributed computing grid.

| Functionality | Status | Description |
| :--- | :--- | :--- |
| **Peer Capability Broadcasting** | 📝 **Planned** | Nodes will announce their hardware capabilities (GPU, RAM, etc.) to the mesh. |
| **Distributed Inference API** | 📝 **Planned** | `POST /api/v1/inference/request` - An API for the Python application to submit inference jobs to the mesh. |
| **P2P Inference Protocol** | 📝 **Planned** | A `libp2p` stream protocol for sending inference jobs and receiving results between capable peers. |
| **Load Balancing & Peer Selection**| 📝 **Planned** | Logic within the node to select the best peer for a task based on capabilities and load. |

### Priority 3: WAN Resilience & Delay-Tolerant Networking (DTN)

These features enhance robustness for operation over the wider internet, especially in environments with high latency or intermittent connectivity, by adopting principles from Delay-Tolerant Networking (DTN).

| Functionality | Status | Description |
| :--- | :--- | :--- |
| **Content Addressing (CID)** | 📝 **Planned** | Transition from file paths to Content Identifiers (CIDs), the core principle of IPFS, for location-independent, verifiable data. |
| **Bundle Protocol (Store-and-Forward)** | 📝 **Planned** | Implement a Delay-Tolerant Networking (DTN) bundle agent to store and forward data, allowing transfers to complete over long, disrupted connections. |
| **Latency-Aware Peer Management** | 📝 **Planned** | Nodes will measure latency to peers to make intelligent routing decisions. |
| **Dynamic Service Disablement** | 📝 **Planned** | The network will automatically disable real-time services like crowd-inference for peers with high latency, while still using them for background DTN file transfers. |

---

## Developer Notes

### Building the Node

The ZephyMesh node is managed and compiled entirely by the main `launcher.py` script. Manual building is only necessary for direct development. From the `zephyMeshNetwork/` directory, run:

```bash
# Ensure all dependencies are present in go.mod
go mod tidy

# Build the binary
go build -o zephymesh_node_compiled ./cmd/zephymesh_node
```

### Code Structure

The project follows the standard Go project layout:

-   `cmd/zephymesh_node/main.go`: The main entry point of the application. Handles command-line flags, signal handling, and initializes the node.
-   `internal/node/node.go`: The core logic for the P2P node. Contains the `Libp2pNode` struct and methods for handling P2P protocols, the HTTP API, and manifest generation.
-   `go.mod` / `go.sum`: Go module files defining the project's dependencies.

### Interaction with the Python Launcher

-   **Lifecycle:** The Python launcher is responsible for compiling (if necessary), running, and terminating the `zephymesh_node_compiled` process.
-   **Working Directory:** The launcher MUST run the binary with the working directory set to the **project root**, not the `zephyMeshNetwork` sub-directory. This is critical for the node to correctly locate asset directories.
-   **IPC (Readiness Probe):** The Go node signals its readiness to the launcher by writing its API port to a `zephymesh_ports.json` file in the project root. The launcher waits for this file before proceeding.