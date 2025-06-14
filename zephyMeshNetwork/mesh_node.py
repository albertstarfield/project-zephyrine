# zephyMeshNetwork/mesh_node.py
#
# This program runs as a background process, turning each instance of the application
# into a peer in the ZephyMesh network. It uses the libp2p library to create a
# resilient, decentralized network for asset distribution, inspired by Syncthing.

import os
import sys
import json
import asyncio
import hashlib
import threading
import time
import random
from typing import Dict, List, Optional

# <<< MODIFICATION START: Resilient Import Block >>>
# This block will patiently retry importing necessary libraries, waiting for the
# main launcher's async pip install process to complete.

def _attempt_imports(max_retries=2, delay_seconds=5):
    """Tries to import necessary libraries, retrying on failure."""
    for attempt in range(max_retries):
        try:
            # Attempt to import all pip-installed dependencies here
            from loguru import logger
            from libp2p import new_host
            # REMOVED: from libp2p.typing import TProtocol
            from libp2p.network.stream.net_stream_interface import INetStream
            from libp2p.crypto.secp256k1 import create_new_key_pair
            from libp2p.peer.id import ID as PeerID
            from libp2p.peer.peerinfo import PeerInfo
            # REMOVED: from libp2p.exceptions import PeerIDException
            import trio

            # If all imports succeed, return the logger module to be used globally
            return logger
        except ModuleNotFoundError as e:
            # Use standard print for logging since loguru is not yet available
            print(
                f"[ZephyMesh Node] Waiting for dependencies... Import failed: {e}. Retrying in {delay_seconds}s (Attempt {attempt + 1}/{max_retries})",
                file=sys.stderr
            )
            time.sleep(delay_seconds)
    return None

logger = _attempt_imports()

if not logger:
    print("[ZephyMesh Node] CRITICAL: Failed to import dependencies after multiple retries. Exiting.", file=sys.stderr)
    sys.exit(1)

# Now that logger is confirmed to be imported, we can import the rest
from libp2p import new_host
from libp2p.typing import TProtocol
from libp2p.network.stream.net_stream_interface import INetStream
from libp2p.crypto.secp256k1 import create_new_key_pair
from libp2p.peer.id import ID as PeerID
from libp2p.peer.peerinfo import PeerInfo
from libp2p.exceptions import PeerIDException
import trio
# <<< MODIFICATION END >>>


# --- Configuration ---
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DIRS_TO_SERVE = {
    "staticmodelpool": os.path.join(PROJECT_ROOT_DIR, "systemCore", "engineMain", "staticmodelpool"),
    "submodules": {
        "llama-cpp-python_build": os.path.join(PROJECT_ROOT_DIR, "llama-cpp-python_build"),
        "stable-diffusion-cpp-python_build": os.path.join(PROJECT_ROOT_DIR, "stable-diffusion-cpp-python_build"),
        "pywhispercpp_build": os.path.join(PROJECT_ROOT_DIR, "pywhispercpp_build"),
    }
}

def get_random_port(start=20000, end=29999):
    return random.randint(start, end)

MESH_API_PORT = int(os.getenv("ZEPHYMESH_API_PORT", get_random_port()))
FILE_SERVER_PORT = int(os.getenv("ZEPHYMESH_FILE_PORT", get_random_port()))

while FILE_SERVER_PORT == MESH_API_PORT:
    FILE_SERVER_PORT = get_random_port()

PORT_INFO_FILE = os.path.join(PROJECT_ROOT_DIR, "zephymesh_ports.json")
IDENTITY_FILE = os.path.join(PROJECT_ROOT_DIR, "zephymesh_identity.key")
MANIFEST_FILENAME = "zephymesh_manifest.json"
BOOTSTRAP_PEERS = [
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTf5gpSchaIs1fl02P9",
    "/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ",
]
PROTOCOL_ID_MANIFEST = TProtocol("/zephymesh/manifest/1.0.0")
PROTOCOL_ID_FILE_TRANSFER = TProtocol("/zephymesh/file_transfer/1.0.0")


class Libp2pMeshNode:
    def __init__(self):
        self.host = None
        self.manifest: Dict[str, Dict] = {}
        self.running = True

    async def _calculate_sha256(self, file_path: str) -> Optional[str]:
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                while True:
                    byte_block = await asyncio.to_thread(f.read, 4096)
                    if not byte_block:
                        break
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Could not calculate SHA256 for {file_path}: {e}")
            return None

    async def _generate_manifest(self):
        logger.info("Generating asset manifest...")
        assets = {}
        for key, path_or_dict in ASSET_DIRS_TO_SERVE.items():
            if key == "submodules":
                logger.warning("Sharing of 'submodules' directory content is not yet implemented.")
                continue
            
            if isinstance(path_or_dict, str) and os.path.isdir(path_or_dict):
                for filename in os.listdir(path_or_dict):
                    file_path = os.path.join(path_or_dict, filename)
                    if os.path.isfile(file_path):
                        checksum = await self._calculate_sha256(file_path)
                        if checksum:
                            relative_path = os.path.join(key, filename).replace("\\", "/")
                            assets[relative_path] = {
                                "sha256": checksum,
                                "size": os.path.getsize(file_path)
                            }
        self.manifest = {"assets": assets}
        logger.success(f"Manifest generated with {len(assets)} assets.")
        try:
            with open(os.path.join(PROJECT_ROOT_DIR, MANIFEST_FILENAME), 'w') as f:
                json.dump(self.manifest, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save local manifest file: {e}")

    async def _file_transfer_handler(self, stream: INetStream) -> None:
        try:
            request_data = await stream.read()
            request = json.loads(request_data.decode())
            relative_path = request.get("path")
            logger.info(f"Received file request for: {relative_path}")

            if not relative_path or not isinstance(relative_path, str):
                await stream.write(b'{"error": "Invalid path"}')
                return

            base_dir = PROJECT_ROOT_DIR
            absolute_path = os.path.normpath(os.path.join(base_dir, relative_path))

            if not absolute_path.startswith(base_dir):
                logger.error(f"SECURITY: Attempt to access file outside of root: {relative_path}")
                await stream.write(b'{"error": "Access denied"}')
                return

            if os.path.isfile(absolute_path):
                file_size = os.path.getsize(absolute_path)
                await stream.write(json.dumps({"status": "sending", "size": file_size}).encode())
                
                with open(absolute_path, "rb") as f:
                    while True:
                        chunk = await asyncio.to_thread(f.read, 4096)
                        if not chunk:
                            break
                        await stream.write(chunk)
                logger.success(f"Successfully sent file: {relative_path}")
            else:
                await stream.write(b'{"error": "File not found"}')
        except Exception as e:
            logger.error(f"Error in file transfer handler: {e}")
        finally:
            await stream.close()
    
    async def _start_local_api(self):
        logger.info("Local API for launcher not yet implemented.")
        pass

    async def run(self):
        await self._generate_manifest()
        
        try:
            with open(IDENTITY_FILE, 'rb') as f:
                key_bytes = f.read()
            identity = create_new_key_pair(key_bytes)
            logger.info(f"Loaded existing node identity from {IDENTITY_FILE}")
        except FileNotFoundError:
            identity = create_new_key_pair()
            with open(IDENTITY_FILE, 'wb') as f:
                f.write(identity.private_key.serialize())
            logger.info(f"Created and saved new node identity to {IDENTITY_FILE}")

        listen_addr = f"/ip4/0.0.0.0/tcp/0"
        self.host = await new_host(key_pair=identity, listen_addrs=[listen_addr])

        logger.info(f"Node started with Peer ID: {self.host.get_id().to_string()}")
        for addr in self.host.get_listen_addrs():
            logger.info(f"Listening on: {addr}")

        self.host.set_stream_handler(PROTOCOL_ID_FILE_TRANSFER, self._file_transfer_handler)

        for peer in BOOTSTRAP_PEERS:
            try:
                ma = peer
                peer_id_str = peer.split("/")[-1]
                peer_id = PeerID.from_string(peer_id_str)
                peer_info = PeerInfo(peer_id, [ma])
                await self.host.connect(peer_info)
                logger.info(f"Connected to bootstrap peer: {peer_id_str}")
            except Exception as e:
                logger.warning(f"Failed to connect to bootstrap peer {peer}: {e}")

        await self._start_local_api()

        try:
            while self.running:
                logger.debug(f"Node running. Connected peers: {len(self.host.get_network().connections)}")
                await asyncio.sleep(60)
        except (KeyboardInterrupt, trio.Cancelled):
            logger.info("Shutdown signal received.")
        finally:
            await self.shutdown()

    async def shutdown(self):
        logger.info("Shutting down Libp2pMeshNode...")
        self.running = False
        if self.host:
            await self.host.close()
            logger.info("Libp2p host closed.")

async def main():
    logger.add("zephymesh_node.log", rotation="10 MB")
    
    node = Libp2pMeshNode()
    await node.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
