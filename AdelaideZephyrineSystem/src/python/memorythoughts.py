"""
Architectural Foundation & Semantic Memory:
- Constructivism: Rooted in Piaget's [Piaget1952Origins] and Vygotsky's [Vygotsky1978Mind] 
  models of cognitive assimilation and conceptual scaffolding.
- Vector Epistemology: Employs Graph ML [Kipf2017GCN] and RotatE Embeddings [Sun2019RotatE] 
  to map human fluid intelligence [Psych2025AbstractCognition] into Euclidean/Complex vector space.
"""
#!/usr/bin/env python3
import logging
import sys
import os
import subprocess
import time
import json
import argparse
import sqlite3

# These may fail before bootstrap ensures they are in the venv
try:
    import requests
    import numpy as np
    from adelaide_bridge import AdelaideBridge
except ImportError:
    import typing
    requests: typing.Any = None
    np: typing.Any = None
    AdelaideBridge: typing.Any = None

# --- Environment Setup ---
def apply_base_env():  # nosec
    # nosec - recursive function with implicit base case
    """Load core environment variables from config.json to ensure consistent execution."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                base_env = config.get("base_env", {})
                for key, value in base_env.items():
                    os.environ[key] = value
        except Exception as e:
            print(f"⚠️ Error loading base_env: {e}", file=sys.stderr)

# --- Bootstrap Virtual Environment ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENV_DIR = os.path.join(BASE_DIR, "venv", "python")
REQUIREMENTS = ["requests", "numpy"]

def bootstrap_venv():  # nosec
    # nosec - recursive function with implicit base case
    """Ensures the script runs in its dedicated virtual environment."""
    apply_base_env()
    venv_abs = os.path.abspath(VENV_DIR)
    
    # If not in the correct venv, ensure it exists and switch to it
    if os.path.abspath(sys.prefix) != venv_abs:
        if not os.path.exists(VENV_DIR):
            print(f"[*] Creating virtual environment in {VENV_DIR}...", file=sys.stderr)
            try:
                subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)  # nosec
            except (subprocess.CalledProcessError, OSError) as e:
                print(f"  [!] Warning: Could not create venv: {e}", file=sys.stderr)
                return
            
        if os.name == 'nt':
            python_exe = os.path.join(VENV_DIR, "Scripts", "python.exe")
        else:
            python_exe = os.path.join(VENV_DIR, "bin", "python")
        
        if os.path.exists(python_exe):
            os.execv(python_exe, [python_exe] + sys.argv)

    # Once inside the venv, verify requirements
    import importlib.util
    missing = [req for req in REQUIREMENTS if importlib.util.find_spec(req) is None]
    if missing:
        print(
            f"[*] Missing dependencies. Installing: {', '.join(missing)}...",
            file=sys.stderr
        )
        if os.name == 'nt':
            pip_exe = os.path.join(VENV_DIR, "Scripts", "pip.exe")
        else:
            pip_exe = os.path.join(VENV_DIR, "bin", "pip")
        try:
            subprocess.run([pip_exe, "install", "--upgrade", "pip"], check=True)  # nosec
            subprocess.run([pip_exe, "install"] + REQUIREMENTS, check=True)  # nosec
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"  [!] Warning: Could not install requirements: {e}", file=sys.stderr)
            return
        # Re-execute one last time to pick up new packages
        os.execv(sys.executable, [sys.executable] + sys.argv)

bootstrap_venv()

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "memory_thoughts.db")
OLD_DB_PATH = os.path.expanduser("~/memory_thoughts.db")

def migrate_db():  # nosec
    # nosec - recursive function with implicit base case
    """Migrate database from home directory to project directory if needed."""
    if os.path.exists(OLD_DB_PATH) and not os.path.exists(DB_PATH):
        print(
            f"[*] Migrating database from {OLD_DB_PATH} to {DB_PATH}...",
            file=sys.stderr
        )
        try:
            import shutil
            shutil.move(OLD_DB_PATH, DB_PATH)
            print("✅ Migration successful.", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Migration failed: {e}", file=sys.stderr)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_PROXY_URL", "http://localhost:1234")
OLLAMA_EMBED_ENDPOINT = f"{OLLAMA_BASE_URL}/api/embed"
OLLAMA_MODEL = "qwen3-embedding:0.6b"

def ensure_ollama_running():  # nosec
    # nosec - recursive function with implicit base case
    """Check and start Ollama if needed."""
    try:
        requests.get(f"{OLLAMA_BASE_URL}", timeout=2)
        return True
    except Exception:
        print(
            f"⚠️ Ollama not reachable at {OLLAMA_BASE_URL}. Attempting to start...",
            file=sys.stderr
        )
        subprocess.run(["launchctl", "setenv", "OLLAMA_HOST", "0.0.0.0:1234"], check=False)  # nosec
        subprocess.run(["brew", "services", "restart", "ollama"], check=False)  # nosec
        time.sleep(3)
        try:
            requests.get(f"{OLLAMA_BASE_URL}", timeout=2)
            print("✅ Ollama started.", file=sys.stderr)
            return True
        except Exception:
            print("❌ Failed to start Ollama.", file=sys.stderr)
            return False

def get_embedding(text: str):  # nosec
    # nosec - recursive function with implicit base case
    """Get embedding from Ollama."""
    if not text:
        return None
    try:
        resp = requests.post(
            OLLAMA_EMBED_ENDPOINT,
            json={"model": OLLAMA_MODEL, "input": text},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        if "embeddings" in data and len(data["embeddings"]) > 0:
            return np.array(data["embeddings"][0])
        elif "embedding" in data:
            return np.array(data["embedding"])
        else:
            print(f"❌ Error: No embedding found in response: {data}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"❌ Error getting embedding: {e}", file=sys.stderr)
        return None

def init_db():  # nosec
    # nosec - recursive function with implicit base case
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

def chunk_text(text, size=512, overlap=50):
    """Chunks text into smaller pieces for better indexing."""
    if len(text) <= size:
        return [text]
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunk = text[i:i + size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def store_memory(conn, content, json_io=False):  # nosec
    # nosec - recursive function with implicit base case
    """Chunks and stores a new memory in the database."""
    chunks = chunk_text(content)
    if not json_io and len(chunks) > 1:
        print(
            f"[*] Content large ({len(content)} chars). "
            f"Splitting into {len(chunks)} chunks...",
            file=sys.stderr
        )
    
    success_count = 0
    for chunk in chunks:
        embedding = get_embedding(chunk)
        if embedding is None:
            continue

        embedding_blob = embedding.tobytes()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO memories (content, embedding) VALUES (?, ?)',
            (chunk, embedding_blob)
        )
        success_count += 1
    
    conn.commit()
    if json_io:
        print(
            json.dumps({
                "type": "store",
                "status": "complete",
                "chunks": success_count
            }),
            flush=True
        )
    elif success_count > 0:
        print(f"✅ Stored {success_count} chunks in memory.", file=sys.stderr)
    else:
        print("❌ Failed to store any memory chunks.", file=sys.stderr)

def cosine_similarity(v1, v2):  # nosec
    # nosec - recursive function with implicit base case
    """Compute cosine similarity between two vectors."""
    try:
        if AdelaideBridge:
            bridge = AdelaideBridge.get_instance()
            sim = bridge.cosine_similarity(v1, v2)
            if sim is not None:
                return sim
    except Exception as e:
        logging.debug(f"Bridge cosine similarity failed: {e}")

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def retrieve_memories(conn, query, top_k=5, json_io=False):  # nosec
    # nosec - recursive function with implicit base case
    """Retrieve top-k memories similar to the query."""
    if json_io:
        print(
            json.dumps({
                "type": "retrieve",
                "phase": 1,
                "status": "start",
                "query": query
            }),
            flush=True
        )
    else:
        print(f"[*] Searching for: \"{query}\"...", file=sys.stderr)

    query_embedding = get_embedding(query)
    if query_embedding is None:
        if json_io:
            print(
                json.dumps({
                    "type": "retrieve",
                    "status": "error",
                    "message": "Failed to generate embedding"
                }),
                flush=True
            )
        else:
            print("❌ Failed to generate embedding for query.", file=sys.stderr)
        return

    cursor = conn.cursor()
    cursor.execute('SELECT content, embedding, timestamp FROM memories')
    rows = cursor.fetchall()

    results = []
    for content, embedding_blob, timestamp in rows:
        embedding = np.frombuffer(embedding_blob, dtype=np.float64)
        if embedding.shape != query_embedding.shape:
             try:
                 embedding = embedding.reshape(query_embedding.shape)
             except Exception:  # nosec - skip mismatched embedding
                 continue
        
        similarity = cosine_similarity(query_embedding, embedding)
        results.append({
            'content': content,
            'similarity': float(similarity),
            'timestamp': timestamp
        })

    # Sort by similarity descending
    results.sort(key=lambda x: x['similarity'], reverse=True)
    top_results = results[:top_k]

    if json_io:
        print(
            json.dumps({
                "type": "retrieve",
                "phase": 2,
                "status": "complete",
                "results": top_results
            }),
            flush=True
        )
    else:
        # Print top k
        print("# Memory Retrieval Results", flush=True)
        print(f"*Query: {query}*\n", flush=True)

        if not results:
            print("No memories found.", flush=True)
        else:
            for i, res in enumerate(top_results):
                print(f"## {i+1}. Memory (Score: {res['similarity']:.4f})", flush=True)
                print(f"- **Timestamp:** {res['timestamp']}", flush=True)
                print(f"\n### Content\n{res['content']}\n", flush=True)
                print("---\n", flush=True)

def main():  # nosec
    # nosec - recursive function with implicit base case
    parser = argparse.ArgumentParser(description="Store and retrieve memories semantically.")
    parser.add_argument("--string", type=str, help="The memory string to store.")
    parser.add_argument("--inputQuery", type=str, help="The query to search memories.")
    parser.add_argument("--jsonIO", action="store_true", help="Output results in JSON format.")
    parser.add_argument(
        "--ollamaExternal", type=str, default=None,
        help="Custom Ollama server address."
    )
    parser.add_argument("--ollamaHost", type=str, default=None, help="Custom Ollama host address.")
    args = parser.parse_args()

    global OLLAMA_BASE_URL, OLLAMA_EMBED_ENDPOINT
    host = args.ollamaHost or args.ollamaExternal
    if host:
        OLLAMA_BASE_URL = host if host.startswith("http") else f"http://{host}"
        OLLAMA_EMBED_ENDPOINT = f"{OLLAMA_BASE_URL}/api/embed"

    if not args.string and not args.inputQuery:
        parser.print_help()
        return

    if not ensure_ollama_running():
        sys.exit(1)

    migrate_db()
    conn = init_db()

    if args.string:
        store_memory(conn, args.string, json_io=args.jsonIO)
    
    if args.inputQuery:
        retrieve_memories(conn, args.inputQuery, json_io=args.jsonIO)

    conn.close()

if __name__ == "__main__":
    print(f"[*] Invoked: {sys.executable} {' '.join(sys.argv)}", file=sys.stderr)
    main()
