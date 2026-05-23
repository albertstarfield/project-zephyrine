import os
import sys
import threading
import sqlite3
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import webview

app = FastAPI()

# Configuration
ADA_BACKEND_URL = "http://localhost:11420"
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "UI_Database", "assistant_session.db")
DIST_DIR = os.path.join(os.path.dirname(__file__), "frontend", "dist")

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.get("/api/messages")
def get_messages():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content, timestamp FROM messages ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    
    # Save User message to DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (role, content) VALUES (?, ?)", ("user", user_message))
    conn.commit()

    # Proxy to Ada Backend (assuming Ollama compatible endpoint or custom endpoint)
    # The Ada backend runs on 11420. If it expects /api/chat like ollama:
    payload = {
        "model": "stella-icarus",  # Or whatever model Ada routes
        "messages": [{"role": "user", "content": user_message}],
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # We send to the Ada backend
            response = await client.post(f"{ADA_BACKEND_URL}/api/chat", json=payload, timeout=60.0)
            if response.status_code == 200:
                resp_json = response.json()
                bot_reply = resp_json.get("message", {}).get("content", "Empty response")
            else:
                bot_reply = f"Backend returned error: {response.status_code} - {response.text}"
    except Exception as e:
        bot_reply = f"Could not connect to Ada backend: {str(e)}"
        
    # Save Bot reply to DB
    cursor.execute("INSERT INTO messages (role, content) VALUES (?, ?)", ("assistant", bot_reply))
    conn.commit()
    conn.close()
    
    return {"reply": bot_reply}

@app.post("/api/exit")
def exit_app():
    # Attempt to gracefully exit the entire application
    os._exit(0)
    return {"status": "exiting"}

@app.get("/api/docs/readme")
def get_readme():
    readme_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "README.md")
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except Exception as e:
        return {"error": str(e)}, 500

@app.get("/api/docs/license")
def get_license():
    license_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "license.md")
    try:
        with open(license_path, "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except Exception as e:
        return {"error": str(e)}, 500

@app.get("/api/user_info")
def get_user_info():
    import getpass
    try:
        username = getpass.getuser()
    except Exception:
        username = "User"
    return {"username": username}

# --- Knowledge Stack Backend ---
from fastapi import UploadFile, File, Form
import fitz  # PyMuPDF
import networkx as nx
import numpy as np
import json
import uuid

LITERATURE_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "UI_Database", "literatureRefIndex.db")
GRAPH_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "UI_Database", "literature.graphml")

# Ensure dir
os.makedirs(os.path.dirname(LITERATURE_DB_PATH), exist_ok=True)

def init_knowledge_db():
    conn = sqlite3.connect(LITERATURE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT,
            domain TEXT,
            content TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()

init_knowledge_db()

embedding_model = None
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

from fastapi.responses import StreamingResponse

@app.post("/api/knowledgestackfrontend/upload")
async def upload_knowledge(files: list[UploadFile] = File(...), domain: str = Form(...)):
    # Read files into memory first to avoid closure issues
    files_data = []
    for file in files:
        content_bytes = await file.read()
        files_data.append((file.filename, content_bytes))
        
    async def process_and_stream():
        conn = sqlite3.connect(LITERATURE_DB_PATH)
        cursor = conn.cursor()
        model = get_embedding_model()
        
        if os.path.exists(GRAPH_PATH):
            try:
                G = nx.read_graphml(GRAPH_PATH)
            except Exception:
                G = nx.Graph()
        else:
            G = nx.Graph()

        all_chunks = []
        # Pre-process and chunk all files
        for filename, content_bytes in files_data:
            content = ""
            ext = filename.split('.')[-1].lower()
            
            if ext == 'txt':
                content = content_bytes.decode('utf-8', errors='ignore')
            elif ext == 'pdf':
                doc = fitz.open(stream=content_bytes, filetype="pdf")
                for page in doc:
                    content += page.get_text() + "\n"
            else:
                continue
                
            if not content.strip(): continue
            
            doc_node = f"doc_{filename}"
            G.add_node(domain, label=domain, type="domain")
            G.add_node(doc_node, label=filename, domain=domain, type="document")
            G.add_edge(domain, doc_node)
            
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            chunks = []
            current_chunk = ""
            for p in paragraphs:
                if len(current_chunk) + len(p) > 500:
                    if current_chunk: chunks.append(current_chunk)
                    current_chunk = p
                else:
                    current_chunk += "\n\n" + p if current_chunk else p
            if current_chunk:
                chunks.append(current_chunk)
                
            if not chunks:
                chunks = [content[i:i+500] for i in range(0, len(content), 500)]
                
            for chunk in chunks:
                all_chunks.append((filename, doc_node, chunk))

        total_chunks = len(all_chunks)
        if total_chunks == 0:
            yield json.dumps({"progress": 100, "status": "success"}) + "\n"
            return

        for i, (filename, doc_node, chunk_text) in enumerate(all_chunks):
            emb = model.encode(chunk_text)
            emb_blob = emb.astype(np.float32).tobytes()
            chunk_id = str(uuid.uuid4())
            
            cursor.execute("INSERT INTO documents (id, filename, domain, content, embedding) VALUES (?, ?, ?, ?, ?)",
                           (chunk_id, filename, domain, chunk_text, emb_blob))
            
            G.add_node(chunk_id, label=f"Chunk {i+1}", type="chunk")
            G.add_edge(doc_node, chunk_id)
            
            # Yield progress every chunk or periodically
            progress = int(((i + 1) / total_chunks) * 100)
            yield json.dumps({"progress": progress}) + "\n"

        conn.commit()
        conn.close()
        nx.write_graphml(G, GRAPH_PATH)
        yield json.dumps({"progress": 100, "status": "success"}) + "\n"

    return StreamingResponse(process_and_stream(), media_type="application/x-ndjson")

@app.get("/api/knowledgestackfrontend/documents")
def get_documents():
    conn = sqlite3.connect(LITERATURE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, domain FROM documents")
    rows = cursor.fetchall()
    conn.close()
    
    result = {}
    for filename, domain in rows:
        if domain not in result:
            result[domain] = []
        result[domain].append(filename)
    return result

@app.get("/api/knowledgestackfrontend/search")
def search_knowledge(q: str):
    conn = sqlite3.connect(LITERATURE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, domain, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return {"results": []}
        
    model = get_embedding_model()
    q_emb = model.encode(q)
    
    results = []
    for filename, domain, content, emb_blob in rows:
        doc_emb = np.frombuffer(emb_blob, dtype=np.float32)
        # Cosine similarity
        score = np.dot(q_emb, doc_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(doc_emb))
        if score > 0.1:  # Simple threshold
            snippet = content[:200].replace('\n', ' ') + "..."
            results.append({
                "filename": filename,
                "domain": domain,
                "score": float(score),
                "snippet": snippet
            })
            
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"results": results[:5]}

@app.get("/api/knowledgestackfrontend/graph")
def get_graph():
    if not os.path.exists(GRAPH_PATH):
        return []
    try:
        G = nx.read_graphml(GRAPH_PATH)
        elements = []
        for n, d in G.nodes(data=True):
            elements.append({"data": {"id": n, "label": d.get("label", n), "type": d.get("type", "unknown")}})
        for u, v in G.edges():
            elements.append({"data": {"source": u, "target": v}})
        return elements
    except Exception:
        return []

# Mount static files
if os.path.exists(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="static")
else:
    @app.get("/")
    def no_dist():
        return HTMLResponse("<h1>Please run `npm run build` inside frontend/</h1>")

import socket

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def run_server(port):
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

if __name__ == "__main__":
    ui_port = get_free_port()
    # Start FastAPI in a background thread
    server_thread = threading.Thread(target=run_server, args=(ui_port,), daemon=True)
    server_thread.start()
    
    # Launch PyWebview native window
    window = webview.create_window(
        "Adelaide Zephyrine Assistant",
        f"http://127.0.0.1:{ui_port}",
        width=1000,
        height=800,
        frameless=False, # Set to True if we want fully custom window frame
        easy_drag=True
    )
    
    webview.start(debug=True)
