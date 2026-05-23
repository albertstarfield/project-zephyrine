import os
import sys
import time
import threading
import sqlite3
import httpx
import uvicorn
import psutil
import tiktoken
import gc
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import webview

# Global Performance Tuning: Disable Garbage Collection
gc.disable()

app = FastAPI()

class EngineStats:
    def __init__(self):
        self.boot_time = time.time()
        self.total_tokens = 0
        self.wcet_elp0 = 0.0
        self.wcet_elp1 = 0.0
        self.wcet_elp2 = 0.0
        self.wcet_elp3 = 0.0
        self.jitter_avg_us = 0.0
        self.jitter_max_us = 0.0
        self.wcel = 0.0
        self.wcet_watchdog_loop_us = 0.0
        self.wcet_main_loop_us = 0.0
        self.wcetr = 0.0
        self.history_1m = []
        self.wcel_history_1m = []
        
        # Histories for deltas
        self.wcet_elp0_hist = []
        self.wcet_elp1_hist = []
        self.wcet_elp2_hist = []
        self.wcet_wtdog_hist = []
        self.wcet_mloop_hist = []

engine_stats = EngineStats()

try:
    enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    enc = None

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

@app.post("/api/telemetry")
async def post_telemetry(req: Request):
    data = await req.json()
    now_ts = time.time()
    
    if "WCET_WatchdogLoop_uS" in data:
        val = float(data["WCET_WatchdogLoop_uS"])
        engine_stats.wcet_watchdog_loop_us = val
        engine_stats.wcet_wtdog_hist.append({"ts": now_ts, "val": val})
        
    if "WCET_mainLoop_uS" in data:
        val = float(data["WCET_mainLoop_uS"])
        engine_stats.wcet_main_loop_us = val
        engine_stats.wcet_mloop_hist.append({"ts": now_ts, "val": val})
        
    if "WCET_ELP0" in data:
        val = float(data["WCET_ELP0"])
        engine_stats.wcet_elp0 = val
        engine_stats.wcet_elp0_hist.append({"ts": now_ts, "val": val})
        
    if "WCET_ELP1" in data:
        val = float(data["WCET_ELP1"])
        engine_stats.wcet_elp1 = val
        engine_stats.wcet_elp1_hist.append({"ts": now_ts, "val": val})

    if "WCET_ELP2" in data:
        val = float(data["WCET_ELP2"])
        engine_stats.wcet_elp2 = val
        engine_stats.wcet_elp2_hist.append({"ts": now_ts, "val": val})

    if "WCET_ELP3" in data:
        val = float(data["WCET_ELP3"])
        engine_stats.wcet_elp3 = val
        # For ELP3 (1ms), we don't store 1000pts/s in history, we'll just track current
    
    if "Jitter_Avg_uS" in data:
        engine_stats.jitter_avg_us = float(data["Jitter_Avg_uS"])
    if "Jitter_Max_uS" in data:
        engine_stats.jitter_max_us = float(data["Jitter_Max_uS"])

    return JSONResponse({"status": "ok"})

@app.get("/api/messages")
def get_messages():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content, timestamp FROM messages ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]

@app.get("/api/adelaideenginestats")
def get_stats(queue_len: int = 0):
    now = time.time()
    uptime = now - engine_stats.boot_time
    
    # Cleanup history older than 60 seconds
    engine_stats.history_1m = [h for h in engine_stats.history_1m if now - h['ts'] <= 60]
    engine_stats.wcel_history_1m = [h for h in engine_stats.wcel_history_1m if now - h['ts'] <= 60]
    engine_stats.wcet_elp0_hist = [h for h in engine_stats.wcet_elp0_hist if now - h['ts'] <= 60]
    engine_stats.wcet_elp1_hist = [h for h in engine_stats.wcet_elp1_hist if now - h['ts'] <= 60]
    engine_stats.wcet_elp2_hist = [h for h in engine_stats.wcet_elp2_hist if now - h['ts'] <= 60]
    engine_stats.wcet_wtdog_hist = [h for h in engine_stats.wcet_wtdog_hist if now - h['ts'] <= 60]
    engine_stats.wcet_mloop_hist = [h for h in engine_stats.wcet_mloop_hist if now - h['ts'] <= 60]
    
    avg_1m_wcel = sum(h['val'] for h in engine_stats.wcel_history_1m) / len(engine_stats.wcel_history_1m) if engine_stats.wcel_history_1m else engine_stats.wcel
    
    def get_delta(hist):
        if not hist: return 0.0
        vals = [h['val'] for h in hist]
        return max(vals) - min(vals)
    
    return {
        "WCET_ELP0": engine_stats.wcet_elp0,
        "WCET_ELP0_delta": get_delta(engine_stats.wcet_elp0_hist),
        "WCET_ELP1": engine_stats.wcet_elp1,
        "WCET_ELP1_delta": get_delta(engine_stats.wcet_elp1_hist),
        "WCET_ELP2": engine_stats.wcet_elp2,
        "WCET_ELP2_delta": get_delta(engine_stats.wcet_elp2_hist),
        "WCET_ELP3": engine_stats.wcet_elp3,
        "Jitter_Avg_uS": engine_stats.jitter_avg_us,
        "Jitter_Max_uS": engine_stats.jitter_max_us,
        "WCEL": engine_stats.wcel,
        "WCEL_delta_1m": avg_1m_wcel,  # Sending the average as requested
        "WCET_WatchdogLoop_uS": engine_stats.wcet_watchdog_loop_us,
        "WCET_WatchdogLoop_uS_delta": get_delta(engine_stats.wcet_wtdog_hist),
        "WCET_mainLoop_uS": engine_stats.wcet_main_loop_us,
        "WCET_mainLoop_uS_delta": get_delta(engine_stats.wcet_mloop_hist),
        "MemoryConsumption_MB": psutil.Process().memory_info().rss / (1024*1024),
        "CPU_Consumption": psutil.Process().cpu_percent(interval=None),
        "sidecarProcessSpawned": engine_stats.boot_time,
        "sidecarProcessRunning": True,
        "WCETR": engine_stats.wcetr,
        "Total_Tokens_Processed": engine_stats.total_tokens,
        "Current_Uptime": uptime,
        "Current_Queue": queue_len,
        "History_1m": engine_stats.history_1m,
        "WCEL_History_1m": engine_stats.wcel_history_1m,
        "WCET_ELP0_Hist": engine_stats.wcet_elp0_hist,
        "WCET_ELP1_Hist": engine_stats.wcet_elp1_hist,
        "WCET_ELP2_Hist": engine_stats.wcet_elp2_hist,
        "WCET_WtDog_Hist": engine_stats.wcet_wtdog_hist,
        "WCET_mLoop_Hist": engine_stats.wcet_mloop_hist
    }

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
        t_start = time.time()
        async with httpx.AsyncClient() as client:
            # We send to the Ada backend
            response = await client.post(f"{ADA_BACKEND_URL}/api/chat", json=payload, timeout=60.0)
            t_end = time.time()
            elapsed = t_end - t_start
            engine_stats.wcet_elp2 = elapsed
            engine_stats.wcet_elp2_hist.append({"ts": t_end, "val": elapsed})            
            if response.status_code == 200:
                resp_json = response.json()
                bot_reply = resp_json.get("message", {}).get("content", "Empty response")
                
                # Calculate tokens and update stats
                if enc:
                    tokens = len(enc.encode(bot_reply))
                    engine_stats.total_tokens += tokens
                    if elapsed > 0:
                        engine_stats.wcetr = tokens / elapsed
                        engine_stats.history_1m.append({"ts": t_end, "val": engine_stats.wcetr})
            else:
                bot_reply = f"Backend returned error: {response.status_code} - {response.text}"
                engine_stats.wcet_elp1 = elapsed
    except Exception as e:
        bot_reply = f"Could not connect to Ada backend: {str(e)}"
        
    # Save Bot reply to DB
    cursor.execute("INSERT INTO messages (role, content) VALUES (?, ?)", ("assistant", bot_reply))
    conn.commit()
    conn.close()
    
    return {"reply": bot_reply}

@app.post("/api/exit")
def exit_app():
    import webview
    import threading
    
    def close_window():
        if webview.windows:
            webview.windows[0].destroy()
        else:
            os._exit(0)
            
    # Run in a separate thread to allow the HTTP response to return
    threading.Timer(0.5, close_window).start()
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
LITERATURE_GRAPH_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "UI_Database", "literature.graphml")

MEMORY_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "UI_Database", "memoryRefIndex.db")
MEMORY_GRAPH_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "UI_Database", "memory.graphml")

# Ensure dir
os.makedirs(os.path.dirname(LITERATURE_DB_PATH), exist_ok=True)

def init_knowledge_db():
    # Initialize Literature DB
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

    # Initialize Memory DB
    conn = sqlite3.connect(MEMORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            session TEXT,
            topic TEXT,
            content TEXT,
            embedding BLOB,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

    # Init GraphML for literature
    if not os.path.exists(LITERATURE_GRAPH_PATH):
        G = nx.DiGraph()
        G.add_node("ROOT", type="system")
        nx.write_graphml(G, LITERATURE_GRAPH_PATH)

    # Init GraphML for memory
    if not os.path.exists(MEMORY_GRAPH_PATH):
        G = nx.DiGraph()
        G.add_node("MEMORY_ROOT", type="system")
        nx.write_graphml(G, MEMORY_GRAPH_PATH)

_embedding_model = None
def init_model():
    global _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        _embedding_model = None

init_knowledge_db()

def update_literature_graph(domain: str, filename: str, doc_id: str, chunk_id: str, content_preview: str):
    G = nx.read_graphml(LITERATURE_GRAPH_PATH)
    
    if not G.has_node(domain):
        G.add_node(domain, type="domain")
        G.add_edge("ROOT", domain)
        
    doc_node_id = f"doc_{filename}"
    if not G.has_node(doc_node_id):
        G.add_node(doc_node_id, type="document", label=filename)
        G.add_edge(domain, doc_node_id)
        
    G.add_node(chunk_id, type="chunk", label=content_preview)
    G.add_edge(doc_node_id, chunk_id)
    
    nx.write_graphml(G, LITERATURE_GRAPH_PATH)

def update_memory_graph(session: str, topic: str, memory_id: str, content_preview: str):
    G = nx.read_graphml(MEMORY_GRAPH_PATH)
    
    session_node_id = f"session_{session}"
    if not G.has_node(session_node_id):
        G.add_node(session_node_id, type="session", label=session)
        G.add_edge("MEMORY_ROOT", session_node_id)
        
    topic_node_id = f"topic_{session}_{topic}"
    if not G.has_node(topic_node_id):
        G.add_node(topic_node_id, type="topic", label=topic)
        G.add_edge(session_node_id, topic_node_id)
        
    G.add_node(memory_id, type="memory", label=content_preview)
    G.add_edge(topic_node_id, memory_id)
    
    nx.write_graphml(G, MEMORY_GRAPH_PATH)

from fastapi.responses import StreamingResponse

@app.post("/api/knowledgestackfrontend/upload")
async def upload_knowledge(files: list[UploadFile] = File(...), domain: str = Form(...)):
    if _embedding_model is None: init_model()
    if _embedding_model is None: return JSONResponse({"error": "Embedding model not available"}, status_code=500)
    
    files_data = []
    for file in files:
        content_bytes = await file.read()
        files_data.append((file.filename, content_bytes))
        
    async def process_and_stream():
        for filename, content_bytes in files_data:
            if not filename: continue
            content = ""
            ext = filename.split('.')[-1].lower()
            if ext == 'txt':
                content = content_bytes.decode('utf-8', errors='ignore')
            elif ext == 'pdf':
                doc = fitz.open(stream=content_bytes, filetype="pdf")
                for page in doc:
                    txt = page.get_text()
                    if isinstance(txt, str):
                        content += txt + "\n"
            
            if not content.strip(): continue
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            chunks = []
            current_chunk = ""
            for p in paragraphs:
                if len(current_chunk) + len(p) > 500:
                    if current_chunk: chunks.append(current_chunk)
                    current_chunk = p
                else:
                    current_chunk += "\n\n" + p if current_chunk else p
            if current_chunk: chunks.append(current_chunk)
            
            doc_id = str(uuid.uuid4())
            for i, chunk in enumerate(chunks):
                if _embedding_model:
                    emb = _embedding_model.encode([chunk])[0]
                    emb_blob = emb.astype(np.float32).tobytes()
                    chunk_id = str(uuid.uuid4())
                    
                    conn = sqlite3.connect(LITERATURE_DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO documents (id, filename, domain, content, embedding) VALUES (?, ?, ?, ?, ?)",
                                  (chunk_id, filename, domain, chunk, emb_blob))
                    conn.commit()
                    conn.close()
                    
                    update_literature_graph(domain, filename, doc_id, chunk_id, chunk[:30] + "...")
                    yield json.dumps({"progress": int(((i+1)/len(chunks))*100)}) + "\n"
        yield json.dumps({"progress": 100, "status": "success"}) + "\n"

    return StreamingResponse(process_and_stream(), media_type="application/x-ndjson")

@app.get("/api/knowledgestackfrontend/search")
def search_literature(q: str):
    if not q: return {"results": []}
    if _embedding_model is None: init_model()
    if _embedding_model is None: return {"results": [], "error": "Embedding model not available"}
    
    query_emb = _embedding_model.encode([q])[0]
    conn = sqlite3.connect(LITERATURE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, domain, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        emb = np.frombuffer(row[4], dtype=np.float32)
        sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        if sim > 0.3:
            results.append({"id": row[0], "filename": row[1], "domain": row[2], "content": row[3], "similarity": float(sim)})
            
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return {"results": results[:10]}

@app.post("/api/knowledgestackfrontend/memory/upload")
async def upload_memory(session: str = Form(...), topic: str = Form(...), content: str = Form(...)):
    if _embedding_model is None: init_model()
    if _embedding_model is None: return {"status": "error", "message": "Embedding model not available"}
        
    chunks = [content[i:i+500] for i in range(0, len(content), 500)]
    for chunk in chunks:
        emb = _embedding_model.encode([chunk])[0]
        emb_blob = emb.astype(np.float32).tobytes()
        memory_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO memories (id, session, topic, content, embedding) VALUES (?, ?, ?, ?, ?)",
                      (memory_id, session, topic, chunk, emb_blob))
        conn.commit()
        conn.close()
        update_memory_graph(session, topic, memory_id, chunk[:30] + "...")
    return {"status": "success"}

@app.get("/api/knowledgestackfrontend/memory/search")
def search_memory(q: str):
    if not q: return {"results": []}
    if _embedding_model is None: init_model()
    if _embedding_model is None: return {"results": [], "error": "Embedding model not available"}
    
    query_emb = _embedding_model.encode([q])[0]
    conn = sqlite3.connect(MEMORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, session, topic, content, embedding, timestamp FROM memories")
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        emb = np.frombuffer(row[4], dtype=np.float32)
        sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        if sim > 0.3:
            results.append({"id": row[0], "session": row[1], "topic": row[2], "content": row[3], "timestamp": row[5], "similarity": float(sim)})
            
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return {"results": results[:10]}

@app.get("/api/knowledgestackfrontend/graph")
def get_literature_graph():
    if not os.path.exists(LITERATURE_GRAPH_PATH): return []
    try:
        G = nx.read_graphml(LITERATURE_GRAPH_PATH)
        elements = []
        for n, d in G.nodes(data=True):
            elements.append({"data": {"id": n, "label": d.get("label", n), "type": d.get("type", "unknown")}})
        for u, v in G.edges():
            elements.append({"data": {"source": u, "target": v}})
        return elements
    except Exception:
        return []

@app.get("/api/knowledgestackfrontend/memory/graph")
def get_memory_graph():
    if not os.path.exists(MEMORY_GRAPH_PATH): return []
    try:
        G = nx.read_graphml(MEMORY_GRAPH_PATH)
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

def perform_pyrefly_integrity_check():
    """
    High-Integrity Static Check: Verify sidecar_ui.py using pyrefly.
    Exits if any errors or warnings are detected to prevent unsafe execution.
    """
    import subprocess
    import shutil
    
    pyrefly_cmd = shutil.which("pyrefly")
    if not pyrefly_cmd:
        # If pyrefly is missing, we consider it a safety violation in this mode
        print("[!] Safety Violation: pyrefly tool not found in PATH.")
        sys.exit(1)
        
    print(f"[*] Running Pyrefly Integrity Check on {os.path.basename(__file__)}...")
    try:
        # Setup environment to include project's site-packages for import resolution
        env = os.environ.copy()
        venv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyvenv")
        # Auto-detect lib/python3.x/site-packages
        site_pkgs = None
        lib_dir = os.path.join(venv_path, "lib")
        if os.path.exists(lib_dir):
            for entry in os.listdir(lib_dir):
                if entry.startswith("python"):
                    potential_path = os.path.join(lib_dir, entry, "site-packages")
                    if os.path.exists(potential_path):
                        site_pkgs = potential_path
                        break
        
        if os.name == 'nt':
            site_pkgs = os.path.join(venv_path, "Lib", "site-packages")
            
        if site_pkgs and os.path.exists(site_pkgs):
            env["PYTHONPATH"] = site_pkgs + os.pathsep + env.get("PYTHONPATH", "")

        # Run pyrefly check.
        result = subprocess.run([pyrefly_cmd, "check", __file__], 
                                capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print("[!] Pyrefly Integrity Check FAILED.")
            print(result.stdout)
            print(result.stderr)
            print("[*] Emergency Shutdown: Integrity violations detected.")
            sys.exit(1)
        print("[+] Pyrefly Integrity Check PASSED.")
    except Exception as e:
        print(f"[!] Error executing Pyrefly: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Perform mandatory safety check before starting any services
    perform_pyrefly_integrity_check()
    
    ui_port = get_free_port()
    port_file = os.path.join(os.path.dirname(DB_PATH), ".sidecar_port")
    with open(port_file, "w") as f:
        f.write(str(ui_port))
        
    def poll_ada_telemetry():
        while True:
            try:
                t0 = time.perf_counter_ns()
                resp = httpx.get("http://127.0.0.1:11420/api/telemetry", timeout=1.0)
                t1 = time.perf_counter_ns()
                now_ts = time.time()
                
                wcel_us = (t1 - t0) / 1000.0
                engine_stats.wcel = wcel_us
                engine_stats.wcel_history_1m.append({"ts": now_ts, "val": wcel_us})
                
                if resp.status_code == 200:
                    data = resp.json()
                    engine_stats.wcet_elp0 = data.get("WCET_ELP0", engine_stats.wcet_elp0)
                    engine_stats.wcet_elp0_hist.append({"ts": now_ts, "val": engine_stats.wcet_elp0})
                    
                    engine_stats.wcet_elp1 = data.get("WCET_ELP1", engine_stats.wcet_elp1)
                    engine_stats.wcet_elp1_hist.append({"ts": now_ts, "val": engine_stats.wcet_elp1})
                    
                    engine_stats.wcet_elp2 = data.get("WCET_ELP2", engine_stats.wcet_elp2)
                    engine_stats.wcet_elp2_hist.append({"ts": now_ts, "val": engine_stats.wcet_elp2})
                    
                    engine_stats.wcet_elp3 = data.get("WCET_ELP3", engine_stats.wcet_elp3)
                    
                    engine_stats.jitter_avg_us = data.get("Jitter_Avg_uS", engine_stats.jitter_avg_us)
                    engine_stats.jitter_max_us = data.get("Jitter_Max_uS", engine_stats.jitter_max_us)

                    engine_stats.wcet_watchdog_loop_us = data.get("WCET_WatchdogLoop_uS", engine_stats.wcet_watchdog_loop_us)
                    engine_stats.wcet_wtdog_hist.append({"ts": now_ts, "val": engine_stats.wcet_watchdog_loop_us})

                    engine_stats.wcet_main_loop_us = data.get("WCET_mainLoop_uS", engine_stats.wcet_main_loop_us)
                    engine_stats.wcet_mloop_hist.append({"ts": now_ts, "val": engine_stats.wcet_main_loop_us})
                    
            except Exception:
                pass
            time.sleep(1)
            
    threading.Thread(target=poll_ada_telemetry, daemon=True).start()

    def run_benchmark():
        time.sleep(2)  # Allow server to fully start
        try:
            httpx.post(f"http://127.0.0.1:{ui_port}/api/chat", json={"message": "test"}, timeout=30.0)
        except Exception:
            pass
            
    threading.Thread(target=run_benchmark, daemon=True).start()

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
