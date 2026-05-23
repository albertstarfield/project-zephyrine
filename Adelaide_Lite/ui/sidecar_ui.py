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
    readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "stellaicarus", "readme.md")
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
