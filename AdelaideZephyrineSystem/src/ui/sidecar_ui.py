import asyncio
import gc
import json
import os
import socket
import sqlite3
import sys
import threading
import time
import uuid
from typing import Optional

import httpx
import networkx as nx
import numpy as np
import psutil
import tiktoken
import uvicorn
import webview
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# Global Performance Tuning: Disable Garbage Collection
gc.disable()

# Configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(base_dir, "data/NetworkMemoryPool", "assistant_session.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# ── Crypto ────────────────────────────────────────────────────────────────
# Load the AdaLang encryption module for field-level AES-256-GCM.
# Sub-keys are derived from the master key (set by run.py as env var).

base_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(base_dir)

# Add python/ to path so adelaide_crypto is importable by static checkers
# and at runtime (it lives in python/, not ui/)
_python_dir = os.path.join(base_dir, "python")
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

from adelaide_crypto import (  # noqa: E402
    CTX_ASSISTANT,
    CTX_LITERATURE,
    CTX_MEMORY_INDEX,
    decrypt_field,
    derive_sub_key,
    encrypt_field,
    is_field_encrypted,
    load_master_key,
)

# Derive sub-keys at module load time
_crypto_available = False
_assistant_sub_key = None
_memory_index_sub_key = None
_literature_sub_key = None

try:
    _master_key = load_master_key()
    _assistant_sub_key = derive_sub_key(_master_key, CTX_ASSISTANT)
    _memory_index_sub_key = derive_sub_key(_master_key, CTX_MEMORY_INDEX)
    _literature_sub_key = derive_sub_key(_master_key, CTX_LITERATURE)
    _crypto_available = True
except Exception as _exc:
    print(f"[CRYPTO] FATAL: Crypto initialization failed ({_exc}). "
          "Aborting — refusing to run with plaintext storage.")
    os.abort()


def _enc(val: str, sub_key=None) -> str:
    """Encrypt a field value. Returns encrypted hex blob."""
    if not _crypto_available or not val:
        return val
    # Skip if already encrypted
    if is_field_encrypted(str(val)):
        return val
    if sub_key is None:
        return val
    return encrypt_field(sub_key, val)


def _cc(val: str, sub_key) -> str:
    """Conditional encrypt: plaintext → hex blob (or pass-through)."""
    if not _crypto_available or not val or is_field_encrypted(str(val)):
        return val
    return encrypt_field(sub_key, val)


def _dc(val: str, sub_key) -> str:
    """Conditional decrypt: hex blob → plaintext (or pass-through)."""
    if not _crypto_available or not val or not is_field_encrypted(str(val)):
        return val
    return decrypt_field(sub_key, val)


# Zephyrine Engine Settings - Configuration dictionary for engine settings
class EngineSettings:
    def __init__(self):  # nosec
        # Load existing settings from DB or use defaults
        # nosec - recursive function with implicit base case
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create settings table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS zephyrine_settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        # Default settings
        defaults = {
            "model_name": "Snowball-Enaga",
            "context_window": 32000,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.6,
            "frequency_penalty": 0.6,
            "n_predict": -1,  # -1 = unlimited
            "streaming": True,
            "system_prompt": "You are Zephyrine, an intelligent AI assistant.",
            "enable_knowledge_search": True,
            "enable_memory_search": True,
            "max_concurrent_requests": 4,
            "keep_alive_seconds": 3600,
            "verbose": False,
        }

        # Insert defaults if table is empty
        cursor.execute("SELECT COUNT(*) FROM zephyrine_settings")
        if cursor.fetchone()[0] == 0:
            for key, value in defaults.items():
                cursor.execute(
                    "INSERT INTO zephyrine_settings (key, value) VALUES (?, ?)",
                    (key, json.dumps(value)),
                )

        conn.commit()
        conn.close()


engine_settings = EngineSettings()


def get_engine_settings():
    """Get all engine settings as a dictionary"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM zephyrine_settings")
    settings = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}
    conn.close()
    return settings


def save_engine_setting(key: str, value):
    """Save a single engine setting"""
    try:
        # Validate type
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(
                f"Invalid value type for {key}: must be str, int, float, or bool"
            )

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO zephyrine_settings (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving setting {key}: {e}")
        return False


def delete_engine_setting(key: str):
    """Delete an engine setting"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM zephyrine_settings WHERE key = ?", (key,))
    conn.commit()
    conn.close()


app = FastAPI()


class EngineStats:
    def __init__(self):  # nosec
        # nosec - recursive function with implicit base case
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
        self.handless_stage = "Idle"
        self.handless_wcet = 0.0
        self.handless_input_text = ""
        self.handless_output_text = ""

        self.history_1m = []
        self.wcel_history_1m = []

        # Histories for deltas
        self.wcet_elp0_hist = []
        self.wcet_elp1_hist = []
        self.wcet_elp2_hist = []
        self.wcet_wtdog_hist = []
        self.wcet_mloop_hist = []

        self.context_faults = 0
        self.virtual_ctx_len = 0


engine_stats = EngineStats()

try:
    enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    enc = None

# Configuration
ADA_BACKEND_URL = "http://localhost:11420"
DIST_DIR = os.path.join(os.path.dirname(__file__), "frontend", "dist")

# ── API Key for Ada backend ──────────────────────────────────────────────────
# Injected by run.py via ADELAIDE_SIDECAR_API_KEY when enforcement is enabled.
# If the env var is not set, we send a default placeholder key so the Ada
# server's non-enforcement mode (which accepts any non-empty key) still works.
_ADELAIDE_API_KEY = os.environ.get("ADELAIDE_SIDECAR_API_KEY", "")
if not _ADELAIDE_API_KEY:
    _ADELAIDE_API_KEY = "adelaide-sidecar-default-key"


def _ada_headers(extra: dict | None = None) -> dict:
    """Return base headers for Ada backend requests, including x-api-key."""
    h = {"User-Agent": "Zephy-Sidecar-UI/1.0", "x-api-key": _ADELAIDE_API_KEY}
    if extra:
        h.update(extra)
    return h


# Initialize SQLite Database
def init_db():  # nosec
    # nosec - recursive function with implicit base case
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Check for sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Check if session_id column exists in messages
    cursor.execute("PRAGMA table_info(messages)")
    columns = [col[1] for col in cursor.fetchall()]
    if "session_id" not in columns:
        print("Migrating database to support sessions...")
        # Create a default session for legacy messages
        cursor.execute("INSERT INTO sessions (title) VALUES (?)", ("Legacy Session",))
        default_session_id = cursor.lastrowid
        # Add column
        cursor.execute("ALTER TABLE messages ADD COLUMN session_id INTEGER")
        # Update existing messages
        cursor.execute("UPDATE messages SET session_id = ?", (default_session_id,))

    conn.commit()
    conn.close()


init_db()

# ── Auto-migration: encrypt any existing unencrypted data in assistant_session.db ──
if _crypto_available:
    try:
        import sqlite3 as _sqlite3

        with _sqlite3.connect(DB_PATH) as _conn:
            _cur = _conn.cursor()
            # Check messages table
            _cur.execute(
                "SELECT COUNT(*) FROM messages WHERE content IS NOT NULL AND content != ''"
            )
            _total = _cur.fetchone()[0]
            if _total > 0:
                _cur.execute(
                    "SELECT rowid, content FROM messages WHERE content IS NOT NULL AND content != ''"
                )
                _migrated = 0
                for _row in _cur.fetchall():
                    if not is_field_encrypted(str(_row[1])):
                        _enc = (
                            _row[1]
                            if _assistant_sub_key is None
                            else encrypt_field(_assistant_sub_key, _row[1])
                        )
                        _cur.execute(
                            "UPDATE messages SET content = ? WHERE rowid = ?",
                            (_enc, _row[0]),
                        )
                        _migrated += 1
                if _migrated > 0:
                    _conn.commit()
                    print(
                        f"[CRYPTO] assistant_session: migrated {_migrated}/{_total} message rows to encrypted"
                    )
    except Exception as _e:
        print(f"[CRYPTO] WARNING: Could not migrate assistant_session.db: {_e}")


@app.post("/api/telemetry")
async def post_telemetry(req: Request):
    data = await req.json()
    now_ts = time.time()

    if "WCET_mainLoop_nS" in data:
        val = float(data["WCET_mainLoop_nS"])
        engine_stats.wcet_main_loop_us = (
            val  # kept as 'us' in field name but now holds nS
        )
        engine_stats.wcet_mloop_hist.append({"ts": now_ts, "val": val})

    if "WCET_ELP0_nS" in data:
        val = float(data["WCET_ELP0_nS"])
        engine_stats.wcet_elp0 = val
        engine_stats.wcet_elp0_hist.append({"ts": now_ts, "val": val})

    if "WCET_ELP1_nS" in data:
        val = float(data["WCET_ELP1_nS"])
        engine_stats.wcet_elp1 = val
        engine_stats.wcet_elp1_hist.append({"ts": now_ts, "val": val})

    if "WCET_ELP2_nS" in data:
        val = float(data["WCET_ELP2_nS"])
        engine_stats.wcet_elp2 = val
        engine_stats.wcet_elp2_hist.append({"ts": now_ts, "val": val})

    if "WCET_ELP3_nS" in data:
        val = float(data["WCET_ELP3_nS"])
        engine_stats.wcet_elp3 = val
        # For ELP3 (1ms paced), we don't store 1000pts/s in history

    if "Context_Faults" in data:
        engine_stats.context_faults = int(data["Context_Faults"])

    if "Virtual_Ctx_Len" in data:
        engine_stats.virtual_ctx_len = int(data["Virtual_Ctx_Len"])

    if "Jitter_Avg_nS" in data:
        engine_stats.jitter_avg_us = float(data["Jitter_Avg_nS"])
    if "Jitter_Max_nS" in data:
        engine_stats.jitter_max_us = float(data["Jitter_Max_nS"])

    return JSONResponse({"status": "ok"})


@app.get("/api/sessions")
def get_sessions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, created_at FROM sessions ORDER BY created_at DESC"
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]


@app.post("/api/sessions")
async def create_session(request: Request):
    data = await request.json()
    title = data.get("title", "New Session")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sessions (title) VALUES (?)", (title,))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"id": session_id, "title": title}


@app.put("/api/sessions/{session_id}")
async def rename_session(session_id: int, request: Request):
    data = await request.json()
    title = data.get("title", "")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.post("/api/sessions/{session_id}/duplicate")
def duplicate_session(session_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    if not row:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    new_title = row[0] + " (Copy)"
    cursor.execute("INSERT INTO sessions (title) VALUES (?)", (new_title,))
    new_session_id = cursor.lastrowid

    # Copy messages (re-encrypt content for the new session)
    cursor.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    )
    messages = cursor.fetchall()
    for m in messages:
        # Decrypt then re-encrypt (ensures consistent encryption for new session)
        plain = _dc(m[1], _assistant_sub_key)
        cursor.execute(
            "INSERT INTO messages (role, content, session_id) VALUES (?, ?, ?)",
            (m[0], _cc(plain, _assistant_sub_key), new_session_id),
        )

    conn.commit()
    conn.close()
    return {"id": new_session_id, "title": new_title}


@app.get("/api/messages")
def get_messages(session_id: Optional[int] = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if session_id:
        cursor.execute(
            "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
    else:
        cursor.execute("SELECT role, content, timestamp FROM messages ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [
        {"role": r[0], "content": _dc(r[1], _assistant_sub_key), "timestamp": r[2]}
        for r in rows
    ]


@app.get("/api/adelaideenginestats")
def get_stats(queue_len: int = 0):
    now = time.time()
    uptime = now - engine_stats.boot_time

    # Cleanup history older than 60 seconds
    engine_stats.history_1m = [
        h for h in engine_stats.history_1m if now - h["ts"] <= 60
    ]
    engine_stats.wcel_history_1m = [
        h for h in engine_stats.wcel_history_1m if now - h["ts"] <= 60
    ]
    engine_stats.wcet_elp0_hist = [
        h for h in engine_stats.wcet_elp0_hist if now - h["ts"] <= 60
    ]
    engine_stats.wcet_elp1_hist = [
        h for h in engine_stats.wcet_elp1_hist if now - h["ts"] <= 60
    ]
    engine_stats.wcet_elp2_hist = [
        h for h in engine_stats.wcet_elp2_hist if now - h["ts"] <= 60
    ]
    engine_stats.wcet_wtdog_hist = [
        h for h in engine_stats.wcet_wtdog_hist if now - h["ts"] <= 60
    ]
    engine_stats.wcet_mloop_hist = [
        h for h in engine_stats.wcet_mloop_hist if now - h["ts"] <= 60
    ]

    avg_1m_wcel = (
        sum(h["val"] for h in engine_stats.wcel_history_1m)
        / len(engine_stats.wcel_history_1m)
        if engine_stats.wcel_history_1m
        else engine_stats.wcel
    )

    def get_delta(hist):
        if not hist:
            return 0.0
        vals = [h["val"] for h in hist]
        return max(vals) - min(vals)

    current_process = psutil.Process()
    return {
        "WCET_ELP0_nS": engine_stats.wcet_elp0,
        "WCET_ELP0_nS_delta": get_delta(engine_stats.wcet_elp0_hist),
        "WCET_ELP1_nS": engine_stats.wcet_elp1,
        "WCET_ELP1_nS_delta": get_delta(engine_stats.wcet_elp1_hist),
        "WCET_ELP2_nS": engine_stats.wcet_elp2,
        "WCET_ELP2_nS_delta": get_delta(engine_stats.wcet_elp2_hist),
        "WCET_ELP3_nS": engine_stats.wcet_elp3,
        "Jitter_Avg_nS": engine_stats.jitter_avg_us,
        "Jitter_Max_nS": engine_stats.jitter_max_us,
        "WCEL": engine_stats.wcel,
        "WCEL_delta_1m": avg_1m_wcel,
        "WCET_mainLoop_nS": engine_stats.wcet_main_loop_us,
        "WCET_mainLoop_nS_delta": get_delta(engine_stats.wcet_mloop_hist),
        "MemoryConsumption_MB": current_process.memory_info().rss / (1024 * 1024),
        "CPU_Consumption": current_process.cpu_percent(interval=None),
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
        "WCET_mLoop_Hist": engine_stats.wcet_mloop_hist,
        "Context_Faults": engine_stats.context_faults,
        "Virtual_Ctx_Len": engine_stats.virtual_ctx_len,
        "Handless_Stage": engine_stats.handless_stage,
        "Handless_WCET_nS": engine_stats.handless_wcet,
        "Handless_Input_Text": engine_stats.handless_input_text,
        "Handless_Output_Text": engine_stats.handless_output_text,
    }


async def _auto_extract_memory(session_id: str, user_msg: str, assistant_msg: str):
    prompt = f'Extract the core topic and a concise memory summary from this interaction.\nUser: {user_msg}\nAssistant: {assistant_msg}\n\nRespond ONLY with a valid JSON object in this format: {{"topic": "Short Topic Name", "memory": "Concise memory text"}}'
    payload = {
        "model": "Snowball-Enaga",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(headers=_ada_headers()) as client:
            response = await client.post(
                f"{ADA_BACKEND_URL}/api/chat", json=payload, timeout=60.0
            )
            if response.status_code == 200:
                import re

                resp_json = response.json()
                content = resp_json.get("message", {}).get("content", "")
                if not content:
                    content = resp_json.get("response", "")

                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    topic = data.get("topic", "Extracted Topic")
                    memory_text = data.get("memory", "")

                    if memory_text:
                        if _embedding_model is None:
                            init_model()
                        if _embedding_model is not None:
                            chunks = [
                                memory_text[i : i + 500]
                                for i in range(0, len(memory_text), 500)
                            ]
                            for chunk in chunks:
                                emb = _embedding_model.encode([chunk])[0]
                                emb_blob = emb.astype(np.float32).tobytes()
                                memory_id = str(uuid.uuid4())

                                conn = sqlite3.connect(MEMORY_DB_PATH)
                                cursor = conn.cursor()
                                cursor.execute(
                                    "INSERT INTO memories (id, session, topic, content, embedding) VALUES (?, ?, ?, ?, ?)",
                                    (
                                        memory_id,
                                        session_id,
                                        topic,
                                        _cc(chunk, _memory_index_sub_key),
                                        emb_blob,
                                    ),
                                )
                                conn.commit()
                                conn.close()
                                update_memory_graph(
                                    session_id, topic, memory_id, chunk[:30] + "..."
                                )
    except Exception as e:
        print(f"Auto-extract memory failed: {e}")


@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    session_id = data.get("session_id")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if not session_id:
        title = user_message[:20] + "..." if len(user_message) > 20 else user_message
        cursor.execute("INSERT INTO sessions (title) VALUES (?)", (title,))
        session_id = cursor.lastrowid

    # Save User message to DB (encrypt content)
    cursor.execute(
        "INSERT INTO messages (role, content, session_id) VALUES (?, ?, ?)",
        ("user", _cc(user_message, _assistant_sub_key), session_id),
    )
    conn.commit()
    conn.close()

    async def event_generator():
        payload = {
            "model": "Snowball-Enaga",
            "messages": [{"role": "user", "content": user_message}],
            "stream": True,
        }

        full_reply = ""
        session_id_local = session_id
        retry_delay = 1.0  # starts at 1s, caps at 30s

        while True:
            try:
                async with httpx.AsyncClient(headers=_ada_headers()) as client:
                    async with client.stream(
                        "POST",
                        f"{ADA_BACKEND_URL}/api/chat",
                        json=payload,
                        timeout=600.0,
                    ) as response:
                        if response.status_code != 200:
                            yield (
                                json.dumps(
                                    {
                                        "error": f"Backend returned error: {response.status_code}, retrying..."
                                    }
                                )
                                + "\n"
                            )
                            await asyncio.sleep(retry_delay)
                            retry_delay = min(retry_delay * 2, 30.0)
                            continue

                        retry_delay = 1.0  # reset on successful connection

                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            try:
                                resp_json = json.loads(line)
                                # Ada returns Ollama format: {"message": {"content": "..."}, "done": false}
                                if "message" in resp_json:
                                    chunk = resp_json["message"].get("content", "")
                                    full_reply += chunk
                                    yield line + "\n"
                                elif "response" in resp_json:  # /api/generate format
                                    chunk = resp_json.get("response", "")
                                    full_reply += chunk
                                    yield line + "\n"

                                if resp_json.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue

                # If we get here, the stream completed successfully
                break

            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            ):
                # Ada backend unreachable or dropped connection — retry indefinitely
                yield (
                    json.dumps(
                        {
                            "message": {
                                "content": f"[Waiting for Ada backend...] (retry in {retry_delay:.0f}s)"
                            },
                            "done": False,
                        }
                    )
                    + "\n"
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)
                continue

            except Exception as e:
                yield (
                    json.dumps(
                        {
                            "message": {
                                "content": f"[Ada backend error: {str(e)}, retrying...]"
                            },
                            "done": False,
                        }
                    )
                    + "\n"
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)
                continue

        # Finalize and save to DB (encrypt content)
        conn_final = sqlite3.connect(DB_PATH)
        cursor_final = conn_final.cursor()
        cursor_final.execute(
            "INSERT INTO messages (role, content, session_id) VALUES (?, ?, ?)",
            ("assistant", _cc(full_reply, _assistant_sub_key), session_id_local),
        )
        conn_final.commit()
        conn_final.close()

        # Fire and forget background memory extraction
        asyncio.create_task(
            _auto_extract_memory(str(session_id_local), user_message, full_reply)
        )

        # Send a final chunk with the session_id so the frontend can update
        yield json.dumps({"session_id": session_id_local, "done": True}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/api/regenerate")
async def regenerate(request: Request):
    """Regenerate the last assistant response in a session.
    Optionally accepts a new user message to replace the last user message before regenerating.
    """
    data = await request.json()
    session_id = data.get("session_id")
    new_message = data.get(
        "message"
    )  # Optional: if provided, replaces last user message

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if not session_id:
        # Create a new session
        title = (new_message or "Regenerated Chat")[:20]
        cursor.execute("INSERT INTO sessions (title) VALUES (?)", (title,))
        session_id = cursor.lastrowid
        conn.commit()

    # Get all messages for this session in order
    cursor.execute(
        "SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    )
    rows = cursor.fetchall()

    # Decrypt all content in-place
    rows = [(row[0], row[1], _dc(row[2], _assistant_sub_key)) for row in rows]

    if not rows:
        conn.close()
        return JSONResponse({"error": "No messages in session"}, status_code=404)

    # If new_message provided, update the last user message
    if new_message:
        # Find last user message and update it (encrypt content)
        for msg_id, role, content in reversed(rows):
            if role == "user":
                cursor.execute(
                    "UPDATE messages SET content = ? WHERE id = ?",
                    (_cc(new_message, _assistant_sub_key), msg_id),
                )
                # Delete all messages after this user message (assistant responses)
                cursor.execute(
                    "DELETE FROM messages WHERE id > ? AND session_id = ?",
                    (msg_id, session_id),
                )
                conn.commit()
                break
        # Re-fetch messages after update
        cursor.execute(
            "SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        rows = cursor.fetchall()
        # Decrypt re-fetched content
        rows = [(row[0], row[1], _dc(row[2], _assistant_sub_key)) for row in rows]
    else:
        # Just delete the last assistant response if the last message is from assistant
        last_role = rows[-1][1] if rows else None
        if last_role == "assistant":
            cursor.execute("DELETE FROM messages WHERE id = ?", (rows[-1][0],))
            conn.commit()
            rows = rows[:-1]

    conn.close()

    # Find the last user message to send to Ada (Ada expects single-message prompts)
    last_user_msg = ""
    for _, role, content in reversed(rows):
        if role == "user":
            last_user_msg = content
            break

    async def event_generator():
        payload = {
            "model": "Snowball-Enaga",
            "messages": [{"role": "user", "content": last_user_msg}],
            "stream": True,
        }

        full_reply = ""
        retry_delay = 1.0  # starts at 1s, caps at 30s

        while True:
            try:
                async with httpx.AsyncClient(headers=_ada_headers()) as client:
                    async with client.stream(
                        "POST",
                        f"{ADA_BACKEND_URL}/api/chat",
                        json=payload,
                        timeout=600.0,
                    ) as response:
                        if response.status_code != 200:
                            yield (
                                json.dumps(
                                    {
                                        "error": f"Backend returned error: {response.status_code}, retrying..."
                                    }
                                )
                                + "\n"
                            )
                            await asyncio.sleep(retry_delay)
                            retry_delay = min(retry_delay * 2, 30.0)
                            continue

                        retry_delay = 1.0  # reset on successful connection

                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            try:
                                resp_json = json.loads(line)
                                if "message" in resp_json:
                                    chunk = resp_json["message"].get("content", "")
                                    full_reply += chunk
                                    yield line + "\n"
                                elif "response" in resp_json:
                                    chunk = resp_json.get("response", "")
                                    full_reply += chunk
                                    yield line + "\n"

                                if resp_json.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue

                break  # stream completed

            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            ):
                yield (
                    json.dumps(
                        {
                            "message": {
                                "content": f"[Waiting for Ada backend...] (retry in {retry_delay:.0f}s)"
                            },
                            "done": False,
                        }
                    )
                    + "\n"
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)
                continue

            except Exception as e:
                yield (
                    json.dumps(
                        {
                            "message": {
                                "content": f"[Ada backend error: {str(e)}, retrying...]"
                            },
                            "done": False,
                        }
                    )
                    + "\n"
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)
                continue

        # Save assistant reply to DB (encrypt content)
        if full_reply:
            conn_final = sqlite3.connect(DB_PATH)
            cursor_final = conn_final.cursor()
            cursor_final.execute(
                "INSERT INTO messages (role, content, session_id) VALUES (?, ?, ?)",
                ("assistant", _cc(full_reply, _assistant_sub_key), session_id),
            )
            conn_final.commit()
            conn_final.close()

            # Fire and forget background memory extraction
            asyncio.create_task(
                _auto_extract_memory(str(session_id), last_user_msg, full_reply)
            )

        yield json.dumps({"session_id": session_id, "done": True}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/api/exit")
def exit_app():
    import threading

    def kill_process():
        try:
            with open(
                os.path.join(os.path.dirname(DB_PATH), ".intentional_exit"), "w"
            ) as f:
                f.write("1")
        except Exception:
            pass
        os._exit(0)

    # Run in a separate thread to allow the HTTP response to return
    threading.Timer(0.5, kill_process).start()
    return {"status": "exiting"}


@app.post("/api/detach_webview")
def detach_webview():
    import threading
    import webbrowser

    import webview

    def close_window_and_open_browser():
        port_file = os.path.join(os.path.dirname(DB_PATH), ".sidecar_port")
        with open(port_file, "r") as f:
            port = f.read().strip()
        webbrowser.open_new(f"http://127.0.0.1:{port}/")

        if webview.windows:
            webview.windows[0].destroy()

    threading.Timer(0.5, close_window_and_open_browser).start()
    return {"status": "detaching"}


@app.get("/api/docs/readme")
def get_readme():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    readme_path = os.path.join(root_dir, "README.md")
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/docs/license")
def get_license():
    license_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "license.md"
    )
    try:
        with open(license_path, "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/user_info")
def get_user_info():
    import getpass

    try:
        username = getpass.getuser()
    except Exception:
        username = "User"
    return {"username": username}


# --- Knowledge Stack Backend ---
LITERATURE_DB_PATH = os.path.join(
    base_dir, "data/NetworkMemoryPool", "literatureRefIndex.db"
)
LITERATURE_GRAPH_PATH = os.path.join(
    base_dir, "data/NetworkMemoryPool", "literature.graphml"
)

MEMORY_DB_PATH = os.path.join(base_dir, "data/NetworkMemoryPool", "memoryRefIndex.db")
MEMORY_GRAPH_PATH = os.path.join(base_dir, "data/NetworkMemoryPool", "memory.graphml")

# Ensure dir
os.makedirs(os.path.dirname(LITERATURE_DB_PATH), exist_ok=True)


def init_knowledge_db():
    # Initialize Literature DB
    conn = sqlite3.connect(LITERATURE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT,
            domain TEXT,
            content TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

    # Initialize Memory DB
    conn = sqlite3.connect(MEMORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            session TEXT,
            topic TEXT,
            content TEXT,
            embedding BLOB,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
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

        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        _embedding_model = None


init_knowledge_db()

# ── Auto-migration: encrypt existing unencrypted data in knowledge DBs ────
if _crypto_available:
    try:
        import sqlite3 as _sqlite3

        # Memory index DB: memories.content
        with _sqlite3.connect(MEMORY_DB_PATH) as _conn:
            _cur = _conn.cursor()
            _cur.execute(
                "SELECT rowid, content FROM memories WHERE content IS NOT NULL AND content != ''"
            )
            _migrated = 0
            for _row in _cur.fetchall():
                if not is_field_encrypted(str(_row[1])):
                    _enc = (
                        _row[1]
                        if _memory_index_sub_key is None
                        else encrypt_field(_memory_index_sub_key, _row[1])
                    )
                    _cur.execute(
                        "UPDATE memories SET content = ? WHERE rowid = ?",
                        (_enc, _row[0]),
                    )
                    _migrated += 1
            if _migrated > 0:
                _conn.commit()
                print(
                    f"[CRYPTO] memoryRefIndex: migrated {_migrated} memory rows to encrypted"
                )

        # Literature DB: documents.content (Ada-side encryption, but catch any
        # existing plaintext that was inserted before crypto was enabled)
        with _sqlite3.connect(LITERATURE_DB_PATH) as _conn:
            _cur = _conn.cursor()
            _cur.execute(
                "SELECT rowid, content FROM documents WHERE content IS NOT NULL AND content != ''"
            )
            _migrated = 0
            for _row in _cur.fetchall():
                if not is_field_encrypted(str(_row[1])):
                    _enc = (
                        _row[1]
                        if _literature_sub_key is None
                        else encrypt_field(_literature_sub_key, _row[1])
                    )
                    _cur.execute(
                        "UPDATE documents SET content = ? WHERE rowid = ?",
                        (_enc, _row[0]),
                    )
                    _migrated += 1
            if _migrated > 0:
                _conn.commit()
                print(
                    f"[CRYPTO] literatureRefIndex: migrated {_migrated} document rows to encrypted"
                )
    except Exception as _e:
        print(f"[CRYPTO] WARNING: Could not migrate knowledge databases: {_e}")


def update_literature_graph(
    domain: str, filename: str, doc_id: str, chunk_id: str, content_preview: str
):
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


@app.post("/api/knowledgestackfrontend/upload")
async def upload_knowledge(
    files: list[UploadFile] = File(...), domain: str = Form(...)
):
    if _embedding_model is None:
        init_model()
    if _embedding_model is None:
        return JSONResponse({"error": "Embedding model not available"}, status_code=500)

    files_data = []
    for file in files:
        content_bytes = await file.read()
        files_data.append((file.filename, content_bytes))

    async def process_and_stream():
        for filename, content_bytes in files_data:
            if not filename:
                continue
            content = ""
            ext = filename.split(".")[-1].lower()
            if ext == "txt":
                content = content_bytes.decode("utf-8", errors="ignore")
            elif ext == "pdf" and fitz:
                doc = fitz.open(stream=content_bytes, filetype="pdf")
                for page in doc:
                    txt = page.get_text()
                    if isinstance(txt, str):
                        content += txt + "\n"

            if not content.strip():
                continue
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            chunks = []
            current_chunk = ""
            for p in paragraphs:
                if len(current_chunk) + len(p) > 500:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = p
                else:
                    current_chunk += "\n\n" + p if current_chunk else p
            if current_chunk:
                chunks.append(current_chunk)

            doc_id = str(uuid.uuid4())
            for i, chunk in enumerate(chunks):
                if _embedding_model:
                    emb = _embedding_model.encode([chunk])[0]
                    emb_blob = emb.astype(np.float32).tobytes()
                    chunk_id = str(uuid.uuid4())

                    conn = sqlite3.connect(LITERATURE_DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO documents (id, filename, domain, content, embedding) VALUES (?, ?, ?, ?, ?)",
                        (chunk_id, filename, domain, chunk, emb_blob),
                    )
                    conn.commit()
                    conn.close()

                    update_literature_graph(
                        domain, filename, doc_id, chunk_id, chunk[:30] + "..."
                    )
                    yield (
                        json.dumps({"progress": int(((i + 1) / len(chunks)) * 100)})
                        + "\n"
                    )
        yield json.dumps({"progress": 100, "status": "success"}) + "\n"

    return StreamingResponse(process_and_stream(), media_type="application/x-ndjson")


@app.get("/api/knowledgestackfrontend/search")
def search_literature(q: str):
    if not q:
        return {"results": []}
    if _embedding_model is None:
        init_model()
    if _embedding_model is None:
        return {"results": [], "error": "Embedding model not available"}

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
            results.append(
                {
                    "id": row[0],
                    "filename": row[1],
                    "domain": row[2],
                    "content": _dc(row[3], _literature_sub_key),
                    "similarity": float(sim),
                }
            )

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return {"results": results[:10]}


@app.post("/api/knowledgestackfrontend/memory/upload")
async def upload_memory(
    session: str = Form(...), topic: str = Form(...), content: str = Form(...)
):
    if _embedding_model is None:
        init_model()
    if _embedding_model is None:
        return {"status": "error", "message": "Embedding model not available"}

    chunks = [content[i : i + 500] for i in range(0, len(content), 500)]
    for chunk in chunks:
        emb = _embedding_model.encode([chunk])[0]
        emb_blob = emb.astype(np.float32).tobytes()
        memory_id = str(uuid.uuid4())

        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO memories (id, session, topic, content, embedding) VALUES (?, ?, ?, ?, ?)",
            (memory_id, session, topic, _cc(chunk, _memory_index_sub_key), emb_blob),
        )
        conn.commit()
        conn.close()
        update_memory_graph(session, topic, memory_id, chunk[:30] + "...")
    return {"status": "success"}


@app.get("/api/knowledgestackfrontend/memory/search")
def search_memory(q: str):
    if not q:
        return {"results": []}
    if _embedding_model is None:
        init_model()
    if _embedding_model is None:
        return {"results": [], "error": "Embedding model not available"}

    query_emb = _embedding_model.encode([q])[0]
    conn = sqlite3.connect(MEMORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, session, topic, content, embedding, timestamp FROM memories"
    )
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        emb = np.frombuffer(row[4], dtype=np.float32)
        sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        if sim > 0.3:
            results.append(
                {
                    "id": row[0],
                    "session": row[1],
                    "topic": row[2],
                    "content": _dc(row[3], _memory_index_sub_key),
                    "timestamp": row[5],
                    "similarity": float(sim),
                }
            )

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return {"results": results[:10]}


@app.get("/api/knowledgestackfrontend/graph")
def get_literature_graph():
    if not os.path.exists(LITERATURE_GRAPH_PATH):
        return []
    try:
        G = nx.read_graphml(LITERATURE_GRAPH_PATH)
        elements = []
        for n, d in G.nodes(data=True):
            elements.append(
                {
                    "data": {
                        "id": n,
                        "label": d.get("label", n),
                        "type": d.get("type", "unknown"),
                    }
                }
            )
        for u, v in G.edges():
            elements.append({"data": {"source": u, "target": v}})
        return elements
    except Exception:
        return []


@app.get("/api/knowledgestackfrontend/memory/graph")
def get_memory_graph():
    if not os.path.exists(MEMORY_GRAPH_PATH):
        return []
    try:
        G = nx.read_graphml(MEMORY_GRAPH_PATH)
        elements = []
        for n, d in G.nodes(data=True):
            elements.append(
                {
                    "data": {
                        "id": n,
                        "label": d.get("label", n),
                        "type": d.get("type", "unknown"),
                    }
                }
            )
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


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_server(port):
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def perform_platform_integrity_check():
    """
    High-Integrity Static Check: Verify sidecar_ui.py using pyrefly and ruff.
    Exits if any errors or warnings are detected to prevent unsafe execution.
    """
    import shutil
    import subprocess

    print("[*] Performing Platform Self-Integrity Check...")

    # 1. Pyrefly Check
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

        if os.name == "nt":
            site_pkgs = os.path.join(venv_path, "Lib", "site-packages")

        if site_pkgs and os.path.exists(site_pkgs):
            env["PYTHONPATH"] = site_pkgs + os.pathsep + env.get("PYTHONPATH", "")

        # Add python/ directory so pyrefly can resolve adelaide_crypto etc.
        python_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "python")
        if os.path.exists(python_dir):
            env["PYTHONPATH"] = python_dir + os.pathsep + env.get("PYTHONPATH", "")

        # Run pyrefly check.
        result = subprocess.run(
            [pyrefly_cmd, "check", __file__], capture_output=True, text=True, env=env
        )  # nosec
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

    # 2. Ruff Check
    ruff_cmd = shutil.which("ruff")
    if not ruff_cmd:
        print("[!] Warning: ruff tool not found in PATH. Skipping secondary check.")
    else:
        try:
            print("[*] Running Ruff Quality Check on platform components...")
            # Run ruff check on the AdelaideZephyrineSystem directory
            adelaide_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            result = subprocess.run(
                [ruff_cmd, "check", adelaide_dir, "--exclude", "vendor,moonshine"],
                capture_output=True,
                text=True,
            )  # nosec
            if result.returncode != 0:
                print("[!] Ruff Integrity Check FAILED.")
                print(result.stdout)
                print("[*] Emergency Shutdown: Quality violations detected.")
                sys.exit(1)
            print("[+] Ruff Quality Check PASSED.")
        except Exception as e:
            print(f"[!] Error executing Ruff: {str(e)}")
            sys.exit(1)

    print("[*] Platform integrity verified.")


class SidecarAPI:
    def log_error(
        self, message, source=None, lineno=None, colno=None, error_stack=None
    ):
        try:
            import glob

            logs_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "logs")
            )
            os.makedirs(logs_dir, exist_ok=True)
            log_files = glob.glob(os.path.join(logs_dir, "run_*.log"))
            if log_files:
                latest_log = max(log_files, key=os.path.getmtime)
            else:
                latest_log = os.path.join(logs_dir, "sidecar_errors.log")

            with open(latest_log, "a", encoding="utf-8") as f:
                f.write(f"\n[FRONTEND ERROR] {message}\n")
                if source:
                    f.write(f"  Source: {source} ({lineno}:{colno})\n")
                if error_stack:
                    f.write(f"  Stack: {error_stack}\n")
                f.flush()
        except Exception as e:
            print(f"Failed to log frontend error: {e}")


if __name__ == "__main__":
    # Perform mandatory safety check before starting any services
    perform_platform_integrity_check()

    ui_port = get_free_port()
    port_file = os.path.join(os.path.dirname(DB_PATH), ".sidecar_port")
    with open(port_file, "w") as f:
        f.write(str(ui_port))

    def poll_ada_telemetry():
        while True:
            try:
                t0 = time.perf_counter_ns()
                resp = httpx.get(
                    f"{ADA_BACKEND_URL}/api/telemetry",
                    headers=_ada_headers(),
                    timeout=1.0,
                )
                t1 = time.perf_counter_ns()
                now_ts = time.time()

                wcel_us = (t1 - t0) / 1000.0
                engine_stats.wcel = wcel_us
                engine_stats.wcel_history_1m.append({"ts": now_ts, "val": wcel_us})

                if resp.status_code == 200:
                    data = resp.json()
                    engine_stats.wcet_elp0 = data.get(
                        "WCET_ELP0_nS", engine_stats.wcet_elp0
                    )
                    engine_stats.wcet_elp0_hist.append(
                        {"ts": now_ts, "val": engine_stats.wcet_elp0}
                    )

                    engine_stats.wcet_elp1 = data.get(
                        "WCET_ELP1_nS", engine_stats.wcet_elp1
                    )
                    engine_stats.wcet_elp1_hist.append(
                        {"ts": now_ts, "val": engine_stats.wcet_elp1}
                    )

                    engine_stats.wcet_elp2 = data.get(
                        "WCET_ELP2_nS", engine_stats.wcet_elp2
                    )
                    engine_stats.wcet_elp2_hist.append(
                        {"ts": now_ts, "val": engine_stats.wcet_elp2}
                    )

                    engine_stats.wcet_elp3 = data.get(
                        "WCET_ELP3_nS", engine_stats.wcet_elp3
                    )

                    engine_stats.jitter_avg_us = data.get(
                        "Jitter_Avg_nS", engine_stats.jitter_avg_us
                    )
                    engine_stats.jitter_max_us = data.get(
                        "Jitter_Max_nS", engine_stats.jitter_max_us
                    )

                    engine_stats.wcet_main_loop_us = data.get(
                        "WCET_mainLoop_nS", engine_stats.wcet_main_loop_us
                    )
                    engine_stats.wcet_mloop_hist.append(
                        {"ts": now_ts, "val": engine_stats.wcet_main_loop_us}
                    )

                    engine_stats.handless_stage = data.get(
                        "Handless_Stage", engine_stats.handless_stage
                    )
                    engine_stats.handless_wcet = data.get(
                        "Handless_WCET_nS", engine_stats.handless_wcet
                    )
                    engine_stats.handless_input_text = data.get(
                        "Handless_Input_Text", engine_stats.handless_input_text
                    )
                    engine_stats.handless_output_text = data.get(
                        "Handless_Output_Text", engine_stats.handless_output_text
                    )

            except Exception:
                pass
            time.sleep(1)

    threading.Thread(target=poll_ada_telemetry, daemon=True).start()

    def run_benchmark():
        time.sleep(2)  # Allow server to fully start
        try:
            httpx.post(
                f"http://127.0.0.1:{ui_port}/api/chat",
                json={"message": "test"},
                timeout=30.0,
            )
        except Exception:
            pass

    # threading.Thread(target=run_benchmark, daemon=True).start()

    # Start FastAPI in a background thread
    server_thread = threading.Thread(target=run_server, args=(ui_port,), daemon=True)
    server_thread.start()

    # [DO NOT REMOVE] macOS App Bundle for Microphone/Camera/Screen Capture Permissions
    # On macOS, pywebview needs a proper .app bundle with Info.plist to request
    # microphone, camera, and screen capture permissions. Without this, WebKit
    # will block getUserMedia() calls silently.
    #
    # The Info.plist must contain:
    #   NSMicrophoneUsageDescription - "Adelaide needs microphone access for voice interaction"
    #   NSCameraUsageDescription - "Adelaide needs camera access for visual context"
    #   NSScreenCaptureUsageDescription - "Adelaide needs screen capture for visual context"
    #
    # For development, we use pywebview debug mode which may prompt for permissions.
    # For production, package as .app bundle using py2app or create manually.

    # Launch PyWebview native window
    api = SidecarAPI()

    if os.environ.get("ADELAIDE_SIDECAR_TEST_MODE") == "1":
        def run_automated_test():
            print("[SIDECAR-TEST] Waiting for FastAPI server to start...", flush=True)
            time.sleep(3)
            
            import urllib.request
            import json

            # Test 1: HTTP API Loopback
            try:
                print("[SIDECAR-TEST] Testing /api/chat endpoint...", flush=True)
                req_data = json.dumps({
                    "message": "ping",
                    "session_id": "test_session_123"
                }).encode("utf-8")
                
                req = urllib.request.Request(
                    f"http://127.0.0.1:{ui_port}/api/chat",
                    data=req_data,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": os.environ.get("ADELAIDE_MASTER_KEY", "fallback"),
                        "User-Agent": "Zephy-Sidecar-UI/1.0"
                    },
                    method="POST"
                )
                
                with urllib.request.urlopen(req, timeout=30) as response:
                    status = response.status
                    if status != 200:
                        raise Exception(f"HTTP {status}")
            except Exception as e:
                print(f"[SIDECAR-TEST] FAILED /api/chat! {e}", flush=True)
                os._exit(1)

            # Test 2: JavaScript UI DOM Interaction
            try:
                print("[SIDECAR-TEST] Testing JavaScript DOM Interaction...", flush=True)
                if window:
                    # Click Knowledge Network Nav
                    window.evaluate_js("document.getElementById('nav-knowledge')?.click()")
                    time.sleep(0.5)
                    
                    # Click Chat Nav
                    window.evaluate_js("document.getElementById('nav-chat')?.click()")
                    time.sleep(0.5)
                    
                    # Type and send
                    script = """
                    (function() {
                        try {
                            let input = document.getElementById('chat-input');
                            let btn = document.getElementById('send-btn');
                            if (input && btn) {
                                input.value = 'hello this is an automated UI interaction test';
                                input.dispatchEvent(new Event('input', { bubbles: true }));
                                btn.click();
                                return "SUCCESS";
                            }
                            return "ERROR: UI elements not found";
                        } catch (e) {
                            return "ERROR: " + e.toString();
                        }
                    })();
                    """
                    print("[SIDECAR-TEST] Executing chat script...", flush=True)
                    res = window.evaluate_js(script)
                    print(f"[SIDECAR-TEST] JS Interaction Result: {res}", flush=True)
                    if res and str(res).startswith("ERROR"):
                        os._exit(1)
                        
                    # Wait for typing indicator to appear
                    time.sleep(0.5)
                    
                    # Poll for typing indicator to disappear
                    print("[SIDECAR-TEST] Waiting for assistant response...", flush=True)
                    for _ in range(60):
                        typing = window.evaluate_js("document.querySelector('.typing-indicator') !== null")
                        if not typing:
                            break
                        time.sleep(0.5)
                        
                    # Extract last message content
                    last_msg = window.evaluate_js("var msgs = document.querySelectorAll('.message .message-content'); msgs.length > 0 ? msgs[msgs.length - 1].textContent : 'none'")
                    print(f"[SIDECAR-TEST] Final Response: {last_msg[:100]}...", flush=True)
            except Exception as e:
                print(f"[SIDECAR-TEST] FAILED DOM Interaction! {e}", flush=True)
                os._exit(1)
                
            print("[SIDECAR-TEST] PASSED! Exiting gracefully.", flush=True)
            time.sleep(1)
            os._exit(0)

        import threading
        threading.Thread(target=run_automated_test, daemon=True).start()
        
    window = webview.create_window(
        "Adelaide Zephyrine Assistant",
        f"http://127.0.0.1:{ui_port}",
        width=1000,
        height=800,
        frameless=False,  # Set to True if we want fully custom window frame
        easy_drag=True,
        js_api=api,
    )

    webview.start(debug=False)

    # Wait for the server thread to keep the FastAPI server running after webview detaches
    server_thread.join()
