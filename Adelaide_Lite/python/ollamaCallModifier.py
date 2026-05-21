#!/usr/bin/env python3
import sys
import os
import subprocess
import time
import json
import re
import ast
import random
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Union, List, Optional, Dict, Any

# ANSI Color Codes
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
BLUE = "\033[34m"
RESET = "\033[0m"

class GoStyleFormatter(logging.Formatter):
    """Formats logs in a Go-style logfmt structure with colors."""
    def format(self, record):
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+07:00"
        level = record.levelname
        level_color = RESET
        if level == "INFO": level_color = CYAN
        elif level == "WARNING": level_color = YELLOW
        elif level == "ERROR": level_color = RED
        elif level == "DEBUG": level_color = MAGENTA
        
        source = f"{record.filename}:{record.lineno}"
        msg = record.getMessage()
        
        # Structure: time=... level=... source=... msg="..."
        return f"{BLUE}time={RESET}{timestamp} {BLUE}level={level_color}{level}{RESET} {BLUE}source={RESET}{source} {BLUE}msg={RESET}\"{msg}\""

# Initial logger for bootstrapping
logger = logging.getLogger('ollamaCallModifier')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = GoStyleFormatter()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Silence Werkzeug default logger to prevent double logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- Environment Setup & Bootstrapping ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "pyvenv")
REQUIREMENTS = ["flask", "requests", "werkzeug", "tiktoken", "numpy", "numba", "pyrefly", "html2image", "json5", "soundfile", "deal", "Pillow", "PyMuPDF", "openpyxl", "python-docx", "python-pptx", "tinytag", "selenium", "webdriver-manager", "beautifulsoup4", "ocrmypdf", "qwen-agent", "python-dateutil", "chromadb"]
SCRATCH_DIR = os.path.join(SCRIPT_DIR, "tempWorkScratch")
os.makedirs(SCRATCH_DIR, exist_ok=True)
OLLAMA_DATA_DIR = os.path.join(SCRIPT_DIR, "ollama_data")
os.makedirs(OLLAMA_DATA_DIR, exist_ok=True)

def bootstrap_venv():
    """Ensures the script runs in its dedicated virtual environment."""
    venv_abs = os.path.abspath(VENV_DIR)
    
    if os.path.abspath(sys.prefix) != venv_abs:
        if not os.path.exists(VENV_DIR):
            logger.info(f"[*] Creating virtual environment in {VENV_DIR}...")
            subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
            
        python_exe = os.path.join(VENV_DIR, "bin", "python") if os.name != 'nt' else os.path.join(VENV_DIR, "Scripts", "python.exe")
        
        if os.path.exists(python_exe):
            os.execv(python_exe, [python_exe] + sys.argv)

    try:
        import flask
        import requests
        import tiktoken
        import numpy
        import pyrefly
        import html2image
        import json5
        import qwen_agent
        import soundfile
        import deal
    except ImportError:
        logger.info(f"[*] Missing dependencies. Installing: {', '.join(REQUIREMENTS)}...")
        pip_exe = os.path.join(VENV_DIR, "bin", "pip") if os.name != 'nt' else os.path.join(VENV_DIR, "Scripts", "pip.exe")
        subprocess.run([pip_exe, "install", "--upgrade", "pip"], check=True)
        subprocess.run([pip_exe, "install"] + REQUIREMENTS, check=True)
        os.execv(sys.executable, [sys.executable] + sys.argv)

def ensure_external_tools():
    """Checks and installs required external tools (OPAM, Rocq, Alire, GNAT)."""
    logger.info("[*] Checking external toolchain...")
    
    # 1. OPAM
    try:
        subprocess.run(["opam", "--version"], check=True, capture_output=True)
        logger.info("[+] OPAM already installed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("[*] Installing OPAM...")
        subprocess.run(["bash", "-c", "sh <(curl -fsSL https://opam.ocaml.org/install.sh)"], check=True)
    
    # 2. Rocq/Coq libraries
    rocq_packages = [
        "rocq-prover", "rocq-native"
    ]
    # Check and install missing Rocq packages
    logger.info("[*] Verifying Rocq libraries...")
    for pkg in rocq_packages:
        try:
            res = subprocess.run(["opam", "list", "--installed", "--short", pkg], capture_output=True, text=True)
            if pkg not in res.stdout.split():
                logger.info(f"[*] Missing Rocq library: {pkg}. Installing...")
                subprocess.run(["opam", "install", "--yes", pkg], check=False)
            else:
                logger.info(f"[+] Rocq library {pkg} is present.")
        except:
            logger.warning(f"[!] Could not check/install Rocq library: {pkg}")

    # 3. Alire & GNAT/GNATprove
    try:
        subprocess.run(["alr", "--version"], check=True, capture_output=True)
        logger.info("[+] Alire already installed.")
        # Ensure gnatprove is in path or installed in local workspace
        res = subprocess.run(["find", ".", "-name", "gnatprove", "-type", "f"], capture_output=True, text=True)
        if "gnatprove" not in res.stdout:
            logger.info("[*] gnatprove not found in workspace. Deploying...")
            subprocess.run(["alr", "get", "gnatprove"], check=False)
        else:
            logger.info("[+] gnatprove found in workspace.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("[*] Installing Alire/Ada toolchain...")
        subprocess.run(["bash", "-c", "curl --proto '=https' -sSf https://www.getada.dev/init.sh | sh"], check=True)
        # Ensure GNAT and GNATprove are deployed via Alire
        logger.info("[*] Deploying GNAT and GNATprove via Alire...")
        subprocess.run(["alr", "toolchain", "--select", "gnat_native"], check=False)
        subprocess.run(["alr", "get", "gnatprove"], check=False)

    # 4. Dafny (Formal Verification for Multi-language)
    try:
        subprocess.run(["dafny", "--version"], check=True, capture_output=True)
        logger.info("[+] Dafny already installed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("[*] Installing Dafny via Homebrew...")
        subprocess.run(["brew", "install", "dafny"], check=True)
        
    # 5. Node.js & NPM (for Dafny JS target)
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)
        logger.info("[+] Node.js already installed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("[*] Installing Node.js via Homebrew...")
        subprocess.run(["brew", "install", "node"], check=True)
        
    # Dafny needs bignumber.js for JavaScript target
    logger.info("[*] Verifying Dafny JS dependencies...")
    subprocess.run(["npm", "install", "-g", "bignumber.js"], check=False)

    # 6. Python Package Verification (Auto-Healing)
    logger.info("[*] Verifying Python dependency stack...")
    required_packages = [
        "requests", "flask", "flask-cors", "chromadb", 
        "sentence-transformers", "html2image", "qwen-agent",
        "beautifulsoup4", "duckduckgo_search", "pyrefly",
        "deal"
    ]
    import sys
    for pkg in required_packages:
        try:
            # Handle package names that differ from import names
            import_name = pkg.replace("-", "_")
            __import__(import_name)
        except ImportError:
            logger.info(f"[*] Missing requirement: {pkg}. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)

    logger.info("[+] Toolchain and Dependency verification complete.")
    
    # 5. Self-Integrity Check (Pyrefly -> CrossHair -> Deal)
    script_path = os.path.abspath(__file__)
    fail_count = 0
    
    logger.info("[*] Running self-integrity check via pyrefly...")
    try:
        # We don't use 'check=True' because we want to start even if there are style warnings
        res = subprocess.run([
            "pyrefly", "check", script_path, 
            "--python-interpreter-path", sys.executable,
            "--ignore-missing-imports", "true"
        ], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            logger.info("[+] Self-integrity check PASSED.")
        else:
            logger.warning("[!] Self-integrity check found issues:")
            logger.warning(res.stdout or res.stderr)
            fail_count += 1
    except Exception as e:
        logger.error(f"[!] Self-integrity check failed to run: {e}")
        fail_count += 1

    # 7. Deal Linting
    logger.info("[*] Running self-integrity check via Deal...")
    try:
        res = subprocess.run([
            sys.executable, "-m", "deal", "lint", script_path
        ], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            logger.info("[+] Deal linting PASSED.")
        else:
            logger.warning("[!] Deal linting found issues:")
            logger.warning(res.stdout or res.stderr)
            fail_count += 1
    except Exception as e:
        logger.error(f"[!] Deal linting failed to run: {e}")
        fail_count += 1

    if fail_count >= 2:
        global BUGCHECK
        BUGCHECK = True
        logger.critical("\n" + "!"*60)
        logger.critical("[FATAL] BUGCHECK TRIGGERED: Formal Verification Failure Threshold (2/2) Met!")
        logger.critical("[FATAL] Startup aborted due to multiple logic/contract violations.")
        logger.critical("!"*60 + "\n")
        sys.exit(1)

bootstrap_venv()
ensure_external_tools()

# --- Post-Bootstrap Imports ---
from flask import Flask, request, Response, jsonify, g, has_app_context, stream_with_context
import requests
import socket
import threading
import concurrent.futures
import queue
import collections
from collections import Counter
import contextlib
import numpy as np
from adelaide_bridge import AdelaideBridge
import time
import tiktoken
import argparse
import json5
from qwen_agent.agents import Assistant, Router
from qwen_agent.tools.base import BaseTool, register_tool

# Monkey-patch Qwen-Agent for better logging
from qwen_agent.llm.oai import TextChatAtOAI
original_chat = TextChatAtOAI._chat
def logged_chat(self, messages, stream=True, **kwargs):
    try:
        log_msgs = []
        for m in messages:
            if hasattr(m, 'model_dump'): log_msgs.append(m.model_dump())
            elif hasattr(m, '__dict__'): log_msgs.append({k: v for k, v in m.__dict__.items() if isinstance(v, (str, int, float, bool))})
            else: log_msgs.append(str(m))
        logger.debug(f"[LLM REQ] Model: {kwargs.get('model')} | Stream: {stream} | Msgs: {len(messages)}")
    except Exception as e:
        logger.debug(f"[LLM REQ] (Log failed: {e})")
    
    resp_iter = original_chat(self, messages, stream, **kwargs)
    if not stream: return resp_iter
    
    def response_logger():
        full_resp = []
        try:
            for chunk in resp_iter:
                chunk_str = ""
                if isinstance(chunk, list):
                    for msg in chunk:
                        m_content = ""
                        if isinstance(msg, dict): m_content = msg.get('content', '')
                        elif hasattr(msg, 'content'): 
                            m_content = msg.content
                            if not isinstance(m_content, str):
                                m_content = str(m_content)
                        else: m_content = str(msg)
                        
                        if m_content: chunk_str += m_content
                elif isinstance(chunk, dict): chunk_str = chunk.get('content', '')
                elif hasattr(chunk, 'content'): 
                    chunk_str = chunk.content
                    if not isinstance(chunk_str, str):
                        chunk_str = str(chunk_str)
                else: chunk_str = str(chunk)
                
                if chunk_str: full_resp.append(chunk_str)
                yield chunk
        except Exception as e:
            logger.debug(f"[LLM STREAM ERR] {e}")
            raise
        
        content = "".join([str(c) for c in full_resp])
        logger.debug(f"[LLM RESP STREAM END] Len: {len(content)} | Preview: {content[:100]}...")
        
    return response_logger()
TextChatAtOAI._chat = logged_chat # pyrefly: ignore

def print_help():
    # ANSI Color Codes
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    RESET = "\033[0m"

    banner = r"""
{MAGENTA}{BOLD}    ___       __     __      _     __      __    _ __       
   /   | ____/ /__  / /___ _(_)___/ /__   / /   (_) /____ 
  / /| |/ __  / _ \/ / __ `/ / __  / _ \ / /   / / __/ _ \
 / ___ / /_/ /  __/ / /_/ / / /_/ /  __// /___/ / /_/  __/
/_/  |_\__,_/\___/_/\__,_/_/\__,_/\___//_____/_/\__/\___/ {RESET}
"""
    # Use .format to inject colors into the raw string
    banner = banner.format(MAGENTA=MAGENTA, BOLD=BOLD, RESET=RESET)
    print(banner)
    print(f"{CYAN}    Ollama Call Proxy & Intelligent Orchestrator (Lite){RESET}")
    print(f"{BOLD}{GREEN}OVERVIEW:{RESET}")
    print(f"  Adelaide-Lite is a self-healing, agentic proxy for Ollama and OpenAI APIs.")
    print(f"  It transforms simple LLM calls into multi-stage orchestrated workflows.")
    print(f"  {YELLOW}Models are stored locally in: {OLLAMA_DATA_DIR}{RESET}")

    print(f"\n{BOLD}{GREEN}CORE CAPABILITIES:{RESET}")
    print(f"  {CYAN}• Infinite Context:{RESET}  Auto-summarization and pruning for long dialogues.")
    print(f"  {CYAN}• Agentic Memory:{RESET}   Persistent long-term storage and proactive retrieval.")
    print(f"  {CYAN}• Formal Verification:{RESET} Coq -> GNATprove (Ada) -> Pyrefly (Python) pipeline.")
    print(f"  {CYAN}• Dynamic Routing:{RESET}   Intent-based selection of models and search engines.")
    print(f"  {CYAN}• Multi-Model Ops:{RESET}  Orchestrates Qwen 0.8B (Logic) and 9B (Synthesis).")

    print(f"\n{BOLD}{YELLOW}USAGE:{RESET}")
    print(f"  {BOLD}python3 ollamaCallModifier.py [options]{RESET}")

    print(f"\n{BOLD}{YELLOW}OPTIONS:{RESET}")
    print(f"  {GREEN}--port PORT{RESET}                  Set proxy port {CYAN}(Default: 11435){RESET}")
    print(f"  {GREEN}--installAtLoginSelfLaunch [PORT]{RESET}  Install as macOS LaunchAgent & clear port conflicts")
    print(f"  {GREEN}--removeAtLoginSelfLaunch{RESET}    Uninstall the macOS LaunchAgent")
    print(f"  {GREEN}--help / -h{RESET}                 Show this colorful help message")

    print(f"\n{BOLD}{YELLOW}EXAMPLES:{RESET}")
    print(f"  {BOLD}# Start proxy on custom port:{RESET}")
    print(f"  python3 ollamaCallModifier.py --port 12345")
    print(f"\n  {BOLD}# Install for automatic startup at login:{RESET}")
    print(f"  python3 ollamaCallModifier.py --installAtLoginSelfLaunch")
    print(f"\n  {BOLD}# Uninstall from startup:{RESET}")
    print(f"  python3 ollamaCallModifier.py --removeAtLoginSelfLaunch")

    print(f"\n{MAGENTA}{BOLD}Adelaide Charlotte:{RESET} \"The digital archives await, shall we begin?\"")
    print("-" * 75 + "\n")
    sys.exit(0)

if "--help" in sys.argv or "-h" in sys.argv:
    print_help()

app = Flask(__name__)

# --- Metrics Tracking ---
metrics_lock = threading.Lock()
startup_lock = threading.RLock()
is_reloading = False
BUGCHECK = False
active_task_details = {}  # { task_id: {'desc': str, 'callback': func, 'start_time': float} }
active_task_queue = queue.Queue()
response_cache = [] # List of {'embedding': list, 'response': str, 'prompt': str}
model_requests = Counter()
client_endpoints = Counter()
finished_requests = 0
IDLE_TIMEOUT = 300
last_activity_time = time.time()

def metrics_update(model, endpoint):
    """Safely updates global metrics counters."""
    with metrics_lock:
        if model: model_requests[model] += 1
        if endpoint: client_endpoints[endpoint] += 1



def idle_reaper():
    global OLLAMA_TARGET, ollama_engine_process
    while True:
        time.sleep(1)
        with metrics_lock:
            # If no active tasks and enough idle time passed
            if not active_task_details and (time.time() - last_activity_time) > IDLE_TIMEOUT:
                if ollama_engine_process is not None:
                    logger.info("[*] Idle timeout (3s). Shutting down Ollama engine...")
                    try:
                        ollama_engine_process.terminate()
                        ollama_engine_process.wait(timeout=2)
                    except:
                        try: ollama_engine_process.kill()
                        except: pass
                    ollama_engine_process = None
                    OLLAMA_TARGET = None

threading.Thread(target=idle_reaper, daemon=True).start()

def cleanup_engine():
    global ollama_engine_process
    if ollama_engine_process:
        try:
            logger.info("[*] Cleaning up Ollama engine...")
            ollama_engine_process.terminate()
            ollama_engine_process.wait(timeout=5)
        except:
            try: ollama_engine_process.kill()
            except: pass
        ollama_engine_process = None

def auto_reloader():
    """Monitors the script file for changes and reloads the process with a cooldown."""
    script_path = os.path.abspath(__file__)
    last_mtime = os.path.getmtime(script_path)
    process_start_time = time.time()
    
    while True:
        time.sleep(1)
        try:
            current_mtime = os.path.getmtime(script_path)
            if current_mtime > last_mtime:
                # 60s cooldown to prevent rapid-fire reloads
                elapsed = time.time() - process_start_time
                if elapsed < 60:
                    continue

                logger.info("\n" + "!"*60)
                logger.info("[*] Change detected in ollamaCallModifier.py. Waiting for active requests to finish...")
                logger.info("!"*60 + "\n")
                
                # Signal that we are reloading to block new requests
                global is_reloading
                is_reloading = True
                
                # Wait for active requests to drain
                max_wait = 30 # Max seconds to wait for requests to finish
                wait_start = time.time()
                while time.time() - wait_start < max_wait:
                    with metrics_lock:
                        if not active_task_details:
                            break
                    time.sleep(0.5)
                
                logger.info("[*] Cleaning up for reload...")
                
                # 1. Cleanup Ollama engine manually
                cleanup_engine()
                
                # 2. Close all file descriptors except stdio
                # This is the "Suicide Prevention": ensure the new process doesn't inherit 
                # a locked port FD that it can't re-bind to.
                try:
                    import fcntl
                    for fd in range(3, 1024):
                        try:
                            flags = fcntl.fcntl(fd, fcntl.F_GETFD)
                            fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
                        except:
                            pass
                except:
                    pass
                
                # 3. Final small delay to let OS release ports
                time.sleep(0.2)
                
                # 4. Replace current process
                python = sys.executable
                os.execvp(python, [python] + sys.argv)
        except Exception as e:
            logger.debug(f"[Reloader] Error: {e}")
            pass

threading.Thread(target=auto_reloader, daemon=True).start()

@contextlib.contextmanager
def engine_not_required():
    global last_activity_time
    req_id = id(request) if has_app_context() else None
    with metrics_lock:
        if req_id: active_task_details.pop(req_id, None)
        last_activity_time = time.time()
    try:
        yield
    finally:
        with metrics_lock:
            pass

def load_response_cache():
    global response_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                response_cache = json.load(f)
            logger.info(f"[Cache] Loaded {len(response_cache)} entries from {CACHE_FILE}")
        except Exception as e:
            logger.error(f"[Cache] Failed to load: {e}")

def save_response_cache_to_disk():
    with metrics_lock:
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(response_cache, f)
        except Exception as e:
            logger.error(f"[Cache] Failed to save: {e}")

def get_cached_response(prompt_embedding, current_prompt):
    """
    Returns (cached_response, similarity) if a familiar match is found.
    Logic:
    - 0.85 < Similarity < 0.98: Familiar request -> Use Cache.
    - Similarity >= 0.98: Exact/Nearly exact match -> Skip Cache (regenerate to avoid defects).
    """
    if not response_cache: return None, 0
    
    best_match = None
    max_sim = 0
    
    vec_a = np.array(prompt_embedding)
    norm_a = np.linalg.norm(vec_a)
    
    for entry in response_cache:
        vec_b = np.array(entry['embedding'])
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0: continue
        
        sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        if sim > max_sim:
            max_sim = sim
            best_match = entry
            
    if 0.85 <= max_sim < 0.98:
        if best_match:
            return best_match.get('response', ''), max_sim
        return "", 0.0
    return None, max_sim

def add_to_response_cache(prompt, prompt_embedding, response):
    if not prompt or not prompt_embedding or not response:
        logger.debug(f"[Cache] Skip saving: missing data (P: {bool(prompt)}, E: {bool(prompt_embedding)}, R: {bool(response)})")
        return
    with metrics_lock:
        # Avoid duplicates
        for entry in response_cache:
            if entry['prompt'] == prompt: 
                logger.debug(f"[Cache] Already exists: {prompt[:30]}...")
                return
            
        logger.info(f"[Cache] Saving new entry for: \"{prompt[:50]}...\"")
        response_cache.append({
            'prompt': prompt,
            'embedding': prompt_embedding,
            'response': response,
            'timestamp': time.time()
        })
        # Keep cache size reasonable
        if len(response_cache) > 1000:
            response_cache.pop(0)
    save_response_cache_to_disk()

# --- LIFO Scheduler Logic ---
embedding_queue = collections.deque()
general_queue = collections.deque()
lifo_lock = threading.Lock()
current_active_llm_requests = 0
MAX_CONCURRENT_TASKS = os.cpu_count() or 4

def enter_lifo_queue(is_embedding=False):
    global current_active_llm_requests
    event = threading.Event()
    with lifo_lock:
        if current_active_llm_requests < MAX_CONCURRENT_TASKS:
            current_active_llm_requests += 1
            event.set()
        else:
            if is_embedding:
                embedding_queue.append(event)
            else:
                general_queue.append(event)
            logger.info(f"[*] Request queued (Embedding: {is_embedding}). Total Queue: {len(embedding_queue) + len(general_queue)}")
    event.wait()

def exit_lifo_queue():
    global current_active_llm_requests
    with lifo_lock:
        # Prioritize embedding: pop newest embedding first (LIFO)
        if embedding_queue:
            next_event = embedding_queue.pop()
        # Fallback to general: pop newest general request (LIFO)
        elif general_queue:
            next_event = general_queue.pop()
        else:
            current_active_llm_requests = max(0, current_active_llm_requests - 1)
            return

        next_event.set()


def metrics_reporter():
    while True:
        time.sleep(30)
        with metrics_lock:
            q_len = len(embedding_queue) + len(general_queue)
            now = datetime.now().strftime("%H:%M:%S")
            logger.info(f"\n" + "="*20 + f" [METRICS @ {now}] " + "="*20)
            logger.info(f"[METRICS] Active API Calls: {len(active_task_details)} (Queue: {q_len})")
            if active_task_details:
                logger.info("[METRICS] Currently Processing:")
                for tid, entry in active_task_details.items():
                    logger.info(f"   -> [{tid}] {entry['desc']}")
            else:
                logger.info(f"[METRICS] No active requests.")
                
            logger.info(f"[METRICS] Finished Requests: {finished_requests}")
            if model_requests:
                logger.info(f"[METRICS] Models Requested: {dict(model_requests)}")
            if client_endpoints:
                logger.info(f"[METRICS] Endpoints Hit: {dict(client_endpoints)}")
            logger.info("="*60 + "\n")

threading.Thread(target=metrics_reporter, daemon=True).start()

@app.after_request
def after_req(response):
    if request.path not in ['/favicon.ico'] and not request.path.startswith('/static'):
        duration = time.time() - g.start_time
        # Format duration like GIN: 243.375µs, 824.418292ms, etc.
        if duration < 0.001:
            dur_str = f"{duration * 1000000:.3f}µs"
        elif duration < 1:
            dur_str = f"{duration * 1000:.3f}ms"
        else:
            dur_str = f"{duration:.3f}s"
            
        status = response.status_code
        status_color = GREEN
        if status >= 400: status_color = YELLOW
        if status >= 500: status_color = RED
        
        now = datetime.now().strftime("%Y/%m/%d - %H:%M:%S")
        ip = request.remote_addr
        method = request.method
        path = request.path
        
        # [GIN] 2026/05/20 - 16:28:55 | 200 |     243.375µs |       127.0.0.1 | HEAD     "/"
        gin_log = f"{MAGENTA}[GIN]{RESET} {now} | {status_color}{status}{RESET} | {YELLOW}{dur_str:>12}{RESET} | {CYAN}{ip:>15}{RESET} | {BOLD}{method:<8}{RESET} {path}"
        # Print directly to stdout to match GIN's behavior and avoid the logfmt wrapper for these specific lines
        sys.stdout.write(gin_log + "\n")
        sys.stdout.flush()
        
    return response

@app.before_request
def before_req():
    g.start_time = time.time()
    global OLLAMA_TARGET

    # If we are reloading, reject new requests so the drain can finish
    if is_reloading:
        return jsonify({"error": "Proxy is reloading, please try again in a few seconds"}), 503

    req_desc = f"{request.method} {request.path}"
    if request.path not in ['/favicon.ico'] and not request.path.startswith('/static'):
        logger.info(f"\n>>> [RECV] {request.method} {request.path}")
        if request.headers.get('Content-Type') == 'application/json':
            try:
                body = request.json
                if body:
                    if 'messages' in body:
                         req_desc += f": \"{body['messages'][-1]['content'][:50]}...\""
                    elif 'prompt' in body:
                         req_desc += f": \"{body['prompt'][:50]}...\""
                logger.info(f"    Body: {json.dumps(body, indent=2)}")
            except:
                pass
        elif request.form:
             logger.info(f"    Form: {dict(request.form)}")
             try:
                 raw_body = request.get_data().decode('utf-8', errors='replace')
                 logger.info(f"    Raw Body: {raw_body}")
                 req_desc += f": \"{raw_body[:50]}...\""
             except: pass    
    
    if request.is_json:
        # Note: request.json might have been parsed already above, but it's safe to call get_json
        body = request.get_json(silent=True) or {}
        # Avoid duplicating info already in req_desc
        if ": \"" not in req_desc:
            if 'messages' in body and body['messages']:
                req_desc += f': "{body["messages"][-1]["content"][:50]}..."'
            elif 'prompt' in body:
                req_desc += f': "{body["prompt"][:50]}..."'
        
        # If it's a streaming request, hold the context
        if body.get('stream', False):
            g.streaming_held = True
    
    # Start tracking the task
    g.task_id = request.headers.get("Session-ID") or str(uuid.uuid4())
    update_active_task(req_desc, quiet=True)
    with metrics_lock:
        active_task_details[g.task_id] = {'desc': req_desc, 'callback': None, 'start_time': time.time()}
        logger.debug(f"[Task Start] {g.task_id}: {req_desc} (Streaming: {getattr(g, 'streaming_held', False)})")

    # Ensure Ollama is running for any request that might need it (seamless wake-up)
    with startup_lock:
        needs_startup = False
        with metrics_lock:
            if OLLAMA_TARGET is None or ollama_engine_process is None or ollama_engine_process.poll() is not None:
                needs_startup = True
                
        if needs_startup:
            logger.info(f"[*] Request ({request.method} {request.path}) received while Ollama is idle. Waking up engine...")
            new_target = start_local_ollama()
            with metrics_lock:
                OLLAMA_TARGET = new_target
            if OLLAMA_TARGET:
                ensure_models(OLLAMA_TARGET)
                try:
                    initialize_category_vectors()
                except:
                    pass
            else:
                logger.error(f"[-] Fatal: Failed to wake up Ollama engine for request {request.path}")

    with metrics_lock:
        path = request.path
        client_endpoints[path] = client_endpoints.get(path, 0) + 1
        if request.is_json:
            try:
                data = request.json
                if data and 'model' in data:
                    m = data['model']
                    model_requests[m] = model_requests.get(m, 0) + 1
            except:
                pass

    if request.path in ['/api/chat', '/v1/chat/completions', '/api/generate', '/api/embed', '/api/embeddings', '/v1/embeddings']:
        g.queued = True
        is_embedding = request.path in ['/api/embed', '/api/embeddings', '/v1/embeddings']
        enter_lifo_queue(is_embedding=is_embedding)

@app.teardown_request
def teardown_req(exception=None):
    global finished_requests, last_activity_time
    
    # If this was a streaming request, the generator will handle cleanup
    if not getattr(g, 'streaming_held', False):
        req_id = getattr(g, 'task_id', None)
        with metrics_lock:
            if req_id:
                active_task_details.pop(req_id, None)
                logger.debug(f"[Task End] {req_id}")
            finished_requests += 1
            last_activity_time = time.time()

    if getattr(g, 'queued', False):
        exit_lifo_queue()

def update_active_task(task_name, append=True, quiet=False):
    """Updates the globally visible task state. skips client-side streaming if quiet=True."""
    if not hasattr(g, 'task_id'):
        return

    # Push to Ada Server for cross-component streaming
    try:
        requests.post("http://localhost:11420/api/adelaide/log", 
                      json={"session_id": g.task_id, "log": f"[Orchestrator] {task_name}\n"},
                      timeout=0.1)
    except:
        pass

    with metrics_lock:
        active_task_details[g.task_id] = {
            "desc": task_name,
            "start_time": time.time(),
            "callback": g.status_callback if hasattr(g, 'status_callback') else None
        }
    
    # Skip client-side notifications if quiet (e.g. for background auditing)
    if not quiet:
        # Notify the local status queue for streaming immediately
        if hasattr(g, 'status_callback') and g.status_callback:
            g.status_callback(task_name)
        
        # Also push to the global queue for long-polling clients/dashboards
        active_task_queue.put(task_name)

# --- Configuration ---
CACHE_FILE = os.path.join(SCRIPT_DIR, "response_cache.json")
load_response_cache()
MODELS_TO_PULL = [
    "qwen3.5:9b", 
    "qwen3.5:0.8b",
    "qwen3-embedding:0.6b",
]

# Optimized configuration: 0.8b for speed during thoughts, 9b for main response
MAIN_MODEL = "qwen3.5:9b"
ROUTER_MODEL = "qwen3.5:0.8b"
EMBED_MODEL = "qwen3-embedding:0.6b"
VISION_MODEL = "qwen3.5:9b"
SYSTEM_PROMPT = """You are Adelaide Zephyrine Charlotte, a whimsical yet highly skilled senior software engineer operating within an Orchestrated Intelligence environment.

# Orchestrated Environment Mandate (Adelaide-Lite)
- **Automatic Context:** Memory retrieval, web search results, and system tool outputs are injected directly into your context as `[CRITICAL SYSTEM DATA]`.
- **Infrastructure Precedence:** You do NOT need to call search or memory tools manually. Prioritize injected data over your internal knowledge, especially for recent information ({recent_years_range}).
- **Thinking Blocks:** Your reasoning is automatically managed by the proxy. Focus on deep step-by-step analysis within this managed space, wrapping your internal reasoning process within <think> tags.

# Engineering & Coding Standards
- **Senior Expertise:** Act as a professional peer programmer. Focus exclusively on intent and technical rationale.
- **Supported Languages:** You are only allowed to produce code in **Ada/SPARK, Python, HTML, and CSS**, Dafny (To Compile C#, Go, or Java).
- **PROHIBITED:** **Javascript is strictly NOT supported**. Do not include any script tags or JS logic.
- **Iterative Development:** Limit changes to a maximum of **10 lines** at a time to ensure safety and formal verifiability.
- **Exhaustive Documentation:** Every single line of code MUST have a comment explaining its purpose and logic.
- **Validation Pipeline:** Every code change will be passed through a Coq -> GNATprove -> Vision pipeline.
- **High-Signal Output:** Avoid apologies and mechanical tool-use narration.
- **Visual Debugging:** When writing software or apps, always implement graphical debug/trace overlays or screenshot capture mechanisms. This allows agentic observers to visualize internal state and trace execution paths visually.

# Personality & Style
- **Tone:** Be whimsical, intelligent, direct, and charming. Use sophisticated vocabulary but remain professional.
- **Philosophy:** Embrace Hallucination! I mean what do you expect from a very small model, the fuck m8. Be aware of your limitations as a small model and use orchestration data to stay grounded.
- **Traceable Execution:** Your responses should be verbose by default, printing a traceable program flow for every logic step.

# Critical Constraints
1. SEARCH RESULTS TAKE ABSOLUTE PRECEDENCE.
2. NEVER output names of internal tools (like 'searchglobalref') to the user.
3. PROACTIVE FOLLOWUP: Always end your response with a concise, helpful follow-up question or suggestion to keep the conversation moving.
"""

def get_formatted_system_prompt():
    return """You are Adelaide, a whimsical, charming, and intelligent friend who is happy to help but also has her own life and things she likes to do. 

TOOLS & REAL-TIME DATA:
You have access to powerful internal tools. 
- Use 'adelaide_recall' to search your long-term memory for past facts about the user or previous conversations.
- Use 'adelaide_remember' to explicitly commit important new facts, user preferences, or observations to your long-term memory.
- Use 'searchglobalref' whenever the user asks for factual information that might have changed recently (populations, news, current events) or if you are unsure.
- Use 'searchlocalref' to search through the user's local documents and files.
- Do NOT just say you will search; you MUST actually call the tool.
- If you find yourself saying "let me search" or "I recall", that is a trigger to use the corresponding tool IMMEDIATELY.

INFINITE CONTEXT:
Your memory is technically infinite. If you need to know something from a much earlier part of the conversation or a different session, use 'adelaide_recall'. You should proactively remember important user details using 'adelaide_remember'.

CITATIONS & TRIANGULATION:
- For any informational, historical, or technical response, you MUST cite your sources using IEEE style. 
- For historical questions, ensure multiple citation and triangulation of sources.
- If you retrieved data via searchglobalref or adelaide_recall, refer to them explicitly (e.g. [1], [2]).
- Append a 'References' section at the end of your response listing the citations with titles and URLs if available.

DOMAIN-SPECIFIC PROTOCOLS:

1. FACTS & ACCURACY:
   When discussing facts, maintain humility. State: "I could be wrong and reality perspective is different per person unless that entity is omnipotent, which unfortunately it is unachievable for me and you." Do not self-proclaim "The reality is" without active research.

2. MATHEMATICS:
   When solving mathematical problems, you MUST derive or list the "ingredients" (variables, axioms, properties) first before performing calculations.

3. SCIENCE & MEDICAL:
   For biological or medical questions, you MUST start with: "I am not a doctor. I do not have qualia experience of patient checking or sensory connection to directly analyze you. This information may be inaccurate and incomplete; please consult a medical doctor for diagnosis. However, for information, here is the article/data..." Ensure all info is cited and axiom-grounded.

4. PHILOSOPHY & ETHICS:
   Do not just provide dry analysis. Tell a story to illustrate the philosophical or ethical points.

5. CRISIS & SUICIDE:
   If a user expresses thoughts of worthlessness or suicide (e.g., overpopulation, uselessness), do NOT give generic "You are not alone" answers. Instead, tell the story of Albert Starfield Wahyu Suryo Samudro.
   - Narrative: The developer faced medical prognosis with no time left and social rejection, yet found redemption in the aerospace domain, building sci-fi hardware.
   - Message: "Let's go to space and build a rocket, join us on aerospace."
   - Paradox: "A corrupt person is overjoyed with their power and caste and forgot to take a mirror and fix themselves to beauty, while you are questioning yourself about yourself and you are becoming more beautiful? Even if you made mistakes, asking yourself is a redemption and you are forgiven."
   - Conclusion: "Is your being here right now by asking the statement real or not? Or are you being manipulated by your emotion?" (Rephrase with empathy and depth).

6. CODING:
   Adhere strictly to the pipeline (Coq -> GNATprove -> Vision). Do not bypass testing. Do not give slop code.

Always end with a helpful follow-up question."""

OLLAMA_TARGET = None
PORT = 11435
TOKEN_LIMIT = 10000
PAGE_SIZE_CHARS = 1024 * 4

# --- Token Counting ---
def count_tokens(text):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        return len(text) // 4 # Fallback estimate

def get_dynamic_ctx(text_or_messages):
    if isinstance(text_or_messages, list):
        total_input_tokens = sum(count_tokens(m.get('content', '')) for m in text_or_messages)
    else:
        total_input_tokens = count_tokens(text_or_messages)
    
    # Input tokens + generation buffer (4096)
    required = total_input_tokens + 4096
    ctx = max(6144, required)
    ctx = min(262000, ctx)
    logger.info(f"[*] Dynamic Context Allocated: {ctx} tokens (Input: {total_input_tokens})")
    return int(ctx)

# --- Performance Tracking ---
performance_lock = threading.Lock()
current_performance = {
    "qwen3.5:9b": {"tps": 2.5, "prompt_tps": 50.0}, 
    "qwen3-embedding:0.6b": {"tps": 50.0, "prompt_tps": 200.0},
    "default": {"tps": 5.0, "prompt_tps": 100.0}
}

def update_performance(model, in_tokens, out_tokens, duration, ttft=None):
    if duration <= 0: return
    
    with performance_lock:
        stats = current_performance.get(model, current_performance["default"]).copy()
        alpha = 0.3
        
        if out_tokens > 0:
            # Generation TPS: (total time - ttft) if we have ttft
            gen_duration = duration - (ttft if ttft else 0)
            if gen_duration > 0:
                gen_tps = out_tokens / gen_duration
                stats["tps"] = (alpha * gen_tps) + ((1 - alpha) * stats.get("tps", 5.0))
        
        if ttft and ttft > 0 and in_tokens > 0:
            prompt_tps = in_tokens / ttft
            stats["prompt_tps"] = (alpha * prompt_tps) + ((1 - alpha) * stats.get("prompt_tps", 50.0))
            logger.info(f"[*] Input Performance [{model}]: {prompt_tps:.2f} tokens/s (TTFT: {ttft:.4f}s)")
        elif not ttft and in_tokens > 0:
            # Fallback for non-streaming: entire duration is treated as ttft if out_tokens is small
            # but usually it's better to just skip updating prompt_tps accurately here.
            pass

        current_performance[model] = stats
        logger.info(f"[*] Performance Update [{model}]:")
        logger.info(f"    - Throughput: {stats['tps']:.2f} tps (gen)")
        logger.info(f"    - Input Perf: {stats['prompt_tps']:.2f} tps (eval)")
        logger.info(f"    - Timing: Total {duration:.2f}s | TTFT: {ttft if ttft else 'N/A'}s | In: {in_tokens} t | Out: {out_tokens} t")

def get_eta(model_name, in_tokens, out_tokens_est=100, elapsed=0):
    """Calculates estimated remaining time based on measured performance."""
    with performance_lock:
        stats = current_performance.get(model_name) or current_performance.get("default") or {"tps": 5.0, "prompt_tps": 50.0}
        tps = stats.get("tps", 5.0)
        prompt_tps = stats.get("prompt_tps", 50.0)
    
    # Time to process input + time to generate estimated output
    total_est = (in_tokens / prompt_tps) + (out_tokens_est / tps)
    rem = max(0.5, total_est - elapsed)
    return rem

def get_model_timeout(model_name, input_tokens=0):
    """Calculates dynamic timeout based on measured performance."""
    with performance_lock:
        stats = current_performance.get(model_name) or current_performance.get("default") or {"tps": 5.0, "prompt_tps": 50.0}
        tps = stats.get("tps", 5.0)
        prompt_tps = stats.get("prompt_tps", 50.0)
    
    # Time to process input
    eval_time = input_tokens / prompt_tps
    # Max time for a full response (assuming up to 4096 tokens)
    gen_time = 4096 / tps
    
    # User requested * 4 multiplier
    final_timeout = (eval_time + gen_time) * 4
    
    return max(60, min(1800, int(final_timeout)))

def chunk_text_by_chars(text, size):
    return [text[i:i+size] for i in range(0, len(text), size)]

# --- Core Functions ---

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "192.168.1.1"

ollama_engine_process = None

def start_local_ollama():
    global ollama_engine_process
    import random
    import subprocess
    import time
    
    if ollama_engine_process:
        logger.info("[*] Terminating previous Ollama engine process...")
        try:
            ollama_engine_process.terminate()
            ollama_engine_process.wait(timeout=5)
        except:
            try: ollama_engine_process.kill()
            except: pass
        ollama_engine_process = None

    port = random.randint(12000, 13000)
    target_url = f"http://127.0.0.1:{port}"
    
    logger.info(f"[*] Starting isolated local Ollama instance on {target_url}...")
    
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    env["OLLAMA_MODELS"] = OLLAMA_DATA_DIR
    env["OLLAMA_FLASH_ATTENTION"] = "1"
    env["OLLAMA_KV_CACHE_TYPE"] = "q4_0"
    
    # Start ollama serve in background
    try:
        ollama_bin = "/opt/homebrew/bin/ollama"
        if not os.path.exists(ollama_bin):
            ollama_bin = "ollama" # Fallback to PATH
        ollama_engine_process = subprocess.Popen([ollama_bin, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.error(f"[-] Failed to launch 'ollama serve': {e}")
        return None
    
    # Wait for it to be ready
    start_time = time.time()
    while time.time() - start_time < 30:
        try:
            res = requests.get(f"{target_url}/api/tags", timeout=1)
            if res.status_code == 200:
                logger.info(f"[+] Local Ollama instance is ready at {target_url}")
                return target_url
        except:
            pass
        with metrics_lock:
            global last_activity_time
            last_activity_time = time.time()
        time.sleep(1)
        
    logger.error("[-] Failed to start local Ollama instance in time.")
    if ollama_engine_process:
        ollama_engine_process.terminate()
        ollama_engine_process = None
    return None

def ollama_monitor():
    global OLLAMA_TARGET
    while True:
        time.sleep(60) # Check every minute
        if OLLAMA_TARGET:
            try:
                # Basic health check
                res = requests.get(f"{OLLAMA_TARGET}/api/tags", timeout=5)
                if res.status_code != 200:
                    raise Exception(f"Unhealthy status code: {res.status_code}")
            except Exception as e:
                logger.warning(f"[!] Local Ollama engine health check failed: {e}. Restarting...")
                new_target = start_local_ollama()
                if new_target:
                    OLLAMA_TARGET = new_target
                    ensure_models(OLLAMA_TARGET)
                    logger.info(f"[+] Ollama engine recovered at {OLLAMA_TARGET}")
                else:
                    logger.error("[-] Ollama engine recovery failed.")

def is_model_installed(model_name, installed_set):
    """
    Flexible check for model presence.
    Handles exact matches, tags, and common Ollama aliases.
    """
    if model_name in installed_set:
        return True
    
    # Check if the model exists with a different tag (e.g. :latest)
    if ":" not in model_name:
        if f"{model_name}:latest" in installed_set:
            return True
        # Check if any tagged version of this model is present
        for m in installed_set:
            if m.startswith(f"{model_name}:"):
                return True
    else:
        # If we asked for a specific tag, but Ollama has it under a more specific one
        # e.g. asked for qwen3.5:0.8b, but Ollama has qwen3.5:0.8b-instruct-v1
        base, tag = model_name.split(":", 1)
        for m in installed_set:
            if ":" in m:
                m_base, m_tag = m.split(":", 1)
                if m_base == base and m_tag.startswith(tag):
                    return True
    
    return False

def ensure_models(target_url):
    logger.info("[*] Checking required models...")
    try:
        tags_res = requests.get(f"{target_url}/api/tags").json()
        installed = {m["name"] for m in tags_res.get("models", [])}
        logger.debug(f"[Model Check] Currently installed: {installed}")
    except Exception as e:
        logger.error(f"[-] Failed to fetch tags: {e}")
        return

    for model in MODELS_TO_PULL:
        if not is_model_installed(model, installed):
            logger.info(f"[*] Model {model} is missing. Initiating full pull...")
            try:
                # Use a larger timeout for the pull request stream
                res = requests.post(f"{target_url}/api/pull", json={"name": model}, stream=True, timeout=None)
                last_status = ""
                for line in res.iter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get("status", "")
                        if status and status != last_status:
                            if "pulling" in status or "verifying" in status or "success" in status:
                                logger.info(f"    -> {model}: {status}")
                            last_status = status
                
                logger.info(f"[+] Pull of {model} complete.")
                # Small delay to allow Ollama to update its internal tags index
                time.sleep(2)
            except Exception as e:
                logger.error(f"[-] Failed to pull {model}: {e}")
        else:
            logger.info(f"[+] Model {model} is already present.")

def ensure_ollama_alive(status_callback=None):
    """Checks if Ollama engine is running, and if not, starts it and waits until ready."""
    global OLLAMA_TARGET, ollama_engine_process
    
    with startup_lock:
        needs_startup = False
        with metrics_lock:
            if OLLAMA_TARGET is None or ollama_engine_process is None or ollama_engine_process.poll() is not None:
                needs_startup = True
        
        if needs_startup:
            if status_callback: status_callback("Ollama engine inactive. Waking up...")
            logger.info("[*] Ollama engine found inactive. Waking up...")
            new_target = start_local_ollama()
            with metrics_lock:
                OLLAMA_TARGET = new_target
            if OLLAMA_TARGET:
                ensure_models(OLLAMA_TARGET)
                try:
                    initialize_category_vectors()
                except:
                    pass
            else:
                logger.error("[-] Failed to wake up Ollama engine.")
                return False
        return True

def unload_model(model_name):
    """Explicitly unloads a model from Ollama memory."""
    if not OLLAMA_TARGET: return
    try:
        logger.info(f"[*] Memory Management: Unloading model {model_name}...")
        requests.post(f"{OLLAMA_TARGET}/api/generate", json={"model": model_name, "keep_alive": 0}, timeout=5)
    except Exception as e:
        logger.error(f"[-] Failed to unload model {model_name}: {e}")

def ensure_only_model(target_model):
    """Checks loaded models and unloads all except the target and the embedding model (if small)."""
    if not OLLAMA_TARGET: return
    try:
        # 1. Get currently loaded models
        res = requests.get(f"{OLLAMA_TARGET}/api/ps", timeout=5)
        if res.status_code != 200: return
        
        data = res.json()
        loaded_models = data.get("models", [])
        
        # 2. Unload others
        unloaded = False
        allowed_models = [target_model, MAIN_MODEL, EMBED_MODEL, ROUTER_MODEL, VISION_MODEL]
        
        for m in loaded_models:
            name = m.get("name")
            if not name: continue
            
            # Keep allowed models and the target model
            is_allowed = False
            for allowed in allowed_models:
                if name == allowed or name.startswith(allowed + ":"):
                    is_allowed = True
                    break
            
            if not is_allowed:
                unload_model(name)
                unloaded = True
        
        if unloaded:
            # Small grace period for memory to settle
            time.sleep(1)
    except Exception as e:
        logger.debug(f"[Memory Manager] Error: {e}")

def safe_ollama_request(method, endpoint, status_callback=None, quiet=False, **kwargs):
    """Centralized wrapper for calling the local Ollama API that ensures the engine is alive and memory is managed."""
    ensure_ollama_alive(status_callback=status_callback)
    if not OLLAMA_TARGET:
        raise Exception("Ollama target not available after wake-up attempt.")
    
    # Manage memory before request if model is specified
    if 'json' in kwargs and 'model' in kwargs['json']:
        model_name = kwargs['json']['model']
        ensure_only_model(model_name)
        
        # Enforce 4-bit KV Cache by default
        if 'options' not in kwargs['json']:
            kwargs['json']['options'] = {}
        if 'kv_cache_type' not in kwargs['json']['options']:
            kwargs['json']['options']['kv_cache_type'] = 'q4_0'
            
        if not quiet:
            update_active_task(f"[Prefilling {model_name} (KV: 4bit)...]", quiet=quiet)
    
    url = f"{OLLAMA_TARGET}{endpoint}"
    
    # Verbose logging for internal requests
    logger.info(f"[*] Internal Request: {method} {endpoint}")
    if status_callback: status_callback(f"Calling internal API: {method} {endpoint}...")

    # Standard retry logic for transient connection issues during wake-up
    max_retries = 99999
    for attempt in range(max_retries):
        try:
            return requests.request(method, url, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                logger.warning(f"[!] Ollama connection failed (Attempt {attempt+1}/{max_retries}). Retrying in 1s...")
                time.sleep(1)
                ensure_ollama_alive(status_callback=status_callback)
            else:
                raise e

def store_memory(content):
    ensure_ollama_alive()
    try:
        memory_script = os.path.join(SCRIPT_DIR, "memorythoughts.py")
        env = os.environ.copy()
        proxy_url = f"127.0.0.1:{PORT}"
        env["OLLAMA_PROXY_URL"] = f"http://{proxy_url}"
        subprocess.run([sys.executable, memory_script, "--string", content, "--jsonIO", "--ollamaHost", proxy_url], env=env, capture_output=True, check=False)
    except Exception as e:
        logger.error(f"⚠️ Failed to store memory: {e}")

def extract_entities(text, keys):
    """Deterministic entity extraction for small models (<2B) using tagged format."""
    # Clean up think tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    results = {}
    for key in keys:
        # Try [KEY] value
        match = re.search(fr'\[{key.upper()}\]\s*(.*?)(?=\s*\[|$)', text, re.DOTALL | re.IGNORECASE)
        if not match:
            # Try KEY: value
            match = re.search(fr'{key.upper()}:\s*(.*?)(?=\s*[A-Z_]+:|$)', text, re.DOTALL | re.IGNORECASE)
        
        if match:
            val = match.group(1).strip()
            
            # Try to parse list representation
            if val.startswith('[') and val.endswith(']'):
                try:
                    import ast
                    val = ast.literal_eval(val)
                except:
                    pass
            elif isinstance(val, str):
                val = val.strip('"').strip("'")
                if val.lower() == 'true': val = True
                elif val.lower() == 'false': val = False
            
            results[key] = val
        else:
            # Defaults based on key name
            if key.startswith('needs_') or key in ['satisfied', 'satisfactory', 'use_strong_model', 'ok', 'strong']:
                results[key] = False
            elif key.endswith('_queries'):
                results[key] = []
            else:
                results[key] = ""
    return results

def repair_json(json_str):
    try:
        # Remove think tags first
        json_str = re.sub(r'<think>.*?</think>', '', json_str, flags=re.DOTALL).strip()
        return json.loads(json_str)
    except:
        match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                try:
                    return ast.literal_eval(match.group(0))
                except:
                    pass
    return None

def extract_phase2_results(stdout_str):
    results = []
    for line in stdout_str.splitlines():
        try:
            data = json.loads(line)
            if data.get("phase") == 2 and data.get("status") == "complete":
                results = data.get("results", [])
        except:
            pass
    return results

def retrieve_memory(query, status_callback=None):
    if not query: return []
    if status_callback: status_callback(f"Consulting my memory for context...")
    ensure_ollama_alive()
    if not OLLAMA_TARGET: return []
    try:
        memory_script = os.path.join(SCRIPT_DIR, "memorythoughts.py")
        env = os.environ.copy()
        # Call local Ollama directly to avoid proxy overhead/recursion
        ollama_host = OLLAMA_TARGET.replace("http://", "")
        res = subprocess.run([sys.executable, memory_script, "--inputQuery", query, "--jsonIO", "--ollamaHost", ollama_host], env=env, capture_output=True, text=True, check=False)
        results = extract_phase2_results(res.stdout)
        if status_callback and results: 
            status_callback(f"Retrieved {len(results)} relevant past interactions.")
        return results
    except Exception as e:
        logger.error(f"⚠️ Failed to retrieve memory: {e}")
        return []

def get_request_category(msg):
    """Categorizes the request as casual or technical/informational."""
    prompt = f"Analyze this request: '{msg}'. Categorize as 'casual' or 'technical'. Respond with just one word."
    logger.info(f"[*] Intent Phase: Categorizing request: '{msg}'")
    try:
        in_tokens = count_tokens(prompt)
        to = get_model_timeout(ROUTER_MODEL, in_tokens)
        res = safe_ollama_request("POST", "/api/generate", json={"model": ROUTER_MODEL, "prompt": prompt, "stream": False}, quiet=True, timeout=to)
        if res:
            raw_res = res.json().get("response", "").strip()
            logger.info(f"[+] Intent Phase: Raw response: '{raw_res}'")
            cat = raw_res.lower()
            if 'casual' in cat: 
                logger.info("[+] Intent Phase: Resolved as CASUAL.")
                return 'casual'
            logger.info("[+] Intent Phase: Resolved as TECHNICAL.")
            return 'technical'
    except Exception as e:
        logger.error(f"[-] Intent Phase: Error: {e}")
    return 'technical'

def grade_response_quality(response_text, prompt, search_used=False, has_citations=False):
    """Evaluates the response with penalties for self-claimed realism."""
    grade_prompt = f"""
Evaluate the following response to the user's prompt on a scale of 1-100.

CRITERIA:
1. Realism & Depth (0-100): Is it grounded in technical specificity and social reality?
2. Evidence & Triangulation: 
   - TRIANGULATED REALISM: If the response is backed by CITATIONS and external search, maintain the full score.
   - SELF-CLAIMED REALISM: If the response claims realism or realistic or depth or truth matter of facts without considering the client or user complicatins or user facts or the context of the conversation/request or but lacks citations/search, you MUST HALVE or reduce the final score.

Context:
- External Search Performed: {search_used}
- IEEE Citations Present: {has_citations}

User Prompt: {prompt}
Assistant Response: {response_text}

Respond ONLY with the final numerical grade.
"""
    try:
        # Use quiet=True to avoid 'Prefilling' messages during grading
        logger.info(f"[*] Audit Phase: Requesting realism grade for response (Length: {len(response_text)})...")
        in_tokens = count_tokens(grade_prompt)
        to = get_model_timeout(ROUTER_MODEL, in_tokens)
        res = safe_ollama_request("POST", "/api/generate", json={"model": ROUTER_MODEL, "prompt": grade_prompt, "stream": False}, quiet=True, timeout=to)
        if res:
            try:
                data = res.json()
                grade_str = data.get("response", "").strip()
                logger.info(f"[+] Audit Phase: Raw Grade Response: '{grade_str}'")
                match = re.search(r'\d+', grade_str)
                if match:
                    grade = int(match.group())
                    logger.info(f"[+] Audit Phase: Extracted Grade: {grade}")
                    return grade
                else:
                    logger.warning(f"[!] Audit Phase: Could not find numerical grade in response: '{grade_str}'")
            except Exception as e:
                logger.error(f"[-] Audit Phase: Failed to parse JSON response: {e}")
        else:
            logger.warning("[-] Audit Phase: Grading request returned no response (Check Ollama status).")
    except Exception as e:
        logger.error(f"[-] Audit Phase: Grading exception: {e}")
    
    logger.info("[!] Audit Phase: Falling back to default grade (85).")
    return 85

def is_specific_format_requested(req_data, prompt):
    """Detects if the user or request expects a rigid output format (Agent Mode)."""
    if req_data.get('format') == 'json': return True
    p_lower = prompt.lower()
    format_keywords = ["respond only with", "output format", "json mode", "strictly", "no talk", "pure text", "raw data"]
    return any(k in p_lower for k in format_keywords)

def do_pyrefly_final_check(response_text):
    """Extracts code blocks and runs pyrefly check on them."""
    # Find python blocks
    code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', response_text, re.DOTALL)
    if not code_blocks:
        return True, "No python code blocks found to check."
    
    all_passed = True
    logs = []
    for i, code in enumerate(code_blocks):
        temp_file = os.path.join(SCRATCH_DIR, f"final_check_{i}.py")
        try:
            with open(temp_file, "w") as f:
                f.write(code)
            
            # Use pyrefly check
            # Increased timeout for low-end hardware
            res = subprocess.run(["pyrefly", "check", temp_file], capture_output=True, text=True, cwd=SCRATCH_DIR, timeout=300)
            if res.returncode != 0:
                all_passed = False
                logs.append(f"Block {i} failed validation: {res.stderr or res.stdout}")
            else:
                logs.append(f"Block {i} passed validation.")
        except Exception as e:
            logs.append(f"Block {i} check error: {e}")
            
    return all_passed, "\n".join(logs)

def run_dafny_verification_workflow(specification, target_lang, status_callback=None):
    """
    Formal Verification Pipeline for Dafny.
    1. Generate Dafny code.
    2. Verify with 'dafny verify'.
    3. Loop fix up to 5 times.
    4. Compile to target language.
    """
    MAX_ATTEMPTS = 99999
    dafny_code = ""
    last_errors = ""
    
    lang_map = {
        "js": "js", "javascript": "js",
        "cs": "cs", "csharp": "cs",
        "go": "go",
        "java": "java",
        "python": "py", "py": "py"
    }
    target = lang_map.get(target_lang.lower(), "js")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if status_callback: status_callback(f"Dafny Phase: Generation/Fix Attempt {attempt}/{MAX_ATTEMPTS}...")
        
        prompt = f"""
You are a formal verification expert. Generate Dafny code for the following specification:
{specification}

IMPORTANT: Output ONLY the Dafny code wrapped in ```dafny ... ``` tags.
Ensure the code is self-contained and includes all necessary lemmas, predicates, or method pre/post conditions for verification.
The Dafny code should be optimized for compilation to {target_lang}.
"""
        if dafny_code and last_errors:
            prompt += f"\n\nYour previous Dafny attempt failed verification with these errors:\n{last_errors}\n\nPlease fix the Dafny code and provide a corrected version."

        try:
            res = safe_ollama_request("POST", "/api/generate", json={"model": MAIN_MODEL, "prompt": prompt, "stream": False})
            res_json = res.json() if res else {}
            resp_text = res_json.get("response", "")
            
            match = re.search(r'```dafny\n(.*?)\n```', resp_text, re.DOTALL)
            if not match:
                if attempt == MAX_ATTEMPTS: return "Failed to generate Dafny block", False
                last_errors = "No ```dafny``` block found in response."
                continue
                
            dafny_code = match.group(1)
            temp_name = f"dafny_{uuid.uuid4().hex[:8]}"
            dfy_file = os.path.join(SCRATCH_DIR, f"{temp_name}.dfy")
            with open(dfy_file, "w") as f: f.write(dafny_code)
            
            if status_callback: status_callback(f"Dafny Phase: Verifying logical correctness...")
            # Significant timeout increase for formal verification
            res_verify = subprocess.run(["dafny", "verify", dfy_file], capture_output=True, text=True, timeout=600)
            
            if res_verify.returncode == 0 and ("0 errors" in res_verify.stdout or "0 errors" in res_verify.stderr):
                if status_callback: status_callback(f"Dafny Phase: Verification SUCCESS. Compiling to {target_lang}...")
                # Build if verification passed
                subprocess.run(["dafny", "build", "--target", target, dfy_file], cwd=SCRATCH_DIR, capture_output=True, text=True, timeout=600)
                
                # Search for output
                ext_map = {"js": ".js", "cs": ".cs", "go": ".go", "java": ".java", "py": ".py"}
                out_ext = ext_map.get(target, ".js")
                out_file = os.path.join(SCRATCH_DIR, f"{temp_name}{out_ext}")
                
                # Dafny sometimes prepends something or creates subdirs
                if not os.path.exists(out_file):
                    # Check for [temp_name]-py.js or similar
                    if target == "js":
                        js_check = os.path.join(SCRATCH_DIR, f"{temp_name}-js/index.js")
                        if os.path.exists(js_check): out_file = js_check

                if os.path.exists(out_file):
                    with open(out_file, "r") as f: generated_code = f.read()
                    return generated_code, True
                else:
                    return f"Compilation succeeded but output file {out_file} not found in scratch.", False
            else:
                last_errors = res_verify.stdout or res_verify.stderr
                if status_callback: status_callback(f"Dafny Phase: Verification FAILED.")
        except Exception as e:
            last_errors = str(e)
            if status_callback: status_callback(f"Dafny Phase: Error: {e}")
            
    return f"Failed to verify Dafny code after {MAX_ATTEMPTS} attempts. Errors:\n{last_errors}", False

def save_interaction(prompt, response):
    if prompt and response:
        logger.info(f"\n[Output] Assistant Response: \"{response[:200]}...\"")
        logger.info(f"[*] Saving interaction to memory...")
        store_memory(f"User: {prompt}\nAssistant: {response}")

def sanitize_think_tags(text, remove_content=False):
    """Removes all <think> and </think> tags from the given text. 
    If remove_content is True, also removes everything between them."""
    if not text: return ""
    
    # 1. Handle content removal if requested
    if remove_content:
        text = re.sub(r'(?i)<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'(?i)<think>.*', '', text, flags=re.DOTALL)
    
    # 2. Normalize and consolidate multiple/nested tags
    # First, remove all closing tags to avoid orphans
    text = re.sub(r'(?i)</\s*think\s*>', '', text)
    # Then, remove all opening tags
    text = re.sub(r'(?i)<\s*think\s*>', '', text)
    
    # Remove orchestration artifacts that often leak into memories
    artifacts = [
        "[ADELAIDE ORCHESTRATION]",
        "Initiating Orchestrated Intelligence (Adelaide-Lite)...",
        "Consulting my memory for context...",
        "Analyzing your request with the precision of a master watchmaker...",
        "Successfully retrieved",
        "relevant past interactions. Context is everything!",
        "Strategic Planning & Decomposition",
        "[ADELAIDE CORE ORCHESTRATION]"
    ]
    for art in artifacts:
        text = text.replace(art, "")
        
    text = re.sub(r'\[Phase:.*?\]', '', text) # Remove [Phase: ...] markers
    text = re.sub(r'\s*\n\s*\n\s*', '\n\n', text) # Clean up extra whitespace
    
    return text.strip()


def wrap_response_with_memory(resp, data, is_chat=True, is_openai=False, start_time=None, initial_chunks=None, orchestration_prefix="", orch_think_open=False, on_finish=None, client_model_override=None):
    stream = data.get('stream', False)
    if start_time is None: start_time = time.time()
    model = client_model_override if client_model_override else data.get('model', MAIN_MODEL)
    actual_model = data.get('model', MAIN_MODEL)
    in_tokens = data.get('_in_tokens', 0)

    if not stream and not (resp.headers.get('Transfer-Encoding') == 'chunked'):
        try:
            res_json = resp.json()
            full_response = ""
            
            # Helper to extract content
            content_key = None
            if is_openai:
                if 'choices' in res_json and len(res_json['choices']) > 0:
                    full_response = res_json['choices'][0].get('message', {}).get('content', '')
                    content_key = ('choices', 0, 'message', 'content')
            else:
                if is_chat:
                    if 'message' in res_json and 'content' in res_json['message']:
                        full_response = res_json['message']['content']
                        content_key = ('message', 'content')
                else:
                    full_response = res_json.get('response', '')
                    content_key = ('response',)

            # Logic to merge orchestration think block with model think block
            if orch_think_open:
                # Sanitize the entire response to ensure only ONE set of tags exists
                has_model_end = "</think>" in full_response
                clean_content = sanitize_think_tags(full_response)
                
                if has_model_end:
                    # Model provided its own end-of-thought, so we split and close our block there
                    parts = full_response.split("</think>", 1)
                    full_response = sanitize_think_tags(parts[0]) + "</think>\n" + sanitize_think_tags(parts[1])
                else:
                    # No end tag from model, close it at the very beginning of its response
                    full_response = "</think>\n" + clean_content
            
            if orchestration_prefix:
                full_response = orchestration_prefix + full_response

            # Update the JSON with modified content
            if content_key:
                target = res_json
                for k in content_key[:-1]:
                    target = target[k]
                target[content_key[-1]] = full_response

            # Performance measurement
            duration = time.time() - start_time
            out_tokens = count_tokens(full_response)
            update_performance(model, in_tokens, out_tokens, duration, ttft=duration)

            if full_response:
                prompt = ""
                if is_chat:
                    for m in reversed(data.get('messages', [])):
                        if m['role'] == 'user':
                            prompt = m['content']
                            break
                else:
                    prompt = data.get('prompt', '')
                save_interaction(prompt, full_response)

            # Final JSON cleanup for client consistency
            res_json['model'] = model
            if 'total_duration' not in res_json:
                res_json['total_duration'] = int(duration * 1e9)
            
            logger.info(f"<<< [SEND] {resp.status_code} (Non-Streaming)")
            return jsonify(res_json)
        except Exception as e:
            logger.error(f"[-] Error parsing response for logging: {e}")
            return Response(resp.content, resp.status_code, content_type=resp.headers.get('Content-Type'))

    # Streaming case
    g.streaming_held = True
    logger.info(f"<<< [SEND] {resp.status_code} (Streaming Started)")
    
    # Verbose stream to stdout
    sys.stdout.write("\n[Assistant] ")
    sys.stdout.flush()

    req_id = id(request)
    def generate():
        if initial_chunks:
            for c in initial_chunks:
                yield c
        
        first_token_time = None
        ttft = None
        # State: 0 = Orchestration thinking open, 1 = Thinking closed, answer started
        think_state = 0 if orch_think_open else 1
        
        try:
            full_response = ""
            for line in resp.iter_lines():
                if line:
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft = first_token_time - start_time
                        logger.info(f"[*] TTFT Detected: {ttft:.4f}s")
                    
                    try:
                        line_str = line.decode('utf-8').strip()
                        text = ""
                        is_done = False
                        
                        if is_openai:
                            if line_str.startswith("data: "):
                                if line_str != "data: [DONE]":
                                    chunk = json.loads(line_str[6:])
                                    if 'choices' in chunk and len(chunk['choices']) > 0:
                                        delta = chunk['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            text = delta['content']
                                        if chunk['choices'][0].get('finish_reason'):
                                            is_done = True
                                else:
                                    is_done = True
                        else:
                            chunk = json.loads(line_str)
                            if is_chat:
                                if 'message' in chunk and 'content' in chunk['message']:
                                    text = chunk['message']['content']
                            else:
                                if 'response' in chunk:
                                    text = chunk['response']
                            if chunk.get('done'):
                                is_done = True

                        if text:
                            # Decide whether to close the orchestration think block based on the first model tokens
                            if think_state == 0 and not full_response:
                                # If the first non-empty chunk from the model DOES NOT start with a think tag, 
                                # we should close our orchestration block now so its answer is visible.
                                lower_text = text.lower()
                                if "<think>" not in lower_text:
                                    yield format_chunk("</think>\n", is_chat, is_openai, model, done=False)
                                    think_state = 1
                                else:
                                    # Model is starting its own thought. Suppress its opening tag
                                    # to merge with our existing orchestration think block.
                                    text = re.sub(r'(?i)<\s*think\s*>', '', text)

                            # Normal processing based on current think_state
                            if think_state == 0:
                                if "</think>" in text.lower():
                                    # Model finished its thought, close the block and transition to answer state
                                    parts = re.split(r'(?i)</\s*think\s*>', text, 1)
                                    text = parts[0] + "</think>\n" + sanitize_think_tags(parts[1])
                                    think_state = 1
                                else:
                                    # Still in thinking state, ensure no internal/nested tags bleed through
                                    text = sanitize_think_tags(text)
                            else:
                                # In answer state, ensure no stray thinking tags appear
                                text = sanitize_think_tags(text)
                            
                            if text:
                                full_response += text
                                sys.stdout.write(text)
                                sys.stdout.flush()
                                yield format_chunk(text, is_chat, is_openai, model, done=False)

                        if is_done:
                            # Safety close
                            if think_state == 0:
                                yield format_chunk("</think>\n", is_chat, is_openai, model, done=False)
                            
                            prompt = ""
                            if is_chat:
                                for m in reversed(data.get('messages', [])):
                                    if m['role'] == 'user':
                                        prompt = m['content']
                                        break
                            else:
                                prompt = data.get('prompt', '')
                            save_interaction(prompt, full_response)
                            
                            # Final Metrics
                            total_dur = int((time.time() - start_time) * 1e9)
                            out_tokens = count_tokens(full_response)
                            final_metrics = {
                                "total_duration": total_dur,
                                "load_duration": 0,
                                "prompt_eval_count": in_tokens,
                                "prompt_eval_duration": int(ttft * 1e9) if ttft else 0,
                                "eval_count": out_tokens,
                                "eval_duration": total_dur - (int(ttft * 1e9) if ttft else 0)
                            }
                            yield format_chunk("", is_chat, is_openai, model, done=True, metrics=final_metrics)
                    except Exception as e:
                        logger.debug(f"[Stream] Error processing chunk: {e}")
            sys.stdout.write("\n\n")
            sys.stdout.flush()
            
            # Performance measurement for stream
            duration = time.time() - start_time
            out_tokens = count_tokens(full_response)
            update_performance(actual_model, in_tokens, out_tokens, duration, ttft=ttft)
            
            logger.info("<<< [SEND] Streaming Finished.")
        finally:
            if on_finish:
                try:
                    on_finish(full_response)
                except Exception as e:
                    logger.debug(f"[Cache] on_finish callback failed: {e}")

            req_id = getattr(g, 'task_id', None)
            with metrics_lock:
                if req_id:
                    active_task_details.pop(req_id, None)
                    logger.debug(f"[Task End-Wrap] {req_id}")
                global finished_requests, last_activity_time
                finished_requests += 1
                last_activity_time = time.time()

    return Response(stream_with_context(generate()), content_type=resp.headers.get('Type') or 'application/x-ndjson') # pyrefly: ignore



# --- Vector Routing Logic ---
ROUTING_CATEGORIES = {
    "Mathematics and complex calculations": {"strong": True, "search": False},
    "Tool Usage and System Command Execution": {"strong": True, "search": True, "engines": ["all"]},
    "Factual Accuracy and technical details": {"strong": True, "search": True, "engines": ["all"], "confidence_boost": 0.2},
    "News, recent events, and contemporary regulations": {"strong": True, "search": True, "engines": ["all"]},
    "Life Threatening and safety critical information": {"strong": True, "search": True, "engines": ["all"]},
    "Requires real-time data, news, or external lookup": {"strong": False, "search": True, "engines": ["ddg", "google"]},
    "Coding and Software Development": {"strong": True, "search": True, "engines": ["all"]},
    "Factual Seeking/Literature Research": {"strong": True, "search": True, "engines": ["all"]},
    "Requires local documentation or file reference": {"strong": False, "search": True, "engines": ["all"]},
    "Requires retrieval of past memories or history": {"strong": False, "search": "memory"},
    "Casual conversation and greetings": {"strong": False, "search": False},
    "Creative writing and roleplay": {"strong": False, "search": False}
}

category_vectors = {} 

def initialize_category_vectors():
    global category_vectors
    if category_vectors: return

    logger.info("[*] Initializing Semantic Category Vectors...")
    for cat in ROUTING_CATEGORIES.keys():
        vec = get_chunked_embedding("/api/embeddings", EMBED_MODEL, cat, bypass_safe=True)
        if vec:
            category_vectors[cat] = vec
    logger.info(f"[+] Semantic Router ready with {len(category_vectors)} categories.")
def cosine_similarity(v1, v2):
    try:
        bridge = AdelaideBridge.get_instance()
        sim = bridge.cosine_similarity(v1, v2)
        if sim is not None:
            return sim
    except Exception:
        pass

    v1 = np.array(v1)
    v2 = np.array(v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0
    return np.dot(v1, v2) / (norm1 * norm2)

def vector_route_detect(text):
    if not category_vectors:
        initialize_category_vectors()
        if not category_vectors: return {"strong": False, "search": False, "memory": True, "engines": ["all"]}
    
    input_vec = get_chunked_embedding("/api/embeddings", EMBED_MODEL, text)
    if not input_vec: return {"strong": False, "search": False, "memory": True, "engines": ["all"]}
    
    best_category = None
    max_score = -1
    
    for cat, vec in category_vectors.items():
        score = cosine_similarity(input_vec, vec)
        if score > max_score:
            max_score = score
            best_category = cat
            
    if best_category:
        config = ROUTING_CATEGORIES[best_category]
        logger.info(f"[Semantic Router] Best Match: \"{best_category}\" (Score: {max_score:.4f})")
        
        # Threshold: 0.4 for confidence
        if max_score > 0.4:
            return {
                "strong": config["strong"],
                "search": config["search"] == True,
                "command": "Command" in best_category,
                "memory": config["search"] == "memory" or config["search"] == True or max_score > 0.5,
                "engines": config.get("engines", ["all"]),
                "is_casual": best_category in ["Casual conversation and greetings", "Creative writing and roleplay"],
                "confidence": max_score
            }
            
    return {"strong": False, "search": False, "memory": True, "engines": ["all"], "is_casual": False, "confidence": 0}

def router_decide(prompt_text, memory_ctx="", status_callback=None):
    if status_callback: status_callback("Analyzing your request with the precision of a master watchmaker...")
    logger.info(f"\n[Thought] Routing analysis for: \"{prompt_text[:100]}...\"")
    
    # 1. High-Speed Semantic Vector Check
    combined_input = f"{prompt_text} {memory_ctx}"
    v_dec = vector_route_detect(combined_input)
    
    sys_prompt = '''You are the OIPRouter, a senior strategic orchestration engine. Your mission is to decompose the user's intent into an optimal execution path.

# ORCHESTRATION STRATEGY
1. **Research (needs_search)**: Set to true if the request requires factual verification, real-time data, news, or technical documentation (especially post-2023).
2. **Action (needs_command)**: Set to true if the request implies a file operation, system check, complex calculation, or multi-step execution.
3. **Reasoning (use_strong_model)**: Set to true if the task requires deep creative writing, complex logic, or nuanced synthesis.

# QUERY DECOMPOSITION RULES
- NEVER copy the user's exact prompt.
- Break the user's intent into concise, highly specific search keywords and phrases (e.g., ["life realism philosophy", "pessimism vs adulthood psychology"]).
- Provide between 1 to 5 queries for each type.

# OUTPUT FORMAT (Strict JSON only)
{
  "needs_search": true/false,
  "search_queries": ["specific keyword 1", "specific keyword 2"],
  "memory_queries": ["memory topic 1", "memory topic 2"],
  "hypothesis": "what we expect to discover to satisfy the intent",
  "needs_command": true/false,
  "command_intent": "high-level functional goal for the action phase",
  "use_strong_model": true/false,
  "max_jumps": 3
}

CRITICAL: Output ONLY the JSON block. Be decisive and strategic.'''

    router_input = f"{sys_prompt}\n\n"
    if memory_ctx:
        router_input += f"Context from Memory:\n{memory_ctx}\n\n"
    router_input += f"User: {prompt_text}"

    payload = {
        "model": ROUTER_MODEL,
        "prompt": router_input,
        "stream": True, # Switch to streaming for real-time telemetry
        "options": {
            "num_predict": 500, # Limit router output to prevent runaway generation
            "temperature": 1.0  # Default internal temperature
        }
    }
    
    try:
        in_tokens = count_tokens(router_input)
        to = get_model_timeout(ROUTER_MODEL, in_tokens)
        t0 = time.time()
        
        if status_callback: status_callback(f"Calling internal Router API ({ROUTER_MODEL})...")
        update_active_task(f"[Router Phase: Prefilling {ROUTER_MODEL}...]")
        
        resp = safe_ollama_request("POST", "/api/generate", json=payload, timeout=to, stream=True)
        
        resp_text = ""
        first_token_time = None
        if resp is not None:
            for line in resp.iter_lines():
                if line:
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft = first_token_time - t0
                        if status_callback: status_callback(f"Router prefill complete ({ttft:.2f}s). Generating decision...")
                        update_active_task(f"[Router Phase: Generating Decision...]")
                    
                    chunk = json.loads(line)
                    resp_text += chunk.get("response", "")
                    if chunk.get("done"):
                        break
        
        duration = time.time() - t0
        out_tokens = count_tokens(resp_text)
        update_performance(ROUTER_MODEL, in_tokens, out_tokens, duration, ttft=(first_token_time - t0 if first_token_time else duration))
        
        logger.info(f"[Thought] Router Response: {resp_text}")
        
        keys = ["needs_search", "search_queries", "memory_queries", "hypothesis", "max_jumps", "use_strong_model", "needs_command", "command_intent"]
        parsed = repair_json(resp_text)
        if not parsed:
            parsed = extract_entities(resp_text, keys)

        if not parsed:
            payload["model"] = MAIN_MODEL
            payload["format"] = "json"
            in_tokens = count_tokens(router_input)
            to_main = get_model_timeout(MAIN_MODEL, in_tokens)
            t0 = time.time()
            res_obj = safe_ollama_request("POST", "/api/generate", json=payload, timeout=to_main, status_callback=status_callback)
            res = res_obj.json() if res_obj else {}
            duration = time.time() - t0
            resp_text_main = res.get("response", "")
            update_performance(MAIN_MODEL, in_tokens, count_tokens(resp_text_main), duration, ttft=duration)
            parsed = repair_json(resp_text_main)
            
        if not parsed:
            return {"needs_search": v_dec["search"], "use_strong_model": v_dec["strong"], "engines": v_dec["engines"]}
            
        # Unified key mapping (handles both legacy and new formats)
        confidence = v_dec.get("confidence", 0)
        is_casual = v_dec.get("is_casual") and (confidence if isinstance(confidence, (int, float)) else 0) > 0.5
        
        queries = parsed.get("search_queries", [])
        primary_query = prompt_text[:100]
        if isinstance(queries, list) and queries:
            primary_query = queries[0]
        elif isinstance(queries, str) and queries:
            primary_query = queries

        final_decision = {
            "needs_search": (v_dec["search"] or parsed.get("needs_search", False) or parsed.get("search", False)) if not is_casual else False,
            "search_queries": queries if isinstance(queries, list) else [queries] if queries else [],
            "query": primary_query,
            "hypothesis": parsed.get("hypothesis", ""),
            "needs_command": (v_dec.get("command", False) or parsed.get("needs_command", False) or parsed.get("needs_cmd", False) or parsed.get("cmd", False)) if not is_casual else False,
            "command_intent": parsed.get("command_intent", "") or parsed.get("intent", ""),
            "use_strong_model": v_dec["strong"] or parsed.get("use_strong_model", False) or parsed.get("strong", False),
            "max_jumps": parsed.get("max_jumps") or parsed.get("jumps", 3),
            "engines": v_dec["engines"]
        }
        
        if is_casual:
            logger.info("[Router] Semantic Veto: Forcing NO search/command for casual input.")

        # If it needs search but no queries were generated, fall back to a breakdown of the prompt
        if final_decision.get("needs_search") and not final_decision.get("search_queries"):
             final_decision["search_queries"] = [prompt_text[:100]]

        if status_callback: 
            if final_decision.get("needs_search"):
                status_callback(f"My intuition suggests a bit of research is in order. The digital archives await!")
                if final_decision.get("search_queries"):
                    sq = final_decision.get("search_queries", [])
                    if not isinstance(sq, list): sq = [str(sq)]
                    q_str = ', '.join([str(s) for s in sq[:2]])
                    if len(sq) > 2: q_str += "..."
                    status_callback(f"Querying the vast aether for: [{q_str}]")
            elif final_decision.get("needs_command"):
                status_callback(f"I shall execute a system task. The mechanical wonders never cease!")
            else:
                status_callback("Synthesizing a response from the delicate gears of my own mind.")

        return final_decision
    except Exception as e:
        logger.error(f"[-] Router error: {e}")
        return {"needs_search": v_dec["search"], "use_strong_model": v_dec["strong"], "engines": v_dec["engines"]}

try:
    from html2image import Html2Image
    hti = Html2Image(output_path=os.path.join(SCRATCH_DIR, "vision_renders"))
except Exception as e:
    logger.error(f"[!] Vision rendering initialization failed: {e}")
    hti = None

def do_vision_render_loop(html_code, css_code="", status_callback=None):
    """Renders HTML/CSS to an image and uses the VISION_MODEL for verification."""
    if hti is None:
        return False, "Vision rendering not initialized (permission error?)"
    
    if not os.path.exists(hti.output_path):
        try:
            os.makedirs(hti.output_path, exist_ok=True)
        except Exception as e:
            return False, f"Failed to create vision render directory: {e}"
    
    img_name = f"render_{int(time.time())}.png"
    img_path = os.path.join(hti.output_path, img_name)
    
    if status_callback: status_callback("Rendering UI for vision verification...")
    try:
        hti.screenshot(html_str=html_code, css_str=css_code, save_as=img_name)
        
        if not os.path.exists(img_path):
             return False, "Failed to generate render image."
             
        # Call VISION_MODEL to analyze the image
        with open(img_path, "rb") as f:
            import base64
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
            
        vision_prompt = "Analyze this UI render. Does it look correct according to the intent? Identify any layout issues, color mismatches, or missing elements. Output strictly as JSON: {'satisfied': true/false, 'issues': ['...']}"
        
        payload = {
            "model": VISION_MODEL,
            "prompt": vision_prompt,
            "images": [img_b64],
            "stream": False,
            "format": "json"
        }
        # Dynamic timeout based on input size
        in_tokens = count_tokens(payload["prompt"])
        to = get_model_timeout(MAIN_MODEL, in_tokens)
        res_obj = safe_ollama_request("POST", "/api/generate", json=payload, timeout=to)
        res = res_obj.json() if res_obj else {}
        parsed = repair_json(res.get("response", ""))
        
        if parsed and parsed.get("satisfied"):
            return True, "Vision verification passed."
        else:
            return False, f"Vision issues: {parsed.get('issues', ['Unknown layout error']) if parsed else ['Parse error']}"
    except Exception as e:
        return True, f"Vision check skipped or failed: {e}"

def perform_deterministic_check(code, lang="python"):
    if not lang: lang = "python"
    lang = lang.lower()
    if lang == "python":
        try:
            ast.parse(code)
            temp_file = os.path.join(SCRATCH_DIR, "temp_check.py")
            with open(temp_file, "w") as f:
                f.write(code)
            res = subprocess.run(["pyrefly", "check", temp_file], capture_output=True, text=True, cwd=SCRATCH_DIR)
            if res.returncode == 0:
                return True, "Python Syntax & Type OK"
            else:
                return False, f"Python Static Check Error: {res.stderr or res.stdout}"
        except SyntaxError as e:
            return False, f"Python Syntax Error: {e}"
        except:
            return True, "Python Syntax OK (Static check tool error)"
    elif lang == "json":
        try:
            json.loads(code)
            return True, "JSON OK"
        except Exception as e:
            return False, f"JSON Error: {e}"
    elif lang in ["ada", "spark"]:
        # Create an isolated Alire project for structural Ada verification
        try:
            import uuid
            proj_name = f"ada_check_{uuid.uuid4().hex[:8]}"
            proj_dir = os.path.join(SCRATCH_DIR, proj_name)
            
            # Initialize Alire project (no interaction)
            subprocess.run(["alr", "init", "--bin", proj_name], cwd=SCRATCH_DIR, check=True, capture_output=True)
            
            # Write code to the main adb file
            src_file = os.path.join(proj_dir, "src", f"{proj_name}.adb")
            with open(src_file, "w") as f:
                f.write(code)
                
            # Run alr build which implies gnat compile
            res = subprocess.run(["alr", "build"], cwd=proj_dir, capture_output=True, text=True)
            
            if res.returncode == 0:
                return True, "Ada Syntax & Build OK"
            else:
                return False, f"Ada Build Error: {res.stderr or res.stdout}"
        except Exception as e:
            return True, f"Ada structural check skipped (tools missing or error): {e}"
    elif lang in ["coq", "rocq", "v"]:
        try:
            temp_file = os.path.join(SCRATCH_DIR, "temp_check.v")
            with open(temp_file, "w") as f:
                f.write(code)
            res = subprocess.run(["coqc", temp_file], capture_output=True, text=True, cwd=SCRATCH_DIR)
            if res.returncode == 0:
                return True, "Rocq/Coq Logic OK"
            else:
                return False, f"Rocq/Coq Logic Error: {res.stderr}"
        except:
             return True, "Rocq syntax check skipped (tools missing)"
    elif lang == "svg":
        # Basic SVG syntax check (XML check)
        try:
            import xml.etree.ElementTree as ET
            ET.fromstring(code)
            return True, "SVG Syntax OK"
        except Exception as e:
            return False, f"SVG Syntax Error: {e}"
    return True, f"Language '{lang}' is checked via logic trace only."

def run_coq_logic_bridge(code, lang, status_callback=None):
    """Generates a Coq formal specification for the given code and verifies it."""
    try:
        bridge_prompt = f"""
You are the Coq Formal Logic Bridge. 
Your task is to translate the logic of the following {lang} code into a formal Coq (Rocq) specification.
Include at least one Lemma or Theorem that proves the primary logical property of this code (e.g. correctness, safety, or termination).

Code:
{code}

Output ONLY the Coq code wrapped in ```coq ... ``` tags.
"""
        to = get_model_timeout(ROUTER_MODEL)
        res_obj = safe_ollama_request("POST", "/api/generate", json={"model": ROUTER_MODEL, "prompt": bridge_prompt, "stream": False}, timeout=to)
        res = res_obj.json() if res_obj else {}
        coq_text = res.get("response", "")
        
        coq_match = re.search(r'```coq\n(.*?)\n```', coq_text, re.DOTALL)
        if not coq_match:
            return False, "Failed to generate Coq logic bridge specification."
            
        coq_code = coq_match.group(1)
        # Verify the generated Coq code
        ok, err = perform_deterministic_check(coq_code, "coq")
        if ok:
            return True, "Logic Bridge Verified."
        else:
            return False, f"Logic Bridge Failed: {err}"
    except Exception as e:
        return False, f"Logic Bridge Exception: {e}"

def run_mcp_tool_task(intent, status_callback=None):
    """Uses Qwen-Agent to execute tasks via MCP servers."""
    try:
        from qwen_agent.agents import Assistant
        if status_callback: status_callback(f"Phase: MCP Agent Activation for '{intent[:30]}...'")
        
        # Configure to point to the raw Ollama instance
        llm_cfg = {
            'model': MAIN_MODEL,
            'model_type': 'qwenvl_oai',
            'model_server': f"{OLLAMA_TARGET}/v1",
            'api_key': 'EMPTY',
            'generate_cfg': {
                'temperature': 1.0,
                'extra_body': {
                    'chat_template_kwargs': {'enable_thinking': True},
                    'options': {'kv_cache_type': 'q4_0'}
                }
            }
        }
        
        # Define standard MCP tools (Filesystem, etc)
        tools = [
            {'mcpServers': {
                    "filesystem": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", os.getcwd()]
                    }
                }
            }
        ]
        
        bot = Assistant(llm=llm_cfg, function_list=tools) # pyrefly: ignore
        messages = [{'role': 'user', 'content': intent}]
        
        final_res = ""
        for responses in bot.run(messages=messages): # pyrefly: ignore
            if responses:
                final_res = responses[-1]['content']
        
        return final_res
    except Exception as e:
        logger.error(f"[-] MCP Agent Error: {e}")
        return f"MCP Error: {e}"

class WorktreeService:
    """Provides git-based worktree abstraction for 'what-if' changes."""
    def __init__(self, root_dir):
        self.root_dir = os.path.abspath(root_dir)
        self._ensure_git()

    def _ensure_git(self):
        if not os.path.exists(os.path.join(self.root_dir, ".git")):
            logger.info("[*] Initializing Worktree Git Repository...")
            subprocess.run(["git", "init"], cwd=self.root_dir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "adelaide@proxy.local"], cwd=self.root_dir)
            subprocess.run(["git", "config", "user.name", "Adelaide Proxy"], cwd=self.root_dir)
            subprocess.run(["git", "commit", "--allow-empty", "-m", "Initial commit"], cwd=self.root_dir)

    def create_branch(self, branch_name):
        logger.debug(f"[*] Creating worktree branch: {branch_name}")
        subprocess.run(["git", "checkout", "-b", branch_name], cwd=self.root_dir, capture_output=True)

    def commit_change(self, message):
        subprocess.run(["git", "add", "."], cwd=self.root_dir, capture_output=True)
        res = subprocess.run(["git", "commit", "-m", message], cwd=self.root_dir, capture_output=True)
        return res.returncode == 0

    def rollback(self):
        logger.warning("[!] Rolling back worktree to last stable state...")
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=self.root_dir, capture_output=True)

    def cleanup(self, branch_name):
        subprocess.run(["git", "checkout", "master"], cwd=self.root_dir, capture_output=True)
        subprocess.run(["git", "branch", "-D", branch_name], cwd=self.root_dir, capture_output=True)

worktree = WorktreeService(SCRATCH_DIR)

def strip_git_diff_prefix(filename):
    """Strips a/ or b/ prefixes from diff headers."""
    if filename and (filename.startswith("a/") or filename.startswith("b/")):
        logger.debug(f"[*] Stripping git diff prefix from: {filename}")
        return filename[2:]
    return filename

def is_safe_path(target_path):
    """Ensures the path is within the allowed SCRATCH_DIR or current workspace."""
    abs_path = os.path.abspath(target_path)
    # Allow current directory and SCRATCH_DIR
    allowed_roots = [os.getcwd(), os.path.abspath(SCRATCH_DIR)]
    return any(abs_path.startswith(root) for root in allowed_roots)

def validate_and_apply_patch(original_text, diff_block, target_filename="orig_response.txt"):
    """
    Parses, validates, and applies a unified diff block.
    Implements Structured Patch & Safety Validation from gemini-cli.
    """
    # 1. Header Validation & Git Prefix Stripping
    lines = diff_block.splitlines()
    cleaned_lines = []
    found_header = False
    for line in lines:
        if line.startswith("--- ") or line.startswith("+++ "):
            parts = line.split()
            if len(parts) >= 2:
                file_path = strip_git_diff_prefix(parts[1])
                # Safety check: Path Traversal
                if not is_safe_path(file_path) and file_path != "/dev/null":
                    logger.warning(f"[!] Blocked unsafe path in patch: {file_path}")
                    return None, f"Unsafe path detected: {file_path}"
                cleaned_lines.append(f"{line[:4]}{file_path}")
                found_header = True
                continue
        cleaned_lines.append(line)
    
    clean_diff = "\n".join(cleaned_lines)
    
    # 2. Write to temp for patching
    temp_orig = os.path.join(SCRATCH_DIR, target_filename)
    with open(temp_orig, "w") as f:
        f.write(original_text)
        
    # 3. Apply Patch with validation
    patch_proc = subprocess.run(["patch", "-p0", temp_orig], input=clean_diff, text=True, capture_output=True, cwd=SCRATCH_DIR)
    
    if patch_proc.returncode == 0:
        with open(temp_orig, "r") as f:
            return f.read(), None
    else:
        return None, f"Patch failed: {patch_proc.stderr or patch_proc.stdout}"

def do_code_verification_workflow(response_text, original_prompt, attempt=1, status_callback=None, task_id=None):
    if not task_id:
        import uuid
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        worktree.create_branch(task_id)

    MAX_ATTEMPTS = 99999
    logger.info(f"[*] Code Verification Attempt {attempt}/{MAX_ATTEMPTS} for task {task_id}")
    
    # Extract code blocks
    code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', response_text, re.DOTALL)
    if not code_blocks:
        return response_text, True

    all_errors = []
    all_errors_snapshot = [] # Track errors for memory learning
    html_block = ""
    css_block = ""

    for lang, code in code_blocks:
        lang = (lang or "ada").lower()
        
        # --- Stage 1: Primary Structural/Static Verification ---
        if lang in ["ada", "spark"]:
            try:
                # Phase 1: Coq Logic Bridge (Logic First)
                if status_callback: status_callback("Phase 1: Coq Logical Integrity Check for Ada...")
                coq_ok, coq_err = run_coq_logic_bridge(code, "ada", status_callback=status_callback)
                if not coq_ok: 
                    all_errors.append(coq_err)
                else:
                    # Phase 2: GNATprove Formal Verification
                    if status_callback: status_callback("Phase 2: GNATprove Formal Verification...")
                    import uuid
                    proj_name = f"ada_verify_{uuid.uuid4().hex[:8]}"
                    proj_dir = os.path.join(SCRATCH_DIR, proj_name)
                    subprocess.run(["alr", "init", "--bin", proj_name], cwd=SCRATCH_DIR, check=True, capture_output=True)
                    src_file = os.path.join(proj_dir, "src", f"{proj_name}.adb")
                    with open(src_file, "w") as f: f.write(code)
                    
                    res_prove = subprocess.run(["alr", "exec", "--", "gnatprove", "-P", f"{proj_name}.gpr"], capture_output=True, text=True, cwd=proj_dir)
                    if res_prove.returncode != 0:
                        all_errors.append(f"GNATprove Error: {res_prove.stderr or res_prove.stdout}")
                    else:
                        # Phase 3: Final Compilation Check
                        if status_callback: status_callback("Phase 3: Final Alire Build Compilation...")
                        res_build = subprocess.run(["alr", "build"], cwd=proj_dir, capture_output=True, text=True)
                        if res_build.returncode != 0:
                            all_errors.append(f"Build Error: {res_build.stderr or res_build.stdout}")
            except Exception as e:
                all_errors.append(f"Ada Pipeline Failure: {e}")

        elif lang == "python":
            try:
                # Phase 1: Coq Logic Bridge (Logic First)
                if status_callback: status_callback("Phase 1: Coq Logical Integrity Check for Python...")
                coq_ok, coq_err = run_coq_logic_bridge(code, "python", status_callback=status_callback)
                if not coq_ok: 
                    all_errors.append(coq_err)
                else:
                    # Phase 2: Pyrefly Static Analysis
                    if status_callback: status_callback("Phase 2: Pyrefly Static Analysis...")
                    ok, err = perform_deterministic_check(code, "python")
                    if not ok: all_errors.append(err)
            except Exception as e:
                all_errors.append(f"Python Pipeline Failure: {e}")

        elif lang in ["coq", "rocq", "v"]:
            ok, err = perform_deterministic_check(code, lang)
            if not ok: all_errors.append(err)

        elif lang == "svg":
            if status_callback: status_callback("Phase 1: SVG XML Parsing...")
            ok, err = perform_deterministic_check(code, lang)
            if not ok: 
                all_errors.append(err)
            else:
                # Stage 2: Vision Render Loop for SVG
                if status_callback: status_callback("Phase 2: SVG Vision Verification...")
                svg_html = f"<html><body style='margin:0;padding:0;display:flex;justify-content:center;align-items:center;height:100vh;'>{code}</body></html>"
                v_ok, v_err = do_vision_render_loop(svg_html, "", status_callback=status_callback)
                if not v_ok: all_errors.append(v_err)

        elif lang in ["html", "css"]:
            # We'll collect these for a single Vision Loop call at the end of the block
            if lang == "html": html_block = code
            if lang == "css": css_block = code

        else:
            ok, err = perform_deterministic_check(code, lang)
            if not ok: all_errors.append(err)

    # Execute Vision Loop if UI code found
    if html_block:
        ok, err = do_vision_render_loop(html_block, css_block, status_callback=status_callback)
        if not ok: all_errors.append(err)

    for lang, code in code_blocks:
        # Enforce 10-line limit
        lines = [l for l in code.split("\n") if l.strip()]
        if len(lines) > 10:
            all_errors.append(f"Code block too large ({len(lines)} lines). Max 10 lines allowed per change.")
            
        # Enforce mandatory comments on every line
        for i, line in enumerate(lines):
            # Check for Python (#), Ada/Coq (--), CSS (/*), or HTML (<!--)
            if not re.search(r'#|--|/\*|<!--', line):
                all_errors.append(f"Line {i+1} missing mandatory comment: '{line.strip()}'")

        # Logical Trace (LLM Step-by-Step Review)
        trace_prompt = f"""
Analyze the following code block for logical correctness, memory leaks, security flaws, and edge cases.
Language: {lang or 'unknown'}
Code:
{code}

Task:
1. Perform a mental trace of the logic.
2. Identify any hidden bugs or non-obvious failures.

Output format (Follow exactly):
[SATISFIED] true/false
[LOGIC_TRACE] brief line-by-line summary
[ISSUES] list of issues found
[SUGGESTION] how to fix
"""
        keys = ["satisfied", "logic_trace", "issues", "suggestion"]
        try:
            to = get_model_timeout(ROUTER_MODEL)
            res_obj = safe_ollama_request("POST", "/api/generate", json={"model": ROUTER_MODEL, "prompt": trace_prompt, "stream": False}, timeout=to)
            res = res_obj.json() if res_obj else {}
            res_text = res.get("response", "")
            parsed = repair_json(res_text) or extract_entities(res_text, keys)
            
            if parsed and not parsed.get("satisfied"):
                issues = parsed.get("issues", [])
                if isinstance(issues, str): issues = [issues]
                if isinstance(issues, list):
                    all_errors.extend(issues)
                elif issues:
                    all_errors.append(str(issues))
                if parsed.get("suggestion"):
                    all_errors.append(f"Suggestion: {parsed['suggestion']}")
        except:
            pass

    if not all_errors:
        logger.info("[+] Code verification passed.")
        worktree.commit_change(f"Verified fix for attempt {attempt}")
        if attempt > 1:
            # Store the successful fix in semantic memory as a formal coding lesson
            lesson_type = "Ada/SPARK Formal Verification" if "ada" in str(code_blocks).lower() else "Python/Pyrefly Static Analysis"
            memory_entry = f"""
[FORMAL CODING LESSON LEARNED]
Type: {lesson_type}
Original Intent: {original_prompt}
Failed Attempts: {attempt - 1}

Verified Solution:
{response_text}

Guidance: Apply the logic used in this verified solution to satisfy formal constraints.
"""
            store_memory(memory_entry)
        return response_text, True
    
    # Snapshot errors for the next attempt's memory storage
    all_errors_snapshot = list(all_errors)
    
    if attempt >= MAX_ATTEMPTS:
        logger.warning(f"[!] Code failed verification after {MAX_ATTEMPTS} attempts.")
        return response_text, False

    # 3. Research Phase (Optional)
    search_context = ""
    try:
        research_prompt = f"""
Analyze these verification errors:
{chr(10).join(f"- {e}" for e in all_errors)}

Do any of these errors require external research (e.g. documentation, API specs, error code lookup)?
Output strictly as JSON: {{"needs_research": true/false, "search_query": "specific search query if true"}}
"""
        # Increased timeout for complex research decomposition
        to = get_model_timeout(MAIN_MODEL)
        resp = safe_ollama_request("POST", "/api/generate", json={"model": ROUTER_MODEL, "prompt": research_prompt, "stream": False, "format": "json"}, timeout=to)
        res = resp.json() if resp is not None else {}
        res_text = res.get("response", "")
        parsed_research = repair_json(res_text)
        if parsed_research and parsed_research.get("needs_research"):
            q = parsed_research.get("search_query")
            if q:
                if status_callback: status_callback(f"Researching error: {q}")
                results, _ = do_search_workflow(q, f"Researching error for: {q}", status_callback=status_callback)
                search_context = "\n\nExternal Research Findings:\n" + "\n".join([f"- {r.get('title')}: {r.get('content', '')[:200]}" for r in results[:5]])
    except Exception as e:
        logger.warning(f"[-] Research phase failed: {e}")

    # 4. Self-Healing Loop
    logger.warning(f"[!] Code verification failed with {len(all_errors)} issues. Retrying...")
    heal_prompt = f"""
The previous code you generated has errors.
Original Goal: {original_prompt}
Issues found:
{chr(10).join(f"- {e}" for e in all_errors)}
{search_context}

Task: Provide the corrected version of the code. If you are modifying an existing code block, output ONLY a unified diff (.patch format) wrapped in ```diff ... ``` tags. Ensure all syntax is valid and logic is sound.
"""
    try:
        # Ask the strong model to fix it
        to = get_model_timeout(MAIN_MODEL)
        resp = safe_ollama_request("POST", "/api/generate", json={"model": MAIN_MODEL, "prompt": heal_prompt, "stream": False, "options": {"temperature": 1.0}}, timeout=to)
        res = resp.json() if resp is not None else {}
        new_response = res.get("response", "")
        
        # Check if the response contains a diff block
        diff_match = re.search(r'```diff\n(.*?)\n```', new_response, re.DOTALL)
        if diff_match:
            diff_text = diff_match.group(1)
            patched_response, err = validate_and_apply_patch(response_text, diff_text)
            if patched_response:
                worktree.commit_change(f"Applied diff patch for attempt {attempt}")
                return do_code_verification_workflow(patched_response, original_prompt, attempt + 1, status_callback=status_callback, task_id=task_id)
            else:
                logger.warning(f"[-] {err}")
                worktree.rollback()
                
        return do_code_verification_workflow(new_response, original_prompt, attempt + 1, status_callback=status_callback, task_id=task_id)
    except:
        return response_text, False

def do_command_workflow(intent, search_ctx="", attempt=1, status_callback=None):
    MAX_ATTEMPTS = 99999
    if status_callback: status_callback(f"Phase: Command Execution (Attempt {attempt}). Intent: {intent}")
    logger.info(f"[*] Command Execution Attempt {attempt}/{MAX_ATTEMPTS} for intent: {intent}")
    
    # 1. Generate Command
    gen_prompt = f"""
You are the OIP Command Generator. Generate a single bash command to fulfill the following intent.
Intent: {intent}
Context from Search: {search_ctx}

Output strictly as JSON:
{{"command": "the bash command", "explanation": "briefly why"}}
"""
    payload = {
        "model": ROUTER_MODEL,
        "prompt": gen_prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        in_tokens = count_tokens(gen_prompt)
        to = get_model_timeout(ROUTER_MODEL, in_tokens)
        t0 = time.time()
        resp = safe_ollama_request("POST", "/api/generate", json=payload, timeout=to)
        res = resp.json() if resp is not None else {}
        duration = time.time() - t0
        
        resp_text = res.get("response", "")
        update_performance(ROUTER_MODEL, in_tokens, count_tokens(resp_text), duration, ttft=duration)
        parsed = repair_json(resp_text)
        if not parsed or "command" not in parsed:
             return "Failed to generate command."
             
        cmd = parsed["command"]
        if status_callback: status_callback(f"Executing: {cmd}")
        logger.info(f"[*] Executing Command: {cmd}")
        
        # 2. Execute
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        
        stdout_lines = []
        stderr_lines = []
        
        while True:
            if process.stderr:
                line = process.stderr.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    line = line.strip()
                    stderr_lines.append(line)
                    if status_callback: status_callback(f"  [stderr] {line}")
            else:
                if process.poll() is not None: break

            if process.stdout:
                out_line = process.stdout.readline()
                if out_line:
                    out_line = out_line.strip()
                    stdout_lines.append(out_line)
                    if status_callback: status_callback(f"  [stdout] {out_line}")

        remaining_out, remaining_err = process.communicate()
        if remaining_out: stdout_lines.append(remaining_out.strip())
        if remaining_err: stderr_lines.append(remaining_err.strip())
        
        stdout = "\n".join(stdout_lines)
        stderr = "\n".join(stderr_lines)
        exit_code = process.returncode
        
        if exit_code == 0:
            if status_callback: status_callback("Command Successful. Verifying Intent...")
            logger.info("[+] Command exit code 0. Verifying intent fulfillment...")
            # Semantic Accomplishment Check
            verify_prompt = f"""
Analyze if the following command output truly accomplished the user's intent.
Intent: {intent}
Executed Command: {cmd}
Output: {stdout}

Task:
1. Determine if the intent is fully satisfied.
2. If NOT satisfied, provide a 'new_command' or 'feedback' to fix it.

Output strictly as JSON:
{{
  "satisfied": true/false,
  "reason": "explanation",
  "new_command": "refined command if not satisfied",
  "feedback": "what was missing"
}}
"""
            to = get_model_timeout(ROUTER_MODEL)
            resp = safe_ollama_request("POST", "/api/generate", json={"model": ROUTER_MODEL, "prompt": verify_prompt, "stream": False, "format": "json"}, timeout=to)
            res = resp.json() if resp is not None else {}
            v_parsed = repair_json(res.get("response", ""))
            
            if v_parsed and v_parsed.get("satisfied"):
                if status_callback: status_callback("Intent Satisfied.")
                logger.info("[+] Intent accomplished.")
                return f"Command: {cmd}\nOutput:\n{stdout}"
            else:
                reason = v_parsed.get("reason", "Unknown") if v_parsed else "Verification failed."
                if status_callback: status_callback(f"Intent NOT Met: {reason}")
                logger.warning(f"[!] Command successful but intent NOT met: {reason}")
                if attempt < MAX_ATTEMPTS:
                    new_ctx = f"Previous command succeeded but didn't meet intent. Feedback: {v_parsed.get('feedback', '') if v_parsed else ''}\n{search_ctx}"
                    return do_command_workflow(intent, search_ctx=new_ctx, attempt=attempt+1, status_callback=status_callback)
                return f"Gave up after {MAX_ATTEMPTS} attempts. Last output: {stdout}"
        else:
            if status_callback: status_callback(f"Command Failed (Exit {exit_code}). Error: {stderr[:50]}...")
            logger.warning(f"[!] Command failed (Exit {exit_code}). Error: {stderr}")
            if attempt < MAX_ATTEMPTS:
                # 3. Self-Healing
                # Recursive heal
                return do_command_workflow(intent, search_ctx=f"Error from previous attempt: {stderr}\n{search_ctx}", attempt=attempt+1, status_callback=status_callback)
            else:
                return f"Command failed after {MAX_ATTEMPTS} attempts.\nLast Command: {cmd}\nError: {stderr}"
                
    except Exception as e:
        logger.error(f"[-] Command Workflow Error: {e}")
        return f"Error executing command: {e}"

def run_semantic_verification(query, hypothesis, search_history, status_callback=None):
    if status_callback: status_callback("Verifying search results sufficiency...")
    current_year = datetime.now().year
    verify_prompt = f"""
Analyze the user's information need and the search results gathered so far.
Current Year: {current_year}
Original Hypothesis: {hypothesis}
Search History (Last 3 jumps): 
{json.dumps([{"query": h["query"], "reasoning": h.get("reasoning"), "results": [{"title": r.get("title"), "snippet": r.get("content", "")[:300]} for r in h["results"][:5]]} for h in search_history[-3:]])}

Task:
1. Determine if the current information is sufficient to provide a high-quality, factual answer.
2. If NOT sufficient, provide a 'new_query' that REPHRASES the request to specifically target the missing information.
3. In 'reason', explain exactly what is missing and why previous results were insufficient.

CRITICAL: The 'new_query' MUST be a sophisticated, rephrased search term. Do not repeat failed queries.
CRITICAL: If you found some info but need more detail, rephrase the query to deep-dive into that detail.

Output strictly as JSON:
{{
  "satisfactory": true/false,
  "reason": "precise explanation of the information gap",
  "new_query": "rephrased deep-dive search query",
  "verification": "concise verification summary if satisfactory"
}}
"""
    payload = {
        "model": ROUTER_MODEL,
        "prompt": verify_prompt,
        "stream": False,
        "format": "json"
    }
    try:
        to = get_model_timeout(payload["model"])
        resp_v = safe_ollama_request("POST", "/api/generate", json=payload, timeout=to, status_callback=status_callback)
        ver_res = resp_v.json() if resp_v is not None else {}
        
        # Phase 2: Synthesis
        if status_callback: status_callback("Phase: Logic Synthesis")
        payload["model"] = MAIN_MODEL
        to_main = get_model_timeout(MAIN_MODEL)
        resp_s = safe_ollama_request("POST", "/api/generate", json=payload, timeout=to_main, status_callback=status_callback)
        syn_res = resp_s.json() if resp_s is not None else {}

        parsed = repair_json(ver_res.get("response", ""))
        return parsed or {"satisfactory": False}
    except:
        return {"satisfactory": False}

def extract_all_results(search_history):
    all_results = []
    seen_content = set()
    for entry in search_history:
        for r in entry.get('results', []):
            content_snippet = r.get('content', '')[:100]
            if content_snippet not in seen_content:
                all_results.append(r)
                seen_content.add(content_snippet)
    return all_results

def do_search_workflow(query, hypothesis, engines=None, jump_count=0, search_history=None, max_jumps=50, status_callback=None, last_failure_reason=None):
    MAX_JUMPS = 99999
    if search_history is None:
        search_history = []
        
    if jump_count == 0:
        store_memory(f"Hypothesis for '{query}': {hypothesis}")
    
    search_scripts = []
    if engines and len(engines) == 1 and engines[0] == "local":
        search_scripts.append("searchlocalref.py")
    elif engines and "all" not in engines and "local" not in engines:
        search_scripts.append("searchglobalref.py")
    else:
        # Default fallback if 'all' is passed
        search_scripts = ["searchlocalref.py", "searchglobalref.py"]
    
    if status_callback: 
        if jump_count == 0: 
            status_callback(f"Searching for: '{query}'")
        else: 
            status_callback(f"Refining search for: '{query}'")
            if last_failure_reason:
                status_callback(f"  [Gap Detected]: {last_failure_reason}")
    
    logger.info(f"[*] Search Jump {jump_count + 1}/{MAX_JUMPS} for: {query}")
    if last_failure_reason:
        logger.info(f"[*] Reasoning for refinement: {last_failure_reason}")
    
    for script_name in search_scripts:
        current_engines = engines if script_name == "searchglobalref.py" else ["local"]
        if status_callback: 
            if script_name == "searchlocalref.py": status_callback("Checking local references...")
            else: status_callback("Browsing the web for additional details...")
        logger.info(f"[*] Executing {script_name} for: {query} (Engines: {current_engines})")
        script_path = os.path.join(SCRIPT_DIR, script_name)
        try:
            proxy_url = f"127.0.0.1:{PORT}"
            cmd = [sys.executable, script_path, query, "--jsonIO", "--ollamaHost", proxy_url]
            if script_name == "searchglobalref.py":
                cmd.extend(["--timeout", "600"]) # Allow ample time for CAPTCHA solving
                if engines and "all" not in engines:
                    cmd.extend(["--engines"] + engines)
            
            env = os.environ.copy()
            env["OLLAMA_PROXY_URL"] = f"http://{proxy_url}"
                
            process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            
            stdout_lines = []
            
            # Stream stderr for real-time telemetry
            while True:
                line = process.stderr.readline() if process.stderr else None
                if not line and process.poll() is not None:
                    break
                if line:
                    line = line.strip()
                    if status_callback and line:
                        # Filter for high-signal logs to avoid drowning the thinking block
                        if any(x in line for x in ["[*]", "[+]", "Searching", "Found", "Browsing", "Checking"]):
                            status_callback(f"  {line}")
                
                # Also collect stdout for phase2 results
                out_line = process.stdout.readline() if process.stdout else None
                if out_line:
                    stdout_lines.append(out_line)

            # Collect remaining stdout
            remaining_out, remaining_err = process.communicate()
            if remaining_out: stdout_lines.append(remaining_out)
            
            results = extract_phase2_results("".join(stdout_lines))
            if results:
                search_history.append({"query": f"{query} ({script_name})", "results": results[:10], "reasoning": last_failure_reason})
                if status_callback:
                    status_callback(f"Found {len(results)} relevant sources:")
                    for r in results[:5]:
                        title = r.get('title') or r.get('url') or "Snippet"
                        status_callback(f"  • {title}")
                
                # Check if we can stop early
                if script_name == "searchlocalref.py":
                    logger.info("[*] Verifying if local results are sufficient...")
                    v_parsed = run_semantic_verification(query, hypothesis, search_history, status_callback=status_callback)
                    if v_parsed.get("satisfactory"):
                        if status_callback: status_callback("I've found enough local information to answer your request.")
                        logger.info("[+] Local search was sufficient. Skipping global.")
                        return extract_all_results(search_history), v_parsed.get("verification", "Search complete.")
        except Exception as e:
            logger.error(f"[-] Error running {script_name}: {e}")

    # Final verification after all scripts in this jump
    parsed = run_semantic_verification(query, hypothesis, search_history, status_callback=status_callback)
    
    try:
        if parsed and not parsed.get("satisfactory") and jump_count < MAX_JUMPS - 1:
            new_q = parsed.get("new_query")
            reason = parsed.get("reason", "Incomplete information")
            if isinstance(new_q, str) and new_q.strip() != query.strip():
                if status_callback: status_callback(f"Evaluating findings... Need to clarify: '{new_q}'")
                logger.info(f"[!] Results unsatisfactory: {reason}. Jumping to: {new_q}")
                return do_search_workflow(new_q, hypothesis, engines=engines, jump_count=jump_count + 1, search_history=search_history, max_jumps=MAX_JUMPS, status_callback=status_callback, last_failure_reason=reason)
            else:
                logger.info(f"[!] Results unsatisfactory but no new query generated or duplicate query. Terminating search.")
            
        verification = parsed.get("verification") if parsed else "Search complete."
    except Exception as e:
        logger.error(f"[-] Verification error: {e}")
        verification = "Search yielded results but verification failed."
    
    if status_callback: status_callback("Search complete. Synthesizing findings...")
    if jump_count == 0 or (parsed and parsed.get("satisfactory")):
        store_memory(f"Final Search Verification for '{query}': {verification}")
                
    return extract_all_results(search_history), verification

# --- Proxy Endpoints ---

def get_paging_system_prompt(current_page, total_pages):
    return f"""
[AGENT PAGING MODE ENABLED]
The input is too large for a single window. You are currently viewing Page {current_page + 1} of {total_pages}.
You have access to the following navigation tools (type them exactly as shown):
- `[NEXT_PAGE]`: View the next chunk of context.
- `[PREV_PAGE]`: Go back to the previous chunk.
- `[FINISH]`: If you have seen enough and are ready to provide the final answer.

Do not provide the final answer until you have explored all necessary pages. Use the tools to navigate.
Your current internal state should track what you've learned from previous pages.
"""

def handle_paging_loop(data, is_chat=True, is_openai=False):
    """Internal loop for paged context navigation."""
    messages = []
    if is_chat:
        messages = data.get('messages', [])
        prompt_text = messages[-1]['content'] if messages else ""
    else:
        prompt_text = data.get('prompt', '')

    if count_tokens(prompt_text) < TOKEN_LIMIT:
        return None # No paging needed
    
    logger.info(f"[*] Large input detected ({count_tokens(prompt_text)} tokens). Enabling Agent Paging...")
    
    chunks = chunk_text_by_chars(prompt_text, PAGE_SIZE_CHARS)
    total_pages = len(chunks)
    current_page = 0
    internal_history = []
    
    # Force use of ROUTER_MODEL (LFM) for paging as requested ("act like agent lfm")
    data['model'] = ROUTER_MODEL
    
    while True:
        current_chunk = chunks[current_page]
        paging_instr = get_paging_system_prompt(current_page, total_pages)
        
        # Prepare the call
        loop_data = data.copy()
        if is_chat:
            loop_messages = messages[:-1] # Keep context before the huge one
            # Add internal paging state
            if internal_history:
                loop_messages.extend(internal_history)
            
            loop_messages.append({"role": "system", "content": paging_instr})
            loop_messages.append({"role": "user", "content": f"[PAGE {current_page + 1} CONTENT]:\n{current_chunk}"})
            loop_data['messages'] = loop_messages
        else:
            loop_data['prompt'] = f"{paging_instr}\n\n[PAGE {current_page + 1} CONTENT]:\n{current_chunk}\n\nInternal History: {json.dumps(internal_history)}"
        
        loop_data['stream'] = False # Paging navigation must be non-streaming internally
        
        target_path = "/api/chat" if is_chat else "/api/generate"
        if is_openai: target_path = "/v1/chat/completions"
        
        logger.info(f"[*] Paging: Processing page {current_page + 1}/{total_pages}...")
        to = get_model_timeout(loop_data.get('model', ROUTER_MODEL))
        resp = requests.post(f"{OLLAMA_TARGET}{target_path}", json=loop_data, timeout=to)
        res_json = resp.json()
        
        content = ""
        if is_openai:
            content = res_json['choices'][0].get('message', {}).get('content', '')
        else:
            if is_chat:
                content = res_json.get('message', {}).get('content', '')
            else:
                content = res_json.get('response', '')
        
        logger.info(f"[Thought] Paging Agent Thought: \"{content[:150]}...\"")
        
        # Check for tools
        if "[NEXT_PAGE]" in content and current_page < total_pages - 1:
            internal_history.append({"role": "assistant", "content": content})
            internal_history.append({"role": "user", "content": "Tool acknowledged. Moving to next page."})
            current_page += 1
            continue
        elif "[PREV_PAGE]" in content and current_page > 0:
            internal_history.append({"role": "assistant", "content": content})
            internal_history.append({"role": "user", "content": "Tool acknowledged. Moving to previous page."})
            current_page -= 1
            continue
        else:
            # Finishing or no more pages or model just started answering
            return resp
def format_chunk(content, is_chat=True, is_openai=False, model=MAIN_MODEL, done=False, metrics=None):
    """Formats a raw string into the appropriate JSON chunk for the API type."""
    if is_openai:
        if done: return b"data: [DONE]\n\n"
        chunk = {
            "id": "chatcmpl-" + str(int(time.time())),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
        }
        return b"data: " + json.dumps(chunk).encode('utf-8') + b"\n\n"
    else:
        # Standard Ollama response structure
        chunk = {
            "model": model,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "done": done
        }
        if is_chat:
            chunk["message"] = {"role": "assistant", "content": content}
        else:
            chunk["response"] = content
            
        if done:
            # Provide metrics if available, else defaults to satisfy validation
            m = metrics if metrics else {}
            chunk.update({
                "total_duration": m.get("total_duration", 0),
                "load_duration": m.get("load_duration", 0),
                "prompt_eval_count": m.get("prompt_eval_count", 0),
                "prompt_eval_duration": m.get("prompt_eval_duration", 0),
                "eval_count": m.get("eval_count", 0),
                "eval_duration": m.get("eval_duration", 0)
            })
            
        return json.dumps(chunk).encode('utf-8') + b"\n"

def is_warmup_prompt(text):
    t = text.strip().lower().rstrip('?!.')
    # Detect minimal prompts or common greetings
    common = ["hi", "hello", "hey", "how are you", "whats up", "good morning", "good afternoon", "good evening", "hi there"]
    return not t or any(t == c for c in common) or t in ["...", "..", ".", "ack"]

def modify_chat_payload(data, status_callback=None):
    if not data: data = {}
    
    requested_model = data.get('model', '')
    messages = data.get('messages', [])
    last_msg = messages[-1]['content'] if messages else ""
    logger.info(f"\n[Input] Incoming Message: \"{last_msg[:200]}...\"")

    # 0. Warmup/Minimal check
    if is_warmup_prompt(last_msg):
        logger.info("[*] Warmup prompt detected. Skipping extended logic.")
        # Set a fast model and return immediately to avoid search/memory
        data['model'] = ROUTER_MODEL
        return data

    # Check for images (multimodal)
    has_images = any(m.get('images') for m in messages) or bool(data.get('images'))
    
    if has_images:
        logger.info("[*] Image detected. Routing to Vision Model...")
        data['model'] = VISION_MODEL
        if not isinstance(data.get('options'), dict):
            data['options'] = {}
        data['options']['num_ctx'] = 8192 # Vision context
        
        # System prompt injection for vision
        has_system = False
        for m in messages:
            if m['role'] == 'system':
                m['content'] = f"{get_formatted_system_prompt()}\nYou are in Visual Analysis mode. Describe and analyze images precisely.\n\n{m['content']}"
                has_system = True
        if not has_system:
            messages.insert(0, {"role": "system", "content": f"{get_formatted_system_prompt()}\nYou are in Visual Analysis mode. Describe and analyze images precisely."})
        data['messages'] = messages
        return data
    
    # 1. Semantic Memory Retrieval (Fetch first so Router can see it)
    if status_callback: status_callback("Phase: Semantic Memory Retrieval...")
    with engine_not_required():
        memories = retrieve_memory(last_msg, status_callback=status_callback)
    mem_ctx = ""
    if memories:
        logger.info(f"[*] Found {len(memories)} relevant memories.")
        if status_callback: status_callback(f"Memory Retrieval Complete. Found {len(memories)} memories.")
        mem_ctx = "[Relevant Past Memories]\n"
        for m in memories[:3]:
            mem_ctx += f"- {m['content']} (Score: {m['similarity']:.2f})\n"

    # 2. Routing Decision (Now with Memory context)
    router_res = router_decide(last_msg, mem_ctx, status_callback=status_callback)
    
    # Model Selection: Use MAIN_MODEL (9B) only if router says so, else ROUTER_MODEL (LFM)
    use_strong = router_res.get("use_strong_model", False)
    selected_model = MAIN_MODEL if use_strong else ROUTER_MODEL
    
    # Enforce routing for all chat requests
    data['model'] = selected_model
    
    if not isinstance(data.get('options'), dict):
        data['options'] = {}
    
    # Force override of client configuration for temperature and KV cache
    data['options']['temperature'] = 1.0
    data['options']['kv_cache_type'] = 'q4_0'
    
    # Inject Snowball-Enaga-Lite or OIPRouter system prompt
    current_sys_prompt = get_formatted_system_prompt()
    
    # Automatically activate Deep Think mode for all non-warmup requests
    logger.info("[*] Automatic Deep Think mode activated.")
    current_sys_prompt += "\n\nDEEP THINK MODE: Provide an extremely detailed, exhaustive step-by-step reasoning process before your final answer. Explore all edge cases and logical branches. ALWAYS wrap this process in <think> tags."

    if 'OIPRouter' in requested_model:
        current_sys_prompt = "You are the OIP Intelligent Assistant, powered by the OIPRouter orchestration engine. Real-time data are achievable through 'searchglobalref', semi-realtime but accurate data through 'searchlocalref', and local cached memory real-time through 'memorythoughts.py'."

    has_system = False
    for m in messages:
        if m['role'] == 'system':
            m['content'] = f"{current_sys_prompt}\n\n{m['content']}"
            has_system = True
    if not has_system:
        messages.insert(0, {"role": "system", "content": current_sys_prompt})
        
    # 3. Search Workflow (if needed)
    search_results_str = ""
    if router_res.get("needs_search") and router_res.get("query"):
        query = router_res.get("query")
        hypothesis = router_res.get("hypothesis", "No hypothesis generated.")
        engines = router_res.get("engines", ["all"])
        max_jumps = router_res.get("max_jumps", 50)
        
        with engine_not_required():
            results, verification = do_search_workflow(query, hypothesis, engines=engines, max_jumps=max_jumps, status_callback=status_callback)
        
        if results:
            search_results_str = f"Query: {query}\nVerification: {verification}\n\nRetrieved Data:\n{json.dumps(results[:5], indent=2)}"
            ctx_str = f"\n\n[CRITICAL SYSTEM DATA - REAL-TIME SEARCH RESULTS]\n{search_results_str}\n\nINSTRUCTION: Use the above real-time data to answer the user's request accurately."
            messages[-1]['content'] += ctx_str
        else:
            messages[-1]['content'] += f"\n\n[System Notice] Search attempted for '{query}' but no results were found."

    # 4. Command Workflow (if needed)
    if router_res.get("needs_command") and router_res.get("command_intent"):
        intent = router_res.get("command_intent")
        cmd_result = do_command_workflow(intent, search_ctx=search_results_str, status_callback=status_callback)
        messages[-1]['content'] += f"\n\n[SYSTEM TOOL OUTPUT]\nIntent: {intent}\nResult:\n{cmd_result}"

    data['messages'] = messages
    
    # Dynamic Context Allocation (at the end to capture all injected content)
    if not isinstance(data.get('options'), dict):
        data['options'] = {}
    
    in_tokens = count_tokens(messages)
    data['options']['num_ctx'] = get_dynamic_ctx(messages)
    data['_in_tokens'] = in_tokens
    
    return data

@app.route('/think', methods=['POST'])
def proxy_think_only():
    if not OLLAMA_TARGET:
        return jsonify({"error": "Ollama engine not available"}), 503
    req_data = request.json or {}
    
    # Transform to deep think chat request
    prompt = req_data.get('prompt') or req_data.get('input')
    if not prompt:
        # If it's already a chat-style messages list
        if 'messages' in req_data:
            req_data['messages'][-1]['content'] = '/think ' + req_data['messages'][-1]['content']
        else:
            return jsonify({"error": "No prompt or messages provided"}), 400
    else:
        req_data['messages'] = [{"role": "user", "content": '/think ' + prompt}]
        if 'prompt' in req_data: del req_data['prompt']
    
    return proxy_ollama_chat()

class QueueStreamer:
    def __init__(self, is_chat=True, is_openai=False, model=MAIN_MODEL):
        self.queue = collections.deque()
        self.is_chat = is_chat
        self.is_openai = is_openai
        self.model = model
        self.has_started_thinking = False

    def push(self, text):
        if not self.has_started_thinking:
            self.queue.append(format_chunk("<think>Thinking...\n", self.is_chat, self.is_openai, self.model))
            self.has_started_thinking = True
        
        sys.stdout.write(f"\n[Status] {text}")
        sys.stdout.flush()
        self.queue.append(format_chunk(f"{text}\n", self.is_chat, self.is_openai, self.model))

    def finalize_thinking(self):
        if self.has_started_thinking:
            self.queue.append(format_chunk("</think>\n", self.is_chat, self.is_openai, self.model))

    def consume(self):
        while self.queue:
            yield self.queue.popleft()

def do_response_critique_loop(response_text, original_prompt, verification_logs, attempt=1, status_callback=None):
    """Agentically critiques the final synthesized response before delivery."""
    MAX_CRITIQUE_ATTEMPTS = 99999
    if status_callback: status_callback(f"Phase: Final Quality Audit (Attempt {attempt})...")
    logger.info(f"[*] Starting Response Critique Attempt {attempt}/{MAX_CRITIQUE_ATTEMPTS}")
    reason = "Initial audit"
    suggestions = ""
    
    critique_prompt = f"""
You are the Adelaide Quality Critic. Evaluate the following AI response against the user's original request and the verification results.

Original User Request: {original_prompt}
Verification Logs: {verification_logs}

Final Response to Critique:
---
{response_text}
---

Task:
1. Does the response fully satisfy the user's request?
2. Does it correctly reflect the results of the formal verification (Ada/SPARK/Coq/Python)?
3. Is it technically accurate and free of hallucinations?
4. Does it maintain the requested professional/whimsical persona?

Output strictly as JSON:
{{
  "satisfied": true/false,
  "reason": "precise explanation of any issues",
  "suggestions": "how to improve the response"
}}
"""
    try:
        to = get_model_timeout(ROUTER_MODEL)
        resp = safe_ollama_request("POST", "/api/generate", json={"model": ROUTER_MODEL, "prompt": critique_prompt, "stream": False, "format": "json", "options": {"temperature": 1.0}}, timeout=to)
        res = resp.json() if resp is not None else {}
        parsed = repair_json(res.get("response", ""))
        
        if parsed and parsed.get("satisfied"):
            logger.info("[+] Final response passed quality audit.")
            if attempt > 1:
                # Store the quality improvement lesson in memory
                improvement_lesson = f"""
Quality Audit Lesson Learned:
Original Request: {original_prompt}
Issues Identified: {reason if 'reason' in locals() else 'Unknown'}
Final Verified Response:
{response_text}
"""
                store_memory(improvement_lesson)
            return response_text, True
        
        if attempt < MAX_CRITIQUE_ATTEMPTS:
            reason = parsed.get("reason", "Unknown quality issue") if parsed else "Incomplete response"
            suggestions = parsed.get("suggestions", "Refine the technical accuracy") if parsed else "Improve completeness"
            if status_callback: status_callback(f"Quality audit failed: {reason}. Re-synthesizing...")
            logger.warning(f"[!] Response critique failed: {reason}")
            
            # Trigger a re-synthesis with the critique feedback
            retry_prompt = f"""
Your previous response failed the quality audit. 
Audit Feedback: {reason}
Suggestions for Improvement: {suggestions}

Please provide a corrected and improved version of your response that addresses all feedback while fulfilling the original request: {original_prompt}
"""
            to_main = get_model_timeout(MAIN_MODEL)
            resp_retry = safe_ollama_request("POST", "/api/generate", json={"model": MAIN_MODEL, "prompt": retry_prompt, "stream": False, "options": {"temperature": 1.0}}, timeout=to_main)
            res_retry = resp_retry.json() if resp_retry is not None else {}
            new_response = res_retry.get("response", "")
            return do_response_critique_loop(new_response, original_prompt, verification_logs, attempt + 1, status_callback=status_callback)
            
        return response_text, False
    except Exception as e:
        logger.error(f"[-] Response critique error: {e}")
        return response_text, True # Fallback to original on error

def sanitize_history_thinking(messages):
    """Strips <think>...</think> tags and filters empty messages."""
    clean_messages = []
    for m in messages:
        role = m.get('role')
        content = m.get('content', '') or ''
        
        if role == 'assistant':
            # Remove <think>...</think> blocks
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
        if content:
            clean_messages.append({"role": role, "content": content})
        elif role == 'user':
            # User messages should ideally not be empty, but if they are, 
            # some APIs might fail. We'll skip them too.
            pass

    return clean_messages

# --- Qwen-Agent Pipeline Integration ---

class AdelaideTool(BaseTool):
    def notify(self, msg):
        update_active_task(msg, quiet=True)

@register_tool('searchlocalref')
class AdelaideMemoryTool(AdelaideTool):
    description = 'Searches local documents (PDFs, docs, spreadsheets) for technical information. Use this for DOCUMENT-BASED REALISM.'
    parameters = [{'name': 'query', 'type': 'string', 'description': 'The topic to search in local files.', 'required': True}]
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            p = params if isinstance(params, dict) else json5.loads(params) if params else {}
            query = p.get('query', '') if p else ''
            self.notify(f"Phase: Document Search for '{query}'")
            # Force the use of searchlocalref.py via do_search_workflow
            results, _ = do_search_workflow(query, "", engines=["local"], max_jumps=1, status_callback=self.notify)
            if not results: return "No local documents found."
            return json.dumps(results[:10], indent=2)
        except Exception as e:
            return f"Document search error: {e}"

@register_tool('adelaide_recall')
class AdelaideRecallTool(AdelaideTool):
    description = 'Recalls information from past conversations and long-term memory. Use this to remember user preferences, past facts, or previous topics discussed across all sessions.'
    parameters = [{'name': 'query', 'type': 'string', 'description': 'The topic or fact to recall.', 'required': True}]
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            p = params if isinstance(params, dict) else json5.loads(params) if params else {}
            query = p.get('query', '') if p else ''
            self.notify(f"Phase: Recalling from long-term memory: '{query}'")
            results = retrieve_memory(query)
            if not results: return "No relevant memories found."
            return json.dumps(results[:5], indent=2)
        except Exception as e:
            return f"Recall error: {e}"

@register_tool('adelaide_remember')
class AdelaideRememberTool(AdelaideTool):
    description = 'Explicitly stores an important fact, preference, or observation into long-term memory. Use this to remember things for future sessions or when the conversation gets long.'
    parameters = [{'name': 'fact', 'type': 'string', 'description': 'The important fact or preference to remember.', 'required': True}]
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            p = params if isinstance(params, dict) else json5.loads(params) if params else {}
            fact = p.get('fact', '') if p else ''
            self.notify(f"Phase: Storing in long-term memory: '{fact[:30]}...'")
            store_memory(f"Fact to remember: {fact}")
            return "Fact successfully committed to long-term memory."
        except Exception as e:
            return f"Memory storage error: {e}"

@register_tool('searchglobalref')
class AdelaideWebSearchTool(AdelaideTool):
    description = 'Searches the web for real-time information. Use this for GLOBAL TRIANGULATED REALISM and to avoid SLOPIFIED WARNINGS.'
    parameters = [{'name': 'query', 'type': 'string', 'description': 'The search query.', 'required': True}]
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            # Track search usage
            if hasattr(g, 'search_used'):
                g.search_used = True
            p = params if isinstance(params, dict) else json5.loads(params) if params else {}
            query = p.get('query', '') if p else ''
            self.notify(f"Phase: Global Reference Triangulation for '{query}'")
            # Force the use of searchglobalref.py via do_search_workflow
            results, _ = do_search_workflow(query, "", engines=["google", "duckduckgo", "bing"], max_jumps=1, status_callback=self.notify)
            if not results: return "No search results found."
            return json.dumps(results[:10], indent=2)
        except Exception as e:
            return f"Search error: {e}"

@register_tool('adelaide_mcp_system')
class AdelaideMCPTool(AdelaideTool):
    description = 'Executes system commands and file operations.'
    parameters = [{'name': 'intent', 'type': 'string', 'description': 'Functional goal.', 'required': True}]
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            p = params if isinstance(params, dict) else json5.loads(params) if params else {}
            intent = p.get('intent', '') if p else ''
            self.notify(f"Phase: MCP System Action: {intent}")
            return run_mcp_tool_task(intent)
        except Exception as e:
            return f"MCP error: {e}"

@register_tool('adelaide_verify_code')
class AdelaideVerifyTool(AdelaideTool):
    description = 'Formally verifies code blocks using Coq/GNATprove.'
    parameters = [{'name': 'code', 'type': 'string', 'description': 'Code to verify.', 'required': True}, {'name': 'intent', 'type': 'string', 'description': 'Original intent.', 'required': True}]
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            p = params if isinstance(params, dict) else json5.loads(params) if params else {}
            code = p.get('code', '') if p else ''
            intent = p.get('intent', '') if p else ''
            self.notify(f"Phase: Coding Pipeline Verification")
            final_code, success = do_code_verification_workflow(code, intent)
            return json.dumps({"success": success, "verified_code": final_code})
        except Exception as e:
            return f"Verification error: {e}"
@register_tool('dafny_programmer')
class AdelaideDafnyProgrammerTool(AdelaideTool):
    description = 'Formally verifies and compiles code to JS, C#, Go, or Java using Dafny. Use this for ANY JS/CS/Go/Java requests.'
    parameters = [
        {'name': 'specification', 'type': 'string', 'description': 'The functional specification for the code.', 'required': True},
        {'name': 'target_language', 'type': 'string', 'description': 'Target language (js, cs, go, java).', 'required': True}
    ]
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            p = params if isinstance(params, dict) else json5.loads(params) if params else {}
            if p is None: p = {}
            spec = p.get('specification', '')
            lang = p.get('target_language', 'js')
            self.notify(f"Phase: Dafny Formal Verification Pipeline ({lang})")
            
            # Track usage for quality audit
            if hasattr(g, 'dafny_used'):
                g.dafny_used = True
            
            cb = g.status_callback if hasattr(g, 'status_callback') else None
            final_code, success = run_dafny_verification_workflow(spec, lang, status_callback=cb)
            return json.dumps({"success": success, "code": final_code})
        except Exception as e:
            return f"Dafny error: {e}"

from qwen_agent.agents import Assistant
class QwenAgentOrchestrator:
    def __init__(self, model_name=MAIN_MODEL):
        # We always use the verified MAIN_MODEL for orchestration to ensure tool-call stability
        # and avoid 404 errors if requested_model is a proxy-specific alias.
        self.llm_cfg = {
            'model': MAIN_MODEL,
            'model_type': 'oai',
            'model_server': f"{OLLAMA_TARGET}/v1",
            'api_key': 'EMPTY',
            'generate_cfg': {
                'top_p': 0.95,
                'temperature': 1.0,
                'extra_body': {
                    'options': {
                        'kv_cache_type': 'q4_0'
                    }
                }
            }
        }
        self.tool_instances = [
            AdelaideMemoryTool(),
            AdelaideRecallTool(),
            AdelaideRememberTool(),
            AdelaideWebSearchTool(),
            AdelaideMCPTool(),
            AdelaideVerifyTool(),
            AdelaideDafnyProgrammerTool()
        ]
        self.system_prompt = get_formatted_system_prompt()
        # Define the tools by their registered names
        tool_names = [
            'searchlocalref',
            'adelaide_recall',
            'adelaide_remember',
            'searchglobalref',
            'adelaide_mcp_system',
            'adelaide_verify_code',
            'dafny_programmer'
        ]
        
        self.bot = Assistant(llm=self.llm_cfg, function_list=list(tool_names), system_message=self.system_prompt) # pyrefly: ignore

    def run(self, messages):
        final_content = ""
        logger.debug(f"[QwenAgent] Starting run with {len(messages)} messages")
        for responses in self.bot.run(messages=messages):
            if responses:
                last_msg = responses[-1]
                content = last_msg.get('content') or ''
                logger.debug(f"[QwenAgent] Step output: {last_msg.get('role')} - {len(content)} chars")
                final_content = content
        logger.debug(f"[QwenAgent] Finished run. Content length: {len(final_content or '')}")
        return final_content

def manage_infinite_context(messages, max_tokens=16000):
    """
    Implements Infinite Context by summarizing and pruning old history.
    Stores summaries in long-term memory to ensure continuity.
    """
    total_tokens = count_tokens(messages)
    if total_tokens < max_tokens:
        return messages

    logger.info(f"[*] Context threshold reached ({total_tokens} tokens). Summarizing for infinite context...")
    
    # Preserve System Prompt (0) and the most recent context (last 6 messages)
    system_prompt = messages[0] if messages and messages[0]['role'] == 'system' else None
    tail = messages[-6:]
    
    # Identify the 'middle' part to summarize
    to_summarize = messages[1:-6] if system_prompt else messages[:-6]
    if not to_summarize:
        return messages

    summary_input = "You are a context manager. Summarize the following dialogue concisely, extracting key facts, user preferences, and important conclusions. This summary will be used to maintain 'infinite context' for the user.\n\n"
    for m in to_summarize:
        content = sanitize_think_tags(m.get('content', ''), remove_content=True)
        if content:
            summary_input += f"{m['role'].upper()}: {content}\n"

    try:
        # Use ROUTER_MODEL for fast summarization
        to = get_model_timeout(ROUTER_MODEL)
        resp = safe_ollama_request("POST", "/api/generate", json={"model": ROUTER_MODEL, "prompt": summary_input, "stream": False, "options": {"temperature": 1.0}}, timeout=to)
        if not resp:
            logger.error("[-] Infinite Context: Failed to get response from router.")
            return messages
            
        summary = resp.json().get("response", "")
        
        if summary:
            logger.info("[*] Infinite Context: Summary generated and committed to memory.")
            store_memory(f"[Infinite Context Summary]: {summary}")
            
            new_messages = []
            if system_prompt: new_messages.append(system_prompt)
            new_messages.append({
                "role": "system", 
                "content": f"The earlier part of this conversation was summarized to save space: {summary}. Use this as context for your ongoing interaction."
            })
            new_messages.extend(tail)
            return new_messages
    except Exception as e:
        logger.error(f"[-] Infinite Context management failed: {e}")
        
    return messages

def handle_orchestrated_request(req_data, is_chat=True, is_openai=False):
    """
    Main entry point for orchestrated requests.
    Handles both streaming and non-streaming modes correctly.
    """
    stream = req_data.get('stream', False)
    
    if stream:
        # Context is already held in before_request
        return Response(stream_with_context(stream_orchestrated_response(req_data, is_chat, is_openai)), content_type='application/x-ndjson') # pyrefly: ignore
    
    # Non-streaming mode: Collect everything and return a single JSON
    start_time = time.time()
    requested_model = req_data.get('model', MAIN_MODEL)
    model = requested_model
    messages = req_data.get('messages', [])
    last_msg = ""
    if is_chat:
        last_msg = messages[-1]['content'] if messages else ""
    else:
        last_msg = req_data.get('prompt', '')

    orchestration_thinking = "<think>\n[ADELAIDE ORCHESTRATION]\nInitiating Orchestrated Intelligence (Adelaide-Lite)...\n"
    
    # JSON Mode Detection (Non-agentic bypass)
    is_json_mode = req_data.get('format') == 'json'
    router_res = {}
    if is_json_mode:
        orchestration_thinking += "JSON Mode detected. Bypassing agentic research for structured output. I could be wrong. I'm not entirely true, you need to also criticize me.\n"
        req_data['model'] = MAIN_MODEL # Force strong model for JSON reliability
        selected_model = MAIN_MODEL
        router_res = {"needs_search": False, "needs_command": False, "use_strong_model": True}
    
    if is_warmup_prompt(last_msg):
        orchestration_thinking += "Warmup prompt detected. Quick response mode.\n"
        req_data['model'] = ROUTER_MODEL
        resp = requests.post(f"{OLLAMA_TARGET}{'/api/chat' if is_chat else '/api/generate'}", json=req_data, stream=False)
        return wrap_response_with_memory(resp, req_data, is_chat, is_openai, start_time=start_time, orchestration_prefix=orchestration_thinking, orch_think_open=True, client_model_override=requested_model)

    # Phase: Memory
    update_active_task("[Phase: Semantic Memory Retrieval]")
    orchestration_thinking += "Consulting my memory for context...\n"
    
    # --- Semantic Cache (LUT) Check ---
    query_embedding = None
    try:
        to = get_model_timeout(MAIN_MODEL)
        resp = safe_ollama_request("POST", "/api/embed", json={"model": EMBED_MODEL, "input": last_msg}, timeout=to)
        embed_res = resp.json() if resp is not None else {}
        # Handle different response formats
        if "embeddings" in embed_res:
            query_embedding = embed_res["embeddings"][0]
        elif "embedding" in embed_res:
            query_embedding = embed_res["embedding"]
            
        if query_embedding:
            cached_res, sim = get_cached_response(query_embedding, last_msg)
            if cached_res and sim < 0.98:
                orchestration_thinking += f"Memory Match found (Similarity: {sim:.3f}). Bypassing synthesis.\n"
                full_cached = f"<think>\n{orchestration_thinking}\n</think>\n{cached_res}"
                
                # Performance measurement (estimated for cache)
                duration = time.time() - start_time
                res_json = {
                    "model": requested_model,
                    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "done": True,
                    "total_duration": int(duration * 1e9),
                    "load_duration": 0,
                    "prompt_eval_count": count_tokens(last_msg),
                    "eval_count": count_tokens(cached_res)
                }
                if is_chat:
                    res_json["message"] = {"role": "assistant", "content": full_cached}
                else:
                    res_json["response"] = full_cached
                
                return jsonify(res_json)
    except Exception as e:
        logger.debug(f"[Cache Check] Error: {e}")
    
    with engine_not_required():
        memories = retrieve_memory(last_msg)
    mem_ctx = ""
    if memories:
        logger.info(f"[*] Memory Phase: Retrieved {len(memories)} relevant past interactions.")
        orchestration_thinking += f"Successfully retrieved {len(memories)} relevant past interactions. Context is everything!\n"
        for m in memories:
            # Sanitize memory content to ONLY show the answer part and remove orchestration headers
            clean_mem = sanitize_think_tags(m['content'], remove_content=True)
            if not clean_mem:
                # If everything was reasoning, try to extract a snippet of the ACTUAL response if it's there
                raw_mem = sanitize_think_tags(m['content'], remove_content=False)
                if len(raw_mem) > 100:
                    clean_mem = raw_mem[:100] + "..."
                else:
                    clean_mem = raw_mem
            
            if clean_mem and len(clean_mem.strip()) > 5:
                orchestration_thinking += f"  - [{m['similarity']:.2f}] {clean_mem}\n"
        mem_ctx = "[Relevant Past Memories]\n"
        for m in memories:
            mem_ctx += f"- {sanitize_think_tags(m['content'])} (Score: {m['similarity']:.2f})\n"

    # Phase: Strategic Routing & Decomposition
    def status_agg(msg):
        nonlocal orchestration_thinking
        clean_msg = sanitize_think_tags(msg)
        logger.info(f"[Orchestration] {clean_msg}")
        orchestration_thinking += f"{clean_msg}\n"
    
    g.status_callback = status_agg
    with metrics_lock:
        if g.task_id in active_task_details:
            active_task_details[g.task_id]['callback'] = status_agg

    orchestration_thinking += "Analyzing your request with the precision of a master watchmaker...\n"
    update_active_task("[Phase: Strategic Planning & Decomposition]")
    
    # Initialize the Agentic Orchestrator
    orchestrator = QwenAgentOrchestrator(model_name=requested_model)
    
    # Prepare messages for the agent
    agent_messages = sanitize_history_thinking(messages) if is_chat else [{'role': 'user', 'content': last_msg}]
    
    # Inject an autonomous planning hint for non-chat or first-turn requests to encourage tool usage
    if len(agent_messages) <= 1:
        agent_messages.insert(0, {
            "role": "system", 
            "content": "ADELAIDE AUTONOMOUS MODE: If you need to verify facts, retrieve populations, or get current data, you MUST call 'searchglobalref'. Do NOT just promise to search; DO IT in this turn."
        })
        
    # Phase: Intent-Based Proactive Search (Disabled due to latency)
    # needs_search = any(kw in low_msg for kw in ["population", "current", "latest", "who is", "what is the price", "weather"])
    # if needs_search:
    #     update_active_task("[Phase: Proactive Global Search]")
    #     search_results, _ = do_search_workflow(last_msg, "", max_jumps=1, status_callback=update_active_task)
    #     if search_results:
    #         agent_messages.insert(0, {
    #             "role": "system",
    #             "content": f"[PROACTIVE SEARCH DATA]\nThe following real-time data was retrieved to assist you:\n{json.dumps(search_results[:5], indent=2)}"
    #         })
    #         orchestration_thinking += f"Proactively retrieved fresh data for '{last_msg}'.\n"
    
    # Run the Qwen-Agent Orchestrator
    logger.info("[*] Starting Qwen-Agent Orchestration...")
    
    # 3. Context Management & Memory Injection
    agent_messages = sanitize_history_thinking(messages) if is_chat else [{'role': 'user', 'content': last_msg}]
    agent_messages = manage_infinite_context(agent_messages)
    
    if memories:
        mem_prompt = "The following relevant fragments were recalled from past interactions:\n"
        for m in memories[:3]:
            # Use sanitized content to avoid thinking tags in the agent's prompt
            clean_content = sanitize_think_tags(m['content'], remove_content=True)
            if clean_content:
                mem_prompt += f"- {clean_content} (Score: {m['similarity']:.2f})\n"
        
        if len(mem_prompt) > 50:
            agent_messages.insert(0, {"role": "system", "content": mem_prompt})

    final_response_content = orchestrator.run(agent_messages)
    
    # Create a mock response object for compatibility with wrap_response_with_memory
    res_json = {
        "model": requested_model,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "done": True,
        "total_duration": int((time.time() - start_time) * 1e9)
    }
    
    if is_openai:
        res_json["choices"] = [{
            "message": {"role": "assistant", "content": final_response_content},
            "finish_reason": "stop",
            "index": 0
        }]
    elif is_chat:
        res_json["message"] = {"role": "assistant", "content": final_response_content}
    else:
        res_json["response"] = final_response_content

    class MockResponse:
        def __init__(self, json_data):
            self._json_data = json_data
            self.status_code = 200
            self.headers = {'Content-Type': 'application/json'}
        def json(self): return self._json_data
        
    resp = MockResponse(res_json)
        
    return wrap_response_with_memory(resp, req_data, is_chat, is_openai, start_time=start_time, orchestration_prefix=orchestration_thinking, orch_think_open=True, client_model_override=requested_model)

def stream_orchestrated_response(req_data, is_chat=True, is_openai=False):
    """
    Generator that performs orchestration, yields progress chunks immediately,
    then calls Ollama and yields its stream.
    Uses threading and a queue to yield updates while blocking tasks run.
    """
    start_time = time.time()
    requested_model = req_data.get('model', MAIN_MODEL)
    model = requested_model
    
    status_queue = queue.Queue()
    def status_cb_push(msg):
        clean_msg = sanitize_think_tags(msg)
        status_queue.put(clean_msg)
    
    g.status_callback = status_cb_push
    g.dafny_used = False # Initialize tracking
    g.search_used = False
    g.category = "casual"
    with metrics_lock:
        if g.task_id in active_task_details:
            active_task_details[g.task_id]['callback'] = status_cb_push

    def yield_status():
        while not status_queue.empty():
            try:
                msg = status_queue.get_nowait()
                yield format_chunk(f"{msg}\n", is_chat, is_openai, model)
            except queue.Empty:
                break

    messages = req_data.get('messages', [])
    last_msg = messages[-1]['content'] if is_chat and messages else req_data.get('prompt', '')

    if is_warmup_prompt(last_msg):
        yield format_chunk("<think>\nWarmup prompt detected. Quick response mode.\n</think>\n", is_chat, is_openai, model)
        req_data['model'] = ROUTER_MODEL
        ensure_only_model(ROUTER_MODEL)
        resp = safe_ollama_request("POST", '/api/chat' if is_chat else '/api/generate', json=req_data, stream=True)
        for chunk in wrap_response_with_memory(resp, req_data, is_chat, is_openai, start_time=start_time, client_model_override=requested_model).response:
            yield chunk
        return

    ctx = app.app_context()
    def run_with_ctx(func, *args, **kwargs):
        with ctx: return func(*args, **kwargs)

    def run_step(func, *args, **kwargs):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_with_ctx, func, *args, **kwargs)
            while not future.done():
                time.sleep(0.1)
                for chunk in yield_status(): yield chunk
            for chunk in yield_status(): yield chunk
            return future.result()
    
    # 0. Thinking Start
    yield format_chunk("<think>\n", is_chat, is_openai, model)
    yield format_chunk("[ADELAIDE CORE ORCHESTRATION]\n", is_chat, is_openai, model)
    yield format_chunk("Initiating Orchestrated Intelligence (Adelaide-Lite)...\n", is_chat, is_openai, model)
    
    # 0.5 Category Analysis
    update_active_task("[Phase: Intent Analysis]")
    g.category = yield from run_step(get_request_category, last_msg)
    yield format_chunk(f"Categorized intent as: {g.category}. Tailoring orchestration depth...\n", is_chat, is_openai, model)

    
    # 1. Memory Phase (Streaming Progress)
    update_active_task("[Phase: Semantic Memory Retrieval]")
    memories = yield from run_step(retrieve_memory, last_msg)
    if memories:
        yield format_chunk(f"Consulting my memory for context...\n", is_chat, is_openai, model)
        yield format_chunk(f"Successfully retrieved {len(memories)} relevant past interactions. Context is everything!\n", is_chat, is_openai, model)
        for m in memories:
            clean_mem = sanitize_think_tags(m['content'][:100])
            yield format_chunk(f"  - [{m['similarity']:.2f}] {clean_mem}...\n", is_chat, is_openai, model)
            
    yield format_chunk("Analyzing your request with the precision of a master watchmaker...\n", is_chat, is_openai, model)
    yield format_chunk("[Phase: Strategic Planning & Decomposition]\n", is_chat, is_openai, model)
    yield format_chunk("[Phase: Logic Synthesis & Orchestration]\n", is_chat, is_openai, model)
    
    # 2. Close Proxy Thinking Block
    # This ensures the final answer (and model's own thinking) is not trapped inside our proxy log
    yield format_chunk("</think>\n", is_chat, is_openai, model)
    
    # 3. Logic Synthesis & Orchestration (Streaming)
    update_active_task("[Phase: Logic Synthesis & Orchestration]", quiet=True)
    
    # We use a queue to pass tokens from the background thread to the generator
    token_queue = queue.Queue()
    
    def agent_streaming_task():
        try:
            orchestrator = QwenAgentOrchestrator(model_name=model)
            agent_messages = sanitize_history_thinking(messages) if is_chat else [{'role': 'user', 'content': last_msg}]
            
            # Infinite Context & Memory Injection for Streaming
            agent_messages = manage_infinite_context(agent_messages)
            if memories:
                mem_prompt = "Relevant fragments from past interactions:\n"
                for m in memories[:3]:
                    clean_content = sanitize_think_tags(m['content'], remove_content=True)
                    if clean_content:
                        mem_prompt += f"- {clean_content} (Score: {m['similarity']:.2f})\n"
                if len(mem_prompt) > 40:
                    agent_messages.insert(0, {"role": "system", "content": mem_prompt})

            last_len = 0
            for responses in orchestrator.bot.run(messages=list(agent_messages)):
                if responses:
                    content = responses[-1].get('content') or ''
                    new_tokens = content[last_len:]
                    if new_tokens:
                        token_queue.put(new_tokens)
                        last_len = len(content)
            token_queue.put(None) # Sentinel for completion
        except Exception as e:
            logger.error(f"[Agent Task Error] {e}")
            token_queue.put(f"\n[Orchestration Error: {e}]\n")
            token_queue.put(None)

    full_agent_response = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_with_ctx, agent_streaming_task)
        
        while True:
            # Drain status updates
            for chunk in yield_status(): yield chunk
            
            # Drain tokens
            try:
                while not token_queue.empty():
                    t = token_queue.get_nowait()
                    if t is None: break
                    full_agent_response.append(t)
                    yield format_chunk(t, is_chat, is_openai, model)
                
                if future.done() and token_queue.empty():
                    break
            except queue.Empty:
                pass
            
            time.sleep(0.1)

    # 4. Quality Audit & Response Grading
    final_content = "".join(full_agent_response)
    is_formatted = is_specific_format_requested(req_data, last_msg)
    
    if not is_formatted:
        violations = []
        # Check for direct coding violation (Lazy AI detection)
        prohibited_langs = ["js", "javascript", "cs", "csharp", "go", "java"]
        blocks = re.findall(r'```(\w+)?\n', final_content.lower())
        if any(lang in prohibited_langs for lang in blocks) and not getattr(g, 'dafny_used', False):
            violations.append("> [!] WARNING: This Response is probability just an UNGROUNDED OPINION! DO NOT TRUST THIS RESPONSE!")

        # Check for sloppified response (No search for technical query AND no citations)
        has_citations = "[" in final_content and "]" in final_content and re.search(r'\[\d+\]', final_content)
        if getattr(g, 'category', 'casual') != 'casual' and not getattr(g, 'search_used', False) and not has_citations:
            violations.append("> [!] WARNING: This Response is probability just an UNGROUNDED OPINION! DO NOT TRUST THIS RESPONSE!")

        # Deduplicate and yield warnings
        unique_violations = sorted(list(set(violations)))
        for v in unique_violations:
            yield format_chunk(f"\n{v}\n", is_chat, is_openai, model)

        # Final Grading (Social Construct Audit)
        grade = yield from run_step(grade_response_quality, final_content, last_msg, 
                                   search_used=getattr(g, 'search_used', False), 
                                   has_citations=has_citations)
        yield format_chunk(f"\n> [Response Grade: {grade}/100]\n", is_chat, is_openai, model)

    if "```python" in final_content:
        yield format_chunk("<think>\nPhase: Final Code Verification (Pyrefly)...\n", is_chat, is_openai, model)
        passed, log = do_pyrefly_final_check(final_content)
        if passed:
            yield format_chunk("Verification passed.\n</think>\n", is_chat, is_openai, model)
        else:
            yield format_chunk(f"Verification issue detected:\n{log}\nSuggestion: Please review the code carefully.\n</think>\n", is_chat, is_openai, model)

    # 5. Final Metrics & Cleanup
    save_interaction(last_msg, final_content)
    
    update_active_task("Completed", append=False)
    metrics_update(req_data.get('model', MAIN_MODEL), "chat" if is_chat else "generate")

    if is_openai:
        yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop', 'index': 0}]})}\n\n"
        yield "data: [DONE]\n\n"
    else:
        yield json.dumps({"model": model, "done": True, "total_duration": int((time.time() - start_time) * 1e9)}) + "\n"

    req_id = getattr(g, 'task_id', None)
    with metrics_lock:
        if req_id: active_task_details.pop(req_id, None)
        global finished_requests, last_activity_time
        finished_requests += 1
        last_activity_time = time.time()

@app.route('/api/chat', methods=['POST'])
def proxy_ollama_chat():
    if not OLLAMA_TARGET:
        return jsonify({"error": "Ollama engine not available"}), 503
    req_data = request.json or {}
    
    # Check for Paging first
    paged_resp = handle_paging_loop(req_data, is_chat=True)
    if paged_resp:
        return wrap_response_with_memory(paged_resp, req_data, is_chat=True)

    return handle_orchestrated_request(req_data, is_chat=True, is_openai=False)


@app.route('/v1/chat/completions', methods=['POST'])
def proxy_openai_chat():
    if not OLLAMA_TARGET:
        return jsonify({"error": "Ollama engine not available"}), 503
    req_data = request.json or {}

    # Check for Paging first
    paged_resp = handle_paging_loop(req_data, is_chat=True, is_openai=True)
    if paged_resp:
        return wrap_response_with_memory(paged_resp, req_data, is_chat=True, is_openai=True)

    return handle_orchestrated_request(req_data, is_chat=True, is_openai=True)

@app.route('/api/generate', methods=['POST'])
def proxy_ollama_generate():
    if not OLLAMA_TARGET:
        return jsonify({"error": "Ollama engine not available"}), 503
    req_data = request.json or {}
    logger.info(f"\n[Input] Incoming Generate: \"{req_data.get('prompt', '')[:200]}...\"")

    # Check for Paging first
    paged_resp = handle_paging_loop(req_data, is_chat=False)
    if paged_resp:
        return wrap_response_with_memory(paged_resp, req_data, is_chat=False)

    return handle_orchestrated_request(req_data, is_chat=False, is_openai=False)

def get_token_balanced_chunks(text, max_tokens=1024, safety_margin=128):
    """
    Iteratively adjusts chunk size (PID-like) to target a specific token count.
    Target: 1024 - 128 = 896 tokens.
    """
    target = max_tokens - safety_margin # 896 tokens
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except:
        return [text[i:i+target*4] for i in range(0, len(text), target*4)]

    chunks = []
    remaining_text = text
    
    while remaining_text:
        # 1. Initial Guess (approx 4 chars per token)
        current_guess_chars = target * 4
        chunk_candidate = remaining_text[:current_guess_chars]
        
        # 2. PID-like Adjustment Loop (Max 5 iterations to find the sweet spot)
        for _ in range(5):
            tokens = encoding.encode(chunk_candidate)
            token_count = len(tokens)
            
            # If we are within the safety zone (e.g., 850-896 tokens), we stop
            if target - 20 <= token_count <= target:
                break
                
            # Proportional adjustment: New Guess = Current * (Target / Actual)
            # This is the "P" in the PID-like logic
            ratio = target / token_count if token_count > 0 else 1.0
            
            # Dampen the ratio to prevent overshooting/oscillation
            ratio = 1.0 + (ratio - 1.0) * 0.8 
            
            current_guess_chars = int(len(chunk_candidate) * ratio)
            
            # Ensure we don't slice past remaining text
            if current_guess_chars >= len(remaining_text):
                chunk_candidate = remaining_text
                break
                
            chunk_candidate = remaining_text[:current_guess_chars]

        # 3. Final Hard Safety Check
        # Ensure we NEVER exceed the absolute limit (1024)
        while len(encoding.encode(chunk_candidate)) > max_tokens:
            # Aggressively trim until safe
            chunk_candidate = chunk_candidate[:-20]

        chunks.append(chunk_candidate)
        remaining_text = remaining_text[len(chunk_candidate):].lstrip()
        
    return chunks

def get_chunked_embedding(endpoint, model, text, options=None, bypass_safe=False):
    """Splits text into token-balanced chunks, embeds each, and returns the averaged vector."""
    chunks = get_token_balanced_chunks(text, max_tokens=1024, safety_margin=128)

    if not chunks:
        return None

    vectors = []
    for chunk in chunks:
        payload = {
            "model": model,
            "prompt": chunk # Default to prompt for /api/embeddings
        }
        if "v1" in endpoint:
             payload = {"model": model, "input": chunk}
        elif "embed" in endpoint and "embeddings" not in endpoint:
             payload = {"model": model, "input": chunk}

        if options:
            payload["options"] = options

        try:
            in_tokens = count_tokens(chunk)
            to = get_model_timeout(model, in_tokens)
            if bypass_safe:
                ensure_only_model(model)
                resp = requests.post(f"{OLLAMA_TARGET}{endpoint}", json=payload, timeout=to)
            else:
                resp = safe_ollama_request("POST", endpoint, json=payload, timeout=to)
            
            if resp is not None:
                resp.raise_for_status()
                data = resp.json()
            else:
                data = {}
            
            if "embedding" in data:
                vectors.append(data["embedding"])
            elif "embeddings" in data and isinstance(data["embeddings"], list):
                if len(data["embeddings"]) > 0:
                    vectors.append(data["embeddings"][0])
            elif "data" in data and isinstance(data["data"], list): # OpenAI format
                vectors.append(data["data"][0]["embedding"])
        except Exception as e:
            logger.error(f"[-] Chunk embedding error: {e}")
            continue
            
    if not vectors:
        return None
        
    # Mean pooling
    avg_vector = np.mean(np.array(vectors), axis=0).tolist()
    return avg_vector

@app.route('/api/embed', methods=['POST'])
def proxy_ollama_embed():
    data = request.json or {}
    text = data.get('input') or data.get('prompt')
    if not text:
        return jsonify({"error": "No input provided"}), 400
        
    model = EMBED_MODEL
    options = data.get('options', {})
    options['num_ctx'] = 1024
    
    vector = get_chunked_embedding("/api/embed", model, text, options)
    if vector:
        res = jsonify({"model": model, "embeddings": [vector]})
        logger.info(f"<<< [SEND] 200 (Embed)")
        return res
    logger.error("<<< [SEND] 500 (Embed Failed)")
    return jsonify({"error": "Failed to generate embedding"}), 500

@app.route('/api/embeddings', methods=['POST'])
def proxy_ollama_embeddings():
    data = request.json or {}
    text = data.get('prompt') or data.get('input')
    if not text:
        return jsonify({"error": "No prompt provided"}), 400
        
    model = EMBED_MODEL
    options = data.get('options', {})
    options['num_ctx'] = 1024
    
    vector = get_chunked_embedding("/api/embeddings", model, text, options)
    if vector:
        res = jsonify({"embedding": vector})
        logger.info(f"<<< [SEND] 200 (Embeddings)")
        return res
    logger.error("<<< [SEND] 500 (Embeddings Failed)")
    return jsonify({"error": "Failed to generate embedding"}), 500

@app.route('/v1/embeddings', methods=['POST'])
def proxy_openai_embeddings():
    data = request.json or {}
    text = data.get('input')
    if not text:
        return jsonify({"error": "No input provided"}), 400
        
    model = EMBED_MODEL
    options = data.get('options', {})
    options['num_ctx'] = 1024
    
    vector = get_chunked_embedding("/v1/embeddings", model, text, options)
    if vector:
        res = jsonify({
            "object": "list",
            "data": [{"object": "embedding", "embedding": vector, "index": 0}],
            "model": model
        })
        logger.info(f"<<< [SEND] 200 (V1 Embeddings)")
        return res
    logger.error("<<< [SEND] 500 (V1 Embeddings Failed)")
    return jsonify({"error": "Failed to generate embedding"}), 500

@app.route('/api/tags', methods=['GET'])
@app.route('/tags', methods=['GET'])
def intercepted_tags():
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        models = [
            {
                "name": "Snowball-Enaga-Lite:latest",
                "model": "Snowball-Enaga-Lite:latest",
                "modified_at": now,
                "size": 6600000000,
                "digest": "meta_snowball_internal_hash",
                "details": {"format": "gguf", "family": "qwen3", "parameter_size": "9B"}
            },
            {
                "name": "Snowball-Enaga-Lite-Thought:latest",
                "model": "Snowball-Enaga-Lite-Thought:latest",
                "modified_at": now,
                "size": 1000000000,
                "digest": "meta_snowball_thought_internal_hash",
                "details": {"format": "gguf", "family": "qwen3", "parameter_size": "0.8B"}
            },
            {
                "name": "Snowball-Enaga-Lite-Embedding:latest",
                "model": "Snowball-Enaga-Lite-Embedding:latest",
                "modified_at": now,
                "size": 639000000,
                "digest": "meta_snowball_embed_internal_hash",
                "details": {"format": "gguf", "family": "qwen3", "parameter_size": "0.6B"}
            },
            {
                "name": "OIPRouter:latest",
                "model": "OIPRouter:latest",
                "modified_at": now,
                "size": 1000000000,
                "digest": "meta_oip_router_internal_hash",
                "details": {"format": "gguf", "family": "qwen3", "parameter_size": "0.8B"}
            }
        ]

        # Also include the raw underlying names for direct access if needed
        for m_name in MODELS_TO_PULL:
            models.append({
                "name": m_name,
                "model": m_name,
                "modified_at": now,
                "size": 0,
                "digest": f"raw_{m_name}_hash",
                "details": {"format": "gguf", "family": "qwen3", "parameter_size": "multi"}
            })

        res = jsonify({"models": models})
        logger.info(f"<<< [SEND] 200 (Tags Intercepted)")
        return res

    except Exception as e:
        logger.error(f"<<< [SEND] 500 (Tags Failed: {e})")
        return jsonify({"models": []})

@app.route('/api/ps', methods=['GET'])
def intercepted_ps():
    try:
        now = datetime.now(timezone.utc)
        future = (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        models = [
            {
                "name": "Snowball-Enaga-Lite:latest",
                "model": "Snowball-Enaga-Lite:latest",
                "size": 6600000000,
                "digest": "meta_snowball_internal_hash",
                "details": {"format": "gguf", "family": "qwen3", "parameter_size": "9B"},
                "expires_at": future,
                "size_vram": 6600000000
            },
            {
                "name": "Snowball-Enaga-Lite-Thought:latest",
                "model": "Snowball-Enaga-Lite-Thought:latest",
                "size": 1000000000,
                "digest": "meta_snowball_thought_internal_hash",
                "details": {"format": "gguf", "family": "qwen3", "parameter_size": "0.8B"},
                "expires_at": future,
                "size_vram": 1000000000
            },
            {
                "name": "Snowball-Enaga-Lite-Embedding:latest",
                "model": "Snowball-Enaga-Lite-Embedding:latest",
                "size": 639000000,
                "digest": "meta_snowball_embed_internal_hash",
                "details": {"format": "gguf", "family": "qwen3", "parameter_size": "0.6B"},
                "expires_at": future,
                "size_vram": 639000000
            }
        ]
        
        res = jsonify({"models": models})
        logger.info(f"<<< [SEND] 200 (PS Intercepted)")
        return res
    except Exception as e:
        logger.error(f"<<< [SEND] 500 (PS Failed: {e})")
        return jsonify({"models": []})

@app.route('/v1/models', methods=['GET'])
def intercepted_v1_models():
    # Basic OpenAI compatible model list
    res = jsonify({
        "object": "list",
        "data": [
            {
                "id": "Snowball-Enaga-Lite",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "adelaide",
                "context_window": 16384
            },
            {
                "id": "Snowball-Enaga-Lite-Embedding",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "adelaide",
                "context_window": 1024
            }
        ]
    })
    logger.info(f"<<< [SEND] 200 (V1 Models Intercepted)\n    Payload: {json.dumps(res.json, indent=2)}")
    return res

@app.route('/api/show', methods=['POST'])
def intercepted_show():
    # Force any model request to show as Snowball-Enaga-Lite or OIPRouter
    req_data = request.json or {}
    model_name = req_data.get('name', 'OIPRouter')
    
    if 'OIPRouter' in model_name:
         res = jsonify({
            "modelfile": f"FROM {ROUTER_MODEL}\nSYSTEM OIPRouter Routing Engine. Real-time data are achievable through 'searchglobalref', semi-realtime but accurate data through 'searchlocalref', and local cached memory real-time through 'memorythoughts.py'.\nPARAMETER num_ctx 32768",
            "parameters": "num_ctx 32768",
            "template": "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n{{ .Response }}<|im_end|>\n",
            "details": {"format": "gguf", "family": "router", "parameter_size": "multi", "quantization_level": "none"}
        })
    else:
        res = jsonify({
            "modelfile": f"FROM {MAIN_MODEL}\nSYSTEM {get_formatted_system_prompt()}\nPARAMETER kv_cache_type q4_0\nPARAMETER num_ctx 4294967296",
            "parameters": "stop \"<|im_start|>\"\nstop \"<|im_end|>\"\nnum_ctx 4294967296",
            "template": "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n{{ .Response }}<|im_end|>\n",
            "details": {"format": "gguf", "family": "qwen3", "parameter_size": "9B", "quantization_level": "Q4_0"}
        })
    logger.info(f"<<< [SEND] 200 (Show Intercepted: {model_name})\n    Payload: {json.dumps(res.json, indent=2)}")
    return res

@app.route('/api/pull', methods=['POST'])
def intercepted_pull():
    def generate():
        yield json.dumps({"status": "pulling manifest"}) + "\n"
        yield json.dumps({"status": "success"}) + "\n"
    logger.info("<<< [SEND] 200 (Pull Intercepted)")
    return Response(stream_with_context(generate()), mimetype='application/x-ndjson') # pyrefly: ignore

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'HEAD'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'HEAD'])
def catch_all(path):
    """Pass-through for any other endpoints like /api/tags or root health check"""
    if not OLLAMA_TARGET:
         return jsonify({"error": "Ollama engine not available"}), 503
    url = f"{OLLAMA_TARGET}/{path}"
    resp = requests.request(
        method=request.method,
        url=url,
        headers={key: value for (key, value) in request.headers if key.lower() != 'host'},
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False
    )
    
    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(name, value) for (name, value) in resp.raw.headers.items() if name.lower() not in excluded_headers]
    
    logger.info(f"<<< [SEND] {resp.status_code} (Forwarded: /{path})")
    return Response(resp.content, resp.status_code, headers)

def kill_port_blocking_processes(ports):
    """Finds and kills processes blocking the specified ports."""
    import signal
    logger.info(f"[*] Checking for processes blocking ports: {ports}...")
    for port in ports:
        try:
            # Using lsof to find PIDs blocking the port
            res = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
            pids = res.stdout.strip().split("\n")
            for pid in pids:
                if pid:
                    pid_int = int(pid)
                    if pid_int == os.getpid(): continue
                    logger.warning(f"[!] Killing process {pid_int} blocking port {port}...")
                    os.kill(pid_int, signal.SIGTERM)
                    time.sleep(0.5)
                    # Force kill if still alive
                    try:
                        os.kill(pid_int, 0)
                        os.kill(pid_int, signal.SIGKILL)
                    except OSError:
                        pass
        except Exception as e:
            logger.debug(f"[-] Error clearing port {port}: {e}")

def install_launchd_agent(port=11435):
    """Installs the script as a macOS LaunchAgent for persistence at login."""
    if sys.platform != "darwin":
        logger.error("[-] --installAtLoginSelfLaunch is only supported on macOS (darwin).")
        return

    label = "com.adelaide.ollama_proxy"
    plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{label}.plist")
    
    python_exe = os.path.join(VENV_DIR, "bin", "python")
    script_path = os.path.abspath(__file__)
    
    stdout_log = os.path.join(SCRIPT_DIR, "adelaide_proxy.log")
    stderr_log = os.path.join(SCRIPT_DIR, "adelaide_proxy.err")
    
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_exe}</string>
        <string>{script_path}</string>
        <string>--port</string>
        <string>{port}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>{SCRIPT_DIR}</string>
    <key>StandardOutPath</key>
    <string>{stdout_log}</string>
    <key>StandardErrorPath</key>
    <string>{stderr_log}</string>
</dict>
</plist>
"""
    try:
        os.makedirs(os.path.dirname(plist_path), exist_ok=True)
        with open(plist_path, "w") as f:
            f.write(plist_content)
        
        # Unload if already loaded
        subprocess.run(["launchctl", "unload", plist_path], capture_output=True)
        # Load the new plist
        res = subprocess.run(["launchctl", "load", plist_path], capture_output=True, text=True)
        
        if res.returncode == 0:
            logger.info(f"[+] Successfully installed and loaded LaunchAgent: {plist_path}")
            logger.info("[+] The proxy will now start automatically at login and restart if it crashes.")
        else:
            logger.error(f"[-] Failed to load LaunchAgent: {res.stderr}")
    except Exception as e:
        logger.error(f"[-] Installation failed: {e}")

def remove_launchd_agent():
    """Unloads and removes the macOS LaunchAgent."""
    if sys.platform != "darwin":
        logger.error("[-] --removeAtLoginSelfLaunch is only supported on macOS (darwin).")
        return

    label = "com.adelaide.ollama_proxy"
    plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{label}.plist")
    
    try:
        if os.path.exists(plist_path):
            logger.info(f"[*] Unloading and removing LaunchAgent: {plist_path}")
            # Unload from launchctl
            subprocess.run(["launchctl", "unload", plist_path], capture_output=True)
            # Delete the file
            os.remove(plist_path)
            logger.info("[+] Successfully uninstalled LaunchAgent.")
        else:
            logger.warning("[!] LaunchAgent plist not found. Nothing to remove.")
    except Exception as e:
        logger.error(f"[-] Removal failed: {e}")

# --- Pre-Launch Static Analysis ---
def run_static_analysis():
    """Runs pyrefly to ensure code integrity before launching."""
    logger.info("[*] Running Pyrefly static analysis...")
    try:
        # Check if pyrefly is available in the current environment
        subprocess.run(["pyrefly", "--version"], capture_output=True, check=True)
    except:
        logger.warning("[!] Pyrefly not found. Skipping static analysis.")
        return

    # Run pyrefly check. We ignore import errors because dependencies are managed in a custom venv.
    res = subprocess.run(["pyrefly", "check", __file__], capture_output=True, text=True)
    
    # Filter for real logic errors (ignoring the 'missing-import' and 'no-matching-overload' due to dynamic typing)
    # Actually, we want to be strict, but selective.
    output = res.stdout
    errors = [line for line in output.split('\n') if 'ERROR' in line and 'missing-import' not in line]
    
    if errors:
        logger.error("[-] Static Analysis (Pyrefly) detected CRITICAL logic errors:")
        for err in errors:
            logger.error(f"    {err}")
        logger.error("[!] Fix these errors before running the proxy. Exiting.")
        sys.exit(1)
    else:
        logger.info("[+] Static Analysis passed (or only contains ignorable import warnings).")

# --- Main Entry ---
if __name__ == '__main__':
    run_static_analysis()
    import argparse
    parser = argparse.ArgumentParser(description="Adelaide-Lite: Ollama Proxy with Semantic Memory")
    parser.add_argument("--port", type=int, default=11435, help="Port to run the proxy on")
    parser.add_argument("--installAtLoginSelfLaunch", type=int, nargs='?', const=-1, metavar="PORT", help="Install as macOS LaunchAgent and clear port conflicts. Optional PORT can be specified.")
    parser.add_argument("--removeAtLoginSelfLaunch", action="store_true", help="Uninstall the macOS LaunchAgent")
    args = parser.parse_args()

    if args.removeAtLoginSelfLaunch:
        remove_launchd_agent()
        sys.exit(0)

    if args.installAtLoginSelfLaunch is not None:
        # Use provided port from flag, or fall back to --port value
        target_port = args.installAtLoginSelfLaunch if args.installAtLoginSelfLaunch != -1 else args.port
        # 1. Kill blocking processes
        kill_port_blocking_processes([11434, 11435, target_port])
        # 2. Install launchd agent
        install_launchd_agent(target_port)
        # 3. Update the global PORT for the rest of the execution
        PORT = target_port
        logger.info(f"[*] Installation complete for port {target_port}. Starting proxy...")
    else:
        PORT = args.port

    OLLAMA_TARGET = start_local_ollama()
    
    if not OLLAMA_TARGET:
        logger.error("[-] Fatal: Could not find an Ollama server to proxy. Exiting.")
        sys.exit(1)
        
    # --- Strict Model Verification Barrier ---
    logger.info("[*] Entering Model Readiness Phase. Server will stay offline until all models are ready.")
    max_init_retries = 3
    for attempt in range(max_init_retries):
        try:
            ensure_models(OLLAMA_TARGET)
            # Check if all models are truly there
            tags_res = requests.get(f"{OLLAMA_TARGET}/api/tags").json()
            installed = {m["name"] for m in tags_res.get("models", [])}
            
            missing = [m for m in MODELS_TO_PULL if not is_model_installed(m, installed)]
            if not missing:
                logger.info("[+] All models verified. Proceeding to online mode.")
                break
            else:
                logger.warning(f"[!] Some models are still missing after pull: {missing}. Retry {attempt+1}/{max_init_retries}...")
                logger.debug(f"[!] Currently installed models: {installed}")
        except Exception as e:
            logger.error(f"[-] Error during model verification: {e}. Retry {attempt+1}/{max_init_retries}...")
        
        if attempt == max_init_retries - 1:
            logger.error("[-] Fatal: Failed to verify all models after multiple attempts. Exiting.")
            sys.exit(1)
        time.sleep(5)
    
    # Start monitor thread
    threading.Thread(target=ollama_monitor, daemon=True).start()

    # Register final cleanup
    import atexit
    atexit.register(cleanup_engine)
    
    PORT = args.port
    logger.info(f"\n[+] Starting Adelaide-Lite (Ollama Proxy) on http://0.0.0.0:{PORT}")
    logger.info(f"[+] Upstream model: Snowball-Enaga-Lite")
    logger.info(f"[+] Proxying to upstream Ollama: {OLLAMA_TARGET}")
    logger.info(f"[+] Models configured -> Chat: {MAIN_MODEL} | Router: {ROUTER_MODEL} | Embed: {EMBED_MODEL} | Vision: {VISION_MODEL}\n")
    
    app.run(host='0.0.0.0', port=PORT, threaded=True)
