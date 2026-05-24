#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import time
import json
from datetime import datetime

# --- Environment Setup ---
def apply_base_env():
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
VENV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyvenv")
REQUIREMENTS = ["numpy", "requests"]

def bootstrap_venv():
    """Ensures the script runs in its dedicated virtual environment."""
    apply_base_env()
    venv_abs = os.path.abspath(VENV_DIR)
    
    # If not in the correct venv, ensure it exists and switch to it
    if os.path.abspath(sys.prefix) != venv_abs:
        if not os.path.exists(VENV_DIR):
            print(f"[*] Creating virtual environment in {VENV_DIR}...", file=sys.stderr)
            subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
            
        if os.name == 'nt':
            python_exe = os.path.join(VENV_DIR, "Scripts", "python.exe")
        else:
            python_exe = os.path.join(VENV_DIR, "bin", "python")
        
        if os.path.exists(python_exe):
            os.execv(python_exe, [python_exe] + sys.argv)

    # Once inside the venv, verify requirements
    try:
        import numpy
        import requests
    except ImportError:
        print(f"[*] Missing dependencies. Installing: {', '.join(REQUIREMENTS)}...", file=sys.stderr)
        if os.name == 'nt':
            pip_exe = os.path.join(VENV_DIR, "Scripts", "pip.exe")
        else:
            pip_exe = os.path.join(VENV_DIR, "bin", "pip")
        subprocess.run([pip_exe, "install", "--upgrade", "pip"], check=True)
        subprocess.run([pip_exe, "install"] + REQUIREMENTS, check=True)
        # Re-execute one last time to pick up new packages
        os.execv(sys.executable, [sys.executable] + sys.argv)

bootstrap_venv()

# --- Post-Bootstrap Imports ---
import numpy as np

# --- Ollama Configuration ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_PROXY_URL", "http://localhost:11435")
OLLAMA_EMBED_ENDPOINT = f"{OLLAMA_BASE_URL}/api/embed"
OLLAMA_MODEL = "qwen3-embedding:0.6b"

# --- Helper Functions ---

def generate_apa7_reference(title, url):
    today = datetime.now().strftime("%Y, %B %d")
    clean_title = str(title).strip().rstrip('.')
    return f"{clean_title}. (Fetched: {today}). {url}"

def ensure_ollama_running():
    import requests
    try:
        requests.get(f"{OLLAMA_BASE_URL}", timeout=2)
        print(f"✅ Ollama reachable at {OLLAMA_BASE_URL}", file=sys.stderr)
        return True
    except:
        print("⚠️ Ollama not reachable. Attempting restart...", file=sys.stderr)
        subprocess.run(["launchctl", "setenv", "OLLAMA_HOST", "0.0.0.0:1234"], check=False)
        subprocess.run(["brew", "services", "restart", "ollama"], check=False)
        time.sleep(3)
        try:
            requests.get(f"{OLLAMA_BASE_URL}", timeout=2)
            return True
        except: return False

def get_embedding(text: str):
    import requests
    if not text: return None
    try:
        resp = requests.post(OLLAMA_EMBED_ENDPOINT, json={"model": OLLAMA_MODEL, "input": text}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "embeddings" in data and len(data["embeddings"]) > 0:
            return np.array(data["embeddings"][0])
        elif "embedding" in data:
            return np.array(data["embedding"])
        return None
    except: return None

def store_in_memory(content, ollama_external=None):
    """Invokes memorythoughts.py to store content."""
    try:
        memory_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memorythoughts.py")
        cmd = [sys.executable, memory_script, "--string", content]
        if ollama_external:
            cmd.extend(["--ollamaHost", ollama_external])
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"⚠️ Failed to store memory: {e}", file=sys.stderr)

def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--engines", nargs='*', default=['all'])
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--jsonIO", action="store_true", help="Output results in JSON format.")
    parser.add_argument("--ollamaExternal", type=str, default=None, help="Custom Ollama server address.")
    parser.add_argument("--ollamaHost", type=str, default=None, help="Custom Ollama host address.")
    args = parser.parse_args()

    global OLLAMA_BASE_URL, OLLAMA_EMBED_ENDPOINT
    host = args.ollamaHost or args.ollamaExternal
    if host:
        OLLAMA_BASE_URL = host if host.startswith("http") else f"http://{host}"
        OLLAMA_EMBED_ENDPOINT = f"{OLLAMA_BASE_URL}/api/embed"

    engines_str = ",".join(args.engines)

    if args.jsonIO:
        print(json.dumps({"phase": 1, "status": "start", "query": args.query}), flush=True)
    else:
        print(f"[*] Dispatching Deno Playwright Scraper for '{args.query}'...", file=sys.stderr)

    # Spawn Deno Sidecar
    scraper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "playwright_scraper.ts")
    cmd = ["deno", "run", "-A", scraper_path, args.query, f"--engines={engines_str}", f"--num={args.num}"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        raw_output = result.stdout.strip()
        # Find the last valid JSON array in stdout (in case Deno printed warnings)
        json_str = "[]"
        for line in reversed(raw_output.splitlines()):
            if line.startswith("[") and line.endswith("]"):
                json_str = line
                break
        all_flat = json.loads(json_str)
    except Exception as e:
        print(f"⚠️ Deno Scraper failed: {e}", file=sys.stderr)
        all_flat = []

    # Inject APA7 references
    for r in all_flat:
        r['apa7_reference'] = generate_apa7_reference(r.get('title', 'Unknown'), r.get('url', ''))

    if args.jsonIO:
        print(json.dumps({"phase": 1, "status": "complete", "results": all_flat}), flush=True)
        print(json.dumps({"phase": 2, "status": "start"}), flush=True)

    ollama_ready = ensure_ollama_running()

    final_results = []
    if ollama_ready and all_flat:
        if not args.jsonIO:
            print(f"[*] Ranking {len(all_flat)} results semantically...", file=sys.stderr)
        q_emb = get_embedding(args.query)
        if q_emb is not None:
            ranked = []
            for r in all_flat:
                # Use snippet if available, otherwise just title
                text_to_embed = f"{r.get('title', '')} {r.get('snippet', '')}"
                r_emb = get_embedding(text_to_embed)
                score = np.dot(q_emb, r_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(r_emb)) if r_emb is not None else 0
                ranked.append((float(score), r))
            ranked.sort(key=lambda x: x[0], reverse=True)
            final_results = [x[1] for x in ranked[:7]]
            for i, r in enumerate(final_results): 
                r['semantic_rank'] = i + 1
                r['semantic_score'] = ranked[i][0]
        else: final_results = all_flat[:7]
    else: final_results = all_flat[:7]

    # --- Store in Memory ---
    for r in final_results:
        memory_content = f"Source: {r.get('url', '')}\nReference: {r.get('apa7_reference', '')}\nSnippet: {r.get('snippet', '')}"
        store_in_memory(memory_content, host)

    if args.jsonIO:
        print(json.dumps({"phase": 2, "status": "complete", "results": final_results}), flush=True)
    else:
        # --- Markdown Output ---
        print("# Global Search Results", flush=True)
        print(f"*Query: {args.query}*\n", flush=True)
        print("> ℹ️ Note: If a tool suggests re-parsing a PDF, it may be an **Invalid trigger**. Refer to the provided snippets and images. **Use these as your primary Reference.**\n", flush=True)

        for i, r in enumerate(final_results):
            print(f"## {i+1}. {r.get('title', 'Unknown')}", flush=True)
            print(f"- **URL:** {r.get('url', 'Unknown')}", flush=True)
            print(f"- **Engine:** {r.get('source_engine', 'unknown')}", flush=True)
            if 'semantic_rank' in r:
                print(f"- **Semantic Rank:** {r['semantic_rank']}", flush=True)
            print(f"- **Reference:** {r.get('apa7_reference', 'Unknown')}", flush=True)
            print(f"\n### Snippet\n{r.get('snippet', 'No snippet available.')}\n", flush=True)
            
            if r.get('screenshot_base64'):
                print(f"### Visual Evidence (Page Snapshot)\n![Page Snapshot]({r['screenshot_base64']})\n", flush=True)
            
            if r.get('web_images'):
                print("### Website Images\n", flush=True)
                for img_b64 in r['web_images']:
                    print(f"![Web Image]({img_b64})\n", flush=True)
            
            print("---\n", flush=True)

if __name__ == "__main__":
    print(f"[*] Invoked: {sys.executable} {' '.join(sys.argv)}", file=sys.stderr)
    main()
