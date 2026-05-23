#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import hashlib
import platform
import signal
import shutil
import glob

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Globals to keep track of background processes
daemon_process = None
server_process = None

def get_files_to_hash():
    patterns = [
        "src/**/*",
        "config/**/*",
        "adelaide_lite.gpr",
        "ui/frontend/src/**/*",
        "ui/frontend/index.html",
        "ui/frontend/package.json"
    ]
    files = []
    for pattern in patterns:
        path = os.path.join(BASE_DIR, pattern)
        if "/**/" in pattern:
            # Recursive glob isn't strictly needed if we just os.walk, but let's do a simple recursive collect
            base = path.split("/**/")[0]
            if os.path.exists(base):
                for root, _, filenames in os.walk(base):
                    for name in filenames:
                        files.append(os.path.join(root, name))
        else:
            if os.path.exists(os.path.join(BASE_DIR, pattern)):
                files.append(os.path.join(BASE_DIR, pattern))
    return sorted(files)

def calculate_hash(file_paths):
    hasher = hashlib.md5()
    for file_path in file_paths:
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                # To closely mimic the bash find | sort | xargs md5 -q
                # we hash the contents of the files in sorted order
                hasher.update(f.read())
    return hasher.hexdigest()

def cleanup(signum=None, frame=None):
    print("\n[*] Shutting down background processes...")
    if daemon_process and daemon_process.poll() is None:
        daemon_process.terminate()
    if server_process and server_process.poll() is None:
        server_process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def main():
    global daemon_process, server_process
    
    print(f"[*] Setting up Adelaide-Lite environment in {BASE_DIR}...")
    start_time = int(time.time() * 1000)

    # Calculate Current Hash
    current_hash = calculate_hash(get_files_to_hash())
    hash_file = os.path.join(BASE_DIR, ".build_hash")
    
    saved_hash = ""
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            saved_hash = f.read().strip()

    daemon_build_flag = ""

    if current_hash != saved_hash:
        print("[*] Changes detected, checking downloads and rebuilding...")
        
        # Check and clone llama.cpp
        llama_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "llama.cpp"))
        if not os.path.exists(llama_dir):
            print("[*] Cloning llama.cpp...")
            subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", llama_dir], check=False)
        else:
            print("[*] llama.cpp already exists, skipping clone.")

        # Check and clone supertonic
        supertonic_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "supertonic"))
        if not os.path.exists(supertonic_dir):
            print("[*] Cloning supertonic...")
            subprocess.run(["git", "clone", "https://github.com/supertone-inc/supertonic.git", supertonic_dir], check=False)
        else:
            print("[*] supertonic already exists, skipping clone.")

        # Ensure Deno Playwright Chromium is installed
        print("[*] Installing Playwright Chromium binary for Deno crawler...")
        # Cross platform deno invocation
        deno_cmd = "deno.exe" if platform.system() == "Windows" else "deno"
        try:
            subprocess.run([deno_cmd, "run", "-A", "npm:playwright", "install", "chromium"], check=False)
        except FileNotFoundError:
            print("[!] Deno not found in PATH, skipping playwright installation.")

        print("[*] Resolving Ada dependencies and building project...")
        
        env = os.environ.copy()
        if platform.system() == "Darwin":
            try:
                sdk_path = subprocess.check_output(["xcrun", "--show-sdk-path"]).decode().strip()
                env["SDKROOT"] = sdk_path
                env["CPATH"] = os.path.join(sdk_path, "usr", "include")
                env["LIBRARY_PATH"] = os.path.join(sdk_path, "usr", "lib")
            except Exception as e:
                print(f"[!] Warning: Could not set macOS SDK paths: {e}")

        # Note for future agents: The user strictly wants Alire to use the local alirevenv
        alr_cmd = "alr.exe" if platform.system() == "Windows" else "alr"
        subprocess.run([alr_cmd, "build"], env=env, cwd=BASE_DIR, check=True)
        
        print("[*] Building Vite Frontend for Sidecar UI...")
        frontend_dir = os.path.join(BASE_DIR, "ui", "frontend")
        if os.path.exists(frontend_dir):
            npm_cmd = "npm.cmd" if platform.system() == "Windows" else "npm"
            subprocess.run([npm_cmd, "install"], cwd=frontend_dir, check=True)
            subprocess.run([npm_cmd, "run", "build"], cwd=frontend_dir, check=True)
        
        with open(hash_file, "w") as f:
            f.write(current_hash)
    else:
        print("[*] No changes detected, skipping build.")
        daemon_build_flag = "--skip-build"

    # Parse arguments
    launch_gui = True
    if "--no-gui" in sys.argv:
        launch_gui = False

    print("[*] Booting StellaIcarus Ada Daemon Manager...")
    python_cmd = sys.executable
    daemon_script = os.path.join(BASE_DIR, "python", "stellaicarus_daemon_runner.py")
    
    daemon_args = [python_cmd, daemon_script]
    if daemon_build_flag:
        daemon_args.append(daemon_build_flag)
        
    daemon_process = subprocess.Popen(daemon_args, cwd=BASE_DIR)

    print("[*] Booting Adelaide Intelligence Server...")
    end_time = int(time.time() * 1000)
    print(f"[*] Startup completed in {end_time - start_time}ms (WCET)")

    server_bin = "adelaide_server.exe" if platform.system() == "Windows" else "adelaide_server"
    server_path = os.path.join(BASE_DIR, "bin", server_bin)

    if launch_gui:
        server_process = subprocess.Popen([server_path], cwd=BASE_DIR)
        print("[*] Booting Python Sidecar UI...")
        ui_dir = os.path.join(BASE_DIR, "ui")
        
        # Check for venv python
        venv_python_win = os.path.join(BASE_DIR, "pyvenv", "Scripts", "python.exe")
        venv_python_unix = os.path.join(BASE_DIR, "pyvenv", "bin", "python")
        
        if os.path.exists(venv_python_win):
            sidecar_python = venv_python_win
        elif os.path.exists(venv_python_unix):
            sidecar_python = venv_python_unix
        else:
            sidecar_python = python_cmd
            
        subprocess.run([sidecar_python, "sidecar_ui.py"], cwd=ui_dir)
    else:
        subprocess.run([server_path], cwd=BASE_DIR)
        
    # Wait for background processes to finish if main blocking process exits
    cleanup()

if __name__ == "__main__":
    main()
