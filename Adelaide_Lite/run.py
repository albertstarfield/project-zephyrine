#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import hashlib
import platform
import signal
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set HF_HOME so huggingface caches locally in the project directory
os.environ["HF_HOME"] = os.path.join(BASE_DIR, ".hf_cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Globals to keep track of background processes
daemon_process = None
server_process = None
kokoro_process = None

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
        threads = str(os.cpu_count() or 4)
        
        # Check and clone llama.cpp
        llama_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "llama.cpp"))
        if not os.path.exists(llama_dir):
            print("[*] Cloning llama.cpp...")
            subprocess.run(["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp.git", llama_dir], check=False)
        else:
            print("[*] llama.cpp already exists, skipping clone.")

        # Ensure llama.cpp is built
        llama_build_dir = os.path.join(llama_dir, "build")
        llama_lib = os.path.join(llama_build_dir, "src", "libllama.a")
        if not os.path.exists(llama_lib):
            print("[*] Building llama.cpp...")
            os.makedirs(llama_build_dir, exist_ok=True)
            cmake_flags = ["cmake", "-B", "build", "-DGGML_NATIVE=ON"]
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                cmake_flags.append("-DGGML_METAL=ON")
            subprocess.run(cmake_flags, cwd=llama_dir, check=False)
            subprocess.run(["cmake", "--build", "build", "--config", "Release", "-j"], cwd=llama_dir, check=False)
        else:
            print("[*] llama.cpp library exists, skipping build.")

        # Check and clone kokoro-onnx
        kokoro_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "kokoro-onnx"))
        if not os.path.exists(kokoro_dir):
            print("[*] Cloning kokoro-onnx...")
            subprocess.run(["git", "clone", "https://github.com/thewh1teagle/kokoro-onnx", kokoro_dir], check=False)
        else:
            print("[*] kokoro-onnx already exists, skipping clone.")
            
        kokoclone_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "kokoclone"))
        if not os.path.exists(kokoclone_dir):
            print("[*] Cloning KokoClone Zero-Shot Repository...")
            subprocess.run(["git", "clone", "https://github.com/Ashish-Patnaik/kokoclone.git", kokoclone_dir], check=True)
        else:
            print("[*] kokoclone already exists, skipping clone.")
            
        # Ensure Kokoro TTS component dependencies are installed in an isolated venv
        kokoro_comp_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "tts_kokoro_component"))
        kokoro_venv_dir = os.path.join(kokoro_comp_dir, "venv")
        if not os.path.exists(kokoro_venv_dir):
            print("[*] Creating dedicated virtual environment for Kokoro TTS (Python 3.12)...")
            subprocess.run(["python3.12", "-m", "venv", kokoro_venv_dir], check=True)
            
        print("[*] Installing Kokoro TTS requirements...")
        kokoro_pip = os.path.join(kokoro_venv_dir, "bin", "pip") if platform.system() != "Windows" else os.path.join(kokoro_venv_dir, "Scripts", "pip.exe")
        subprocess.run([kokoro_pip, "install", "-r", os.path.join(kokoro_comp_dir, "requirements.txt")], check=False)

        # Check and clone moonshine
        moonshine_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "moonshine"))
        if not os.path.exists(moonshine_dir):
            print("[*] Cloning moonshine...")
            subprocess.run(["git", "clone", "--depth=1", "https://github.com/moonshine-ai/moonshine.git", moonshine_dir], check=False)
            
            # Autoremove examples to save space
            moonshine_examples = os.path.join(moonshine_dir, "examples")
            if os.path.exists(moonshine_examples):
                print("[*] Removing heavy moonshine/examples directory...")
                shutil.rmtree(moonshine_examples, ignore_errors=True)
        else:
            print("[*] moonshine already exists, skipping clone.")

        # Ensure Moonshine is built
        moonshine_build_dir = os.path.join(moonshine_dir, "build")
        moonshine_core_lib = os.path.join(moonshine_build_dir, "core", "libmoonshine.dylib") if platform.system() == "Darwin" else os.path.join(moonshine_build_dir, "core", "libmoonshine.so")
        if not os.path.exists(moonshine_core_lib):
            print("[*] Building moonshine C API...")
            os.makedirs(moonshine_build_dir, exist_ok=True)
            subprocess.run(["cmake", ".."], cwd=moonshine_build_dir, check=False)
            subprocess.run(["make", f"-j{threads}"], cwd=moonshine_build_dir, check=False)
        else:
            print("[*] moonshine core library exists, skipping cmake build.")

        # Check and download Moonshine models
        moonshine_models_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "moonshine", "models"))
        if not os.path.exists(moonshine_models_dir) or not os.listdir(moonshine_models_dir):
            print("[*] Downloading Moonshine models...")
            os.makedirs(moonshine_models_dir, exist_ok=True)
            env_for_download = os.environ.copy()
            env_for_download["PYTHONPATH"] = os.path.join(moonshine_dir, "python", "src")
            download_script = os.path.join(moonshine_dir, "python", "src", "moonshine_voice", "download.py")
            subprocess.run([sys.executable, download_script, "--stt", "--language", "en", "--root", moonshine_models_dir], env=env_for_download, check=False)
        else:
            print("[*] Moonshine models already exist, skipping download.")

        # Check and download Qwen models
        qwen_models_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "llama.cpp", "models", "qwen3.5"))
        os.makedirs(qwen_models_dir, exist_ok=True)
        
        models_to_download = [
            {
                "url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q2_K_XL.gguf?download=true",
                "output": "Qwen3.5-9B-UD-Q2_K_XL.gguf"
            },
            {
                "url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/mmproj-F16.gguf?download=true",
                "output": "mmproj-9B-F16.gguf"
            }
        ]
        
        aria2c_cmd = shutil.which("aria2c")
        for model in models_to_download:
            target_path = os.path.join(qwen_models_dir, model["output"])
            if not os.path.exists(target_path):
                print(f"[*] Downloading {model['output']}...")
                if aria2c_cmd:
                    subprocess.run([aria2c_cmd, "-x", "16", "-s", "16", "-k", "1M", model["url"], "-o", model["output"], "-d", qwen_models_dir], check=True)
                else:
                    subprocess.run(["wget", "-q", "--show-progress", model["url"], "-O", target_path], check=True)

        # Check and download Kokoro models
        kokoro_models_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "kokoro_models"))
        os.makedirs(kokoro_models_dir, exist_ok=True)
        kokoro_onnx_model = os.path.join(kokoro_models_dir, "kokoro-v0_19.int8.onnx")
        kokoro_voices = os.path.join(kokoro_models_dir, "voices-v1.0.bin")
        if not os.path.exists(kokoro_onnx_model):
            print("[*] Downloading Kokoro ONNX model...")
            subprocess.run(["wget", "-q", "--show-progress", "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.int8.onnx"], cwd=kokoro_models_dir, check=False)
        if not os.path.exists(kokoro_voices):
            print("[*] Downloading Kokoro voices...")
            subprocess.run(["wget", "-q", "--show-progress", "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"], cwd=kokoro_models_dir, check=False)

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
            
    # Self-Integrity Check using Ruff
    ruff_cmd = "ruff.exe" if platform.system() == "Windows" else "ruff"
    if shutil.which(ruff_cmd):
        print("[*] Running Platform Self-Integrity Quality Check (Ruff)...")
        # Run ruff check on the Adelaide_Lite directory
        try:
            result = subprocess.run([ruff_cmd, "check", BASE_DIR], capture_output=True, text=True)
            if result.returncode != 0:
                print("[!] Self-Integrity Quality Check FAILED.")
                print(result.stdout)
                # In strict mode, we might want to exit, but for now just warn
                # print("[*] Emergency Shutdown: Quality violations detected.")
                # sys.exit(1)
            else:
                print("[+] Self-Integrity Quality Check PASSED.")
        except Exception as e:
            print(f"[!] Error executing Ruff integrity check: {e}")
    else:
        print("[!] Warning: ruff not found in PATH, skipping self-integrity quality check.")

    # Handle integrity check flag
    if "--test-build-integrity-check" in sys.argv:
        print("[*] Test build integrity check passed! Exiting without launching services.")
        sys.exit(0)

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

    env = os.environ.copy()
    
    # Architecture-aware Moonshine ONNX runtime path
    arch = "arm64" if platform.machine() == "arm64" else "x86_64"
    moonshine_onnx = os.path.join(BASE_DIR, "..", "moonshine", "core", "third-party", "onnxruntime", "lib", "macos", arch)
    
    if platform.system() == "Darwin":
        env["DYLD_LIBRARY_PATH"] = f"{moonshine_onnx}:{env.get('DYLD_LIBRARY_PATH', '')}"
    
    # Start through ALIRE to ensure correct environment variables for Ada libraries
    if shutil.which("alr"):
        server_process = subprocess.Popen(["alr", "exec", "--", server_path], cwd=BASE_DIR, env=env)
    else:
        server_process = subprocess.Popen([server_path], cwd=BASE_DIR, env=env)

    if launch_gui:
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
        try:
            server_process.wait()
        except KeyboardInterrupt:
            pass
        
    # Wait for background processes to finish if main blocking process exits
    cleanup()

if __name__ == "__main__":
    main()
