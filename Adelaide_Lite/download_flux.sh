#!/bin/bash
# download_flux.sh — Resilient FLUX model downloader with infinite retry
# Resumes interrupted downloads, chains all FLUX components automatically.
# Safe to run multiple times — skips completed files.

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "$0")" && pwd)/model"
LOG="/tmp/flux_download.log"
mkdir -p "$MODEL_DIR"

download() {
    local url="$1"
    local name="$2"
    local target="$MODEL_DIR/$name"
    local expected_size="${3:-0}"
    
    if [ -f "$target" ]; then
        local actual_size
        actual_size=$(stat -f%z "$target" 2>/dev/null || echo 0)
        if [ "$expected_size" -eq 0 ] || [ "$actual_size" -ge "$expected_size" ]; then
            echo "[SKIP] $name already exists ($(du -h "$target" | cut -f1))"
            return 0
        else
            echo "[RESUME] $name incomplete ($actual_size / $expected_size bytes), resuming..."
        fi
    fi
    
    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        echo "[ATTEMPT #$attempt] Downloading $name ..."
        wget -c -t 0 --timeout=30 --waitretry=5 --tries=inf \
             --show-progress "$url" -O "$target" 2>&1 | tee -a "$LOG"
        
        if [ $? -eq 0 ] && [ -f "$target" ]; then
            echo "[OK] $name downloaded ($(du -h "$target" | cut -f1))"
            return 0
        fi
        
        echo "[RETRY] $name failed, retrying in 5s..."
        rm -f "$target" "$target.aria2"
        sleep 5
    done
}

echo "=== FLUX Schnell Model Downloads ==="
echo "Resume-safe: interrupted downloads continue where they left off"
echo "Log: $LOG"
echo ""

# 1. Diffusion model (Q2_K, ~4GB = 4010296352 bytes)
download "https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q2_K.gguf?download=true" \
         "flux1-schnell.gguf" 4010296352

# 2. Text encoder t5xxl (Q4_0 GGUF, ~2.9GB = 2924546752 bytes)
download "https://huggingface.co/Phil2Sat/T5XXL-Unchained-GGUF/resolve/main/Kaoru8-t5xxl-unchained-Q4_0.gguf?download=true" \
         "flux1-t5xxl.gguf" 2924546752

# 3. CLIP-L text encoder (safetensors, ~246MB)
download "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true" \
         "clip_l.safetensors" 0

# 4. VAE (safetensors, ~335MB)
download "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors?download=true" \
         "ae.safetensors" 0

# 5. SD refinement model (~1.9GB)
download "https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium.safetensors?download=true" \
         "sd-refinement.safetensors" 0

echo ""
echo "=== All FLUX models downloaded ==="
ls -lh "$MODEL_DIR"/flux1-* "$MODEL_DIR"/clip_l* "$MODEL_DIR"/ae.safetensors 2>/dev/null
