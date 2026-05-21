#!/bin/bash

# Agentic.sh - Shim for launching Adelaide-Lite Agentic reasoning
# Usage: ./Agentic.sh [workspace_dir] [port]

WORKSPACE="${1:-$(pwd)}"
PORT="${2:-11420}"
MODEL="${3:-adelaide-hybrid}"

# Ensure workspace exists
if [ ! -d "$WORKSPACE" ]; then
    echo "[!] Error: Workspace directory '$WORKSPACE' does not exist."
    exit 1
fi

echo "[*] Launching Adelaide Agentic Loop..."
echo "[*] Workspace: $WORKSPACE"
echo "[*] Endpoint:  http://127.0.0.1:$PORT/v1"
echo "[*] Model:     $MODEL"

# Environment Setup for OpenAI-compatible client (qwen-cli)
export OPENAI_API_KEY="xd"
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_MODEL="$MODEL"

# Navigate to the target operation directory
cd "$WORKSPACE"

# Execute the agentic command
# Note: Using the user-provided instruction path as the session baseline
qwen -y --prompt-interactive \
    "I want you to read @/Users/albertstarfield/Documents/JournalingNotebook/JournalingNotebook/midnighthelper/Instruction.md for this session. (Don't forget to use absolute_path to read this)"
