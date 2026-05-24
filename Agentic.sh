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

# Determine the instruction file path
INSTRUCTION_FILE="${4:-./Instruction.md}"

if [ ! -f "$INSTRUCTION_FILE" ]; then
    echo "[!] Warning: Instruction file '$INSTRUCTION_FILE' not found. Using empty baseline."
    INSTRUCTION_PROMPT="Start a new agentic session."
else
    echo "[*] Using Instruction: $INSTRUCTION_FILE"
    INSTRUCTION_PROMPT="I want you to read $INSTRUCTION_FILE for this session. (Don't forget to use absolute_path to read this)"
fi

# Execute the agentic command
qwen -y --prompt-interactive "$INSTRUCTION_PROMPT"
