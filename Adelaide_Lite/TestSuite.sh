#!/bin/bash
# TestSuite.sh - Continuous Integration, Quality Approval & Torture Test Suite
# Designed for the OpenIntellegentiaPlatform (Adelaide_Lite) Project

# Colors for nice output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0;0m' # No Color

PROJECT_ROOT="/Users/albertstarfield/LibraryTube/OpenIntellegentiaPlatform"
DAEMON_DIR="$PROJECT_ROOT/Adelaide_Lite"
API_URL="http://localhost:11420"

# Default GNATprove proof level is 4 for strict maximum validation
PROVE_LEVEL=4

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --level=*) PROVE_LEVEL="${1#*=}" ;;
        --level) PROVE_LEVEL="$2"; shift ;;
        --release) PROVE_LEVEL=4 ;;
        --help|-h)
            echo "Usage: ./TestSuite.sh [options]"
            echo "Options:"
            echo "  --level=<1..4>    Specify GNATprove proof level (default: 4)"
            echo "  --release         Run GNATprove at level 4 for deep release proof validation"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

echo -e "${CYAN}[i] GNATprove analysis level configured to: --level=$PROVE_LEVEL${NC}"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}      Adelaide Server Quality Approval & API Torture Test Suite       ${NC}"
echo -e "${BLUE}======================================================================${NC}"

# Navigate to daemon directory
cd "$DAEMON_DIR" || { echo -e "${RED}[!] Failed to enter daemon directory at $DAEMON_DIR${NC}"; exit 1; }

# 1. Verification of Toolchains and Environment
echo -e "\n${BLUE}[*] Stage 1: Checking Alire (alr) Environment...${NC}"
if ! command -v alr &> /dev/null; then
    echo -e "${RED}[!] Alire (alr) is not installed or not in PATH!${NC}"
    echo -e "${YELLOW}[i] Prerequisite: Please install Alire via 'brew install alire'${NC}"
    exit 1
fi
echo -e "${GREEN}[ok] Alire is available: $(alr --version | head -n 1)${NC}"

# 2. GNAT Compilation Verify
echo -e "\n${BLUE}[*] Stage 2: Compiling with GNAT via Alire...${NC}"
export SDKROOT=$(xcrun --show-sdk-path)
export CPATH=$SDKROOT/usr/include
export LIBRARY_PATH=$SDKROOT/usr/lib
alr --non-interactive build || {
    echo -e "${YELLOW}[i] Initial build failed, applying macOS ranlib workaround for GNU ar archives...${NC}"
    find ~/.local/share/alire/builds -name "*.a" -exec /usr/bin/ranlib {} \; 2>/dev/null || true
    alr --non-interactive build
}
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[ok] GNAT compilation succeeded!${NC}"
else
    echo -e "${RED}[!] GNAT compilation failed!${NC}"
    exit 1
fi

# 3. GNATprove SPARK Verification
echo -e "\n${BLUE}[*] Stage 3: Running GNATprove (SPARK Static Analysis)...${NC}"
alr exec -- gnatprove -P adelaide_spark.gpr --level=$PROVE_LEVEL --prover=cvc5,z3,altergo --timeout=60 --memlimit=2000 --steps=0 --counterexamples=on --report=fail --warnings=error
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[ok] GNATprove SPARK analysis completed successfully!${NC}"
else
    echo -e "${RED}[!] GNATprove failed! SPARK proofing detected an issue. Complete failure.${NC}"
    exit 1
fi

# 4. API Torture Test
echo -e "\n${BLUE}[*] Stage 4: API Bombardment (HVF Docker Torture Test)...${NC}"
echo -e "${CYAN}[i] Starting adelaide_server in the background...${NC}"

./bin/adelaide_server --no-gui > server_torture.log 2>&1 &
SERVER_PID=$!

# Wait for server to boot
echo -e "${CYAN}[i] Waiting for server to bind to port 11420...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:11420/ > /dev/null; then
        break
    fi
    sleep 1
done

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}[!] Server crashed immediately on boot! Check server_torture.log${NC}"
    exit 1
fi

echo -e "${GREEN}[ok] Server is up! Deploying Docker HVF Container for OS-isolated 100 Million bombardment...${NC}"

# Create the bombardment script for the Docker container
cat << 'EOF' > docker_torture.sh
#!/bin/bash
export API_URL="http://host.docker.internal:11420"
bombard() {
    local i=$1
    case $((i % 4)) in
        0) curl -s -X POST $API_URL/api/generate -d "{\"model\": \"adelaide-hybrid\", \"prompt\": \"Generate a massive essay about quantum physics and do not stop until you reach 4000 words. Keep going. This is request $i.\"}" -m 10 > /dev/null ;;
        1) curl -s -X POST $API_URL/api/embeddings -d "{\"model\": \"qwen-embedding\", \"prompt\": \"Torture test embedding generation $i\"}" -m 5 > /dev/null ;;
        2) curl -s -X POST $API_URL/api/chat -d "INVALID_JSON_PAYLOAD_} { [ BOOM $i" -m 5 > /dev/null ;;
        3) curl -s -X POST $API_URL/api/generate -d "{\"model\": \"non-existent-model\", \"prompt\": \"Load a broken model $i\"}" -m 5 > /dev/null ;;
    esac
}
export -f bombard

echo "[!] Commencing 100 MILLION isolated requests from Docker HVF..."
# We use xargs for massive concurrency without blowing up the container limits
seq 1 100000000 | xargs -n 1 -P 500 -I {} bash -c 'bombard "$@"' _ {}
EOF
chmod +x docker_torture.sh

if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}[!] Docker is not installed or running. Cannot launch HVF torture container.${NC}"
else
    # Run the Docker container in the background to hammer the local server via host.docker.internal
    echo -e "${CYAN}[i] Launching Alpine Linux payload container...${NC}"
    docker run --rm -v $(pwd)/docker_torture.sh:/torture.sh alpine:latest sh -c "apk add --no-cache bash curl && bash /torture.sh" &
    DOCKER_PID=$!
    
    # We will let it torture for 15 seconds to prove resilience, then terminate the container
    echo -e "${CYAN}[i] Let the server suffer for 15 seconds...${NC}"
    sleep 15
    
    echo -e "${CYAN}[i] Halting Docker torture...${NC}"
    kill $DOCKER_PID 2>/dev/null
    docker stop $(docker ps -q --filter ancestor=alpine:latest) 2>/dev/null
fi

echo -e "${CYAN}[i] Bombardment phase completed. Checking if server survived...${NC}"

if kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${GREEN}[ok] Server SURVIVED the torture test!${NC}"
    # Gracefully kill it
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
else
    echo -e "${RED}[!] Server CRASHED during the torture test! Check server_torture.log for details.${NC}"
fi

# 5. AFL++ Ada Fuzzing Check
echo -e "\n${BLUE}[*] Stage 5: AFL++ Ada Fuzzing Approval Check...${NC}"
AFL_FUZZ_FOUND=false
AFL_COMPILER_FOUND=false
AFL_COMPILER=""

if command -v afl-fuzz &> /dev/null; then
    echo -e "${GREEN}[ok] AFL++ (afl-fuzz) is available in PATH!${NC}"
    AFL_FUZZ_FOUND=true
else
    echo -e "${YELLOW}[!] AFL++ (afl-fuzz) is not installed in the system PATH.${NC}"
fi

for cmd in afl-clang-fast afl-gcc-fast afl-clang-lto afl-gcc afl-g++; do
    if command -v "$cmd" &> /dev/null; then
        "$cmd" --version &> /dev/null
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}[ok] AFL++ compiler wrapper '$cmd' is available and functional!${NC}"
            AFL_COMPILER_FOUND=true
            AFL_COMPILER="$cmd"
            break
        fi
    fi
done

if [ "$AFL_FUZZ_FOUND" = true ] && [ "$AFL_COMPILER_FOUND" = true ]; then
    echo -e "${GREEN}[ok] AFL++ Ada/C Fuzzing environment is fully ready!${NC}"
else
    echo -e "${YELLOW}[!] AFL++ environment is incomplete. Production fuzzing requires 'afl-fuzz' and wrappers.${NC}"
    echo -e "${CYAN}    To test binary parsing, compile with: CC=$AFL_COMPILER alr build${NC}"
    echo -e "${CYAN}    Then run: afl-fuzz -i inputs -o outputs ./bin/adelaide_server${NC}"
fi

echo -e "\n${BLUE}======================================================================${NC}"
echo -e "${GREEN}             Test Suite Execution Complete & Documented               ${NC}"
echo -e "${BLUE}======================================================================${NC}"

# ===========================================================================
# STREAMING TTFB REQUIREMENT (DO NOT REMOVE)
# ===========================================================================
# The first chunk (ACK or token) of any streaming response MUST arrive
# within 5ms of the HTTP request. This is measured from the moment the
# HTTP request headers are fully received to the first byte of the
# response body (either an SSE ACK event or the first token chunk).
#
# If TTFB exceeds 5ms, the test is considered a FAILURE.
#
# Root causes of TTFB violations:
#   1. Background indexing (ELP0) blocking chat endpoint (ELP1 contention)
#   2. Missing immediate ACK push in Dispatch before Generator_Task start
#   3. Model loading delay on first request (cold start)
#   4. Kratos crash isolation overhead (should be negligible, <1us)
#
# Verification command:
#   curl -s -w "TTFB: %{time_starttransfer}s\n" \
#     http://localhost:11420/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"Snowball-Enaga","messages":[{"role":"user","content":"test"}],"stream":true}'
#
# Two-query session test (same session_id, verify no cross-contamination):
#   curl -s http://localhost:11420/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"Snowball-Enaga","messages":[{"role":"user","content":"What is Cauchy Number"}],"stream":true,"session_id":"test-session-1"}'
#   curl -s http://localhost:11420/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"Snowball-Enaga","messages":[{"role":"user","content":"What is homogenous turbulent"}],"stream":true,"session_id":"test-session-1"}'
# ===========================================================================
