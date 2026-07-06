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
