#!/bin/bash
set +x
# --- Configuration ---
TARGET_URL="http://127.0.0.1:11434/v1/chat/completions"
MODEL_NAME="Zephy-Direct0.6-async-51.0B"
TOTAL_REQUESTS=1000
# ---------------------

echo "🚀 STARTING SEQUENTIAL KRAEPELIN TEST"
echo "🎯 Target: $TARGET_URL"
echo "🔢 Requests: $TOTAL_REQUESTS"
echo "---------------------------------------------------"

for ((i=1; i<=TOTAL_REQUESTS; i++)); do
    # Generate random numbers for the math problem
    A=$((RANDOM % 100))
    B=$((RANDOM % 100))
    OPS=("+" "-" "*" "/")
    OP=${OPS[$((RANDOM % 4))]}
    
    # Avoid division by zero
    if [ "$OP" == "/" ] && [ "$B" -eq 0 ]; then B=1; fi

    PROMPT="Calculate $A $OP $B. Reply with ONLY the number."

    PROMPT="Calculate $A $OP $B"
    
    # Construct JSON payload
    # We use jq if available for safety, but raw string is faster for bash here
    JSON_DATA=$(cat <<EOF
{
  "model": "$MODEL_NAME",
  "stream": false,
  "messages": [
    { "role": "user", "content": "$PROMPT" }
  ]
}
EOF
)

    # Perform Request
    # -s : Silent (no progress bar)
    # -o /dev/null : Discard the response body (we just want to know if it worked)
    # -w : Write out custom format (HTTP Code and Total Time)
    # --max-time 30 : Timeout safety
    
    RESULT=$(curl -s -X POST "$TARGET_URL" \
         -H "Content-Type: application/json" \
         -d "$JSON_DATA" \
         -o /dev/null \
         -w "%{http_code},%{time_total}")

    echo $RESULT
    # Parse result (Format is "200,0.123456")
    HTTP_CODE=$(echo "$RESULT" | cut -d',' -f1)
    TIME_TAKEN=$(echo "$RESULT" | cut -d',' -f2)

    if [ "$HTTP_CODE" -eq 200 ]; then
        STATUS="✅ Success"
    else
        STATUS="❌ Failed ($HTTP_CODE)"
    fi

    echo "[$i/$TOTAL_REQUESTS] $STATUS | Time: ${TIME_TAKEN}s | Q: $A $OP $B"
done

echo "---------------------------------------------------"
echo "🏁 Test Complete."
