#!/bin/bash

# Project Zephyrine Health Check and Agentic Pipeline Verification
# This script performs automated curl tests to verify system integrity.

API_BASE="http://localhost:11434"
LOG_FILE="TestRun.log"

echo "--- [HEALTH CHECK] Starting Automated Verification ---" | tee -a "$LOG_FILE"

# 1. Wait for system to be ready
echo "Waiting for system to be 'Primed and Ready'..." | tee -a "$LOG_FILE"
MAX_RETRIES=30
RETRY_COUNT=0
while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE/v1/primedready")
    if [[ "$STATUS" == "200" ]]; then
        echo "✅ System is ONLINE." | tee -a "$LOG_FILE"
        break
    fi
    echo "   ...waiting (Attempt $((RETRY_COUNT+1))/$MAX_RETRIES)"
    sleep 60
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [[ "$STATUS" != "200" ]]; then
    echo "❌ ERROR: System failed to come online within 30 minutes." | tee -a "$LOG_FILE"
    exit 1
fi

# 2. Normal Conversation Test
echo "--- [TEST] Normal Conversation ---" | tee -a "$LOG_FILE"
RESPONSE=$(curl -s -X POST "$API_BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zephyrine",
    "messages": [{"role": "user", "content": "Hello Zephy, how are you today?"}],
    "stream": false
  }')

if echo "$RESPONSE" | grep -q "choices"; then
    echo "✅ Normal conversation test PASSED." | tee -a "$LOG_FILE"
else
    echo "❌ Normal conversation test FAILED." | tee -a "$LOG_FILE"
    echo "DEBUG: $RESPONSE" | tee -a "$LOG_FILE"
fi

# 3. Agentic Pipeline Test (Triggering ELP1 escalation)
echo "--- [TEST] Agentic Pipeline Escalation ---" | tee -a "$LOG_FILE"
# We use keywords like "analyze code" and "SPARK" to trigger agentic logic
AGENTIC_RESPONSE=$(curl -s -X POST "$API_BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zephyrine",
    "messages": [{"role": "user", "content": "Analyze this Ada/SPARK code for memory safety: procedure Test is begin null; end Test;"}],
    "stream": false
  }')

if echo "$AGENTIC_RESPONSE" | grep -q "choices"; then
    echo "✅ Agentic pipeline test PASSED (Response received)." | tee -a "$LOG_FILE"
else
    echo "❌ Agentic pipeline test FAILED." | tee -a "$LOG_FILE"
    echo "DEBUG: $AGENTIC_RESPONSE" | tee -a "$LOG_FILE"
fi

echo "--- [HEALTH CHECK] Verification Complete ---" | tee -a "$LOG_FILE"
