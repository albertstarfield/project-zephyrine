#!/bin/bash

# Configuration
ENDPOINT="localhost:11434/v1/chat/completions"
MODEL_NAME="Amaryllis-Adelaide-LegacyMoEArch-IdioticRecursiveLearner-FlaskStream" # Or your stream model
SESSION_BASE_ID="pauli_stress_test_$(date +%s)"
TOTAL_OPERATIONS=1000
ROWS_PER_OPERATOR=100
STREAM_ENABLED=true # Set to false to test non-streaming if needed

# Operators cycle: +, -, *, /
OPERATORS=("+" "-" "*" "/")
NUM_OPERATORS=${#OPERATORS[@]}

# Function to generate a random number (integer or float)
get_random_number() {
  # Generate integer between 1 and 1000 or float between 1.0 and 100.0
  if (( RANDOM % 2 == 0 )); then
    echo "$((RANDOM % 1000 + 1))"
  else
    printf "%.2f" "$(bc -l <<< "scale=2; $RANDOM/327.67 * 99 + 1")" # Generates float between 1.00 and 100.00
  fi
}

# --- Main Test Loop ---
successful_requests=0
failed_requests=0
total_request_time_ms=0

echo "Starting Kraepelin/Pauli-like Stress Test..."
echo "Targeting Endpoint: $ENDPOINT"
echo "Total Operations: $TOTAL_OPERATIONS"
echo "Rows per Operator Block: $ROWS_PER_OPERATOR"
echo "Streaming: $STREAM_ENABLED"
echo "----------------------------------------------------"

overall_start_time=$(date +%s%N) # Nanosecond precision for overall timing

for (( i=0; i<$TOTAL_OPERATIONS; i++ )); do
  operator_index=$(( (i / ROWS_PER_OPERATOR) % NUM_OPERATORS ))
  current_operator=${OPERATORS[$operator_index]}

  num1=$(get_random_number)
  num2=$(get_random_number)

  # Ensure num2 is not zero for division, and not too small to cause issues
  if [[ "$current_operator" == "/" ]]; then
    while [[ "$num2" == "0" || "$num2" == "0.0" || "$num2" == "0.00" || $(echo "$num2 < 0.01" | bc -l) -eq 1 ]]; do
      # echo "  DEBUG: Regenerating num2 for division (was $num2)" # Optional debug
      num2=$(get_random_number)
    done
  fi

  prompt_content="calc ${num1} ${current_operator} ${num2}"
  session_id="${SESSION_BASE_ID}_op_${i}"

  # Construct JSON payload
  json_payload=$(cat <<EOF
{
  "model": "$MODEL_NAME",
  "messages": [
    {
      "role": "user",
      "content": "$prompt_content"
    }
  ],
  "stream": $STREAM_ENABLED,
  "session_id": "$session_id"
}
EOF
)

  request_start_time_ns=$(date +%s%N)

  # Execute curl command
  # For streaming, we're interested in the fact that it completes without error.
  # For non-streaming, we could check the content.
  # Using -s for silent, -o /dev/null to discard output (we only care about success/timing)
  # Using -w to extract HTTP code and total time for the request
  http_code_and_time=$(curl -s -N -w "%{http_code}:%{time_total}" \
    -H "Content-Type: application/json" \
    -d "$json_payload" \
    "$ENDPOINT" -o /dev/null) # -N is for streaming, -o /dev/null discards body

  request_end_time_ns=$(date +%s%N)
  request_duration_ns=$((request_end_time_ns - request_start_time_ns))
  request_duration_ms=$(bc -l <<< "scale=3; $request_duration_ns / 1000000")
  total_request_time_ms=$(bc -l <<< "scale=3; $total_request_time_ms + $request_duration_ms")


  http_code=$(echo "$http_code_and_time" | cut -d':' -f1)
  # time_total_curl=$(echo "$http_code_and_time" | cut -d':' -f2) # This is curl's own timing

  if [[ "$http_code" -eq 200 ]]; then
    successful_requests=$((successful_requests + 1))
  else
    failed_requests=$((failed_requests + 1))
    echo "  ERROR: Request $((i+1)) failed! HTTP Code: $http_code. Prompt: '$prompt_content'"
    # Optionally, add a small delay on error to avoid overwhelming a failing server
    # sleep 0.1
  fi

  # Progress Report (every 10 operations or so)
  if (( (i+1) % 10 == 0 || (i+1) == TOTAL_OPERATIONS )); then
    progress_percent=$(( (i+1) * 100 / TOTAL_OPERATIONS ))
    current_elapsed_ns=$(( $(date +%s%N) - overall_start_time ))
    current_elapsed_s=$(bc -l <<< "scale=2; $current_elapsed_ns / 1000000000")
    avg_req_time_ms=$(bc -l <<< "scale=3; if($successful_requests + $failed_requests > 0) { $total_request_time_ms / ($successful_requests + $failed_requests) } else { 0 }")

    printf "Progress: %3d%% (%5d/%d) | Elapsed: %s s | Last prompt: '%-25s' | HTTP: %s | AvgReqTime: %.3f ms\n" \
      "$progress_percent" "$((i+1))" "$TOTAL_OPERATIONS" "$current_elapsed_s" "$prompt_content" "$http_code" "$avg_req_time_ms"
  fi

done

overall_end_time=$(date +%s%N)
overall_duration_ns=$((overall_end_time - overall_start_time))
overall_duration_s=$(bc -l <<< "scale=3; $overall_duration_ns / 1000000000")

echo "----------------------------------------------------"
echo "Stress Test Complete!"
echo "Total Operations Attempted: $TOTAL_OPERATIONS"
echo "Successful Requests: $successful_requests"
echo "Failed Requests: $failed_requests"
echo "Total Time Taken: $overall_duration_s seconds"

if (( TOTAL_OPERATIONS > 0 )); then
  avg_ops_per_second=$(bc -l <<< "scale=2; $TOTAL_OPERATIONS / $overall_duration_s")
  avg_time_per_op_ms=$(bc -l <<< "scale=3; $total_request_time_ms / $TOTAL_OPERATIONS")
  echo "Average Operations Per Second: $avg_ops_per_second ops/s"
  echo "Average Request Time (client-side): $avg_time_per_op_ms ms/op"
fi
echo "----------------------------------------------------"