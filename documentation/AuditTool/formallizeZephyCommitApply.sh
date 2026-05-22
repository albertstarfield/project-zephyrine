#!/bin/bash

# --- FIX: DEFINE THE API URL HERE ---
# This was the missing variable causing the curl error.
API_URL="http://localhost:18141/v1/chat/completions"

# Function to rephrase a commit message with diff context
rephrase_commit() {
  local commit_details="$1"
  local commit_author="$2"

  if [[ -z "$commit_details" ]]; then
    echo "Error: No commit details provided to rephrase_commit function." >&2
    return 1
  fi
  if [[ -z "$commit_author" ]]; then
    echo "Warning: Commit author not found for current commit. Using 'Unknown Author'." >&2
    commit_author="Unknown Author" # Fallback
  fi

  local llm_prompt="Author for this commit: ${commit_author}\n\nRephrase the following Git commit details (including original message, list of files changed, and code diffs) into a formal CM management style. Use the provided diff information to make the description of changes more precise and concrete, referencing specific files or parts of the code where appropriate.\n\n${commit_details}\n\n"

  # Safely build the JSON payload using jq's --arg to handle special characters
  json_payload=$(jq -n \
    --arg model "google/gemma-3n-e4b" \
    --arg system_content "You are a professional technical writer specializing in creating highly detailed, formal commit messages for Configuration Management (CM) in professional avionics software development. You will be provided with full Git commit details, including the original commit message, list of files changed, and the code diffs.

    Your goal is to transform this information into a well-structured, comprehensive, and informative formal commit message following a consistent format.

    **Required Sections in the Output:**
    1.  **Title:** A clear, concise summary (under 70 characters) indicating the core change.
    2.  **Author:** The contributor's name. This will be provided as 'Author for this commit: [Name]' in the user input. You MUST use this provided name for the 'Author' section.
    3.  **Description of Changes (How it Works):** A detailed explanation of *what* was changed, *how* the change was implemented, and *how* it functions at a code level, referencing specific files, functions, or variable changes identified in the diff.
    4.  **Discovery/Rationale (How it was Discovered/Why it was Needed):** Provide a precise reasoning for the change, derived directly from the original commit message, diff content, and commit type. **Strictly avoid speculative phrasing such as 'likely', 'most likely', 'potentially', or 'it seems'.** Instead, infer or deduce the method of discovery (e.g., 'Identified during unit testing of X module', 'Discovered during code review of Y feature', 'Addressed an observed anomaly during system integration testing', 'Implemented to enhance feature Y'). If the specific method of discovery or rationale cannot be precisely deduced from the provided commit information, clearly state that 'The method of discovery/specific rationale is not explicitly detailed in the commit information.'
    5.  **References:** Relevant requirements, problem reports (PRs), or test plans. If no specific PR, requirement, or test plan is provided in the original message or diff context, assume it's a self-audit or internal task and reference it as 'Ref: Self-Audit / Internal Task'.

    **Additional Strict Instructions:**
    * **Proper Name Preservation:** The term 'Zephy' is a proper name and must NOT be changed to 'Zephyr' or any other variation. Preserve 'Zephy' exactly as is.
    * **Output Format:** Use Markdown formatting for better readability, with clear headings for sections.
    * **Strict Output:** Your output must ONLY be the rephrased commit message, with no conversational filler, introductions, or examples." \
    --arg user_content "$llm_prompt" \
    '{
      "model": $model,
      "messages": [
        { "role": "system", "content": $system_content },
        { "role": "user", "content": $user_content }
      ],
      "temperature": 0.7,
      "max_tokens": 800,
      "stream": false
    }')

  # Make the API call using curl, piping the json_payload
  response=$(echo "$json_payload" | curl -s \
    -X POST \
    -H "Content-Type: application/json" \
    --data-binary @- \
    "$API_URL") # <-- This now correctly uses the defined variable

  # --- MODIFICATION START ---
  rephrased_message="" # Initialize as empty

  if [[ -z "$response" ]]; then
    echo "API Call received EMPTY response for commit." >&2
  elif ! echo "$response" | jq -e . > /dev/null; then # Check if response is valid JSON
    echo "API Response is NOT valid JSON for commit." >&2
    echo "Full API Response: $response" >&2 # Print response if not valid JSON
  elif ! echo "$response" | jq -e '.choices[0].message.content' > /dev/null; then # Check if desired path exists
    echo "API Response is valid JSON, but '.choices[0].message.content' path NOT found." >&2
    echo "Full API Response: $response" >&2 # Print response if path not found
  else
    # If all checks pass, then extract the content
    rephrased_message=$(echo "$response" | jq -r '.choices[0].message.content')
  fi
  # --- MODIFICATION END ---

  echo "$rephrased_message"
}

# --- Main Script to Read Commits and Rephrase ---

echo "--- Processing Git Commits ---"

# Use process substitution to avoid the while loop running in a subshell
# This is a good practice, though not strictly necessary for this script's logic
while read -r commit_hash; do
  if [[ -n "$commit_hash" ]]; then
    echo "--- Processing Commit: $commit_hash ---"

    commit_details=$(git show --unified=0 --no-color --format=%B "$commit_hash")
    commit_author=$(git show -s --format='%an' "$commit_hash")

    rephrased_output=$(rephrase_commit "$commit_details" "$commit_author")

    echo "Original Commit Details (first 10 lines):"
    echo "$commit_details" | head -n 10
    echo "..."
    echo ""

    echo "Rephrased Commit Message:"
    if [[ -n "$rephrased_output" ]]; then
      echo "$rephrased_output"
    else
      echo "Error: Failed to rephrase commit message for commit $commit_hash."
      C_RED='\033[0;31m' # Red color
      C_NC='\033[0m'    # No Color
      echo -e "${C_RED}No rephrased message was generated. See above for potential API errors.${C_NC}"
    fi
    echo ""
  fi
done < <(git log --pretty=format:%H)

echo "--- Processing Complete ---"
