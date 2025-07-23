#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import textwrap

# --- CONFIGURATION ---
# The API endpoint for your local LLM
API_URL = "http://localhost:18141/v1/chat/completions"

# The system prompt that guides the LLM
SYSTEM_PROMPT = """
You are a professional technical writer specializing in creating highly detailed, formal commit messages for Configuration Management (CM) in professional avionics software development. You will be provided with full Git commit details, including the original commit message, list of files changed, and the code diffs.

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
* **Strict Output:** Your output must ONLY be the rephrased commit message, with no conversational filler, introductions, or examples.
"""

# --- Colors for Terminal Output ---
C_RED = '\033[0;31m'
C_GREEN = '\033[0;32m'
C_YELLOW = '\033[0;33m'
C_BLUE = '\033[0;34m'
C_NC = '\033[0m' # No Color

def run_git_command(command):
    """Executes a Git command and returns its stdout."""
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"{C_RED}Error executing command: {' '.join(command)}{C_NC}", file=sys.stderr)
        print(f"{C_RED}Stderr: {e.stderr.strip()}{C_NC}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"{C_RED}Error: 'git' command not found. Is Git installed and in your PATH?{C_NC}", file=sys.stderr)
        sys.exit(1)


def rephrase_commit_message(commit_details: str, author: str) -> str | None:
    """
    Sends commit details to an LLM API and returns the rephrased message.
    Returns None on failure.
    """
    # Lazy import requests so the script can run with --help without it
    try:
        import requests
    except ImportError:
        print(f"{C_RED}Error: 'requests' library not found.{C_NC}", file=sys.stderr)
        print("Please install it by running: pip install requests", file=sys.stderr)
        sys.exit(1)

    llm_prompt = (
        f"Author for this commit: {author}\n\n"
        "Rephrase the following Git commit details (including original message, list of files changed, "
        "and code diffs) into a formal CM management style. Use the provided diff information to make "
        "the description of changes more precise and concrete, referencing specific files or parts "
        "of the code where appropriate.\n\n"
        f"{commit_details}\n\n"
    )

    json_payload = {
        "model": "google/gemma-3n-e4b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": llm_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1024, # Increased for potentially long diffs
        "stream": False
    }

    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json=json_payload,
            timeout=120 # 2-minute timeout for slow models
        )
        response.raise_for_status() # Raises an exception for 4xx/5xx errors

        response_data = response.json()
        rephrased_message = response_data['choices'][0]['message']['content']
        return rephrased_message.strip()

    except requests.exceptions.RequestException as e:
        print(f"{C_RED}API call failed: {e}{C_NC}", file=sys.stderr)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"{C_RED}Failed to parse API response: {e}{C_NC}", file=sys.stderr)
        print(f"Full response: {response.text}", file=sys.stderr)

    return None


def process_in_test_mode():
    """
    Fetches all commits, rephrases them, and prints the results without changing history.
    """
    print(f"{C_BLUE}--- Running in Test Mode (No changes will be made) ---{C_NC}")
    commit_hashes = run_git_command(['git', 'log', '--pretty=format:%H']).splitlines()
    if not commit_hashes:
        print("No commits found.")
        return

    for commit_hash in reversed(commit_hashes): # Process from oldest to newest
        print(f"\n{C_YELLOW}--- Processing Commit: {commit_hash} ---{C_NC}")

        # Fetch full details including diff
        commit_details = run_git_command(['git', 'show', '--unified=0', '--no-color', '--format=%B', commit_hash])
        commit_author = run_git_command(['git', 'show', '-s', '--format=%an', commit_hash])

        if commit_details is None or commit_author is None:
            print(f"{C_RED}Failed to get details for commit {commit_hash}. Skipping.{C_NC}")
            continue

        print(f"\n{C_BLUE}Original Commit Message:{C_NC}")
        original_message = run_git_command(['git', 'show', '-s', '--format=%B', commit_hash])
        print(textwrap.indent(original_message, '> '))


        rephrased_message = rephrase_commit_message(commit_details, commit_author)

        print(f"\n{C_GREEN}Rephrased Commit Message:{C_NC}")
        if rephrased_message:
            print(textwrap.indent(rephrased_message, '| '))
        else:
            print(f"{C_RED}Error: Failed to rephrase commit message.{C_NC}")


def apply_history_rewrite():
    """
    Uses 'git filter-branch' to rewrite the history of the current branch.
    This is a destructive operation.
    """
    print(f"{C_RED}--- WARNING: HISTORY REWRITE MODE ---{C_NC}")
    print("This will permanently rewrite the Git history for the current branch.")
    print("It is a destructive operation. Make sure you have a backup.")
    print("This script will call 'git filter-branch' to process each commit.")

    try:
        answer = input(f"Are you sure you want to continue? (yes/no): ").lower()
        if answer != 'yes':
            print("Operation cancelled.")
            sys.exit(0)
    except (EOFError, KeyboardInterrupt):
        print("\nOperation cancelled.")
        sys.exit(0)

    print(f"{C_YELLOW}Starting history rewrite... This may take a while.{C_NC}")

    # This command tells git to run this script for every commit's message.
    # The script detects the '--internal-filter-mode' and acts accordingly.
    # It also preserves the original author name and date.
    script_path = os.path.abspath(__file__)
    msg_filter_command = f'"{sys.executable}" "{script_path}" --internal-filter-mode'

    # The --env-filter preserves the original author and date, while the
    # committer becomes the person running the script (which is correct).
    env_filter_command = (
        'export GIT_AUTHOR_NAME="$GIT_AUTHOR_NAME"; '
        'export GIT_AUTHOR_EMAIL="$GIT_AUTHOR_EMAIL"; '
        'export GIT_AUTHOR_DATE="$GIT_AUTHOR_DATE";'
    )

    filter_branch_command = [
        'git', 'filter-branch', '-f',
        '--env-filter', env_filter_command,
        '--msg-filter', msg_filter_command,
        '--' # This ensures it only rewrites the current branch
    ]

    print(f"Executing: {' '.join(filter_branch_command)}")

    try:
        # We use Popen to stream the output in real-time
        process = subprocess.Popen(
            ' '.join(filter_branch_command),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            print(f"{C_RED}git filter-branch failed with exit code {return_code}{C_NC}")
        else:
            print(f"\n{C_GREEN}--- History rewrite complete! ---{C_NC}")
            print("Original refs were backed up in 'refs/original/'.")
            print("Review the new history with 'git log'.")

    except Exception as e:
        print(f"{C_RED}An error occurred during git filter-branch: {e}{C_NC}")


def internal_filter_mode():
    """
    Special mode executed by 'git filter-branch'.
    Reads original message from stdin, gets commit info from env vars,
    rephrases, and prints new message to stdout.
    """
    original_message = sys.stdin.read()

    # git filter-branch provides the commit hash in an environment variable
    commit_hash = os.environ.get('GIT_COMMIT')
    if not commit_hash:
        # If something goes wrong, print the original message to avoid breaking the filter
        print(original_message, end='')
        return

    # We need the full diff, not just the message, for the best rephrasing
    commit_details = run_git_command(['git', 'show', '--unified=0', '--no-color', '--format=%B', commit_hash])
    commit_author = os.environ.get('GIT_AUTHOR_NAME', 'Unknown Author')

    rephrased_message = rephrase_commit_message(commit_details, commit_author)

    if rephrased_message:
        # Print the new message to stdout for git to use
        print(rephrased_message, end='')
    else:
        # IMPORTANT: If the API fails, print the original message back.
        # This prevents losing a commit message entirely.
        print(f"API call failed for commit {commit_hash[:7]}. Keeping original message.", file=sys.stderr)
        print(original_message, end='')


def main():
    """Main function to parse arguments and direct traffic."""
    parser = argparse.ArgumentParser(
        description="Rephrase Git commit messages using an LLM.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help="Run in test mode. Prints rephrased messages without changing Git history."
    )
    # This is a hidden argument for the script to call itself
    parser.add_argument(
        '--internal-filter-mode',
        action='store_true',
        help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    # Check if we are in a git repository
    if not os.path.isdir('.git') and not args.internal_filter_mode:
        print(f"{C_RED}Error: This script must be run from the root of a Git repository.{C_NC}", file=sys.stderr)
        sys.exit(1)

    if args.internal_filter_mode:
        internal_filter_mode()
    elif args.test_mode:
        process_in_test_mode()
    else:
        apply_history_rewrite()


if __name__ == "__main__":
    main()