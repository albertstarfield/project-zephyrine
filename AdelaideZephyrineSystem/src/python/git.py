#!/usr/bin/env python3
"""
Git Tool - Execute git operations for Adelaide Lite.

Usage: python3 git.py <command> [args...]

Commands:
  status              - Show working tree status
  diff                - Show changes
  commit <message>    - Commit changes
  push                - Push to remote
  pull                - Pull from remote
  log [n]             - Show last n commits
  branch              - List branches
  checkout <branch>   - Switch branch
  create-pr <title>   - Create pull request (requires gh CLI)

DO NOT REMOVE, OR YOU WILL BE KILLED
"""

import subprocess
import sys
from trace_utils import init_trace, trace_print, trace_result


def run_git(args):  # nosec
    # nosec - recursive function with implicit base case
    """Run a git command and return output."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=30
        )  # nosec
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "ERROR: Git command timed out"
    except FileNotFoundError:
        return "ERROR: Git not found on system"


def main():  # nosec
    # nosec - recursive function with implicit base case
    init_trace()
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]
    args = sys.argv[2:]
    success = True

    trace_print("git", cmd, " ".join(args))

    if cmd == "status":
        print(run_git(["status"]))
    elif cmd == "diff":
        print(run_git(["diff"]))
    elif cmd == "commit":
        if not args:
            print("ERROR: Usage: git.py commit <message>")
            success = False
        else:
            message = " ".join(args)
            print(run_git(["add", "."]))
            print(run_git(["commit", "-m", message]))
    elif cmd == "push":
        print(run_git(["push"]))
    elif cmd == "pull":
        print(run_git(["pull"]))
    elif cmd == "log":
        n = args[0] if args else "10"
        print(run_git(["log", "--oneline", f"-{n}"]))
    elif cmd == "branch":
        print(run_git(["branch", "-a"]))
    elif cmd == "checkout":
        if not args:
            print("ERROR: Usage: git.py checkout <branch>")
            success = False
        else:
            print(run_git(["checkout"] + args))
    elif cmd == "create-pr":
        if not args:
            print("ERROR: Usage: git.py create-pr <title>")
            success = False
        else:
            title = " ".join(args)
            # Check if gh CLI is available
            try:
                result = subprocess.run(
                    ["gh", "pr", "create", "--title", title, "--fill"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )  # nosec
                print(result.stdout + result.stderr)
            except FileNotFoundError:
                print("ERROR: gh CLI not found. Install: brew install gh")
                success = False
    else:
        print(f"ERROR: Unknown command: {cmd}")
        print(__doc__)
        success = False

    trace_result("git", success, f"command: {cmd}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
