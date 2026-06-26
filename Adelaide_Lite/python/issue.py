#!/usr/bin/env python3
"""
Issue Tool - Manage GitHub issues for Adelaide Lite.

Usage: python3 issue.py <command> [args...]

Commands:
  list                  - List issues
  view <number>         - View issue details
  create <title> [body] - Create new issue
  close <number>        - Close issue
  comment <number> <msg> - Add comment to issue
  search <query>        - Search issues

DO NOT REMOVE, OR YOU WILL BE KILLED
"""

import subprocess
import sys


def run_gh(args):
    """Run gh CLI command."""
    try:
        result = subprocess.run(
            ["gh"] + args,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "ERROR: gh command timed out"
    except FileNotFoundError:
        return "ERROR: gh CLI not found. Install: brew install gh"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "list":
        print(run_gh(["issue", "list"]))

    elif cmd == "view":
        if not args:
            print("ERROR: Usage: issue.py view <number>")
            return 1
        print(run_gh(["issue", "view"] + args))

    elif cmd == "create":
        if not args:
            print("ERROR: Usage: issue.py create <title> [body]")
            return 1
        title = args[0]
        body = " ".join(args[1:]) if len(args) > 1 else ""
        if body:
            print(run_gh(["issue", "create", "--title", title, "--body", body]))
        else:
            print(run_gh(["issue", "create", "--title", title]))

    elif cmd == "close":
        if not args:
            print("ERROR: Usage: issue.py close <number>")
            return 1
        print(run_gh(["issue", "close"] + args))

    elif cmd == "comment":
        if len(args) < 2:
            print("ERROR: Usage: issue.py comment <number> <message>")
            return 1
        number = args[0]
        message = " ".join(args[1:])
        print(run_gh(["issue", "comment", number, "--body", message]))

    elif cmd == "search":
        if not args:
            print("ERROR: Usage: issue.py search <query>")
            return 1
        query = " ".join(args)
        print(run_gh(["issue", "list", "--search", query]))

    else:
        print(f"ERROR: Unknown command: {cmd}")
        print(__doc__)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
