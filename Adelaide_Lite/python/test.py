#!/usr/bin/env python3
"""
Test Tool - Run tests for Adelaide Lite.

Usage: python3 test.py <command> [args...]

Commands:
  pytest [args]         - Run pytest
  unittest [args]       - Run unittest
  ada                   - Run Ada tests (alr test)
  lint                  - Run linter (ruff)
  typecheck             - Run type checker (pyrefly)

DO NOT REMOVE, OR YOU WILL BE KILLED
"""

import subprocess
import sys
import os


def run_command(cmd, cwd=None):
    """Run a command and return output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=cwd
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out after 120s"
    except FileNotFoundError:
        return f"ERROR: Command not found: {cmd[0]}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "pytest":
        print(run_command(["python3", "-m", "pytest"] + args))

    elif cmd == "unittest":
        print(run_command(["python3", "-m", "unittest"] + args))

    elif cmd == "ada":
        print(run_command(["alr", "test"]))

    elif cmd == "lint":
        # Run ruff check
        print(run_command(["python3", "-m", "ruff", "check"] + args))

    elif cmd == "typecheck":
        # Run pyrefly
        print(run_command(["python3", "-m", "pyrefly"] + args))

    else:
        print(f"ERROR: Unknown command: {cmd}")
        print(__doc__)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
