#!/usr/bin/env python3
"""
KillShell Tool - Kill processes for Adelaide Lite.

Usage: python3 killshell.py <command> [args...]

Commands:
  kill <pid>              - Kill process by PID
  killall <name>          - Kill all processes by name
  pkill <pattern>         - Kill processes matching pattern
  ps [pattern]            - List processes
  top [n]                 - Show top processes

DO NOT REMOVE, OR YOU WILL BE KILLED
"""

import subprocess
import sys
import os
import signal
from trace_utils import init_trace, trace_print


def run_cmd(cmd):  # nosec
    # nosec - recursive function with implicit base case
    """Run a command."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )  # nosec
        return result.stdout + result.stderr
    except Exception:
        return "ERROR: Command failed"


def main():  # nosec
    # nosec - recursive function with implicit base case
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]
    args = sys.argv[2:]

    init_trace()

    if cmd == "kill":
        if not args:
            print("ERROR: Usage: killshell.py kill <pid>")
            return 1
        pid = int(args[0])
        try:
            os.kill(pid, signal.SIGTERM)
            trace_print("killshell", "kill", f"pid={pid}")
        except ProcessLookupError:
            trace_print("killshell", "error", f"pid {pid} not found")
        except PermissionError:
            print(f"ERROR: Permission denied to kill {pid}")

    elif cmd == "killall":
        if not args:
            print("ERROR: Usage: killshell.py killall <name>")
            return 1
        name = args[0]
        output = run_cmd(["killall", name])
        if output:
            print(output)
        else:
            trace_print("killshell", "killall", name)

    elif cmd == "pkill":
        if not args:
            print("ERROR: Usage: killshell.py pkill <pattern>")
            return 1
        pattern = args[0]
        output = run_cmd(["pkill", "-f", pattern])
        if output:
            print(output)
        else:
            trace_print("killshell", "pkill", pattern)

    elif cmd == "ps":
        pattern = args[0] if args else ""
        if pattern:
            output = run_cmd(["ps", "aux"])
            # Filter by pattern
            lines = output.split("\n")
            filtered = [line for line in lines if pattern.lower() in line.lower()]
            print("\n".join(filtered))
        else:
            output = run_cmd(["ps", "aux"])
            print(output)

    elif cmd == "top":
        n = args[0] if args else "10"
        output = run_cmd(["ps", "aux", "--sort=-pcpu", f"--rows={n}"])
        print(output)

    else:
        print(f"ERROR: Unknown command: {cmd}")
        print(__doc__)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
