#!/usr/bin/env python3
"""
Grep Tool - Search file contents for Adelaide Lite.

Usage: python3 grep.py <command> [args...]

Commands:
  search <pattern> [path]    - Search for pattern in files
  regex <pattern> [path]     - Search with regex
  fixed <string> [path]      - Search for fixed string
  count <pattern> [path]     - Count matches
  files <pattern> [path]     - List files containing pattern

Options:
  --include <ext>            - Include only files with extension
  --exclude <ext>            - Exclude files with extension
  --ignore-case              - Case insensitive search

DO NOT REMOVE, OR YOU WILL BE KILLED
"""

import subprocess
import sys
import re
from trace_utils import init_trace, trace_print


def run_grep(pattern, path=".", options=None):
    """Run grep with options."""
    cmd = ["grep", "-r"]
    
    if options:
        if options.get("ignore_case"):
            cmd.append("-i")
        if options.get("include"):
            cmd.extend(["--include", options["include"]])
        if options.get("exclude"):
            cmd.extend(["--exclude", options["exclude"]])
    
    cmd.extend([pattern, path])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return "ERROR: Grep timed out"
    except Exception as e:
        return f"ERROR: {e}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]
    args = sys.argv[2:]
    
    # Parse options
    options = {}
    paths = []
    i = 0
    while i < len(args):
        if args[i] == "--include" and i + 1 < len(args):
            options["include"] = args[i + 1]
            i += 2
        elif args[i] == "--exclude" and i + 1 < len(args):
            options["exclude"] = args[i + 1]
            i += 2
        elif args[i] == "--ignore-case":
            options["ignore_case"] = True
            i += 1
        else:
            paths.append(args[i])
            i += 1

    init_trace()

    if cmd == "search" or cmd == "regex":
        if not paths:
            print("ERROR: Usage: grep.py search <pattern> [path]")
            return 1
        pattern = paths[0]
        path = paths[1] if len(paths) > 1 else "."
        trace_print("grep", cmd, f"pattern: {pattern}, path: {path}")
        output = run_grep(pattern, path, options)
        print(output if output else "No matches found")

    elif cmd == "fixed":
        if not paths:
            print("ERROR: Usage: grep.py fixed <string> [path]")
            return 1
        pattern = re.escape(paths[0])
        path = paths[1] if len(paths) > 1 else "."
        trace_print("grep", cmd, f"pattern: {pattern}, path: {path}")
        output = run_grep(pattern, path, options)
        print(output if output else "No matches found")

    elif cmd == "count":
        if not paths:
            print("ERROR: Usage: grep.py count <pattern> [path]")
            return 1
        pattern = paths[0]
        path = paths[1] if len(paths) > 1 else "."
        trace_print("grep", cmd, f"pattern: {pattern}, path: {path}")
        cmd_list = ["grep", "-r", "-c", pattern, path]
        if options.get("ignore_case"):
            cmd_list.insert(2, "-i")
        try:
            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=30
            )
            print(result.stdout if result.stdout else "No matches found")
        except Exception:
            print("ERROR: Grep failed")

    elif cmd == "files":
        if not paths:
            print("ERROR: Usage: grep.py files <pattern> [path]")
            return 1
        pattern = paths[0]
        path = paths[1] if len(paths) > 1 else "."
        trace_print("grep", cmd, f"pattern: {pattern}, path: {path}")
        cmd_list = ["grep", "-r", "-l", pattern, path]
        if options.get("ignore_case"):
            cmd_list.insert(2, "-i")
        try:
            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=30
            )
            print(result.stdout if result.stdout else "No matching files")
        except Exception:
            print("ERROR: Grep failed")

    else:
        print(f"ERROR: Unknown command: {cmd}")
        print(__doc__)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
