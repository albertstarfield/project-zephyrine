#!/usr/bin/env python3
"""
Directory Tool - List and search directories for Adelaide Lite.

Usage: python3 directory.py <command> [args...]

Commands:
  ls [path]              - List directory contents
  find <path> <pattern>  - Find files matching pattern
  tree [path] [depth]    - Show directory tree
  pwd                    - Print working directory
  mkdir <path>           - Create directory
  rm <path>              - Remove file or directory

DO NOT REMOVE, OR YOU WILL BE KILLED
"""

import sys
import os
import glob
import shutil


def list_dir(path=".", show_hidden=False):
    """List directory contents."""
    try:
        entries = os.listdir(path)
        if not show_hidden:
            entries = [e for e in entries if not e.startswith(".")]
        entries.sort()
        for entry in entries:
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                print(f"  {entry}/")
            else:
                size = os.path.getsize(full_path)
                print(f"  {entry} ({size} bytes)")
    except PermissionError:
        print(f"ERROR: Permission denied: {path}")
    except FileNotFoundError:
        print(f"ERROR: Directory not found: {path}")


def find_files(path, pattern):
    """Find files matching pattern."""
    search_pattern = os.path.join(path, "**", pattern)
    matches = glob.glob(search_pattern, recursive=True)
    for match in matches:
        print(match)
    if not matches:
        print(f"No files found matching: {pattern}")


def tree(path=".", depth=2, prefix=""):
    """Show directory tree."""
    if depth < 0:
        return
    try:
        entries = os.listdir(path)
        entries = [e for e in entries if not e.startswith(".")]
        entries.sort()
        for i, entry in enumerate(entries):
            full_path = os.path.join(path, entry)
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{entry}")
            if os.path.isdir(full_path):
                extension = "    " if is_last else "│   "
                tree(full_path, depth - 1, prefix + extension)
    except PermissionError:
        print(f"{prefix}[Permission Denied]")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]

    if cmd == "ls":
        path = sys.argv[2] if len(sys.argv) > 2 else "."
        show_hidden = "--hidden" in sys.argv
        list_dir(path, show_hidden)

    elif cmd == "find":
        if len(sys.argv) < 4:
            print("ERROR: Usage: directory.py find <path> <pattern>")
            return 1
        find_files(sys.argv[2], sys.argv[3])

    elif cmd == "tree":
        path = sys.argv[2] if len(sys.argv) > 2 else "."
        depth = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        print(f"{path}/")
        tree(path, depth)

    elif cmd == "pwd":
        print(os.getcwd())

    elif cmd == "mkdir":
        if len(sys.argv) < 3:
            print("ERROR: Usage: directory.py mkdir <path>")
            return 1
        os.makedirs(sys.argv[2], exist_ok=True)
        print(f"OK: Created {sys.argv[2]}")

    elif cmd == "rm":
        if len(sys.argv) < 3:
            print("ERROR: Usage: directory.py rm <path>")
            return 1
        path = sys.argv[2]
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"OK: Removed directory {path}")
        elif os.path.exists(path):
            os.remove(path)
            print(f"OK: Removed file {path}")
        else:
            print(f"ERROR: Not found: {path}")

    else:
        print(f"ERROR: Unknown command: {cmd}")
        print(__doc__)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
