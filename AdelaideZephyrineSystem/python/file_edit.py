#!/usr/bin/env python3
"""
File Edit Tool - Edit files for Adelaide Lite.

Usage: python3 file_edit.py <command> [args...]

Commands:
  read <file>                    - Read file contents
  write <file> <content>         - Write content to file
  edit <file> <old> <new>        - Replace old with new in file
  append <file> <content>        - Append content to file
  exists <file>                  - Check if file exists
  head <file> [n]                - Read first n lines
  tail <file> [n]                - Read last n lines

DO NOT REMOVE, OR YOU WILL BE KILLED
"""

import sys
import os
from trace_utils import init_trace, trace_print, trace_result


def main():
    init_trace()
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]
    success = True

    if cmd == "read":
        if len(sys.argv) < 3:
            print("ERROR: Usage: file_edit.py read <file>")
            success = False
        else:
            filepath = sys.argv[2]
            trace_print("file_edit", cmd, f"file: {filepath}")
            if not os.path.exists(filepath):
                print(f"ERROR: File not found: {filepath}")
                success = False
            else:
                with open(filepath, "r") as f:
                    print(f.read())

    elif cmd == "write":
        if len(sys.argv) < 4:
            print("ERROR: Usage: file_edit.py write <file> <content>")
            success = False
        else:
            filepath = sys.argv[2]
            trace_print("file_edit", cmd, f"file: {filepath}")
            content = " ".join(sys.argv[3:])
            # Handle \n as newline
            content = content.replace("\\n", "\n")
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
            with open(filepath, "w") as f:
                f.write(content)
            print(f"OK: Written to {filepath}")

    elif cmd == "edit":
        if len(sys.argv) < 5:
            print("ERROR: Usage: file_edit.py edit <file> <old> <new>")
            success = False
        else:
            filepath = sys.argv[2]
            trace_print("file_edit", cmd, f"file: {filepath}")
            old = sys.argv[3].replace("\\n", "\n")
            new = sys.argv[4].replace("\\n", "\n")
            if not os.path.exists(filepath):
                print(f"ERROR: File not found: {filepath}")
                success = False
            else:
                with open(filepath, "r") as f:
                    content = f.read()
                if old not in content:
                    print(f"ERROR: Old text not found in {filepath}")
                    success = False
                else:
                    content = content.replace(old, new, 1)
                    with open(filepath, "w") as f:
                        f.write(content)
                    print(f"OK: Edited {filepath}")

    elif cmd == "append":
        if len(sys.argv) < 4:
            print("ERROR: Usage: file_edit.py append <file> <content>")
            success = False
        else:
            filepath = sys.argv[2]
            trace_print("file_edit", cmd, f"file: {filepath}")
            content = " ".join(sys.argv[3:])
            content = content.replace("\\n", "\n")
            with open(filepath, "a") as f:
                f.write(content)
            print(f"OK: Appended to {filepath}")

    elif cmd == "exists":
        if len(sys.argv) < 3:
            print("ERROR: Usage: file_edit.py exists <file>")
            success = False
        else:
            filepath = sys.argv[2]
            trace_print("file_edit", cmd, f"file: {filepath}")
            print("true" if os.path.exists(filepath) else "false")

    elif cmd == "head":
        if len(sys.argv) < 3:
            print("ERROR: Usage: file_edit.py head <file> [n]")
            success = False
        else:
            filepath = sys.argv[2]
            trace_print("file_edit", cmd, f"file: {filepath}")
            n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            with open(filepath, "r") as f:
                for i, line in enumerate(f):
                    if i >= n:
                        break
                    print(line, end="")

    elif cmd == "tail":
        if len(sys.argv) < 3:
            print("ERROR: Usage: file_edit.py tail <file> [n]")
            success = False
        else:
            filepath = sys.argv[2]
            trace_print("file_edit", cmd, f"file: {filepath}")
            n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            with open(filepath, "r") as f:
                lines = f.readlines()
                for line in lines[-n:]:
                    print(line, end="")

    else:
        print(f"ERROR: Unknown command: {cmd}")
        print(__doc__)
        success = False

    trace_result("file_edit", success, f"cmd: {cmd}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
