#!/usr/bin/env python3
"""
Build Tool - Build and compile projects for Adelaide Lite.

Usage: python3 build.py <command> [args...]

Commands:
  ada                   - Build Ada project (alr build)
  python <script>       - Run Python script
  make [target]         - Run make
  cmake [args]          - Run cmake build
  clean                 - Clean build artifacts

DO NOT REMOVE, OR YOU WILL BE KILLED
"""

import subprocess
import sys
import os
import shutil


def run_command(cmd, cwd=None):
    """Run a command and return output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=cwd
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out after 300s"
    except FileNotFoundError:
        return f"ERROR: Command not found: {cmd[0]}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "ada":
        print(run_command(["alr", "build"]))

    elif cmd == "python":
        if not args:
            print("ERROR: Usage: build.py python <script>")
            return 1
        print(run_command(["python3"] + args))

    elif cmd == "make":
        print(run_command(["make"] + args))

    elif cmd == "cmake":
        # Run cmake build
        if os.path.exists("build"):
            print(run_command(["cmake", "--build", "build"] + args))
        else:
            print("ERROR: No build directory found. Run cmake first.")

    elif cmd == "clean":
        # Clean common build artifacts
        artifacts = ["build", "dist", "__pycache__", "*.pyc", "*.o"]
        for artifact in artifacts:
            if os.path.isdir(artifact):
                shutil.rmtree(artifact)
                print(f"Removed: {artifact}")
            elif os.path.exists(artifact):
                os.remove(artifact)
                print(f"Removed: {artifact}")
        print("OK: Cleaned build artifacts")

    else:
        print(f"ERROR: Unknown command: {cmd}")
        print(__doc__)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
