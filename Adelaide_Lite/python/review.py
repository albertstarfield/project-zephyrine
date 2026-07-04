#!/usr/bin/env python3
"""
Review Tool - Code review for Adelaide Lite.

Usage: python3 review.py <command> [args...]

Commands:
  diff [branch]         - Review diff against branch
  file <file>           - Review a specific file
  security <file>       - Security review of file
  quality <file>        - Code quality review

DO NOT REMOVE, OR YOU WILL BE KILLED
"""

import subprocess
import sys
import os
import re
from trace_utils import init_trace, trace_print, trace_result


def run_command(cmd):
    """Run a command and return output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout
    except Exception:
        return ""


def security_check(filepath):
    """Check file for security issues."""
    issues = []
    
    try:
        with open(filepath, "r") as f:
            content = f.read()
            lines = content.split("\n")
    except Exception:
        return ["ERROR: Cannot read file"]
    
    # Security patterns
    patterns = [
        (r"eval\s*\(", "Use of eval() - potential code injection"),
        (r"exec\s*\(", "Use of exec() - potential code injection"),
        (r"os\.system\s*\(", "Use of os.system() - use subprocess instead"),
        (r"subprocess\.call.*shell=True", "shell=True in subprocess - command injection risk"),
        (r"pickle\.loads?\s*\(", "Untrusted pickle deserialization"),
        (r"__import__\s*\(", "Dynamic import - potential security risk"),
        (r"input\s*\(", "Use of input() - verify not used for sensitive data"),
        (r"password\s*=\s*['\"]", "Hardcoded password detected"),
        (r"secret\s*=\s*['\"]", "Hardcoded secret detected"),
        (r"api[_-]?key\s*=\s*['\"]", "Hardcoded API key detected"),
    ]
    
    for i, line in enumerate(lines, 1):
        for pattern, message in patterns:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append(f"Line {i}: {message}")
                issues.append(f"  {line.strip()}")
    
    return issues


def quality_check(filepath):
    """Check file for code quality issues."""
    issues = []
    
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except Exception:
        return ["ERROR: Cannot read file"]
    
    # Quality patterns
    for i, line in enumerate(lines, 1):
        # Long lines
        if len(line.rstrip()) > 120:
            issues.append(f"Line {i}: Line too long ({len(line.rstrip())} > 120)")
        
        # TODO/FIXME
        if re.search(r"#\s*(TODO|FIXME|HACK|XXX)", line, re.IGNORECASE):
            issues.append(f"Line {i}: Unresolved TODO/FIXME")
        
        # Bare except
        if re.search(r"except\s*:", line):
            issues.append(f"Line {i}: Bare except - use specific exceptions")
        
        # Global variables
        if re.match(r"^[A-Z_]+\s*=", line.strip()):
            issues.append(f"Line {i}: Possible global variable")
    
    return issues


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]
    args = sys.argv[2:]

    init_trace()

    if cmd == "diff":
        branch = args[0] if args else "main"
        print(run_command(["git", "diff", branch]))

    elif cmd == "file":
        if not args:
            print("ERROR: Usage: review.py file <file>")
            return 1
        filepath = args[0]
        if not os.path.exists(filepath):
            print(f"ERROR: File not found: {filepath}")
            return 1
        trace_print("review", "file", filepath)
        trace_print("review", "security", "scanning...")
        sec_issues = security_check(filepath)
        if sec_issues:
            for issue in sec_issues:
                print(issue)
        else:
            print("No security issues found")
        trace_print("review", "quality", "scanning...")
        qual_issues = quality_check(filepath)
        if qual_issues:
            for issue in qual_issues:
                print(issue)
        else:
            print("No quality issues found")
        total = len(sec_issues) + len(qual_issues)
        trace_result("review", True, f"found {total} issues")

    elif cmd == "security":
        if not args:
            print("ERROR: Usage: review.py security <file>")
            return 1
        filepath = args[0]
        issues = security_check(filepath)
        if issues:
            for issue in issues:
                print(issue)
        else:
            print("No security issues found")

    elif cmd == "quality":
        if not args:
            print("ERROR: Usage: review.py quality <file>")
            return 1
        filepath = args[0]
        issues = quality_check(filepath)
        if issues:
            for issue in issues:
                print(issue)
        else:
            print("No quality issues found")

    else:
        print(f"ERROR: Unknown command: {cmd}")
        print(__doc__)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
