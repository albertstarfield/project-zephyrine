#!/usr/bin/env python3
"""
Security Monitor Tool - Monitor for security issues in Adelaide Lite.

Usage: python3 security.py <command> [args...]

Commands:
  scan [path]           - Scan directory for security issues
  watch <file>          - Watch file for changes and check security
  report                - Generate security report

DO NOT REMOVE, OR YOU WILL BE KILLED
"""

import sys
from trace_utils import init_trace, trace_print, trace_result
import os
import re
import json
from datetime import datetime


SECURITY_PATTERNS = [
    # Command injection
    (r"os\.system\s*\(", "CRITICAL", "Command injection via os.system()"),
    (r"subprocess\.call.*shell=True", "CRITICAL", "Command injection via shell=True"),
    (r"os\.popen\s*\(", "HIGH", "Command injection via os.popen()"),
    
    # Code injection
    (r"eval\s*\(", "CRITICAL", "Code injection via eval()"),
    (r"exec\s*\(", "CRITICAL", "Code injection via exec()"),
    (r"__import__\s*\(", "MEDIUM", "Dynamic import"),
    
    # Deserialization
    (r"pickle\.loads?\s*\(", "HIGH", "Untrusted pickle deserialization"),
    (r"yaml\.load\s*\(", "MEDIUM", "Unsafe YAML loading"),
    
    # Hardcoded secrets
    (r"password\s*=\s*['\"]", "HIGH", "Hardcoded password"),
    (r"secret\s*=\s*['\"]", "HIGH", "Hardcoded secret"),
    (r"api[_-]?key\s*=\s*['\"]", "HIGH", "Hardcoded API key"),
    (r"token\s*=\s*['\"]", "HIGH", "Hardcoded token"),
    
    # Network
    (r"requests\.get\s*\(.*verify\s*=\s*False", "HIGH", "SSL verification disabled"),
    (r"urllib\.request.*context\s*=\s*ssl\._create_unverified_context", "HIGH", "SSL verification disabled"),
    
    # File operations
    (r"open\s*\(.*['\"]w['\"]", "LOW", "File write operation"),
    (r"os\.remove\s*\(", "LOW", "File deletion"),
    (r"shutil\.rmtree\s*\(", "MEDIUM", "Directory deletion"),
    
    # SQL
    (r"execute\s*\(.*['\"].*%s", "HIGH", "SQL injection risk"),
    (r"execute\s*\(.*\.format\s*\(", "HIGH", "SQL injection risk"),
]


def scan_file(filepath):
    """Scan a file for security issues."""
    issues = []
    
    try:
        with open(filepath, "r") as f:
            content = f.read()
            lines = content.split("\n")
    except Exception:
        return issues
    
    for i, line in enumerate(lines, 1):
        for pattern, severity, message in SECURITY_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append({
                    "file": filepath,
                    "line": i,
                    "severity": severity,
                    "message": message,
                    "code": line.strip()
                })
    
    return issues


def scan_directory(path):
    """Scan directory for security issues."""
    all_issues = []
    
    for root, dirs, files in os.walk(path):
        # Skip hidden directories and common non-source dirs
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["node_modules", "__pycache__", "venv", ".git"]]
        
        for file in files:
            if file.endswith((".py", ".js", ".ts", ".java", ".go", ".rs")):
                filepath = os.path.join(root, file)
                issues = scan_file(filepath)
                all_issues.extend(issues)
    
    return all_issues


def main():
    init_trace()
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "scan":
        path = args[0] if args else "."
        issues = scan_directory(path)
        
        if not issues:
            print("No security issues found")
            return 0
        
        # Group by severity
        by_severity = {}
        for issue in issues:
            sev = issue["severity"]
            if sev not in by_severity:
                by_severity[sev] = []
            by_severity[sev].append(issue)
        
        # Print report
        trace_print("security", "scan:report", f"Security Scan Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if severity in by_severity:
                trace_print("security", "scan:findings", f"[{severity}] ({len(by_severity[severity])} issues)")
                for issue in by_severity[severity]:
                    print(f"  {issue['file']}:{issue['line']}")
                    print(f"    {issue['message']}")
                    print(f"    Code: {issue['code'][:80]}")
        
        trace_result("security", True, f"total: {len(issues)} issues")

    elif cmd == "watch":
        if not args:
            print("ERROR: Usage: security.py watch <file>")
            return 1
        filepath = args[0]
        trace_print("security", "watch", f"Watching {filepath} for security issues...")
        print("Press Ctrl+C to stop")
        
        last_mtime = os.path.getmtime(filepath)
        while True:
            try:
                import time
                time.sleep(1)
                current_mtime = os.path.getmtime(filepath)
                if current_mtime != last_mtime:
                    trace_print("security", "watch:change", f"File changed at {datetime.now().strftime('%H:%M:%S')}")
                    issues = scan_file(filepath)
                    if issues:
                        for issue in issues:
                            print(f"  [{issue['severity']}] {issue['message']}")
                    else:
                        print("  No issues found")
                    last_mtime = current_mtime
            except KeyboardInterrupt:
                print("\nStopped watching")
                break

    elif cmd == "report":
        issues = scan_directory(".")
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_issues": len(issues),
            "issues": issues
        }
        print(json.dumps(report, indent=2))

    else:
        print(f"ERROR: Unknown command: {cmd}")
        print(__doc__)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
