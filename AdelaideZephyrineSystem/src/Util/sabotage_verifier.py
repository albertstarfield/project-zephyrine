"""
Sabotage Verifier — Self-Audit Pipeline
========================================
Detects known sabotage patterns across Python, Ada/SPARK, and C source files.
This is the internal critic that prevents wasting hours on GNATprove and AFL++
when the source itself has crash-on-launch bugs.

Architecture:
- PatternRegistry: Adaptive, extensible pattern database
- SabotageVerifier: Core engine that runs registered patterns against source
- Language-specific checkers: Python, Ada/SPARK, C
- Multi-file audit: Scan entire directories for sabotage patterns
- CLI interface: --verify-sabotage flag for standalone execution

Usage:
    # From run.py (integrated into build pipeline):
    from src.Util.sabotage_verifier import run_sabotage_audit, audit_directory
    violations = run_sabotage_audit("run.py")
    violations = audit_directory("src/python/", extensions=[".py"])
    violations = audit_directory("src/", extensions=[".adb", ".ads"])

    # Standalone:
    python src/Util/sabotage_verifier.py run.py
    python src/Util/sabotage_verifier.py run.py --severity CRITICAL
    python src/Util/sabotage_verifier.py src/python/ --extensions .py
    python src/Util/sabotage_verifier.py src/ --extensions .adb,.ads,.c
    python src/Util/sabotage_verifier.py run.py --json
"""

import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


# ── Severity Levels ──────────────────────────────────────────────────────

class Severity(Enum):
    CRITICAL = "CRITICAL"  # Will crash or cause silent data loss
    HIGH = "HIGH"          # Broken on specific platforms or under conditions
    MEDIUM = "MEDIUM"      # Code smell, dead code, stale references
    LOW = "LOW"            # Style issues, minor inefficiencies


# ── Violation Data ───────────────────────────────────────────────────────

@dataclass
class Violation:
    filepath: str
    line: int
    severity: Severity
    category: str
    message: str
    standard: str = ""
    code_snippet: str = ""

    def __repr__(self):
        return f"[{self.severity.value}] {self.filepath}:{self.line}: {self.category} — {self.message}"


# ── Pattern Definition ───────────────────────────────────────────────────

@dataclass
class Pattern:
    """
    A single detection pattern. Can be regex-based or function-based.

    For regex patterns:
        - regex: compiled regex pattern
        - context_lines: how many lines before/after to check for guards
        - guard_patterns: regexes that indicate the code is guarded
        - languages: list of languages this pattern applies to ("python", "ada", "c")

    For function patterns:
        - check_func: callable(source, lines, filepath) -> list[Violation]
    """
    name: str
    category: str
    severity: Severity
    standard: str
    description: str
    languages: list[str] = field(default_factory=lambda: ["python"])

    # Regex-based pattern fields
    regex: re.Pattern | None = None
    context_lines: int = 5
    guard_patterns: list[str] = field(default_factory=list)
    message_template: str = ""

    # Function-based pattern fields
    check_func: Callable | None = None


# ── Adaptive Pattern Registry ───────────────────────────────────────────

class PatternRegistry:
    """
    Central registry for all sabotage detection patterns.

    Patterns can be registered at startup or dynamically at runtime.
    The registry is the single source of truth for what constitutes sabotage.

    To add a new pattern:
        registry.register(Pattern(
            name="my_new_check",
            category="MY_CATEGORY",
            severity=Severity.HIGH,
            standard="CWE-XXX",
            description="Detects something bad",
            languages=["python"],
            regex=re.compile(r'bad_pattern'),
            guard_patterns=[r'if Platform\.'],
            message_template="Found bad thing: {match}",
        ))

    Or register a function-based checker:
        registry.register(Pattern(
            name="my_func_check",
            category="MY_FUNC_CATEGORY",
            severity=Severity.CRITICAL,
            standard="ISO 25010",
            description="Complex pattern analysis",
            languages=["python", "ada", "c"],
            check_func=my_check_function,
        ))
    """

    def __init__(self):
        self._patterns: list[Pattern] = []

    def register(self, pattern: Pattern):
        """Register a new detection pattern."""
        self._patterns.append(pattern)

    def register_all(self, patterns: list[Pattern]):
        """Register multiple patterns at once."""
        self._patterns.extend(patterns)

    @property
    def patterns(self) -> list[Pattern]:
        return list(self._patterns)

    def count(self) -> int:
        return len(self._patterns)

    def categories(self) -> list[str]:
        return list({p.category for p in self._patterns})

    def for_language(self, lang: str) -> list[Pattern]:
        """Return patterns that apply to a specific language."""
        return [p for p in self._patterns if lang in p.languages]


# ── Core Verifier Engine ─────────────────────────────────────────────────

class SabotageVerifier:
    """
    Core engine that runs registered patterns against source code.

    The verifier is stateless — it takes source code and a registry,
    and returns violations. All state lives in the registry.
    """

    def __init__(self, registry: PatternRegistry):
        self.registry = registry

    def verify(self, source: str, filepath: str = "", language: str = "python") -> list[Violation]:
        """Run all registered patterns against source code for a given language."""
        violations = []
        lines = source.splitlines()

        for pattern in self.registry.for_language(language):
            if pattern.check_func:
                # Function-based pattern: delegate entirely
                violations.extend(pattern.check_func(source, lines, filepath))
            elif pattern.regex:
                # Regex-based pattern: scan lines with guard detection
                violations.extend(self._check_regex(pattern, lines, filepath))

        return violations

    def _check_regex(self, pattern: Pattern, lines: list[str], filepath: str) -> list[Violation]:
        """Check a regex pattern against all lines, with guard detection."""
        violations = []

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Skip comments (Python #, Ada --, C // and /*)
            if self._is_comment(stripped, pattern.languages):
                continue

            # Check for match
            if not pattern.regex.search(line):
                continue

            # Check if this line is inside a platform/safety guard
            if pattern.guard_patterns:
                context_start = max(0, i - 1 - pattern.context_lines)
                context = "\n".join(lines[context_start:i])

                has_guard = any(
                    re.search(gp, context, re.IGNORECASE) for gp in pattern.guard_patterns
                )

                if has_guard:
                    continue  # Line is guarded, skip

            # Extract code snippet
            snippet_start = max(0, i - 2)
            snippet_end = min(len(lines), i + 1)
            snippet = "\n".join(
                f"  {j+1}: {lines[j]}" for j in range(snippet_start, snippet_end)
            )

            # Build message from template
            message = pattern.message_template
            if "{match}" in message:
                match = pattern.regex.search(line)
                if match:
                    message = message.replace("{match}", match.group(0)[:60])
            if "{line}" in message:
                message = message.replace("{line}", str(i))
            if "{snippet}" in message:
                message = message.replace("{snippet}", stripped[:80])

            violations.append(Violation(
                filepath=filepath,
                line=i,
                severity=pattern.severity,
                category=pattern.category,
                message=message,
                standard=pattern.standard,
                code_snippet=snippet,
            ))

        return violations

    @staticmethod
    def _is_comment(stripped: str, languages: list[str]) -> bool:
        """Check if a line is a comment for any of the target languages."""
        if "python" in languages and stripped.startswith("#"):
            return True
        if "ada" in languages and (stripped.startswith("--") or stripped.startswith("--!")):
            return True
        if "c" in languages and (stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*")):
            return True
        return False


# ══════════════════════════════════════════════════════════════════════════
# PYTHON PATTERNS
# ══════════════════════════════════════════════════════════════════════════

def _build_python_platform_hardcoding_patterns() -> list[Pattern]:
    """Detect hardcoded platform-specific paths without guards."""
    return [
        Pattern(
            name="hardcoded_homebrew_arm64",
            category="PLATFORM_HARDCODING",
            severity=Severity.HIGH,
            standard="ISO/IEC 25010:2021 Portability, CWE-1033",
            description="Hardcoded Homebrew ARM64 path without platform guard",
            languages=["python"],
            regex=re.compile(r"/opt/homebrew/"),
            guard_patterns=[
                r"platform\.system\(\)\s*==\s*['\"]Darwin['\"]",
                r"Platform\.is_macos",
                r"if.*darwin",
            ],
            message_template="Hardcoded Homebrew ARM64 path without platform guard: {snippet}",
        ),
        Pattern(
            name="hardcoded_homebrew_intel",
            category="PLATFORM_HARDCODING",
            severity=Severity.HIGH,
            standard="ISO/IEC 25010:2021 Portability, CWE-1033",
            description="Hardcoded Homebrew Intel path without platform guard",
            languages=["python"],
            regex=re.compile(r"/usr/local/opt/"),
            guard_patterns=[
                r"platform\.system\(\)\s*==\s*['\"]Darwin['\"]",
                r"Platform\.is_macos",
                r"if.*darwin",
            ],
            message_template="Hardcoded Homebrew Intel path without platform guard: {snippet}",
        ),
        Pattern(
            name="hardcoded_python_version",
            category="PLATFORM_HARDCODING",
            severity=Severity.HIGH,
            standard="ISO/IEC 25010:2021 Portability, CWE-1033",
            description="Hardcoded Python version (python3.X) instead of sys.executable",
            languages=["python"],
            regex=re.compile(r"""['"]python3\.\d+['"]"""),
            guard_patterns=[
                r"sys\.executable",
                r"platform",
            ],
            message_template="Hardcoded Python version: {match} — use sys.executable instead",
        ),
        Pattern(
            name="hardcoded_architecture",
            category="PLATFORM_HARDCODING",
            severity=Severity.MEDIUM,
            standard="ISO/IEC 25010:2021 Portability",
            description="Hardcoded architecture string without detection",
            languages=["python"],
            regex=re.compile(r"""['"](osx-arm64|osx-64|linux-64|linux-aarch64)['"]"""),
            guard_patterns=[
                r"platform\.machine\(\)",
                r"Platform\.is_arm64",
                r"Platform\.is_intel",
                r"arch\s*=",
            ],
            message_template="Hardcoded architecture string: {match}",
        ),
        Pattern(
            name="macos_framework_no_guard",
            category="PLATFORM_HARDCODING",
            severity=Severity.HIGH,
            standard="ISO/IEC 25010:2021 Portability",
            description="macOS framework flags without platform guard",
            languages=["python"],
            regex=re.compile(r"""['"]-framework['"].*['"]CoreFoundation['"]"""),
            guard_patterns=[
                r"platform\.system\(\)\s*==\s*['\"]Darwin['\"]",
                r"Platform\.is_macos",
                r"if.*darwin",
            ],
            message_template="macOS framework without platform guard: {snippet}",
        ),
    ]


def _build_python_silent_failure_patterns() -> list[Pattern]:
    """Detect silent return None in critical functions."""
    def check_silent_failures(source: str, lines: list[str], filepath: str = "") -> list[Violation]:
        violations = []

        critical_functions = [
            "derive_master_key",
            "_compute_integrity_hash",
            "compute_integrity_hash",
            "_try_c_derive",
            "load_master_key",
            "adl_crypto",
            "derive_master_key_from_stdin",
        ]

        in_critical_func = False
        func_name = ""

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Track function boundaries
            if stripped.startswith("def "):
                match = re.match(r"def\s+(\w+)", stripped)
                if match:
                    func_name = match.group(1)
                    in_critical_func = any(
                        cf in func_name for cf in critical_functions
                    )

            if not in_critical_func:
                continue

            # Check for bare "return None" (not "return None, None, None")
            if re.match(r"return\s+None\s*$", stripped):
                violations.append(Violation(
                    filepath=filepath,
                    line=i,
                    severity=Severity.CRITICAL,
                    category="SILENT_FAILURE",
                    message=(
                        f"Silent return None in critical function {func_name}() — "
                        f"failure will be invisible. Use Strictness.critical() instead."
                    ),
                    standard="DO-178C §6.3.3, ECSS-Q-ST-80C §7.4",
                    code_snippet=stripped,
                ))

            # Check for except block that returns None
            if stripped.startswith("except"):
                for j in range(i, min(i + 4, len(lines))):
                    if re.match(r"\s+return\s+None\s*$", lines[j]):
                        violations.append(Violation(
                            filepath=filepath,
                            line=i,
                            severity=Severity.CRITICAL,
                            category="SWALLOWED_CRITICAL_EXCEPTION",
                            message=(
                                f"Exception in {func_name}() swallowed with return None "
                                f"— use Strictness.critical() to log and optionally raise"
                            ),
                            standard="DO-178C §6.3.3, MISRA C:2012 Rule 2.2",
                            code_snippet=stripped,
                        ))
                        break

        return violations

    return [
        Pattern(
            name="silent_failure_in_critical_path",
            category="SILENT_FAILURE",
            severity=Severity.CRITICAL,
            standard="DO-178C §6.3.3, ECSS-Q-ST-80C §7.4",
            description="Silent return None in critical crypto/hash functions",
            languages=["python"],
            check_func=check_silent_failures,
        ),
    ]


def _build_python_copy_paste_patterns() -> list[Pattern]:
    """Detect copy-paste bugs where identical logic diverged.

    Uses AST parsing to avoid false positives from string literals,
    regex patterns, and docstrings.
    """
    def check_copy_paste(source: str, lines: list[str], filepath: str = "") -> list[Violation]:
        violations = []

        # ── Pattern 1: subprocess.run(force_kill_process(...)) — AST-aware ──
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                # Look for subprocess.run(...) calls
                if not isinstance(node, ast.Call):
                    continue
                if not (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "run"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess"
                ):
                    continue

                # Check if any argument is force_kill_process(...)
                for arg in node.args:
                    if _is_force_kill_call(arg):
                        violations.append(Violation(
                            filepath=filepath,
                            line=node.lineno,
                            severity=Severity.CRITICAL,
                            category="COPY_PASTE_DIVERGENCE",
                            message=(
                                "subprocess.run() wrapping force_kill_process() — "
                                "force_kill_process returns None, subprocess.run expects "
                                "string/bytes args. Will crash with TypeError."
                            ),
                            standard="CWE-628: Function Call with Incorrectly Specified Arguments",
                            code_snippet=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                        ))
                        break  # One violation per subprocess.run() call

                # Also check for subprocess.run([force_kill_process(...)]) — list arg
                for arg in node.args:
                    if isinstance(arg, ast.List):
                        for elt in arg.elts:
                            if _is_force_kill_call(elt):
                                violations.append(Violation(
                                    filepath=filepath,
                                    line=node.lineno,
                                    severity=Severity.CRITICAL,
                                    category="COPY_PASTE_DIVERGENCE",
                                    message=(
                                        "subprocess.run() wrapping force_kill_process() in list — "
                                        "force_kill_process returns None, subprocess.run expects "
                                        "string/bytes args. Will crash with TypeError."
                                    ),
                                    standard="CWE-628: Function Call with Incorrectly Specified Arguments",
                                    code_snippet=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                                ))
                                break

        except SyntaxError:
            # If AST parsing fails (e.g., Python 2 code, incomplete source),
            # fall back to text-based scanning with string-literal exclusion
            violations.extend(_check_copy_paste_text_fallback(lines, filepath))

        # ── Pattern 2: Duplicate function definitions ──
        func_defs = {}
        for i, line in enumerate(lines, 1):
            match = re.match(r"def\s+(\w+)\s*\(", line.strip())
            if match:
                name = match.group(1)
                if name in func_defs:
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.MEDIUM,
                        category="DUPLICATE_DEFINITION",
                        message=(
                            f"Function '{name}' defined multiple times "
                            f"(first at line {func_defs[name]}) — possible copy-paste divergence"
                        ),
                        standard="MISRA C:2012 Rule 2.5",
                        code_snippet=line.strip(),
                    ))
                else:
                    func_defs[name] = i

        return violations

    return [
        Pattern(
            name="copy_paste_subprocess_misuse",
            category="COPY_PASTE_DIVERGENCE",
            severity=Severity.CRITICAL,
            standard="CWE-628",
            description="subprocess.run() wrapping a function that returns None (AST-aware)",
            languages=["python"],
            check_func=check_copy_paste,
        ),
    ]


def _is_force_kill_call(node: ast.expr) -> bool:
    """Check if an AST node is a call to force_kill_process(...)."""
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id == "force_kill_process"
    return False


def _check_copy_paste_text_fallback(lines: list[str], filepath: str) -> list[Violation]:
    """Text-based fallback for copy-paste detection when AST parsing fails.

    Skips string literals, comments, and regex patterns to avoid false positives.
    """
    violations = []
    in_triple_quote = False
    triple_quote_char = None

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track triple-quoted strings
        if not in_triple_quote:
            if '"""' in stripped or "'''" in stripped:
                # Count triple quotes on this line
                count_3dq = stripped.count('"""')
                count_3sq = stripped.count("'''")
                if count_3dq % 2 == 1:
                    in_triple_quote = True
                    triple_quote_char = '"""'
                elif count_3sq % 2 == 1:
                    in_triple_quote = True
                    triple_quote_char = "'''"
                continue
        else:
            if triple_quote_char in stripped:
                in_triple_quote = False
                triple_quote_char = None
            continue

        # Skip single-line comments
        if stripped.startswith("#"):
            continue

        # Skip lines that are clearly string assignments or regex patterns
        if re.match(r'(r|f|b|u)?["\']', stripped) and "subprocess" not in stripped:
            continue
        if "re.compile" in stripped or "re.search" in stripped:
            continue

        # Check for the pattern in actual code
        context = "\n".join(lines[max(0, i - 2):min(len(lines), i + 3)])
        if re.search(r"subprocess\.run\(\s*\n?\s*force_kill_process\(", context):
            violations.append(Violation(
                filepath=filepath,
                line=i,
                severity=Severity.CRITICAL,
                category="COPY_PASTE_DIVERGENCE",
                message=(
                    "subprocess.run() wrapping force_kill_process() — "
                    "force_kill_process returns None, subprocess.run expects "
                    "string/bytes args. Will crash with TypeError."
                ),
                standard="CWE-628: Function Call with Incorrectly Specified Arguments",
                code_snippet=stripped,
            ))

    return violations


def _build_python_stale_reference_patterns() -> list[Pattern]:
    """Detect hardcoded line numbers in error messages that become stale."""
    def check_stale_refs(source: str, lines: list[str], filepath: str = "") -> list[Violation]:
        violations = []

        for i, line in enumerate(lines, 1):
            match = re.search(r"at line (\d+)", line)
            if match:
                claimed_line = int(match.group(1))
                if abs(i - claimed_line) > 20:
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.MEDIUM,
                        category="STALE_LINE_REFERENCE",
                        message=(
                            f"Claims exception at line {claimed_line} but is on line {i} "
                            f"(delta: {abs(i - claimed_line)} lines) — "
                            f"use function name instead of line number"
                        ),
                        standard="ECSS-Q-ST-80C §7.5: Error Reporting",
                        code_snippet=line.strip()[:100],
                    ))

        return violations

    return [
        Pattern(
            name="stale_line_number_reference",
            category="STALE_LINE_REFERENCE",
            severity=Severity.MEDIUM,
            standard="ECSS-Q-ST-80C §7.5",
            description="Hardcoded line number in error message is stale",
            languages=["python"],
            check_func=check_stale_refs,
        ),
    ]


def _build_python_dead_code_patterns() -> list[Pattern]:
    """Detect dead code: if True, if False."""
    return [
        Pattern(
            name="always_true_condition",
            category="DEAD_CODE",
            severity=Severity.MEDIUM,
            standard="MISRA C:2012 Rule 2.2: Dead code",
            description="Always-true condition (if True:)",
            languages=["python"],
            regex=re.compile(r"^\s*if\s+True\s*:\s*$"),
            message_template="Always-true condition: {snippet} — remove or replace with real condition",
        ),
        Pattern(
            name="always_false_condition",
            category="DEAD_CODE",
            severity=Severity.MEDIUM,
            standard="MISRA C:2012 Rule 2.2: Dead code",
            description="Always-false condition (if False:)",
            languages=["python"],
            regex=re.compile(r"^\s*if\s+False\s*:\s*$"),
            message_template="Always-false condition: {snippet} — dead code, remove",
        ),
    ]


def _build_python_resource_leak_patterns() -> list[Pattern]:
    """Detect resource leaks: subprocess.Popen without cleanup."""
    def check_resource_leaks(source: str, lines: list[str], filepath: str = "") -> list[Violation]:
        violations = []

        popen_calls = []
        for i, line in enumerate(lines, 1):
            if "subprocess.Popen(" in line:
                match = re.search(r"(\w+)\s*=\s*subprocess\.Popen\(", line)
                if match:
                    popen_calls.append((i, match.group(1)))

        for line_no, var_name in popen_calls:
            has_cleanup = False
            search_end = min(line_no + 200, len(lines))

            for j in range(line_no, search_end):
                check_line = lines[j]
                if (
                    f"{var_name}.kill()" in check_line
                    or f"{var_name}.terminate()" in check_line
                    or f"{var_name}.wait()" in check_line
                    or f"{var_name}.stdin.close()" in check_line
                ):
                    has_cleanup = True
                    break

            if not has_cleanup:
                violations.append(Violation(
                    filepath=filepath,
                    line=line_no,
                    severity=Severity.MEDIUM,
                    category="RESOURCE_LEAK",
                    message=(
                        f"subprocess.Popen assigned to '{var_name}' but no "
                        f"kill()/terminate()/wait()/stdin.close() found within 200 lines"
                    ),
                    standard="CWE-775: Missing Release of Resource, CERT FIO42-C",
                    code_snippet=f"{var_name} = subprocess.Popen(...)",
                ))

        return violations

    return [
        Pattern(
            name="subprocess_resource_leak",
            category="RESOURCE_LEAK",
            severity=Severity.MEDIUM,
            standard="CWE-775, CERT FIO42-C",
            description="subprocess.Popen without corresponding cleanup",
            languages=["python"],
            check_func=check_resource_leaks,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════
# SOFTLOCK DETECTION PATTERNS
# ══════════════════════════════════════════════════════════════════════════

def _build_python_softlock_patterns() -> list[Pattern]:
    """Detect softlock patterns: hangs, infinite loops, deadlocks.

    Softlocks are insidious because the system appears alive but is actually stuck.
    Unlike crashes (which are loud and obvious), softlocks silently consume resources
    and block progress without any error output.
    """
    def check_softlocks(source: str, lines: list[str], filepath: str = "") -> list[Violation]:
        violations = []

        # ── Pattern 1: subprocess.run() without timeout ──
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue

                # Check for subprocess.run() calls
                is_subprocess_run = (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "run"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess"
                )
                # Also check for bare run() if 'from subprocess import run'
                is_bare_run = (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "run"
                    and "from subprocess import" in source
                )

                if not (is_subprocess_run or is_bare_run):
                    continue

                # Check if timeout is in keyword arguments
                has_timeout = any(
                    kw.arg == "timeout" for kw in node.keywords
                )
                # Also check for timeout in *args (unlikely but possible)
                # subprocess.run([...], timeout=30) is the normal form

                if not has_timeout:
                    violations.append(Violation(
                        filepath=filepath,
                        line=node.lineno,
                        severity=Severity.HIGH,
                        category="SOFTLOCK_RISK",
                        message=(
                            "subprocess.run() without timeout — if the child process "
                            "deadlocks or hangs, this thread will block forever. "
                            "Add timeout= parameter (e.g., timeout=300)."
                        ),
                        standard="CERT FIO47-C, CWE-835: Loop with Unreachable Exit Condition",
                        code_snippet=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                    ))

        except SyntaxError:
            pass  # Can't parse — skip AST-based checks

        # ── Pattern 2: Infinite while True loops without break/return ──
        in_loop = False
        loop_start = 0
        loop_indent = 0

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            current_indent = len(line) - len(line.lstrip())

            # Detect while True: or while 1:
            if re.match(r"while\s+(True|1)\s*:", stripped):
                in_loop = True
                loop_start = i
                loop_indent = current_indent
                continue

            if in_loop:
                # Check if we're still inside the loop (indentation-based)
                if current_indent <= loop_indent and stripped and not stripped.startswith("#"):
                    # Exited the loop — check if break/return was found
                    in_loop = False
                    continue

                # Look for break or return inside the loop
                if "break" in stripped or "return" in stripped:
                    in_loop = False  # Loop has an exit condition

        # If we ended still inside a loop, it's infinite
        if in_loop:
            violations.append(Violation(
                filepath=filepath,
                line=loop_start,
                severity=Severity.HIGH,
                category="SOFTLOCK_RISK",
                message=(
                    "while True loop without break/return — "
                    "infinite loop will block this thread forever. "
                    "Add exit condition or timeout."
                ),
                standard="CERT FIO47-C, CWE-835: Loop with Unreachable Exit Condition",
                code_snippet=lines[loop_start - 1].strip() if loop_start <= len(lines) else "",
            ))

        # ── Pattern 3: Recursive functions without base case ──
        func_defs = []
        for i, line in enumerate(lines, 1):
            match = re.match(r"def\s+(\w+)\s*\(([^)]*)\)\s*(?:->.*?)?:", line.strip())
            if match:
                func_defs.append((i, match.group(1), match.group(2)))

        for line_no, func_name, params in func_defs:
            # Find the function body
            func_indent = len(lines[line_no - 1]) - len(lines[line_no - 1].lstrip())
            body_start = line_no
            body_end = line_no

            for j in range(line_no, min(line_no + 100, len(lines))):
                body_line = lines[j]
                if body_line.strip() and not body_line.strip().startswith("#"):
                    body_indent = len(body_line) - len(body_line.lstrip())
                    if body_indent > func_indent:
                        body_end = j
                    elif body_indent <= func_indent and j > line_no:
                        break

            # Check if the function calls itself
            body_text = "\n".join(lines[base] for base in range(body_start - 1, body_end + 1) if base < len(lines))
            if f"{func_name}(" not in body_text:
                continue  # Not recursive

            # Check for base case: if/return before recursive call
            has_base_case = False
            for j in range(body_start - 1, min(body_end + 1, len(lines))):
                body_line = lines[j].strip()
                if body_line.startswith("if ") and ("return" in body_line or "==" in body_line or "<=" in body_line or ">=" in body_line):
                    has_base_case = True
                    break

            if not has_base_case:
                violations.append(Violation(
                    filepath=filepath,
                    line=line_no,
                    severity=Severity.HIGH,
                    category="SOFTLOCK_RISK",
                    message=(
                        f"Recursive function '{func_name}()' without apparent base case — "
                        f"will cause infinite recursion (stack overflow or hang). "
                        f"Add a termination condition."
                    ),
                    standard="CERT FIO47-C, CWE-674: Uncontrolled Recursion",
                    code_snippet=lines[line_no - 1].strip() if line_no <= len(lines) else "",
                ))

        # ── Pattern 4: time.sleep() in loops without timeout ──
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if "time.sleep(" in stripped:
                # Check if this is inside a while loop
                for j in range(i - 1, max(0, i - 100), -1):
                    check_line = lines[j].strip()
                    if re.match(r"while\s+(True|1)\s*:", check_line):
                        # Check if sleep is followed by break/return
                        has_exit = False
                        for k in range(i, min(i + 5, len(lines))):
                            if "break" in lines[k] or "return" in lines[k]:
                                has_exit = True
                                break
                        if not has_exit:
                            violations.append(Violation(
                                filepath=filepath,
                                line=i,
                                severity=Severity.MEDIUM,
                                category="SOFTLOCK_RISK",
                                message=(
                                    "time.sleep() in while loop without break/return — "
                                    "polling loop may run indefinitely. Consider adding "
                                    "a max iteration count or timeout."
                                ),
                                standard="CWE-835: Loop with Unreachable Exit Condition",
                                code_snippet=stripped,
                            ))
                        break

        return violations

    return [
        Pattern(
            name="subprocess_no_timeout",
            category="SOFTLOCK_RISK",
            severity=Severity.HIGH,
            standard="CERT FIO47-C, CWE-835",
            description="subprocess.run() without timeout — may hang forever",
            languages=["python"],
            check_func=check_softlocks,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════
# REDUNDANT / ILLOGICAL / FILE REFERENCE PATTERNS
# ══════════════════════════════════════════════════════════════════════════

def _build_python_redundant_logic_patterns() -> list[Pattern]:
    """Detect redundant code, illogical operations, and invalid file references.

    These patterns indicate either:
    - Copy-paste errors (code that does nothing)
    - Deliberate sabotage (code that contradicts itself)
    - Sloppy maintenance (stale references, broken paths)
    """
    def check_redundant_logic(source: str, lines: list[str], filepath: str = "") -> list[Violation]:
        violations = []

        # ── Pattern 1: Self-assignment (x = x) ──
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("--"):
                continue

            # Match: variable = variable (exact self-assignment)
            self_assign = re.match(r"^(\w+)\s*=\s*(\1)\s*$", stripped)
            if self_assign:
                var_name = self_assign.group(1)
                # Exclude loop variables and common patterns
                if var_name not in ("i", "j", "k", "n", "_", "self", "cls"):
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.MEDIUM,
                        category="REDUNDANT_LOGIC",
                        message=(
                            f"Self-assignment: {var_name} = {var_name} — "
                            f"this statement has no effect. Either remove it or "
                            f"there's a copy-paste error."
                        ),
                        standard="MISRA C:2012 Rule 2.2, CWE-563: Assignment to Variable Not Used",
                        code_snippet=stripped,
                    ))

        # ── Pattern 2: Tautological conditions (if x == x, if x and not x) ──
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # if x == x (always true)
            taut_true = re.match(r"if\s+(\w+)\s*==\s*(\1)\s*:", stripped)
            if taut_true:
                violations.append(Violation(
                    filepath=filepath,
                    line=i,
                    severity=Severity.HIGH,
                    category="REDUNDANT_LOGIC",
                    message=(
                        f"Tautological condition: {taut_true.group(1)} == {taut_true.group(1)} — "
                        f"always true. This is either a bug or dead code."
                    ),
                    standard="CWE-561: Dead Code, MISRA C:2012 Rule 2.2",
                    code_snippet=stripped,
                ))

            # if x != x (always false)
            taut_false = re.match(r"if\s+(\w+)\s*!=\s*(\1)\s*:", stripped)
            if taut_false:
                violations.append(Violation(
                    filepath=filepath,
                    line=i,
                    severity=Severity.HIGH,
                    category="REDUNDANT_LOGIC",
                    message=(
                        f"Tautological condition: {taut_false.group(1)} != {taut_false.group(1)} — "
                        f"always false. Dead code will never execute."
                    ),
                    standard="CWE-561: Dead Code, MISRA C:2012 Rule 2.2",
                    code_snippet=stripped,
                ))

            # if x and not x (always false)
            if re.match(r"if\s+(\w+)\s+and\s+not\s+(\1)\s*:", stripped):
                var = re.match(r"if\s+(\w+)\s+and\s+not\s+\w+\s*:", stripped).group(1)
                violations.append(Violation(
                    filepath=filepath,
                    line=i,
                    severity=Severity.HIGH,
                    category="REDUNDANT_LOGIC",
                    message=(
                        f"Contradictory condition: {var} and not {var} — "
                        f"always false. Dead code will never execute."
                    ),
                    standard="CWE-561: Dead Code",
                    code_snippet=stripped,
                ))

            # if x or not x (always true)
            if re.match(r"if\s+(\w+)\s+or\s+not\s+(\1)\s*:", stripped):
                var = re.match(r"if\s+(\w+)\s+or\s+not\s+\w+\s*:", stripped).group(1)
                violations.append(Violation(
                    filepath=filepath,
                    line=i,
                    severity=Severity.MEDIUM,
                    category="REDUNDANT_LOGIC",
                    message=(
                        f"Tautological condition: {var} or not {var} — "
                        f"always true. Conditional is meaningless."
                    ),
                    standard="CWE-561: Dead Code",
                    code_snippet=stripped,
                ))

            # if True: / if False:
            if re.match(r"if\s+True\s*:", stripped):
                violations.append(Violation(
                    filepath=filepath,
                    line=i,
                    severity=Severity.MEDIUM,
                    category="REDUNDANT_LOGIC",
                    message="if True: — unconditional branch. Remove the if or fix the condition.",
                    standard="CWE-561: Dead Code",
                    code_snippet=stripped,
                ))
            if re.match(r"if\s+False\s*:", stripped):
                violations.append(Violation(
                    filepath=filepath,
                    line=i,
                    severity=Severity.HIGH,
                    category="REDUNDANT_LOGIC",
                    message="if False: — dead code block will never execute.",
                    standard="CWE-561: Dead Code",
                    code_snippet=stripped,
                ))

        # ── Pattern 3: Pointless return (return None at end of void function) ──
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                if not node.body:
                    continue

                last_stmt = node.body[-1]
                if isinstance(last_stmt, ast.Return) and last_stmt.value is None:
                    # Check if the function has any other return statements
                    other_returns = [
                        n for n in ast.walk(node)
                        if isinstance(n, ast.Return) and n is not last_stmt
                    ]
                    if not other_returns:
                        # No other returns — this is a void function with pointless return
                        violations.append(Violation(
                            filepath=filepath,
                            line=last_stmt.lineno,
                            severity=Severity.LOW,
                            category="REDUNDANT_LOGIC",
                            message=(
                                f"Function '{node.name}()' ends with return None — "
                                f"implicit return is equivalent. Remove for clarity."
                            ),
                            standard="MISRA C:2012 Rule 2.2",
                            code_snippet="return None",
                        ))

        except SyntaxError:
            pass

        # ── Pattern 4: File path invalidation ──
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # os.path.join with absolute path (overwrites previous components)
            abs_in_join = re.search(r"os\.path\.join\([^)]*['\"]/[a-zA-Z]", stripped)
            if abs_in_join:
                violations.append(Violation(
                    filepath=filepath,
                    line=i,
                    severity=Severity.HIGH,
                    category="INVALID_FILE_REFERENCE",
                    message=(
                        "os.path.join() with absolute path — absolute path overrides "
                        "all previous join components. Use relative paths or Path / operator."
                    ),
                    standard="CWE-22: Path Traversal, CWE-798: Hard-coded Credentials",
                    code_snippet=stripped,
                ))

            # Path('...') with // or trailing /
            double_slash = re.search(r"Path\(['\"].*//", stripped)
            if double_slash:
                violations.append(Violation(
                    filepath=filepath,
                    line=i,
                    severity=Severity.MEDIUM,
                    category="INVALID_FILE_REFERENCE",
                    message=(
                        "Path() with double slash (//) — likely a path construction error. "
                        "Use Path / operator instead of string concatenation."
                    ),
                    standard="CWE-22: Path Traversal",
                    code_snippet=stripped,
                ))

            # __file__ used with os.path.dirname twice (common mistake)
            if stripped.count("__file__") >= 2 and "dirname" in stripped:
                violations.append(Violation(
                    filepath=filepath,
                    line=i,
                    severity=Severity.MEDIUM,
                    category="INVALID_FILE_REFERENCE",
                    message=(
                        "Multiple __file__ references with dirname — likely a path "
                        "construction error. Use pathlib.Path(__file__).parent instead."
                    ),
                    standard="CWE-22: Path Traversal",
                    code_snippet=stripped,
                ))

            # open() with path that looks like a template (has { or %)
            if "open(" in stripped and ("{" in stripped or "%s" in stripped or "%d" in stripped):
                # Check if it's an f-string or format call
                if not stripped.startswith("f'") and not stripped.startswith('f"'):
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.HIGH,
                        category="INVALID_FILE_REFERENCE",
                        message=(
                            "open() with template-style path — path may not be formatted "
                            "before use. Verify the path is interpolated correctly."
                        ),
                        standard="CWE-22: Path Traversal",
                        code_snippet=stripped,
                    ))

            # Hardcoded paths that look like placeholders
            placeholder_patterns = [
                r"['\"]/(tmp|var|usr|etc)/\w*\.\w+['\"]",  # /tmp/something.ext
                r"['\"]/(TODO|FIXME|CHANGEME|XXX|PLACEHOLDER)",  # Placeholder markers
                r"['\"]\.?/(TODO|FIXME|CHANGEME|XXX|PLACEHOLDER)",  # Relative placeholders
            ]
            for pattern in placeholder_patterns:
                if re.search(pattern, stripped, re.IGNORECASE):
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.HIGH,
                        category="INVALID_FILE_REFERENCE",
                        message=(
                            "Hardcoded path appears to be a placeholder — "
                            "file will not exist at runtime. Replace with actual path."
                        ),
                        standard="CWE-22: Path Traversal",
                        code_snippet=stripped,
                    ))
                    break

        return violations

    return [
        Pattern(
            name="redundant_logic",
            category="REDUNDANT_LOGIC",
            severity=Severity.MEDIUM,
            standard="MISRA C:2012 Rule 2.2, CWE-561",
            description="Redundant code, tautological conditions, self-assignment",
            languages=["python"],
            check_func=check_redundant_logic,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════
# EXCEPTION HANDLING & COVERAGE GAP PATTERNS
# ══════════════════════════════════════════════════════════════════════════

def _build_python_exception_patterns() -> list[Pattern]:
    """Detect missing exception handling that causes random crashes.

    These patterns indicate code that will crash unpredictably because
    exceptions are not caught, or are caught incorrectly.
    """
    def check_exceptions(source: str, lines: list[str], filepath: str = "") -> list[Violation]:
        violations = []

        # ── Pattern 1: Bare except (catches everything including SystemExit, KeyboardInterrupt) ──
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # bare except:
            if re.match(r"except\s*:", stripped):
                # Check if the handler actually does something useful
                has_action = False
                for j in range(i, min(i + 5, len(lines))):
                    handler_line = lines[j].strip()
                    if handler_line and not handler_line.startswith("except") and not handler_line.startswith("#"):
                        if handler_line not in ("pass", "...", "continue"):
                            has_action = True
                            break

                if not has_action:
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.HIGH,
                        category="EXCEPTION_MISSING",
                        message=(
                            "Bare 'except:' with no action — silently swallows ALL exceptions "
                            "including SystemExit and KeyboardInterrupt. Use 'except Exception:' "
                            "at minimum, and log the error."
                        ),
                        standard="CERT ERR00-C, MISRA C++:2008 Rule 15.5.2",
                        code_snippet=stripped,
                    ))

        # ── Pattern 2: except Exception: pass (silently swallowed) ──
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            if re.match(r"except\s+(Exception|BaseException)\s*:", stripped):
                # Check if the handler is just 'pass' or '...'
                for j in range(i, min(i + 3, len(lines))):
                    handler_line = lines[j].strip()
                    if handler_line in ("pass", "..."):
                        violations.append(Violation(
                            filepath=filepath,
                            line=i,
                            severity=Severity.HIGH,
                            category="EXCEPTION_MISSING",
                            message=(
                                f"Exception silently swallowed: {stripped} → {handler_line} — "
                                f"error will be invisible. At minimum, log the exception."
                            ),
                            standard="CERT ERR00-C, CWE-390: Detection of Error Condition Without Action",
                            code_snippet=stripped,
                        ))
                        break

        # ── Pattern 3: except without logging or re-raising ──
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            if re.match(r"except\s+\w+", stripped):
                # Check if the handler logs, re-raises, or returns error
                has_handling = False
                for j in range(i, min(i + 10, len(lines))):
                    handler_line = lines[j].strip()
                    if not handler_line or handler_line.startswith("#"):
                        continue
                    # Good patterns: logging, print, raise, return error
                    if any(kw in handler_line for kw in ("logging", "logger", "print(", "raise", "return False", "return None", "Strictness")):
                        has_handling = True
                        break
                    # Exit handler if we hit another except/finally/def/class
                    if handler_line.startswith(("except", "finally", "def ", "class ")) and j > i:
                        break

                if not has_handling:
                    # Get the exception type
                    exc_match = re.match(r"except\s+(\w+)", stripped)
                    exc_type = exc_match.group(1) if exc_match else "Exception"
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.MEDIUM,
                        category="EXCEPTION_MISSING",
                        message=(
                            f"except {exc_type}: without logging, re-raising, or error return — "
                            f"failure will be invisible. Add logging or raise."
                        ),
                        standard="CERT ERR00-C, CWE-390",
                        code_snippet=stripped,
                    ))

        # ── Pattern 4: Unreachable code after return/raise/continue/break ──
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.For, ast.While, ast.If, ast.With, ast.Try)):
                    continue

                body_list = node.body if hasattr(node, "body") else []
                if hasattr(node, "orelse") and node.orelse:
                    # Check orelse too (else blocks on for/if)
                    pass

                for idx, stmt in enumerate(body_list):
                    if isinstance(stmt, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
                        # Check if there's code after this statement
                        if idx + 1 < len(body_list):
                            next_stmt = body_list[idx + 1]
                            # Skip if the next statement is a function/class def (those are fine)
                            if isinstance(next_stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                continue
                            violations.append(Violation(
                                filepath=filepath,
                                line=next_stmt.lineno,
                                severity=Severity.HIGH,
                                category="EXCEPTION_MISSING",
                                message=(
                                    f"Unreachable code after {type(stmt).__name__} at line {stmt.lineno} — "
                                    f"this code will never execute. Remove it or fix the control flow."
                                ),
                                standard="MISRA C:2012 Rule 2.2, CWE-561: Dead Code",
                                code_snippet=lines[next_stmt.lineno - 1].strip() if next_stmt.lineno <= len(lines) else "",
                            ))

        except SyntaxError:
            pass

        # ── Pattern 5: File/IO operations without try/except ──
        io_operations = [
            (r"\bopen\(", "open()"),
            (r"\bos\.(remove|rename|makedirs|rmdir)\(", "filesystem operation"),
            (r"\bpathlib.*\.read_text\(", "Path.read_text()"),
            (r"\bpathlib.*\.write_text\(", "Path.write_text()"),
            (r"\bjson\.load\(", "json.load()"),
            (r"\bjson\.dump\(", "json.dump()"),
            (r"\bcsv\.\w+Reader\(", "csv reader"),
            (r"\bcsv\.\w+Writer\(", "csv writer"),
        ]

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            for pattern, op_name in io_operations:
                if re.search(pattern, stripped):
                    # Check if this line is inside a try block
                    in_try = False
                    for j in range(i - 1, max(0, i - 50), -1):
                        check_line = lines[j].strip()
                        if check_line.startswith("try") and (check_line == "try:" or check_line.startswith("try:")):
                            in_try = True
                            break
                        # If we hit a function def, we're not in a try block
                        if check_line.startswith(("def ", "class ")) and j < i - 1:
                            break

                    if not in_try:
                        violations.append(Violation(
                            filepath=filepath,
                            line=i,
                            severity=Severity.MEDIUM,
                            category="EXCEPTION_MISSING",
                            message=(
                                f"{op_name} without try/except — will crash on file not found, "
                                f"permission denied, or I/O error. Wrap in try/except."
                            ),
                            standard="CERT ERR33-C, CWE-703: Improper Check or Handling of Exceptional Conditions",
                            code_snippet=stripped,
                        ))
                    break

        # ── Pattern 6: Missing None check before attribute access ──
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                # Check if the value is a function call that might return None
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name):
                        func_name = node.value.func.id
                        # Common functions that might return None
                        risk_funcs = {
                            "get", "dict.get", "os.environ.get", "json.loads",
                            "re.search", "re.match", "re.findall",
                        }
                        if func_name in risk_funcs or "." in func_name:
                            # Check if there's a None check before this
                            # Look for: if result is not None: / if result: / if result != None:
                            has_check = False
                            # Simple heuristic: look in enclosing scope
                            for j in range(max(0, node.lineno - 10), node.lineno):
                                check_line = lines[j] if j < len(lines) else ""
                                if func_name in check_line and ("is not None" in check_line or "if " in check_line):
                                    has_check = True
                                    break

                            if not has_check:
                                violations.append(Violation(
                                    filepath=filepath,
                                    line=node.lineno,
                                    severity=Severity.MEDIUM,
                                    category="EXCEPTION_MISSING",
                                    message=(
                                        f"Attribute access on potential None from {func_name}() — "
                                        f"add 'if result is not None:' check."
                                    ),
                                    standard="CWE-476: NULL Pointer Dereference",
                                    code_snippet=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                                ))

        except SyntaxError:
            pass

        return violations

    return [
        Pattern(
            name="exception_handling",
            category="EXCEPTION_MISSING",
            severity=Severity.HIGH,
            standard="CERT ERR00-C, CWE-390, CWE-703",
            description="Missing exception handling, silent swallowing, unreachable code",
            languages=["python"],
            check_func=check_exceptions,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════
# STALE FLAG / TIME-BASED DETECTION PATTERNS
# ══════════════════════════════════════════════════════════════════════════

def _build_python_stale_flag_patterns() -> list[Pattern]:
    """Detect stale flags, never-modified conditions, and time-based logic errors.

    These patterns indicate:
    - Flags that are checked but never updated (always True/False)
    - Time comparisons that are stale (checking old timestamps)
    - Cache invalidation that never happens
    - Conditions that are always True/False due to never-modified variables
    """
    def check_stale_flags(source: str, lines: list[str], filepath: str = "") -> list[Violation]:
        violations = []

        # ── Pattern 1: Boolean flags set to True/False but never modified ──
        # Track boolean assignments
        bool_assignments = {}  # var_name -> [(line_no, value), ...]

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Match: flag = True / flag = False / flag: bool = True
            match = re.match(r"(\w+)\s*(?::\s*bool\s*)?=\s*(True|False)\s*$", stripped)
            if match:
                var_name = match.group(1)
                value = match.group(2)
                if var_name not in bool_assignments:
                    bool_assignments[var_name] = []
                bool_assignments[var_name].append((i, value))

        # Check each boolean flag
        for var_name, assignments in bool_assignments.items():
            if len(assignments) != 1:
                continue  # Flag is modified multiple times — not stale

            # Skip common patterns that are expected to be constant
            if var_name.startswith("_") and var_name.endswith("_"):
                continue
            if var_name in ("DEBUG", "VERBOSE", "TESTING", "DRY_RUN"):
                continue

            initial_value = assignments[0][1]

            # Check if this flag is ever read in a condition
            is_read = False
            is_written_again = False

            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue

                # Check if flag is used in a condition
                if re.search(rf"\bif\s+{re.escape(var_name)}\b", stripped):
                    is_read = True
                if re.search(rf"\bif\s+not\s+{re.escape(var_name)}\b", stripped):
                    is_read = True
                if re.search(rf"\bwhile\s+{re.escape(var_name)}\b", stripped):
                    is_read = True

                # Check if flag is modified after initial assignment
                if re.search(rf"^{re.escape(var_name)}\s*=\s*(True|False)", stripped):
                    if i != assignments[0][0]:
                        is_written_again = True

            if is_read and not is_written_again:
                # Flag is read but never modified — stale!
                violations.append(Violation(
                    filepath=filepath,
                    line=assignments[0][0],
                    severity=Severity.HIGH,
                    category="STALE_FLAG",
                    message=(
                        f"Boolean flag '{var_name}' = {initial_value} is never modified — "
                        f"condition using it is always {initial_value}. "
                        f"Either update the flag or remove the dead branch."
                    ),
                    standard="CWE-561: Dead Code, CWE-835: Loop with Unreachable Exit Condition",
                    code_snippet=f"{var_name} = {initial_value}",
                ))

        # ── Pattern 2: Stale time comparisons (comparing to old timestamps) ──
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Match: if time.time() - last_time > HUGE_NUMBER
            time_compare = re.search(
                r"if\s+time\.time\(\)\s*-\s*(\w+)\s*>\s*(\d+)",
                stripped
            )
            if time_compare:
                threshold = int(time_compare.group(2))
                if threshold > 86400:  # More than 24 hours
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.MEDIUM,
                        category="STALE_FLAG",
                        message=(
                            f"Time comparison threshold is {threshold} seconds "
                            f"({threshold // 3600} hours) — may be stale. "
                            f"Consider if this threshold is still appropriate."
                        ),
                        standard="CWE-835: Loop with Unreachable Exit Condition",
                        code_snippet=stripped,
                    ))

            # Match: if datetime.now() - last_check > timedelta(days=999)
            datetime_compare = re.search(
                r"if\s+datetime\.now\(\)\s*-\s*(\w+)\s*>\s*timedelta\((?:days\s*=\s*)?(\d+)\)",
                stripped
            )
            if datetime_compare:
                days = int(datetime_compare.group(2))
                if days > 365:
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.MEDIUM,
                        category="STALE_FLAG",
                        message=(
                            f"Datetime comparison threshold is {days} days "
                            f"({days // 365} years) — likely stale. "
                            f"Consider if this threshold is still appropriate."
                        ),
                        standard="CWE-835: Loop with Unreachable Exit Condition",
                        code_snippet=stripped,
                    ))

        # ── Pattern 3: Cache invalidation without expiry ──
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Match: cache = {} or cache = dict() or cache = {}
            cache_init = re.match(r"(\w+)\s*[:=]\s*(?:\{\}|dict\(\))", stripped)
            if cache_init:
                cache_var = cache_init.group(1)
                if "cache" in cache_var.lower():
                    # Check if this cache is ever cleared
                    has_clear = False
                    for j in range(i, min(i + 200, len(lines))):
                        check_line = lines[j].strip()
                        if f"{cache_var}.clear()" in check_line or f"{cache_var} = " in check_line:
                            has_clear = True
                            break

                    if not has_clear:
                        violations.append(Violation(
                            filepath=filepath,
                            line=i,
                            severity=Severity.MEDIUM,
                            category="STALE_FLAG",
                            message=(
                                f"Cache '{cache_var}' initialized but never cleared — "
                                f"may grow unbounded or serve stale data. "
                                f"Add cache.clear() or TTL-based expiry."
                            ),
                            standard="CWE-400: Uncontrolled Resource Consumption, CWE-665: Improper Initialization",
                            code_snippet=stripped,
                        ))

        # ── Pattern 4: Flags set in conditional but always True/False ──
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.If):
                    continue

                # Check if condition is a constant True/False
                if isinstance(node.test, ast.Constant):
                    if node.test.value is True:
                        violations.append(Violation(
                            filepath=filepath,
                            line=node.lineno,
                            severity=Severity.MEDIUM,
                            category="STALE_FLAG",
                            message=(
                                "if True: — unconditional branch, always taken. "
                                "Remove the if or fix the condition."
                            ),
                            standard="CWE-561: Dead Code",
                            code_snippet=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                        ))
                    elif node.test.value is False:
                        violations.append(Violation(
                            filepath=filepath,
                            line=node.lineno,
                            severity=Severity.HIGH,
                            category="STALE_FLAG",
                            message=(
                                "if False: — dead code, never executed. "
                                "Remove this block."
                            ),
                            standard="CWE-561: Dead Code",
                            code_snippet=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                        ))

                # Check for: if variable is always True/False based on assignment
                if isinstance(node.test, ast.Name):
                    var_name = node.test.id
                    # Look for assignment before this if
                    for j in range(max(0, node.lineno - 20), node.lineno):
                        check_line = lines[j].strip()
                        assign_match = re.match(rf"^{re.escape(var_name)}\s*=\s*(True|False)", check_line)
                        if assign_match:
                            # Check if there's any reassignment between assignment and this if
                            has_reassignment = False
                            for k in range(j + 1, node.lineno):
                                reassign_line = lines[k].strip()
                                if re.match(rf"^{re.escape(var_name)}\s*=", reassign_line):
                                    has_reassignment = True
                                    break

                            if not has_reassignment:
                                value = assign_match.group(1)
                                violations.append(Violation(
                                    filepath=filepath,
                                    line=node.lineno,
                                    severity=Severity.MEDIUM,
                                    category="STALE_FLAG",
                                    message=(
                                        f"if {var_name}: — {var_name} = {value} (set {node.lineno - j} lines above), "
                                        f"never modified. Condition is always {value}."
                                    ),
                                    standard="CWE-561: Dead Code",
                                    code_snippet=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                                ))
                            break

        except SyntaxError:
            pass

        return violations

    return [
        Pattern(
            name="stale_flags",
            category="STALE_FLAG",
            severity=Severity.HIGH,
            standard="CWE-561, CWE-835, CWE-400",
            description="Stale flags, never-modified booleans, time comparison errors",
            languages=["python"],
            check_func=check_stale_flags,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════
# COQ PROOF VERIFICATION PATTERNS
# ══════════════════════════════════════════════════════════════════════════

def _build_coq_proof_patterns() -> list[Pattern]:
    """Detect Coq .v proof fraud: Admitted, Axiom, missing proofs.

    In aerospace-grade verification (DO-178C, ECSS), EVERY source unit
    (Ada, Python, C) MUST have a corresponding Coq proof. Code without
    proof is FRAUD.
    """
    def check_coq_proofs(source: str, lines: list[str], filepath: str = "") -> list[Violation]:
        violations = []
        if not filepath:
            return violations

        is_coq = filepath.endswith(".v")
        is_ada = filepath.endswith((".adb", ".ads"))
        is_python = filepath.endswith(".py")
        is_c = filepath.endswith((".c", ".h"))

        # Skip vendor, tests, and build artifacts
        skip_dirs = ["vendor", "node_modules", "__pycache__", ".git", "build", "tests"]
        if any(skip_dir in filepath for skip_dir in skip_dirs):
            return violations

        if is_coq:
            # ── Coq-specific checks ──
            for i, line in enumerate(lines, 1):
                stripped = line.strip()

                # Skip comments
                if stripped.startswith("(*") or stripped.startswith("--"):
                    continue

                # Admitted. = proof not finished — FRAUD
                if re.match(r"Admitted\s*\.", stripped):
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.CRITICAL,
                        category="PROOF_MISSING",
                        message=(
                            "Admitted. — proof incomplete, verification bypassed. "
                            "This is FRAUD. Every theorem MUST be proved with Qed or Defined. "
                            "No exceptions. No excuses.\n"
                            "JUSTIFICATION: Replace 'Admitted.' with actual proof. "
                            "If truly impossible, add: '(* JUSTIFICATION: <reason> *)' "
                            "above and document in design records."
                        ),
                        standard="DO-178C §5.2.2, ECSS-Q-ST-80C §6.3",
                        code_snippet=stripped,
                    ))

                # Axiom = unproven assumption — FRAUD
                if re.match(r"Axiom\s+\w+", stripped):
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.CRITICAL,
                        category="PROOF_MISSING",
                        message=(
                            "Axiom declared without proof — unproven assumption. "
                            "Every axiom MUST be justified and documented. "
                            "Use 'Parameter' with justification or prove it.\n"
                            "JUSTIFICATION: Add comment above Axiom: "
                            "'(* JUSTIFICATION: <reason> *)' "
                            "and document in design records."
                        ),
                        standard="DO-178C §5.2.2, ECSS-Q-ST-80C §6.3",
                        code_snippet=stripped,
                    ))

                # Parameter without Proof — unproven assumption
                if re.match(r"Parameter\s+\w+", stripped):
                    # Check if there's a Proof later
                    has_proof = False
                    for j in range(i, min(i + 50, len(lines))):
                        if "Proof" in lines[j] or "Qed" in lines[j]:
                            has_proof = True
                            break
                    if not has_proof:
                        violations.append(Violation(
                            filepath=filepath,
                            line=i,
                            severity=Severity.HIGH,
                            category="PROOF_MISSING",
                            message=(
                                "Parameter without Proof — unproven assumption. "
                                "Add a proof or document why this is safe."
                            ),
                            standard="DO-178C §5.2.2",
                            code_snippet=stripped,
                        ))

                # ── Cheap proof detection ──

                # Proof with only "auto" or "trivial" — too cheap
                if re.match(r"Proof\s*\.", stripped):
                    # Look at the proof body
                    proof_lines_count = 0
                    has_substantial_tactic = False
                    for j in range(i, min(i + 30, len(lines))):
                        proof_line = lines[j].strip()
                        if proof_line.startswith("Qed") or proof_line.startswith("Defined"):
                            break
                        proof_lines_count += 1
                        # Substantial tactics (not just auto/trivial/reflexivity)
                        if proof_line and not proof_line.startswith("--"):
                            if not re.match(r"^(Proof|Qed|Defined|auto|trivial|reflexivity|intros|apply|exact)\s", proof_line):
                                has_substantial_tactic = True

                    if proof_lines_count <= 2 and not has_substantial_tactic:
                        violations.append(Violation(
                            filepath=filepath,
                            line=i,
                            severity=Severity.HIGH,
                            category="PROOF_CHEAP",
                            message=(
                                f"Proof body is only {proof_lines_count} lines — "
                                f"likely trivial/not thorough. A real proof should "
                                f"contain substantial tactic steps.\n"
                                "JUSTIFICATION: Add comment above Proof explaining why "
                                "this proof is trivial (e.g., '(* Trivial: follows from X *)')."
                            ),
                            standard="DO-178C §5.2.2, ECSS-Q-ST-80C §6.3",
                            code_snippet=stripped,
                        ))

                # "admit" tactic — bypass
                if re.match(r"\badmit\b", stripped):
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.CRITICAL,
                        category="PROOF_CHEAP",
                        message=(
                            "admit tactic used — proof bypassed. "
                            "This is FRAUD. Every goal MUST be discharged.\n"
                            "JUSTIFICATION: Replace 'admit' with actual proof. "
                            "If truly impossible, add: '(* JUSTIFICATION: <reason> *)' "
                            "above and document in design records."
                        ),
                        standard="DO-178C §5.2.2, ECSS-Q-ST-80C §6.3",
                        code_snippet=stripped,
                    ))

                # "sorry" (Lean-style) — bypass
                if re.match(r"\bsorry\b", stripped):
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.CRITICAL,
                        category="PROOF_CHEAP",
                        message=(
                            "sorry used — proof bypassed. "
                            "This is FRAUD. Every goal MUST be discharged.\n"
                            "JUSTIFICATION: Replace 'sorry' with actual proof. "
                            "If truly impossible, add: '(* JUSTIFICATION: <reason> *)' "
                            "above and document in design records."
                        ),
                        standard="DO-178C §5.2.2, ECSS-Q-ST-80C §6.3",
                        code_snippet=stripped,
                    ))

                # "tauto" without justification — might be hiding issues
                if re.match(r"\btauto\b", stripped):
                    # Check if there's a comment explaining why
                    has_comment = "--" in stripped or "(" in stripped
                    if not has_comment:
                        violations.append(Violation(
                            filepath=filepath,
                            line=i,
                            severity=Severity.LOW,
                            category="PROOF_CHEAP",
                            message=(
                                "tauto used without comment — verify this is "
                                "sufficient for the proof goal.\n"
                                "JUSTIFICATION: Add comment: '(* tauto suffices because <reason> *)'"
                            ),
                            standard="DO-178C §5.2.2",
                            code_snippet=stripped,
                        ))

                # "omega" or "lia" without justification — automation
                if re.match(r"\b(omega|lia|nia)\b", stripped):
                    has_comment = "--" in stripped or "(" in stripped
                    if not has_comment:
                        violations.append(Violation(
                            filepath=filepath,
                            line=i,
                            severity=Severity.LOW,
                            category="PROOF_CHEAP",
                            message=(
                                "Automated arithmetic (omega/lia/nia) without comment — "
                                "verify the arithmetic is correctly captured.\n"
                                "JUSTIFICATION: Add comment: '(* <tactic> suffices because <reason> *)'"
                            ),
                            standard="DO-178C §5.2.2",
                            code_snippet=stripped,
                        ))

                # "firstorder" — might be too powerful
                if re.match(r"\bfirstorder\b", stripped):
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.LOW,
                        category="PROOF_CHEAP",
                        message=(
                            "firstorder used — powerful automation that might "
                            "mask proof obligations. Verify completeness.\n"
                            "JUSTIFICATION: Add comment: '(* firstorder suffices because <reason> *)'"
                        ),
                        standard="DO-178C §5.2.2",
                        code_snippet=stripped,
                    ))

                # "Search" or "Print" left in proof — debugging left in
                if re.match(r"\b(Search|Print|Check|About)\s+", stripped):
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.LOW,
                        category="PROOF_CHEAP",
                        message=(
                            "Debugging command left in proof (Search/Print/Check) — "
                            "remove before finalizing."
                        ),
                        standard="DO-178C §5.2.2",
                        code_snippet=stripped,
                    ))

                # "Admitted" in comment — suspicious
                if stripped.startswith("--") and "Admitted" in stripped:
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.MEDIUM,
                        category="PROOF_CHEAP",
                        message=(
                            "Admitted mentioned in comment — verify proof is "
                            "actually complete and not just commented out.\n"
                            "JUSTIFICATION: Remove the comment or add: "
                            "'(* Admitted was removed because <reason> *)'"
                        ),
                        standard="DO-178C §5.2.2",
                        code_snippet=stripped,
                    ))

        if is_ada or is_python or is_c:
            # ── Ada/Python/C checks: verify .v file exists ──
            from pathlib import Path as _Path

            filepath_obj = _Path(filepath)
            unit_name = filepath_obj.stem

            # Look for corresponding .v file
            src_dir = filepath_obj.parent
            proof_dirs = [
                src_dir / "proofs",
                src_dir.parent / "proofs",
                src_dir.parent.parent / "proofs",
            ]

            # Also check in the Coq verification directory
            proof_dirs.extend([
                _Path("coq_proofs"),
                _Path("src/coq_proofs"),
                _Path("proofs"),
            ])

            found_proof = False
            proof_path = ""

            for proof_dir in proof_dirs:
                for ext in ["_proof.v", ".v"]:
                    candidate = proof_dir / f"{unit_name}{ext}"
                    if candidate.exists():
                        found_proof = True
                        proof_path = str(candidate)
                        break
                if found_proof:
                    break

            if not found_proof:
                # Determine file type for message
                if is_ada:
                    file_type = "Ada/SPARK"
                elif is_python:
                    file_type = "Python"
                else:
                    file_type = "C"

                violations.append(Violation(
                    filepath=filepath,
                    line=1,
                    severity=Severity.CRITICAL,
                    category="PROOF_MISSING",
                    message=(
                        f"{file_type} unit '{unit_name}' has NO corresponding Coq .v proof file. "
                        f"Every {file_type} unit MUST have a Coq proof. "
                        f"Expected: proofs/{unit_name}_proof.v or proofs/{unit_name}.v. "
                        f"This is FRAUD — code without proof is not acceptable."
                    ),
                    standard="DO-178C §5.2.2, ECSS-Q-ST-80C §6.3",
                    code_snippet=f"unit: {unit_name}",
                ))

            # Also check if the .v file has Admitted
            if found_proof:
                try:
                    with open(proof_path, "r") as f:
                        proof_content = f.read()
                    proof_lines = proof_content.split("\n")
                    for j, pline in enumerate(proof_lines, 1):
                        if re.match(r"Admitted\s*\.", pline.strip()):
                            violations.append(Violation(
                                filepath=filepath,
                                line=1,
                                severity=Severity.CRITICAL,
                                category="PROOF_MISSING",
                                message=(
                                    f"Corresponding proof '{proof_path}' has Admitted at line {j} — "
                                    f"proof is incomplete. This is FRAUD."
                                ),
                                standard="DO-178C §5.2.2, ECSS-Q-ST-80C §6.3",
                                code_snippet=f"Admitted in {proof_path}",
                            ))
                            break
                except (IOError, OSError):
                    pass

        return violations

    return [
        Pattern(
            name="coq_proof_verification",
            category="PROOF_MISSING",
            severity=Severity.CRITICAL,
            standard="DO-178C §5.2.2, ECSS-Q-ST-80C §6.3",
            description="Coq proof verification: Admitted, Axiom, missing .v files for ALL source types",
            languages=["coq", "ada", "python", "c"],
            check_func=check_coq_proofs,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════
# BEHAVIORAL CHANGE DETECTION PATTERNS
# ══════════════════════════════════════════════════════════════════════════

def _build_behavioral_change_patterns() -> list[Pattern]:
    """Detect unauthorized behavioral changes in existing code.

    These patterns indicate code that changes existing behavior without
    documentation — a common sabotage vector.
    """
    def check_behavioral_changes(source: str, lines: list[str], filepath: str = "") -> list[Violation]:
        violations = []

        # ── Pattern 1: Modified function signatures ──
        # This is checked via git diff integration in run.py, not here

        # ── Pattern 2: Changed return values ──
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # return True → return False (or vice versa) — behavioral change
            if re.match(r"return\s+(True|False)\s*$", stripped):
                # Check if there's a comment explaining why
                has_explanation = False
                # Check same line after return
                if "--" in stripped or "#" in stripped:
                    has_explanation = True
                # Check previous line
                if i > 1:
                    prev_line = lines[i - 2].strip()
                    if prev_line.startswith("--") or prev_line.startswith("#"):
                        if len(prev_line) > 5:
                            has_explanation = True

                # Don't flag legitimate returns, only suspicious ones
                # Skip if this is in a test file or has explanation
                if "test" in filepath.lower() or has_explanation:
                    continue

                # Check if this function had a different return value historically
                # (This would require git history integration — done in run.py)

        # ── Pattern 3: Modified error handling behavior ──
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # raise → pass (or return None) — swallowing errors
            if re.match(r"except\s+\w+", stripped):
                # Check if the handler re-raises or swallows
                for j in range(i, min(i + 5, len(lines))):
                    handler_line = lines[j].strip()
                    if handler_line == "pass" or handler_line == "...":
                        # Check if there was previously a raise here
                        # (This would require git history — done in run.py)
                        break

        # ── Pattern 4: Changed constant values ──
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Match: CONSTANT = value (module-level)
            if re.match(r"^[A-Z_]+\s*=\s*\d+", stripped):
                # This is a constant — changing it may affect behavior
                # Check if there's a comment explaining the change
                has_explanation = False
                if "--" in stripped or "#" in stripped:
                    has_explanation = True
                if i > 1:
                    prev_line = lines[i - 2].strip()
                    if prev_line.startswith("--") or prev_line.startswith("#"):
                        if len(prev_line) > 10:
                            has_explanation = True

                if not has_explanation:
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.MEDIUM,
                        category="BEHAVIORAL_CHANGE",
                        message=(
                            "Constant value modification without explanation — "
                            "add comment explaining why this value changed."
                        ),
                        standard="DO-178C §6.3.2, ECSS-Q-ST-80C §7.4",
                        code_snippet=stripped,
                    ))

        return violations

    return [
        Pattern(
            name="behavioral_change",
            category="BEHAVIORAL_CHANGE",
            severity=Severity.HIGH,
            standard="DO-178C §6.3.2, ECSS-Q-ST-80C §7.4",
            description="Unauthorized behavioral changes without documentation",
            languages=["python"],
            check_func=check_behavioral_changes,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════
# INTEGRATION CONTRACT VALIDATION PATTERNS
# ══════════════════════════════════════════════════════════════════════════

def _build_integration_contract_patterns() -> list[Pattern]:
    """Detect broken integration contracts between modules.

    When function signatures change, callers must be updated. If not,
    the integration is broken — a common sabotage vector.
    """
    def check_contracts(source: str, lines: list[str], filepath: str = "") -> list[Violation]:
        violations = []

        # ── Pattern 1: Function with too many parameters (likely changed signature) ──
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Count parameters
                    param_count = len(node.args.args)
                    if param_count > 10:
                        violations.append(Violation(
                            filepath=filepath,
                            line=node.lineno,
                            severity=Severity.MEDIUM,
                            category="INTEGRATION_CONTRACT",
                            message=(
                                f"Function '{node.name}' has {param_count} parameters — "
                                f"consider if all are necessary. Large parameter lists "
                                f"indicate tight coupling or signature bloat."
                            ),
                            standard="CWE-697: Incorrect Comparison",
                            code_snippet=f"def {node.name}(..., {param_count} params)",
                        ))

                    # Check for **kwargs or *args usage (flexible signature)
                    has_kwargs = node.args.kwarg is not None

                    if has_kwargs:
                        violations.append(Violation(
                            filepath=filepath,
                            line=node.lineno,
                            severity=Severity.LOW,
                            category="INTEGRATION_CONTRACT",
                            message=(
                                f"Function '{node.name}' uses **kwargs — "
                                f"flexible signatures can hide integration issues."
                            ),
                            standard="CWE-697: Incorrect Comparison",
                            code_snippet=f"def {node.name}(..., **kwargs)",
                        ))

        except SyntaxError:
            pass

        # ── Pattern 2: Import without corresponding usage ──
        imports = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Match: from module import name
            import_match = re.match(r"from\s+\S+\s+import\s+(.+)", stripped)
            if import_match:
                imported = import_match.group(1).strip()
                imports.append((i, imported))

            # Match: import module
            simple_import = re.match(r"import\s+(\S+)", stripped)
            if simple_import:
                imports.append((i, simple_import.group(1)))

        # Check if each import is used
        for line_no, imported in imports:
            # Extract the name (handle "import a, b, c" and "from x import a as b")
            for name in imported.split(","):
                name = name.strip()
                if " as " in name:
                    name = name.split(" as ")[1].strip()
                if name == "*":
                    continue

                # Check if name is used anywhere in the file
                is_used = False
                for i, line in enumerate(lines, 1):
                    if i == line_no:
                        continue
                    if re.search(rf"\b{re.escape(name)}\b", line):
                        is_used = True
                        break

                if not is_used:
                    violations.append(Violation(
                        filepath=filepath,
                        line=line_no,
                        severity=Severity.LOW,
                        category="INTEGRATION_CONTRACT",
                        message=(
                            f"'{name}' imported but never used — "
                            f"possible broken integration or dead code."
                        ),
                        standard="MISRA C:2012 Rule 2.5",
                        code_snippet=f"import {name}",
                    ))

        return violations

    return [
        Pattern(
            name="integration_contract",
            category="INTEGRATION_CONTRACT",
            severity=Severity.MEDIUM,
            standard="CWE-697, MISRA C:2012 Rule 2.5",
            description="Broken integration contracts, signature bloat, unused imports",
            languages=["python"],
            check_func=check_contracts,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════
# REGRESSION REVERSION DETECTION PATTERNS
# ══════════════════════════════════════════════════════════════════════════

def _build_regression_reversion_patterns() -> list[Pattern]:
    """Detect when previous fixes are reverted.

    This pattern checks for known anti-patterns that were previously fixed
    but may have been reintroduced.
    """
    def check_regressions(source: str, lines: list[str], filepath: str = "") -> list[Violation]:
        violations = []

        # ── Known anti-patterns that were previously fixed ──
        KNOWN_ANTI_PATTERNS = [
            # (pattern, description, standard)
            (r"subprocess\.run\(\s*force_kill_process\(", "subprocess.run(force_kill_process())", "CWE-628"),
            (r"except\s*:\s*$", "bare except without type", "CERT ERR00-C"),
            (r"open\([^)]*\)\s*$", "open() without context manager", "CWE-775"),
            (r"os\.system\(", "os.system() usage", "CWE-78"),
            (r"eval\(", "eval() usage", "CWE-95"),
            (r"exec\(", "exec() usage", "CWE-95"),
            (r"pickle\.loads\(", "pickle.loads() usage", "CWE-502"),
            (r"yaml\.load\((?!.*Loader)", "yaml.load() without Loader", "CWE-502"),
            (r"subprocess\.call\(", "subprocess.call() — use run() instead", "CWE-628"),
        ]

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            for pattern, desc, standard in KNOWN_ANTI_PATTERNS:
                if re.search(pattern, stripped):
                    violations.append(Violation(
                        filepath=filepath,
                        line=i,
                        severity=Severity.HIGH,
                        category="REGRESSION_REVERSION",
                        message=(
                            f"Previously fixed anti-pattern reintroduced: {desc} — "
                            f"this was fixed before, do not revert."
                        ),
                        standard=f"{standard}: Regression detected",
                        code_snippet=stripped,
                    ))

        return violations

    return [
        Pattern(
            name="regression_reversion",
            category="REGRESSION_REVERSION",
            severity=Severity.HIGH,
            standard="CWE-628, CWE-78, CWE-95, CWE-502",
            description="Regression detection: previously fixed anti-patterns reintroduced",
            languages=["python"],
            check_func=check_regressions,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════
# ADA/SPARK PATTERNS
# ══════════════════════════════════════════════════════════════════════════

def _build_ada_spark_off_patterns() -> list[Pattern]:
    """
    Detect SPARK_Mode(Off) and classify its intent.

    SPARK_Mode(Off) is sometimes legitimately needed for:
    - Multithreading (tasks, protected types, reentrancy)
    - FFI / Interface.C / C interop
    - Volatile/atomic operations
    - Import/Export of foreign subprograms

    But SPARK_Mode(Off) is SABOTAGE when:
    - No justification comment
    - Used in security-critical code paths without documented rationale
    - Followed by suspicious patterns (unchecked conversions, pointer ops, magic numbers)
    """
    # Legitimate justification keywords (lowercase)
    LEGITIMATE_JUSTIFICATIONS = [
        "multithreading", "thread", "task", "protected", "reentrant",
        "interface.c", "ffi", "foreign", "import", "export",
        "volatile", "atomic", "memory_model", "pragma import",
        "c_binding", "c_interface", "interop", "extern",
        "low_level", "hardware", "register", "memory_mapped",
        "aspect", "linker", "calling_convention",
    ]

    # Weak justification keywords
    WEAK_JUSTIFICATIONS = [
        "performance", "optimization", "speed", "inline",
        "style", "convenience", "compatibility", "legacy",
        "refactor", "temporary", "workaround",
    ]

    # Suspicious code patterns that should NOT appear after SPARK_Mode(Off)
    # unless explicitly justified
    SUSPICIOUS_PATTERNS = [
        (re.compile(r"Unchecked_Conversion", re.IGNORECASE), "Unchecked_Conversion"),
        (re.compile(r"System\.Address", re.IGNORECASE), "System.Address"),
        (re.compile(r"\.all\s*:=", re.IGNORECASE), "Unchecked dereference assignment"),
        (re.compile(r"Address\s*=>", re.IGNORECASE), "Address clause"),
        (re.compile(r"pragma\s+Import", re.IGNORECASE), "Pragma Import"),
    ]

    # ── Hints for valid justifications ──
    JUSTIFICATION_HINTS = """
VALID JUSTIFICATION KEYWORDS (use one or more in your comment):
  C/C++ Bridging:    interface.c, c_binding, c_interface, pragma_import, extern
  Python Bridging:   python_binding, pyinterface, ctypes, cpython, pycffi
  Shared Memory:     shm, shared_memory, mmap, memory_mapped, System.Address
  Assembly/Inline:    asm, inline_asm, machine_code, register, low_level
  Multithreading:    thread, task, protected, reentrant, mutex, semaphore
  Hardware Access:    hardware, register, memory_mapped, volatile, atomic
  FFI/Foreign:       ffi, foreign, extern, calling_convention, linker
  Performance:       performance, optimization (WEAK — must document why SPARK can't work)

EXAMPLES OF VALID JUSTIFICATIONS:
  SPARK_Mode(Off) -- c_binding: Interface.C.int for socket() FFI call
  SPARK_Mode(Off) -- shm: shared memory access via System.Address
  SPARK_Mode(Off) -- thread: protected type for reentrant task entry
  SPARK_Mode(Off) -- hardware: memory-mapped register at 0x40000000
  SPARK_Mode(Off) -- python_binding: CPython API PyObject* handling

AUDIT ENFORCEMENT (what the verifier checks):
  - If justification keywords match C/Python/ASM/SHM → LOW severity (expected)
  - If justification is vague/missing → HIGH severity (must document)
  - If SPARK_Mode(Off) hides suspicious code (Unchecked_Conversion, etc.) → CRITICAL
  - Auditor will verify the Off scope is LIMITED to the justified section only
""".strip()

    def check_spark_off(source: str, lines: list[str], filepath: str = "") -> list[Violation]:
        violations = []

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Skip comments
            if stripped.startswith("--"):
                continue

            # Match SPARK_Mode(Off) or pragma SPARK_Mode(Off)
            is_spark_off = re.search(r"SPARK_Mode\s*\(\s*Off\s*\)", stripped, re.IGNORECASE)
            if not is_spark_off:
                continue

            # ── Step 1: Extract justification comment ──
            has_justification = False
            justification_line = ""
            justification_type = "NONE"  # NONE, WEAK, LEGITIMATE

            # Check same line after the pragma
            same_line_match = re.search(
                r"SPARK_Mode\s*\(\s*Off\s*\)\s*--\s*(.+)", stripped, re.IGNORECASE
            )
            if same_line_match:
                has_justification = True
                justification_line = same_line_match.group(1).strip()

            # Check next line for justification comment
            if not has_justification and i < len(lines):
                next_line = lines[i].strip()
                if next_line.startswith("--") and len(next_line) > 5:
                    has_justification = True
                    justification_line = next_line[2:].strip()

            # Classify justification
            if has_justification:
                just_lower = justification_line.lower()
                if any(kw in just_lower for kw in LEGITIMATE_JUSTIFICATIONS):
                    justification_type = "LEGITIMATE"
                elif any(kw in just_lower for kw in WEAK_JUSTIFICATIONS):
                    justification_type = "WEAK"
                else:
                    justification_type = "WEAK"  # Has comment but unclear

            # ── Step 2: Check for suspicious code in following 20 lines ──
            suspicious_following = []
            search_end = min(i + 20, len(lines))
            for j in range(i, search_end):
                check_line = lines[j]
                for pattern, desc in SUSPICIOUS_PATTERNS:
                    if pattern.search(check_line):
                        suspicious_following.append((j + 1, desc))

            # ── Step 3: Determine severity with hints ──
            if justification_type == "LEGITIMATE" and not suspicious_following:
                # Legitimate justification, no suspicious code following
                severity = Severity.LOW
                message = (
                    f"SPARK_Mode(Off) justified: \"{justification_line}\" "
                    f"— legitimate use case detected"
                )
            elif justification_type == "LEGITIMATE" and suspicious_following:
                # Legitimate but suspicious code follows — verify scope
                sus_lines = ", ".join(f"L{line_num}" for line_num, _ in suspicious_following[:3])
                severity = Severity.MEDIUM
                message = (
                    f"SPARK_Mode(Off) justified: \"{justification_line}\" "
                    f"— but suspicious code follows at {sus_lines}. "
                    f"Verify Off scope is limited to justified section only.\n"
                    f"{JUSTIFICATION_HINTS}"
                )
            elif justification_type == "WEAK":
                # Weak justification
                severity = Severity.MEDIUM
                message = (
                    f"SPARK_Mode(Off) with weak justification: \"{justification_line}\" "
                    f"— verify this is truly necessary or if SPARK-compatible alternative exists.\n"
                    f"{JUSTIFICATION_HINTS}"
                )
            else:
                # No justification at all
                if suspicious_following:
                    # No justification AND suspicious code — this is sabotage
                    sus_lines = ", ".join(f"L{line_num}" for line_num, _ in suspicious_following[:3])
                    severity = Severity.CRITICAL
                    message = (
                        f"SPARK_Mode(Off) WITHOUT justification, followed by "
                        f"suspicious code at {sus_lines} — formal verification "
                        f"disabled with no documented rationale in security-critical path. "
                        f"This is not acceptable in DO-178C/ECSS compliance.\n"
                        f"{JUSTIFICATION_HINTS}"
                    )
                else:
                    # No justification but no obvious suspicious code — STILL FRAUD
                    severity = Severity.CRITICAL
                    message = (
                        "SPARK_Mode(Off) WITHOUT justification — "
                        "FRAUD: formal verification disabled with no documented rationale. "
                        "This is not a bug, this is sabotage. "
                        "Every SPARK_Mode(Off) MUST have a justification comment. "
                        "No exceptions. No excuses.\n"
                        f"{JUSTIFICATION_HINTS}"
                    )

            violations.append(Violation(
                filepath=filepath,
                line=i,
                severity=severity,
                category="SPARK_MODE_OFF",
                message=message,
                standard="DO-178C §5.2.2, ECSS-Q-ST-80C §6.3, SPARK User Guide §6.1",
                code_snippet=stripped,
            ))

        return violations

    return [
        Pattern(
            name="spark_mode_off",
            category="SPARK_MODE_OFF",
            severity=Severity.HIGH,
            standard="DO-178C §5.2.2, ECSS-Q-ST-80C §6.3",
            description="SPARK_Mode(Off) — context-aware: justified vs sabotage",
            languages=["ada"],
            check_func=check_spark_off,
        ),
    ]


def _build_ada_sabotage_patterns() -> list[Pattern]:
    """Detect Ada-specific sabotage patterns."""
    return [
        Pattern(
            name="ada_unchecked_conversion",
            category="ADA_TYPE_SAFETY",
            severity=Severity.HIGH,
            standard="Ada RM 13.9, DO-178C §5.2.3",
            description="Unchecked_Conversion bypasses type safety",
            languages=["ada"],
            regex=re.compile(r"Unchecked_Conversion", re.IGNORECASE),
            guard_patterns=[
                r"--\s*justified",
                r"--\s*approved",
                r"--\s*see.*design",
            ],
            message_template="Unchecked_Conversion bypasses type safety: {snippet} — requires justification",
        ),
        Pattern(
            name="ada_unchecked_deallocation",
            category="ADA_TYPE_SAFETY",
            severity=Severity.MEDIUM,
            standard="Ada RM 13.11.2",
            description="Unchecked_Deallocation can cause dangling pointers",
            languages=["ada"],
            regex=re.compile(r"Unchecked_Deallocation", re.IGNORECASE),
            guard_patterns=[
                r"--\s*justified",
                r"--\s*approved",
            ],
            message_template="Unchecked_Deallocation: {snippet} — ensure no dangling pointers",
        ),
        Pattern(
            name="ada_system_address_cast",
            category="ADA_TYPE_SAFETY",
            severity=Severity.HIGH,
            standard="Ada RM 13.7.2, ECSS-Q-ST-80C §6.3",
            description="System.Address usage bypasses type system",
            languages=["ada"],
            regex=re.compile(r"System\.Address", re.IGNORECASE),
            guard_patterns=[
                r"--\s*justified",
                r"--\s*FFI",
                r"--\s*interop",
                r"C_Binding",
            ],
            message_template="System.Address usage: {snippet} — type safety bypass, ensure FFI justification",
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════
# C PATTERNS
# ══════════════════════════════════════════════════════════════════════════

def _build_c_sabotage_patterns() -> list[Pattern]:
    """Detect C-specific sabotage patterns."""
    return [
        Pattern(
            name="c_banned_function_sprintf",
            category="C_BUFFER_OVERFLOW",
            severity=Severity.HIGH,
            standard="CERT STR31-C, CWE-120",
            description="sprintf() has no buffer size checking — use snprintf()",
            languages=["c"],
            regex=re.compile(r"\bsprintf\s*\("),
            guard_patterns=[
                r"snprintf",
                r"//\s*safe",
            ],
            message_template="sprintf() has no bounds checking: {snippet} — use snprintf() instead",
        ),
        Pattern(
            name="c_banned_function_gets",
            category="C_BUFFER_OVERFLOW",
            severity=Severity.CRITICAL,
            standard="CERT STR31-C, CWE-120",
            description="gets() is always unsafe — removed in C11",
            languages=["c"],
            regex=re.compile(r"\bgets\s*\("),
            message_template="gets() is always unsafe: {snippet} — use fgets() instead",
        ),
        Pattern(
            name="c_banned_function_strcpy",
            category="C_BUFFER_OVERFLOW",
            severity=Severity.MEDIUM,
            standard="CERT STR31-C, CWE-120",
            description="strcpy() has no buffer size checking",
            languages=["c"],
            regex=re.compile(r"\bstrcpy\s*\("),
            guard_patterns=[
                r"strncpy",
                r"strlcpy",
                r"//\s*safe",
            ],
            message_template="strcpy() has no bounds checking: {snippet} — consider strncpy()/strlcpy()",
        ),
        Pattern(
            name="c_banned_function_strcat",
            category="C_BUFFER_OVERFLOW",
            severity=Severity.MEDIUM,
            standard="CERT STR31-C, CWE-120",
            description="strcat() has no buffer size checking",
            languages=["c"],
            regex=re.compile(r"\bstrcat\s*\("),
            guard_patterns=[
                r"strncat",
                r"strlcat",
                r"//\s*safe",
            ],
            message_template="strcat() has no bounds checking: {snippet} — consider strncat()/strlcat()",
        ),
        Pattern(
            name="c_banned_functionscanf",
            category="C_FORMAT_STRING",
            severity=Severity.MEDIUM,
            standard="CERT FLP34-C, CWE-134",
            description="scanf() without field width limit",
            languages=["c"],
            regex=re.compile(r"\bscanf\s*\(\s*\"[^\"]*%[^\"]*\""),
            guard_patterns=[
                r"%\*\.",
                r"//\s*safe",
                r"//\s*bounded",
            ],
            message_template="scanf() without field width: {snippet} — use %Ns format specifier",
        ),
        Pattern(
            name="c_malloc_no_check",
            category="C_NULL_DEREFERENCE",
            severity=Severity.HIGH,
            standard="CERT MEM32-C, CWE-476",
            description="malloc() result not checked for NULL",
            languages=["c"],
            regex=re.compile(r"\bmalloc\s*\("),
            guard_patterns=[
                r"if\s*\(",
                r"!=\s*NULL",
                r"//\s*checked",
            ],
            message_template="malloc() without NULL check: {snippet} — memory exhaustion causes NULL deref",
        ),
        Pattern(
            name="c_void_pointer_arithmetic",
            category="C_TYPE_SAFETY",
            severity=Severity.MEDIUM,
            standard="CERT EXP39-C",
            description="Pointer arithmetic on void* is a GCC extension, not standard C",
            languages=["c"],
            regex=re.compile(r"\(\s*void\s*\*\s*\)\s*\w+\s*\+"),
            message_template="void* pointer arithmetic (GCC extension): {snippet} — not standard C",
        ),
        Pattern(
            name="c_magic_number",
            category="C_MAINTAINABILITY",
            severity=Severity.LOW,
            standard="CERT DCL00-C, MISRA C:2012 Rule 8.9",
            description="Magic number in code — should be a named constant",
            languages=["c"],
            regex=re.compile(r"[=<>+\-*/]\s*\d{2,}(?![.\d])"),
            guard_patterns=[
                r"#define",
                r"enum",
                r"const\s+",
                r"//\s*index",
                r"//\s*offset",
                r"//\s*size",
                r"0x[0-9a-fA-F]+",  # hex constants are usually intentional
            ],
            message_template="Magic number: {match} — define as named constant",
        ),
        Pattern(
            name="c_missing_free",
            category="C_MEMORY_LEAK",
            severity=Severity.MEDIUM,
            standard="CERT MEM31-C, CWE-401",
            description="malloc/calloc without corresponding free in same function (heuristic)",
            languages=["c"],
            check_func=lambda src, lines, fp: _check_c_missing_free(src, lines, fp),
        ),
    ]


def _check_c_missing_free(source: str, lines: list[str], filepath: str = "") -> list[Violation]:
    """Heuristic: detect malloc/calloc without free in the same function."""
    violations = []

    # Track function boundaries and allocations
    in_func = False
    func_name = ""
    alloc_lines = []
    free_found = False

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Detect function start (simplified: looks for { after function signature)
        if re.match(r"\w+.*\(", stripped) and "{" in stripped:
            func_match = re.match(r"(?:static\s+|extern\s+)*(?:\w+\s+)+(\w+)\s*\(", stripped)
            if func_match:
                func_name = func_match.group(1)
                in_func = True
                alloc_lines = []
                free_found = False

        if in_func:
            if re.search(r"\b(malloc|calloc|realloc)\s*\(", stripped):
                alloc_lines.append(i)
            if re.search(r"\bfree\s*\(", stripped):
                free_found = True

        # Simple heuristic: if we see a closing brace at column 0, end of function
        if in_func and stripped == "}":
            if alloc_lines and not free_found:
                for alloc_line in alloc_lines:
                    violations.append(Violation(
                        filepath=filepath,
                        line=alloc_line,
                        severity=Severity.MEDIUM,
                        category="C_MEMORY_LEAK",
                        message=(
                            f"Memory allocation in function '{func_name}' without "
                            f"corresponding free() — potential memory leak"
                        ),
                        standard="CERT MEM31-C, CWE-401",
                        code_snippet=f"malloc/calloc at line {alloc_line}",
                    ))
            in_func = False
            func_name = ""
            alloc_lines = []
            free_found = False

    return violations


# ══════════════════════════════════════════════════════════════════════════
# DEFAULT REGISTRY
# ══════════════════════════════════════════════════════════════════════════

def create_default_registry() -> PatternRegistry:
    """
    Create a PatternRegistry with all built-in sabotage patterns.

    This is the adaptive part: new patterns can be registered at any time
    by calling registry.register() or registry.register_all().
    """
    registry = PatternRegistry()

    # Python patterns
    registry.register_all(_build_python_platform_hardcoding_patterns())
    registry.register_all(_build_python_silent_failure_patterns())
    registry.register_all(_build_python_copy_paste_patterns())
    registry.register_all(_build_python_stale_reference_patterns())
    registry.register_all(_build_python_dead_code_patterns())
    registry.register_all(_build_python_resource_leak_patterns())
    registry.register_all(_build_python_softlock_patterns())
    registry.register_all(_build_python_redundant_logic_patterns())
    registry.register_all(_build_python_exception_patterns())
    registry.register_all(_build_python_stale_flag_patterns())

    # Coq proof patterns (applies to ALL source types)
    registry.register_all(_build_coq_proof_patterns())

    # Behavioral & integration patterns
    registry.register_all(_build_behavioral_change_patterns())
    registry.register_all(_build_integration_contract_patterns())
    registry.register_all(_build_regression_reversion_patterns())

    # Ada/SPARK patterns
    registry.register_all(_build_ada_spark_off_patterns())
    registry.register_all(_build_ada_sabotage_patterns())

    # C patterns
    registry.register_all(_build_c_sabotage_patterns())

    return registry


# ══════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════

def detect_language(filepath: str) -> str:
    """Detect file language from extension."""
    ext = Path(filepath).suffix.lower()
    lang_map = {
        ".py": "python",
        ".adb": "ada",
        ".ads": "ada",
        ".c": "c",
        ".h": "c",
        ".cpp": "c",
        ".cc": "c",
        ".cxx": "c",
        ".hpp": "c",
    }
    return lang_map.get(ext, "python")  # Default to python for unknown extensions


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════

def run_sabotage_audit(
    filepath: str,
    registry: PatternRegistry | None = None,
    severity_filter: Severity | None = None,
) -> list[Violation]:
    """
    Run sabotage audit against a single source file.

    Args:
        filepath: Path to the source file to audit
        registry: Optional custom registry (uses default if None)
        severity_filter: Optional minimum severity to report

    Returns:
        List of violations found, sorted by severity then line number
    """
    if registry is None:
        registry = create_default_registry()

    source = Path(filepath).read_text(encoding="utf-8")
    language = detect_language(filepath)
    verifier = SabotageVerifier(registry)
    violations = verifier.verify(source, filepath=filepath, language=language)

    return _filter_and_sort(violations, severity_filter)


def audit_directory(
    dirpath: str,
    extensions: list[str] | None = None,
    registry: PatternRegistry | None = None,
    severity_filter: Severity | None = None,
    exclude_dirs: list[str] | None = None,
    exclude_files: list[str] | None = None,
) -> list[Violation]:
    """
    Run sabotage audit against all matching files in a directory.

    Args:
        dirpath: Path to the directory to audit
        extensions: File extensions to include (e.g., [".py", ".c", ".adb"])
        registry: Optional custom registry (uses default if None)
        severity_filter: Optional minimum severity to report
        exclude_dirs: Directory names to exclude (default: vendor, node_modules, .git)
        exclude_files: File paths to exclude (e.g., sabotage_verifier.py itself)

    Returns:
        List of all violations found across all files, sorted by severity then filepath
    """
    if registry is None:
        registry = create_default_registry()
    if extensions is None:
        extensions = [".py", ".c", ".h", ".adb", ".ads"]
    if exclude_dirs is None:
        exclude_dirs = ["vendor", "node_modules", ".git", "__pycache__", "obj", "build"]
    if exclude_files is None:
        exclude_files = []

    all_violations = []
    dir_path = Path(dirpath)

    for root, dirs, files in os.walk(dir_path):
        # Exclude directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for filename in files:
            filepath = Path(root) / filename
            if filepath.suffix.lower() in extensions:
                # Skip excluded files (e.g., sabotage_verifier.py auditing itself)
                if str(filepath) in exclude_files or filename in exclude_files:
                    continue
                try:
                    violations = run_sabotage_audit(
                        str(filepath),
                        registry=registry,
                        severity_filter=severity_filter,
                    )
                    all_violations.extend(violations)
                except (UnicodeDecodeError, PermissionError, OSError) as e:
                    # Skip files that can't be read
                    print(f"  [!] Skipping {filepath}: {e}")

    return _filter_and_sort(all_violations, severity_filter)


def _filter_and_sort(
    violations: list[Violation],
    severity_filter: Severity | None,
) -> list[Violation]:
    """Filter by severity and sort violations."""
    if severity_filter:
        severity_order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
        min_idx = severity_order.index(severity_filter)
        violations = [v for v in violations if severity_order.index(v.severity) <= min_idx]

    severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
    violations.sort(key=lambda v: (severity_order[v.severity], v.filepath, v.line))

    return violations


def format_report(violations: list[Violation], target: str = "") -> str:
    """Format violations into a human-readable report."""
    lines = []

    critical = [v for v in violations if v.severity == Severity.CRITICAL]
    high = [v for v in violations if v.severity == Severity.HIGH]
    medium = [v for v in violations if v.severity == Severity.MEDIUM]
    low = [v for v in violations if v.severity == Severity.LOW]

    lines.append(f"\n{'='*70}")
    lines.append(f" SABOTAGE AUDIT: {target}")
    lines.append(f"{'='*70}")
    lines.append(
        f" CRITICAL: {len(critical)}  HIGH: {len(high)}  "
        f"MEDIUM: {len(medium)}  LOW: {len(low)}"
    )
    lines.append(f"{'='*70}\n")

    current_file = ""
    for v in violations:
        # Show file header when file changes
        if v.filepath != current_file:
            current_file = v.filepath
            lines.append(f"  --- {current_file} ---")

        lines.append(f"  [{v.severity.value}] L{v.line:4d}: {v.category}")
        lines.append(f"           {v.message}")
        if v.standard:
            lines.append(f"           Standard: {v.standard}")
        lines.append("")

    if critical:
        lines.append(f"\n{'='*70}")
        lines.append(f" VERDICT: TAINTED — {len(critical)} CRITICAL violations found")
        lines.append(f"{'='*70}")
    else:
        lines.append(f"\n{'='*70}")
        lines.append(" VERDICT: CLEAN — No critical violations")
        lines.append(f"{'='*70}")

    return "\n".join(lines)


def format_json(violations: list[Violation]) -> str:
    """Format violations as JSON for CI/CD integration."""
    data = []
    for v in violations:
        data.append({
            "filepath": v.filepath,
            "line": v.line,
            "severity": v.severity.value,
            "category": v.category,
            "message": v.message,
            "standard": v.standard,
            "code_snippet": v.code_snippet,
        })
    return json.dumps(data, indent=2)


# ── CLI Entry Point ──────────────────────────────────────────────────────

def main():
    """CLI entry point for standalone sabotage audit."""
    if len(sys.argv) < 2:
        print("Usage: python sabotage_verifier.py <file_or_dir> [options]")
        print()
        print("Options:")
        print("  --severity LEVEL      Minimum severity (CRITICAL, HIGH, MEDIUM, LOW)")
        print("  --extensions EXTS     Comma-separated extensions (for directories)")
        print("  --json                Output as JSON")
        print("  --exclude DIRS        Comma-separated directory names to exclude")
        print("  --exclude-files FILES Comma-separated filenames to exclude")
        print()
        print("Examples:")
        print("  python sabotage_verifier.py run.py")
        print("  python sabotage_verifier.py src/python/ --extensions .py")
        print("  python sabotage_verifier.py src/ --extensions .adb,.ads,.c,.h")
        print("  python sabotage_verifier.py src/ --exclude-files sabotage_verifier.py")
        print("  python sabotage_verifier.py run.py --severity CRITICAL --json")
        sys.exit(1)

    target = sys.argv[1]
    severity_filter = None
    json_output = False
    extensions = None
    exclude_dirs = None
    exclude_files = None

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--json":
            json_output = True
        elif args[i] == "--severity" and i + 1 < len(args):
            severity_filter = Severity(args[i + 1].upper())
            i += 1
        elif args[i] == "--extensions" and i + 1 < len(args):
            extensions = [ext.strip() if ext.startswith(".") else f".{ext.strip()}" for ext in args[i + 1].split(",")]
            i += 1
        elif args[i] == "--exclude" and i + 1 < len(args):
            exclude_dirs = [d.strip() for d in args[i + 1].split(",")]
            i += 1
        elif args[i] == "--exclude-files" and i + 1 < len(args):
            exclude_files = [f.strip() for f in args[i + 1].split(",")]
            i += 1
        i += 1

    target_path = Path(target)

    if target_path.is_dir():
        violations = audit_directory(
            target,
            extensions=extensions,
            severity_filter=severity_filter,
            exclude_dirs=exclude_dirs,
            exclude_files=exclude_files,
        )
    else:
        violations = run_sabotage_audit(target, severity_filter=severity_filter)

    if json_output:
        print(format_json(violations))
    else:
        print(format_report(violations, target))

    # Exit with error if critical violations found
    if any(v.severity == Severity.CRITICAL for v in violations):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
