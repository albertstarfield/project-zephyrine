#!/usr/bin/env python3
"""
Trace Utility Module — Standardized verbosity for all Adelaide tool scripts.

Provides a `trace_print()` function used by every Python tool to emit
consistent `[prefix][Toolcall][+uptime]` format messages.

Usage:
    from trace_utils import trace_print, init_trace
    init_trace()                              # one call at script start
    trace_print("searchglobalref", "phase1", "Dispatching Deno scraper...")
    trace_print("searchglobalref", "phase2:embed", f"Ranking {n} results")

Output format:
    [ADA][Toolcall][+120] dispatch:search("quantum computing")
    [PY][Toolcall][+120] searchglobalref:phase1:Dispatching Deno scraper...

Env vars:
    ADELAIDE_TOOL_TRACE_PREFIX   — trace prefix (default: "[PY]")
    ADELAIDE_TOOL_TRACE_ENABLED  — set to "0" to suppress trace prints
"""

import os
import sys
import time

_START_TIME: float = 0.0
_PREFIX: str = "[PY]"
_ENABLED: bool = True


def init_trace(prefix: str | None = None) -> None:
    """Initialize the trace module.  Call once at script start."""
    global _START_TIME, _PREFIX, _ENABLED
    _START_TIME = time.monotonic()
    _PREFIX = prefix or os.environ.get("ADELAIDE_TOOL_TRACE_PREFIX", "[PY]")
    _ENABLED = os.environ.get("ADELAIDE_TOOL_TRACE_ENABLED", "1") != "0"


def _uptime() -> int:
    """Return whole seconds since init_trace() was called."""
    return int(time.monotonic() - _START_TIME)


def trace_print(toolcall: str, step: str = "", message: str = "",
                file=sys.stderr) -> None:
    """Emit a [prefix][Toolcall][+uptime] trace line to *file* (stderr).

    Args:
        toolcall: Tool name, e.g. "searchglobalref", "git", "package".
        step:     Optional sub-step, e.g. "phase1", "extract", "embed".
        message:  Free-form human-readable description.
        file:     Output stream (default stderr — keeps stdout clean for data).
    """
    if not _ENABLED:
        return

    uptime = _uptime()
    label = toolcall
    if step:
        label += f":{step}"

    # Sanitize message — collapse whitespace, truncate to 200 chars
    msg = " ".join(message.split())[:200]
    if msg:
        print(f"{_PREFIX}[Toolcall][+{uptime}] {label}: {msg}", file=file, flush=True)
    else:
        print(f"{_PREFIX}[Toolcall][+{uptime}] {label}", file=file, flush=True)


def trace_result(toolcall: str, success: bool, detail: str = "",
                 file=sys.stderr) -> None:
    """Trace the final result of a tool invocation."""
    status = "OK" if success else "FAIL"
    uptime = _uptime()
    msg = f"{status}"
    if detail:
        msg += f" — {detail}"
    print(f"{_PREFIX}[Toolcall][+{uptime}] {toolcall}:{msg}", file=file, flush=True)
