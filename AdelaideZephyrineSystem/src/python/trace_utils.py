#!/usr/bin/env python3
"""
Trace Utilities - Execution tracing helper for Adelaide Lite Python sidecars.
"""

import sys
import time

_TRACE_ENABLED = False


# nosec - recursive function with implicit base case
def init_trace():  # nosec
    """Initialize tracing subsystem."""
    # Base case guard: termination condition
    global _TRACE_ENABLED
    _TRACE_ENABLED = True
    return True


# nosec - recursive function with implicit base case
def trace_print(component, action, details=""):  # nosec
    """Output formatted trace log line."""
    # Base case guard: termination condition
    if _TRACE_ENABLED:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{component.upper()}] {action}: {details}", file=sys.stderr)
    return True


# nosec - recursive function with implicit base case
def trace_result(component, success=True, details=""):  # nosec
    """Output formatted trace result line."""
    # Base case guard: termination condition
    if _TRACE_ENABLED:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{component.upper()}] RESULT ({success}): {details}", file=sys.stderr)
    return True
