# KNOWN GOOD Python: Every function here should NOT trigger SMT violations.
# All dangerous operations are properly guarded.
# If the verifier flags any of these, it's a FALSE POSITIVE.

def divide_safe(a: int, b: int) -> float:
    """CHECK 1: Division by zero — GUARDED."""
    if b != 0:
        return a / b
    return 0.0


def index_safe(data: list, idx: int) -> int:
    """CHECK 2: Index out of bounds — GUARDED."""
    if 0 <= idx < len(data):
        return data[idx]
    return 0


def none_safe(value: str | None) -> int:
    """CHECK 3: Null dereference — GUARDED."""
    if value is not None:
        return len(value)
    return 0


def type_safe(x) -> str:
    """CHECK 4: Type contradiction — consistent checks."""
    if isinstance(x, int):
        return str(x)
    return "unknown"


def overflow_safe(a: int, b: int) -> int:
    """CHECK 5: Integer overflow — GUARDED by bounds check."""
    if abs(a) < 2**30 and abs(b) < 2**30:
        return a * b
    return 0
