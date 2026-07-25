# KNOWN BAD Python: Every function here SHOULD trigger SMT violations.
# The test runner expects specific violations from each function.
# If the verifier misses any, it's a FALSE NEGATIVE.

def divide_by_zero(a: int, b: int) -> int:
    """CHECK 1: Division by zero — b can be 0."""
    return a / b


def index_oob(data: list, idx: int) -> int:
    """CHECK 2: Index out of bounds — no bounds check."""
    return data[idx]


def none_deref(value: str | None) -> int:
    """CHECK 3: Null dereference — value used without None check."""
    return len(value)


def type_contradiction(x):
    """CHECK 4: Type contradiction — x checked as both str and int."""
    if isinstance(x, str):
        return int(x)
    if isinstance(x, int):
        return str(x)
    return 0


def overflow_demo(a: int, b: int) -> int:
    """CHECK 5: Integer overflow — multiplication unchecked."""
    return a * b
