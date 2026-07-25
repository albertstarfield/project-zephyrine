// KNOWN GOOD TypeScript: Every function here should NOT trigger SMT violations.

/* CHECK 1: Division by zero — GUARDED. */
function divideSafe(a: number, b: number): number {
    if (b !== 0) {
        return a / b;
    }
    return 0;
}

/* CHECK 2: Index out of bounds — GUARDED. */
function indexSafe(data: number[], idx: number): number {
    if (idx >= 0 && idx < data.length) {
        return data[idx];
    }
    return 0;
}

/* CHECK 3: Null dereference — GUARDED. */
function noneSafe(value: string | null): number {
    if (value !== null) {
        return value.length;
    }
    return 0;
}

/* CHECK 4: Type contradiction — consistent checks. */
function typeSafe(x: any): string {
    if (typeof x === "number") {
        return x.toString();
    }
    return "unknown";
}

/* CHECK 5: Integer overflow — GUARDED. */
function overflowSafe(a: number, b: number): number {
    if (Math.abs(a) < 2**26 && Math.abs(b) < 2**26) {
        return a * b;
    }
    return 0;
}
