// KNOWN BAD TypeScript: Every function here SHOULD trigger SMT violations.

/* CHECK 1: Division by zero — b can be 0. */
function divideByZero(a: number, b: number): number {
    return a / b;
}

/* CHECK 2: Index out of bounds — no bounds check. */
function indexOOB(data: number[], idx: number): number {
    return data[idx];
}

/* CHECK 3: Null dereference — value used without null check. */
function noneDeref(value: string | null): number {
    return value.length;
}

/* CHECK 4: Type contradiction — contradictory typeof checks. */
function typeContradiction(x: any): string {
    if (typeof x === "string") {
        return parseInt(x).toString();
    }
    if (typeof x === "number") {
        return x.toString();
    }
    return "unknown";
}

/* CHECK 5: Integer overflow — unchecked multiplication. */
function overflowDemo(a: number, b: number): number {
    return a * b;
}
