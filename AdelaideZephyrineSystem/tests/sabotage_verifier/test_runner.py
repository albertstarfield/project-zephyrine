#!/usr/bin/env python3
"""Sabotage Verifier Self-Verification Test Harness.

This is the logic tracer. It proves the sabotage verifier's SMT checks
are correct against known attack vectors.

For each language (Ada, C, Python, TypeScript), we have:
  - known_bad_*.{ext}  : Code that MUST trigger specific violations
  - known_good_*.{ext} : Code that MUST NOT trigger violations

The test harness:
  1. Runs the sabotage verifier on each file
  2. Extracts which SMT checks fired
  3. Compares against expected results
  4. Reports FALSE NEGATIVES (missed violations) and FALSE POSITIVES (false alarms)

A test PASSES only if:
  - All expected violations are found (no false negatives)
  - No unexpected violations are found (no false positives)

Usage:
  python -m tests.sabotage_verifier.test_runner
  python tests/sabotage_verifier/test_runner.py
"""

import os
import sys
import importlib.util

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_verifier():
    """Dynamically load sabotage_verifier to avoid import issues."""
    verifier_path = os.path.join(
        os.path.dirname(PROJECT_ROOT), "src", "Util", "sabotage_verifier.py"
    )
    if not os.path.exists(verifier_path):
        # Try alternate path
        verifier_path = os.path.join(
            PROJECT_ROOT, "src", "Util", "sabotage_verifier.py"
        )
    spec = importlib.util.spec_from_file_location("sabotage_verifier", verifier_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_verifier_on_file(verifier_mod, filepath):
    """Run the sabotage verifier's SMT checks on a single file.

    Returns list of (category, line, message) tuples for SMT violations found.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError as e:
        return [], f"Failed to read {filepath}: {e}"

    violations = []

    # Get the SMT logic verification pattern's check function
    # We call it directly to avoid the full audit pipeline
    try:
        from z3 import sat  # noqa: F401
    except ImportError:
        return [], "z3 not installed"

    filepath_lower = filepath.lower()
    is_python = filepath_lower.endswith(".py")
    is_c = filepath_lower.endswith((".c", ".h"))
    is_ada = filepath_lower.endswith((".adb", ".ads"))
    is_tsjs = filepath_lower.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"))

    if is_python:
        functions = verifier_mod._parse_python_functions_ast(source)
        for func in functions:
            func["filepath"] = filepath
            issues = verifier_mod._verify_python_function_with_z3(func)
            for issue in issues:
                violations.append({
                    "category": issue["category"],
                    "line": issue["line"],
                    "message": issue["message"],
                    "solvers": issue.get("solvers", []),
                })

    elif is_c:
        functions = verifier_mod._parse_c_functions(source)
        for func in functions:
            func["filepath"] = filepath
            issues = verifier_mod._verify_c_function_with_z3(func)
            for issue in issues:
                violations.append({
                    "category": issue["category"],
                    "line": issue["line"],
                    "message": issue["message"],
                    "solvers": issue.get("solvers", []),
                })

    elif is_ada:
        functions = verifier_mod._parse_ada_functions(source)
        for func in functions:
            func["filepath"] = filepath
            issues = verifier_mod._verify_ada_function_with_z3(func)
            for issue in issues:
                violations.append({
                    "category": issue["category"],
                    "line": issue["line"],
                    "message": issue["message"],
                    "solvers": issue.get("solvers", []),
                })

    elif is_tsjs:
        functions = verifier_mod._parse_tsjs_functions(source)
        for func in functions:
            func["filepath"] = filepath
            issues = verifier_mod._verify_tsjs_function_with_z3(func)
            for issue in issues:
                violations.append({
                    "category": issue["category"],
                    "line": issue["line"],
                    "message": issue["message"],
                    "solvers": issue.get("solvers", []),
                })

    return violations, None


# ═══════════════════════════════════════════════════════════════════════════════
# Test Definitions
# ═══════════════════════════════════════════════════════════════════════════════

# Each test defines:
#   file: test file path (relative to TESTS_DIR)
#   expected_violations: dict of {category: True} for violations that MUST exist
#   expected_clean: list of categories that MUST NOT appear
#   description: what this test proves

TESTS = [
    # ── Ada Tests ──
    {
        "name": "Ada: Known Bad — all violations present",
        "file": "known_bad_ada.adb",
        "expected_violations": {
            "DIVISION_BY_ZERO",
            "INDEX_OUT_OF_BOUNDS",
            "NULL_DEREFERENCE",
            "CONSTRAINT_ERROR",
            "INTEGER_OVERFLOW",
            "PRECONDITION_CONTRADICTION",
            "POSTCONDITION_NOT_ENFORCED",
            "FLOAT_NAN_INF",
        },
        "expected_clean": [],
        "description": "Proves Ada SMT verifier detects all 8 violation categories",
    },
    {
        "name": "Ada: Known Good — no false positives",
        "file": "known_good_ada.adb",
        "expected_violations": set(),
        "expected_clean": {
            "DIVISION_BY_ZERO",
            "INDEX_OUT_OF_BOUNDS",
            "NULL_DEREFERENCE",
            "CONSTRAINT_ERROR",
            "INTEGER_OVERFLOW",
            "PRECONDITION_CONTRADICTION",
            "POSTCONDITION_NOT_ENFORCED",
            "FLOAT_NAN_INF",
        },
        "description": "Proves Ada SMT verifier does NOT false-positive on guarded code",
    },

    # ── Python Tests ──
    {
        "name": "Python: Known Bad — all violations present",
        "file": "known_bad_python.py",
        "expected_violations": {
            "DIVISION_BY_ZERO",
            "INDEX_OUT_OF_BOUNDS",
            "NONE_DEREFERENCE",
            "TYPE_CONTRADICTION",
            "INTEGER_OVERFLOW",
        },
        "expected_clean": [],
        "description": "Proves Python SMT verifier detects all 5 violation categories",
    },
    {
        "name": "Python: Known Good — no false positives",
        "file": "known_good_python.py",
        "expected_violations": set(),
        "expected_clean": {
            "DIVISION_BY_ZERO",
            "INDEX_OUT_OF_BOUNDS",
            "NONE_DEREFERENCE",
            "TYPE_CONTRADICTION",
            "INTEGER_OVERFLOW",
        },
        "description": "Proves Python SMT verifier does NOT false-positive on guarded code",
    },

    # ── C Tests ──
    {
        "name": "C: Known Bad — all violations present",
        "file": "known_bad_c.c",
        "expected_violations": {
            "NULL_POINTER_DEREFERENCE",
            "INTEGER_OVERFLOW",
        },
        "expected_clean": [],
        "description": "Proves C SMT verifier detects null deref and overflow",
    },
    {
        "name": "C: Known Good — no false positives",
        "file": "known_good_c.c",
        "expected_violations": set(),
        "expected_clean": {
            "NULL_POINTER_DEREFERENCE",
            "INTEGER_OVERFLOW",
        },
        "description": "Proves C SMT verifier does NOT false-positive on guarded code",
    },

    # ── TypeScript Tests ──
    {
        "name": "TypeScript: Known Bad — all violations present",
        "file": "known_bad_typescript.ts",
        "expected_violations": {
            "DIVISION_BY_ZERO",
            "INDEX_OUT_OF_BOUNDS",
            "NULL_DEREFERENCE",
            "TYPE_CONTRADICTION",
            "INTEGER_OVERFLOW",
        },
        "expected_clean": [],
        "description": "Proves TS/JS SMT verifier detects all 5 violation categories",
    },
    {
        "name": "TypeScript: Known Good — no false positives",
        "file": "known_good_typescript.ts",
        "expected_violations": set(),
        "expected_clean": {
            "DIVISION_BY_ZERO",
            "INDEX_OUT_OF_BOUNDS",
            "NULL_DEREFERENCE",
            "TYPE_CONTRADICTION",
            "INTEGER_OVERFLOW",
        },
        "description": "Proves TS/JS SMT verifier does NOT false-positive on guarded code",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Test Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    """Run all self-verification tests and report results."""
    print("=" * 80)
    print("SABOTAGE VERIFIER SELF-VERIFICATION TEST HARNESS")
    print("=" * 80)
    print()

    try:
        verifier_mod = load_verifier()
        print("[+] sabotage_verifier.py loaded successfully")
    except Exception as e:
        print(f"[-] FATAL: Failed to load sabotage_verifier.py: {e}")
        return 1

    print(f"[+] Tests defined: {len(TESTS)}")
    print()

    passed = 0
    failed = 0
    total_violations_found = 0

    for test in TESTS:
        test_name = test["name"]
        test_file = os.path.join(TESTS_DIR, test["file"])

        print(f"{'─' * 80}")
        print(f"TEST: {test_name}")
        print(f"  File: {test['file']}")
        print(f"  Description: {test['description']}")

        if not os.path.exists(test_file):
            print(f"  [-] FAIL: Test file not found: {test_file}")
            failed += 1
            continue

        violations, error = run_verifier_on_file(verifier_mod, test_file)
        if error:
            print(f"  [-] FAIL: {error}")
            failed += 1
            continue

        total_violations_found += len(violations)

        # Extract categories found
        found_categories = {v["category"] for v in violations}

        # Check for false negatives (expected but not found)
        false_negatives = test["expected_violations"] - found_categories
        # Check for false positives (unexpected but found)
        false_positives = found_categories - set(test["expected_clean"]) if test["expected_clean"] else found_categories - test["expected_violations"]

        # For clean tests: any violation found is a false positive
        if not test["expected_violations"]:
            false_positives = found_categories
            false_negatives = set()

        test_passed = len(false_negatives) == 0 and len(false_positives) == 0

        if test_passed:
            print("  [+] PASS")
            print(f"      Violations found: {sorted(found_categories)}")
            passed += 1
        else:
            print("  [-] FAIL")
            if false_negatives:
                print("      FALSE NEGATIVES (missed violations):")
                for cat in sorted(false_negatives):
                    print(f"        - {cat}: Expected but NOT found by verifier")
            if false_positives:
                print("      FALSE POSITIVES (false alarms):")
                for cat in sorted(false_positives):
                    print(f"        - {cat}: Found but should NOT be flagged")
            print(f"      Actual violations found: {sorted(found_categories)}")
            failed += 1

        # Print details of violations found
        if violations:
            print("      Violation details:")
            for v in violations:
                solvers_str = ",".join(v.get("solvers", []))
                print(f"        L{v['line']}: [{v['category']}] solvers={solvers_str}")

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"  Passed: {passed}/{len(TESTS)}")
    print(f"  Failed: {failed}/{len(TESTS)}")
    print(f"  Total violations found: {total_violations_found}")
    print()

    if failed == 0:
        print("  VERDICT: SABOTAGE VERIFIER LOGIC IS SOUND")
        print("  All expected violations detected. No false positives.")
        print("  The verifier's SMT checks are proven correct against known vectors.")
    else:
        print("  VERDICT: SABOTAGE VERIFIER HAS LOGICAL FALLACIES")
        print(f"  {failed} test(s) FAILED. The verifier has bugs that must be fixed.")
        print("  FALSE NEGATIVES = missed violations = dangerous code slips through")
        print("  FALSE POSITIVES = false alarms = build blocked for no reason")

    print("=" * 80)
    return failed


if __name__ == "__main__":
    sys.exit(run_all_tests())
