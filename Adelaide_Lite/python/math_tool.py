import sys
import math
import sympy

if __name__ == "__main__":
    if len(sys.argv) > 1:
        expr = " ".join(sys.argv[1:])
        try:
            # Using sympy for safe mathematical evaluation
            res = sympy.sympify(expr)
            print(float(res))
        except Exception:
            try:
                print(eval(expr, {"__builtins__": None}, math.__dict__))
            except Exception as e2:
                print(f"Error evaluating math expression: {e2}")
