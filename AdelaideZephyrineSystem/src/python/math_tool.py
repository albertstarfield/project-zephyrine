import sys
import math
import sympy
from trace_utils import init_trace, trace_print, trace_result

if __name__ == "__main__":
    init_trace()
    if len(sys.argv) > 1:
        expr = " ".join(sys.argv[1:])
        trace_print("math", "evaluate", f"expr: {expr}")
        try:
            # Using sympy for safe mathematical evaluation
            res = sympy.sympify(expr)
            print(float(res))
            trace_result("math", True, f"result: {float(res)}")
        except Exception:
            try:
                result = eval(expr, {"__builtins__": None}, math.__dict__)  # nosec - sandboxed eval
                print(result)
                trace_result("math", True, f"result: {result}")
            except Exception as e2:
                print(f"Error evaluating math expression: {e2}")
                trace_result("math", False, str(e2))
