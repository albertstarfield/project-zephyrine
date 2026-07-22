import sys
import os
from trace_utils import init_trace, trace_print, trace_result

if __name__ == "__main__":
    init_trace()
    if len(sys.argv) > 1:
        path = sys.argv[1]
        trace_print("cat", "read", f"file: {path}")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                print(f.read())
            trace_result("cat", True, f"read {path}")
        else:
            print(f"File not found: {path}")
            trace_result("cat", False, f"file not found: {path}")
