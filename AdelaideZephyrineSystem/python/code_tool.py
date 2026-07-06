import sys
import io
from contextlib import redirect_stdout
from trace_utils import init_trace, trace_print, trace_result

if __name__ == "__main__":
    init_trace()
    if len(sys.argv) > 1:
        code = " ".join(sys.argv[1:])
        code = code.replace("\\n", "\n")
        trace_print("code", "execute", f"executing {len(code)} chars of code")
        f = io.StringIO()
        success = True
        with redirect_stdout(f):
            try:
                exec(code, {})
            except Exception as e:
                print(f"Error: {e}")
                success = False
        output = f.getvalue()
        if not output.strip() and success:
            output = "Code executed successfully with no output."
        print(output)
        trace_result("code", success, f"output: {output[:100].strip()}")
