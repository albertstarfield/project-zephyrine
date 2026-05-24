import sys
import io
from contextlib import redirect_stdout

if __name__ == "__main__":
    if len(sys.argv) > 1:
        code = " ".join(sys.argv[1:])
        code = code.replace("\\n", "\n")
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                exec(code, {})
            except Exception as e:
                print(f"Error: {e}")
        output = f.getvalue()
        if not output.strip():
            output = "Code executed successfully with no output."
        print(output)
