import sys
import os

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                print(f.read())
        else:
            print(f"File not found: {path}")
