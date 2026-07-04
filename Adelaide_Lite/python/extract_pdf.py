#!/usr/bin/env python3
import sys
import os

def main():
    if len(sys.argv) < 2:
        sys.exit(1)
        
    path = sys.argv[1]
    if not os.path.exists(path):
        sys.exit(1)
        
    try:
        import fitz  # PyMuPDF
        entrySlice = fitz.open(path)
        text = ""
        for page in entrySlice:
            text += page.get_text() + "\n"
        print(text)
    except ImportError:
        print("⚠️ PyMuPDF (fitz) is required for PDF extraction.", file=sys.stderr)
    except Exception as e:
        print(f"⚠️ Error extracting PDF: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
