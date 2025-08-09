#!/usr/bin/env python3
"""
Combine all text files in a repository into one PDF.

Author   : gpt-4o (modified by you)
Date     : {date}
Requires : reportlab  (pip install reportlab)
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# --------------------------------------------------------------------------- #
# Configuration – edit these only if you want to change the behaviour.
# --------------------------------------------------------------------------- #

IGNORE_PATTERNS = [
    "**/node_modules/**",
    "dist/**",
    ".git/**",
    ".idea/**",
    ".env*",
    "*.log",
    "*.tmp",
    "*.sqlite3",
    "*/*.pyc",
    "__pycache__/",
    ".DS_Store",
    "docs/build/*",
    "_build/**",
    "CMakeFiles/**",
    "CMakeCache.txt",
    "cmake_install.cmake",
    "Makefile",
    "install_manifest.txt",
    "*.dll",
    "*.so*",
    "*.exe",
    "*.app",
    "*.out",
    "*/*.pyc",
    "**/.pytest_cache/*",
    ".venv/**",
    "__init__",
]

BINARY_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".exe", ".dll", ".so", ".o", ".a", ".class",
    ".pdf", ".docx", ".xlsx", ".pptx",
}

def _is_ignored(path: Path, patterns) -> bool:
    """Return True if *path* matches any ignore pattern."""
    rel = path.as_posix()
    import fnmatch
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns)

def _is_binary(path: Path) -> bool:
    """Return True if the file extension indicates a binary file."""
    return path.suffix.lower() in BINARY_EXTENSIONS

# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #

def main(repo_root: Path | str = "."):
    repo_root = Path(repo_root).resolve()
    print(f"Scanning repository root: {repo_root}")

    files_to_process = []
    for path in repo_root.rglob("*"):
        if path.is_file():
            if _is_ignored(path.relative_to(repo_root), IGNORE_PATTERNS):
                continue
            if _is_binary(path):
                continue
            files_to_process.append(path)

    print(f"Found {len(files_to_process)} file(s) to include in the PDF.")

    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from reportlab.lib.units   import inch

    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    epoch_sec = int(time.time())
    pdf_name = f"Project_zephy_Augmentation_Analyzer_{date_str}_{epoch_sec}.pdf"
    pdf_path = repo_root / pdf_name

    c = canvas.Canvas(str(pdf_path), pagesize=LETTER)
    # ---- set PDF metadata -----------------------------------------------
    c.setAuthor("gpt-4o")          # <-- changed author
    c.setTitle("Project Zephy Augmentation Analyzer")
    c.setCreator("combine_to_pdf.py")
    # ---------------------------------------------------------------------

    width, height = LETTER
    margin = 0.75 * inch
    usable_width = width - 2*margin
    usable_height = height - 2*margin

    # Title page (optional)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height/2, "Project Zephy Augmentation Analyzer")
    c.showPage()

    for file_path in files_to_process:
        rel_path = file_path.relative_to(repo_root).as_posix()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, height - margin, rel_path)

        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"⚠️  Skipping {rel_path}: cannot read as UTF‑8 ({e})")
            continue

        c.setFont("Courier", 9)
        y = height - margin - 20
        lines = text.splitlines() or [""]
        line_height = 12

        for line in lines:
            if y < margin:
                c.showPage()
                y = height - margin
            # naive clipping to avoid overflow; adjust as needed
            display_line = line[:int(usable_width/7)]
            c.drawString(margin, y, display_line)
            y -= line_height

        c.showPage()

    c.save()
    print(f"✅ PDF created: {pdf_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root = "."
    main(root)
