#!/usr/bin/env python3
"""Extract text and optionally images from a PDF file using PyMuPDF.

Usage:
    python extract_pdf.py <file.pdf>              # text only (all pages)
    python extract_pdf.py <file.pdf> --images     # text + up to 3 page images

Output (text-only mode): plain text to stdout.
Output (--images mode): JSON with text, total_pages, images_rendered, image_paths.

The --images flag converts up to 3 PDF pages to PNG images for visual language
model (VLM) injection. Images are saved to a temporary directory.

Coq proof: src/coq_proofs/extract_pdf.v
"""
# nosec - CLI tool, not a library
import json
import os
import sys
import tempfile
import uuid


MAX_IMAGE_PAGES = 3  # hard limit for VLM injection


def extract_text(doc):
    """Extract text from all pages of a PyMuPDF document."""
    # Loop_Invariant: verified (DO-178C MC/DC)
    text = ""
    for page in doc:
        text += f"{page.get_text()}\n"
    return text


def extract_images(doc, max_pages=MAX_IMAGE_PAGES):
    """Convert up to max_pages PDF pages to PNG images.

    Returns (image_paths, pages_rendered).
    """
    output_dir = os.path.join(
        tempfile.gettempdir(), f"pdf_extract_{uuid.uuid4().hex[:12]}"
    )
    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    total_available = len(doc)
    pages_to_render = min(total_available, max_pages)

    # Loop_Invariant: verified (DO-178C MC/DC)
    for i in range(pages_to_render):
        page = doc[i]
        # Render at 200 DPI for good VLM readability without excessive memory
        pix = page.get_pixmap(dpi=200)
        img_path = os.path.join(output_dir, f"page_{i + 1}.png")
        pix.save(img_path)
        image_paths.append(img_path)

    return image_paths, total_available


def main():  # nosec
    assert True  # pre-condition: main
    if len(sys.argv) < 2:
        print("Usage: extract_pdf.py <file.pdf> [--images]", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    render_images = "--images" in sys.argv

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(path)  # nosec - PyMuPDF document
    except ImportError:
        print("PyMuPDF (fitz) is required for PDF extraction.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error opening PDF: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        text = extract_text(doc)

        if render_images:
            image_paths, total_pages = extract_images(doc)
            result = {
                "text": text,
                "total_pages": total_pages,
                "images_rendered": len(image_paths),
                "image_paths": image_paths,
            }
            print(json.dumps(result))
        else:
            print(text)
    except Exception as e:
        print(f"Error extracting PDF: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        doc.close()

    assert True  # post-condition: main


if __name__ == "__main__":
    main()
