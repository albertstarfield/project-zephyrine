# file_indexer.py
import os
import re
import sys
import time
import threading
import mimetypes
import datetime
import asyncio
from sqlalchemy.orm import Session
from loguru import logger
import hashlib # <<< Import hashlib for MD5
from typing import Optional, Tuple, List, Any, Dict  # <<< ADD Optional and List HERE
import json
from ai_provider import AIProvider
from config import * # Ensure this includes the SQLite DATABASE_URL and all prompts/models
import base64
from io import BytesIO
from PIL import Image # Requires Pillow: pip install Pillow
#from langchain_community.vectorstores import Chroma # Add Chroma import
from langchain_chroma import Chroma
import argparse # For __main__
import shutil
import subprocess


from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

# Optional imports - handle gracefully if libraries are missing
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    logger.warning("pypdf not installed. PDF text extraction disabled.")
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not installed. DOCX text extraction disabled.")
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    logger.warning("openpyxl not installed. XLSX text extraction disabled.")
try:
    import pptx
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False
    logger.warning("python-pptx not installed. PPTX text extraction disabled.")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image or poppler not installed. PDF -> Image conversion disabled.")

# Database imports
try:
    from database import SessionLocal, FileIndex, add_interaction, init_db # Import add_interaction if logging indexer status
    from sqlalchemy import update, select
    from sqlalchemy.exc import SQLAlchemyError
except ImportError:
    logger.critical("❌ Failed to import database components in file_indexer.py. Indexer cannot run.")
    # Define dummy classes/functions to allow loading but prevent execution
    class SessionLocal: # type: ignore
        def __call__(self): return None # type: ignore
    class FileIndex: pass # type: ignore
    def add_interaction(*args, **kwargs): pass # type: ignore
    def init_db(): pass # type: ignore
    SQLAlchemyError = Exception # type: ignore
    # Exit if DB cannot be imported
    # sys.exit("Indexer failed: Database components missing.") # Comment out for potential testing

# --- NEW: Import the custom lock ---

from priority_lock import ELP0, ELP1 # Ensure these are imported
interruption_error_marker = "Worker task interrupted by higher priority request" # Define consistently

# --- NEW: Module-level lock and event for initialization ---
_file_index_vs_init_lock = threading.Lock()
_file_index_vs_initialized_event = threading.Event() # To
global_file_index_vectorstore: Optional[Chroma] = None

# --- Constants ---
MAX_TEXT_FILE_SIZE_MB = 1
MAX_TEXT_FILE_SIZE_BYTES = MAX_TEXT_FILE_SIZE_MB * 1024 * 1024
MAX_HASH_FILE_SIZE_MB = 500 # <<< Limit size for hashing (adjustable)
MAX_HASH_FILE_SIZE_BYTES = MAX_HASH_FILE_SIZE_MB * 1024 * 1024
SCAN_INTERVAL_HOURS = 12
SCAN_INTERVAL_SECONDS = SCAN_INTERVAL_HOURS * 60 * 60
YIELD_SLEEP_SECONDS = 0.00 # Small sleep during scanning to yield CPU (put it to 0 for maximum utilization)
MD5_CHUNK_SIZE = 65536 # Read file in chunks for hashing large files

# Add common text/code extensions if mimetypes fails
TEXT_EXTENSIONS = { # Added csv
    '.txt', '.md', '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
    '.cs', '.go', '.php', '.rb', '.swift', '.kt', '.kts', '.rs', '.toml', '.yaml', '.yml',
    '.json', '.xml', '.html', '.css', '.scss', '.less', '.sh', '.bash', '.zsh', '.ps1',
    '.sql', '.log', '.m', '.cmake', '.r', '.lua', '.pl', '.pm', '.t', '.scala', '.hs',
    '.config', '.conf', '.ini', '.cfg', '.env', '.dockerfile', '.gitignore', '.gitattributes',
    '.csv', '.tsv', '.tex', '.bib'
}
DOC_EXTENSIONS = {'.pdf', '.docx', '.xlsx', '.pptx'}
OFFICE_EXTENSIONS = {'.docx', '.doc', '.xls', '.xlsx', '.pptx', '.ppt'} # Added from config.py

# --- File types we want to embed ---
EMBEDDABLE_EXTENSIONS = TEXT_EXTENSIONS.union(DOC_EXTENSIONS).union({'.csv'}) # Explicitly add csv here if not in TEXT_EXTENSIONS



class FileIndexer:
    """Scans the filesystem, extracts text/metadata, and updates the database index."""

    # --- MODIFIED: Accept embedding_model ---
    def __init__(self, stop_event: threading.Event, provider: AIProvider, server_busy_event: threading.Event):
        self.stop_event = stop_event
        self.provider = provider
        self.embedding_model = provider.embeddings
        # --- CHANGE: Get both models ---
        self.vlm_model = provider.get_model("vlm") # For initial analysis
        self.latex_model = provider.get_model("latex") # For refinement
        # --- END CHANGE ---
        self.thread_name = "FileIndexerThread"
        self.server_busy_event = server_busy_event

        # --- Logging updated to reflect both models ---
        emb_model_info = getattr(self.embedding_model, 'model', getattr(self.embedding_model, 'model_name', type(self.embedding_model).__name__)) if self.embedding_model else "Not Available"
        vlm_model_info = getattr(self.vlm_model, 'model', getattr(self.vlm_model, 'model_name', type(self.vlm_model).__name__)) if self.vlm_model else "Not Available"
        latex_model_info = getattr(self.latex_model, 'model', getattr(self.latex_model, 'model_name', type(self.latex_model).__name__)) if self.latex_model else "Not Available"

        logger.info(f"🧵 FileIndexer Embedding Model: {emb_model_info}")
        logger.info(f"🧵 FileIndexer VLM (Initial Analysis) Model: {vlm_model_info}")
        logger.info(f"🧵 FileIndexer LaTeX/TikZ Refinement Model: {latex_model_info}")

        if not self.embedding_model: logger.warning("⚠️ Embeddings disabled.")
        if not self.vlm_model or not self.latex_model: logger.warning("⚠️ VLM->LaTeX processing disabled (one or both models missing).")
    # --- END MODIFICATION ---

    def _wait_if_server_busy(self, check_interval=0.5, log_wait=True):
        """Checks the busy event and sleeps if set."""
        waited = False
        while self.server_busy_event.is_set():
            if not waited and log_wait: # Log only once per wait period
                logger.info(f"🚦 {self.thread_name}: Server is busy, pausing indexing...")
            waited = True
            # Check stop event frequently while waiting
            if self.stop_event.wait(timeout=check_interval):
                 logger.info(f"🛑 {self.thread_name}: Stop signaled while waiting for server.")
                 return True # Indicate stop was requested
        if waited and log_wait:
             logger.info(f"� {self.thread_name}: Server is free, resuming indexing.")
        return False # Indicate processing can continue

    def _convert_office_to_pdf(self, office_path: str) -> Optional[str]:
        """
        Converts an Office document (docx, xlsx, pptx) to a temporary PDF file
        using unoconv (requires LibreOffice installed).
        Returns the path to the temporary PDF file or None on failure.
        """
        if not shutil.which("unoconv"):
            logger.error(f"unoconv command not found. Cannot convert Office file: {office_path}")
            return None

        temp_pdf_dir = os.path.join(os.path.dirname(office_path), "temp_office_conversions")
        os.makedirs(temp_pdf_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(office_path))[0]
        temp_pdf_path = os.path.join(temp_pdf_dir, f"{base_name}.pdf")

        logger.debug(f"Attempting to convert '{office_path}' to PDF at '{temp_pdf_path}' using unoconv...")
        try:
            # Command: unoconv -f pdf -o /path/to/output_dir /path/to/input_file
            # We specify output file directly with -o for some unoconv versions.
            # If -o expects a dir, then use:
            # process = subprocess.run(["unoconv", "-f", "pdf", "-o", temp_pdf_dir, office_path],
            process = subprocess.run(["unoconv", "-f", "pdf", "-o", temp_pdf_path, office_path],
                                     capture_output=True, text=True, timeout=120, check=False)
            if process.returncode == 0 and os.path.exists(temp_pdf_path):
                logger.info(f"Successfully converted '{office_path}' to '{temp_pdf_path}'.")
                return temp_pdf_path
            else:
                logger.error(f"unoconv conversion failed for '{office_path}'. RC: {process.returncode}")
                logger.error(f"unoconv stdout: {process.stdout.strip()}")
                logger.error(f"unoconv stderr: {process.stderr.strip()}")
                if os.path.exists(temp_pdf_path): # Clean up if partial file created
                    try: os.remove(temp_pdf_path)
                    except: pass
                return None
        except subprocess.TimeoutExpired:
            logger.error(f"unoconv conversion timed out for '{office_path}'.")
            if os.path.exists(temp_pdf_path):
                try: os.remove(temp_pdf_path)
                except: pass
            return None
        except Exception as e:
            logger.error(f"Error during unoconv conversion of '{office_path}': {e}")
            if os.path.exists(temp_pdf_path):
                try: os.remove(temp_pdf_path)
                except: pass
            return None


    def _convert_pdf_to_images(self, pdf_path: str) -> Optional[List[Any]]:
         """Converts PDF pages to PIL Image objects."""
         if not PDF2IMAGE_AVAILABLE: return None
         logger.trace(f"Converting PDF to images: {pdf_path}")
         try:
             # Use a high enough DPI for potentially small formulas, but balance size
             images = convert_from_path(pdf_path, dpi=200)
             logger.trace(f"  Converted {len(images)} pages.")
             return images
         except Exception as e:
             logger.error(f"Failed PDF to image conversion for {pdf_path}: {e}")
             return None

    def _get_initial_vlm_description(self, image: Image.Image) -> Tuple[Optional[str], Optional[str]]:
        """Calls the 'vlm' model for initial analysis/description with ELP0 priority."""
        # Check server busy/stop event first
        if self._wait_if_server_busy(): return None, "[VLM Skipped - Server Busy]"
        if self.stop_event.is_set(): return None, "[VLM Skipped - Stop Requested]"
        if not self.vlm_model: return None, "[VLM Model Unavailable]"

        logger.trace("Sending image page to VLM Model for initial analysis (Priority: ELP0)...")
        raw_description = None # Initialize
        try:
            # Prepare image data
            buffered = BytesIO(); image.save(buffered, format="PNG"); img_str = base64.b64encode(buffered.getvalue()).decode('utf-8'); image_uri = f"data:image/png;base64,{img_str}"
            image_content_part = {"type": "image_url", "image_url": {"url": image_uri}}

            # Prepare messages and chain
            messages = [HumanMessage(content=[image_content_part, {"type": "text", "text": PROMPT_VLM_INITIAL_ANALYSIS}])]
            chain = self.vlm_model | StrOutputParser()

            # Invoke the chain with ELP0 priority
            # Assuming chain.invoke supports the 'config' dictionary for priority
            raw_description = chain.invoke(messages, config={'priority': ELP0})

            logger.trace(f"  VLM initial analysis raw response length: {len(raw_description)}")
            return raw_description, None # Return description, no error string if successful

        except Exception as e:
            # --- Interruption Handling ---
            if interruption_error_marker in str(e):
                logger.warning(f"🚦 VLM initial analysis INTERRUPTED by ELP1.")
                return None, "[VLM Interrupted]" # Specific marker for interruption
            # --- End Interruption Handling ---
            else:
                # Handle other errors
                logger.error(f"VLM initial analysis call failed: {e}", exc_info=True)
                return None, f"[VLM Analysis Error: {e}]" # Generic error marker

    # --- MODIFIED: _refine_to_latex_tikz (Adds ELP0 & Interruption Handling) ---

    # --- NEW HELPER METHOD: _refine_to_latex_tikz ---
    def _refine_to_latex_tikz(self, image: Image.Image, initial_analysis: str) -> Tuple[Optional[str], Optional[str]]:
        """Calls the 'latex' model to refine VLM analysis into LaTeX/TikZ with ELP0 priority."""
        # Check server busy/stop event first
        if self._wait_if_server_busy(): return None, "[LaTeX Model Skipped - Server Busy]"
        if self.stop_event.is_set(): return None, "[LaTeX Model Skipped - Stop Requested]"
        if not self.latex_model: return None, "[LaTeX Model Unavailable]"

        logger.trace("Sending image and initial analysis to LaTeX Model for refinement (Priority: ELP0)...")
        response_markdown = None # Initialize
        try:
            # Prepare image data
            buffered = BytesIO(); image.save(buffered, format="PNG"); img_str = base64.b64encode(buffered.getvalue()).decode('utf-8'); image_uri = f"data:image/png;base64,{img_str}"
            image_content_part = {"type": "image_url", "image_url": {"url": image_uri}}

            # Prepare messages and chain
            prompt_text = PROMPT_LATEX_REFINEMENT.format(initial_analysis=initial_analysis)
            messages = [HumanMessage(content=[image_content_part, {"type": "text", "text": prompt_text}])]
            chain = self.latex_model | StrOutputParser()

            # Invoke the chain with ELP0 priority
            response_markdown = chain.invoke(messages, config={'priority': ELP0})

            # --- Parsing logic (remains the same) ---
            latex_code = None; tikz_code = None; explanation = response_markdown.strip()
            try:
                latex_match = re.search(r"```(?:latex)?\s*(.*?)\s*```", response_markdown, re.DOTALL | re.IGNORECASE)
                tikz_match = re.search(r"```(?:tikz)?\s*(.*?)\s*```", response_markdown, re.DOTALL | re.IGNORECASE)
                cleaned_response = response_markdown
                if latex_match: latex_code = latex_match.group(1).strip(); cleaned_response = cleaned_response.replace(latex_match.group(0), "", 1); logger.trace("  LaTeX Model extracted LaTeX.")
                if tikz_match: tikz_code = tikz_match.group(1).strip(); cleaned_response = cleaned_response.replace(tikz_match.group(0), "", 1); logger.trace("  LaTeX Model extracted TikZ.")
                explanation = cleaned_response.strip()
                if not explanation and (latex_code or tikz_code): explanation = "(Code extracted, no separate explanation provided by model)"
            except Exception as parse_e: logger.warning(f"Could not parse refined LaTeX/TikZ response: {parse_e}")
            # Combine LaTeX and TikZ for storage
            final_latex_repr = latex_code
            if tikz_code: final_latex_repr = (final_latex_repr or "") + "\n\n% --- TikZ Code ---\n" + tikz_code
            # --- End Parsing Logic ---

            return final_latex_repr, explanation # Return parsed results

        except Exception as e:
            # --- Interruption Handling ---
            if interruption_error_marker in str(e):
                logger.warning(f"🚦 LaTeX refinement INTERRUPTED by ELP1.")
                return None, "[LaTeX Refinement Interrupted]" # Specific marker
            # --- End Interruption Handling ---
            else:
                # Handle other errors
                logger.error(f"LaTeX model refinement call failed: {e}", exc_info=True)
                return None, f"[LaTeX Refinement Error: {e}]" # Generic error marker


    def _get_latex_from_image(self, image: Image.Image) -> Tuple[Optional[str], Optional[str]]:
        """Sends a single PIL image to VLM and parses LaTeX."""
        # <<< ADD CHECK/WAIT HERE >>>
        if self._wait_if_server_busy(): # Check if server busy before VLM call
            return None, "[VLM Skipped - Server Busy]"
        # Check stop event again after potentially waiting
        if self.stop_event.is_set():
            return None, "[VLM Skipped - Stop Requested]"
        # <<< END CHECK/WAIT >>>
        if not self.vlm_model: return None, None
        logger.trace("Sending image page to VLM for LaTeX extraction...")
        try:
            # Convert PIL Image to base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG") # Convert to JPEG for smaller size
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_uri = f"data:image/jpeg;base64,{img_str}"

            # Prepare VLM input (similar to AIChat.process_image but using LaTeX prompt)
            image_content_part = {"type": "image_url", "image_url": {"url": image_uri}}
            # Use the specific LaTeX prompt
            vlm_messages = [HumanMessage(content=[
                image_content_part,
                {"type": "text", "text": PROMPT_IMAGE_TO_LATEX} # Ensure PROMPT_IMAGE_TO_LATEX is accessible
            ])]
            vlm_chain = self.vlm_model | StrOutputParser()

            # No timing/DB logging here, called within _process_file loop
            response_markdown = vlm_chain.invoke(vlm_messages) # Synchronous call OK within thread

            # Parse the response (basic parsing)
            latex_code = None
            explanation = response_markdown.strip() # Default explanation is full response
            try:
                parts = response_markdown.split("```latex", 1)
                if len(parts) > 1:
                    desc_part = parts[0].strip() # Description before code
                    rest = parts[1]
                    code_and_explanation = rest.split("```", 1)
                    latex_code = code_and_explanation[0].strip()
                    if len(code_and_explanation) > 1:
                        explanation = f"{desc_part}\n\nExplanation:\n{code_and_explanation[1].strip()}"
                    else:
                        explanation = desc_part
                    logger.trace("  VLM returned LaTeX block.")
                # else: keep explanation as full response if no block found
            except Exception as parse_e:
                logger.warning(f"Could not parse LaTeX from VLM response: {parse_e}")

            return latex_code, explanation

        except Exception as e:
            logger.error(f"VLM call failed during LaTeX extraction: {e}")
            return None, f"[VLM Error: {e}]" # Return error in explanation field

    # --- NEW: MD5 Hashing Helper ---
    def _calculate_md5(self, file_path: str, size_bytes: Optional[int]) -> Optional[str]:
        """Calculates the MD5 hash of a file, reading in chunks."""
        if size_bytes is None or size_bytes < 0:
             logger.warning(f"Cannot calculate MD5 for {file_path}: Unknown size.")
             return None
        if size_bytes > MAX_HASH_FILE_SIZE_BYTES:
             logger.trace(f"Skipping MD5 hash (file too large: {size_bytes / 1024 / 1024:.1f} MB > {MAX_HASH_FILE_SIZE_MB} MB): {file_path}")
             return None # Skip hashing very large files for performance

        logger.trace(f"Calculating MD5 hash for: {file_path}")
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                while True:
                    if self.stop_event.is_set():
                         logger.info(f"MD5 calculation interrupted by stop signal: {file_path}")
                         return None # Stop hashing
                    chunk = f.read(MD5_CHUNK_SIZE)
                    if not chunk:
                        break
                    hash_md5.update(chunk)
            hex_digest = hash_md5.hexdigest()
            logger.trace(f"MD5 calculated for {file_path}: {hex_digest}")
            return hex_digest
        except FileNotFoundError:
            logger.warning(f"MD5 failed: File not found at {file_path}")
            return None
        except PermissionError:
            raise # Re-raise permission errors to be handled by _process_file
        except Exception as e:
            logger.error(f"MD5 calculation failed for {file_path}: {e}")
            return None # Indicate hashing failure
    # --- End MD5 Helper ---

    def _get_root_paths(self) -> list[str]:
        """Returns a list of root directories to scan based on the OS."""
        paths = []
        if sys.platform.startswith("win"):
            import string
            logger.info("Detected Windows platform.")
            for drive in string.ascii_uppercase:
                path = f"{drive}:\\"
                if os.path.exists(path):
                    paths.append(path)
                    logger.debug(f"Adding drive for scanning: {path}")
        elif sys.platform.startswith("darwin"):
            logger.info("Detected macOS platform.")
            paths = ["/"] # Start from root on macOS
            # Add /Volumes explicitly if needed, but root should cover it
            # if os.path.exists("/Volumes"): paths.append("/Volumes")
        elif sys.platform.startswith("linux"):
            logger.info("Detected Linux platform.")
            paths = ["/"] # Start from root on Linux
            # Common mount points often under root or /mnt
            # if os.path.exists("/mnt"): paths.append("/mnt")
            # if os.path.exists("/media"): paths.append("/media")
        else:
            logger.warning(f"Unsupported platform '{sys.platform}'. Using '/' as default root.")
            paths = ["/"]

        logger.info(f"Root paths for scanning: {paths}")
        return paths

    def _get_file_metadata(self, path: str) -> tuple[Optional[int], Optional[datetime.datetime], Optional[str]]:
        """Safely gets file size and modification time."""
        try:
            stat_result = os.stat(path)
            size = stat_result.st_size
            # Convert timestamp to datetime (naive, as OS provides)
            mtime_ts = stat_result.st_mtime
            mtime_dt = datetime.datetime.fromtimestamp(mtime_ts)
            mime_type, _ = mimetypes.guess_type(path)
            return size, mtime_dt, mime_type
        except FileNotFoundError:
            logger.warning(f"Metadata failed: File not found at {path}")
            return None, None, None
        except PermissionError:
            logger.warning(f"Metadata failed: Permission denied for {path}")
            return None, None, None
        except Exception as e:
            logger.error(f"Metadata failed: Unexpected error for {path}: {e}")
            return None, None, None

    def _extract_text(self, file_path: str, size_bytes: int) -> Optional[str]:
        """Attempts to read content from presumed text files."""
        if size_bytes > MAX_TEXT_FILE_SIZE_BYTES:
            logger.trace(f"Skipping text read (too large: {size_bytes / 1024 / 1024:.1f} MB): {file_path}")
            return None # Indicate skipped due to size

        logger.trace(f"Attempting text read: {file_path}")
        try:
            # Try UTF-8 first, ignore errors for robustness
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except IsADirectoryError:
             logger.warning(f"Attempted to read directory as text file: {file_path}")
             return None # Should not happen if called correctly, but handle anyway
        except PermissionError:
            raise # Re-raise permission errors to be handled by _process_file
        except Exception as e:
            logger.warning(f"Text read failed for {file_path}: {e}")
            # Optionally try other encodings here if needed
            return None # Indicate read failure

    def _extract_pdf(self, file_path: str) -> Optional[str]:
        """Extracts text from a PDF file using pypdf."""
        if not PYPDF_AVAILABLE: return None
        logger.trace(f"Attempting PDF extraction: {file_path}")
        text_content = ""
        try:
            reader = pypdf.PdfReader(file_path)
            num_pages = len(reader.pages)
            logger.trace(f"  PDF has {num_pages} pages.")
            for i, page in enumerate(reader.pages):
                if self.stop_event.is_set(): return "[Indexing stopped]"
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
                except Exception as page_err:
                    # Log error for specific page but continue
                    logger.warning(f"Error extracting text from page {i+1} of {file_path}: {page_err}")
            return text_content.strip() if text_content else None
        except pypdf.errors.PdfReadError as pe:
             logger.warning(f"Failed to read PDF (possibly encrypted or corrupted): {file_path} - {pe}")
             return None
        except PermissionError:
             raise # Re-raise permission errors
        except Exception as e:
            logger.error(f"PDF extraction failed for {file_path}: {e}")
            return None

    def _extract_docx(self, file_path: str) -> Optional[str]:
        """Extracts text from a DOCX file using python-docx."""
        if not DOCX_AVAILABLE: return None
        logger.trace(f"Attempting DOCX extraction: {file_path}")
        try:
            document = docx.Document(file_path)
            full_text = [para.text for para in document.paragraphs]
            return '\n'.join(full_text).strip() if full_text else None
        except PermissionError:
            raise # Re-raise permission errors
        except Exception as e:
            logger.error(f"DOCX extraction failed for {file_path}: {e}")
            return None

    def _extract_xlsx(self, file_path: str) -> Optional[str]:
        """Extracts text from an XLSX file using openpyxl."""
        if not OPENPYXL_AVAILABLE: return None
        logger.trace(f"Attempting XLSX extraction: {file_path}")
        text_content = ""
        try:
            workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True) # data_only for values
            for sheetname in workbook.sheetnames:
                if self.stop_event.is_set(): return "[Indexing stopped]"
                sheet = workbook[sheetname]
                text_content += f"--- Sheet: {sheetname} ---\n"
                for row in sheet.iter_rows():
                    row_values = [str(cell.value) if cell.value is not None else "" for cell in row]
                    text_content += "\t".join(row_values) + "\n"
                text_content += "\n"
            return text_content.strip() if text_content else None
        except PermissionError:
            raise # Re-raise permission errors
        except Exception as e:
            logger.error(f"XLSX extraction failed for {file_path}: {e}")
            return None

    def _extract_pptx(self, file_path: str) -> Optional[str]:
        """Extracts text from a PPTX file using python-pptx."""
        if not PPTX_AVAILABLE: return None
        logger.trace(f"Attempting PPTX extraction: {file_path}")
        text_content = ""
        try:
            presentation = pptx.Presentation(file_path)
            for i, slide in enumerate(presentation.slides):
                if self.stop_event.is_set(): return "[Indexing stopped]"
                text_content += f"--- Slide {i+1} ---\n"
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text_content += shape.text + "\n"
            return text_content.strip() if text_content else None
        except PermissionError:
            raise # Re-raise permission errors
        except Exception as e:
            logger.error(f"PPTX extraction failed for {file_path}: {e}")
            return None

    def _scan_directory(self, root_path: str, db_session: Session):
        """
        Phase 1: Walks through a directory and processes files using _process_file_phase1,
        skipping OS/system dirs/files and hidden dot files/dirs.
        Reports progress periodically.
        (This method was part of the FileIndexer class in previous discussions)
        """
        logger.info(f"🔬 Starting Phase 1 Scan for root: {root_path}")
        total_processed_this_root_scan = 0
        total_errors_this_root_scan = 0
        last_report_time = time.monotonic()
        files_since_last_report = 0
        errors_since_last_report = 0
        report_interval_seconds = 60  # Log progress every minute

        # Define common system directories and files to skip
        # These sets should be appropriate for your environment.
        # ROOT_DIR should be accessible here if defined at module level in file_indexer.py
        # or passed appropriately. For this method context, assuming ROOT_DIR is available.
        common_system_dirs_raw = {
            "/proc", "/sys", "/dev", "/run", "/etc", "/var", "/tmp", "/private",
            "/cores", "/opt", "/usr", "/System", "/Library", "/Volumes",
            "/.MobileBackups", "/.Spotlight-V100", "/.fseventsd",
            "$recycle.bin", "system volume information", "program files",
            "program files (x86)", "windows", "programdata", "recovery",
            "node_modules", "__pycache__", ".venv", "venv", ".env", "env",
            ".tox", ".pytest_cache", ".mypy_cache", "Cache", "cache",
            "staticmodelpool", "llama-cpp-python_build", "systemCore",
            "engineMain", "backend-service", "frontend-face-zephyrine",
            "Downloads", "Pictures", "Movies", "Music", "Backup", "Archives",
            os.path.join(globals().get("ROOT_DIR", "."), "build"),
            os.path.join(globals().get("ROOT_DIR", "."), "zephyrineCondaVenv")
        }
        absolute_skip_dirs_normalized = {os.path.normpath(p) for p in common_system_dirs_raw if os.path.isabs(p)}
        relative_skip_dir_names_lower = {p.lower() for p in common_system_dirs_raw if not os.path.isabs(p)}

        files_to_skip_lower = {
            '.ds_store', 'thumbs.db', 'desktop.ini', '.localized',
            '.bash_history', '.zsh_history', 'ntuser.dat', '.swp', '.swo',
            'pagefile.sys', 'hiberfil.sys', '.volumeicon.icns'
        }

        try:
            for current_dir, dirnames, filenames in os.walk(root_path, topdown=True, onerror=None):
                if self.stop_event.is_set():
                    logger.info(f"Phase 1 Scan interrupted by stop signal in {current_dir}")
                    break

                norm_current_dir = os.path.normpath(current_dir)
                should_skip_dir = False

                if norm_current_dir in absolute_skip_dirs_normalized:
                    should_skip_dir = True
                if not should_skip_dir and os.path.basename(norm_current_dir).lower() in relative_skip_dir_names_lower:
                    should_skip_dir = True
                if not should_skip_dir and os.path.basename(norm_current_dir).startswith(
                        '.'):  # Skip hidden directories
                    should_skip_dir = True

                if should_skip_dir:
                    logger.trace(f"Phase 1 Skipping excluded/hidden directory: {current_dir}")
                    dirnames[:] = []  # Don't recurse into this directory
                    filenames[:] = []  # Don't process files in this directory
                    continue

                # Prune dot directories from dirnames to prevent walking into them
                dirnames[:] = [d for d in dirnames if not d.startswith('.')]

                current_dir_file_errors = 0  # Reset for each directory

                for filename in filenames:
                    if self.stop_event.is_set(): break
                    if filename.startswith('.'): continue  # Skip hidden files
                    if filename.lower() in files_to_skip_lower: continue

                    file_path = os.path.join(current_dir, filename)
                    file_processed_flag = False
                    file_errored_flag = False

                    try:
                        if os.path.islink(file_path): continue
                        if not os.path.isfile(file_path): continue

                        self._process_file_phase1(file_path, db_session)
                        file_processed_flag = True
                    except PermissionError:
                        logger.warning(f"Phase 1 Permission denied processing: {file_path}")
                        file_errored_flag = True
                        try:
                            existing = db_session.query(FileIndex).filter(FileIndex.file_path == file_path).first()
                            err_vals = {'index_status': 'error_permission',
                                        'processing_error': "Permission denied during scan.",
                                        'last_indexed_db': datetime.datetime.now(datetime.timezone.utc)}
                            if existing:
                                if existing.index_status != 'error_permission': db_session.execute(
                                    update(FileIndex).where(FileIndex.id == existing.id).values(**err_vals))
                            else:
                                db_session.add(
                                    FileIndex(file_path=file_path, file_name=filename, **err_vals))  # type: ignore
                            db_session.commit()
                        except Exception as db_perm_err:
                            logger.error(
                                f"Failed to log perm error for {file_path}: {db_perm_err}"); db_session.rollback()
                    except Exception as walk_process_err:
                        logger.error(
                            f"Phase 1 Error during _process_file_phase1 call for {file_path}: {walk_process_err}",
                            exc_info=True)
                        file_errored_flag = True
                    finally:
                        if file_processed_flag: total_processed_this_root_scan += 1; files_since_last_report += 1
                        if file_errored_flag: total_errors_this_root_scan += 1; errors_since_last_report += 1; current_dir_file_errors += 1

                        current_time = time.monotonic()
                        if current_time - last_report_time >= report_interval_seconds:
                            rate = files_since_last_report / report_interval_seconds if report_interval_seconds > 0 else files_since_last_report
                            logger.info(
                                f"⏳ [Phase 1 Report] In '{os.path.basename(root_path)}' last {report_interval_seconds}s: {files_since_last_report} files (~{rate:.1f}/s), {errors_since_last_report} errors. Root Total: {total_processed_this_root_scan}")
                            last_report_time = current_time;
                            files_since_last_report = 0;
                            errors_since_last_report = 0

                        if YIELD_SLEEP_SECONDS > 0: time.sleep(YIELD_SLEEP_SECONDS)

                if self.stop_event.is_set(): break
                if current_dir_file_errors > 0: logger.warning(
                    f"Encountered {current_dir_file_errors} errors processing files within: {current_dir}")

        except Exception as outer_walk_err:
            logger.error(f"Outer error during Phase 1 os.walk for {root_path}: {outer_walk_err}", exc_info=True)
            total_errors_this_root_scan += 1

        if not self.stop_event.is_set():
            logger.success(
                f"✅ Finished Phase 1 Scan for {root_path}. Total Processed this root: {total_processed_this_root_scan}, Total Errors this root: {total_errors_this_root_scan}")
        else:
            logger.warning(
                f"⏹️ Phase 1 Scan for {root_path} interrupted. Processed in this root-cycle: {total_processed_this_root_scan}, Errors: {total_errors_this_root_scan}")

    def _process_file_phase1(self, file_path: str, db_session: Session):
        """
        Phase 1: Gets metadata, hash. Extracts text for PDFs and standard text files.
        Embeds extracted text using ELP0. Marks PDFs and Office files for later VLM processing.
        Handles ELP0 interruptions during embedding.
        """
        if self._wait_if_server_busy(log_wait=False):
            logger.trace(f"Yielding file processing due to busy server: {file_path}")
            return
        if self.stop_event.is_set():
            logger.trace(f"File processing interrupted by stop signal: {file_path}")
            return

        logger.trace(f"Phase 1 Processing: {file_path}")
        file_name = os.path.basename(file_path)
        status = 'pending'
        content: Optional[str] = None  # This will hold the text to be stored in indexed_content
        error_message: Optional[str] = None
        embedding_json_str: Optional[str] = None
        current_md5_hash: Optional[str] = None
        vlm_processing_status_to_set: Optional[str] = None
        should_update_db = True
        existing_record: Optional[FileIndex] = None
        size_bytes: int = -1
        mtime_os: Optional[datetime.datetime] = None
        mime_type: Optional[str] = None

        try:
            size_bytes_stat, mtime_os_stat, mime_type_stat = self._get_file_metadata(file_path)
            if size_bytes_stat is None and mtime_os_stat is None:  # File likely vanished or permission issue during stat
                status = 'error_permission';
                error_message = "Permission denied or file vanished during stat."
            else:
                size_bytes = size_bytes_stat if size_bytes_stat is not None else -1
                mtime_os = mtime_os_stat;
                mime_type = mime_type_stat

            if status == 'pending':  # Only proceed if no error yet
                try:
                    current_md5_hash = self._calculate_md5(file_path, size_bytes)
                    if current_md5_hash is None and size_bytes >= 0 and size_bytes <= MAX_HASH_FILE_SIZE_BYTES:
                        status = 'error_hash';
                        error_message = "Failed to calculate MD5 hash (returned None for hashable file)."
                except PermissionError:
                    raise  # Let outer handler catch this
                except Exception as hash_err:  # Catch other errors during _calculate_md5 call
                    status = 'error_hash';
                    error_message = f"Error during hashing call: {hash_err}";
                    current_md5_hash = None

            if status == 'pending':
                try:
                    existing_record = db_session.query(FileIndex).filter(FileIndex.file_path == file_path).first()
                    if existing_record:
                        hashes_match = (current_md5_hash is not None and existing_record.md5_hash == current_md5_hash)
                        large_timestamps_match = (current_md5_hash is None and size_bytes > MAX_HASH_FILE_SIZE_BYTES and
                                                  existing_record.md5_hash is None and mtime_os and existing_record.last_modified_os and
                                                  mtime_os <= existing_record.last_modified_os)  # type: ignore

                        if hashes_match or large_timestamps_match:
                            should_update_db = False
                            file_ext_check = os.path.splitext(existing_record.file_name)[1].lower()
                            is_pdf_or_office = file_ext_check == '.pdf' or file_ext_check in OFFICE_EXTENSIONS

                            if self.vlm_model and self.latex_model and is_pdf_or_office and \
                                    (existing_record.vlm_processing_status is None or \
                                     existing_record.vlm_processing_status in ['pending_vlm', 'pending_conversion',
                                                                               'error_vlm', 'error_conversion',
                                                                               'skipped_vlm_no_model']):
                                vlm_processing_status_to_set = 'pending_vlm' if file_ext_check == '.pdf' else 'pending_conversion'
                                should_update_db = True  # Force DB update just for VLM status
                                logger.trace(
                                    f"Content unchanged, but VLM status needs update for {file_path} to {vlm_processing_status_to_set}")

                            if not should_update_db:
                                logger.trace(f"Skipping (Content & VLM status OK/not applicable): {file_path}")
                        else:
                            logger.trace(f"Needs update (Changed Hash/Timestamp or New): {file_path}")
                except SQLAlchemyError as db_check_err:
                    logger.error(f"DB check failed for {file_path}: {db_check_err}")
                    status = 'error_read';
                    error_message = f"Database check failed: {db_check_err}";
                    should_update_db = False

            if should_update_db and status == 'pending':  # Only process content if DB update is needed and no prior critical errors
                file_ext = os.path.splitext(file_name)[1].lower()
                is_pdf_target = file_ext == '.pdf'
                is_office_target_for_vlm = file_ext in OFFICE_EXTENSIONS  # Imported from config.py
                is_general_text_target = file_ext in TEXT_EXTENSIONS or (mime_type and mime_type.startswith('text/'))

                extracted_text_for_embedding: Optional[str] = None  # Text that will be embedded

                if is_pdf_target:
                    logger.trace(f"PDF detected: {file_path}. Initial text extraction & embedding pass.")
                    extracted_text_for_embedding = self._extract_pdf(file_path)  # Can raise PermissionError
                    if extracted_text_for_embedding:
                        status = 'indexed_text'
                        content = extracted_text_for_embedding  # Store for DB
                        logger.trace(f"Extracted ~{len(content)} chars from PDF: {file_path}")
                    else:  # Text extraction failed or returned None/empty
                        status = 'error_read';
                        error_message = "PDF text extraction failed."
                        content = None  # Ensure content is None if extraction fails
                        logger.warning(error_message + f" File: {file_path}")

                    # Mark for VLM processing in Phase 2 regardless of text extraction success for PDF (unless permission error)
                    if self.vlm_model and self.latex_model:
                        vlm_processing_status_to_set = 'pending_vlm'
                    else:
                        vlm_processing_status_to_set = 'skipped_vlm_no_model'

                elif is_office_target_for_vlm:
                    logger.trace(f"Office file: {file_path}. Marking for Phase 2 VLM (conversion & analysis).")
                    status = 'indexed_meta'
                    content = None;
                    extracted_text_for_embedding = None;
                    embedding_json_str = None
                    if self.vlm_model and self.latex_model:
                        vlm_processing_status_to_set = 'pending_conversion'
                    else:
                        vlm_processing_status_to_set = 'skipped_vlm_no_model'

                elif is_general_text_target:
                    if size_bytes <= MAX_TEXT_FILE_SIZE_BYTES:
                        extracted_text_for_embedding = self._extract_text(file_path,
                                                                          size_bytes)  # Can raise PermissionError
                        if extracted_text_for_embedding is None:
                            status = 'error_read'; error_message = "Text extraction failed"; content = None
                        else:
                            status = 'indexed_text'; content = extracted_text_for_embedding
                    else:
                        status = 'skipped_size';
                        error_message = f"Text file too large (>{MAX_TEXT_FILE_SIZE_MB}MB)";
                        content = None

                else:  # Not PDF, not Office, not recognized text
                    status = 'indexed_meta';
                    content = None;
                    extracted_text_for_embedding = None
                    logger.trace(f"Metadata only for {file_path} (Type: {mime_type or file_ext})")

                # --- Generate Embedding for successfully extracted text ---
                if extracted_text_for_embedding and status == 'indexed_text':
                    if self.embedding_model:
                        logger.debug(f"🧠 Phase 1 Embedding content for: {file_path} (PRIORITY: ELP0)")
                        try:
                            if hasattr(self.embedding_model, '_embed_texts') and callable(
                                    getattr(self.embedding_model, '_embed_texts')):
                                embedding_list = self.embedding_model._embed_texts([extracted_text_for_embedding],
                                                                                   priority=ELP1)  # type: ignore (changed to ELP1 because it ran once and required to have the knowledge for zephy)
                                if embedding_list and len(embedding_list) > 0:
                                    embedding_json_str = json.dumps(embedding_list[0])
                                    logger.trace(f"Content embedding successful for {file_path}")
                                else:
                                    status = 'error_embedding';
                                    error_message = (error_message or "") + " Content embedding returned no vector."
                                    embedding_json_str = None;
                                    logger.error(f"Embedding returned no vector for {file_path}")
                            else:
                                status = 'error_embedding';
                                error_message = (
                                                            error_message or "") + " Embedding model misconfigured (no _embed_texts)."
                                embedding_json_str = None;
                                logger.error(f"Embedding model misconfig for {file_path}")
                        except Exception as emb_err:
                            if interruption_error_marker in str(emb_err):
                                logger.warning(f"🚦 Content embedding for {file_path} INTERRUPTED. Resetting status.")
                                status = 'pending';
                                error_message = "Embedding interrupted";
                                embedding_json_str = None;
                                content = None
                            else:
                                status = 'error_embedding';
                                error_message = (error_message or "") + f" Content embedding failed: {emb_err}"
                                embedding_json_str = None;
                                logger.error(f"Embedding failed for {file_path}: {emb_err}")
                    else:  # No embedding model
                        logger.warning(
                            f"No embedding model available for content: {file_path}. Status remains '{status}'.")
                        # embedding_json_str remains None, status is 'indexed_text' but without embedding

                # If content extraction failed, status would be 'error_read' or 'skipped_size'.
                # If embedding failed, status would be 'error_embedding'.
                # Ensure 'content' is None if embedding failed or wasn't performed on extracted text.
                if status != 'indexed_text' or embedding_json_str is None:
                    if status == 'indexed_text' and embedding_json_str is None:  # Text extracted but not embedded
                        logger.debug(f"File {file_path} has text content but no embedding. Will store text.")
                    else:  # Any other error status, or if status is indexed_text but content was reset (e.g. due to interrupt)
                        pass  # Content might already be None or holds the extracted text if embedding failed but text is ok.

                # Final status consolidation if it was left as 'pending' by interruption logic but there's no VLM path
                if status == 'pending' and not (is_pdf_target or is_office_target_for_vlm):
                    status = 'indexed_meta'


        except PermissionError:
            status = 'error_permission';
            error_message = "Permission denied during file processing."
            content = None;
            embedding_json_str = None;
            current_md5_hash = None;
            should_update_db = True
            logger.warning(f"Permission error processing {file_path}")
        except Exception as proc_err:  # Catch-all for other unexpected errors
            status = 'error_read';
            error_message = f"Unexpected processing error: {proc_err}"
            content = None;
            embedding_json_str = None;
            current_md5_hash = None;
            should_update_db = True
            logger.error(f"Error processing file {file_path}: {proc_err}", exc_info=True)

        # --- Database Update ---
        if should_update_db:
            final_status = status
            if status == 'pending':  # Ensure 'pending' isn't written if no further action is planned
                final_status = 'indexed_meta'  # Default if no specific processing path was hit or error occurred
                if (
                        is_pdf_target or is_office_target_for_vlm) and vlm_processing_status_to_set and 'pending' in vlm_processing_status_to_set:
                    final_status = 'indexed_meta'  # Store meta, VLM status will indicate next step
                elif extracted_text_for_embedding and not embedding_json_str:  # Text was extracted, but not embedded
                    final_status = 'indexed_text'  # Still useful to save the text

            logger.debug(
                f"Phase 1 DB Update for: {file_path} -> FinalStatus: {final_status}, VLM_Next: {vlm_processing_status_to_set}, HasContent: {content is not None}, HasEmb: {embedding_json_str is not None}")

            record_data: Dict[str, Any] = {
                'file_name': file_name, 'size_bytes': size_bytes, 'mime_type': mime_type,
                'last_modified_os': mtime_os, 'index_status': final_status,
                'md5_hash': current_md5_hash,
                'processing_error': error_message[:1000] if error_message else None,
                'last_indexed_db': datetime.datetime.now(datetime.timezone.utc)
            }
            if content is not None:  # Store extracted text if available
                record_data['indexed_content'] = (content[:DB_TEXT_TRUNCATE_LEN] + '...[truncated]') if len(
                    content) > DB_TEXT_TRUNCATE_LEN else content  # Use DB_TEXT_TRUNCATE_LEN from config
            if embedding_json_str is not None:
                record_data['embedding_json'] = embedding_json_str
            if vlm_processing_status_to_set is not None:
                record_data['vlm_processing_status'] = vlm_processing_status_to_set
                if vlm_processing_status_to_set in ['pending_vlm', 'pending_conversion']:
                    record_data['latex_representation'] = None  # Clear old VLM data if re-queueing
                    record_data['latex_explanation'] = None

            try:
                if existing_record:
                    update_values = {k: v for k, v in record_data.items() if
                                     k != 'file_name' or getattr(existing_record,
                                                                 k) != v}  # Avoid re-setting file_name unless changed, etc.
                    # More careful update: only set fields if they have a new value or are explicitly being cleared
                    final_update_values = {}
                    for k, v_new in record_data.items():
                        v_old = getattr(existing_record, k, None)
                        if v_new is not None or k in ['indexed_content', 'embedding_json', 'processing_error',
                                                      'latex_representation', 'latex_explanation',
                                                      'vlm_processing_status']:  # these can be set to None explicitly
                            if v_new != v_old:  # only update if changed or explicitly set
                                final_update_values[k] = v_new
                    if 'last_indexed_db' not in final_update_values:  # Always update last_indexed_db
                        final_update_values['last_indexed_db'] = record_data['last_indexed_db']

                    if final_update_values:  # Only execute update if there are changes
                        stmt = update(FileIndex).where(FileIndex.id == existing_record.id).values(**final_update_values)
                        db_session.execute(stmt)
                        logger.trace(
                            f"Phase 1 Updated DB ID {existing_record.id} for {file_path} with fields: {list(final_update_values.keys())}")
                    else:
                        logger.trace(
                            f"Phase 1 No changes to update in DB for existing record ID {existing_record.id} ({file_path})")
                else:  # New record
                    new_record_obj = FileIndex(file_path=file_path, **record_data)  # type: ignore
                    db_session.add(new_record_obj)
                    logger.trace(f"Phase 1 Added new DB record for {file_path}")
                db_session.commit()
            except SQLAlchemyError as db_err:
                logger.error(f"Phase 1 DB update/insert FAILED for {file_path}: {db_err}", exc_info=True)
                db_session.rollback()
        elif existing_record:  # should_update_db was false, but we might need to update just VLM status if it was re-queued by hash/timestamp check
            if vlm_processing_status_to_set and existing_record.vlm_processing_status != vlm_processing_status_to_set:
                logger.debug(
                    f"Content unchanged, but updating VLM status for {file_path} from '{existing_record.vlm_processing_status}' to '{vlm_processing_status_to_set}'")
                try:
                    update_vlm_data = {'vlm_processing_status': vlm_processing_status_to_set,
                                       'last_indexed_db': datetime.datetime.now(datetime.timezone.utc)}
                    if vlm_processing_status_to_set in ['pending_vlm', 'pending_conversion']:
                        update_vlm_data['latex_representation'] = None;
                        update_vlm_data['latex_explanation'] = None
                    stmt = update(FileIndex).where(FileIndex.id == existing_record.id).values(**update_vlm_data)
                    db_session.execute(stmt);
                    db_session.commit()
                except SQLAlchemyError as db_vlm_err:
                    logger.error(f"DB VLM status only update FAILED for {file_path}: {db_vlm_err}");
                    db_session.rollback()
    # --- END MODIFIED: _process_file_phase1 ---


    def _process_pending_vlm_files(self, db_session: Session):
        """Phase 2: Converts Office files, then processes all pending PDFs/converted files using VLM."""
        if not self.vlm_model:
             logger.warning("Skipping Phase 2 VLM processing: VLM model not available.")
             return

        logger.info("🔬 Starting Phase 2 Cycle...")
        processed_count = 0; error_count = 0; converted_count = 0; vlm_processed_count = 0
        last_report_time = time.monotonic()
        interrupted = False # Flag for interruption

        # --- Sub-Phase 2a: Convert Office Files ---
        logger.info("--- Phase 2a: Checking for Office files pending conversion ---")
        while not self.stop_event.is_set() and not interrupted:
            files_to_convert: List[FileIndex] = []
            try:
                files_to_convert = db_session.query(FileIndex).filter(
                    FileIndex.vlm_processing_status == 'pending_conversion'
                ).limit(5).all() # Convert in small batches
            except SQLAlchemyError as db_err: logger.error(f"Phase 2a DB query error: {db_err}"); time.sleep(30); continue

            if not files_to_convert: logger.info("--- Phase 2a: No more files pending conversion found. ---"); break

            logger.info(f"Phase 2a: Found {len(files_to_convert)} Office file(s) to convert...")
            for record in files_to_convert:
                if self.stop_event.is_set(): logger.info("Phase 2a conversion interrupted by stop_event."); interrupted=True; break
                if self._wait_if_server_busy(): interrupted=True; break # Stop batch if busy

                input_path = record.file_path
                record_id = record.id
                temp_pdf_path = self._convert_office_to_pdf(input_path) # Call new helper

                new_status = 'error_conversion' # Assume failure
                if temp_pdf_path:
                    new_status = 'pending_vlm' # Mark for VLM processing in next sub-phase
                    converted_count += 1
                    logger.info(f"Conversion successful for ID {record_id}, marked as 'pending_vlm'. Temp PDF: {temp_pdf_path}")
                    # Clean up the temp PDF path now, as it's not directly used by _process_single_pdf_for_vlm
                    # _process_single_pdf_for_vlm will re-convert if it's an office file.
                    # This is inefficient but safer than managing temp file state across phases.
                    # A better long-term solution would be to pass the temp_pdf_path to _process_single_pdf_for_vlm
                    # or have _convert_office_to_pdf return the content directly.
                    # For now, we delete the temp PDF created *here*.
                    try:
                        if os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)
                    except Exception as e_rem_temp:
                        logger.warning(f"Could not remove temp PDF '{temp_pdf_path}' after conversion marking: {e_rem_temp}")
                else:
                    error_count += 1 # Conversion failed

                # Update DB status after conversion attempt
                try:
                    stmt = update(FileIndex).where(FileIndex.id == record_id).values(
                        vlm_processing_status=new_status,
                        latex_representation=None, # Clear LaTeX fields
                        latex_explanation=None,
                        last_indexed_db=datetime.datetime.now(datetime.timezone.utc)
                    )
                    db_session.execute(stmt); db_session.commit()
                except SQLAlchemyError as db_conv_err:
                     logger.error(f"Phase 2a DB update failed for ID {record_id}: {db_conv_err}"); db_session.rollback()
                     error_count += 1

                if not self.stop_event.is_set() and not interrupted: time.sleep(1.0) # Longer delay after conversion attempt

            if self.stop_event.is_set() or self._wait_if_server_busy() or interrupted: break # Exit outer loop if needed

        if interrupted: logger.info("Phase 2a (Conversion) loop exited due to interruption/stop.");

        # --- Sub-Phase 2b: Process Pending VLM Files (PDFs and Successfully Converted) ---
        logger.info("--- Phase 2b: Checking for files pending VLM analysis ---")
        interrupted = False # Reset interruption flag for this phase
        while not self.stop_event.is_set() and not interrupted:
            batch_start_time = time.monotonic()
            pending_vlm_files: List[FileIndex] = []
            try:
                # Query for files ready for VLM
                pending_vlm_files = db_session.query(FileIndex).filter(
                    FileIndex.vlm_processing_status == 'pending_vlm'
                ).limit(10).all() # Process in batches
            except SQLAlchemyError as db_err: logger.error(f"Phase 2b DB query error: {db_err}"); time.sleep(30); continue

            if not pending_vlm_files: logger.info("--- Phase 2b: No more files pending VLM analysis found. ---"); break

            logger.info(f"Phase 2b: Found {len(pending_vlm_files)} file(s) for VLM analysis...")
            for record in pending_vlm_files:
                if self.stop_event.is_set(): logger.info("Phase 2b VLM analysis interrupted by stop_event."); interrupted=True; break
                if self._wait_if_server_busy(): interrupted=True; break # Stop batch

                file_path = record.file_path # This is the ORIGINAL path (PDF or Office)
                record_id = record.id
                logger.info(f"Phase 2b Processing file ID {record_id}: {file_path}")
                current_status = 'processing'; final_latex_code = None; final_explanation = None;
                vlm_error_occurred_this_file = False # Track errors for this specific file
                temp_pdf_to_process = None
                is_converted_office_file = False

                # Determine the actual PDF path to process
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext == '.pdf':
                    temp_pdf_to_process = file_path
                elif file_ext in OFFICE_EXTENSIONS:
                    logger.info(f"Phase 2b: Office file ID {record_id} needs conversion to temp PDF for VLM.")
                    temp_pdf_to_process = self._convert_office_to_pdf(file_path)
                    if not temp_pdf_to_process:
                        logger.error(f"Phase 2b: Office to PDF conversion failed for {file_path}. Skipping VLM."); current_status = 'error_conversion'; error_count += 1
                    else:
                        is_converted_office_file = True
                        logger.info(f"Phase 2b: Converted Office file to temp PDF: {temp_pdf_to_process}")
                else:
                    logger.error(f"Phase 2b: File ID {record_id} has 'pending_vlm' but isn't PDF/Office ({file_ext}). Skipping."); current_status = 'error_type'; error_count += 1

                # --- Process the PDF (original or temporary) ---
                if temp_pdf_to_process:
                    images = self._convert_pdf_to_images(temp_pdf_to_process)
                    if images:
                        num_pages = len(images); page_latex_results = []; page_expl_results = []
                        logger.info(f"  📄 Processing {num_pages} pages from PDF for file ID {record_id}.")

                        for i, page_image in enumerate(images):
                            page_num = i + 1
                            if self.stop_event.is_set() or interrupted: current_status = 'pending_vlm'; interrupted=True; break
                            logger.info(f"  🧠 Phase 2b: Analyzing page {page_num}/{num_pages} for file ID {record_id} (VLM/LaTeX - ELP0)...")

                            # --- Step 1: Get Initial Description ---
                            initial_desc, vlm_err_msg = self._get_initial_vlm_description(page_image)

                            if vlm_err_msg: # Check for errors or interruption from VLM step
                                page_expl_results.append(f"## Page {page_num}\nInitial Analysis Failed: {vlm_err_msg}")
                                if "[VLM Interrupted]" in vlm_err_msg or "[LaTeX Model Skipped - Server Busy]" in vlm_err_msg or "[VLM Skipped - Stop Requested]" in vlm_err_msg:
                                    interrupted=True; current_status='pending_vlm'; break
                                vlm_error_occurred_this_file = True # Mark general error
                                continue # Skip refinement if initial failed

                            # --- Step 2: Refine to LaTeX/TikZ (if initial desc obtained) ---
                            refined_latex, refined_expl_msg = self._refine_to_latex_tikz(page_image, initial_desc or "")

                            if refined_expl_msg and ("[LaTeX Refinement Interrupted]" in refined_expl_msg or "[LaTeX Model Skipped - Server Busy]" in refined_expl_msg or "[LaTeX Model Skipped - Stop Requested]" in refined_expl_msg):
                                interrupted=True; current_status='pending_vlm'; break
                            elif refined_expl_msg and "[LaTeX Refinement Error:" in refined_expl_msg:
                                page_expl_results.append(f"## Page {page_num}\nLaTeX Refinement Failed: {refined_expl_msg}")
                                vlm_error_occurred_this_file = True # Mark general error
                            else:
                                # Success for this page
                                if refined_latex: page_latex_results.append(f"% Page {page_num}\n{refined_latex}")
                                page_expl_results.append(f"## Page {page_num}\n{refined_expl_msg or '(No explanation provided)'}")

                        # --- After processing all pages (or interruption) ---
                        if not interrupted: # Determine final status only if not stopped/interrupted
                            if vlm_error_occurred_this_file: # If any page had an error (VLM or LaTeX)
                                current_status = 'partial_vlm_error' if (page_latex_results or page_expl_results) else 'error_vlm'
                            else:
                                current_status = 'success'
                        # Else: status was already set to 'pending_vlm' during interruption

                        final_latex_code = "\n\n".join(page_latex_results) if page_latex_results else None
                        final_explanation = "\n\n".join(page_expl_results) if page_expl_results else None

                    else: # PDF conversion failed
                        logger.error(f"Phase 2b Skipping file ID {record_id}: PDF to image conversion failed for '{temp_pdf_to_process}'.")
                        current_status = 'error_conversion'
                        error_count +=1

                # --- Clean up temporary PDF if created from Office file ---
                if is_converted_office_file and temp_pdf_to_process and os.path.exists(temp_pdf_to_process):
                    try: os.remove(temp_pdf_to_process); logger.trace(f"Cleaned up temporary PDF: {temp_pdf_to_process}")
                    except Exception as e: logger.warning(f"Failed to clean up temp PDF '{temp_pdf_to_process}': {e}")

                # --- Update DB Record for this file ---
                try:
                    stmt = update(FileIndex).where(FileIndex.id == record_id).values(
                        vlm_processing_status=current_status,
                        latex_representation=final_latex_code,
                        latex_explanation=final_explanation,
                        last_indexed_db=datetime.datetime.now(datetime.timezone.utc))
                    db_session.execute(stmt); db_session.commit()
                    vlm_processed_count += 1
                    logger.info(f"Phase 2b DB Updated ID {record_id}. Status: {current_status}.")
                except SQLAlchemyError as db_upd_err:
                    logger.error(f"Phase 2b DB Update FAILED ID {record_id}: {db_upd_err}"); db_session.rollback(); error_count += 1

                if not self.stop_event.is_set() and not interrupted: time.sleep(0.5) # Small delay

            if self.stop_event.is_set() or self._wait_if_server_busy() or interrupted: break # Exit outer VLM loop
            if not pending_vlm_files: time.sleep(5) # Pause if queue empty

        # --- End of outer Phase 2b while loop ---
        if interrupted: logger.info("Phase 2b (VLM Analysis) loop exited due to interruption/stop.");
        total_duration = time.monotonic() - last_report_time
        logger.success(f"✅ Finished Phase 2 Cycle. Converted: {converted_count}, VLM Processed: {vlm_processed_count}, Errors: {error_count}. Duration: {total_duration:.2f}s")

    # --- (Existing run method - orchestrates Phase 1 then Phase 2) ---
    def run(self):
        """Main execution loop: Runs Phase 1 scan, then Phase 2 VLM, then waits."""
        logger.info(f"✅ {self.thread_name} started.")
        db: Optional[Session] = None # Initialize db to Optional[Session]

        while not self.stop_event.is_set():
            cycle_start_time = time.monotonic()
            # --- Phase 1: File Scanning ---
            logger.info(f"--- {self.thread_name}: Starting Scan Cycle (Phase 1) ---")
            try:
                db = SessionLocal() # type: ignore
                if db is None: raise Exception("Failed to create DB session for Phase 1")
                root_paths = self._get_root_paths()
                for root in root_paths:
                    if self.stop_event.is_set(): break
                    self._scan_directory(root, db) # Executes Phase 1 logic
            except Exception as e:
                logger.error(f"💥 Unhandled error during Phase 1: {e}", exc_info=True)
            finally:
                if db: db.close(); db = None # Close session after Phase 1
            if self.stop_event.is_set(): break
            phase1_duration = time.monotonic() - cycle_start_time
            logger.info(f"--- {self.thread_name}: Phase 1 Scan Cycle completed in {phase1_duration:.2f} seconds ---")

            # --- Phase 2: VLM/LaTeX Processing ---
            logger.info(f"--- {self.thread_name}: Starting VLM/LaTeX Cycle (Phase 2) ---")
            phase2_start_time = time.monotonic()
            try:
                db = SessionLocal() # type: ignore
                if db is None: raise Exception("Failed to create DB session for Phase 2")
                self._process_pending_vlm_files(db) # Executes Phase 2 logic
            except Exception as e:
                 logger.error(f"💥 Unhandled error during Phase 2: {e}", exc_info=True)
            finally:
                 if db: db.close(); db = None # Close session after Phase 2
            if self.stop_event.is_set(): break
            phase2_duration = time.monotonic() - phase2_start_time
            logger.info(f"--- {self.thread_name}: Phase 2 VLM/LaTeX Cycle completed in {phase2_duration:.2f} seconds ---")

            # --- Wait for Next Cycle ---
            total_cycle_duration = time.monotonic() - cycle_start_time
            wait_time = max(10, SCAN_INTERVAL_SECONDS - total_cycle_duration)
            logger.info(f"{self.thread_name}: Full cycle complete ({total_cycle_duration:.1f}s). Waiting {wait_time:.1f}s...")
            self.stop_event.wait(timeout=wait_time)

        logger.info(f"🛑 {self.thread_name} received stop signal and is exiting.")
    # --- END MODIFIED Run Method ---


def _locked_initialization_task(provider_ref: AIProvider) -> Dict[str, Any]:
    global global_file_index_vectorstore

    task_status: str = "unknown_error_before_processing"
    task_message: str = "Initialization did not complete as expected."
    initialization_succeeded_or_known_empty: bool = False

    overall_start_time: float = time.monotonic()
    logger.info(">>> FileIndex VS Init: Attempting to acquire _file_index_vs_init_lock... <<<")

    with _file_index_vs_init_lock:
        lock_acquired_time: float = time.monotonic()
        logger.info(
            f">>> FileIndex VS Init: ACQUIRED _file_index_vs_init_lock (waited {lock_acquired_time - overall_start_time:.3f}s). <<<")

        if _file_index_vs_initialized_event.is_set():
            if global_file_index_vectorstore is not None:
                logger.info(
                    ">>> FileIndex VS Init: SKIPPING - _file_index_vs_initialized_event is ALREADY SET and VS exists. <<<")
                return {"status": "skipped_already_initialized",
                        "message": "Initialization event was already set and VS exists."}
            else:
                logger.warning(
                    ">>> FileIndex VS Init: Event was set, but global_file_index_vectorstore is None. This indicates a previous partial/failed init. Clearing event and re-attempting full initialization under lock.")
                _file_index_vs_initialized_event.clear()

        db_session: Optional[Session] = None
        try:
            # STAGE 0: Validate Provider and Embeddings
            logger.info(">>> FileIndex VS Init: Stage 0: Validating Provider and Embeddings. <<<")
            if not provider_ref or not provider_ref.embeddings:
                task_status = "error_provider_missing"
                task_message = "AIProvider or its embeddings module is missing. Cannot initialize FileIndex VS."
                logger.error(task_message)
                initialization_succeeded_or_known_empty = True
                global_file_index_vectorstore = None
                if not _file_index_vs_initialized_event.is_set(): _file_index_vs_initialized_event.set()  # Allow app to know init was attempted
                return {"status": task_status, "message": task_message}
            logger.debug("Stage 0 (Provider Check) completed.")

            # STAGE 1: Attempt to load persisted Chroma DB
            current_stage_start_time_s1: float = time.monotonic()
            # These globals() calls assume FILE_INDEX_CHROMA_PERSIST_DIR and FILE_INDEX_CHROMA_COLLECTION_NAME
            # are available from config.py, imported via 'from config import *'
            _persist_dir_to_use = globals().get("FILE_INDEX_CHROMA_PERSIST_DIR")
            _collection_name_to_use = globals().get("FILE_INDEX_CHROMA_COLLECTION_NAME")

            if not _persist_dir_to_use or not _collection_name_to_use:
                task_status = "error_config_missing_chroma_paths"
                task_message = "Chroma DB persist directory or collection name not configured."
                logger.error(task_message)
                initialization_succeeded_or_known_empty = True
                global_file_index_vectorstore = None
                if not _file_index_vs_initialized_event.is_set(): _file_index_vs_initialized_event.set()
                return {"status": task_status, "message": task_message}

            logger.info(
                f">>> FileIndex VS Init: Stage 1: Attempting to load persisted Chroma VS from: '{_persist_dir_to_use}' collection '{_collection_name_to_use}' <<<")

            should_rebuild_from_sql = True
            if os.path.exists(_persist_dir_to_use) and os.path.isdir(_persist_dir_to_use):
                try:
                    if any(fname.endswith(('.sqlite3', '.duckdb', '.parquet')) for fname in
                           os.listdir(_persist_dir_to_use)):
                        logger.info(
                            f"Found persisted Chroma data files in '{_persist_dir_to_use}'. Attempting to load collection '{_collection_name_to_use}'...")
                        loaded_store = Chroma(
                            collection_name=_collection_name_to_use,
                            embedding_function=provider_ref.embeddings,
                            persist_directory=_persist_dir_to_use
                        )
                        collection_count = 0
                        # Check if the collection attribute exists and is not None before calling count()
                        if hasattr(loaded_store,
                                   '_collection') and loaded_store._collection is not None:  # type: ignore
                            collection_count = loaded_store._collection.count()  # type: ignore
                        else:
                            logger.warning(
                                f"Loaded Chroma store from '{_persist_dir_to_use}' appears to have no active collection attribute or it's None. Will attempt rebuild.")

                        if collection_count > 0:
                            global_file_index_vectorstore = loaded_store
                            task_status = "success_loaded_from_persist"
                            task_message = f"Successfully loaded FileIndex VS from '{_persist_dir_to_use}' with {collection_count} items."
                            initialization_succeeded_or_known_empty = True
                            should_rebuild_from_sql = False
                            logger.success(
                                task_message + f" Stage 1 took {time.monotonic() - current_stage_start_time_s1:.3f}s.")
                        else:
                            logger.warning(
                                f"Loaded persisted Chroma from '{_persist_dir_to_use}', but it's EMPTY ({collection_count} items). Will attempt rebuild from SQL.")
                            # should_rebuild_from_sql remains True
                    else:
                        logger.info(
                            f"No Chroma data files (e.g. *.sqlite3, *.duckdb, *.parquet) found in '{_persist_dir_to_use}'. Will build from SQL DB.")
                        # should_rebuild_from_sql remains True
                except Exception as e_persist_load:
                    logger.warning(
                        f"Failed to load/check persisted Chroma from '{_persist_dir_to_use}' or it was empty: {e_persist_load}. Will rebuild.")
                    # should_rebuild_from_sql remains True
            else:  # Persist directory doesn't exist
                logger.info(
                    f"Persist directory '{_persist_dir_to_use}' does not exist or is not a directory. Will build from SQL DB.")
                try:
                    os.makedirs(_persist_dir_to_use, exist_ok=True)  # Attempt to create it
                except OSError as e_mkdir:
                    logger.error(
                        f"Could not create persist directory '{_persist_dir_to_use}': {e_mkdir}.")  # Non-fatal for rebuild

            if should_rebuild_from_sql:
                logger.info(">>> FileIndex VS Init: Proceeding to rebuild from SQL Database. <<<")
                current_stage_start_time_s2: float = time.monotonic()
                db_session = SessionLocal()  # type: ignore
                if not db_session:  # Should not happen if SessionLocal is configured by init_db
                    task_status = "error_db_session_rebuild";
                    task_message = "Failed to create DB session for rebuild phase."
                    logger.error(task_message);
                    initialization_succeeded_or_known_empty = True;
                    global_file_index_vectorstore = None
                    if not _file_index_vs_initialized_event.is_set(): _file_index_vs_initialized_event.set()
                    return {"status": task_status, "message": task_message}

                logger.info(f">>> FileIndex VS Init: Stage 2: Querying SQL DB for records with embeddings... <<<")
                indexed_files = db_session.query(FileIndex).filter(  # type: ignore
                    FileIndex.embedding_json.isnot(None), FileIndex.indexed_content.isnot(None),
                    FileIndex.index_status.in_(['indexed_text', 'success', 'partial_vlm_error'])
                ).all()
                total_records_from_db = len(indexed_files)
                logger.info(
                    f"Stage 2 (DB Query) completed in {time.monotonic() - current_stage_start_time_s2:.3f}s. Found {total_records_from_db} candidates with embeddings in SQL.")

                if not indexed_files:
                    logger.warning("No files with embeddings in SQL DB to rebuild. Creating empty persistent VS.")
                    global_file_index_vectorstore = Chroma(collection_name=_collection_name_to_use,
                                                           embedding_function=provider_ref.embeddings,
                                                           persist_directory=_persist_dir_to_use)
                    # Persistence is handled by chromadb client when persist_directory is set
                    task_status = "success_empty_db_rebuild";
                    task_message = "Init (rebuild): No files with embeddings in SQL. Created empty VS.";
                    initialization_succeeded_or_known_empty = True
                else:
                    current_stage_start_time_s3: float = time.monotonic()
                    logger.info(
                        f">>> FileIndex VS Init: Stage 3: Processing {total_records_from_db} DB records for Chroma... <<<")
                    texts_for_vs, embeddings_for_vs, metadatas_for_vs, ids_for_vs = [], [], [], []
                    processed_records_for_chroma = 0
                    report_interval = max(1, total_records_from_db // 20) if total_records_from_db > 0 else 1;
                    report_interval = min(report_interval, 2000)
                    time_last_report_s3, recs_since_report_s3 = time.monotonic(), 0

                    for record_idx, db_record in enumerate(indexed_files):
                        rec_id_val = getattr(db_record, 'id', f'unk_idx_{record_idx}')
                        try:
                            emb_json_str = getattr(db_record, 'embedding_json', None)
                            content_str = getattr(db_record, 'indexed_content', None)
                            if not emb_json_str or not content_str or not content_str.strip(): continue
                            vector = json.loads(emb_json_str)
                            if not (isinstance(vector, list) and vector and all(
                                isinstance(val, (float, int)) for val in vector)): continue

                            texts_for_vs.append(content_str)
                            embeddings_for_vs.append([float(val) for val in vector])

                            source_val = getattr(db_record, 'file_path')
                            fname_val = getattr(db_record, 'file_name')
                            lmod_val = getattr(db_record, 'last_modified_os')
                            idx_status_val = getattr(db_record, 'index_status')
                            mtype_val = getattr(db_record, 'mime_type')

                            metadatas_for_vs.append({
                                "source": source_val if source_val is not None else "UnknownPath",
                                "file_id": int(str(rec_id_val)) if str(rec_id_val).isdigit() else -1,
                                "file_name": fname_val if fname_val is not None else "UnknownFile",
                                "last_modified": str(lmod_val) if lmod_val is not None else "N/A",
                                "index_status": idx_status_val if idx_status_val is not None else "UnknownStatus",
                                "mime_type": mtype_val if mtype_val is not None else "UnknownMIME"
                            })
                            ids_for_vs.append(f"file_{rec_id_val}")
                            processed_records_for_chroma += 1
                        except Exception as e_prep:
                            logger.warning(f"Skipping record ID {rec_id_val} due to prep error: {e_prep}")

                        recs_since_report_s3 += 1
                        if recs_since_report_s3 >= report_interval or (record_idx + 1) == total_records_from_db:
                            now_s3 = time.monotonic();
                            batch_dur_s3 = now_s3 - time_last_report_s3;
                            loop_dur_s3 = now_s3 - current_stage_start_time_s3
                            rate_b = recs_since_report_s3 / batch_dur_s3 if batch_dur_s3 > 0 else float('inf')
                            rate_t = processed_records_for_chroma / loop_dur_s3 if loop_dur_s3 > 0 else float('inf')
                            prog_pct = (
                                        processed_records_for_chroma / total_records_from_db * 100) if total_records_from_db > 0 else 0
                            logger.info(
                                f"  Prep {processed_records_for_chroma}/{total_records_from_db} for VS ({prog_pct:.1f}%). Batch: {recs_since_report_s3} in {batch_dur_s3:.2f}s (~{rate_b:.0f}r/s). Loop: {loop_dur_s3:.2f}s (~{rate_t:.0f}r/s avg).")
                            recs_since_report_s3 = 0;
                            time_last_report_s3 = now_s3

                    logger.info(
                        f"Stage 3 (DB Record Processing) completed. Prepared {processed_records_for_chroma} valid items.")

                    if processed_records_for_chroma > 0:
                        current_stage_start_time_s4: float = time.monotonic()
                        logger.info(
                            f">>> FileIndex VS Init: Stage 4: Populating Chroma with {processed_records_for_chroma} items (pre-computed embeddings)... <<<")

                        temp_chroma_store = Chroma(collection_name=_collection_name_to_use,
                                                   embedding_function=provider_ref.embeddings,
                                                   persist_directory=_persist_dir_to_use)

                        batch_size = 500
                        num_batches = (processed_records_for_chroma + batch_size - 1) // batch_size
                        logger.info(
                            f"Adding items to Chroma collection in {num_batches} batches of size up to {batch_size}.")
                        all_batches_ok = True
                        for i in range(num_batches):
                            start_idx, end_idx = i * batch_size, min((i + 1) * batch_size, processed_records_for_chroma)
                            b_texts, b_embs, b_metas, b_ids = texts_for_vs[start_idx:end_idx], embeddings_for_vs[
                                                                                               start_idx:end_idx], metadatas_for_vs[
                                                                                                                   start_idx:end_idx], ids_for_vs[
                                                                                                                                       start_idx:end_idx]

                            logger.info(
                                f"  Adding batch {i + 1}/{num_batches} ({len(b_texts)} items) to Chroma collection...")
                            batch_add_start = time.monotonic()
                            try:
                                if not temp_chroma_store._collection: raise RuntimeError(
                                    "Chroma collection not initialized.")  # type: ignore
                                temp_chroma_store._collection.add(ids=b_ids, embeddings=b_embs, metadatas=b_metas,
                                                                  documents=b_texts)  # type: ignore
                                logger.info(
                                    f"  Batch {i + 1} added successfully in {time.monotonic() - batch_add_start:.2f}s.")
                            except Exception as e_batch_add_exc:
                                logger.error(f"Error adding Chroma batch {i + 1}/{num_batches}: {e_batch_add_exc}")
                                all_batches_ok = False
                                task_status = "error_chroma_batch_add_rebuild"
                                task_message = (
                                                   task_message if task_message != "Initialization did not complete as expected." else "") + f" Error on Chroma rebuild batch {i + 1}: {str(e_batch_add_exc)[:200]}."

                        if _persist_dir_to_use:
                            logger.info(
                                f"Chroma store operations complete. Data should be persisted to '{_persist_dir_to_use}' by the underlying chromadb client.")

                        global_file_index_vectorstore = temp_chroma_store
                        if all_batches_ok: task_status = "success_rebuilt_from_sql_batched"; task_message = f"FileIndex VS rebuilt from SQL with {processed_records_for_chroma} items (batched)."
                        initialization_succeeded_or_known_empty = True
                        logger.success(
                            task_message + f" Stage 4 took {time.monotonic() - current_stage_start_time_s4:.3f}s.")
                    else:
                        logger.warning("No valid data with embeddings from SQL. Creating empty persistent VS.");
                        global_file_index_vectorstore = Chroma(collection_name=_collection_name_to_use,
                                                               embedding_function=provider_ref.embeddings,
                                                               persist_directory=_persist_dir_to_use)
                        task_status = "success_no_valid_data_rebuild";
                        task_message = "Init (rebuild): No valid data from SQL. Created empty VS.";
                        initialization_succeeded_or_known_empty = True
            # End of should_rebuild_from_sql block

        except Exception as e_init_critical:
            task_status = "critical_error_overall_init";
            task_message = f"CRITICAL ERROR FileIndex VS init: {e_init_critical}"
            logger.error(task_message);
            logger.exception("FileIndex VS Init Traceback (critical):")
            global_file_index_vectorstore = None;
            initialization_succeeded_or_known_empty = False

        finally:
            if db_session:
                try:
                    db_session.close()
                except Exception as e_close:
                    logger.warning(f"Error closing DB session in FileIndex VS init: {e_close}")

            if initialization_succeeded_or_known_empty:
                if not _file_index_vs_initialized_event.is_set():
                    _file_index_vs_initialized_event.set()
                    logger.info(
                        f">>> SET _file_index_vs_initialized_event. Final Status: {task_status}. Total Task Time: {time.monotonic() - overall_start_time:.3f}s <<<")
            else:
                logger.error(
                    f">>> _file_index_vs_initialized_event NOT SET due to critical error. Status: {task_status}. Total: {time.monotonic() - overall_start_time:.3f}s <<<")

            logger.info(
                f">>> FileIndex VS Init: Exiting _file_index_vs_init_lock context. Final Status: '{task_status}', Message: \"{task_message}\" <<<")
            return {"status": task_status, "message": task_message}

    logger.error(
        ">>> FileIndex VS Init: Exited WITHOUT properly returning from 'with _file_index_vs_init_lock' block. <<<")
    return {"status": "error_unexpected_exit_from_lock", "message": "Task exited lock context unexpectedly."}


async def initialize_global_file_index_vectorstore(provider: AIProvider):
    global global_file_index_vectorstore  # Still need this for the outer function's scope
    logger.info(">>> Entered initialize_global_file_index_vectorstore (async version) <<<")

    if not provider:
        logger.error("  INIT_VS_ERROR (Outer): AIProvider instance is None! Cannot initialize file index vector store.")
        return
    if not provider.embeddings:
        logger.error(
            "  INIT_VS_ERROR (Outer): AIProvider.embeddings is None! Cannot initialize file index vector store.")
        return
    logger.info(f"  INIT_VS_INFO (Outer): Using AIProvider embeddings of type: {type(provider.embeddings)}")

    if _file_index_vs_initialized_event.is_set():
        logger.info(">>> SKIPPING FileIndex VS Init (Outer): _file_index_vs_initialized_event is ALREADY SET. <<<")
        return

    try:
        # Pass the provider instance to the threaded function _locked_initialization_task
        result_dict = await asyncio.to_thread(_locked_initialization_task, provider)

        status_from_thread = result_dict.get("status", "unknown_thread_exit_status")
        message_from_thread = result_dict.get("message", "No message from thread.")
        logger.info(
            f"Locked initialization task for FileIndex VS completed with status: '{status_from_thread}'. Message: '{message_from_thread}'")

        # The event _file_index_vs_initialized_event is set (or not set) inside _locked_initialization_task's finally block.
        # No need to set it again here unless _locked_initialization_task failed to be scheduled by to_thread.

    except Exception as e_thread_task:
        logger.error(f"Exception running _locked_initialization_task via asyncio.to_thread: {e_thread_task}")
        logger.exception("FileIndex VS _locked_initialization_task Thread Execution Traceback:")
        # If to_thread itself fails, the event won't be set by the inner task.
        # We might choose to set it here to prevent hangs, indicating init was attempted but failed at thread level.
        if not _file_index_vs_initialized_event.is_set():
            logger.warning(
                ">>> Setting _file_index_vs_initialized_event due to THREAD EXECUTION ERROR (VS is None). <<<")
            _file_index_vs_initialized_event.set()
            global_file_index_vectorstore = None  # Ensure it's None

    logger.info(
        f">>> EXITED initialize_global_file_index_vectorstore. Event set: {_file_index_vs_initialized_event.is_set()} Global VS is None: {global_file_index_vectorstore is None}<<<")


def get_global_file_index_vectorstore(timeout_seconds=60) -> Optional[Chroma]:
    if not _file_index_vs_initialized_event.is_set():
        logger.warning(f"Global file index VS not ready, waiting up to {timeout_seconds}s for initialization event...")
        event_was_set = _file_index_vs_initialized_event.wait(timeout=timeout_seconds)
        if not event_was_set:
            logger.error(f"Timeout waiting for file index VS initialization after {timeout_seconds}s.")
            return None
        logger.info("File index VS initialization event received after waiting.")

    # Event is now set (or was already set)
    if global_file_index_vectorstore is None:
        logger.warning(
            "File index VS event is set, but global_file_index_vectorstore is still None. Data might be empty or init issue.")
    return global_file_index_vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="File Indexer Test CLI")
    parser.add_argument("--test", action="store_true", help="Run test initialization and optional search.")
    parser.add_argument("--search_query", type=str, default=None, help="Query string to search the file index vector store.")
    cli_args = parser.parse_args()

    if cli_args.test:
        logger.remove() # Remove default loguru handler
        logger.add(sys.stderr, level="DEBUG") # Add a new one with DEBUG level for testing
        logger.info("--- File Indexer Test Mode ---")

        # 1. Initialize Database (critical for SessionLocal and schema)
        try:
            logger.info("Initializing database via database.init_db()...")
            init_db() # This function from database.py handles Alembic migrations etc.
            logger.info("Database initialization complete.")
        except Exception as e_db_init:
            logger.error(f"Failed to initialize database: {e_db_init}")
            sys.exit(1)

        # 2. Initialize AIProvider (needed for embeddings)
        #    Ensure PROVIDER is set in your environment or config.py for this to work.
        #    The AIProvider constructor might try to load all models, which could be slow.
        #    For testing search, only provider.embeddings is strictly needed by initialize_global_file_index_vectorstore.
        test_ai_provider: Optional[AIProvider] = None
        try:
            logger.info(f"Initializing AIProvider (Provider: {PROVIDER})...") # PROVIDER from config.py
            test_ai_provider = AIProvider(PROVIDER)
            if not test_ai_provider.embeddings:
                raise ValueError("AIProvider initialized, but embeddings are not available.")
            logger.info("AIProvider initialized successfully for embeddings.")
        except Exception as e_ai_provider:
            logger.error(f"Failed to initialize AIProvider: {e_ai_provider}")
            sys.exit(1)

        # 3. Initialize Global File Index Vector Store (runs the async function)
        try:
            logger.info("Initializing global file index vector store (async task)...")
            asyncio.run(initialize_global_file_index_vectorstore(test_ai_provider)) # Run the async init
            logger.info("Global file index vector store initialization process completed.")
        except Exception as e_vs_init:
            logger.error(f"Error during file index vector store initialization: {e_vs_init}")
            sys.exit(1)

        # 4. Get the vector store instance
        file_vs = get_global_file_index_vectorstore(timeout_seconds=10) # Wait a bit if still initializing
        if file_vs:
            logger.success("Successfully retrieved global file index vector store instance.")
            try:
                collection_count = file_vs._collection.count() # type: ignore
                logger.info(f"File index Chroma collection contains {collection_count} items.")
            except Exception as e_count:
                logger.warning(f"Could not get count from Chroma collection: {e_count}")

            # 5. Perform Search if query provided
            if cli_args.search_query:
                query = cli_args.search_query
                logger.info(f"Performing search with query: '{query}'")
                try:
                    # Chroma's similarity_search takes the query string directly
                    # It will use the embedding_function configured during Chroma init.
                    search_results = file_vs.similarity_search_with_relevance_scores(query, k=5) # Get top 5 with scores
                    if search_results:
                        logger.info(f"Found {len(search_results)} results:")
                        for i, (doc, score) in enumerate(search_results):
                            print(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
                            print(f"  Source: {doc.metadata.get('source', 'N/A')}")
                            print(f"  File Name: {doc.metadata.get('file_name', 'N/A')}")
                            print(f"  Content Snippet:\n    {doc.page_content[:300].replace(os.linesep, ' ')}...")
                    else:
                        logger.info("No results found for the query.")
                except Exception as e_search:
                    logger.error(f"Error during vector search: {e_search}")
            else:
                logger.info("No search query provided. Skipping search.")
        else:
            logger.error("Failed to get global file index vector store instance after initialization attempt.")

        logger.info("--- File Indexer Test Mode Finished ---")
    else:
        print("Run with --test to initialize and optionally --search_query \"your query\" to search.")

