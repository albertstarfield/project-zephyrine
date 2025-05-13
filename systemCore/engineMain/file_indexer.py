# file_indexer.py
import os
import re
import sys
import time
import threading
import mimetypes
import datetime
from sqlalchemy.orm import Session
from loguru import logger
import hashlib # <<< Import hashlib for MD5
from typing import Optional, Tuple, List, Any # <<< ADD Optional and List HERE
import json
from ai_provider import AIProvider
from config import * # Ensure this includes the SQLite DATABASE_URL and all prompts/models
import base64
from io import BytesIO
from PIL import Image # Requires Pillow: pip install Pillow
#from langchain_community.vectorstores import Chroma # Add Chroma import
from langchain_chroma import Chroma





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
    from database import SessionLocal, FileIndex, add_interaction # Import add_interaction if logging indexer status
    from sqlalchemy import update, select
    from sqlalchemy.exc import SQLAlchemyError
except ImportError:
    logger.critical("❌ Failed to import database components in file_indexer.py. Indexer cannot run.")
    # Define dummy classes/functions to allow loading but prevent execution
    class SessionLocal:
        def __call__(self): return None # type: ignore
    class FileIndex: pass
    def add_interaction(*args, **kwargs): pass
    SQLAlchemyError = Exception
    # Exit if DB cannot be imported
    sys.exit("Indexer failed: Database components missing.")



# --- NEW: Import the custom lock ---

from priority_lock import ELP0, ELP1 # Ensure these are imported
interruption_error_marker = "Worker task interrupted by higher priority request" # Define consistently

# --- NEW: Module-level lock and event for initialization ---
_file_index_vs_init_lock = threading.Lock()
_file_index_vs_initialized_event = threading.Event() # To

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
             logger.info(f"🟢 {self.thread_name}: Server is free, resuming indexing.")
        return False # Indicate processing can continue

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
    
    def _process_file_phase1(self, file_path: str, db_session: Session):
        """
        Phase 1: Gets metadata, hash. Extracts text ONLY for non-Office/non-PDF files.
        Marks Office/PDF files for later processing. Embeds extracted text using ELP0.
        Handles ELP0 interruptions during embedding.
        """
        # Check/Wait at the START of processing a file
        if self._wait_if_server_busy(log_wait=False): # Don't log busy waits per file
            logger.trace(f"Yielding file processing due to busy server: {file_path}")
            return # Skip this file entirely for now if server busy at start
        if self.stop_event.is_set():
            return # Exit if stop requested

        logger.trace(f"Phase 1 Processing: {file_path}")
        # Initialize variables
        file_name = os.path.basename(file_path)
        status = 'pending' # Initial status
        content: Optional[str] = None
        error_message: Optional[str] = None
        embedding_json_str: Optional[str] = None
        current_md5_hash: Optional[str] = None
        vlm_processing_status_to_set: Optional[str] = None
        should_update_db = True # Assume update needed unless skipped
        existing_record: Optional[FileIndex] = None
        size_bytes: int = -1 # Default size
        mtime_os: Optional[datetime.datetime] = None
        mime_type: Optional[str] = None

        try:
            # 1. Get File Metadata
            size_bytes_stat, mtime_os_stat, mime_type_stat = self._get_file_metadata(file_path)
            if size_bytes_stat is None and mtime_os_stat is None:
                 # Likely permission error or file vanished after os.walk found it
                 status = 'error_permission'
                 error_message = "Permission denied or file vanished during stat."
            else:
                 # Store valid metadata, use -1 for unknown size
                 size_bytes = size_bytes_stat if size_bytes_stat is not None else -1
                 mtime_os = mtime_os_stat
                 mime_type = mime_type_stat

            # 2. Calculate Current MD5 Hash (if possible and no prior error)
            if status == 'pending':
                try:
                    current_md5_hash = self._calculate_md5(file_path, size_bytes)
                    # If hashing returns None for a file that *should* be hashable, mark as error
                    if current_md5_hash is None and size_bytes >= 0 and size_bytes <= MAX_HASH_FILE_SIZE_BYTES:
                        status = 'error_hash'
                        error_message = "Failed to calculate MD5 hash."
                        logger.warning(f"MD5 hash calculation failed for: {file_path}")
                except PermissionError:
                    # Let the outer handler catch PermissionError from hashing
                    raise
                except Exception as hash_err:
                     # Catch unexpected errors during hashing call
                     status = 'error_hash'
                     error_message = f"Error during hashing: {hash_err}"
                     current_md5_hash = None
                     logger.error(f"Unexpected error calling _calculate_md5 for {file_path}: {hash_err}")

            # 3. Check Database using MD5 (if possible and no prior error)
            if status == 'pending':
                try:
                    existing_record = db_session.query(FileIndex).filter(FileIndex.file_path == file_path).first()
                    if existing_record:
                        # Determine if file needs re-processing based on hash/timestamp
                        hashes_match = (current_md5_hash is not None and existing_record.md5_hash == current_md5_hash)
                        # Check timestamp only if hashing was skipped due to size on *both* attempts
                        large_file_timestamps_match = (
                            current_md5_hash is None and size_bytes > MAX_HASH_FILE_SIZE_BYTES and
                            existing_record.md5_hash is None and
                            mtime_os and existing_record.last_modified_os and
                            mtime_os <= existing_record.last_modified_os
                        )

                        if hashes_match or large_file_timestamps_match:
                            logger.trace(f"Skipping DB update (Content Unchanged): {file_path}")
                            should_update_db = False # File hasn't changed significantly
                        else:
                            logger.trace(f"Needs update (Changed Hash/Timestamp/New): {file_path}")
                            # Keep status 'pending' for content processing
                except SQLAlchemyError as db_check_err:
                     logger.error(f"DB check failed for {file_path}: {db_check_err}")
                     status = 'error_read' # Treat DB error as a read error for the file state
                     error_message = f"Database check failed: {db_check_err}"
                     should_update_db = False # Don't proceed if DB check fails

            # 4. Process Content (Extract/Embed/Mark) if Update Needed & No Prior Error
            if should_update_db and status == 'pending':
                file_ext = os.path.splitext(file_name)[1].lower()
                # Determine file type and if it's a target for VLM processing later
                is_vlm_target = file_ext in VLM_TARGET_EXTENSIONS # Check against VLM target types
                is_text_target = file_ext in TEXT_EXTENSIONS or (mime_type and mime_type.startswith('text/'))

                if is_vlm_target:
                    # Mark for Phase 2, do not extract text or embed now
                    status = 'indexed_meta' # Only index metadata in this phase
                    content = None
                    should_embed = False
                    if self.vlm_model:
                         # Set appropriate pending status based on type (PDF direct, Office needs conversion)
                         vlm_processing_status_to_set = 'pending_vlm' if file_ext == '.pdf' else 'pending_conversion'
                    else:
                         logger.trace(f"Skipping VLM marking (VLM model unavailable): {file_path}")
                         # Inherit status if record exists, otherwise None
                         vlm_processing_status_to_set = existing_record.vlm_processing_status if existing_record else None

                elif is_text_target:
                    # Handle standard text files
                    if size_bytes <= MAX_TEXT_FILE_SIZE_BYTES:
                        content = self._extract_text(file_path, size_bytes)
                        if content is None: # Check if extraction failed
                             status = 'error_read'; error_message = "Text extraction failed"
                             should_embed = False
                        else:
                             should_embed = True # Text extracted, attempt embedding
                    else:
                        # Text file too large
                        status = 'skipped_size'
                        error_message = f"Skipped content (exceeds {MAX_TEXT_FILE_SIZE_MB} MB)"
                        content = None
                        should_embed = False
                else:
                     # File is not VLM target and not recognized text, index metadata only
                     status = 'indexed_meta'
                     content = None
                     should_embed = False
                     logger.trace(f"Indexing metadata only for {file_path} (Type: {mime_type or file_ext})")

                # --- Generate Embedding (if text extracted and should_embed is True) ---
                if status == 'pending' and content is not None and should_embed: # Check content explicitly not None
                    if self.embedding_model:
                        logger.debug(f"🧠 Phase 1 Generating embedding for: {file_path} (PRIORITY: ELP0)")
                        try:
                            # Call embed_query with ELP0 priority
                            vector = self.embedding_model.embed_query(content, priority=ELP0)
                            embedding_json_str = json.dumps(vector)
                            status = 'indexed_text' # Mark as successfully indexed with text
                            logger.trace(f"Embedding generated successfully for {file_path}")
                        except Exception as emb_err:
                            # Check if the error is the specific interruption marker
                            if interruption_error_marker in str(emb_err):
                                logger.warning(f"🚦 Embedding for {file_path} INTERRUPTED by ELP1. Resetting to pending.")
                                status = 'pending' # Reset status to retry later
                                error_message = "Embedding interrupted by higher priority task"
                                embedding_json_str = None
                                content = None # Discard potentially partial content if interrupted? Optional.
                                # Ensure DB update happens to save the 'pending' status
                                should_update_db = True
                            else:
                                # Handle other embedding errors
                                logger.error(f"Embedding generation failed for {file_path}: {emb_err}")
                                status = 'error_embedding'
                                error_message = f"Embedding failed: {emb_err}"
                                embedding_json_str = None
                                # Keep content if it was extracted, just mark embedding error
                    else:
                        # No embedding model, but text was extracted
                        status = 'indexed_text'
                        logger.warning(f"Skipping embedding (no model) for: {file_path}")
                elif status == 'pending':
                     # If status is still pending here, it means conditions for embedding weren't met
                     # or it wasn't a text file/VLM target initially. Set final status.
                     status = 'indexed_meta' if not is_vlm_target else status # Keep pending_vlm/conversion if set

        except PermissionError:
             # Catch permission errors occurring anywhere during the process
             status = 'error_permission'
             error_message = "Permission denied during file processing."
             content = None; embedding_json_str = None; current_md5_hash = None;
             should_update_db = True # Ensure we update the DB record with the error status
             logger.warning(f"Permission error processing {file_path}")
        except Exception as proc_err:
            # Catch any other unexpected errors during processing
            status = 'error_read' # Generic read/processing error
            error_message = f"Processing error: {proc_err}"
            content = None; embedding_json_str = None; current_md5_hash = None;
            should_update_db = True # Update DB with error
            logger.error(f"Error processing file {file_path}: {proc_err}")
            logger.exception("File Processing Phase 1 Traceback:")

        # --- 5. Update Database Record ---
        if should_update_db:
            # Determine final status if logic above left it as 'pending' unexpectedly
            if status == 'pending':
                 logger.warning(f"File {file_path} reached DB update with pending status. Defaulting status to indexed_meta.")
                 status = 'indexed_meta'

            logger.debug(f"Phase 1 DB Update: {file_path} -> Status: {status}, Hash: {current_md5_hash}, VLM Marked: {vlm_processing_status_to_set}")

            # Prepare data, ensuring None is used for fields not set
            record_data = {
                'file_name': file_name,
                'size_bytes': size_bytes, # Already defaulted to -1 if needed
                'mime_type': mime_type,
                'last_modified_os': mtime_os,
                'index_status': status,
                'indexed_content': (content[:50000] + '...[truncated]') if content and len(content) > 50000 else content, # Truncate long content
                'embedding_json': embedding_json_str,
                'md5_hash': current_md5_hash,
                'processing_error': error_message[:1000] if error_message else None, # Limit error message length
                'last_indexed_db': datetime.datetime.now(datetime.timezone.utc),
                'vlm_processing_status': vlm_processing_status_to_set, # Explicitly set from logic above
                'latex_representation': None, # Always None/cleared in Phase 1 if VLM is pending/marked
                'latex_explanation': None,    # Always None/cleared in Phase 1 if VLM is pending/marked
            }

            # If the file existed before, and VLM processing was NOT marked as pending
            # this time (e.g., status became 'error_read' or 'indexed_meta'),
            # we might want to preserve the *old* VLM status and results if they existed.
            if existing_record and record_data.get('vlm_processing_status') is None:
                  record_data['vlm_processing_status'] = existing_record.vlm_processing_status
                  # Only keep old LaTeX if VLM status isn't being reset to pending
                  if existing_record.vlm_processing_status not in ['pending_vlm', 'pending_conversion']:
                      record_data['latex_representation'] = existing_record.latex_representation
                      record_data['latex_explanation'] = existing_record.latex_explanation

            try:
                if existing_record:
                    # Update existing record
                    # Filter record_data to only valid columns before update
                    valid_columns = {c.name for c in FileIndex.__table__.columns}
                    update_values = {k: v for k, v in record_data.items() if k in valid_columns}
                    stmt = update(FileIndex).where(FileIndex.id == existing_record.id).values(**update_values)
                    db_session.execute(stmt)
                    logger.trace(f"Phase 1 Updated existing DB record ID {existing_record.id} for {file_path}")
                else:
                    # Insert new record
                    # Filter record_data to only valid columns before insert
                    valid_columns = {c.name for c in FileIndex.__table__.columns}
                    insert_values = {k: v for k, v in record_data.items() if k in valid_columns}
                    new_record = FileIndex(file_path=file_path, **insert_values)
                    db_session.add(new_record)
                    # Commit immediately to get ID if needed (optional)
                    # db_session.flush()
                    # logger.trace(f"Phase 1 Inserted new DB record ID {new_record.id} for {file_path}")

                db_session.commit() # Commit the transaction for this file

            except SQLAlchemyError as db_err:
                 logger.error(f"Phase 1 DB update/insert FAILED for {file_path}: {db_err}")
                 db_session.rollback() # Rollback changes for this specific file on error
                 logger.exception("DB Update/Insert Traceback:")

        elif existing_record:
             # Case: MD5 matched (should_update_db=False), but maybe VLM status needs updating
             # This logic path might be redundant if VLM marking happens only when should_update_db is True.
             # Check if the *calculated* VLM status differs from the existing one.
             calculated_vlm_status = None # Recalculate what it *should* be based on type
             file_ext = os.path.splitext(file_name)[1].lower()
             is_vlm_target = file_ext in VLM_TARGET_EXTENSIONS
             if is_vlm_target and self.vlm_model:
                  calculated_vlm_status = 'pending_vlm' if file_ext == '.pdf' else 'pending_conversion'

             if calculated_vlm_status is not None and existing_record.vlm_processing_status != calculated_vlm_status:
                 logger.debug(f"Updating only VLM status to '{calculated_vlm_status}' for {file_path} (Content Unchanged)")
                 try:
                     update_data = {'vlm_processing_status': calculated_vlm_status}
                     # Reset LaTeX fields if marking as pending
                     if calculated_vlm_status in ['pending_vlm', 'pending_conversion']:
                          update_data['latex_representation'] = None
                          update_data['latex_explanation'] = None
                     stmt = update(FileIndex).where(FileIndex.id == existing_record.id).values(**update_data)
                     db_session.execute(stmt)
                     db_session.commit()
                 except SQLAlchemyError as db_vlm_err:
                     logger.error(f"DB VLM status only update FAILED for {file_path}: {db_vlm_err}")
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

        # --- Sub-Phase 2a: Convert Office Files ---
        logger.info("--- Phase 2a: Checking for Office files pending conversion ---")
        while not self.stop_event.is_set():
            files_to_convert: List[FileIndex] = []
            try:
                files_to_convert = db_session.query(FileIndex).filter(
                    FileIndex.vlm_processing_status == 'pending_conversion'
                ).limit(5).all() # Convert in small batches
            except SQLAlchemyError as db_err: logger.error(f"Phase 2a DB query error: {db_err}"); time.sleep(30); continue

            if not files_to_convert: logger.info("--- Phase 2a: No more files pending conversion found. ---"); break

            logger.info(f"Phase 2a: Found {len(files_to_convert)} Office file(s) to convert...")
            for record in files_to_convert:
                if self.stop_event.is_set(): logger.info("Phase 2a conversion interrupted."); break
                if self._wait_if_server_busy(): break # Stop batch if busy

                input_path = record.file_path
                record_id = record.id
                temp_pdf_path = self._convert_office_to_pdf(input_path) # Call new helper

                new_status = 'error_conversion' # Assume failure
                if temp_pdf_path:
                    new_status = 'pending_vlm' # Mark for VLM processing in next sub-phase
                    converted_count += 1
                    # --- Optionally process the temp PDF immediately ---
                    # self._process_single_pdf_for_vlm(db_session, record, temp_pdf_path)
                    # os.remove(temp_pdf_path) # Clean up temp PDF
                    # --- OR ---
                    # Just update status and let the next loop handle it
                    logger.info(f"Conversion successful for ID {record_id}, marked as 'pending_vlm'. Temp PDF: {temp_pdf_path}")
                    # We need to decide whether to process now or later. Let's mark and process later for simplicity.
                    # Clean up the temp PDF path *now* if we don't process immediately
                    # This is complex - maybe conversion should output to a specific cache dir?
                    # For now, let's assume we need the temp path later, which is risky.
                    # SAFER: Process immediately. Let's adjust the loop below.

                # Update DB status after conversion attempt
                try:
                    stmt = update(FileIndex).where(FileIndex.id == record_id).values(
                        vlm_processing_status=new_status,
                        # Clear LaTeX fields if conversion failed or pending VLM
                        latex_representation=None if new_status != 'success' else record.latex_representation,
                        latex_explanation=None if new_status != 'success' else record.latex_explanation,
                        last_indexed_db=datetime.datetime.now(datetime.timezone.utc)
                    )
                    db_session.execute(stmt); db_session.commit()
                except SQLAlchemyError as db_conv_err:
                     logger.error(f"Phase 2a DB update failed for ID {record_id}: {db_conv_err}"); db_session.rollback()
                     error_count += 1

                if not self.stop_event.is_set(): time.sleep(1.0) # Longer delay after conversion attempt

            if self.stop_event.is_set() or self._wait_if_server_busy(): break # Exit outer loop if needed

        # --- Sub-Phase 2b: Process Pending VLM Files (PDFs and Successfully Converted) ---
        logger.info("--- Phase 2b: Checking for files pending VLM analysis ---")
        while not self.stop_event.is_set():
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
                if self.stop_event.is_set(): logger.info("Phase 2b VLM analysis interrupted."); break
                if self._wait_if_server_busy(): break # Stop batch

                file_path = record.file_path # This is the ORIGINAL path (PDF or Office)
                record_id = record.id
                logger.info(f"Phase 2b Processing file ID {record_id}: {file_path}")
                current_status = 'processing'; final_latex_code = None; final_explanation = None; page_errors = 0; num_pages = 0
                temp_pdf_to_process = None
                is_converted = False

                # Check if this is an Office file that needs the temp PDF path
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in OFFICE_EXTENSIONS:
                     # Problem: We didn't store the temp PDF path from Phase 2a!
                     # Need to re-convert or change Phase 2a to process immediately.
                     # Let's re-convert for now, acknowledging inefficiency.
                     logger.warning(f"Phase 2b: Re-converting Office file ID {record_id} to temporary PDF for VLM analysis.")
                     temp_pdf_to_process = self._convert_office_to_pdf(file_path)
                     if not temp_pdf_to_process:
                         logger.error(f"Phase 2b: Re-conversion failed for {file_path}. Skipping VLM."); current_status = 'error_conversion'; error_count += 1
                         # Go straight to DB update below loop
                     else:
                         is_converted = True # Mark that we need to delete temp file later
                         logger.info(f"Phase 2b: Re-converted to {temp_pdf_to_process}")
                elif file_ext == '.pdf':
                     temp_pdf_to_process = file_path # Process the original PDF path
                else:
                     logger.error(f"Phase 2b: File ID {record_id} has status 'pending_vlm' but is not PDF or known Office type ({file_ext}). Skipping."); current_status = 'error_type'; error_count += 1

                # --- Process the PDF (original or temporary) ---
                if temp_pdf_to_process:
                    images = self._convert_pdf_to_images(temp_pdf_to_process)
                    if images:
                        num_pages = len(images); page_latex_results = []; page_expl_results = []
                        logger.info(f"  📄 Processing {num_pages} pages from PDF for file ID {record_id}.")

                        for i, page_image in enumerate(images):
                            page_num = i + 1
                            if self.stop_event.is_set(): current_status = 'pending_vlm'; interrupted=True; break
                            logger.info(f"  🧠 Phase 2b: Analyzing page {page_num}/{num_pages} for file ID {record_id} (VLM/LaTeX - ELP0)...")

                            # --- Step 1: Get Initial Description ---
                            initial_desc, vlm_err_msg = self._get_initial_vlm_description(page_image)

                            if vlm_err_msg: # Check for errors or interruption from VLM step
                                page_expl_results.append(f"## Page {page_num}\nInitial Analysis Failed: {vlm_err_msg}")
                                if "[VLM Interrupted]" in vlm_err_msg: interrupted=True; current_status='pending_vlm'; break
                                vlm_error_occurred = True # Mark general error
                                continue # Skip refinement if initial failed

                            # --- Step 2: Refine to LaTeX/TikZ (if initial desc obtained) ---
                            refined_latex, refined_expl_msg = self._refine_to_latex_tikz(page_image, initial_desc or "")

                            if refined_expl_msg and "[LaTeX Refinement Interrupted]" in refined_expl_msg:
                                interrupted=True; current_status='pending_vlm'; break
                            elif refined_expl_msg and "[LaTeX Refinement Error:" in refined_expl_msg:
                                page_expl_results.append(f"## Page {page_num}\nLaTeX Refinement Failed: {refined_expl_msg}")
                                vlm_error_occurred = True # Mark general error
                            else:
                                # Success for this page
                                if refined_latex: page_latex_results.append(f"% Page {page_num}\n{refined_latex}")
                                page_expl_results.append(f"## Page {page_num}\n{refined_expl_msg or '(No explanation provided)'}")

                        # --- After processing all pages (or interruption) ---
                        if not interrupted: # Determine final status only if not stopped/interrupted
                            if vlm_error_occurred: # If any page had an error (VLM or LaTeX)
                                current_status = 'partial_vlm_error' if (page_latex_results or page_expl_results) else 'error_vlm'
                            else:
                                current_status = 'success'
                        # Else: status was already set to 'pending_vlm' during interruption

                        final_latex_code = "\n\n".join(page_latex_results) if page_latex_results else None
                        final_explanation = "\n\n".join(page_expl_results) if page_expl_results else None

                    else: # PDF conversion failed
                        logger.error(f"Phase 2b Skipping file ID {record_id}: PDF conversion failed for '{temp_pdf_to_process}'.")
                        current_status = 'error_conversion'

                # --- Clean up temporary PDF if created ---
                if is_converted and temp_pdf_to_process and os.path.exists(temp_pdf_to_process):
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

                if not self.stop_event.is_set(): time.sleep(0.5) # Small delay

            if self.stop_event.is_set() or self._wait_if_server_busy(): break # Exit outer VLM loop
            if not pending_vlm_files: time.sleep(5) # Pause if queue empty

        # --- End of outer Phase 2b while loop ---
        total_duration = time.monotonic() - last_report_time
        logger.success(f"✅ Finished Phase 2 Cycle. Converted: {converted_count}, VLM Processed: {vlm_processed_count}, Errors: {error_count}. Duration: {total_duration:.2f}s")

    # --- (Existing run method - orchestrates Phase 1 then Phase 2) ---
    def run(self):
        """Main execution loop: Runs Phase 1 scan, then Phase 2 VLM, then waits."""
        logger.info(f"✅ {self.thread_name} started.")
        db: Session = None
        while not self.stop_event.is_set():
            cycle_start_time = time.monotonic()
            # --- Phase 1 ---
            logger.info(f"--- {self.thread_name}: Starting Scan Cycle (Phase 1) ---")
            try: db = SessionLocal(); root_paths = self._get_root_paths()
            except Exception as db_err: logger.error(f"Failed get DB session for Phase 1: {db_err}"); time.sleep(60); continue
            try:
                 for root in root_paths:
                      if self.stop_event.is_set(): break
                      self._scan_directory(root, db) # Calls Phase 1 logic
            except Exception as e: logger.error(f"💥 Unhandled error during Phase 1: {e}", exc_info=True)
            finally:
                 if db: db.close(); db = None
            if self.stop_event.is_set(): break
            phase1_duration = time.monotonic() - cycle_start_time
            logger.info(f"--- {self.thread_name}: Phase 1 Completed ({phase1_duration:.2f}s) ---")

            # --- Phase 2 ---
            logger.info(f"--- {self.thread_name}: Starting VLM/Conversion Cycle (Phase 2) ---")
            phase2_start_time = time.monotonic()
            try: db = SessionLocal()
            except Exception as db_err: logger.error(f"Failed get DB session for Phase 2: {db_err}"); time.sleep(60); continue
            try: self._process_pending_vlm_files(db) # Calls Phase 2 logic
            except Exception as e: logger.error(f"💥 Unhandled error during Phase 2: {e}", exc_info=True)
            finally:
                 if db: db.close(); db = None
            if self.stop_event.is_set(): break
            phase2_duration = time.monotonic() - phase2_start_time
            logger.info(f"--- {self.thread_name}: Phase 2 Completed ({phase2_duration:.2f}s) ---")

            # --- Wait ---
            total_cycle_duration = time.monotonic() - cycle_start_time
            wait_time = max(10, SCAN_INTERVAL_SECONDS - total_cycle_duration)
            logger.info(f"{self.thread_name}: Full cycle ({total_cycle_duration:.1f}s). Waiting {wait_time:.1f}s...")
            self.stop_event.wait(timeout=wait_time)

        logger.info(f"🛑 {self.thread_name} exiting.")

    def _scan_directory(self, root_path: str, db_session: Session):
        """
        Phase 1: Walks through a directory and processes files using _process_file_phase1,
        skipping OS/system dirs/files and hidden dot files/dirs.
        Reports progress periodically.
        """
        logger.info(f"🔬 Starting Phase 1 Scan Cycle for root: {root_path}")
        total_processed_this_cycle = 0
        total_errors_this_cycle = 0
        last_report_time = time.monotonic()
        files_since_last_report = 0
        errors_since_last_report = 0
        report_interval_seconds = 60 # Log progress every minute

        # --- Skip Logic Initialization ---
        # Define common system directories (absolute paths mostly)
        common_system_dirs_raw = {
            # General Unix/Linux/BSD/macOS (Absolute)
            "/proc", "/sys", "/dev", "/run", "/etc", "/var", "/tmp", "/private",
            "/cores", "/opt", "/usr",
            # Linux Specific (Absolute)
            "/snap", "/lost+found", "/boot",
            # macOS Specific (Absolute)
            "/System", "/Library", "/Users/Shared", "/Network", "/Volumes",
            "/.MobileBackups", "/.Spotlight-V100", "/.fseventsd",
            # Windows Specific (Drive Letter Agnostic - checking startswith later)
            "$recycle.bin", "system volume information", "program files", "program files (x86)",
            "windows", "programdata", "recovery", "$windows.~bt", "$windows.~ws",
            "config.msi",
            # Common Dev/Temp Dirs (Relative name check - avoid common large/generated dirs)
            "node_modules", "__pycache__", ".venv", "venv", ".env", "env",
            ".tox", ".pytest_cache", ".mypy_cache", "Cache", "cache", "Library",
            # Specific to your project structure to avoid indexing itself
            "staticmodelpool", "llama-cpp-python_build", "systemCore", "engineMain", "backend-service", "frontend-face-zephyrine",
            # Common user data dirs that might be large/redundant
            "Downloads", "Pictures", "Movies", "Music",
            # Common lib/bin dirs (might catch venv dirs too)
             "python3", "lib", "bin", "database",
             # Example: Add more potentially large/unwanted directories
             "Backup", "Archives"
        }

        # Separate lists for different matching strategies
        absolute_skip_dirs_normalized = {os.path.normpath(p) for p in common_system_dirs_raw if p.startswith('/') or ':' in p}
        relative_skip_dir_names_lower = {p.lower() for p in common_system_dirs_raw if not (p.startswith('/') or ':' in p)}
        absolute_skip_dirs_normalized_lower = {p.lower() for p in absolute_skip_dirs_normalized}

        # Specific system files to skip by name (case-insensitive)
        files_to_skip_lower = {
            '.ds_store', 'thumbs.db', 'desktop.ini', '.localized',
            '.bash_history', '.zsh_history',
            'ntuser.dat', '.swp', '.swo',
            'pagefile.sys', 'hiberfil.sys',
            '.volumeicon.icns', '.cfusertextencoding',
            '.traceroute.log',
            # Add potentially large single files if needed
            '.bash_sessions_disable'
        }
        # --- End Skip Logic Init ---

        try:
            # Using os.walk
            for current_dir, dirnames, filenames in os.walk(root_path, topdown=True, onerror=None):
                if self.stop_event.is_set():
                    logger.info(f"Phase 1 Scan interrupted by stop signal in {current_dir}")
                    break

                # --- Apply Skip Logic for Directories ---
                norm_current_dir = os.path.normpath(current_dir)
                norm_current_dir_lower = norm_current_dir.lower()
                current_dir_basename_lower = os.path.basename(norm_current_dir).lower()
                should_skip_dir = False

                # 1. Check against absolute system paths
                for skip_dir in absolute_skip_dirs_normalized:
                    # Use os.path.commonpath for more robust checking if needed, but startswith is often sufficient
                    if norm_current_dir == skip_dir or norm_current_dir.startswith(skip_dir + os.sep):
                        should_skip_dir = True; break
                if not should_skip_dir:
                     for skip_dir_lower in absolute_skip_dirs_normalized_lower:
                         if norm_current_dir_lower == skip_dir_lower or norm_current_dir_lower.startswith(skip_dir_lower + os.sep):
                              should_skip_dir = True; break

                # 2. Check if the directory *name* matches a relative skip name
                if not should_skip_dir:
                    if current_dir_basename_lower in relative_skip_dir_names_lower:
                        should_skip_dir = True

                # 3. Skip hidden directories (dot directories)
                # We check basename here too and prune walk below
                if not should_skip_dir and os.path.basename(norm_current_dir).startswith('.'):
                    should_skip_dir = True
                # --- End Directory Skip Logic ---

                if should_skip_dir:
                    logger.trace(f"Phase 1 Skipping excluded/hidden directory: {current_dir}")
                    dirnames[:] = [] # Prune walk - essential for efficiency
                    filenames[:] = [] # Clear filenames list for this dir
                    continue

                # --- Prune Hidden Subdirectories from Further Walk (if not already skipped) ---
                # Filter dirnames *in place* to remove dot-directories before iterating files
                original_dir_count = len(dirnames)
                dirnames[:] = [d for d in dirnames if not d.startswith('.')]
                skipped_dot_dirs = original_dir_count - len(dirnames)
                if skipped_dot_dirs > 0:
                     logger.trace(f"Phase 1 Pruning {skipped_dot_dirs} hidden subdirs from walk in: {current_dir}")
                # --- End Hidden Directory Pruning ---

                logger.trace(f"Phase 1 Scanning {current_dir}...")
                current_dir_file_errors = 0

                # Process files in the current (non-skipped) directory
                for filename in filenames:
                    if self.stop_event.is_set(): break

                    # --- Apply Skip Logic for Files ---
                    # 1. Skip hidden files (dot files) - Double check needed? Basename check covers parent.
                    if filename.startswith('.'):
                        logger.trace(f"Phase 1 Skipping hidden file: {filename}")
                        continue

                    # 2. Skip specific system/config files by name
                    filename_lower = filename.lower()
                    if filename_lower in files_to_skip_lower:
                         logger.trace(f"Phase 1 Skipping excluded system/config file: {filename}")
                         continue
                    # --- End File Skip Logic ---

                    file_path = os.path.join(current_dir, filename)
                    file_processed = False
                    file_errored = False

                    try:
                        # Check link/file status *after* initial name checks
                        if os.path.islink(file_path):
                             logger.trace(f"Phase 1 Skipping symbolic link: {file_path}")
                             continue
                        # Ensure it's actually a file (could be socket, etc.)
                        if not os.path.isfile(file_path):
                            logger.trace(f"Phase 1 Skipping non-file entry: {file_path}")
                            continue

                        # --- Call PHASE 1 processing ---
                        self._process_file_phase1(file_path, db_session)
                        file_processed = True
                        # --- End Phase 1 call ---

                    except PermissionError:
                        logger.warning(f"Phase 1 Permission denied accessing path info: {file_path}")
                        file_errored = True
                        # Attempt to log error to DB if permission denied during the walk itself
                        try:
                            existing = db_session.query(FileIndex).filter(FileIndex.file_path == file_path).first()
                            if existing:
                                if existing.index_status != 'error_permission': existing.index_status = 'error_permission'; existing.processing_error = "Permission denied during scan."; db_session.commit()
                            else: perm_error_record = FileIndex(file_path=file_path, file_name=filename, index_status='error_permission', processing_error="Permission denied during scan."); db_session.add(perm_error_record); db_session.commit()
                        except Exception as db_perm_err: logger.error(f"Failed to log permission error to DB for {file_path}: {db_perm_err}"); db_session.rollback()

                    except Exception as walk_err:
                        logger.error(f"Phase 1 Error processing entry {file_path} during walk: {walk_err}", exc_info=True) # Add traceback
                        file_errored = True

                    finally:
                        # --- Update Counters & Report ---
                        if file_processed:
                            total_processed_this_cycle += 1; files_since_last_report += 1
                        if file_errored:
                            total_errors_this_cycle += 1; errors_since_last_report += 1; current_dir_file_errors += 1

                        current_time = time.monotonic()
                        if current_time - last_report_time >= report_interval_seconds:
                            rate = files_since_last_report / report_interval_seconds
                            logger.info(f"⏳ [Phase 1 Report] Last {report_interval_seconds}s: {files_since_last_report} files (~{rate:.1f}/s), {errors_since_last_report} errors. Dir: '...{os.path.basename(current_dir)}'. Cycle Total: {total_processed_this_cycle}")
                            last_report_time = current_time; files_since_last_report = 0; errors_since_last_report = 0
                        # --- End Counters & Report ---

                        # Yield CPU slightly between files if needed
                        if YIELD_SLEEP_SECONDS > 0:
                            time.sleep(YIELD_SLEEP_SECONDS)

                # Check stop event after processing files in a directory
                if self.stop_event.is_set(): break

                if current_dir_file_errors > 0:
                     logger.warning(f"Encountered {current_dir_file_errors} errors processing files within: {current_dir}")

            # End of os.walk loop
        except Exception as outer_walk_err:
             logger.error(f"Outer error during Phase 1 os.walk for {root_path}: {outer_walk_err}", exc_info=True)
             total_errors_this_cycle += 1

        # Log final summary for this root path scan cycle
        if not self.stop_event.is_set():
            logger.success(f"✅ Finished Phase 1 Scan for {root_path}. Total Processed: {total_processed_this_cycle}, Total Errors: {total_errors_this_cycle}")
        else:
            logger.warning(f"⏹️ Phase 1 Scan for {root_path} interrupted by stop signal. Processed in this cycle: {total_processed_this_cycle}, Errors: {total_errors_this_cycle}")

    def run(self):
        """Main execution loop: Runs Phase 1 scan, then Phase 2 VLM, then waits."""
        logger.info(f"✅ {self.thread_name} started.")
        db: Session = None

        while not self.stop_event.is_set():
            cycle_start_time = time.monotonic()
            # --- Phase 1: File Scanning ---
            logger.info(f"--- {self.thread_name}: Starting Scan Cycle (Phase 1) ---")
            try:
                db = SessionLocal()
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
                db = SessionLocal()
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

async def initialize_global_file_index_vectorstore(provider: AIProvider):  # Pass AIProvider instance
    logger.info(">>> initialize_global_file_index_vectorstore CALLED <<<")  # Add this
    logger.info(">>> initialize_global_file_index_vectorstore CALLED <<<")
    if not provider:
        logger.error("  INITIALIZE_VS_ERROR: Passed 'provider' is None!")
        return
    logger.info(f"  INITIALIZE_VS_INFO: Passed 'provider' type: {type(provider)}")
    if not provider.embeddings:
        logger.error("  INITIALIZE_VS_ERROR: Passed 'provider.embeddings' is None!")
        return
    logger.info(f"  INITIALIZE_VS_INFO: Passed 'provider.embeddings' type: {type(provider.embeddings)}")

    global global_file_index_vectorstore  # Refer to the module-level global

    # Prevent re-initialization or concurrent initialization
    if _file_index_vs_initialized_event.is_set():
        logger.info("Global FileIndex vector store already initialized. Skipping.")
        return

    with _file_index_vs_init_lock:
        # Double-check after acquiring lock
        if _file_index_vs_initialized_event.is_set():
            logger.info("Global FileIndex vector store was initialized while waiting for lock. Skipping.")
            return

        if not provider or not provider.embeddings:  # Use passed provider
            logger.error("Cannot init global file index VS: AIProvider or embeddings missing.")
            return

        db: Optional[Session] = None
        try:
            db = SessionLocal()
            logger.info("Initializing GLOBAL FileIndex vector store from file_indexer.py...")
            # Fetch records that have embeddings and content
            indexed_files = db.query(FileIndex).filter(
                FileIndex.embedding_json.isnot(None),
                FileIndex.indexed_content.isnot(None),
                FileIndex.index_status.in_(['indexed_text', 'success'])
                # Or 'pending_vlm', 'pending_conversion' if text is already there
            ).all()

            if not indexed_files:
                logger.warning("No files found in DB with embeddings and content for global vector store.")
                _file_index_vs_initialized_event.set()  # Mark as "initialized" (even if empty) to prevent retries
                return

            texts_for_vs = []
            embeddings_for_vs = []
            metadatas_for_vs = []
            ids_for_vs = []

            for record in indexed_files:
                try:
                    vector = json.loads(record.embedding_json)
                    # Ensure content is not excessively long.
                    # The text stored in Chroma is what gets *returned* after a match.
                    # Store enough for context, but perhaps not gigantic files verbatim.
                    # For simplicity, using full indexed_content now. Consider truncation if memory becomes an issue.
                    content_to_store = record.indexed_content

                    texts_for_vs.append(content_to_store)
                    embeddings_for_vs.append(vector)
                    metadatas_for_vs.append({
                        "source": record.file_path,
                        "file_id": record.id,
                        "file_name": record.file_name,
                        "last_modified": str(record.last_modified_os),
                        "index_status": record.index_status,
                        "mime_type": record.mime_type
                    })
                    ids_for_vs.append(f"file_{record.id}")
                except json.JSONDecodeError:
                    logger.warning(f"Skipping FileIndex ID {record.id} for global VS: bad embedding JSON.")
                except Exception as e:
                    logger.warning(f"Skipping FileIndex ID {record.id} for global VS processing its data: {e}")

            if texts_for_vs and embeddings_for_vs:
                logger.info(
                    f"Building global FileIndex Chroma store with {len(texts_for_vs)} items using pre-computed embeddings...")
                global_file_index_vectorstore = Chroma.from_embeddings(
                    text_embeddings=embeddings_for_vs,
                    embedding=provider.embeddings,  # Use passed provider's embeddings
                    documents=texts_for_vs,
                    metadatas=metadatas_for_vs,
                    ids=ids_for_vs,
                    # persist_directory="./chroma_file_index_store" # Optional: for disk persistence
                )
                if global_file_index_vectorstore:
                    logger.success(
                        f"Global FileIndex VS created. Type: {type(global_file_index_vectorstore)}. Embedding function type: {type(global_file_index_vectorstore.embedding_function)}")
                    if hasattr(global_file_index_vectorstore.embedding_function, 'embed_query'):
                        logger.info("  Chroma VS embedding function HAS 'embed_query' method.")
                    else:
                        logger.error("  CRITICAL: Chroma VS embedding function LACKS 'embed_query' method!")
                logger.success("Global FileIndex vector store initialized successfully.")
            else:
                logger.warning("No valid texts/embeddings to build global FileIndex vector store.")

            _file_index_vs_initialized_event.set()  # Signal completion

        except Exception as e:
            logger.error(f"Failed to initialize global FileIndex vector store: {e}")
            logger.exception("Global FileIndex VS Init Traceback:")
            # Don't set the event on failure, so it might be retried or indicates a problem
        finally:
            if db:
                db.close()

def get_global_file_index_vectorstore() -> Optional[Chroma]:
    """Returns the initialized global file index vector store, or None if not ready."""
    if _file_index_vs_initialized_event.is_set():
        return global_file_index_vectorstore
    else:
        logger.warning("Attempted to get global file index vector store before it was initialized.")
        return None