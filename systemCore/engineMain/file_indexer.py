# file_indexer.py
import os
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
    def __init__(self, stop_event: threading.Event, provider: AIProvider, server_busy_event: threading.Event): # <<< CHANGE 'embedding_model' to 'provider' here
        self.stop_event = stop_event
        self.provider = provider # Store the whole provider object
        self.embedding_model = provider.embeddings # Get embeddings from the provider
        self.vlm_model = provider.get_model("vlm") # Get VLM model from the provider
        self.thread_name = "FileIndexerThread"
        self.server_busy_event = server_busy_event # <<< Store the busy event
        if not self.embedding_model:
             logger.warning("⚠️ FileIndexer initialized WITHOUT an embedding model. Text embedding will be skipped.")
        else:
            model_info = getattr(self.embedding_model, 'model_name', type(self.embedding_model).__name__) # Example getting model info
            logger.info(f"🧵 FileIndexer Embedding Model: {model_info}")

        if not self.vlm_model:
            logger.warning("⚠️ FileIndexer initialized WITHOUT a VLM model. LaTeX generation will be skipped.")
        else:
            model_info = getattr(self.vlm_model, 'model', type(self.vlm_model).__name__) # Example getting model info
            logger.info(f"🧵 FileIndexer VLM Model: {model_info}")
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

    def _process_file(self, file_path: str, db_session: Session):
        """Processes a single file: check metadata & MD5, extract, embed, update DB."""
        # --- Check/Wait at the START of processing a file ---
        if self._wait_if_server_busy():
            logger.trace(f"Yielding file processing due to busy server: {file_path}")
            return # Skip this file entirely for now if server busy at start
        if self.stop_event.is_set(): return
        # ---
        if self.stop_event.is_set():
            logger.info(f"Stop event set, skipping processing for: {file_path}")
            return
        
        logger.trace(f"Processing file: {file_path}")
        file_name = os.path.basename(file_path)
        status = 'pending'
        content = None
        error_message = None
        embedding_json_str = None
        current_md5_hash = None
        latex_representation = None # <<< NEW
        latex_explanation = None # <<< NEW
        vlm_processing_status = None # <<< NEW
        should_update_db = True
        existing_record: Optional[FileIndex] = None

        try:
            # 1. Get File Metadata
            size_bytes, mtime_os, mime_type = self._get_file_metadata(file_path)
            if size_bytes is None and mtime_os is None:
                status = 'error_permission'; error_message = "Permission denied or file vanished during stat."
            elif size_bytes is None:
                 size_bytes = -1
                 if mtime_os is None: status = 'error_read'; error_message = "Failed to get consistent metadata."

            # --- 2. Calculate Current MD5 Hash (if possible) ---
            if status == 'pending':
                try:
                    current_md5_hash = self._calculate_md5(file_path, size_bytes)
                    if current_md5_hash is None and size_bytes <= MAX_HASH_FILE_SIZE_BYTES and size_bytes >= 0:
                        # Hashing failed for a file that *should* have been hashable
                        status = 'error_hash'
                        error_message = "Failed to calculate MD5 hash."
                        logger.warning(f"MD5 hash calculation failed for: {file_path}")
                except PermissionError:
                    # Re-catch permission error specifically from hashing
                    raise # Let the outer handler catch this
                except Exception as hash_err: # Catch unexpected errors during hashing call
                     status = 'error_hash'; error_message = f"Error during hashing: {hash_err}"; current_md5_hash = None
                     logger.error(f"Unexpected error calling _calculate_md5 for {file_path}: {hash_err}")


            # --- 3. Check Database using MD5 ---
            existing_record: Optional[FileIndex] = None
            if status == 'pending': # Only check DB if no prior metadata/hash error
                try:
                    existing_record = db_session.query(FileIndex).filter(FileIndex.file_path == file_path).first()
                    if existing_record:
                        # --- Compare Hashes ---
                        if current_md5_hash is not None and existing_record.md5_hash == current_md5_hash:
                            # Hashes match, file assumed unchanged
                            logger.trace(f"Skipping DB update (MD5 match): {file_path}")
                            should_update_db = False # File hasn't changed
                        elif current_md5_hash is None and size_bytes > MAX_HASH_FILE_SIZE_BYTES and existing_record.md5_hash is None:
                             # Both current and previous hash skipped due to size, check mtime as fallback
                             if mtime_os and existing_record.last_modified_os and mtime_os <= existing_record.last_modified_os:
                                  logger.trace(f"Skipping DB update (Large file, MD5 skipped, timestamp match): {file_path}")
                                  should_update_db = False
                             else:
                                  logger.trace(f"Updating large file (MD5 skipped, timestamp changed or missing): {file_path}")
                        else:
                            # Hashes differ, or one is missing (and not skipped due to size) - need to re-index
                            logger.trace(f"Updating DB (MD5 mismatch or missing): Current={current_md5_hash}, DB={existing_record.md5_hash} for {file_path}")
                            # Status remains 'pending' for now, will be updated after content processing
                except SQLAlchemyError as db_check_err:
                     logger.error(f"DB check failed for {file_path}: {db_check_err}")
                     status = 'error_read'; error_message = f"Database check failed: {db_check_err}"
                     should_update_db = False # Don't proceed if DB check fails

            # --- 4. Process Content & Embed if Update Needed ---
            if should_update_db and status == 'pending':
                file_ext = os.path.splitext(file_name)[1].lower()
                should_embed = file_ext in EMBEDDABLE_EXTENSIONS
                is_office_doc = file_ext in {'.docx', '.xlsx', '.pptx'}
                is_pdf = file_ext == '.pdf'
                is_text = file_ext in TEXT_EXTENSIONS or (mime_type and mime_type.startswith('text/'))
                can_attempt_vlm = is_pdf # <<< STARTING WITH PDF ONLY FOR VLM->LATEX

                # Extract content (logic remains the same)
                if file_ext in DOC_EXTENSIONS:
                    # ... (pdf, docx, xlsx, pptx extraction) ...
                    if file_ext == '.pdf': content = self._extract_pdf(file_path)
                    elif file_ext == '.docx': content = self._extract_docx(file_path)
                    elif file_ext == '.xlsx': content = self._extract_xlsx(file_path)
                    elif file_ext == '.pptx': content = self._extract_pptx(file_path)
                    if content is None: status = 'error_read'; error_message = "Content extraction returned None."; should_embed = False
                elif file_ext in TEXT_EXTENSIONS or (mime_type and mime_type.startswith('text/')):
                    if size_bytes > MAX_TEXT_FILE_SIZE_BYTES: # Check read size limit
                        status = 'skipped_size'; error_message = f"Skipped content (exceeds {MAX_TEXT_FILE_SIZE_MB} MB)"; content = None; should_embed = False
                    else:
                        content = self._extract_text(file_path, size_bytes)
                        if content is None: status = 'error_read'; error_message = "Text extraction returned None."; should_embed = False
                else:
                     status = 'indexed_meta'; content = None; should_embed = False
                     logger.trace(f"Indexing metadata only for {file_path} (Type: {mime_type or file_ext})")

                # Generate Embedding (logic remains the same, depends on successful content extraction)
                if status == 'pending' and content and should_embed:
                    if self.embedding_model:
                        logger.debug(f"🧠 Generating embedding for: {file_path}")
                        try:
                            vector = self.embedding_model.embed_query(content)
                            embedding_json_str = json.dumps(vector)
                            status = 'indexed_text'
                            logger.trace(f"Embedding generated successfully for {file_path}")
                        except Exception as emb_err:
                            logger.error(f"Embedding generation failed for {file_path}: {emb_err}")
                            status = 'error_embedding'; error_message = f"Embedding failed: {emb_err}"; embedding_json_str = None
                    else:
                        status = 'indexed_text'; logger.warning(f"Skipping embedding (no model) for: {file_path}")
                elif status == 'pending' and not content and should_embed:
                     status = 'indexed_text'; logger.trace(f"Extracted empty content, skipping embedding for {file_path}")
                elif status == 'pending' and not should_embed:
                     status = 'indexed_meta'




            # Handle stop event during processing
            if content == "[Indexing stopped]": # Check if extraction was interrupted
                 status = "pending"; should_update_db = False
                 logger.info(f"Indexing stopped during content extraction for: {file_path}")
            # If current_md5_hash is None after hashing logic, it might mean stop occurred there
            # but it's harder to check directly without passing status back.

        # --- 4b. Attempt VLM -> LaTeX Processing (if needed) ---
            # Trigger this *after* text processing, only if text was indexed successfully OR if it's a type we want to VLM anyway
            # And importantly, only if the file hasn't fundamentally changed (MD5 match already skipped)
            # We also check if the existing record *already has* LaTeX data
            needs_latex_check = (should_update_db and self.vlm_model and can_attempt_vlm and
                                     (existing_record is None or existing_record.latex_representation is None))

            if needs_latex_check:
                    logger.info(f"🖼️ Attempting VLM->LaTeX processing for: {file_path}")
                    vlm_processing_status = 'processing'
                    vlm_proc_start_time = time.monotonic() # Start timer for whole VLM block

                    images = self._convert_pdf_to_images(file_path)
                    if images:
                        num_pages = len(images)
                        logger.info(f"  📄 Found {num_pages} pages for VLM processing.")
                        all_latex = []
                        all_expl = []
                        page_errors = 0
                        time_first_page = 0.0 # Track time for first page
                        estimated_total_time = 0.0

                        for i, page_image in enumerate(images):
                            page_num = i + 1
                            if self.stop_event.is_set():
                                logger.info("VLM->LaTeX stopped by signal."); vlm_processing_status = 'pending'; break

                            # --- Logging BEFORE processing page ---
                            eta_str = ""
                            if i > 0 and estimated_total_time > 0:
                                elapsed_since_start = time.monotonic() - vlm_proc_start_time
                                remaining_time = max(0, estimated_total_time - elapsed_since_start)
                                eta_str = f" (ETA: {remaining_time:.0f}s)"
                            elif i == 0:
                                eta_str = "" # No ETA yet
                            else: # Estimating failed or first page took no time
                                eta_str = ""

                            logger.info(f"  🧠 Processing page {page_num}/{num_pages} with VLM...{eta_str}")
                            # --- End Logging ---

                            page_start_time = time.monotonic()
                            l_code, l_expl = self._get_latex_from_image(page_image)
                            page_duration = time.monotonic() - page_start_time

                            # --- Logging AFTER processing page ---
                            logger.debug(f"    Page {page_num} VLM processing took {page_duration:.2f}s. Found LaTeX: {'Yes' if l_code else 'No'}")

                            # Estimate total time after first page completes
                            if i == 0 and page_duration > 0.1: # Avoid division by zero or tiny times skewing ETA
                                time_first_page = page_duration
                                estimated_total_time = time_first_page * num_pages
                                logger.info(f"    First page processed in {time_first_page:.2f}s. Estimated total VLM time: ~{estimated_total_time:.0f}s")
                            # --- End Logging ---

                            if l_expl and "[VLM Error:" in l_expl: page_errors += 1
                            if l_code: all_latex.append(f"% Page {page_num}\n{l_code}")
                            if l_expl: all_expl.append(f"## Page {page_num}\n{l_expl}")
                            # time.sleep(0.1) # Optional sleep

                        # After loop finishes or breaks
                        if self.stop_event.is_set():
                            pass # Status already set
                        elif page_errors == num_pages and num_pages > 0 : # All pages failed VLM
                            vlm_processing_status = 'error_vlm'
                            error_message = (error_message or "") + f" [VLM errors on ALL {page_errors} pages]"
                        elif page_errors > 0: # Some pages failed
                            vlm_processing_status = 'partial_vlm_error'
                            error_message = (error_message or "") + f" [VLM errors on {page_errors}/{num_pages} pages]"
                        else: # No errors reported by VLM helper
                            vlm_processing_status = 'success'

                        latex_representation = "\n\n".join(all_latex) if all_latex else None
                        latex_explanation = "\n\n".join(all_expl) if all_expl else None
                        vlm_total_duration = time.monotonic() - vlm_proc_start_time
                        logger.info(f"  VLM->LaTeX processing finished for {file_path}. Status: {vlm_processing_status}. Found LaTeX: {'Yes' if latex_representation else 'No'}. Total Time: {vlm_total_duration:.2f}s")

                    else: # Failed to get images
                        logger.warning(f"Skipping VLM->LaTeX for {file_path}: Failed to convert to images.")
                        vlm_processing_status = 'error_conversion'
                        error_message = (error_message or "") + " [Failed PDF->Image conversion]"
                    # Keep existing LaTeX data if file content changed but LaTeX already exists
                    latex_representation = existing_record.latex_representation
                    latex_explanation = existing_record.latex_explanation
                    vlm_processing_status = existing_record.vlm_processing_status # Keep old status

        except PermissionError:
             status = 'error_permission'; error_message = "Permission denied during file processing."; content = None; embedding_json_str = None; current_md5_hash = None; should_update_db = True
             logger.warning(f"Permission error processing {file_path}")
        except Exception as proc_err:
            status = 'error_read'; error_message = f"Processing error: {proc_err}"; content = None; embedding_json_str = None; current_md5_hash = None; should_update_db = True
            logger.error(f"Error processing file {file_path}: {proc_err}")
            logger.exception("File Processing Traceback:")

        # --- 5. Update Database Record ---
        if should_update_db:
            # Ensure final status is set if it was still pending
            if status == 'pending':
                 if content is not None: status = 'indexed_text' # Assume text index if content exists but no embedding done
                 else: status = 'indexed_meta' # Otherwise just metadata

            logger.debug(f"Updating DB for {file_path} -> Status: {status}, Hash: {current_md5_hash}")
            record_data = {
                'file_name': file_name,
                'size_bytes': size_bytes if size_bytes is not None else -1,
                'mime_type': mime_type,
                'last_modified_os': mtime_os,
                'index_status': status,
                'indexed_content': (content[:50000] + '...[truncated]') if content and len(content) > 50000 else content,
                'embedding_json': embedding_json_str,
                'md5_hash': current_md5_hash,
                'processing_error': error_message[:1000] if error_message else None,
                'last_indexed_db': datetime.datetime.now(datetime.timezone.utc),
                'latex_representation': latex_representation,
                'latex_explanation': latex_explanation,
                'vlm_processing_status': vlm_processing_status
            }
            try:
                if existing_record: stmt = update(FileIndex).where(FileIndex.id == existing_record.id).values(**record_data); db_session.execute(stmt)
                else: new_record = FileIndex(file_path=file_path, **record_data); db_session.add(new_record)
                db_session.commit()
                logger.trace(f"DB commit successful for {file_path}")
            except SQLAlchemyError as db_upd_err: logger.error(f"DB update/insert FAILED for {file_path}: {db_upd_err}"); db_session.rollback()
        elif existing_record and existing_record.vlm_processing_status != vlm_processing_status:
             # Handle case where only VLM status might need update even if MD5 matched
             logger.debug(f"Updating only VLM status for {file_path} (MD5 matched)")
             try:
                  stmt = update(FileIndex).where(FileIndex.id == existing_record.id).values(vlm_processing_status=vlm_processing_status)
                  db_session.execute(stmt)
                  db_session.commit()
             except SQLAlchemyError as db_vlm_err: logger.error(f"DB VLM status update FAILED for {file_path}: {db_vlm_err}"); db_session.rollback()


    def _scan_directory(self, root_path: str, db_session: Session):
        """
        Walks through a directory and processes files, skipping OS/system dirs/files
        and hidden dot files/dirs. Reports progress periodically.
        """
        logger.info(f"🔬 Starting scan cycle for root: {root_path}")
        total_processed_this_cycle = 0
        total_errors_this_cycle = 0
        last_report_time = time.monotonic()
        files_since_last_report = 0
        errors_since_last_report = 0
        report_interval_seconds = 60

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
            ".tox", ".pytest_cache", ".mypy_cache", "Cache", "cache", "Library", "python3", "lib", "database"
            # NOTE: General dot files/dirs like .git, .config, .cache, .local etc.
            # will be skipped by the explicit dot-prefix check below.
            # No need to list them individually here unless you want to skip them even earlier.
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
            '.volumeicon.icns', '.cfusertextencoding', # From user list examples
            '.traceroute.log' # Example of another hidden log file
        }
        # --- End Skip Logic Init ---

        try:
            # Using os.walk
            for current_dir, dirnames, filenames in os.walk(root_path, topdown=True, onerror=None):
                if self.stop_event.is_set():
                    logger.info(f"Stop event detected, halting scan in {current_dir}")
                    break

                # --- Apply Skip Logic for Directories ---
                norm_current_dir = os.path.normpath(current_dir)
                norm_current_dir_lower = norm_current_dir.lower()
                current_dir_basename_lower = os.path.basename(norm_current_dir).lower()
                should_skip_dir = False

                # 1. Check against absolute system paths
                for skip_dir in absolute_skip_dirs_normalized:
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
                # --- End Directory Skip Logic ---

                if should_skip_dir:
                    logger.trace(f"Skipping excluded system/dev directory: {current_dir}")
                    dirnames[:] = [] # Prune walk
                    filenames[:] = [] # Clear filenames
                    continue

                # --- Skip Hidden Directories (Dot Directories) and Prune Walk ---
                # Filter dirnames *in place* to remove dot-directories before iterating files
                original_dir_count = len(dirnames)
                dirnames[:] = [d for d in dirnames if not d.startswith('.')]
                skipped_dot_dirs = original_dir_count - len(dirnames)
                if skipped_dot_dirs > 0:
                     logger.trace(f"Pruning {skipped_dot_dirs} hidden subdirectories from walk in: {current_dir}")
                # --- End Hidden Directory Pruning ---


                logger.trace(f"Scanning {current_dir}...")
                current_dir_file_errors = 0

                # Process files in the current directory
                for filename in filenames:
                    if self.stop_event.is_set():
                        break

                    # --- Apply Skip Logic for Files ---
                    # 1. Skip hidden files (dot files)
                    if filename.startswith('.'):
                        logger.trace(f"Skipping hidden file: {filename}")
                        continue

                    # 2. Skip specific system/config files by name
                    filename_lower = filename.lower()
                    if filename_lower in files_to_skip_lower:
                         logger.trace(f"Skipping excluded system/config file: {filename}")
                         continue
                    # --- End File Skip Logic ---

                    file_path = os.path.join(current_dir, filename)
                    file_processed = False
                    file_errored = False

                    try:
                        # Check link/file status *after* initial name checks
                        if os.path.islink(file_path):
                             logger.trace(f"Skipping symbolic link: {file_path}")
                             continue
                        if not os.path.isfile(file_path):
                            logger.trace(f"Skipping non-file entry: {file_path}")
                            continue

                        # Process the file (updates DB internally)
                        self._process_file(file_path, db_session)
                        file_processed = True

                    except PermissionError:
                        logger.warning(f"Permission denied accessing path info: {file_path}")
                        file_errored = True
                        # ... (DB error logging for permission error - unchanged) ...
                        try:
                            existing = db_session.query(FileIndex).filter(FileIndex.file_path == file_path).first()
                            if existing:
                                if existing.index_status != 'error_permission': existing.index_status = 'error_permission'; existing.processing_error = "Permission denied during scan."; db_session.commit()
                            else: perm_error_record = FileIndex(file_path=file_path, file_name=filename, index_status='error_permission', processing_error="Permission denied during scan."); db_session.add(perm_error_record); db_session.commit()
                        except Exception as db_perm_err: logger.error(f"Failed to log permission error to DB for {file_path}: {db_perm_err}"); db_session.rollback()

                    except Exception as walk_err:
                        logger.error(f"Error processing entry {file_path} during walk: {walk_err}")
                        file_errored = True

                    finally:
                        # --- Update Counters & Report ---
                        if file_processed:
                            total_processed_this_cycle += 1; files_since_last_report += 1
                        if file_errored:
                            total_errors_this_cycle += 1; errors_since_last_report += 1; current_dir_file_errors += 1

                        current_time = time.monotonic()
                        if current_time - last_report_time >= report_interval_seconds:
                            logger.info(f"⏳ [Indexer Report] Last {report_interval_seconds}s: {files_since_last_report} files processed, {errors_since_last_report} errors. Current Dir: '{os.path.basename(current_dir)}'. Total Cycle: {total_processed_this_cycle} files, {total_errors_this_cycle} errors.")
                            last_report_time = current_time; files_since_last_report = 0; errors_since_last_report = 0
                        # --- End Counters & Report ---
                        time.sleep(YIELD_SLEEP_SECONDS)


                if self.stop_event.is_set(): break
                if current_dir_file_errors > 0:
                     logger.warning(f"Encountered {current_dir_file_errors} errors processing files within: {current_dir}")

            # End of os.walk loop
        except Exception as outer_walk_err:
             logger.error(f"Outer error during os.walk for {root_path}: {outer_walk_err}")
             logger.exception("os.walk Traceback:")
             total_errors_this_cycle += 1


        # Log final summary for this root path scan cycle
        if not self.stop_event.is_set(): logger.success(f"✅ Finished scan cycle for {root_path}. Total Processed: {total_processed_this_cycle}, Total Errors: {total_errors_this_cycle}")
        else: logger.warning(f"⏹️ Scan cycle for {root_path} interrupted by stop signal. Processed in this cycle: {total_processed_this_cycle}, Errors: {total_errors_this_cycle}")

    def run(self):
        """Main execution loop for the indexer thread."""
        logger.info(f"✅ {self.thread_name} started.")
        db: Session = None # Initialize
        initial_scan_done = False

        while not self.stop_event.is_set():
            scan_start_time = time.monotonic()
            try:
                # --- Get DB Session ---
                logger.debug(f"{self.thread_name}: Creating DB session.")
                db = SessionLocal()
                if not db:
                    logger.error(f"{self.thread_name}: Failed to acquire DB session. Retrying later.")
                    raise RuntimeError("Failed to get DB Session")

                # --- Perform Scan ---
                root_paths = self._get_root_paths()
                logger.info(f"{self.thread_name}: Starting filesystem scan cycle...")
                for root in root_paths:
                    if self.stop_event.is_set(): break
                    self._scan_directory(root, db)

                if self.stop_event.is_set():
                     logger.info(f"{self.thread_name}: Scan interrupted by stop signal.")
                     break # Exit main loop

                initial_scan_done = True # Mark initial scan as complete after first full pass
                scan_duration = time.monotonic() - scan_start_time
                logger.success(f"{self.thread_name}: Filesystem scan cycle completed in {scan_duration:.2f} seconds.")

            except Exception as e:
                logger.error(f"💥 Error in {self.thread_name} main loop: {e}")
                logger.exception("Indexer Thread Traceback:")
                # Avoid busy-looping on persistent errors
                sleep_duration = 60 # Sleep for a minute before retrying on error
            finally:
                # --- Close DB Session ---
                if db:
                    try:
                        db.close()
                        logger.debug(f"{self.thread_name}: DB session closed.")
                    except Exception as db_close_err:
                        logger.error(f"{self.thread_name}: Error closing DB session: {db_close_err}")
                db = None # Ensure session is cleared

                # --- Wait for next cycle or stop ---
                if initial_scan_done and not self.stop_event.is_set():
                    logger.info(f"{self.thread_name}: Scan complete. Sleeping for {SCAN_INTERVAL_HOURS} hours...")
                    sleep_duration = SCAN_INTERVAL_SECONDS
                elif not initial_scan_done and not self.stop_event.is_set():
                     # If initial scan failed, retry sooner
                     sleep_duration = 300 # Sleep 5 minutes before retrying initial scan
                     logger.warning(f"{self.thread_name}: Initial scan may have failed. Retrying in {sleep_duration} seconds.")
                else:
                     sleep_duration = 1 # Short sleep if stopping

                # Use wait with timeout for responsiveness to stop_event
                self.stop_event.wait(timeout=sleep_duration)

        logger.info(f"🛑 {self.thread_name} received stop signal and is exiting.")