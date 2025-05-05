# database.py (V16 - Full Code with Integrated Migrations)

import os
import sys
import datetime
import time
import atexit
import shutil
import hashlib
import subprocess # Needed for sqlite3 calls and alembic autogen
import tempfile # Needed for temporary dump file

# --- Zstandard Import ---
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
# --- End Zstandard Import ---

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, Float, Boolean,
    ForeignKey, Index, MetaData, update, desc, select, inspect as sql_inspect,
    UniqueConstraint, text, CheckConstraint
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, Session
from sqlalchemy.exc import OperationalError, ProgrammingError, SQLAlchemyError
from typing import List, Optional, Dict, Any
from sqlalchemy.sql import func
# Ensure DATETIME from sqlalchemy is imported if needed, otherwise use the one from sqlalchemy.dialects.sqlite
# from sqlalchemy import DateTime as SQLAlchemyDateTime
from sqlalchemy.dialects.sqlite import DATETIME as SQLITE_DATETIME # Using SQLite specific for potential benefits

# Alembic imports
from alembic.config import Config
from alembic import command
from alembic.script import ScriptDirectory
from alembic.runtime.environment import EnvironmentContext
from alembic.runtime.migration import MigrationContext
from alembic.util import CommandError

from loguru import logger

# --- Configuration & Paths ---
# Assume APP_DIR is the directory containing this database.py file
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_BASENAME = "mappedknowledge.db"
RUNTIME_DB_PATH = os.path.join(APP_DIR, DB_BASENAME) # Path for the runtime DB
DB_PATH_COMPRESSED = f"{RUNTIME_DB_PATH}.zst"        # Path for the compressed archive
RUNTIME_DATABASE_URL = f"sqlite:///{os.path.abspath(RUNTIME_DB_PATH)}" # URL for runtime

# --- Zstd Compression Level ---
# Use environment variable or default to 9
ZSTD_COMPRESSION_LEVEL = int(os.getenv("ZSTD_COMPRESSION_LEVEL", 9))

logger.info(f"⚙️ Runtime DB Path: {RUNTIME_DB_PATH}")
logger.info(f"📦 Compressed DB Path: {DB_PATH_COMPRESSED}")
logger.info(f"🔗 Runtime DB URL: {RUNTIME_DATABASE_URL}")

if not ZSTD_AVAILABLE:
    logger.critical("🔥🔥 zstandard library not found! Install with 'pip install zstandard'. Exiting.")
    sys.exit(1)

# --- Alembic Paths (Relative to this file/APP_DIR) ---
ALEMBIC_DIR = os.path.join(APP_DIR, 'alembic')
ALEMBIC_INI_PATH = os.path.join(APP_DIR, 'alembic.ini')
ALEMBIC_VERSIONS_PATH = os.path.join(ALEMBIC_DIR, 'versions')
ALEMBIC_ENV_PY_PATH = os.path.join(ALEMBIC_DIR, 'env.py')
ALEMBIC_SCRIPT_MAKO_PATH = os.path.join(ALEMBIC_DIR, "script.py.mako")

# --- SQLAlchemy Setup ---
Base = declarative_base()
_engine = None # Global engine variable, managed by get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False) # Defined early, bound later
logger.debug("SessionLocal factory structure created (unbound initially).")


# --- Database Models ---
# Define your models AFTER Base = declarative_base()
class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    session_id = Column(String, index=True, nullable=True)
    mode = Column(String, default="chat", index=True)
    input_type = Column(String, default="text")
    user_input = Column(Text, nullable=True)
    llm_response = Column(Text, nullable=True)
    classification = Column(String, nullable=True)
    classification_reason = Column(Text, nullable=True)
    emotion_context_analysis = Column(Text, nullable=True)
    tool_name = Column(String, nullable=True)
    tool_parameters = Column(Text, nullable=True)
    tool_result = Column(Text, nullable=True)
    image_data = Column(Text, nullable=True) # Store base64 or indicator? Indicator better.
    url_processed = Column(String, nullable=True)
    image_description = Column(Text, nullable=True)
    latex_representation = Column(Text, nullable=True)
    latex_explanation = Column(Text, nullable=True)
    execution_time_ms = Column(Float, nullable=True)
    reflection_completed = Column(Boolean, default=False, nullable=False, index=True) # Added column
    rag_source_url = Column(String, nullable=True)
    rag_history_ids = Column(String, nullable=True)
    requires_deep_thought = Column(Boolean, nullable=True)
    deep_thought_reason = Column(Text, nullable=True)
    tot_analysis_requested = Column(Boolean, default=False)
    tot_result = Column(Text, nullable=True)
    tot_delivered = Column(Boolean, default=False, index=True, server_default='0') # Explicit default 0 for SQLite Boolean
    assistant_action_analysis_json = Column(Text, nullable=True)
    assistant_action_type = Column(String, nullable=True, index=True)
    assistant_action_params = Column(Text, nullable=True)
    assistant_action_executed = Column(Boolean, default=False, server_default='0')
    assistant_action_result = Column(Text, nullable=True)
    # Add last_modified_db for easier tracking of reflection updates
    last_modified_db = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    __table_args__ = (
        Index('ix_interactions_session_mode_timestamp', 'session_id', 'mode', 'timestamp'),
        Index('ix_interactions_undelivered_tot', 'session_id', 'mode', 'tot_delivered', 'timestamp'),
        Index('ix_interactions_action_type_time', 'assistant_action_type', 'timestamp'),
        Index('ix_interactions_reflection_pending', 'reflection_completed', 'mode', 'input_type', 'timestamp'), # New index
    )

class AppleScriptAttempt(Base):
    __tablename__ = "applescript_attempts"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    session_id = Column(String, index=True, nullable=False)
    triggering_interaction_id = Column(Integer, ForeignKey('interactions.id', ondelete='SET NULL'), index=True, nullable=True)
    action_type = Column(String, index=True, nullable=False)
    parameters_json = Column(Text, nullable=False) # Store params as JSON string
    attempt_number = Column(Integer, default=1, nullable=False)
    generated_script = Column(Text, nullable=True)
    execution_success = Column(Boolean, index=True, nullable=True)
    execution_return_code = Column(Integer, nullable=True)
    execution_stdout = Column(Text, nullable=True)
    execution_stderr = Column(Text, nullable=True)
    execution_duration_ms = Column(Float, nullable=True)
    error_summary = Column(Text, nullable=True) # Extracted summary of stderr if failed
    triggering_interaction = relationship("Interaction", backref="applescript_attempts")
    __table_args__ = (
        UniqueConstraint('session_id', 'action_type', 'parameters_json', 'attempt_number', name='uq_applescript_attempt'),
        Index('ix_applescript_attempts_lookup', 'action_type', 'parameters_json', 'execution_success', 'timestamp'),
    )

class FileIndex(Base):
    __tablename__ = "file_index"
    id = Column(Integer, primary_key=True)
    file_path = Column(String, unique=True, index=True, nullable=False)
    file_name = Column(String, index=True, nullable=False)
    size_bytes = Column(Integer, nullable=True) # Use -1 for unknown/error?
    mime_type = Column(String, nullable=True)
    last_modified_os = Column(DateTime(timezone=False), nullable=True) # OS mtime (naive)
    last_indexed_db = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    index_status = Column(String, default='pending', index=True, nullable=False) # pending, indexed_text, indexed_meta, skipped_*, error_*
    indexed_content = Column(Text, nullable=True) # Store extracted text (can be large)
    embedding_json = Column(Text, nullable=True) # Store embedding vector as JSON string
    md5_hash = Column(String(32), nullable=True, index=True) # MD5 hash for change detection
    processing_error = Column(Text, nullable=True) # Store error message if indexing failed
    # VLM related fields
    latex_representation = Column(Text, nullable=True)
    latex_explanation = Column(Text, nullable=True)
    vlm_processing_status = Column(String, nullable=True, index=True) # e.g., pending_vlm, processing, success, error_vlm, partial_vlm_error, error_conversion
    __table_args__ = (
        Index('ix_file_index_name_status', 'file_name', 'index_status'),
        Index('ix_file_index_status_modified', 'index_status', 'last_modified_os'),
        Index('ix_file_index_path_hash', 'file_path', 'md5_hash'),
        CheckConstraint(index_status.in_([
            'pending', 'indexed_text', 'indexed_meta', 'skipped_size',
            'skipped_type', 'error_read', 'error_permission', 'processing',
            'error_embedding', 'error_hash', 'error_vlm', 'partial_vlm_error',
            'error_conversion', 'pending_vlm', 'pending_conversion', 'success' # Include VLM statuses
        ]), name='ck_file_index_status')
    )
    def __repr__(self):
        return f"<FileIndex(path='{self.file_path[:50]}...', status='{self.index_status}', hash='{self.md5_hash}')>"

# --- Compression/Decompression Helpers ---
def _compress_db():
    """Compresses the RUNTIME DB to the compressed path and DELETES the runtime DB."""
    if not os.path.exists(RUNTIME_DB_PATH):
        logger.warning("Compression skipped: Runtime DB file not found at shutdown.")
        return False

    logger.info(f"Compressing '{RUNTIME_DB_PATH}' to '{DB_PATH_COMPRESSED}' (Level {ZSTD_COMPRESSION_LEVEL})...")
    start_time = time.monotonic()
    compression_successful = False
    original_size = os.path.getsize(RUNTIME_DB_PATH) if os.path.exists(RUNTIME_DB_PATH) else 0

    try:
        cctx = zstd.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL, threads=-1) # Use multiple threads if available
        with open(RUNTIME_DB_PATH, 'rb') as ifh, open(DB_PATH_COMPRESSED, 'wb') as ofh:
            cctx.copy_stream(ifh, ofh)
        duration = time.monotonic() - start_time
        compressed_size = os.path.getsize(DB_PATH_COMPRESSED) if os.path.exists(DB_PATH_COMPRESSED) else 0
        ratio = compressed_size / original_size if original_size > 0 else 0
        logger.success(f"✅ Compression complete ({duration:.2f}s). Size: {original_size/1024/1024:.2f}MB -> {compressed_size/1024/1024:.2f}MB (Ratio: {ratio:.2f}).")
        compression_successful = True
    except Exception as e:
        logger.error(f"❌ Compression failed: {e}")
        logger.exception("Compression Traceback:")
        # Attempt to clean up partial compressed file if it exists
        if os.path.exists(DB_PATH_COMPRESSED):
            try: os.remove(DB_PATH_COMPRESSED); logger.warning("Removed partial compressed file after error.")
            except Exception as rm_err: logger.error(f"Failed remove partial compressed file: {rm_err}")
        return False

    # Delete Runtime DB only if compression was successful
    if compression_successful:
        logger.info(f"Removing runtime database file: {RUNTIME_DB_PATH}")
        try:
            # Add extra check to prevent deleting non-existent file
            if os.path.exists(RUNTIME_DB_PATH):
                os.remove(RUNTIME_DB_PATH)
                logger.debug("Runtime database file removed.")
            else:
                 logger.warning("Runtime database file already gone before removal attempt.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to remove runtime database file after successful compression: {e}")
            logger.error(f"Compressed file '{DB_PATH_COMPRESSED}' exists, but runtime file '{RUNTIME_DB_PATH}' could not be removed.")
            return False
    else:
        return False # Should have already returned False above


def _decompress_db():
    """Decompresses the compressed DB to the RUNTIME path and DELETES the compressed file."""
    if not os.path.exists(DB_PATH_COMPRESSED):
        logger.debug("Decompression skipped: Compressed DB file not found (normal for first run).")
        return False

    logger.info(f"Decompressing '{DB_PATH_COMPRESSED}' to '{RUNTIME_DB_PATH}'...")
    start_time = time.monotonic()
    decompression_successful = False
    try:
        # Ensure target directory exists
        os.makedirs(os.path.dirname(RUNTIME_DB_PATH), exist_ok=True)

        # Overwrite existing runtime DB if it exists (e.g., from unclean shutdown)
        if os.path.exists(RUNTIME_DB_PATH):
             logger.warning(f"Overwriting existing runtime file before decompression: {RUNTIME_DB_PATH}")
             # No need to explicitly remove, open('wb') truncates

        dctx = zstd.ZstdDecompressor()
        with open(DB_PATH_COMPRESSED, 'rb') as ifh, open(RUNTIME_DB_PATH, 'wb') as ofh:
            dctx.copy_stream(ifh, ofh)
        duration = time.monotonic() - start_time
        logger.success(f"✅ Decompression complete ({duration:.2f}s).")
        decompression_successful = True
    except Exception as e:
        logger.error(f"❌ Decompression failed: {e}")
        logger.exception("Decompression Traceback:")
        # Attempt to clean up potentially partial runtime file
        if os.path.exists(RUNTIME_DB_PATH):
             try: os.remove(RUNTIME_DB_PATH); logger.warning("Removed partial runtime file after failed decompression.")
             except Exception as rm_err: logger.error(f"Failed to remove partial runtime file: {rm_err}")
        return False

    # Delete Compressed DB only if decompression was successful
    if decompression_successful:
        logger.info(f"Removing compressed file: {DB_PATH_COMPRESSED}")
        try:
            if os.path.exists(DB_PATH_COMPRESSED): # Add check
                os.remove(DB_PATH_COMPRESSED)
                logger.debug("Compressed file removed.")
            else:
                logger.warning("Compressed file already gone before removal attempt.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to remove compressed file after successful decompression: {e}")
            logger.error(f"Runtime file '{RUNTIME_DB_PATH}' exists, but compressed file '{DB_PATH_COMPRESSED}' could not be removed.")
            return False
    else:
        return False


def _check_and_repair_db(db_path: str) -> bool:
    """Checks SQLite DB integrity and attempts repair via dump/reload if needed."""
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        logger.debug(f"Integrity check skipped: DB file not found or empty at {db_path}")
        return True # Treat as OK, will be created/populated by Alembic

    logger.info(f"🩺 Performing integrity check on '{os.path.basename(db_path)}'...")
    check_start_time = time.monotonic()
    is_ok = False
    sqlite3_cmd = shutil.which("sqlite3")
    if not sqlite3_cmd:
        logger.error("❌ Cannot run integrity check: 'sqlite3' command not found in PATH.")
        return False

    try:
        # Run integrity check
        check_command = [sqlite3_cmd, db_path, "PRAGMA integrity_check;"]
        logger.debug(f"Running command: {' '.join(check_command)}")
        process = subprocess.run(check_command, capture_output=True, text=True, check=False, timeout=300) # 5 min timeout
        check_output = process.stdout.strip()
        check_duration = time.monotonic() - check_start_time
        logger.info(f"Integrity check completed in {check_duration:.2f}s.")

        if process.returncode != 0:
            logger.error(f"Integrity check process failed (RC={process.returncode}): {process.stderr.strip()}")
            is_ok = False
        elif check_output.lower() == "ok": # Case-insensitive check
            logger.success("✅ Integrity check passed.")
            is_ok = True
        else:
            logger.warning(f"🔥 Integrity check FAILED. Errors reported:\n{check_output}")
            is_ok = False

    except subprocess.TimeoutExpired:
         logger.error("❌ Integrity check timed out after 300 seconds.")
         return False
    except Exception as e:
        logger.error(f"❌ Error during integrity check execution: {e}")
        return False

    # --- Attempt Repair if Check Failed ---
    if not is_ok:
        logger.warning(f"🚨 Attempting automatic repair via dump/reload for '{os.path.basename(db_path)}'...")
        repair_start_time = time.monotonic()
        temp_dump_file = None
        repair_successful = False
        try:
            # 1. Create a temporary file for the dump
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".sql", encoding='utf-8') as tmp_sql:
                temp_dump_file = tmp_sql.name
                logger.debug(f"Dumping database to temporary file: {temp_dump_file}")
                dump_command = [sqlite3_cmd, db_path, ".dump"]
                dump_process = subprocess.run(dump_command, stdout=tmp_sql, stderr=subprocess.PIPE, text=True, check=False, timeout=600)

            if dump_process.returncode != 0:
                 logger.error(f"Database dump FAILED (RC={dump_process.returncode}). Cannot repair.")
                 logger.error(f"Stderr: {dump_process.stderr.strip()}")
                 return False # Cannot proceed

            logger.info("Database dump successful. Recreating database from dump...")

            # 2. Rename corrupted DB
            corrupted_backup_path = f"{db_path}.corrupted_{int(time.time())}"
            logger.warning(f"Renaming corrupted database to: {corrupted_backup_path}")
            try:
                shutil.move(db_path, corrupted_backup_path)
            except Exception as move_err:
                 logger.error(f"❌ Failed to rename corrupted database: {move_err}. Cannot complete repair.")
                 return False

            # 3. Reload dump into a new file
            reload_command = [sqlite3_cmd, db_path]
            logger.debug(f"Reloading dump into new database: {db_path}")
            with open(temp_dump_file, 'r', encoding='utf-8') as dump_fh:
                reload_process = subprocess.run(reload_command, stdin=dump_fh, capture_output=True, text=True, check=False, timeout=600)

            if reload_process.returncode != 0:
                logger.error(f"Database reload from dump FAILED (RC={reload_process.returncode}).")
                logger.error(f"Stderr: {reload_process.stderr.strip()}")
                logger.error(f"DB may be missing at '{db_path}'. Corrupted at '{corrupted_backup_path}'. Dump at '{temp_dump_file}'.")
                return False # Repair failed

            repair_duration = time.monotonic() - repair_start_time
            logger.success(f"✅ Automatic repair attempt finished ({repair_duration:.2f}s).")
            logger.warning("   NOTE: Data loss may have occurred.")

            # Run integrity check again
            logger.info("Running final integrity check on repaired database...")
            final_check_command = [sqlite3_cmd, db_path, "PRAGMA integrity_check;"]
            final_process = subprocess.run(final_check_command, capture_output=True, text=True, check=False, timeout=300)
            final_output = final_process.stdout.strip()
            if final_output.lower() == "ok":
                 logger.success("✅ Repaired database passed final integrity check.")
                 repair_successful = True
            else:
                 logger.error(f"❌ Repaired database FAILED final integrity check! Errors:\n{final_output}")
                 repair_successful = False

        except subprocess.TimeoutExpired as te:
             logger.error(f"❌ Repair process (dump/reload) timed out: {te}")
             repair_successful = False
        except Exception as repair_err:
            logger.error(f"❌ Error during automatic repair: {repair_err}")
            logger.exception("Repair Traceback:")
            repair_successful = False
        finally:
            # Clean up temporary dump file
            if temp_dump_file and os.path.exists(temp_dump_file):
                try: os.remove(temp_dump_file); logger.debug(f"Removed temporary dump file.")
                except Exception as del_err: logger.warning(f"Could not remove tmp dump file '{temp_dump_file}': {del_err}")

        return repair_successful

    return True # Return True if check passed initially

# --- Shutdown Hook ---
def _shutdown_hook():
    """Handles clean shutdown: Compresses runtime DB, then deletes it."""
    logger.info("Executing database shutdown hook...")
    global _engine
    if _engine:
        try:
            logger.debug("Disposing SQLAlchemy engine pool...")
            _engine.dispose()
            logger.debug("Engine pool disposed.")
        except Exception as e:
            logger.warning(f"Error disposing engine pool during shutdown: {e}")

    # Compress the runtime DB (which also deletes it on success)
    compression_final_success = _compress_db()

    if compression_final_success:
        logger.info("Database compressed and runtime file removed successfully.")
    else:
        logger.error("Shutdown hook: Compression and/or runtime file cleanup failed!")
        logger.error(f"Ensure data is saved. Check '{RUNTIME_DB_PATH}' and '{DB_PATH_COMPRESSED}'.")
    logger.info("Database shutdown hook finished.")

atexit.register(_shutdown_hook)

# --- Engine Creation Wrapper ---
def get_engine():
    """Manages DB decompression, integrity check/repair, and returns the engine."""
    global _engine, SessionLocal
    if _engine is None:
        logger.info("Initializing SQLAlchemy engine (Direct Runtime DB Mode)...")

        compressed_exists = os.path.exists(DB_PATH_COMPRESSED)
        runtime_exists = os.path.exists(RUNTIME_DB_PATH)

        # Decompress if compressed file exists
        if compressed_exists:
            logger.info("Compressed database found, attempting decompression...")
            if not _decompress_db(): # Decompresses to RUNTIME_DB_PATH
                logger.critical("🔥🔥 FATAL: Failed to decompress database. Cannot start.")
                sys.exit(1)
            runtime_exists = True
        elif runtime_exists:
            logger.warning(f"Using existing runtime DB '{os.path.basename(RUNTIME_DB_PATH)}' (Compressed file not found).")
        else:
            logger.info("No existing database found (.db or .zst). Will create new.")

        # Check Integrity and Repair (if runtime DB exists)
        if runtime_exists:
            if not _check_and_repair_db(RUNTIME_DB_PATH):
                 logger.critical(f"🔥🔥 FATAL: DB integrity check/repair failed for '{os.path.basename(RUNTIME_DB_PATH)}'.")
                 try: # Attempt removal
                     logger.warning(f"Attempting to remove corrupted runtime DB file: {RUNTIME_DB_PATH}")
                     os.remove(RUNTIME_DB_PATH)
                     logger.info("Removed corrupted runtime DB. Will attempt fresh creation.")
                     runtime_exists = False # Ensure Alembic creates it
                 except Exception as rm_err:
                      logger.critical(f"Failed to remove corrupted DB file: {rm_err}")
                      sys.exit(1) # Exit if repair fails and removal fails
            else:
                 logger.info("Database integrity check/repair passed.")
        else:
             logger.info("Skipping integrity check as runtime DB does not exist (will be created).")

        # Create Engine pointing to the RUNTIME DB file
        # Use longer timeout for SQLite
        engine_args_internal = {"echo": False, "connect_args": {"check_same_thread": False, "timeout": 60.0}}
        logger.info(f"Creating SQLAlchemy engine for runtime DB: {RUNTIME_DATABASE_URL}")
        try:
            _engine = create_engine(RUNTIME_DATABASE_URL, **engine_args_internal)
            # Test connection immediately
            with _engine.connect() as connection:
                logger.debug("Engine connection test successful.")
            # Configure SessionLocal AFTER engine is confirmed working
            SessionLocal.configure(bind=_engine)
            logger.info("SQLAlchemy SessionLocal configured and bound to runtime DB engine.")
        except Exception as e:
            logger.critical(f"🔥🔥 DATABASE ENGINE CREATION FAILED for runtime DB: {e}")
            logger.exception("Engine Creation Traceback:")
            sys.exit(1)

    # Final check on SessionLocal binding
    if SessionLocal is None or not SessionLocal.kw.get('bind'):
        logger.error("SessionLocal is None or was not bound to an engine after get_engine() call!")
        raise RuntimeError("Failed to configure SessionLocal binding")

    return _engine

# --- Alembic Helper Functions ---
def _get_alembic_config() -> Optional[Config]:
    """Loads the Alembic configuration, ensuring the URL points to the RUNTIME DB."""
    if not os.path.exists(ALEMBIC_INI_PATH):
        logger.error(f"Alembic configuration file not found at: {ALEMBIC_INI_PATH}")
        return None
    try:
        alembic_cfg = Config(ALEMBIC_INI_PATH)
        # Use the RUNTIME_DATABASE_URL defined at the top of this file
        alembic_cfg.set_main_option("sqlalchemy.url", RUNTIME_DATABASE_URL)
        # Ensure script location is correct relative to APP_DIR (where database.py is)
        alembic_cfg.set_main_option("script_location", os.path.relpath(ALEMBIC_DIR, APP_DIR).replace("\\", "/"))
        logger.debug("Alembic config loaded and verified for runtime DB.")
        return alembic_cfg
    except Exception as e:
        logger.error(f"Failed to load or configure Alembic Config from {ALEMBIC_INI_PATH}: {e}")
        return None

def _create_default_env_py():
    """Creates a robust default alembic env.py if it doesn't exist."""
    if not os.path.exists(ALEMBIC_ENV_PY_PATH):
        logger.warning(f"🔧 Alembic env.py not found. Creating default at {ALEMBIC_ENV_PY_PATH}")
        # Use RUNTIME_DATABASE_URL in the generated content
        # Ensure relative import works from env.py back to database.py
        env_py_content = f"""
import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool, create_engine
from alembic import context

# --- Detect Project Root and Add to Path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is alembic/
APP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) # This should be engineMain/
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
print(f"Alembic env.py: Added {{APP_DIR}} to sys.path")

# --- Import Base and Models ---
target_metadata = None
RUNTIME_DATABASE_URL_FROM_DB_PY = None
try:
    # Import Base AND URL from the database module (sibling to alembic dir)
    from database import Base, RUNTIME_DATABASE_URL
    RUNTIME_DATABASE_URL_FROM_DB_PY = RUNTIME_DATABASE_URL
    print(f"Alembic env.py: Imported Base. Using RUNTIME DB URL: {{RUNTIME_DATABASE_URL_FROM_DB_PY}}")

    # IMPORTANT: Import all models defined in database.py or models.py
    # This ensures they are registered with Base.metadata for autogenerate
    from database import Interaction, AppleScriptAttempt, FileIndex # Adjust if models are elsewhere
    print(f"Alembic env.py: Explicitly imported models.")

    target_metadata = Base.metadata # Assign metadata AFTER models are imported
    print(f"Alembic env.py: Assigned Base.metadata.")

except ImportError as import_err:
    print(f"Alembic env.py: ERROR importing from database.py: {{import_err}}")
    print(f"Alembic env.py: Check file structure and ensure models are importable.")
except Exception as e:
    print(f"Alembic env.py: UNEXPECTED ERROR during import: {{e}}")

# --- Alembic Configuration ---
config = context.config

# Ensure target_metadata is loaded before proceeding
if target_metadata is None:
    raise RuntimeError("Alembic env.py: target_metadata not loaded. Check imports and model definitions in database.py.")

# Set the effective URL from the imported variable
effective_db_url = RUNTIME_DATABASE_URL_FROM_DB_PY
if not effective_db_url:
    # Fallback to ini file if import failed, but warn heavily
    print("Alembic env.py: WARNING - Failed to import URL from database.py, falling back to alembic.ini URL.")
    effective_db_url = config.get_main_option("sqlalchemy.url")
    if not effective_db_url:
         raise ValueError("Alembic env.py: Database URL not found in database.py or alembic.ini.")

print(f"Alembic env.py: Configuring Alembic with effective URL: {{effective_db_url}}")
config.set_main_option("sqlalchemy.url", effective_db_url)

# Interpret the config file for Python logging.
# This line needs to be placed after configuring the URL.
if config.config_file_name is not None:
    try:
        fileConfig(config.config_file_name)
    except Exception as fc_err:
        print(f"Alembic env.py: Error processing logging config from alembic.ini: {{fc_err}}")


# Other settings from the environment, ensure batch mode for SQLite
is_sqlite = effective_db_url.startswith("sqlite")
print(f"Alembic env.py: Detected SQLite: {{is_sqlite}}")

def run_migrations_offline() -> None:
    \"\"\"Run migrations in 'offline' mode.\"\"\"
    print("Running migrations offline...")
    if not effective_db_url: raise ValueError("DB URL not set for offline mode.")
    context.configure(
        url=effective_db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={{"paramstyle": "named"}},
        render_as_batch=is_sqlite # Use batch mode for SQLite offline
    )
    with context.begin_transaction():
        context.run_migrations()
    print("Offline migrations finished.")


def run_migrations_online() -> None:
    \"\"\"Run migrations in 'online' mode.\"\"\"
    print("Running migrations online...")
    if not effective_db_url: raise ValueError("DB URL not set for online mode.")

    connectable_args = {{}}
    # Specific connect args for SQLite in online mode
    if is_sqlite:
        connectable_args["connect_args"] = {{"check_same_thread": False, "timeout": 60.0}} # Longer timeout for migrations
        print("Applying SQLite connect_args for online migration.")

    # Create engine specifically for migration
    connectable = create_engine(
        effective_db_url,
        poolclass=pool.NullPool, # Avoid pooling for migrations
        **connectable_args
    )
    print("Migration engine created.")
    try:
        with connectable.connect() as connection:
            print("Established connection for online migration.")
            # Configure context, MUST include render_as_batch for SQLite ALTER support
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                render_as_batch=is_sqlite, # <<< CRITICAL FOR SQLITE ALTERS >>>
                compare_type=True # Compare column types during autogenerate checks
            )
            print("Beginning transaction and running migrations...")
            try:
                with context.begin_transaction():
                    context.run_migrations()
                print("Online migrations completed successfully.")
            except Exception as mig_err:
                print(f"Alembic online migration execution ERROR: {{mig_err}}")
                raise # Re-raise to be caught by caller if needed
    except Exception as conn_err:
        print(f"Alembic online database connection ERROR: {{conn_err}}")
        raise # Re-raise
    finally:
        if 'connectable' in locals() and connectable:
             connectable.dispose()
             print("Migration engine disposed.")


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

print("Alembic env.py finished.")
"""
        try:
            os.makedirs(os.path.dirname(ALEMBIC_ENV_PY_PATH), exist_ok=True)
            with open(ALEMBIC_ENV_PY_PATH, 'w') as f:
                f.write(env_py_content)
            logger.success(f"✅ Default env.py created/overwritten.")
        except IOError as e:
            logger.error(f"❌ Failed to write default env.py: {e}")
            raise # Fail init if this happens

def _create_default_script_mako():
    """Creates the default Alembic script template if it's missing."""
    if not os.path.exists(ALEMBIC_SCRIPT_MAKO_PATH):
        logger.warning(f"🔧 Alembic script template not found. Creating default at {ALEMBIC_SCRIPT_MAKO_PATH}")
        mako_content = """\"\"\"${message}\"\"\"
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    ${upgrades if upgrades else "pass"}
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    ${downgrades if downgrades else "pass"}
    # ### end Alembic commands ###
"""
        try:
            with open(ALEMBIC_SCRIPT_MAKO_PATH, 'w') as f:
                f.write(mako_content)
            logger.success(f"✅ Default script.py.mako created.")
        except IOError as e:
            logger.error(f"❌ Failed to write default script.py.mako: {e}")
            raise # Fail init

# --- Migration Application Function ---
def check_and_apply_migrations() -> bool: # Added return type hint
    """Applies pending Alembic migrations to the RUNTIME database."""
    logger.info("🔬 Applying pending Alembic migrations (to runtime DB)...")
    alembic_cfg = _get_alembic_config() # Ensures URL points to runtime DB
    if not alembic_cfg:
        logger.error("Cannot apply migrations: Alembic config failed.")
        return False # Indicate failure
    try:
        logger.info("Running 'alembic upgrade head'...")
        # Use the programmatic API
        command.upgrade(alembic_cfg, "head")
        logger.success("✅ 'alembic upgrade head' command finished successfully (or DB was already at head).")
        return True # Indicate success
    except CommandError as ce:
        logger.error(f"❌ Alembic 'upgrade' command failed: {ce}")
        err_str = str(ce).lower()
        # Add specific SQLite advice
        if "sqlite" in err_str and ("alter table" in err_str or "add column" in err_str):
             logger.critical("   >>> Potential SQLite migration failure! Ensure 'render_as_batch=True' in alembic/env.py. <<<")
        if "already exists" in err_str:
             logger.critical("   >>> 'Table already exists' error during upgrade. Alembic state might be inconsistent. <<<")
        else:
             # General advice if specific patterns aren't matched
             logger.error("   Manual Alembic intervention likely required. Check logs and Alembic state.")
        return False # Indicate failure
    except Exception as upg_err:
        logger.error(f"❌ Unexpected error during 'alembic upgrade': {upg_err}")
        logger.exception("Upgrade Error Traceback:")
        return False # Indicate failure

# --- init_db ---
def init_db():
    """
    Initializes the Database: Handles decompression, ensures schema is up-to-date
    via Alembic migrations (auto-generating initial if needed), sets up engine/session
    for the RUNTIME DB, and logs stats.
    """
    logger.info("🚀 Initializing Database (Direct Runtime DB Mode)...")
    global _engine, SessionLocal

    # Prevent re-initialization
    if _engine is not None and SessionLocal is not None and SessionLocal.kw.get('bind'):
        logger.warning("Database already initialized. Skipping re-initialization.")
        return

    engine_instance = None
    migration_successful = False

    try:
        # --- Step 1: Ensure Engine/Session Factory exist ---
        engine_instance = get_engine() # Handles decompression, integrity, sets globals
        if not engine_instance:
             raise RuntimeError("get_engine() failed to return a valid engine instance.")
        logger.info("Engine initialization/check complete.")

        # --- Step 2: Ensure Alembic config files exist ---
        _create_default_env_py()  # Create/verify env.py FIRST
        _create_default_script_mako() # Verify mako template exists

        # Verify/create alembic.ini last, as it depends on paths
        if not os.path.exists(ALEMBIC_INI_PATH):
            logger.warning(f"🔧 Alembic config file not found. Creating default at {ALEMBIC_INI_PATH}")
            alembic_dir_for_ini = os.path.relpath(ALEMBIC_DIR, APP_DIR).replace("\\", "/")
            alembic_cfg_content = f"""[alembic]
script_location = {alembic_dir_for_ini}
sqlalchemy.url = {RUNTIME_DATABASE_URL}

[loggers]
keys = root,sqlalchemy,alembic
[handlers]
keys = console
[formatters]
keys = generic
[logger_root]
level = WARN
handlers = console
qualname =
[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine
[logger_alembic]
level = INFO
handlers =
qualname = alembic
[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic
[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""
            try:
                with open(ALEMBIC_INI_PATH, 'w') as f: f.write(alembic_cfg_content)
                logger.success(f"📝 Default alembic.ini created.")
            except IOError as e: logger.error(f"❌ Failed to write default alembic.ini: {e}"); raise
        logger.info("Alembic configuration files ensured.")


        # --- Step 3: Check/Run Initial Auto-Generation ---
        try:
            # Check if versions directory exists AND is empty (ignoring __pycache__)
            versions_exist = os.path.isdir(ALEMBIC_VERSIONS_PATH)
            version_files = []
            if versions_exist:
                version_files = [f for f in os.listdir(ALEMBIC_VERSIONS_PATH) if f.endswith('.py')]

            if not versions_exist or not version_files:
                if not versions_exist:
                    logger.warning(f"Alembic versions directory missing at {ALEMBIC_VERSIONS_PATH}, creating...")
                    os.makedirs(ALEMBIC_VERSIONS_PATH, exist_ok=True)
                logger.warning("🤔 Alembic versions directory empty. Attempting auto-generate initial migration...")
                autogen_success = False
                try:
                    # Run using subprocess module approach
                    command_args = [
                        sys.executable, "-m", "alembic",
                        "-c", ALEMBIC_INI_PATH, # Explicitly use config
                        "revision", "--autogenerate",
                        "-m", "Auto-generated initial schema"
                    ]
                    logger.info(f"Running command: {' '.join(command_args)}")
                    # Run from APP_DIR (where database.py is, allowing env.py imports)
                    process = subprocess.run(
                        command_args, cwd=APP_DIR,
                        capture_output=True, text=True, check=True, timeout=60
                    )
                    logger.info("Alembic Autogenerate Output:\n" + process.stdout)
                    if process.stderr: logger.warning("Alembic Autogenerate Stderr:\n" + process.stderr)
                    logger.success("✅ Successfully auto-generated initial migration script.")
                    autogen_success = True
                except subprocess.CalledProcessError as cpe:
                     logger.error(f"❌ Failed auto-generate migration (RC: {cpe.returncode}).")
                     logger.error(f"   Cmd: {' '.join(cpe.cmd)}")
                     logger.error(f"   Stdout: {cpe.stdout}")
                     logger.error(f"   Stderr: {cpe.stderr}")
                     logger.critical("   Check imports/metadata in alembic/env.py & model definitions!")
                except FileNotFoundError: logger.error(f"❌ Failed auto-generate: '{sys.executable} -m alembic' not found.")
                except subprocess.TimeoutExpired: logger.error("❌ Auto-generate timed out.")
                except Exception as auto_err: logger.error(f"❌ Unexpected error during auto-generation: {auto_err}")

                if not autogen_success:
                    logger.critical("‼️ Failed to create initial migration. DB schema likely incomplete.")
            else:
                logger.debug(f"Found existing migration scripts. Skipping auto-generation.")

        except Exception as check_err:
            logger.error(f"❌ Error checking/running initial auto-generation: {check_err}")


        # --- Step 4: Apply All Available Migrations ---
        migration_successful = check_and_apply_migrations()


        # --- Step 5: Log Schema Statistics (Verification) ---
        if not migration_successful:
            logger.warning("Skipping schema verification as migration step failed or reported errors.")
        else:
            logger.info("📊 Logging Database Schema Statistics (Verification)...")
            try:
                if not engine_instance: raise RuntimeError("Engine lost before inspection.")
                inspector = sql_inspect(engine_instance)
                table_names = inspector.get_table_names()
                logger.info(f"  Tables ({len(table_names)}): {', '.join(table_names)}")
                expected_tables = {'alembic_version', 'interactions', 'applescript_attempts', 'file_index'}
                found_tables = set(table_names); missing_tables = expected_tables - found_tables
                if missing_tables:
                     logger.error(f"‼️ Expected tables MISSING: {', '.join(missing_tables)}")
                     if 'alembic_version' in missing_tables: logger.critical("   'alembic_version' MISSING - Migrations failed!")
                     migration_successful = False # Overall init failed
                total_columns = 0
                if table_names:
                    with engine_instance.connect() as connection:
                        for table_name in table_names:
                             try:
                                columns = inspector.get_columns(table_name)
                                column_names = [c['name'] for c in columns]; col_count = len(columns); total_columns += col_count
                                row_count = -1
                                try: row_count = connection.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar_one()
                                except: pass # Ignore count errors
                                logger.info(f"    Table '{table_name}': Rows={row_count}, Cols={col_count} ({', '.join(column_names)})")
                             except Exception as ti_err: logger.error(f"Err inspecting '{table_name}': {ti_err}")
                    logger.info(f"  Total Columns: {total_columns}")
                else:
                     logger.warning("  No tables found. Migrations likely failed.")
                     migration_successful = False
            except Exception as inspect_err:
                logger.error(f"❌ Schema inspection failed: {inspect_err}")
                migration_successful = False

        # --- Final Init Log ---
        if migration_successful:
            logger.success("✅ Database initialization sequence completed successfully.")
        else:
            logger.critical("🔥🔥 Database initialization FAILED or schema verification failed.")
            raise RuntimeError("Database initialization failed migration or verification.")

    except Exception as init_err:
        logger.critical(f"🔥🔥 UNEXPECTED FAILURE during database initialization: {init_err}")
        logger.exception("Initialization Traceback:")
        sys.exit("CRITICAL: Database Initialization Failure")


# --- Database Interaction Functions ---
# (Keep add_interaction, get_recent_interactions, get_pending_tot_result,
#  mark_tot_delivered, get_past_tot_interactions, get_global_recent_interactions,
#  get_past_applescript_attempts, search_file_index functions exactly as provided before)

def add_interaction(db: Session, **kwargs) -> Optional[Interaction]:
    """Saves interaction or log entry to the interactions table."""
    interaction: Optional[Interaction] = None
    try:
        valid_keys = {c.name for c in Interaction.__table__.columns}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        # Set sensible defaults if not provided
        filtered_kwargs.setdefault('mode', 'chat')
        if 'input_type' not in filtered_kwargs:
            if filtered_kwargs.get('user_input'): filtered_kwargs['input_type'] = 'text'
            elif filtered_kwargs.get('llm_response'): filtered_kwargs['input_type'] = 'llm_response'
            else: filtered_kwargs['input_type'] = 'log_info' # Default to info log

        # Ensure boolean defaults are correct for SQLite (0/1) or handled by SQLAlchemy
        filtered_kwargs.setdefault('reflection_completed', False)
        filtered_kwargs.setdefault('tot_delivered', False)
        filtered_kwargs.setdefault('tot_analysis_requested', False)
        filtered_kwargs.setdefault('assistant_action_executed', False)
        filtered_kwargs.setdefault('requires_deep_thought', None) # Allow Null

        interaction = Interaction(**filtered_kwargs)
        db.add(interaction)
        db.commit()
        db.refresh(interaction)

        # Simplified logging
        log_level = "INFO"
        log_extra = ""
        if interaction.input_type == 'error' or interaction.input_type == 'log_error':
            log_level = "ERROR"
            log_extra = " ❌ ERROR"
        elif interaction.input_type == 'log_warning':
             log_level = "WARNING"
             log_extra = " ⚠️ WARN"

        logger.log(log_level, f"💾 Interaction {interaction.id} ({interaction.mode}/{interaction.input_type}){log_extra}")
        return interaction

    except SQLAlchemyError as e:
        logger.error(f"❌ Error saving interaction: {e}")
        err_str = str(e).lower()
        if isinstance(e, (OperationalError, ProgrammingError)) and ("column" in err_str or "table" in err_str):
             logger.critical(f"SCHEMA MISMATCH during save: {e}. Run migrations!")
        else:
             logger.exception("DB Save Traceback:")
        try:
            db.rollback()
            logger.warning("Database transaction rolled back.")
        except Exception as rb_err:
            logger.error(f"Rollback failed: {rb_err}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error saving interaction: {e}")
        logger.exception("Unexpected DB Save Traceback:")
        try:
            db.rollback()
        except Exception as rb_err:
            logger.error(f"Rollback failed: {rb_err}")
        return None

def get_recent_interactions(db: Session, limit=5, session_id=None, mode="chat", include_logs=False) -> List[Interaction]:
    """Gets recent interactions for a session/mode, optionally including logs."""
    query_desc = f"session '{session_id}', mode '{mode}', limit {limit}, logs={include_logs}"
    logger.debug(f"Fetching recent interactions: {query_desc}")
    try:
        base_query = db.query(Interaction).filter(Interaction.mode == mode)
        if session_id:
            base_query = base_query.filter(Interaction.session_id == session_id)

        query = base_query # Start with the base query

        if not include_logs:
            log_types_to_exclude = ['log_warning', 'log_error', 'log_debug', 'log_info', 'error', 'system', 'url', 'image', 'latex_analysis_result'] # Exclude more types
            query = query.filter(Interaction.input_type.notin_(log_types_to_exclude))

        results = query.order_by(desc(Interaction.timestamp)).limit(limit).all()

        # If logs *are* requested, fetch them separately and merge/sort if needed (more complex)
        # For simplicity now, the above logic handles exclusion correctly. If merge needed:
        # if include_logs and len(results) < limit:
        #    ... (fetch log entries query) ...
        #    ... (merge and sort results) ...

        logger.debug(f"Fetched {len(results)} interactions for {query_desc}.")
        results.reverse() # Return oldest first for context flow
        return results
    except (OperationalError, ProgrammingError) as e:
        logger.critical(f"DATABASE SCHEMA MISMATCH! Query failed: {e}. Run Alembic migrations!")
        logger.exception("DB Fetch Traceback:")
        return []
    except Exception as e:
        logger.error(f"Error fetching recent interactions for {query_desc}: {e}")
        logger.exception("DB Fetch Traceback:")
        return []

def get_pending_tot_result(db: Session, session_id: str) -> Optional[Interaction]:
    """Finds the most recent undelivered, successful ToT result for the session."""
    logger.debug(f"Checking for pending ToT result for session '{session_id}'")
    try:
        # Ensure tot_delivered column exists before querying
        if 'tot_delivered' not in Interaction.__table__.columns:
            logger.error("Schema Error: 'tot_delivered' column missing. Run migrations.")
            return None

        result = db.query(Interaction).filter(
            Interaction.session_id == session_id,
            Interaction.mode == 'chat',
            Interaction.tot_analysis_requested == True,
            Interaction.tot_result.isnot(None),
            Interaction.tot_result.notlike('Error%'),
            # Use standard boolean comparison
            Interaction.tot_delivered == False
        ).order_by(desc(Interaction.timestamp)).first()

        if result: logger.info(f"Found pending ToT result from Interaction ID: {result.id}")
        else: logger.debug("No pending ToT result found.")
        return result
    except (OperationalError, ProgrammingError) as e:
        err_str = str(e).lower()
        if "no such column" in err_str and "tot_" in err_str: logger.critical(f"DB SCHEMA MISMATCH (ToT)! Query failed: {e}. Run migrations!")
        else: logger.error(f"Error fetching pending ToT: {e}"); logger.exception("Pending ToT Fetch Traceback:")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching pending ToT for session '{session_id}': {e}")
        logger.exception("Pending ToT Fetch Traceback:")
        return None

def mark_tot_delivered(db: Session, interaction_id: int):
    """Marks a ToT interaction result as delivered in the database."""
    logger.debug(f"Attempting to mark ToT delivered for Interaction ID {interaction_id}")
    try:
        # Ensure tot_delivered column exists before updating
        if 'tot_delivered' not in Interaction.__table__.columns:
            logger.error("Schema Error: 'tot_delivered' column missing. Cannot mark. Run migrations.")
            return

        stmt = update(Interaction).where(Interaction.id == interaction_id).values(tot_delivered=True)
        result = db.execute(stmt)
        db.commit()
        if result.rowcount > 0: logger.info(f"Marked ToT result for Interaction ID {interaction_id} as delivered.")
        else: logger.warning(f"No row updated marking ToT delivered for ID {interaction_id}.")
    except (OperationalError, ProgrammingError) as e:
        err_str = str(e).lower()
        if "no such column" in err_str and "tot_delivered" in err_str: logger.critical(f"DB SCHEMA MISMATCH! Cannot mark ToT: {e}. Run migrations!")
        else: logger.error(f"Error marking ToT delivered: {e}"); logger.exception("Mark ToT Delivered Traceback:")
        try: db.rollback()
        except Exception as rb_err: logger.error(f"Rollback failed: {rb_err}")
    except Exception as e:
        logger.error(f"Unexpected error marking ToT delivered for ID {interaction_id}: {e}")
        logger.exception("Mark ToT Delivered Traceback:")
        try: db.rollback()
        except Exception as rb_err: logger.error(f"Rollback failed: {rb_err}")

def get_past_tot_interactions(db: Session, limit=50) -> List[Interaction]:
    """Gets past successfully completed ToT interactions for chat mode."""
    logger.debug(f"Fetching past {limit} completed ToT interactions.")
    try:
        # Check required columns exist
        required_cols = {'tot_analysis_requested', 'tot_result'}
        if not required_cols.issubset(Interaction.__table__.columns.keys()):
            logger.error("Schema Error: ToT columns missing. Run migrations.")
            return []

        results = db.query(Interaction).filter(
                    Interaction.mode == "chat",
                    Interaction.tot_analysis_requested == True,
                    Interaction.tot_result.isnot(None),
                    Interaction.tot_result.notlike('Error%')
                ).order_by(desc(Interaction.timestamp)).limit(limit).all()
        logger.debug(f"Found {len(results)} past completed ToT interactions.")
        return results
    except (OperationalError, ProgrammingError) as e:
        err_str = str(e).lower()
        if "no such column" in err_str and "tot_" in err_str: logger.critical(f"DB SCHEMA MISMATCH (ToT)! Query failed: {e}. Run migrations!")
        else: logger.error(f"Error fetching past ToT: {e}"); logger.exception("Past ToT Fetch Traceback:")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching past ToT: {e}")
        logger.exception("Past ToT Fetch Traceback:")
        return []

def get_global_recent_interactions(db: Session, limit: int = 5) -> List[Interaction]:
    """Gets the most recent global chat interactions (User input / LLM response pairs)."""
    logger.debug(f"Fetching last {limit} global chat interactions (text/llm_response/image+text).")
    try:
        results = db.query(Interaction).filter(
            Interaction.mode == 'chat',
            Interaction.input_type.in_(['text', 'llm_response', 'image+text']) # Filter for conversational turns
        ).order_by(desc(Interaction.timestamp)).limit(limit).all()
        logger.debug(f"Fetched {len(results)} global interactions.")
        results.reverse() # Oldest first for chronological display
        return results
    except (OperationalError, ProgrammingError) as e:
        logger.critical(f"DATABASE SCHEMA MISMATCH! Query failed: {e}. Run migrations!")
        logger.exception("DB Fetch Traceback:")
        return []
    except Exception as e:
        logger.error(f"Error fetching global recent interactions: {e}")
        logger.exception("Global History Fetch Traceback:")
        return []

def get_past_applescript_attempts(db: Session, action_type: str, parameters_json: str, limit: int = 5) -> List[AppleScriptAttempt]:
    """Retrieves recent successful/failed AppleScript attempts for a similar action/params."""
    engine_instance = get_engine() # Ensure engine is ready
    if not engine_instance: logger.error("Cannot get past AppleScript attempts: Engine not initialized."); return []
    logger.debug(f"Fetching past {limit} AppleScript attempts for action '{action_type}'")
    # Check if table exists before querying
    try:
        inspector = sql_inspect(engine_instance)
        if not inspector.has_table(AppleScriptAttempt.__tablename__):
            logger.error(f"Table '{AppleScriptAttempt.__tablename__}' missing. Run migrations."); return []
    except Exception as e: logger.error(f"Inspector failed checking for {AppleScriptAttempt.__tablename__}: {e}"); return []

    try:
        results = db.query(AppleScriptAttempt).filter(
            AppleScriptAttempt.action_type == action_type,
            AppleScriptAttempt.parameters_json == parameters_json # Exact match on serialized params
        ).order_by(desc(AppleScriptAttempt.timestamp)).limit(limit).all()
        logger.debug(f"Found {len(results)} past attempts for exact action/params: '{action_type}'.")
        return results
    except (OperationalError, ProgrammingError) as e:
        logger.critical(f"DB SCHEMA MISMATCH (AppleScriptAttempts)! Query failed: {e}. Run migrations!")
        logger.exception("Fetch AppleScript Attempts Traceback:")
        return []
    except Exception as e:
        logger.error(f"Error fetching past AppleScript attempts: {e}")
        logger.exception("Fetch AppleScript Attempts Traceback:")
        return []

def search_file_index(db: Session, query: str, limit: int = 10) -> List[FileIndex]:
    """Basic search across file paths and indexed content."""
    engine_instance = get_engine() # Ensure engine is ready
    if not engine_instance: logger.error("Cannot search file index: Engine not initialized."); return []
    logger.debug(f"Searching file index for '{query}' (limit {limit})")
    # Check if table exists
    try:
        inspector = sql_inspect(engine_instance)
        if not inspector.has_table(FileIndex.__tablename__):
            logger.error(f"Cannot search file index: Table '{FileIndex.__tablename__}' does not exist. Run migrations."); return []
    except Exception as inspect_e: logger.error(f"Inspector failed checking for {FileIndex.__tablename__}: {inspect_e}"); return []

    try:
        search_term = f"%{query}%" # Simple LIKE search
        results = db.query(FileIndex).filter(
            (FileIndex.file_path.like(search_term)) |
            (FileIndex.indexed_content.like(search_term)) # Search content too
        ).order_by(
            desc(FileIndex.last_modified_os) # Prioritize recently modified? Or add relevance score later?
        ).limit(limit).all()
        logger.debug(f"Found {len(results)} potential matches in file index.")
        return results
    except (OperationalError, ProgrammingError) as e:
        logger.critical(f"DB SCHEMA MISMATCH (FileIndex)! Query failed: {e}. Run migrations!")
        logger.exception("File Index Search Traceback:")
        return []
    except Exception as e:
        logger.error(f"Error searching file index: {e}")
        logger.exception("File Index Search Traceback:")
        return []

# --- End of database.py ---