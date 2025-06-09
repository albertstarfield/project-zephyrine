# database.py (V17 - With Background Snapshotting & Restore)

import os
import sys
import datetime
import time
import atexit
import shutil
import hashlib
import subprocess
import tempfile
import threading  # For the snapshotter thread

from sqlalchemy.cyextension import collections

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
from typing import List, Optional, Dict, Any, Tuple  # Added Tuple
from sqlalchemy.sql import func
from sqlalchemy.dialects.sqlite import DATETIME as SQLITE_DATETIME

# Alembic imports
from alembic.config import Config
from alembic import command
# from alembic.script import ScriptDirectory # Not directly used in this file after env.py is set up
# from alembic.runtime.environment import EnvironmentContext # Not directly used
# from alembic.runtime.migration import MigrationContext # Not directly used
from alembic.util import CommandError
from collections import deque
from loguru import logger
from config import *

# --- Configuration & Paths ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_BASENAME = "mappedknowledge.db"
RUNTIME_DB_PATH = os.path.join(APP_DIR, DB_BASENAME)
DB_PATH_COMPRESSED = f"{RUNTIME_DB_PATH}.zst"
RUNTIME_DATABASE_URL = f"sqlite:///{os.path.abspath(RUNTIME_DB_PATH)}"

# --- NEW: Snapshot Configuration ---
ENABLE_DB_SNAPSHOTS = os.getenv("ENABLE_DB_SNAPSHOTS", "true").lower() in ('true', '1', 't', 'yes', 'y')
DB_SNAPSHOT_INTERVAL_MINUTES = int(os.getenv("DB_SNAPSHOT_INTERVAL_MINUTES", 10))  # Default: 1 minute
DB_SNAPSHOT_DIR_NAME = "db_snapshots"
DB_SNAPSHOT_DIR = os.path.join(APP_DIR, DB_SNAPSHOT_DIR_NAME)
DB_SNAPSHOT_RETENTION_COUNT = int(
    os.getenv("DB_SNAPSHOT_RETENTION_COUNT", 30))  # Keep last 60 snapshots (e.g., 1 hour worth if 1-min interval)
DB_SNAPSHOT_FILENAME_PREFIX = "snapshot_"
DB_SNAPSHOT_FILENAME_SUFFIX = ".db.zst"
# --- END NEW: Snapshot Configuration ---

ZSTD_COMPRESSION_LEVEL = int(os.getenv("ZSTD_COMPRESSION_LEVEL", 9))

logger.info(f"⚙️ Runtime DB Path: {RUNTIME_DB_PATH}")
logger.info(f"📦 Compressed DB Path (Shutdown Archive): {DB_PATH_COMPRESSED}")
logger.info(f"🔗 Runtime DB URL: {RUNTIME_DATABASE_URL}")
if ENABLE_DB_SNAPSHOTS:
    logger.info(
        f"📸 DB Snapshots Enabled: Dir='{DB_SNAPSHOT_DIR}', Interval={DB_SNAPSHOT_INTERVAL_MINUTES}min, Retention={DB_SNAPSHOT_RETENTION_COUNT}")
else:
    logger.info("📸 DB Snapshots Disabled.")

if not ZSTD_AVAILABLE:
    logger.critical("🔥🔥 zstandard library not found! Install with 'pip install zstandard'. Exiting.")
    sys.exit(1)

ALEMBIC_DIR = os.path.join(APP_DIR, 'alembic')
ALEMBIC_INI_PATH = os.path.join(APP_DIR, 'alembic.ini')
ALEMBIC_VERSIONS_PATH = os.path.join(ALEMBIC_DIR, 'versions')
ALEMBIC_ENV_PY_PATH = os.path.join(ALEMBIC_DIR, 'env.py')
ALEMBIC_SCRIPT_MAKO_PATH = os.path.join(ALEMBIC_DIR, "script.py.mako")

Base = declarative_base()
_engine = None
SessionLocal = sessionmaker(autocommit=False, autoflush=False)
logger.debug("SessionLocal factory structure created (unbound initially).")


# --- Database Models (Interaction, AppleScriptAttempt, FileIndex) ---
class UploadedFileRecord(Base):
    __tablename__ = "uploaded_file_records"
    id = Column(Integer, primary_key=True)
    ingestion_id = Column(String, unique=True, index=True, nullable=False) # UUID from app.py
    original_filename = Column(String, nullable=False)
    stored_path = Column(String, nullable=False) # Full path to the temporarily stored file
    purpose = Column(String, nullable=False) # e.g., "fine-tune"
    status = Column(String, default="received", nullable=False) # "received", "queued_for_processing", "processing", "completed", "failed"
    processed_entries_count = Column(Integer, default=0) # Number of rows/lines processed
    spawned_tasks_count = Column(Integer, default=0) # Number of background_generate tasks spawned
    processing_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index('ix_uploaded_file_records_status', 'status'),
    )

    def __repr__(self):
        return f"<UploadedFileRecord(id={self.id}, filename='{self.original_filename}', status='{self.status}')>"


# (Keep your existing model definitions here - unchanged from previous version)
class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    session_id = Column(String, index=True, nullable=True)

    # Ensuring server_default for string columns if they are NOT NULL and have a Python default
    mode = Column(String, nullable=False, server_default="chat", index=True)
    input_type = Column(String, nullable=False, server_default="text")

    user_input = Column(Text, nullable=True)
    llm_response = Column(Text, nullable=True)
    classification = Column(String, nullable=True)
    classification_reason = Column(Text, nullable=True)
    emotion_context_analysis = Column(Text, nullable=True)
    tool_name = Column(String, nullable=True)
    tool_parameters = Column(Text, nullable=True)
    image_data = Column(Text, nullable=True)  # For user-provided image b64 snippet
    url_processed = Column(String, nullable=True)
    image_description = Column(Text, nullable=True)  # For VLM desc of user-provided image
    latex_representation = Column(Text, nullable=True)
    latex_explanation = Column(Text, nullable=True)
    execution_time_ms = Column(Float, nullable=True)

    # Boolean columns that are NOT NULL need server_default for SQLite
    reflection_completed = Column(Boolean, nullable=False, server_default=text("0"), index=True)

    rag_source_url = Column(String, nullable=True)
    rag_history_ids = Column(String, nullable=True)  # Comma-separated IDs

    requires_deep_thought = Column(Boolean, nullable=True)  # Nullable, Python default is fine
    deep_thought_reason = Column(Text, nullable=True)

    tot_analysis_requested = Column(Boolean, nullable=False, server_default=text("0"))
    tot_analysis_spawned = Column(Boolean, nullable=False,
                                  server_default=text("0"))  # <<< CORRECTLY ADDED WITH SERVER_DEFAULT
    tot_result = Column(Text, nullable=True)
    tot_delivered = Column(Boolean, nullable=False, server_default=text("0"), index=True)

    assistant_action_analysis_json = Column(Text, nullable=True)
    assistant_action_type = Column(String, nullable=True, index=True)
    assistant_action_params = Column(Text, nullable=True)
    assistant_action_executed = Column(Boolean, nullable=False, server_default=text("0"))
    assistant_action_result = Column(Text, nullable=True)

    last_modified_db = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    imagined_image_prompt = Column(Text, nullable=True)
    imagined_image_b64 = Column(Text, nullable=True)  # For AI-generated image b64 snippet
    imagined_image_vlm_description = Column(Text, nullable=True)  # For VLM desc of AI-generated image

    reflection_indexed_in_vs = Column(Boolean, nullable=False, server_default=text("0"),
                                      index=True)  # <<< CORRECTLY ADDED WITH SERVER_DEFAULT

    __table_args__ = (
        Index('ix_interactions_session_mode_timestamp', 'session_id', 'mode', 'timestamp'),
        Index('ix_interactions_undelivered_tot', 'session_id', 'mode', 'tot_delivered', 'timestamp'),
        Index('ix_interactions_action_type_time', 'assistant_action_type', 'timestamp'),
        Index('ix_interactions_reflection_pending', 'reflection_completed', 'mode', 'input_type', 'timestamp'),
        Index('ix_interactions_reflection_indexed_vs', 'reflection_indexed_in_vs', 'input_type', 'timestamp'),
    # Index for new field
    )


class AppleScriptAttempt(Base):
    __tablename__ = "applescript_attempts"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    session_id = Column(String, index=True, nullable=False)
    triggering_interaction_id = Column(Integer, ForeignKey('interactions.id', ondelete='SET NULL'), index=True,
                                       nullable=True)
    action_type = Column(String, index=True, nullable=False)
    parameters_json = Column(Text, nullable=False)
    attempt_number = Column(Integer, default=1, nullable=False)
    generated_script = Column(Text, nullable=True)
    execution_success = Column(Boolean, index=True, nullable=True)
    execution_return_code = Column(Integer, nullable=True)
    execution_stdout = Column(Text, nullable=True)
    execution_stderr = Column(Text, nullable=True)
    execution_duration_ms = Column(Float, nullable=True)
    error_summary = Column(Text, nullable=True)
    triggering_interaction = relationship("Interaction", backref="applescript_attempts")
    __table_args__ = (
        UniqueConstraint('session_id', 'action_type', 'parameters_json', 'attempt_number',
                         name='uq_applescript_attempt'),
        Index('ix_applescript_attempts_lookup', 'action_type', 'parameters_json', 'execution_success', 'timestamp'),
    )


class FileIndex(Base):
    __tablename__ = "file_index"
    id = Column(Integer, primary_key=True)
    file_path = Column(String, unique=True, index=True, nullable=False)
    file_name = Column(String, index=True, nullable=False)
    size_bytes = Column(Integer, nullable=True)
    mime_type = Column(String, nullable=True)
    last_modified_os = Column(DateTime(timezone=False), nullable=True)
    last_indexed_db = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    index_status = Column(String, default='pending', index=True, nullable=False)
    indexed_content = Column(Text, nullable=True)
    embedding_json = Column(Text, nullable=True)
    md5_hash = Column(String(32), nullable=True, index=True)
    processing_error = Column(Text, nullable=True)
    latex_representation = Column(Text, nullable=True)
    latex_explanation = Column(Text, nullable=True)
    vlm_processing_status = Column(String, nullable=True, index=True)
    __table_args__ = (
        Index('ix_file_index_name_status', 'file_name', 'index_status'),
        Index('ix_file_index_status_modified', 'index_status', 'last_modified_os'),
        Index('ix_file_index_path_hash', 'file_path', 'md5_hash'),
        CheckConstraint(index_status.in_([
            'pending', 'indexed_text', 'indexed_meta', 'skipped_size',
            'skipped_type', 'error_read', 'error_permission', 'processing',
            'error_embedding', 'error_hash', 'error_vlm', 'partial_vlm_error',
            'error_conversion', 'pending_vlm', 'pending_conversion', 'success'
        ]), name='ck_file_index_status')
    )

    def __repr__(self):
        return f"<FileIndex(path='{self.file_path[:50]}...', status='{self.index_status}', hash='{self.md5_hash}')>"


# --- END Database Models ---


# --- Compression/Decompression Helpers (_compress_db, _decompress_db) ---
# (Keep your existing _compress_db and _decompress_db functions here - unchanged)
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
        cctx = zstd.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL, threads=-1)
        with open(RUNTIME_DB_PATH, 'rb') as ifh, open(DB_PATH_COMPRESSED, 'wb') as ofh:
            cctx.copy_stream(ifh, ofh)
        duration = time.monotonic() - start_time
        compressed_size = os.path.getsize(DB_PATH_COMPRESSED) if os.path.exists(DB_PATH_COMPRESSED) else 0
        ratio = compressed_size / original_size if original_size > 0 else 0
        logger.success(
            f"✅ Compression complete ({duration:.2f}s). Size: {original_size / 1024 / 1024:.2f}MB -> {compressed_size / 1024 / 1024:.2f}MB (Ratio: {ratio:.2f}).")
        compression_successful = True
    except Exception as e:
        logger.error(f"❌ Compression failed: {e}")
        logger.exception("Compression Traceback:")
        if os.path.exists(DB_PATH_COMPRESSED):
            try:
                os.remove(DB_PATH_COMPRESSED); logger.warning("Removed partial compressed file after error.")
            except Exception as rm_err:
                logger.error(f"Failed remove partial compressed file: {rm_err}")
        return False

    if compression_successful:
        logger.info(f"Removing runtime database file: {RUNTIME_DB_PATH}")
        try:
            if os.path.exists(RUNTIME_DB_PATH):
                os.remove(RUNTIME_DB_PATH)
            else:
                logger.warning("Runtime database file already gone before removal attempt.")
            logger.debug("Runtime database file removed.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to remove runtime database file after successful compression: {e}")
            return False
    return False


def _decompress_db(source_path: str, target_path: str) -> bool:
    """Decompresses a ZSTD file from source_path to target_path."""
    if not os.path.exists(source_path):
        logger.debug(f"Decompression skipped: Source file '{source_path}' not found.")
        return False

    logger.info(f"Decompressing '{source_path}' to '{target_path}'...")
    start_time = time.monotonic()
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        if os.path.exists(target_path):
            logger.warning(f"Overwriting existing file at target: {target_path}")

        dctx = zstd.ZstdDecompressor()
        with open(source_path, 'rb') as ifh, open(target_path, 'wb') as ofh:
            dctx.copy_stream(ifh, ofh)
        duration = time.monotonic() - start_time
        logger.success(f"✅ Decompression of '{os.path.basename(source_path)}' complete ({duration:.2f}s).")
        return True
    except Exception as e:
        logger.error(f"❌ Decompression failed for '{source_path}': {e}")
        logger.exception("Decompression Traceback:")
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
                logger.warning(
                    f"Removed partial target file '{target_path}' after failed decompression.")
            except Exception as rm_err:
                logger.error(f"Failed to remove partial target file: {rm_err}")
        return False


# --- END Compression/Decompression Helpers ---

# --- NEW: Read-Only Check ---
def _is_db_readonly(db_path: str) -> bool:
    """Checks if the SQLite database file is effectively read-only."""
    if not os.path.exists(db_path):
        return False  # Cannot be read-only if it doesn't exist

    # Try a simple write operation that doesn't change data content if successful
    # but requires write access. PRAGMA user_version is good for this.
    sqlite3_cmd = shutil.which("sqlite3")
    if not sqlite3_cmd:
        logger.warning("Cannot check read-only status: 'sqlite3' command not found. Assuming not read-only.")
        return False

    # Get current user_version to set it back
    current_user_version = 0
    try:
        get_version_proc = subprocess.run([sqlite3_cmd, db_path, "PRAGMA user_version;"], capture_output=True,
                                          text=True, check=False, timeout=5)
        if get_version_proc.returncode == 0:
            current_user_version = int(get_version_proc.stdout.strip())
    except Exception:
        pass  # Ignore if getting current version fails

    try:
        # Attempt to write the same user_version back
        test_write_command = [sqlite3_cmd, db_path, f"PRAGMA user_version = {current_user_version};"]
        process = subprocess.run(test_write_command, capture_output=True, text=True, check=False, timeout=5)

        if process.returncode != 0:
            # Check stderr for "attempt to write a readonly database" or similar
            stderr_lower = process.stderr.lower()
            if "readonly database" in stderr_lower or "attempt to write a readonly database" in stderr_lower:
                logger.warning(f"Database at '{db_path}' appears to be READ-ONLY. Stderr: {process.stderr.strip()}")
                return True
            else:
                # Some other error occurred during the write test, but not necessarily read-only
                logger.warning(
                    f"Test write to '{db_path}' failed, but not clearly due to read-only. RC={process.returncode}, Stderr: {process.stderr.strip()}")
                return False  # Assume not read-only for other errors for now
        return False  # Write test succeeded
    except subprocess.TimeoutExpired:
        logger.warning(f"Read-only check for '{db_path}' timed out. Assuming not read-only for safety.")
        return False
    except Exception as e:
        logger.warning(f"Error during read-only check for '{db_path}': {e}. Assuming not read-only.")
        return False


# --- END Read-Only Check ---

def _check_db_integrity_quick(db_path: str, context_msg: str = "DB") -> bool:
    """Quickly checks SQLite DB integrity using 'PRAGMA integrity_check'."""
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        logger.debug(f"Integrity check (quick) skipped for {context_msg}: File not found or empty at {db_path}")
        return True
    logger.info(f"🩺 Performing QUICK integrity check on {context_msg} ('{os.path.basename(db_path)}')...")
    sqlite3_cmd = shutil.which("sqlite3")
    if not sqlite3_cmd:
        logger.error("❌ Cannot run integrity check: 'sqlite3' command not found. Assuming OK for now.")
        return True
    try:
        check_command = [sqlite3_cmd, db_path, "PRAGMA integrity_check;"]
        # Using a timeout for the integrity check itself
        process = subprocess.run(check_command, capture_output=True, text=True, check=False,
                                 timeout=120)  # 2-minute timeout for full check
        if process.returncode == 0 and process.stdout.strip().lower() == "ok":
            logger.success(f"✅ Quick integrity check passed for {context_msg}.")
            return True
        else:
            logger.warning(
                f"🔥 Quick integrity check FAILED for {context_msg}. Output: '{process.stdout.strip()}' Stderr: '{process.stderr.strip()}' RC: {process.returncode}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"❌ Quick integrity check for {context_msg} timed out. Assuming potential issue.")
        return False
    except Exception as e:
        logger.error(f"❌ Error during quick integrity check for {context_msg}: {e}")
        return False


def _check_and_repair_db(db_path: str, context_msg: str = "runtime DB") -> bool:
    """
    Performs a full integrity check on the SQLite DB. If it fails,
    attempts to repair it by dumping to SQL and reloading into a new DB file.
    This is typically a last resort after simpler checks or snapshot restores.
    Returns True if the DB is OK or successfully repaired, False otherwise.
    """
    if not os.path.exists(db_path):
        logger.info(f"Full Check/Repair skipped for {context_msg}: DB file '{db_path}' not found.")
        return False  # Cannot repair a non-existent file; new one should be created by Alembic if needed.

    if os.path.getsize(db_path) == 0:
        logger.warning(
            f"Full Check/Repair skipped for {context_msg}: DB file '{db_path}' is empty. Alembic should create schema.")
        # Delete the empty file so Alembic can create it fresh if this is the runtime DB
        if db_path == globals().get("RUNTIME_DB_PATH"):  # Check against the global RUNTIME_DB_PATH
            try:
                os.remove(db_path)
                logger.info(f"Removed empty DB file '{db_path}' to allow fresh creation.")
            except Exception as e_rm_empty:
                logger.error(f"Could not remove empty DB file '{db_path}': {e_rm_empty}")
        return False  # Indicate that this empty file is not "OK" for use yet.

    logger.info(f"🩺 Performing FULL integrity check on {context_msg} ('{os.path.basename(db_path)}')...")
    full_check_start_time = time.monotonic()
    is_initially_ok = False
    sqlite3_cmd = shutil.which("sqlite3")

    if not sqlite3_cmd:
        logger.error("❌ Cannot run full integrity check/repair: 'sqlite3' command not found in PATH.")
        return False  # Cannot proceed without sqlite3 CLI

    try:
        # Use a longer timeout for a full integrity check on potentially large DBs
        check_command = [sqlite3_cmd, db_path, "PRAGMA integrity_check;"]
        process = subprocess.run(check_command, capture_output=True, text=True, check=False,
                                 timeout=300)  # 5-minute timeout
        check_output = process.stdout.strip()
        check_duration = time.monotonic() - full_check_start_time
        logger.info(f"Full integrity check for {context_msg} completed in {check_duration:.2f}s.")

        if process.returncode == 0 and check_output.lower() == "ok":
            logger.success(f"✅ Full integrity check PASSED for {context_msg}.")
            return True  # DB is OK, no repair needed
        else:
            logger.warning(
                f"🔥 Full integrity check FAILED for {context_msg}. RC={process.returncode}. Output:\n{check_output}\nStderr:\n{process.stderr.strip()}")
            # Proceed to repair attempt
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Full integrity check for {context_msg} TIMED OUT.")
        # Treat timeout as a failure requiring repair attempt
    except Exception as e_check:
        logger.error(f"❌ Error during full integrity check execution for {context_msg}: {e_check}")
        # Treat any error as a failure requiring repair attempt

    # If we reach here, the initial full integrity check failed or timed out.
    logger.warning(
        f"🚨 Attempting automatic repair via SQL dump/reload for {context_msg} ('{os.path.basename(db_path)}')...")
    repair_start_time = time.monotonic()
    temp_dump_sql_file: Optional[str] = None
    repair_successful = False

    # Define a unique name for the corrupted backup
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    corrupted_db_backup_path = f"{db_path}.corrupted_before_repair_{timestamp_str}"

    try:
        # Create a named temporary file for the SQL dump
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".sql", encoding='utf-8') as tmp_sql_fh:
            temp_dump_sql_file = tmp_sql_fh.name
        logger.debug(f"DB Repair: Using temporary SQL dump file: {temp_dump_sql_file}")

        # 1. Dump the database to an SQL file using .dump command
        #    The .dump command is generally good at extracting data even from corrupted DBs.
        logger.info(f"DB Repair: Dumping '{db_path}' to SQL...")
        dump_command = [sqlite3_cmd, db_path, ".dump"]
        with open(temp_dump_sql_file, 'w', encoding='utf-8') as dump_file_handle:
            dump_process = subprocess.run(
                dump_command,
                stdout=dump_file_handle,  # Redirect stdout to the file
                stderr=subprocess.PIPE,
                text=True, errors='replace',  # Handle potential encoding issues in stderr
                check=False,  # Don't raise exception on non-zero exit for dump
                timeout=600  # 10-minute timeout for dump
            )

        if dump_process.returncode != 0:
            # Even if dump has non-zero RC, it might have written some data.
            # Log stderr if present.
            logger.warning(f"DB Repair: Dump command finished with RC={dump_process.returncode}.")
            if dump_process.stderr:
                logger.warning(f"DB Repair: Dump command STDERR:\n{dump_process.stderr.strip()}")
            if not os.path.exists(temp_dump_sql_file) or os.path.getsize(temp_dump_sql_file) == 0:
                logger.error("DB Repair: Dump FAILED to produce any SQL output. Cannot proceed with repair.")
                return False  # Abort repair if dump is empty

        logger.info(
            f"DB Repair: Dumped to SQL. Size: {os.path.getsize(temp_dump_sql_file) if os.path.exists(temp_dump_sql_file) else 0} bytes.")

        # 2. Backup the original (potentially corrupted) database file
        logger.info(f"DB Repair: Backing up original DB '{db_path}' to '{corrupted_db_backup_path}'...")
        try:
            shutil.move(db_path, corrupted_db_backup_path)
        except Exception as e_mv_orig:
            logger.error(f"DB Repair: FAILED to backup original DB: {e_mv_orig}. Cannot proceed with repair safely.")
            return False

        # 3. Reload the SQL dump into a new database file (at original db_path)
        logger.info(f"DB Repair: Reloading SQL dump into new DB file at '{db_path}'...")
        reload_command = [sqlite3_cmd, db_path]  # sqlite3 CLI creates the DB if it doesn't exist
        with open(temp_dump_sql_file, 'r', encoding='utf-8') as dump_sql_handle:
            reload_process = subprocess.run(
                reload_command,
                stdin=dump_sql_handle,  # Pipe SQL commands to sqlite3
                capture_output=True,
                text=True, errors='replace',
                check=False,  # Don't raise exception on non-zero exit for reload
                timeout=1800  # 30-minute timeout for reload (can be large)
            )

        if reload_process.returncode != 0:
            logger.error(f"DB Repair: Reload FAILED (RC={reload_process.returncode}).")
            if reload_process.stderr: logger.error(f"DB Repair: Reload STDERR:\n{reload_process.stderr.strip()}")
            if reload_process.stdout: logger.warning(f"DB Repair: Reload STDOUT:\n{reload_process.stdout.strip()}")

            logger.warning(
                f"DB Repair: Attempting to restore original (corrupted) DB from '{corrupted_db_backup_path}' due to reload failure.")
            try:
                if os.path.exists(db_path): os.remove(db_path)  # Remove partially reloaded DB
                shutil.move(corrupted_db_backup_path, db_path)
                logger.info("DB Repair: Original (corrupted) DB restored.")
            except Exception as e_restore_corr:
                logger.error(
                    f"DB Repair: FAILED to restore original (corrupted) DB: {e_restore_corr}. DB state is uncertain at '{db_path}'.")
            return False  # Repair failed

        logger.info(f"DB Repair: Reloaded SQL dump into '{db_path}'.")
        repair_duration = time.monotonic() - repair_start_time
        logger.success(f"✅ DB Repair: Dump/Reload attempt completed in {repair_duration:.2f}s.")

        # 4. Final integrity check on the reloaded database
        logger.info("DB Repair: Performing final integrity check on reloaded DB...")
        if _check_db_integrity_quick(db_path, "repaired DB via dump/reload"):  # Use the quick check
            logger.success("✅ DB Repair: Repaired DB passed final integrity check.")
            repair_successful = True
            # Successfully repaired, corrupted backup can be kept or deleted.
            # For now, we keep it. Add cleanup logic if desired.
        else:
            logger.error(
                "❌ DB Repair: Repaired DB FAILED final integrity check! Data loss may have occurred or repair was ineffective.")
            logger.error(f"   The original corrupted DB is at: {corrupted_db_backup_path}")
            logger.error(f"   The (still problematic) reloaded DB is at: {db_path}")
            repair_successful = False

    except subprocess.TimeoutExpired as e_timeout:
        logger.error(f"❌ DB Repair: Process (dump/reload) timed out: {e_timeout}")
        repair_successful = False
    except Exception as repair_err:
        logger.error(f"❌ DB Repair: Unexpected error during repair process: {repair_err}")
        logger.exception("DB Repair Process Traceback:")
        repair_successful = False
    finally:
        # Clean up the temporary SQL dump file
        if temp_dump_sql_file and os.path.exists(temp_dump_sql_file):
            try:
                os.remove(temp_dump_sql_file)
                logger.debug(f"DB Repair: Cleaned up temporary SQL dump file: {temp_dump_sql_file}")
            except Exception as e_rm_temp:
                logger.warning(f"DB Repair: Failed to clean up temp SQL dump file '{temp_dump_sql_file}': {e_rm_temp}")

    if repair_successful:
        logger.info(f"DB Repair: Successfully repaired '{db_path}'.")
    else:
        logger.error(f"DB Repair: FAILED to repair '{db_path}'. Check logs and backups.")

    return repair_successful

# --- Integrity Check and Repair (_check_and_repair_db) ---
# (Keep your existing _check_and_repair_db function - it's still useful as a last resort)
# Small modification: it will now be called *after* snapshot restore attempts.
def _check_db_integrity_quick(db_path: str, context_msg: str = "DB") -> bool:
    """Quickly checks SQLite DB integrity using 'PRAGMA integrity_check'."""
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        logger.debug(f"Integrity check (quick) skipped for {context_msg}: File not found or empty at {db_path}")
        return True # Consider non-existent or empty as "not corrupt" for this check's purpose

    logger.info(f"🩺 Performing QUICK integrity check on {context_msg} ('{os.path.basename(db_path)}')...")
    sqlite3_cmd = shutil.which("sqlite3")
    if not sqlite3_cmd:
        logger.error("❌ Cannot run integrity check: 'sqlite3' command not found. Assuming OK for now.")
        return True # Cannot check, assume OK to proceed with caution

    try:
        # Limit integrity check to first few pages for speed, or run full check if preferred.
        # For a quick check, 'PRAGMA quick_check;' can also be used.
        # 'PRAGMA integrity_check(10);' checks first 10 pages.
        # For this initial check, let's use the full integrity_check but with a shorter timeout.
        check_command = [sqlite3_cmd, db_path, "PRAGMA integrity_check;"]
        process = subprocess.run(check_command, capture_output=True, text=True, check=False, timeout=60) # Shorter timeout for quick check

        if process.returncode == 0 and process.stdout.strip().lower() == "ok":
            logger.success(f"✅ Quick integrity check passed for {context_msg}.")
            return True
        else:
            logger.warning(f"🔥 Quick integrity check FAILED for {context_msg}. Output: {process.stdout.strip()} Stderr: {process.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"❌ Quick integrity check for {context_msg} timed out. Assuming potential issue.")
        return False # Treat timeout as a potential issue
    except Exception as e:
        logger.error(f"❌ Error during quick integrity check for {context_msg}: {e}")
        return False # Treat any error as a potential issue


# --- END Integrity Check and Repair ---


# --- Shutdown Hook (_shutdown_hook) ---
# (Keep your existing _shutdown_hook function - unchanged)
def _shutdown_hook():
    logger.info("Executing database shutdown hook...")
    global _engine
    if _engine:
        try:
            _engine.dispose(); logger.debug("Engine pool disposed.")
        except Exception as e:
            logger.warning(f"Error disposing engine pool during shutdown: {e}")
    if not _compress_db():  # Compresses RUNTIME_DB_PATH to DB_PATH_COMPRESSED
        logger.error("Shutdown hook: Compression and/or runtime file cleanup failed!")
    logger.info("Database shutdown hook finished.")


# --- END Shutdown Hook ---
# atexit.register(_shutdown_hook) # Registered in init_db after snapshotter setup

# --- NEW: Snapshot Management Helpers ---
def _get_snapshot_files() -> List[str]:
    """Returns a sorted list of snapshot file paths (oldest first)."""
    if not os.path.isdir(DB_SNAPSHOT_DIR): # DB_SNAPSHOT_DIR is defined globally
        return []
    try:
        snapshot_files = [
            os.path.join(DB_SNAPSHOT_DIR, f)
            for f in os.listdir(DB_SNAPSHOT_DIR)
            if f.startswith(DB_SNAPSHOT_FILENAME_PREFIX) and f.endswith(DB_SNAPSHOT_FILENAME_SUFFIX)
        ]
        def get_timestamp_from_filename(filepath):
            filename = os.path.basename(filepath)
            try:
                ts_part = filename[len(DB_SNAPSHOT_FILENAME_PREFIX):-len(DB_SNAPSHOT_FILENAME_SUFFIX)]
                return datetime.datetime.strptime(ts_part, "%Y%m%d_%H%M%S")
            except ValueError:
                # If filename format is unexpected, treat it as very old for sorting robustness
                logger.warning(f"Could not parse timestamp from snapshot filename: {filename}")
                return datetime.datetime.min
        snapshot_files.sort(key=get_timestamp_from_filename) # Sorts oldest first
        return snapshot_files
    except Exception as e:
        logger.error(f"Error listing snapshot files: {e}")
        return []


def _prune_old_snapshots():
    """Deletes the oldest snapshots if the count exceeds retention."""
    if not ENABLE_DB_SNAPSHOTS or DB_SNAPSHOT_RETENTION_COUNT <= 0: # Check if pruning is enabled/valid
        return
    try:
        snapshots = _get_snapshot_files() # Gets them sorted, oldest first
        if len(snapshots) > DB_SNAPSHOT_RETENTION_COUNT:
            num_to_delete = len(snapshots) - DB_SNAPSHOT_RETENTION_COUNT
            logger.info(f"Pruning {num_to_delete} old snapshot(s) (retention limit: {DB_SNAPSHOT_RETENTION_COUNT})...")
            # Delete the oldest ones from the beginning of the sorted list
            for i in range(num_to_delete):
                snapshot_to_delete = snapshots[i]
                try:
                    os.remove(snapshot_to_delete)
                    logger.debug(f"  Deleted old snapshot: {os.path.basename(snapshot_to_delete)}")
                except Exception as e:
                    logger.warning(f"  Failed to delete old snapshot {os.path.basename(snapshot_to_delete)}: {e}")
    except Exception as e:
        logger.error(f"Error during snapshot pruning: {e}")


def _restore_from_latest_snapshot() -> bool:
    """Attempts to restore the database from the latest valid snapshot."""
    if not ENABLE_DB_SNAPSHOTS:
        logger.info("Snapshot restoration skipped: Snapshots are disabled.")
        return False

    snapshots = _get_snapshot_files()
    if not snapshots:
        logger.info("No snapshots found to restore from.")
        return False

    # Try from newest to oldest
    for snapshot_path in reversed(snapshots):
        logger.info(f"Attempting to restore from snapshot: {os.path.basename(snapshot_path)}...")
        # Decompress to a temporary path first to avoid corrupting runtime if snapshot is bad
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_restored_db:
            temp_restored_db_path = tmp_restored_db.name

        restored_successfully_to_temp = False
        try:
            if _decompress_db(snapshot_path, temp_restored_db_path):
                logger.info(f"Successfully decompressed snapshot to temporary file: {temp_restored_db_path}")
                # Check integrity of the decompressed temporary snapshot
                if _check_and_repair_db(temp_restored_db_path,
                                        context_msg=f"restored snapshot {os.path.basename(snapshot_path)}"):
                    logger.info(f"Restored snapshot '{os.path.basename(snapshot_path)}' passed integrity check.")
                    # Atomically replace the runtime DB with the good snapshot
                    try:
                        if os.path.exists(RUNTIME_DB_PATH):
                            backup_corrupt_path = f"{RUNTIME_DB_PATH}.pre_snapshot_restore_{int(time.time())}"
                            logger.warning(
                                f"Backing up existing (potentially corrupt/readonly) runtime DB to {backup_corrupt_path}")
                            shutil.move(RUNTIME_DB_PATH, backup_corrupt_path)

                        shutil.move(temp_restored_db_path, RUNTIME_DB_PATH)
                        logger.success(
                            f"✅ Database successfully restored from snapshot: {os.path.basename(snapshot_path)}")
                        restored_successfully_to_temp = True  # Flag that the move was successful
                        return True  # Overall success
                    except Exception as move_err:
                        logger.error(f"Failed to move verified snapshot to runtime path: {move_err}")
                else:
                    logger.warning(
                        f"Restored snapshot '{os.path.basename(snapshot_path)}' FAILED integrity check. Trying older snapshot if available.")
            else:
                logger.warning(f"Failed to decompress snapshot '{os.path.basename(snapshot_path)}'.")
        finally:
            # Clean up the temporary decompressed file only if it wasn't successfully moved
            if os.path.exists(temp_restored_db_path) and not restored_successfully_to_temp:
                try:
                    os.remove(temp_restored_db_path)
                except Exception as e:
                    logger.warning(f"Could not remove temporary restored DB file '{temp_restored_db_path}': {e}")

    logger.error("❌ All available snapshots failed to restore or pass integrity checks.")
    return False


# --- END Snapshot Management Helpers ---

# --- NEW: Database Snapshotter Thread ---
class DatabaseSnapshotter(threading.Thread):
    def __init__(self, stop_event: threading.Event):
        super().__init__(name="DatabaseSnapshotterThread", daemon=True)
        self.stop_event = stop_event
        self.sqlite3_cmd = shutil.which("sqlite3")
        if not self.sqlite3_cmd:
            logger.error("SQLite3 command-line tool not found. Database snapshotting will be disabled.")
            # Optionally, could set a flag to prevent run() from doing anything.
        logger.info(f"DatabaseSnapshotter initialized. Interval: {DB_SNAPSHOT_INTERVAL_MINUTES} min.")

    def _take_snapshot(self):
        if not self.sqlite3_cmd: return
        if not os.path.exists(RUNTIME_DB_PATH) or os.path.getsize(RUNTIME_DB_PATH) == 0:
            logger.debug("Snapshot skipped: Runtime DB does not exist or is empty.")
            return

        snapshot_start_time = time.monotonic()
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure DB_SNAPSHOT_DIR exists for the temp backup as well
        os.makedirs(DB_SNAPSHOT_DIR, exist_ok=True)
        temp_backup_db_path = os.path.join(DB_SNAPSHOT_DIR,
                                           f"temp_uncompressed_backup_{timestamp_str}.db")  # More descriptive temp name
        final_snapshot_path = os.path.join(DB_SNAPSHOT_DIR,
                                           f"{DB_SNAPSHOT_FILENAME_PREFIX}{timestamp_str}{DB_SNAPSHOT_FILENAME_SUFFIX}")

        # os.makedirs(DB_SNAPSHOT_DIR, exist_ok=True) # Already done above
        logger.info(f"📸 Creating database snapshot for {timestamp_str}...")
        compression_successful_flag = False  # Flag to track success

        try:
            # 1. Use SQLite's online backup to a temporary file
            if os.path.exists(temp_backup_db_path):
                try:
                    os.remove(temp_backup_db_path)
                except Exception as e_rem_old_temp:
                    logger.warning(f"Could not remove old temp backup '{temp_backup_db_path}': {e_rem_old_temp}")

            backup_command = [self.sqlite3_cmd, RUNTIME_DB_PATH, f".backup '{temp_backup_db_path}'"]
            logger.debug(f"Running SQLite backup: {' '.join(backup_command)}")
            process = subprocess.run(backup_command, capture_output=True, text=True, check=False, timeout=120)

            if process.returncode != 0:
                logger.error(
                    f"SQLite .backup command failed (RC={process.returncode}). Stderr: {process.stderr.strip()}")
                if os.path.exists(temp_backup_db_path): os.remove(temp_backup_db_path)
                return

            if not os.path.exists(temp_backup_db_path) or os.path.getsize(temp_backup_db_path) == 0:
                logger.error("SQLite .backup seemed to succeed but temporary backup file is missing or empty.")
                if os.path.exists(temp_backup_db_path): os.remove(temp_backup_db_path)
                return

            logger.debug(f"SQLite online backup to '{temp_backup_db_path}' successful.")

            # 2. Compress the temporary backup file
            logger.debug(f"Compressing temporary backup '{temp_backup_db_path}' to '{final_snapshot_path}'...")

            # --- CORRECTED COMPRESSION LOGIC ---
            cctx = zstd.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL, threads=-1)
            with open(temp_backup_db_path, 'rb') as ifh, open(final_snapshot_path, 'wb') as ofh:
                cctx.copy_stream(ifh, ofh)
            logger.success(f"Snapshot compressed to '{os.path.basename(final_snapshot_path)}'.")
            compression_successful_flag = True
            # --- END CORRECTED COMPRESSION LOGIC ---

            if compression_successful_flag:
                # 3. Prune old snapshots
                _prune_old_snapshots()
            # else: # No need for an else here, if compression fails, it will be caught by the outer try-except

        except subprocess.TimeoutExpired:
            logger.error("SQLite .backup command timed out during snapshot.")
        except Exception as e:
            logger.error(f"Error during snapshot creation: {e}")
            logger.exception("Snapshot Creation Traceback:")
            # If compression failed, ensure final_snapshot_path (if partially created) is removed
            if not compression_successful_flag and os.path.exists(final_snapshot_path):
                try:
                    os.remove(final_snapshot_path)
                    logger.warning(
                        f"Removed partial snapshot file '{final_snapshot_path}' after error.")
                except Exception as e_rem_final:
                    logger.error(f"Failed to remove partial snapshot file '{final_snapshot_path}': {e_rem_final}")
        finally:
            if os.path.exists(temp_backup_db_path):
                try:
                    os.remove(temp_backup_db_path)
                except Exception as e_rem:
                    logger.warning(f"Could not remove temp backup DB '{temp_backup_db_path}': {e_rem}")

        snapshot_duration = time.monotonic() - snapshot_start_time
        logger.info(f"Snapshot attempt finished in {snapshot_duration:.2f}s. Success: {compression_successful_flag}")

    def run(self):
        if not self.sqlite3_cmd:
            logger.warning("DatabaseSnapshotter exiting: sqlite3 CLI not found.")
            return

        logger.info("💾 DatabaseSnapshotter thread started.")
        while not self.stop_event.is_set():
            try:
                # Check if server is busy (from app.py, if that state is shared, otherwise remove)
                # For now, let's assume no direct server_busy_event check here to simplify
                # The main concern is contention on the DB file itself.
                self._take_snapshot()
            except Exception as e:
                logger.error(f"Unhandled error in DatabaseSnapshotter loop: {e}")
                logger.exception("Snapshotter Loop Traceback:")

            # Wait for the configured interval or until stop_event is set
            wait_seconds = DB_SNAPSHOT_INTERVAL_MINUTES * 60
            logger.debug(f"Snapshotter waiting for {wait_seconds} seconds...")
            self.stop_event.wait(timeout=wait_seconds)

        logger.info("💾 DatabaseSnapshotter thread stopped.")


_snapshotter_thread: Optional[DatabaseSnapshotter] = None
_snapshotter_stop_event = threading.Event()


def start_db_snapshotter():
    global _snapshotter_thread
    if not ENABLE_DB_SNAPSHOTS:
        logger.info("DB Snapshotter disabled by configuration.")
        return
    if _snapshotter_thread is None or not _snapshotter_thread.is_alive():
        logger.info("🚀 Starting Database Snapshotter service...")
        _snapshotter_stop_event.clear()
        _snapshotter_thread = DatabaseSnapshotter(_snapshotter_stop_event)
        _snapshotter_thread.start()
    else:
        logger.warning("🤔 Database Snapshotter thread already running.")


def stop_db_snapshotter():
    global _snapshotter_thread
    if not ENABLE_DB_SNAPSHOTS: return
    if _snapshotter_thread and _snapshotter_thread.is_alive():
        logger.info("Signaling Database Snapshotter thread to stop...")
        _snapshotter_stop_event.set()
        # _snapshotter_thread.join(timeout=10) # Optional: wait for it to finish
        logger.info("Stop signal sent to Database Snapshotter.")


# --- END Database Snapshotter Thread ---


# --- Engine Creation Wrapper (get_engine) - Modified for Snapshot Restore ---
def get_engine():
    global _engine, SessionLocal
    if _engine is not None:
        return _engine

    logger.info("🚀 Initializing SQLAlchemy engine (Runtime DB with Enhanced Restore Logic)...")
    db_successfully_prepared = False
    runtime_db_exists_initially = os.path.exists(RUNTIME_DB_PATH)
    runtime_db_is_empty = runtime_db_exists_initially and os.path.getsize(RUNTIME_DB_PATH) == 0

    # --- Step 1: Handle Decompression of Shutdown Archive (if exists) ---
    if os.path.exists(DB_PATH_COMPRESSED):
        logger.info(
            f"Found shutdown archive '{DB_PATH_COMPRESSED}'. Attempting decompression to '{RUNTIME_DB_PATH}'...")
        if os.path.exists(RUNTIME_DB_PATH):  # If runtime DB also exists, archive takes precedence
            logger.warning(
                f"Runtime DB '{RUNTIME_DB_PATH}' also exists. It will be overwritten by shutdown archive if decompression is successful.")
            try:
                # Backup current runtime DB before overwriting, just in case archive is bad
                backup_path = f"{RUNTIME_DB_PATH}.before_archive_restore_{int(time.time())}"
                shutil.move(RUNTIME_DB_PATH, backup_path)
                logger.info(f"Backed up existing runtime DB to '{backup_path}'.")
            except Exception as e_mv:
                logger.error(
                    f"Could not backup existing runtime DB before archive restore: {e_mv}. Proceeding with caution.")

        if _decompress_db(DB_PATH_COMPRESSED, RUNTIME_DB_PATH):
            logger.success(f"Successfully decompressed shutdown archive to '{RUNTIME_DB_PATH}'.")
            try:
                os.remove(DB_PATH_COMPRESSED)
                logger.info(f"Removed shutdown archive '{DB_PATH_COMPRESSED}'.")
            except Exception as e_rm_arc:
                logger.warning(f"Could not remove shutdown archive after decompression: {e_rm_arc}")
            db_successfully_prepared = True
            runtime_db_exists_initially = True  # It exists now
            runtime_db_is_empty = os.path.getsize(RUNTIME_DB_PATH) == 0
        else:
            logger.error(
                f"Failed to decompress shutdown archive '{DB_PATH_COMPRESSED}'. Current runtime DB (if any) is preserved.")
            # db_successfully_prepared remains False, or True if runtime_db_exists_initially was true and not overwritten.
            db_successfully_prepared = runtime_db_exists_initially and not runtime_db_is_empty

    # --- Step 2: Validate Current/Restored Runtime DB ---
    if runtime_db_exists_initially and not runtime_db_is_empty:
        logger.info(f"Validating existing/restored runtime DB at '{RUNTIME_DB_PATH}'...")
        is_readonly = _is_db_readonly(RUNTIME_DB_PATH)
        is_corrupt = False

        if is_readonly:
            logger.warning(f"Runtime DB '{RUNTIME_DB_PATH}' is READ-ONLY. Attempting to restore from snapshot.")
            db_successfully_prepared = False  # Mark as not prepared, force snapshot restore
        else:
            if not _check_db_integrity_quick(RUNTIME_DB_PATH, "current runtime DB"):
                logger.warning(
                    f"Runtime DB '{RUNTIME_DB_PATH}' FAILED quick integrity check or is potentially corrupt.")
                is_corrupt = True
                db_successfully_prepared = False  # Mark as not prepared, try snapshot restore
            else:
                logger.info(f"Runtime DB '{RUNTIME_DB_PATH}' passed quick integrity check.")
                db_successfully_prepared = True  # Tentatively okay

    elif runtime_db_is_empty:
        logger.warning(
            f"Runtime DB '{RUNTIME_DB_PATH}' exists but is empty. Will attempt snapshot restore or create new.")
        db_successfully_prepared = False
    else:  # Runtime DB does not exist
        logger.info(f"Runtime DB '{RUNTIME_DB_PATH}' does not exist. Will attempt snapshot restore or create new.")
        db_successfully_prepared = False

    # --- Step 3: Attempt Snapshot Restore if Current DB is Not Prepared/Valid ---
    if not db_successfully_prepared:
        logger.info(
            f"Attempting to restore DB from the latest valid snapshot (Snapshots enabled: {ENABLE_DB_SNAPSHOTS})...")
        if _restore_from_latest_snapshot():  # This function now includes integrity check of snapshot
            logger.success(f"✅ Database successfully restored from snapshot to '{RUNTIME_DB_PATH}'.")
            db_successfully_prepared = True
            # Re-check if it's empty after restore (should not be if snapshot was good)
            if os.path.exists(RUNTIME_DB_PATH) and os.path.getsize(RUNTIME_DB_PATH) == 0:
                logger.error("CRITICAL: DB restored from snapshot but is EMPTY. Snapshot might be bad.")
                db_successfully_prepared = False  # Treat as failure
        else:
            logger.warning(
                f"Snapshot restore failed or no suitable snapshots found. Runtime DB path: '{RUNTIME_DB_PATH}'")
            # If runtime DB existed but was corrupt/readonly, it might still be there or moved.
            # If it didn't exist, it still doesn't.
            # db_successfully_prepared remains False

    # --- Step 4: Final Integrity Check / Repair (if DB exists but wasn't just successfully restored from snapshot) ---
    # This step is reached if:
    #   - Initial runtime DB passed quick check (db_successfully_prepared = True)
    #   - OR Snapshot restore failed, but an older (potentially problematic) DB file still exists at RUNTIME_DB_PATH.
    #   - OR No DB existed, no archive, no snapshots -> RUNTIME_DB_PATH doesn't exist, this block is skipped.
    if os.path.exists(RUNTIME_DB_PATH) and os.path.getsize(RUNTIME_DB_PATH) > 0:
        if not db_successfully_prepared:  # Implies snapshot restore failed, but a DB file might still exist
            logger.warning(
                f"Snapshot restore failed. Performing a full integrity check and potential repair on existing DB file: '{RUNTIME_DB_PATH}' as a last resort.")
            if _check_and_repair_db(RUNTIME_DB_PATH, "existing DB after failed snapshot restore"):
                logger.info("Full check/repair on existing DB passed or was successful.")
                db_successfully_prepared = True
            else:
                logger.error(f"Full check/repair FAILED for existing DB '{RUNTIME_DB_PATH}'. Moving corrupted file.")
                # Move the problematic DB file to avoid Alembic trying to migrate a corrupt schema
                try:
                    corrupt_final_path = f"{RUNTIME_DB_PATH}.final_corruption_{int(time.time())}"
                    shutil.move(RUNTIME_DB_PATH, corrupt_final_path)
                    logger.info(f"Moved final corrupted DB to '{corrupt_final_path}'. A new DB will be created.")
                except Exception as mv_err:
                    logger.error(f"Could not move final corrupted DB: {mv_err}. Alembic might fail.")
                db_successfully_prepared = False  # Will lead to new DB creation by Alembic
    elif not os.path.exists(RUNTIME_DB_PATH):
        logger.info("No database file found after all restore attempts. A new DB will be created by Alembic.")
        db_successfully_prepared = False  # To be clear, though it already is

    # --- Step 5: Create Engine ---
    engine_args_internal = {"echo": False, "connect_args": {"check_same_thread": False, "timeout": 60.0}}
    logger.info(f"Creating SQLAlchemy engine for: {RUNTIME_DATABASE_URL}")
    try:
        _engine = create_engine(RUNTIME_DATABASE_URL, **engine_args_internal)
        # Test connection
        with _engine.connect() as connection:
            logger.debug("Engine connection test successful after all preparations.")
        SessionLocal.configure(bind=_engine)
        logger.info("SQLAlchemy SessionLocal configured and bound to engine.")
    except Exception as e_engine:
        logger.critical(f"🔥🔥 DATABASE ENGINE CREATION FAILED for '{RUNTIME_DATABASE_URL}': {e_engine}")
        sys.exit(1)  # Fatal if engine cannot be created

    # SessionLocal should be bound now
    if SessionLocal is None or not SessionLocal.kw.get('bind'):  # type: ignore
        logger.critical("🔥🔥 Failed to configure SessionLocal binding after engine creation.")
        sys.exit(1)

    return _engine


# --- END Engine Creation Wrapper ---


# --- Alembic Helper Functions (_get_alembic_config, _create_default_env_py, etc.) ---
# (Keep your existing Alembic helpers here - unchanged)
def _get_alembic_config() -> Optional[Config]:
    if not os.path.exists(ALEMBIC_INI_PATH):
        logger.error(f"Alembic ini file not found at {ALEMBIC_INI_PATH}. Cannot proceed.")
        return None
    try:
        alembic_cfg = Config(ALEMBIC_INI_PATH)
        # env.py will use PROJECT_CONFIG_DATABASE_URL, but ini needs a valid placeholder
        alembic_cfg.set_main_option("sqlalchemy.url", DATABASE_URL)
        alembic_cfg.set_main_option("script_location", os.path.relpath(ALEMBIC_DIR, MODULE_DIR).replace("\\", "/"))
        return alembic_cfg
    except Exception as e:
        logger.error(f"Failed to load/configure Alembic from {ALEMBIC_INI_PATH}: {e}")
        return None


def _create_default_env_py():
    logger.info(f"Ensuring Alembic env.py at {ALEMBIC_ENV_PY_PATH} is correctly configured...")
    env_py_content = f"""\
import os
import sys
from logging.config import fileConfig
from sqlalchemy import create_engine, pool
from alembic import context

ENV_PY_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_MAIN_DIR = os.path.abspath(os.path.join(ENV_PY_SCRIPT_DIR, '..'))

# Critical: Print CWD when env.py is run
print(f"[alembic/env.py] Current Working Directory: {{os.getcwd()}}", file=sys.stderr)
print(f"[alembic/env.py] ENGINE_MAIN_DIR determined as: {{ENGINE_MAIN_DIR}}", file=sys.stderr)
print(f"[alembic/env.py] Initial sys.path: {{sys.path}}", file=sys.stderr)

if ENGINE_MAIN_DIR not in sys.path:
    sys.path.insert(0, ENGINE_MAIN_DIR)
    print(f"[alembic/env.py] Added {{ENGINE_MAIN_DIR}} to sys.path.", file=sys.stderr)
    print(f"[alembic/env.py] Modified sys.path: {{sys.path}}", file=sys.stderr)

alembic_ini_config = context.config
target_metadata = None
EFFECTIVE_DATABASE_URL_FOR_ALEMBIC = None

print(f"[alembic/env.py] === STARTING IMPORTS ===", file=sys.stderr)
try:
    print(f"[alembic/env.py] Attempting: from database import Base, Interaction, AppleScriptAttempt, FileIndex", file=sys.stderr)
    from database import Base, Interaction, AppleScriptAttempt, FileIndex
    print(f"[alembic/env.py]   Successfully imported from 'database'. Type of Base: {{type(Base)}}", file=sys.stderr)

    if Base is not None and hasattr(Base, 'metadata'):
        target_metadata = Base.metadata
        if target_metadata is not None and hasattr(target_metadata, 'tables'):
            print(f"[alembic/env.py]   Base.metadata is valid. Tables found: {{list(target_metadata.tables.keys())}}", file=sys.stderr)
            if not target_metadata.tables:
                print("[alembic/env.py]   WARNING: Base.metadata.tables collection is EMPTY!", file=sys.stderr)
        else:
            print("[alembic/env.py]   CRITICAL: Base.metadata is None or has no 'tables' attribute!", file=sys.stderr)
            target_metadata = None # Ensure it's None if problematic
    else:
        print("[alembic/env.py]   CRITICAL: Imported 'Base' is None or has no 'metadata' attribute!", file=sys.stderr)
        target_metadata = None # Ensure it's None

    print(f"[alembic/env.py] Attempting: from config import PROJECT_CONFIG_DATABASE_URL", file=sys.stderr)
    from config import PROJECT_CONFIG_DATABASE_URL 
    EFFECTIVE_DATABASE_URL_FOR_ALEMBIC = PROJECT_CONFIG_DATABASE_URL
    print(f"[alembic/env.py]   Successfully imported PROJECT_CONFIG_DATABASE_URL='{EFFECTIVE_DATABASE_URL_FOR_ALEMBIC}'", file=sys.stderr)

except ImportError as e_import_env:
    print(f"[alembic/env.py] CRITICAL IMPORT ERROR: {{e_import_env}}", file=sys.stderr)
    raise 
except Exception as e_env_setup:
    print(f"[alembic/env.py] CRITICAL UNEXPECTED SETUP ERROR: {{e_env_setup}}", file=sys.stderr)
    raise
print(f"[alembic/env.py] === IMPORTS FINISHED ===", file=sys.stderr)

if target_metadata is None: # Check after try-except
    print("[alembic/env.py] CRITICAL: target_metadata is None after import attempts. Autogenerate will be empty.", file=sys.stderr)
    # Optionally, could create a dummy MetaData here to prevent Alembic from crashing,
    # but an empty migration is the symptom we want to see if this is the case.
    # from sqlalchemy import MetaData
    # target_metadata = MetaData() # This would lead to empty script

if not EFFECTIVE_DATABASE_URL_FOR_ALEMBIC:
    raise ValueError("PROJECT_CONFIG_DATABASE_URL from config.py was None. Alembic cannot proceed.")

alembic_ini_config.set_main_option("sqlalchemy.url", EFFECTIVE_DATABASE_URL_FOR_ALEMBIC)

if alembic_ini_config.config_file_name is not None:
    try: fileConfig(alembic_ini_config.config_file_name)
    except Exception as fc_err: print(f"[alembic/env.py] Warn: Error log config from alembic.ini: {{fc_err}}", file=sys.stderr)

is_sqlite = EFFECTIVE_DATABASE_URL_FOR_ALEMBIC.startswith("sqlite")

def run_migrations_offline() -> None:
    print("[alembic/env.py] Configuring for OFFLINE migration.", file=sys.stderr)
    context.configure(url=EFFECTIVE_DATABASE_URL_FOR_ALEMBIC, target_metadata=target_metadata, literal_binds=True, dialect_opts={{"paramstyle": "named"}}, render_as_batch=is_sqlite, compare_server_default=True)
    with context.begin_transaction(): context.run_migrations()

def run_migrations_online() -> None:
    print("[alembic/env.py] Configuring for ONLINE migration.", file=sys.stderr)
    connectable_args = {{"connect_args": {{"check_same_thread": False, "timeout": 60.0}}}} if is_sqlite else {{}}
    connectable = create_engine(EFFECTIVE_DATABASE_URL_FOR_ALEMBIC, poolclass=pool.NullPool, **connectable_args) # type: ignore
    try:
        with connectable.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=is_sqlite, compare_type=True, compare_server_default=True)
            print("[alembic/env.py] Context configured for online. Beginning transaction and running migrations...", file=sys.stderr)
            with context.begin_transaction():
                context.run_migrations()
            print("[alembic/env.py] Online migrations ran.", file=sys.stderr)
    finally:
        if connectable: connectable.dispose()

if context.is_offline_mode(): run_migrations_offline()
else: run_migrations_online()
print("[alembic/env.py] Script execution finished.", file=sys.stderr)
"""
    # Logic to write/overwrite env.py if needed (as before)
    write_env_py = True
    if os.path.exists(ALEMBIC_ENV_PY_PATH):
        try:
            with open(ALEMBIC_ENV_PY_PATH, 'r') as f_current_env: current_content = f_current_env.read()
            if current_content.strip() == env_py_content.strip(): write_env_py = False; logger.trace(f"Alembic env.py at {ALEMBIC_ENV_PY_PATH} is up-to-date.")
            else: logger.warning(f"Alembic env.py at {ALEMBIC_ENV_PY_PATH} differs. Overwriting.")
        except Exception as e_read_env: logger.warning(f"Could not read existing env.py, will overwrite: {e_read_env}")
    if write_env_py:
        try:
            os.makedirs(os.path.dirname(ALEMBIC_ENV_PY_PATH), exist_ok=True)
            with open(ALEMBIC_ENV_PY_PATH, 'w') as f: f.write(env_py_content)
            logger.success(f"✅ Alembic env.py created/overwritten at {ALEMBIC_ENV_PY_PATH}.")
        except IOError as e: logger.error(f"❌ Failed to write default env.py: {e}"); raise

def _create_default_script_mako():
    # Overwrite mako template each time to ensure it's the correct version,
    # or check content before writing if you prefer.
    # For simplicity of ensuring the fix, let's overwrite if different or missing.
    logger.info(f"Ensuring Alembic script.py.mako at {ALEMBIC_SCRIPT_MAKO_PATH} is correctly formatted...")

    # Note the f"""\ at the beginning for a clean multi-line f-string
    mako_content = f"""\
\"\"\"${{message}}\"\"\"
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

${{imports if imports else ""}}

# revision identifiers, used by Alembic.
revision: str = ${{repr(up_revision)}}
down_revision: Union[str, None] = ${{repr(down_revision)}}
branch_labels: Union[str, Sequence[str], None] = ${{repr(branch_labels)}}
depends_on: Union[str, Sequence[str], None] = ${{repr(depends_on)}}


def upgrade() -> None:
    ${{upgrades if upgrades else "pass"}}


def downgrade() -> None:
    ${{downgrades if downgrades else "pass"}}
"""
    write_mako_file = True
    if os.path.exists(ALEMBIC_SCRIPT_MAKO_PATH):
        try:
            with open(ALEMBIC_SCRIPT_MAKO_PATH, 'r') as f_current_mako:
                current_mako_content = f_current_mako.read()
            if current_mako_content.strip() == mako_content.strip(): # Compare stripped content
                write_mako_file = False
                logger.trace(f"Alembic script.py.mako at {ALEMBIC_SCRIPT_MAKO_PATH} is already up-to-date.")
            else:
                logger.warning(f"Alembic script.py.mako at {ALEMBIC_SCRIPT_MAKO_PATH} differs. Overwriting.")
        except Exception as e_read_mako:
            logger.warning(f"Could not read existing script.py.mako to compare, will overwrite: {e_read_mako}")

    if write_mako_file:
        try:
            # Ensure the parent directory exists (alembic/)
            os.makedirs(os.path.dirname(ALEMBIC_SCRIPT_MAKO_PATH), exist_ok=True)
            with open(ALEMBIC_SCRIPT_MAKO_PATH, 'w') as f:
                f.write(mako_content)
            logger.success(f"✅ Default script.py.mako created/overwritten at {ALEMBIC_SCRIPT_MAKO_PATH}.")
        except IOError as e:
            logger.error(f"❌ Failed to write default script.py.mako: {e}")
            raise # Re-raise if writing fails, as it's a critical setup file
# --- END Alembic Helper Functions ---

# --- END Alembic Helpers ---

# --- Migration Application Function (check_and_apply_migrations) ---
# (Keep your existing check_and_apply_migrations - unchanged)
def check_and_apply_migrations() -> bool:
    logger.info("🔬 Applying pending Alembic migrations (to runtime DB)...")
    alembic_cfg = _get_alembic_config()
    if not alembic_cfg: logger.error("Cannot apply migrations: Alembic config failed."); return False
    try:
        command.upgrade(alembic_cfg, "head")
        logger.success("✅ 'alembic upgrade head' command finished successfully (or DB was already at head).")
        return True
    except CommandError as ce:
        logger.error(f"❌ Alembic 'upgrade' command failed: {ce}")
        err_str = str(ce).lower()
        if "sqlite" in err_str and ("alter table" in err_str or "add column" in err_str):
            logger.critical(
                "   >>> Potential SQLite migration failure! Ensure 'render_as_batch=True' in alembic/env.py. <<<")
        elif "already exists" in err_str:
            logger.critical("   >>> 'Table already exists' error. Alembic state might be inconsistent. <<<")
        else:
            logger.error("   Manual Alembic intervention likely required. Check logs and Alembic state.")
        return False
    except Exception as upg_err:
        logger.error(f"❌ Unexpected error during 'alembic upgrade': {upg_err}"); return False


# --- END Migration Application ---

def start_log_batch_writer():
    global _log_writer_thread
    if not (_log_writer_thread and _log_writer_thread.is_alive()):
        if SessionLocal is None or not SessionLocal.kw.get('bind'): logger.error("Cannot start Log Batch Writer: SessionLocal not configured."); return # type: ignore
        logger.info("🚀 Starting DB Log Batch Writer..."); _log_writer_stop_event.clear(); _log_writer_thread = DatabaseLogBatchWriter(_log_writer_stop_event); _log_writer_thread.start()
def stop_log_batch_writer(wait_for_flush_timeout: Optional[float] = 10.0):
    global _log_writer_thread
    if _log_writer_thread and isinstance(_log_writer_thread, DatabaseLogBatchWriter) and _log_writer_thread.is_alive():
        logger.info("Signaling Log Batch Writer to shutdown..."); _log_writer_thread.signal_shutdown_and_flush(); _log_writer_stop_event.set()
        if wait_for_flush_timeout and wait_for_flush_timeout > 0: _log_writer_thread.join(timeout=wait_for_flush_timeout)
        logger.info(f"Log Batch Writer {'finished' if not _log_writer_thread.is_alive() else 'did not finish in timeout'}.")

# --- init_db - Modified to start snapshotter ---
def init_db():
    global _engine, SessionLocal  # To ensure they are assigned if this is the first init
    logger.info("🚀 Initializing Database (Strategy: Always Fresh Alembic Baseline)...")

    # Block to prevent re-entry if already initialized in this process
    if _engine is not None and SessionLocal is not None and SessionLocal.kw.get('bind'):  # type: ignore
        logger.warning("Database already initialized in this process. Skipping re-initialization.")
        # Ensure background services are running if app is reloaded (e.g., by Hypercorn worker restart)
        if ENABLE_DB_SNAPSHOTS and (_snapshotter_thread is None or not _snapshotter_thread.is_alive()):  # type: ignore
            start_db_snapshotter()
        if (_log_writer_thread is None or not _log_writer_thread.is_alive()):  # type: ignore
            if SessionLocal and SessionLocal.kw.get('bind'):  # type: ignore
                start_log_batch_writer()
            else:  # Should not happen if already initialized
                logger.error("init_db (re-entry): SessionLocal not bound, cannot start log writer.")
        return

    # Create SessionLocal factory once if it doesn't exist. get_engine() will bind it.
    if SessionLocal is None:
        SessionLocal = sessionmaker(autocommit=False, autoflush=False)
        logger.debug("database.py init_db: SessionLocal factory created.")

    # --- 1. Get/Prepare Engine and Ensure DB File is Ready ---
    # get_engine() handles archive decompression, snapshot restore, basic integrity checks.
    # It initializes global _engine and binds global SessionLocal.
    engine_instance = get_engine()  # This must succeed and configure SessionLocal.
    if not engine_instance or not _engine or not SessionLocal or not SessionLocal.kw.get('bind'):  # type: ignore
        # get_engine() should sys.exit on critical failure, this is a fallback assertion.
        logger.critical("CRITICAL: Engine or SessionLocal not properly initialized by get_engine(). Cannot proceed.")
        raise RuntimeError("Fatal error in get_engine() preventing Alembic setup or DB operations.")
    logger.info("Engine and SessionLocal are ready. Runtime database file prepared by get_engine().")

    # --- 2. Aggressively Reset Alembic State for a Fresh Baseline ---
    logger.warning("INIT_DB STRATEGY: Deleting existing Alembic versions and DB history for a fresh baseline.")

    # 2a. Clear contents of alembic/versions directory
    if os.path.isdir(ALEMBIC_VERSIONS_PATH):
        logger.info(f"Clearing Alembic versions directory: {ALEMBIC_VERSIONS_PATH}")
        try:
            for filename in os.listdir(ALEMBIC_VERSIONS_PATH):
                file_path = os.path.join(ALEMBIC_VERSIONS_PATH, filename)
                if os.path.isfile(file_path) and file_path.endswith((".py", ".pyc")):
                    os.unlink(file_path)
                elif os.path.isdir(file_path) and filename == "__pycache__":
                    shutil.rmtree(file_path)
            logger.info("Alembic versions directory cleared.")
        except Exception as e_clear_versions:
            logger.error(f"Failed to fully clear alembic/versions: {e_clear_versions}. Proceeding cautiously.")
    else:  # Ensure the directory exists for Alembic to write into
        os.makedirs(ALEMBIC_VERSIONS_PATH, exist_ok=True)
        logger.info(f"Created Alembic versions directory: {ALEMBIC_VERSIONS_PATH}")

    # 2b. Drop the alembic_version table from the database to reset history
    logger.info("Dropping 'alembic_version' table from the database if it exists...")
    try:
        with engine_instance.connect() as connection:  # Use the engine_instance from get_engine()
            connection.execute(text("DROP TABLE IF EXISTS alembic_version;"))
            connection.commit()
        logger.info("'alembic_version' table dropped (or did not exist).")
    except Exception as e_drop_table:
        logger.error(
            f"Failed to drop 'alembic_version' table: {e_drop_table}. This might be okay if DB is truly new and table never existed.")
        # If this fails on an existing DB, subsequent Alembic commands might still error,
        # but the goal is to make it behave like a fresh DB.

    # --- 3. Ensure Alembic Configuration Files Exist and are Correct ---
    # These functions create default ini, env.py, script.py.mako if missing.
    # _create_default_env_py() is critical as it writes an env.py that correctly
    # imports Base.metadata from this database.py and PROJECT_CONFIG_DATABASE_URL from config.py.
    _create_default_env_py()  # Must be defined in this file
    _create_default_script_mako()  # Must be defined in this file
    if not os.path.exists(ALEMBIC_INI_PATH):
        logger.warning(f"🔧 Alembic config file alembic.ini not found. Creating default at {ALEMBIC_INI_PATH}")
        alembic_dir_for_ini = os.path.relpath(ALEMBIC_DIR, MODULE_DIR).replace("\\", "/")
        # alembic.ini's sqlalchemy.url will be overridden by env.py with PROJECT_CONFIG_DATABASE_URL,
        # but it needs a valid placeholder. Using DATABASE_URL from config is fine.
        alembic_cfg_content = f"""
[alembic]
# path to migration scripts
script_location = {alembic_dir_for_ini}

# SQLAlchemy database URL
sqlalchemy.url = {DATABASE_URL}

# Other Alembic settings can go here if needed
# e.g. file_template, timezone, etc.

[post_write_hooks]
# This section is optional, used for code auto-formatting after generation
# hooks = autopep8
# autopep8.type = command
# autopep8.entrypoint = autopep8
# autopep8.options = --in-place --aggressive --aggressive

# Logging configuration (simplified to avoid 'command' formatter issue)
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
handlers = # No specific handler, inherits from root or default
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers = console
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
            with open(ALEMBIC_INI_PATH, 'w') as f:
                f.write(alembic_cfg_content)
            logger.success(f"📝 Default alembic.ini created at {ALEMBIC_INI_PATH}.")
        except IOError as e:
            logger.error(f"❌ Failed to write default alembic.ini: {e}")
            raise  # This is a critical setup step

    alembic_cfg = _get_alembic_config()  # Must be defined in this file
    if not alembic_cfg:
        raise RuntimeError("Failed to load/create Alembic configuration (alembic.ini).")

    # --- 4. Autogenerate and Apply Fresh Baseline Migration ---
    migration_successful = False
    try:
        logger.info("Attempting to auto-generate a new baseline migration script...")
        autogen_message = f"Baseline_schema_on_startup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            # With versions folder empty and alembic_version table gone (or empty),
            # this should create a new initial script based on current models.
            command.revision(alembic_cfg, message=autogen_message, autogenerate=True)
            logger.success(
                f"Alembic 'revision --autogenerate' for baseline executed. Check 'alembic/versions' for the new script.")
        except Exception as auto_err:
            logger.error(f"CRITICAL: Failed to autogenerate Alembic baseline migration: {auto_err}")
            logger.exception("Alembic autogenerate baseline TRACEBACK:")
            # If autogeneration itself fails, it often indicates issues with env.py, model definitions,
            # or Alembic's ability to connect to the DB (even if just for inspection).
            raise RuntimeError(f"Alembic baseline autogeneration failed critically: {auto_err}") from auto_err

        logger.info("Attempting to apply migrations to head (should apply the new baseline)...")
        try:
            command.upgrade(alembic_cfg, "head")
            logger.success("✅ 'alembic upgrade head' command finished successfully.")
            migration_successful = True
        except Exception as upg_err:
            logger.error(f"❌ Alembic 'upgrade head' FAILED: {upg_err}")
            logger.exception("Alembic upgrade TRACEBACK:")
            # This is where "NOT NULL constraint failed" errors would typically appear
            # if model definitions (server_default) are not correctly handled by the autogenerated script.
            migration_successful = False  # Ensure it's false

        if not migration_successful:
            # This means either autogen produced a bad script or the upgrade itself failed.
            # With the "always fresh" strategy, this indicates a fundamental problem.
            raise RuntimeError(
                "Alembic upgrade failed after attempting fresh baseline. Check model definitions (esp. NOT NULL Booleans need server_default=text('0')) and the autogenerated script in alembic/versions/.")

        # --- 5. Schema Verification (Optional but Recommended) ---
        logger.info("📊 Verifying Database Schema after fresh baseline and upgrade...")
        try:
            inspector = sql_inspect(engine_instance)
            table_names = inspector.get_table_names()
            expected_tables = {'alembic_version', 'interactions', 'applescript_attempts',
                               'file_index'}  # Add all your tables
            if not expected_tables.issubset(set(table_names)):
                missing_tbl_str = f"Missing expected tables: {expected_tables - set(table_names)}"
                logger.error(f"‼️ {missing_tbl_str}")
                raise RuntimeError(f"Schema verification failed: {missing_tbl_str}")
            logger.info("✅ All expected tables are present.")

            # Example detailed check for a specific table and key columns
            if "interactions" in table_names:
                interaction_cols_info = inspector.get_columns("interactions")
                interaction_col_names = {col['name'] for col in interaction_cols_info}
                expected_interaction_cols_subset = {"id", "timestamp", "user_input", "llm_response",
                                                    "reflection_indexed_in_vs", "tot_analysis_spawned"}
                if not expected_interaction_cols_subset.issubset(interaction_col_names):
                    missing_interaction_cols_str = f"Missing key columns in 'interactions': {expected_interaction_cols_subset - interaction_col_names}"
                    logger.error(f"‼️ {missing_interaction_cols_str}")
                    raise RuntimeError(f"Schema verification failed: {missing_interaction_cols_str}")
                logger.info("✅ Table 'interactions' contains key expected columns.")
            logger.success("✅ Database schema verification passed (basic checks).")
        except Exception as verify_err:
            logger.error(f"❌ Schema verification FAILED: {verify_err}")
            raise RuntimeError(f"Schema verification failed after migration: {verify_err}") from verify_err

        logger.success("✅ Database initialization (fresh Alembic baseline strategy) completed successfully.")

        # --- Start Background Services ---
        # These are started only if all previous steps were successful.
        start_db_snapshotter()
        start_log_batch_writer()

        # Register shutdown hooks (order can be important for graceful shutdown)
        atexit.register(stop_log_batch_writer)  # Flush logs first
        atexit.register(stop_db_snapshotter)  # Stop snapshotting
        atexit.register(_shutdown_hook)  # Then compress DB

    except Exception as init_err_outer:  # Catches RuntimeErrors raised from within or other exceptions
        logger.critical(f"🔥🔥 OVERALL DATABASE INITIALIZATION FAILURE: {init_err_outer}")
        # Traceback should have been logged by the block that raised the RuntimeError.
        # If not, or for other exceptions:
        if not isinstance(init_err_outer, RuntimeError):  # Log if it wasn't one of our explicit RuntimeErrors
            logger.exception("Database Initialization Full Traceback (Outer Catch):")

        # Attempt to gracefully stop threads if they were somehow started before the failure
        if _log_writer_thread and _log_writer_thread.is_alive():  # type: ignore
            stop_log_batch_writer(wait_for_flush_timeout=1.0)  # Quick flush attempt
        if _snapshotter_thread and _snapshotter_thread.is_alive():  # type: ignore
            stop_db_snapshotter()

        # Re-raise so app.py's top-level try-except during import can catch it and sys.exit.
        # No need for another sys.exit() here, let the caller handle final termination.
        raise


# --- END init_db ---


# --- Database Interaction Functions (add_interaction, get_recent_interactions, etc.) ---
# (Keep your existing DB interaction functions here - unchanged from V16)
def add_interaction(db_session_param: Optional[Session] = None, **kwargs) -> Optional[Interaction]:
    created_session_locally = False
    actual_db_session: Optional[Session] = db_session_param

    if actual_db_session is None:
        if SessionLocal is None:
            logger.error("add_interaction (direct): SessionLocal is None. Cannot create DB session.")
            return None
        try:
            actual_db_session = SessionLocal()  # type: ignore
            created_session_locally = True
        except Exception as e_sess:
            logger.error(f"add_interaction (direct): Failed to create local DB session: {e_sess}")
            return None

    if actual_db_session is None:  # Should not happen if SessionLocal() worked
        logger.error("add_interaction (direct): DB session is unexpectedly None after creation attempt.")
        return None

    interaction_instance: Optional[Interaction] = None
    try:
        interaction_instance = _core_add_interaction_to_session(actual_db_session, **kwargs)
        actual_db_session.commit()
        actual_db_session.refresh(interaction_instance)

        log_level = "INFO"
        log_extra = ""
        if interaction_instance.input_type == 'error' or interaction_instance.input_type == 'log_error':
            log_level = "ERROR"
            log_extra = " ❌ ERROR (Immediate)"
        elif interaction_instance.input_type == 'log_warning':
            log_level = "WARNING"
            log_extra = " ⚠️ WARN (Immediate)"
        logger.log(log_level,
                   f"💾 Interaction {interaction_instance.id} ({interaction_instance.mode}/{interaction_instance.input_type}){log_extra} - Immediate Write")
        return interaction_instance
    except Exception as e:
        logger.error(f"add_interaction (Direct Write): Failed: {e}")
        if actual_db_session: actual_db_session.rollback()
        return None
    finally:
        if created_session_locally and actual_db_session:
            actual_db_session.close()


def get_recent_interactions(db: Session, limit=5, session_id=None, mode="chat", include_logs=False) -> List[
    Interaction]:
    try:
        base_query = db.query(Interaction).filter(Interaction.mode == mode)
        if session_id: base_query = base_query.filter(Interaction.session_id == session_id)
        if not include_logs:
            log_types_to_exclude = ['log_warning', 'log_error', 'log_debug', 'log_info', 'error', 'system', 'url',
                                    'image', 'latex_analysis_result']
            base_query = base_query.filter(Interaction.input_type.notin_(log_types_to_exclude))
        results = base_query.order_by(desc(Interaction.timestamp)).limit(limit).all()
        results.reverse()
        return results
    except Exception as e:
        logger.error(f"Error fetching recent interactions: {e}"); return []

_log_batch_queue = deque()
_log_batch_queue_lock = threading.Lock()
_log_writer_stop_event = threading.Event() # <<<< DEFINITION

_log_writer_thread: Optional[threading.Thread] = None # This is the instance of DatabaseLogBatchWriter
# LOG_QUEUE_MAX_SIZE is imported from config

def _core_add_interaction_to_session(db_session: Session, **kwargs) -> Interaction:
    valid_keys = {c.name for c in Interaction.__table__.columns}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    filtered_kwargs.setdefault('mode', 'chat')
    if 'input_type' not in filtered_kwargs:
        if filtered_kwargs.get('user_input'):
            filtered_kwargs['input_type'] = 'text'
        elif filtered_kwargs.get('llm_response'):
            filtered_kwargs['input_type'] = 'llm_response'
        else:
            filtered_kwargs['input_type'] = 'log_info'  # Default for unspecified logs

    # Ensure all boolean NOT NULL fields have defaults if not provided in kwargs,
    # matching the server_default which is for DDL. This is for Python object creation.
    defaults_to_set = {
        'reflection_completed': False, 'tot_delivered': False, 'tot_analysis_requested': False,
        'tot_analysis_spawned': False, 'assistant_action_executed': False,
        'reflection_indexed_in_vs': False,
        'requires_deep_thought': None  # This one is nullable
    }
    for field, default_val in defaults_to_set.items():
        # Only set default if field is part of the model (valid_keys check already did this broadly)
        # and if it's not already in filtered_kwargs
        if field in valid_keys and field not in filtered_kwargs:
            filtered_kwargs[field] = default_val
        elif field in valid_keys and filtered_kwargs.get(
                field) is None and default_val is not None:  # If explicitly passed as None but shouldn't be
            if not Interaction.__table__.columns[field].nullable and default_val is not None:
                filtered_kwargs[field] = default_val

    interaction = Interaction(**filtered_kwargs)
    db_session.add(interaction)
    return interaction



def queue_interaction_for_batch_logging(**kwargs) -> None:
    if _log_writer_thread is None or not _log_writer_thread.is_alive():
        logger.warning("Log writer thread not active. Logging directly (fallback) for: input_type='{}'".format(kwargs.get("input_type", "unknown")))
        add_interaction(**kwargs)
        return

    with _log_batch_queue_lock:
        if len(_log_batch_queue) < LOG_QUEUE_MAX_SIZE:
            kwargs.setdefault('timestamp', datetime.datetime.now(datetime.timezone.utc))
            _log_batch_queue.append(kwargs)
        else:
            logger.warning(f"Log batch queue is full (size: {LOG_QUEUE_MAX_SIZE}). Discarding log: {kwargs.get('input_type')}")
            try:
                add_interaction(input_type="log_error", user_input=f"[Log Q Full] Discarded: {kwargs.get('input_type')}",
                                llm_response=f"OrigData: {str(kwargs)[:500]}",
                                session_id=kwargs.get('session_id'), mode=kwargs.get('mode', 'chat'))
            except Exception as e_crit_log:
                logger.error(f"Failed to log 'Log Queue Full' event directly: {e_crit_log}")

def get_pending_tot_result(db: Session, session_id: str) -> Optional[Interaction]:
    try:
        if 'tot_delivered' not in Interaction.__table__.columns: return None
        result = db.query(Interaction).filter(
            Interaction.session_id == session_id, Interaction.mode == 'chat',
            Interaction.tot_analysis_requested == True, Interaction.tot_result.isnot(None),
            Interaction.tot_result.notlike('Error%'), Interaction.tot_delivered == False
        ).order_by(desc(Interaction.timestamp)).first()
        return result
    except Exception as e:
        logger.error(f"Error fetching pending ToT for {session_id}: {e}"); return None


def mark_tot_delivered(db: Session, interaction_id: int):
    try:
        if 'tot_delivered' not in Interaction.__table__.columns: return
        stmt = update(Interaction).where(Interaction.id == interaction_id).values(tot_delivered=True)
        db.execute(stmt)
        db.commit()
    except Exception as e:
        logger.error(f"Error marking ToT delivered for ID {interaction_id}: {e}"); db.rollback()


def get_past_tot_interactions(db: Session, limit=50) -> List[Interaction]:
    try:
        if not {'tot_analysis_requested', 'tot_result'}.issubset(Interaction.__table__.columns.keys()): return []
        results = db.query(Interaction).filter(
            Interaction.mode == "chat", Interaction.tot_analysis_requested == True,
            Interaction.tot_result.isnot(None), Interaction.tot_result.notlike('Error%')
        ).order_by(desc(Interaction.timestamp)).limit(limit).all()
        return results
    except Exception as e:
        logger.error(f"Error fetching past ToT: {e}"); return []


def get_global_recent_interactions(db: Session, limit: int = 5) -> List[Interaction]:
    try:
        results = db.query(Interaction).filter(
            Interaction.mode == 'chat', Interaction.input_type.in_(['text', 'llm_response', 'image+text'])
        ).order_by(desc(Interaction.timestamp)).limit(limit).all()
        results.reverse()
        return results
    except Exception as e:
        logger.error(f"Error fetching global recent interactions: {e}"); return []


def get_past_applescript_attempts(db: Session, action_type: str, parameters_json: str, limit: int = 5) -> List[
    AppleScriptAttempt]:
    engine_instance = get_engine()
    if not engine_instance: return []
    try:
        inspector = sql_inspect(engine_instance)
        if not inspector.has_table(AppleScriptAttempt.__tablename__): return []
        results = db.query(AppleScriptAttempt).filter(
            AppleScriptAttempt.action_type == action_type,
            AppleScriptAttempt.parameters_json == parameters_json
        ).order_by(desc(AppleScriptAttempt.timestamp)).limit(limit).all()
        return results
    except Exception as e:
        logger.error(f"Error fetching past AppleScript attempts: {e}"); return []


def search_file_index(db: Session, query: str, limit: int = 10) -> List[FileIndex]:
    engine_instance = get_engine()
    if not engine_instance: return []
    try:
        inspector = sql_inspect(engine_instance)
        if not inspector.has_table(FileIndex.__tablename__): return []
        search_term = f"%{query}%"
        results = db.query(FileIndex).filter(
            (FileIndex.file_path.like(search_term)) | (FileIndex.indexed_content.like(search_term))
        ).order_by(desc(FileIndex.last_modified_os)).limit(limit).all()
        return results
    except Exception as e:
        logger.error(f"Error searching file index: {e}"); return []


class DatabaseLogBatchWriter(threading.Thread):  # <<<< DEFINITION
    def __init__(self, stop_event: threading.Event):  # stop_event is the global _log_writer_stop_event
        super().__init__(name="DatabaseLogBatchWriterThread", daemon=True)
        self.stop_event = stop_event
        self.is_shutting_down = False
        logger.info(
            f"DatabaseLogBatchWriter initialized. BatchSize: {LOG_BATCH_SIZE}, FlushInterval: {LOG_FLUSH_INTERVAL_SECONDS}s")

    def _write_batch(self, items_to_write: List[Dict]):
        if not items_to_write:
            return

        db_session_writer: Optional[Session] = None
        successful_items_in_batch = 0
        try:
            if SessionLocal is None:  # Should be configured by init_db -> get_engine
                logger.error("DatabaseLogBatchWriter: SessionLocal factory is None. Cannot write batch.")
                return

            db_session_writer = SessionLocal()  # type: ignore

            for item_data in items_to_write:
                try:
                    item_data.setdefault('input_type', 'log_info_batched')  # Default for batched logs
                    _core_add_interaction_to_session(db_session_writer, **item_data)
                    successful_items_in_batch += 1
                except Exception as e_item:
                    logger.error(
                        f"DatabaseLogBatchWriter: Error adding single item to batch session: {e_item}. Data: {str(item_data)[:200]}")

            if successful_items_in_batch > 0:
                db_session_writer.commit()
                logger.debug(f"DatabaseLogBatchWriter: Committed batch of {successful_items_in_batch} log entries.")
            elif items_to_write:
                logger.warning("DatabaseLogBatchWriter: No items successfully added to session in this batch.")

        except SQLAlchemyError as e_sql:
            logger.error(f"DatabaseLogBatchWriter: SQLAlchemyError during batch processing/commit: {e_sql}")
            if db_session_writer: db_session_writer.rollback()
        except Exception as e_batch:
            logger.error(f"DatabaseLogBatchWriter: Unexpected error during batch write: {e_batch}")
            if db_session_writer: db_session_writer.rollback()
        finally:
            if db_session_writer:
                db_session_writer.close()

    def run(self):
        logger.info("💾 DatabaseLogBatchWriter thread started.")
        while not self.stop_event.is_set() or self.is_shutting_down:
            items_this_run: List[Dict] = []
            is_final_flush_attempt = self.is_shutting_down and self.stop_event.is_set()

            with _log_batch_queue_lock:
                count = 0
                # Drain fully on final flush, otherwise up to batch size
                while _log_batch_queue and (count < LOG_BATCH_SIZE or is_final_flush_attempt):
                    items_this_run.append(_log_batch_queue.popleft())
                    count += 1

            if items_this_run:
                self._write_batch(items_this_run)

            if is_final_flush_attempt and not _log_batch_queue:  # Final flush done and queue is empty
                logger.info("DatabaseLogBatchWriter: Final flush complete. Queue empty. Exiting run loop.")
                break

            if not items_this_run and not is_final_flush_attempt:  # Queue was empty, not shutting down
                # logger.trace(f"DatabaseLogBatchWriter: Queue empty, waiting for {LOG_FLUSH_INTERVAL_SECONDS}s or stop signal...")
                self.stop_event.wait(timeout=LOG_FLUSH_INTERVAL_SECONDS)
            elif not is_final_flush_attempt:  # Processed items or still flushing, yield briefly
                time.sleep(0.01)
        logger.info("💾 DatabaseLogBatchWriter thread stopped.")

    def signal_shutdown_and_flush(self):
        logger.info("DatabaseLogBatchWriter: Shutdown signaled. Will attempt final flush on next opportunity.")
        self.is_shutting_down = True
        # The main stop_event.set() will wake up the run loop if it's waiting on the timeout.
# --- End of database.py ---

if __name__ != "__main__":
    logger.debug("---------------------------------------------------------------------")
    logger.debug("DATABASE.PY MODULE-LEVEL DIAGNOSTIC (when imported by app.py):")
    try:
        if 'Base' in globals() and Base is not None and hasattr(Base, 'metadata') and Base.metadata is not None: # type: ignore
            logger.debug(f"  database.py: Base.metadata tables: {list(Base.metadata.tables.keys())}") # type: ignore
            if not Base.metadata.tables: # type: ignore
                logger.warning("  database.py WARNING: Base.metadata has no tables registered when database.py finishes importing!")
        else:
            logger.error("  database.py ERROR: 'Base' or 'Base.metadata' is None or malformed at module import time.")
    except Exception as e_diag:
        logger.error(f"  database.py DIAGNOSTIC ERROR: {e_diag}")
    logger.debug("---------------------------------------------------------------------")