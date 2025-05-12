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

from loguru import logger

# --- Configuration & Paths ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_BASENAME = "mappedknowledge.db"
RUNTIME_DB_PATH = os.path.join(APP_DIR, DB_BASENAME)
DB_PATH_COMPRESSED = f"{RUNTIME_DB_PATH}.zst"
RUNTIME_DATABASE_URL = f"sqlite:///{os.path.abspath(RUNTIME_DB_PATH)}"

# --- NEW: Snapshot Configuration ---
ENABLE_DB_SNAPSHOTS = os.getenv("ENABLE_DB_SNAPSHOTS", "true").lower() in ('true', '1', 't', 'yes', 'y')
DB_SNAPSHOT_INTERVAL_MINUTES = int(os.getenv("DB_SNAPSHOT_INTERVAL_MINUTES", 1))  # Default: 1 minute
DB_SNAPSHOT_DIR_NAME = "db_snapshots"
DB_SNAPSHOT_DIR = os.path.join(APP_DIR, DB_SNAPSHOT_DIR_NAME)
DB_SNAPSHOT_RETENTION_COUNT = int(
    os.getenv("DB_SNAPSHOT_RETENTION_COUNT", 60))  # Keep last 60 snapshots (e.g., 1 hour worth if 1-min interval)
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
# (Keep your existing model definitions here - unchanged from previous version)
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
    # tool_result field removed as it was not in the latest schema
    image_data = Column(Text, nullable=True)
    url_processed = Column(String, nullable=True)
    image_description = Column(Text, nullable=True)
    latex_representation = Column(Text, nullable=True)
    latex_explanation = Column(Text, nullable=True)
    execution_time_ms = Column(Float, nullable=True)
    reflection_completed = Column(Boolean, default=False, nullable=False, index=True)
    rag_source_url = Column(String, nullable=True)
    rag_history_ids = Column(String, nullable=True)
    requires_deep_thought = Column(Boolean, nullable=True)
    deep_thought_reason = Column(Text, nullable=True)
    tot_analysis_requested = Column(Boolean, default=False)
    tot_result = Column(Text, nullable=True)
    tot_delivered = Column(Boolean, default=False, index=True, server_default='0')
    assistant_action_analysis_json = Column(Text, nullable=True)
    assistant_action_type = Column(String, nullable=True, index=True)
    assistant_action_params = Column(Text, nullable=True)
    assistant_action_executed = Column(Boolean, default=False, server_default='0')
    assistant_action_result = Column(Text, nullable=True)
    last_modified_db = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    # --- NEW: Fields for imagined image (from app.py) ---
    imagined_image_prompt = Column(Text, nullable=True)
    imagined_image_b64 = Column(Text, nullable=True)
    imagined_image_vlm_description = Column(Text, nullable=True)
    # --- END NEW ---
    __table_args__ = (
        Index('ix_interactions_session_mode_timestamp', 'session_id', 'mode', 'timestamp'),
        Index('ix_interactions_undelivered_tot', 'session_id', 'mode', 'tot_delivered', 'timestamp'),
        Index('ix_interactions_action_type_time', 'assistant_action_type', 'timestamp'),
        Index('ix_interactions_reflection_pending', 'reflection_completed', 'mode', 'input_type', 'timestamp'),
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
                os.remove(target_path); logger.warning(
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

# --- Integrity Check and Repair (_check_and_repair_db) ---
# (Keep your existing _check_and_repair_db function - it's still useful as a last resort)
# Small modification: it will now be called *after* snapshot restore attempts.
def _check_and_repair_db(db_path: str, context_msg: str = "runtime DB") -> bool:
    """Checks SQLite DB integrity and attempts repair via dump/reload if needed."""
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        logger.debug(f"Integrity check skipped for {context_msg}: DB file not found or empty at {db_path}")
        return True

    logger.info(f"🩺 Performing integrity check on {context_msg} ('{os.path.basename(db_path)}')...")
    check_start_time = time.monotonic()
    is_ok = False
    sqlite3_cmd = shutil.which("sqlite3")
    if not sqlite3_cmd:
        logger.error("❌ Cannot run integrity check: 'sqlite3' command not found in PATH.")
        return False

    try:
        check_command = [sqlite3_cmd, db_path, "PRAGMA integrity_check;"]
        process = subprocess.run(check_command, capture_output=True, text=True, check=False, timeout=300)
        check_output = process.stdout.strip()
        check_duration = time.monotonic() - check_start_time
        logger.info(f"Integrity check for {context_msg} completed in {check_duration:.2f}s.")

        if process.returncode != 0:
            logger.error(
                f"Integrity check process for {context_msg} failed (RC={process.returncode}): {process.stderr.strip()}")
            is_ok = False
        elif check_output.lower() == "ok":
            logger.success(f"✅ Integrity check passed for {context_msg}.")
            is_ok = True
        else:
            logger.warning(f"🔥 Integrity check FAILED for {context_msg}. Errors reported:\n{check_output}")
            is_ok = False
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Integrity check for {context_msg} timed out.")
        return False
    except Exception as e:
        logger.error(f"❌ Error during integrity check execution for {context_msg}: {e}")
        return False

    if not is_ok:
        logger.warning(
            f"🚨 Attempting automatic repair via dump/reload for {context_msg} ('{os.path.basename(db_path)}')...")
        # (Keep the existing repair logic from your _check_and_repair_db)
        # ...
        # For brevity, assuming the dump/reload logic is the same as your V16
        # It should return True if repair was successful and final check passed, False otherwise.
        # Let's simulate that part:
        repair_start_time = time.monotonic()
        temp_dump_file = None
        repair_successful = False
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".sql", encoding='utf-8') as tmp_sql:
                temp_dump_file = tmp_sql.name
                dump_command = [sqlite3_cmd, db_path, ".dump"]
                dump_process = subprocess.run(dump_command, stdout=tmp_sql, stderr=subprocess.PIPE, text=True,
                                              check=False, timeout=600)
            if dump_process.returncode != 0: logger.error(
                f"Dump FAILED (RC={dump_process.returncode}). Stderr: {dump_process.stderr.strip()}"); return False
            corrupted_backup_path = f"{db_path}.corrupted_{int(time.time())}"
            shutil.move(db_path, corrupted_backup_path)
            reload_command = [sqlite3_cmd, db_path]
            with open(temp_dump_file, 'r', encoding='utf-8') as dump_fh:
                reload_process = subprocess.run(reload_command, stdin=dump_fh, capture_output=True, text=True,
                                                check=False, timeout=600)
            if reload_process.returncode != 0: logger.error(
                f"Reload FAILED (RC={reload_process.returncode}). Stderr: {reload_process.stderr.strip()}"); return False
            repair_duration = time.monotonic() - repair_start_time;
            logger.success(f"✅ Repair attempt ({repair_duration:.2f}s).")
            final_check_command = [sqlite3_cmd, db_path, "PRAGMA integrity_check;"]
            final_process = subprocess.run(final_check_command, capture_output=True, text=True, check=False,
                                           timeout=300)
            if final_process.stdout.strip().lower() == "ok":
                logger.success("✅ Repaired DB passed final check."); repair_successful = True
            else:
                logger.error(
                    f"❌ Repaired DB FAILED final check! Errors:\n{final_process.stdout.strip()}"); repair_successful = False
        except Exception as repair_err:
            logger.error(f"❌ Error during repair: {repair_err}"); repair_successful = False
        finally:
            if temp_dump_file and os.path.exists(temp_dump_file):
                try:
                    os.remove(temp_dump_file)
                except Exception:
                    pass
        return repair_successful
    return True


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
                    os.remove(final_snapshot_path); logger.warning(
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
    if _engine is not None: return _engine

    logger.info("Initializing SQLAlchemy engine (Runtime DB with Snapshot Restore)...")
    engine_instance = None
    db_successfully_prepared = False

    # 1. Try decompressing shutdown archive first
    if os.path.exists(DB_PATH_COMPRESSED):
        logger.info("Shutdown archive found, attempting decompression...")
        if _decompress_db(DB_PATH_COMPRESSED, RUNTIME_DB_PATH):  # Decompresses to RUNTIME_DB_PATH
            if os.path.exists(DB_PATH_COMPRESSED):  # Remove archive after successful decompression
                try:
                    os.remove(DB_PATH_COMPRESSED)
                except Exception as e:
                    logger.warning(f"Could not remove shutdown archive '{DB_PATH_COMPRESSED}': {e}")
            logger.info("Successfully prepared runtime DB from shutdown archive.")
            db_successfully_prepared = True
        else:
            logger.error("Failed to decompress shutdown archive. Will check for snapshots or create new.")
            db_successfully_prepared = False  # Continue to check snapshots

    # 2. If no runtime DB from archive, or if it exists but is read-only/corrupt, try snapshots
    if not db_successfully_prepared:
        is_readonly = _is_db_readonly(RUNTIME_DB_PATH) if os.path.exists(RUNTIME_DB_PATH) else False
        if not os.path.exists(RUNTIME_DB_PATH) or is_readonly:
            if is_readonly:
                logger.warning("Runtime DB is read-only. Attempting snapshot restore.")
            else:
                logger.info("Runtime DB not found. Attempting snapshot restore.")

            if _restore_from_latest_snapshot():
                logger.info("Successfully restored runtime DB from snapshot.")
                db_successfully_prepared = True
            else:
                logger.warning("Snapshot restore failed or no snapshots available.")
                # If restore failed AND the original was read-only, this is a problem.
                if is_readonly:
                    logger.critical(
                        "🔥🔥 FATAL: Runtime DB was read-only and snapshot restore failed. Manual intervention likely required.")
                    sys.exit(1)
                db_successfully_prepared = False  # Will lead to new DB creation by Alembic if file still missing
        else:  # Runtime DB exists and is not read-only (or read-only check failed to confirm)
            logger.info("Runtime DB exists and is not detected as read-only. Proceeding to integrity check.")
            db_successfully_prepared = True  # Tentatively true, pending integrity check

    # 3. Integrity Check / Repair on the prepared/existing runtime DB
    if db_successfully_prepared and os.path.exists(RUNTIME_DB_PATH):
        if not _check_and_repair_db(RUNTIME_DB_PATH, context_msg="prepared runtime DB"):
            logger.critical(f"🔥🔥 FATAL: DB integrity check/repair FAILED for '{os.path.basename(RUNTIME_DB_PATH)}'.")
            try:
                corrupt_final_path = f"{RUNTIME_DB_PATH}.final_corruption_{int(time.time())}"
                logger.warning(f"Moving final corrupted DB to {corrupt_final_path}")
                shutil.move(RUNTIME_DB_PATH, corrupt_final_path)
            except Exception as mv_err:
                logger.error(f"Could not move final corrupted DB: {mv_err}")
            db_successfully_prepared = False  # Leads to new DB creation
    elif not os.path.exists(RUNTIME_DB_PATH):  # If still no DB file after all attempts
        logger.info(
            "No existing database found (no archive, no snapshots, or all failed). Will create new via Alembic.")
        db_successfully_prepared = False  # Explicitly false, though covered

    # 4. Create Engine
    engine_args_internal = {"echo": False, "connect_args": {"check_same_thread": False, "timeout": 60.0}}
    logger.info(f"Creating SQLAlchemy engine for runtime DB: {RUNTIME_DATABASE_URL}")
    try:
        _engine = create_engine(RUNTIME_DATABASE_URL, **engine_args_internal)
        with _engine.connect() as connection:
            logger.debug("Engine connection test successful.")
        SessionLocal.configure(bind=_engine)
        logger.info("SQLAlchemy SessionLocal configured and bound to runtime DB engine.")
    except Exception as e:
        logger.critical(f"🔥🔥 DATABASE ENGINE CREATION FAILED for runtime DB: {e}")
        sys.exit(1)

    if SessionLocal is None or not SessionLocal.kw.get('bind'):
        raise RuntimeError("Failed to configure SessionLocal binding")
    return _engine


# --- END Engine Creation Wrapper ---


# --- Alembic Helper Functions (_get_alembic_config, _create_default_env_py, etc.) ---
# (Keep your existing Alembic helpers here - unchanged)
def _get_alembic_config() -> Optional[Config]:
    if not os.path.exists(ALEMBIC_INI_PATH): logger.error(f"Alembic ini not found: {ALEMBIC_INI_PATH}"); return None
    try:
        alembic_cfg = Config(ALEMBIC_INI_PATH)
        alembic_cfg.set_main_option("sqlalchemy.url", RUNTIME_DATABASE_URL)
        alembic_cfg.set_main_option("script_location", os.path.relpath(ALEMBIC_DIR, APP_DIR).replace("\\", "/"))
        return alembic_cfg
    except Exception as e:
        logger.error(f"Failed load/config Alembic from {ALEMBIC_INI_PATH}: {e}"); return None


def _create_default_env_py():
    if not os.path.exists(ALEMBIC_ENV_PY_PATH):
        logger.warning(f"🔧 Alembic env.py not found. Creating default at {ALEMBIC_ENV_PY_PATH}")
        env_py_content = f"""
import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool, create_engine
from alembic import context
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if APP_DIR not in sys.path: sys.path.insert(0, APP_DIR)
target_metadata = None; RUNTIME_DATABASE_URL_FROM_DB_PY = None
try:
    from database import Base, RUNTIME_DATABASE_URL
    RUNTIME_DATABASE_URL_FROM_DB_PY = RUNTIME_DATABASE_URL
    from database import Interaction, AppleScriptAttempt, FileIndex # Ensure all models are imported
    target_metadata = Base.metadata
except ImportError as e: print(f"Alembic env.py: ERROR importing: {{e}}"); raise
config = context.config
effective_db_url = RUNTIME_DATABASE_URL_FROM_DB_PY or config.get_main_option("sqlalchemy.url")
if not effective_db_url: raise ValueError("DB URL not found.")
config.set_main_option("sqlalchemy.url", effective_db_url)
if config.config_file_name is not None:
    try: fileConfig(config.config_file_name)
    except Exception as fc_err: print(f"Alembic env.py: Error log config: {{fc_err}}")
is_sqlite = effective_db_url.startswith("sqlite")
def run_migrations_offline() -> None:
    context.configure(url=effective_db_url, target_metadata=target_metadata, literal_binds=True, dialect_opts={{"paramstyle": "named"}}, render_as_batch=is_sqlite)
    with context.begin_transaction(): context.run_migrations()
def run_migrations_online() -> None:
    connectable_args = {{"connect_args": {{"check_same_thread": False, "timeout": 60.0}}}} if is_sqlite else {{}}
    connectable = create_engine(effective_db_url, poolclass=pool.NullPool, **connectable_args)
    try:
        with connectable.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=is_sqlite, compare_type=True)
            with context.begin_transaction(): context.run_migrations()
    finally:
        if connectable: connectable.dispose()
if context.is_offline_mode(): run_migrations_offline()
else: run_migrations_online()
"""
        try:
            os.makedirs(os.path.dirname(ALEMBIC_ENV_PY_PATH), exist_ok=True)
            with open(ALEMBIC_ENV_PY_PATH, 'w') as f:
                f.write(env_py_content)
            logger.success(f"✅ Default env.py created/overwritten.")
        except IOError as e:
            logger.error(f"❌ Failed to write default env.py: {e}"); raise


def _create_default_script_mako():
    if not os.path.exists(ALEMBIC_SCRIPT_MAKO_PATH):
        logger.warning(f"🔧 Alembic script template not found. Creating default at {ALEMBIC_SCRIPT_MAKO_PATH}")
        mako_content = """\"\"\"${message}\"\"\"
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}
def upgrade() -> None:
    ${upgrades if upgrades else "pass"}
def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
"""
        try:
            with open(ALEMBIC_SCRIPT_MAKO_PATH, 'w') as f:
                f.write(mako_content)
            logger.success(f"✅ Default script.py.mako created.")
        except IOError as e:
            logger.error(f"❌ Failed to write default script.py.mako: {e}"); raise


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

# --- init_db - Modified to start snapshotter ---
def init_db():
    """
    Initializes the Database: Handles decompression, ensures schema is up-to-date
    via Alembic migrations (auto-generating if needed), sets up engine/session
    for the RUNTIME DB, and logs stats.
    """
    logger.info("🚀 Initializing Database (Runtime DB with Snapshot Restore)...")
    global _engine, SessionLocal

    if _engine is not None and SessionLocal is not None and SessionLocal.kw.get('bind'):
        logger.warning("Database already initialized. Skipping re-initialization.")
        return

    engine_instance = None
    migration_successful = False

    try:
        engine_instance = get_engine()
        if not engine_instance:
            raise RuntimeError("get_engine() failed to return a valid engine instance.")
        logger.info("Engine initialization/check complete.")

        _create_default_env_py()
        _create_default_script_mako()
        if not os.path.exists(ALEMBIC_INI_PATH):
            logger.warning(f"🔧 Alembic config file not found. Creating default at {ALEMBIC_INI_PATH}")
            alembic_dir_for_ini = os.path.relpath(ALEMBIC_DIR, APP_DIR).replace("\\", "/")
            alembic_cfg_content = f"""[alembic]
script_location = {alembic_dir_for_ini}
sqlalchemy.url = {RUNTIME_DATABASE_URL}
# ... (rest of your ini content from V17) ...
[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""
            try:
                with open(ALEMBIC_INI_PATH, 'w') as f:
                    f.write(alembic_cfg_content)
                logger.success(f"📝 Default alembic.ini created.")
            except IOError as e:
                logger.error(f"❌ Failed to write default alembic.ini: {e}"); raise
        logger.info("Alembic configuration files ensured.")

        # --- MODIFIED PART FOR AUTOGENERATION ---
        # Always attempt to generate a revision to detect changes,
        # then upgrade. Alembic will only create a new file if changes are detected.
        logger.info("Attempting to auto-generate migration script for any schema changes...")
        autogen_success = False
        autogen_message = f"Autodetect schema changes {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            command_args = [
                sys.executable, "-m", "alembic",
                "-c", ALEMBIC_INI_PATH,
                "revision", "--autogenerate",
                "-m", autogen_message
            ]
            logger.info(f"Running command: {' '.join(command_args)}")
            process = subprocess.run(
                command_args, cwd=APP_DIR,
                capture_output=True, text=True, check=False, timeout=60  # check=False initially
            )

            # Alembic revision --autogenerate returns 0 even if no changes are detected.
            # We need to check its output to see if a new file was generated.
            if process.returncode == 0:
                if "Generating" in process.stdout or "Generated" in process.stdout:  # Check for output indicating a new file
                    logger.success(f"✅ Alembic autogenerated new migration script. Output:\n{process.stdout}")
                    autogen_success = True
                else:
                    logger.info("Alembic autogenerate: No new schema changes detected to create a migration file.")
                    autogen_success = True  # Considered success as no changes means nothing to generate
            else:  # Error in running alembic command itself
                logger.error(f"❌ Failed running 'alembic revision --autogenerate' (RC: {process.returncode}).")
                logger.error(f"   Cmd: {' '.join(command_args)}")
                logger.error(f"   Stdout: {process.stdout}")
                logger.error(f"   Stderr: {process.stderr}")
                logger.critical("   Check imports/metadata in alembic/env.py & model definitions!")
                # Do not proceed to upgrade if autogen command itself failed

            if process.stderr and "Target database is not up to date." in process.stderr:
                logger.warning(
                    "Alembic indicated target database is not up to date during autogen. Upgrade will attempt to fix.")
                # This is not necessarily a failure of autogen itself, but a state Alembic noticed.

        except subprocess.CalledProcessError as cpe:  # Should not be hit if check=False
            logger.error(
                f"❌ Error during 'alembic revision --autogenerate' (CalledProcessError): {cpe.stderr or cpe.stdout}")
        except FileNotFoundError:
            logger.error(f"❌ Failed auto-generate: '{sys.executable} -m alembic' not found.")
        except subprocess.TimeoutExpired:
            logger.error("❌ Auto-generate timed out.")
        except Exception as auto_err:
            logger.error(f"❌ Unexpected error during auto-generation: {auto_err}")
        # --- END MODIFIED PART ---

        migration_successful = check_and_apply_migrations()  # This will apply any newly generated or pending migrations

        if not migration_successful:
            logger.warning("Skipping schema verification as migration step failed or reported errors.")
            # If migrations fail critically, we might want to exit or take other actions.
            # For now, it will proceed and potentially fail later if schema is indeed bad.
        else:
            logger.info("📊 Logging Database Schema Statistics (Verification)...")
            try:
                if not engine_instance: raise RuntimeError("Engine lost before inspection.")
                inspector = sql_inspect(engine_instance)
                table_names = inspector.get_table_names()
                logger.info(f"  Tables ({len(table_names)}): {', '.join(table_names)}")
                expected_tables = {'alembic_version', 'interactions', 'applescript_attempts', 'file_index'}
                found_tables = set(table_names);
                missing_tables = expected_tables - found_tables
                if missing_tables:
                    logger.error(f"‼️ Expected tables MISSING: {', '.join(missing_tables)}")
                    if 'alembic_version' in missing_tables: logger.critical(
                        "   'alembic_version' MISSING - Migrations failed!")
                    migration_successful = False
                total_columns = 0
                if table_names:
                    with engine_instance.connect() as connection:
                        for table_name in table_names:
                            try:
                                columns = inspector.get_columns(table_name)
                                column_names = [c['name'] for c in columns];
                                col_count = len(columns);
                                total_columns += col_count
                                row_count = -1
                                try:
                                    row_count = connection.execute(
                                        text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar_one()
                                except:
                                    pass
                                logger.info(
                                    f"    Table '{table_name}': Rows={row_count}, Cols={col_count} ({', '.join(column_names)})")
                                # <<< ADD COLUMN CHECK FOR INTERACTIONS TABLE HERE >>>
                                if table_name == "interactions":
                                    interaction_cols = {c['name'] for c in columns}
                                    expected_interaction_cols = {"imagined_image_prompt", "imagined_image_b64",
                                                                 "imagined_image_vlm_description"}
                                    missing_interaction_cols = expected_interaction_cols - interaction_cols
                                    if missing_interaction_cols:
                                        logger.error(
                                            f"‼️ Table 'interactions' is MISSING expected columns: {', '.join(missing_interaction_cols)}")
                                        migration_successful = False  # Mark as failure if new columns are missing
                                    else:
                                        logger.info(
                                            "✅ Table 'interactions' contains the new 'imagined_image_*' columns.")
                                # <<< END COLUMN CHECK >>>
                            except Exception as ti_err:
                                logger.error(f"Err inspecting '{table_name}': {ti_err}")
                    logger.info(f"  Total Columns: {total_columns}")
                else:
                    logger.warning("  No tables found. Migrations likely failed.")
                    migration_successful = False
            except Exception as inspect_err:
                logger.error(f"❌ Schema inspection failed: {inspect_err}")
                migration_successful = False

        if migration_successful:
            logger.success("✅ Database initialization sequence completed successfully.")
        else:
            logger.critical("🔥🔥 Database initialization FAILED or schema verification (including new columns) failed.")
            raise RuntimeError("Database initialization failed migration or verification.")

        start_db_snapshotter()
        atexit.register(_shutdown_hook)
        atexit.register(stop_db_snapshotter)

    except Exception as init_err:
        logger.critical(f"🔥🔥 UNEXPECTED FAILURE during database initialization: {init_err}")
        logger.exception("Initialization Traceback:")
        sys.exit("CRITICAL: Database Initialization Failure")


# --- END init_db ---


# --- Database Interaction Functions (add_interaction, get_recent_interactions, etc.) ---
# (Keep your existing DB interaction functions here - unchanged from V16)
def add_interaction(db: Session, **kwargs) -> Optional[Interaction]:
    interaction: Optional[Interaction] = None
    try:
        valid_keys = {c.name for c in Interaction.__table__.columns}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        filtered_kwargs.setdefault('mode', 'chat')
        if 'input_type' not in filtered_kwargs:
            if filtered_kwargs.get('user_input'):
                filtered_kwargs['input_type'] = 'text'
            elif filtered_kwargs.get('llm_response'):
                filtered_kwargs['input_type'] = 'llm_response'
            else:
                filtered_kwargs['input_type'] = 'log_info'
        filtered_kwargs.setdefault('reflection_completed', False)
        filtered_kwargs.setdefault('tot_delivered', False)
        filtered_kwargs.setdefault('tot_analysis_requested', False)
        filtered_kwargs.setdefault('assistant_action_executed', False)
        filtered_kwargs.setdefault('requires_deep_thought', None)
        interaction = Interaction(**filtered_kwargs)
        db.add(interaction);
        db.commit();
        db.refresh(interaction)
        log_level = "INFO";
        log_extra = ""
        if interaction.input_type == 'error' or interaction.input_type == 'log_error':
            log_level = "ERROR"; log_extra = " ❌ ERROR"
        elif interaction.input_type == 'log_warning':
            log_level = "WARNING"; log_extra = " ⚠️ WARN"
        logger.log(log_level,
                   f"💾 Interaction {interaction.id} ({interaction.mode}/{interaction.input_type}){log_extra}")
        return interaction
    except SQLAlchemyError as e:
        logger.error(f"❌ Error saving interaction: {e}")
        if isinstance(e, (OperationalError, ProgrammingError)) and (
                "column" in str(e).lower() or "table" in str(e).lower()):
            logger.critical(f"SCHEMA MISMATCH: {e}. Run migrations!")
        try:
            db.rollback()
        except Exception as rb_err:
            logger.error(f"Rollback failed: {rb_err}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error saving interaction: {e}"); return None


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
        db.execute(stmt);
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
# --- End of database.py ---