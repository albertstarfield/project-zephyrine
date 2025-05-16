# reflection_indexer.py
import os
import json
import threading
import time  # For potential periodic tasks
from typing import Optional, List, Dict, Any
from loguru import logger
import sys
from sqlalchemy.orm import Session
from sqlalchemy.orm import attributes
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Or your preferred splitter

# Assuming these are accessible or passed in
from database import Interaction  # The SQLAlchemy model for interactions
from ai_provider import AIProvider  # To get the embedding function
from config import *

# --- NEW: Import the custom lock ---

try:
    from priority_lock import PriorityQuotaLock, ELP0, ELP1
    logger.info("✅ Successfully imported PriorityQuotaLock, ELP0, ELP1.")
except ImportError as e:
    logger.error(f"❌ Failed to import from priority_lock.py: {e}")
    logger.warning("    Falling back to standard threading.Lock for priority lock (NO PRIORITY/QUOTA).")
    # Define fallbacks so the rest of the code doesn't crash immediately
    import threading
    PriorityQuotaLock = threading.Lock # type: ignore
    ELP0 = 0
    ELP1 = 1
    # You might want to sys.exit(1) here if priority locking is critical
    sys.exit(1)
interruption_error_marker = "Worker task interrupted by higher priority request" # Define consistently

# --- Globals for this module ---
global_reflection_vectorstore: Optional[Chroma] = None
_reflection_vs_init_lock = threading.Lock()
_reflection_vs_write_lock = threading.Lock()  # For thread-safe writes if multiple sources could update
_reflection_vs_initialized_event = threading.Event()

# --- Constants for this module ---
# Example: Could be in config.py if preferred
REFLECTION_VS_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_reflection_store")


def initialize_global_reflection_vectorstore(provider: AIProvider,
                                             db_session: Session):  # db_session might not be needed if loading persisted
    global global_reflection_vectorstore
    logger.info(">>> ReflectionIndexer: Entered initialize_global_reflection_vectorstore <<<")

    if not provider or not provider.embeddings:
        logger.error("ReflectionIndexer Init: AIProvider or embeddings missing. Cannot initialize.")
        return

    if _reflection_vs_initialized_event.is_set():
        logger.info("ReflectionIndexer Init: Skipping, event already set.")
        return

    with _reflection_vs_init_lock:
        if _reflection_vs_initialized_event.is_set():
            logger.info("ReflectionIndexer Init: Skipping (double check), event set while waiting for lock.")
            return
        try:
            # Use the imported constants directly from config.py
            if REFLECTION_INDEX_CHROMA_PERSIST_DIR:  # Check if a path is configured
                os.makedirs(REFLECTION_INDEX_CHROMA_PERSIST_DIR, exist_ok=True)  # Ensure dir exists

            contains_chroma_files = False
            if REFLECTION_INDEX_CHROMA_PERSIST_DIR and os.path.exists(REFLECTION_INDEX_CHROMA_PERSIST_DIR):
                try:
                    if any(fname.endswith(('.sqlite3', '.duckdb', '.parquet')) for fname in
                           os.listdir(REFLECTION_INDEX_CHROMA_PERSIST_DIR)):
                        contains_chroma_files = True
                except OSError as e_scan:
                    logger.warning(f"Could not scan persist directory {REFLECTION_INDEX_CHROMA_PERSIST_DIR}: {e_scan}")

            if contains_chroma_files:
                logger.info(
                    f"ReflectionIndexer Init: Loading existing persisted Reflection Chroma DB from: {REFLECTION_INDEX_CHROMA_PERSIST_DIR}")
                global_reflection_vectorstore = Chroma(
                    collection_name=REFLECTION_INDEX_CHROMA_COLLECTION_NAME,  # Use imported constant
                    persist_directory=REFLECTION_INDEX_CHROMA_PERSIST_DIR,  # Use imported constant
                    embedding_function=provider.embeddings
                )
                logger.success("ReflectionIndexer Init: Successfully loaded persisted Global Reflection vector store.")
            else:
                logger.info(
                    f"ReflectionIndexer Init: No persisted Reflection Chroma DB at {REFLECTION_INDEX_CHROMA_PERSIST_DIR}. Creating new.")
                global_reflection_vectorstore = Chroma(
                    collection_name=REFLECTION_INDEX_CHROMA_COLLECTION_NAME,  # Use imported constant
                    embedding_function=provider.embeddings,
                    persist_directory=REFLECTION_INDEX_CHROMA_PERSIST_DIR  # Use imported constant
                )
                if REFLECTION_INDEX_CHROMA_PERSIST_DIR and hasattr(global_reflection_vectorstore, 'persist'):
                    logger.info(
                        f"ReflectionIndexer Init: Persisting newly created empty structure to {REFLECTION_INDEX_CHROMA_PERSIST_DIR}")
                    global_reflection_vectorstore.persist()  # Persist the empty collection structure if directory is set
                logger.success(
                    "ReflectionIndexer Init: New empty persistent Global Reflection vector store initialized.")

            _reflection_vs_initialized_event.set()
            logger.success("ReflectionIndexer Init: Global Reflection vector store ready. Event SET.")
        except Exception as e:
            logger.error(f"ReflectionIndexer Init: Failed to initialize/load global Reflection vector store: {e}")
            logger.exception("Reflection VS Init Traceback:")
            global_reflection_vectorstore = None  # Ensure it's None on failure
            # Do NOT set event on critical failure here, so app knows it's not ready.
    logger.info(">>> ReflectionIndexer: Exited initialize_global_reflection_vectorstore <<<")


def index_single_reflection(
        reflection_interaction: Interaction,
        provider: AIProvider,
        db_session: Session,
        priority: int = ELP0
):
    global global_reflection_vectorstore
    log_prefix = f"ReflIndex|ID:{reflection_interaction.id}|ELP{priority}"

    if not _reflection_vs_initialized_event.is_set():
        logger.warning(f"{log_prefix}: Reflection VS not initialized (_event not set). Skipping indexing.")
        return
    if global_reflection_vectorstore is None:
        logger.error(
            f"{log_prefix}: CRITICAL - Reflection VS event IS SET, but global_reflection_vectorstore object IS None. Skipping.")
        return
    if not provider or not provider.embeddings:
        logger.error(f"{log_prefix}: AIProvider or embeddings unavailable. Skipping.")
        return
    if not reflection_interaction.llm_response or reflection_interaction.input_type != 'reflection_result':
        logger.debug(f"{log_prefix}: Not a reflection result or no content. Skipping.")
        return

    llm_response_content = reflection_interaction.llm_response
    if not isinstance(llm_response_content, str):
        logger.warning(f"{log_prefix}: llm_response not a string (type: {type(llm_response_content)}). Skipping.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=YOUR_REFLECTION_CHUNK_SIZE,
        chunk_overlap=YOUR_REFLECTION_CHUNK_OVERLAP,
        length_function=len, is_separator_regex=False,
    )
    chunks = text_splitter.split_text(llm_response_content)
    if not chunks:
        logger.warning(f"{log_prefix}: No text chunks from llm_response. Skipping.")
        return

    logger.info(f"{log_prefix}: Starting indexing process for {len(chunks)} chunks...")
    try:
        chunk_embeddings: List[List[float]] = provider.embeddings.embed_documents(chunks,
                                                                                  priority=priority)  # type: ignore
        if not chunk_embeddings or len(chunk_embeddings) != len(chunks):
            logger.error(
                f"{log_prefix}: Embedding failed or mismatched vectors. Expected {len(chunks)}, Got {len(chunk_embeddings) if chunk_embeddings else 0}.")
            return

        metadatas: List[Dict[str, Any]] = [
            {"source_interaction_id": reflection_interaction.id,
             "timestamp": str(reflection_interaction.timestamp),
             "original_user_input_snippet": (getattr(reflection_interaction, 'user_input', None) or "")[:100]}
            for _ in range(len(chunks))
        ]
        ids: List[str] = [f"reflection_{reflection_interaction.id}_chunk_{i}" for i in range(len(chunks))]

        with _reflection_vs_write_lock:
            if global_reflection_vectorstore is None:  # Re-check after lock
                logger.error(
                    f"{log_prefix}: CRITICAL - global_reflection_vectorstore is None within write lock. Aborting add.")
                return

            logger.debug(f"{log_prefix}: Acquired write lock. Adding {len(chunks)} items to Chroma reflection store.")
            task_message_suffix = ""
            try:
                logger.debug(
                    f"{log_prefix}: Attempting global_reflection_vectorstore.add_embeddings(texts=..., embeddings=...)")
                global_reflection_vectorstore.add_embeddings(
                    texts=chunks, embeddings=chunk_embeddings, metadatas=metadatas, ids=ids
                )
                task_message_suffix = "(used add_embeddings with pre-computed vectors)"
                logger.info(f"{log_prefix}: Data added via add_embeddings. {task_message_suffix}")
            except (AttributeError, TypeError) as e_add_embed:
                logger.warning(
                    f"{log_prefix}: global_reflection_vectorstore.add_embeddings failed ({type(e_add_embed).__name__}: {e_add_embed}).")
                logger.info(
                    f"Methods on store ({type(global_reflection_vectorstore)}): {dir(global_reflection_vectorstore)}")
                logger.warning(f"{log_prefix}: Falling back to add_texts (WILL RE-EMBED).")
                global_reflection_vectorstore.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
                task_message_suffix = "(fallback: add_texts with RE-EMBEDDING)"
                logger.info(f"{log_prefix}: Fallback add_texts (re-embedding) completed. {task_message_suffix}")

            # Use the imported constant for persistence check
            if REFLECTION_INDEX_CHROMA_PERSIST_DIR and \
                    hasattr(global_reflection_vectorstore, 'persist') and \
                    callable(getattr(global_reflection_vectorstore, 'persist')) and \
                    getattr(global_reflection_vectorstore, '_persist_directory',
                            None) == REFLECTION_INDEX_CHROMA_PERSIST_DIR:
                logger.debug(
                    f"{log_prefix}: Persisting reflection store to {REFLECTION_INDEX_CHROMA_PERSIST_DIR} after adding ID {reflection_interaction.id}")
                global_reflection_vectorstore.persist()  # type: ignore
            else:
                logger.trace(
                    f"{log_prefix}: Reflection store not configured for persistence to '{REFLECTION_INDEX_CHROMA_PERSIST_DIR}' or persist method unavailable. Skipping persist().")

        logger.success(f"{log_prefix}: Successfully indexed {len(chunks)} chunks. {task_message_suffix}")

        if hasattr(reflection_interaction, 'reflection_indexed_in_vs'):
            try:
                if not attributes.instance_state(reflection_interaction).session_id:  # Check if detached
                    reflection_interaction = db_session.merge(reflection_interaction)  # type: ignore
                setattr(reflection_interaction, 'reflection_indexed_in_vs', True)  # Use setattr for safety
                setattr(reflection_interaction, 'last_modified_db', time.strftime("%Y-%m-%d %H:%M:%S"))
                db_session.commit()
                logger.info(f"{log_prefix}: Marked reflection ID {reflection_interaction.id} as indexed in SQLite.")
            except Exception as e_db_update:
                logger.error(
                    f"{log_prefix}: Failed to mark reflection ID {reflection_interaction.id} as indexed in SQLite: {e_db_update}")
                db_session.rollback()
        else:
            logger.warning(f"{log_prefix}: Interaction model no 'reflection_indexed_in_vs' attr.")

    except TaskInterruptedException as tie:
        logger.warning(f"🚦 {log_prefix}: Embedding for reflection INTERRUPTED: {tie}")
    except Exception as e_index:
        logger.error(f"{log_prefix}: Failed to index chunks: {e_index}")
        logger.exception(f"{log_prefix} Reflection Indexing Traceback:")


def get_global_reflection_vectorstore() -> Optional[Chroma]:
    """Returns the initialized global reflection vector store, or None if not ready."""
    if _reflection_vs_initialized_event.is_set():
        return global_reflection_vectorstore
    else:
        logger.warning("Attempted to get global reflection vector store before it was initialized or init failed.")
        return None

# --- Optional: Periodic task to scan for unindexed reflections (if not done immediately) ---
# def periodic_reflection_indexing_task(stop_event: threading.Event, provider: AIProvider, session_factory): ...

class TaskInterruptedException(Exception):
    """Custom exception raised when an ELP0 task is interrupted."""
    pass