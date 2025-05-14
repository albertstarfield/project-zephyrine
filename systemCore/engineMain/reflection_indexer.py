# reflection_indexer.py
import os
import json
import threading
import time  # For potential periodic tasks
from typing import Optional, List, Dict, Any
from loguru import logger
import sys
from sqlalchemy.orm import Session
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Or your preferred splitter

# Assuming these are accessible or passed in
from database import Interaction  # The SQLAlchemy model for interactions
from ai_provider import AIProvider  # To get the embedding function
from config import YOUR_REFLECTION_CHUNK_SIZE, YOUR_REFLECTION_CHUNK_OVERLAP  # Add these to config.py


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


def initialize_global_reflection_vectorstore(provider: AIProvider, db_session: Session):
    """
    Initializes the global reflection vector store at startup.
    Loads existing indexed reflection chunks from the database OR from a persisted Chroma directory.
    This version assumes we might persist Chroma directly.
    """
    global global_reflection_vectorstore

    if _reflection_vs_initialized_event.is_set():
        logger.info("Global Reflection vector store already initialized. Skipping.")
        return

    with _reflection_vs_init_lock:
        if _reflection_vs_initialized_event.is_set():
            logger.info("Global Reflection vector store was initialized while waiting for lock. Skipping.")
            return

        if not provider or not provider.embeddings:
            logger.error("Cannot init global reflection VS: AIProvider or embeddings missing.")
            return

        logger.info(f"Initializing GLOBAL Reflection vector store (Persist Dir: {REFLECTION_VS_PERSIST_DIR})...")
        try:
            if os.path.exists(REFLECTION_VS_PERSIST_DIR) and any(os.scandir(REFLECTION_VS_PERSIST_DIR)):
                logger.info(f"Loading existing persisted Reflection Chroma DB from: {REFLECTION_VS_PERSIST_DIR}")
                global_reflection_vectorstore = Chroma(
                    persist_directory=REFLECTION_VS_PERSIST_DIR,
                    embedding_function=provider.embeddings
                )
                logger.success("Successfully loaded persisted Global Reflection vector store.")
            else:
                logger.info(
                    f"No existing persisted Reflection Chroma DB found at {REFLECTION_VS_PERSIST_DIR}. Creating new empty store.")
                # Create an empty store that can be added to.
                # We need at least one document to initialize Chroma this way, or use a different init.
                # For an empty start that will be added to later:
                # One way is to initialize with a dummy document and then delete it,
                # or see if Chroma offers an "empty init" that supports persistence.
                # For now, let's assume it's okay if it's empty initially and gets populated by index_single_reflection.
                # If Chroma needs initial data for `persist_directory` to work, this needs adjustment.
                # Let's try initializing it empty but with the persist directory set.
                # This might require Chroma to be created and then `add_texts` (or similar) for the first time.
                # A simpler approach for an empty start that persists:
                # Create it with a dummy, then if you want, you can clear it.
                # For now, let's assume it will be populated by index_single_reflection.
                # If we intend to load ALL past reflections from SQLite DB on startup:
                # This is more like the file_indexer's global VS init.
                # Let's assume for now that new reflections are indexed as they come.
                # So, on startup, we just load what was persisted.
                # If the dir is empty, global_reflection_vectorstore remains None or we make an empty one.

                # To ensure Chroma can be added to later, we create an empty one if dir is empty/new
                logger.info(
                    f"Creating a new (or empty) persistent Reflection Chroma DB at: {REFLECTION_VS_PERSIST_DIR}")
                global_reflection_vectorstore = Chroma(
                    collection_name="reflections_persistent",  # Give it a name
                    embedding_function=provider.embeddings,
                    persist_directory=REFLECTION_VS_PERSIST_DIR
                )
                # global_reflection_vectorstore.persist() # Persist the empty structure
                logger.success("New empty persistent Global Reflection vector store initialized.")

            # --- OPTIONAL: Load un-indexed reflections from SQLite DB at startup ---
            # This would be similar to initialize_global_file_index_vectorstore,
            # querying for Interaction records with input_type='reflection_result'
            # that haven't been marked as "indexed_into_reflection_vs" (new DB field needed).
            # For simplicity, we'll skip this full DB scan on startup for now and assume
            # reflections are indexed as they are created.
            # logger.info("Startup: (Skipping) Check for unindexed past reflections in SQLite DB...")

            _reflection_vs_initialized_event.set()
            logger.success("Global Reflection vector store ready.")

        except Exception as e:
            logger.error(f"Failed to initialize/load global Reflection vector store: {e}")
            logger.exception("Global Reflection VS Init/Load Traceback:")


def index_single_reflection(reflection_interaction: Interaction, provider: AIProvider, db_session: Session,
                            priority: int = ELP0):
    """
    Chunks, embeds, and adds a single reflection_result interaction to the global reflection vector store.
    Marks the interaction in SQLite DB once indexed. (Requires a new field in Interaction model).
    """
    global global_reflection_vectorstore
    if not _reflection_vs_initialized_event.is_set():  # Ensure this event is being set correctly during init
        logger.warning(
            f"Reflection VS not initialized. Cannot index reflection ID {reflection_interaction.id}. Triggering initialization might be needed or check startup.")
        # Optionally, attempt to initialize if not done:
        # if not global_reflection_vectorstore:
        #     initialize_global_reflection_vectorstore(provider, db_session) # This function needs to be sync or handled in thread
        #     if not _reflection_vs_initialized_event.is_set(): # Check again
        #         logger.error(f"Reflection VS still not ready after init attempt for reflection ID {reflection_interaction.id}. Skipping.")
        #         return
        # For now, assuming init happens at startup:
        if not global_reflection_vectorstore:
            logger.error(f"Reflection VS is None though event might be set. Skipping ID {reflection_interaction.id}.")
            return

    if not reflection_interaction.llm_response or reflection_interaction.input_type != 'reflection_result':
        logger.debug(
            f"Skipping indexing for interaction ID {reflection_interaction.id}: Not a reflection result or no content.")
        return

    logger.info(
        f"Indexing reflection ID {reflection_interaction.id} into global reflection vector store (Priority ELP{priority} for embedding)...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=YOUR_REFLECTION_CHUNK_SIZE,
        chunk_overlap=YOUR_REFLECTION_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    # Ensure llm_response is not None before splitting
    if reflection_interaction.llm_response is None:
        logger.warning(f"LLM response is None for reflection ID {reflection_interaction.id}. Skipping.")
        return

    chunks = text_splitter.split_text(reflection_interaction.llm_response)

    if not chunks:
        logger.warning(f"No text chunks generated for reflection ID {reflection_interaction.id}. Skipping.")
        return

    try:
        logger.debug(
            f"Embedding {len(chunks)} chunks for reflection ID {reflection_interaction.id} with priority ELP{priority}...")
        if not provider.embeddings:
            logger.error(f"Embeddings provider not available for reflection ID {reflection_interaction.id}. Skipping.")
            return

        # LlamaCppEmbeddingsWrapper.embed_documents should accept 'priority'
        chunk_embeddings: List[List[float]] = provider.embeddings.embed_documents(chunks,
                                                                                  priority=priority)  # type: ignore

        metadatas: List[Dict[str, Any]] = [{
            "source_interaction_id": reflection_interaction.id,
            "timestamp": str(reflection_interaction.timestamp),
            "original_user_input_snippet": (getattr(reflection_interaction, 'user_input', None) or "")[:100]
        } for _ in range(len(chunks))]

        ids: List[str] = [f"reflection_{reflection_interaction.id}_chunk_{i}" for i in range(len(chunks))]

        with _reflection_vs_write_lock:
            if global_reflection_vectorstore is None:  # Should not happen if init was successful
                logger.error(
                    "CRITICAL: global_reflection_vectorstore is None within write lock during index_single_reflection. Aborting.")
                return

            # --- CORRECTED CALL to add_embeddings ---
            global_reflection_vectorstore.add_embeddings(
                embeddings=chunk_embeddings,  # Changed from text_embeddings
                texts=chunks,  # Changed from documents
                metadatas=metadatas,
                ids=ids
            )
            # --- END CORRECTION ---

            # Persist if the store is configured for persistence
            if hasattr(global_reflection_vectorstore, 'persist') and callable(
                    getattr(global_reflection_vectorstore, 'persist')):
                if hasattr(global_reflection_vectorstore,
                           '_persist_directory') and global_reflection_vectorstore._persist_directory:  # type: ignore
                    logger.debug(
                        f"Persisting global_reflection_vectorstore to {global_reflection_vectorstore._persist_directory}")  # type: ignore
                    global_reflection_vectorstore.persist()
                else:
                    logger.trace("Global reflection vector store not configured for persistence, skipping persist().")
            else:
                logger.trace("Global reflection vector store does not have a persist() method or _persist_directory.")

        logger.success(f"Successfully indexed {len(chunks)} chunks from reflection ID {reflection_interaction.id}")

        # --- Mark as indexed in SQLite DB (Example) ---
        # This requires adding a boolean field like `reflection_indexed_in_vs` to your Interaction model
        # and then:
        # try:
        #     stmt = update(Interaction).where(Interaction.id == reflection_interaction.id).values(reflection_indexed_in_vs=True)
        #     db_session.execute(stmt)
        #     db_session.commit()
        #     logger.info(f"Marked reflection ID {reflection_interaction.id} as indexed in SQLite.")
        # except Exception as e_db_update:
        #     logger.error(f"Failed to mark reflection ID {reflection_interaction.id} as indexed in SQLite: {e_db_update}")
        #     db_session.rollback()

    except TaskInterruptedException as tie:
        logger.warning(f"🚦 Embedding for reflection ID {reflection_interaction.id} INTERRUPTED by ELP1: {tie}")
        # Do not mark as indexed, let it be picked up again or handled
    except Exception as e:
        logger.error(f"Failed to index chunks for reflection ID {reflection_interaction.id}: {e}")
        logger.exception(f"Reflection Indexing Traceback ID {reflection_interaction.id}:")


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