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
import argparse # For __main__
import asyncio # For running async init in __main__
import datetime
from chromadb.config import Settings


# Assuming these are accessible or passed in
from database import Interaction, init_db, SessionLocal  # The SQLAlchemy model for interactions
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
    # sys.exit(1) # Commented out for testing
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
                    collection_name=REFLECTION_INDEX_CHROMA_COLLECTION_NAME,
                    persist_directory=REFLECTION_INDEX_CHROMA_PERSIST_DIR,
                    embedding_function=provider.embeddings,
                    client_settings=Settings(anonymized_telemetry=False)
                )
                logger.success("ReflectionIndexer Init: Successfully loaded persisted Global Reflection vector store.")
            else:
                logger.info(
                    f"ReflectionIndexer Init: No persisted Reflection Chroma DB at {REFLECTION_INDEX_CHROMA_PERSIST_DIR}. Creating new.")
                global_reflection_vectorstore = Chroma(
                    collection_name=REFLECTION_INDEX_CHROMA_COLLECTION_NAME,
                    embedding_function=provider.embeddings,
                    persist_directory=REFLECTION_INDEX_CHROMA_PERSIST_DIR,
                    client_settings=Settings(anonymized_telemetry=False)
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
    log_prefix = f"ReflIndex|ID:{getattr(reflection_interaction, 'id', 'Unknown')}|ELP{priority}"

    if not _reflection_vs_initialized_event.is_set():  # type: ignore
        logger.warning(f"{log_prefix}: Reflection VS not initialized (_event not set). Skipping indexing.")
        return
    if global_reflection_vectorstore is None:
        logger.error(
            f"{log_prefix}: CRITICAL - Reflection VS event IS SET, but global_reflection_vectorstore object IS None. Skipping.")
        return
    if not provider or not provider.embeddings:
        logger.error(f"{log_prefix}: AIProvider or its embeddings are not available. Skipping indexing.")
        return

    llm_response_content = getattr(reflection_interaction, 'llm_response', None)
    interaction_input_type = getattr(reflection_interaction, 'input_type', None)

    if not llm_response_content or interaction_input_type != 'reflection_result':
        logger.debug(f"{log_prefix}: Not a reflection result or no content. Skipping.")
        return
    if not isinstance(llm_response_content, str):
        logger.warning(f"{log_prefix}: llm_response is not a string (type: {type(llm_response_content)}). Skipping.")
        return

    # YOUR_REFLECTION_CHUNK_SIZE and YOUR_REFLECTION_CHUNK_OVERLAP should be imported from config
    # Define fallbacks if not found, or ensure they are always imported.
    _chunk_size = globals().get("YOUR_REFLECTION_CHUNK_SIZE", 500)
    _chunk_overlap = globals().get("YOUR_REFLECTION_CHUNK_OVERLAP", 50)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size, chunk_overlap=_chunk_overlap,
        length_function=len, is_separator_regex=False,
    )
    chunks = text_splitter.split_text(llm_response_content)
    if not chunks:
        logger.warning(f"{log_prefix}: No text chunks from llm_response. Skipping.")
        return

    logger.info(f"{log_prefix}: Starting indexing process for {len(chunks)} chunks...")
    try:
        logger.debug(f"{log_prefix}: Embedding {len(chunks)} chunks with priority ELP{priority}...")

        # --- MODIFIED CALL: Use the internal _embed_texts for priority ---
        if not hasattr(provider.embeddings, '_embed_texts') or \
                not callable(getattr(provider.embeddings, '_embed_texts')):
            logger.error(
                f"{log_prefix}: Embeddings object is missing '_embed_texts' method. Cannot embed with priority. Skipping.")
            return

        chunk_embeddings: List[List[float]] = provider.embeddings._embed_texts(chunks,
                                                                               priority=priority)  # type: ignore
        # --- END MODIFIED CALL ---

        if not chunk_embeddings or len(chunk_embeddings) != len(chunks):
            logger.error(
                f"{log_prefix}: Embedding failed or returned mismatched vectors. Expected {len(chunks)}, Got {len(chunk_embeddings) if chunk_embeddings else 0}.")
            return

        metadatas: List[Dict[str, Any]] = [{
            "source_interaction_id": getattr(reflection_interaction, 'id', -1),
            "timestamp": str(getattr(reflection_interaction, 'timestamp', "N/A")),
            "original_user_input_snippet": (getattr(reflection_interaction, 'user_input', None) or "")[:100]
        } for _ in range(len(chunks))]
        ids: List[str] = [f"reflection_{getattr(reflection_interaction, 'id', 'Unknown')}_chunk_{i}" for i in
                          range(len(chunks))]

        with _reflection_vs_write_lock:  # type: ignore
            if global_reflection_vectorstore is None:
                logger.error(
                    f"{log_prefix}: CRITICAL - global_reflection_vectorstore became None before write. Aborting.")
                return

            logger.debug(f"{log_prefix}: Acquired write lock. Adding {len(chunks)} items to Chroma reflection store.")
            task_message_suffix = ""
            try:
                # Using the standard add_embeddings method with corrected keyword arguments
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

            _persist_dir = globals().get("REFLECTION_INDEX_CHROMA_PERSIST_DIR")  # Get from globals if imported
            if _persist_dir and hasattr(global_reflection_vectorstore, 'persist') and \
                    callable(getattr(global_reflection_vectorstore, 'persist')) and \
                    getattr(global_reflection_vectorstore, '_persist_directory', None) == _persist_dir:
                logger.debug(
                    f"{log_prefix}: Persisting reflection store to {_persist_dir} after adding ID {reflection_interaction.id}")
                global_reflection_vectorstore.persist()  # type: ignore
            else:
                logger.trace(
                    f"{log_prefix}: Reflection store not configured for persistence to '{_persist_dir}'. Skipping persist().")

        logger.success(f"{log_prefix}: Successfully indexed {len(chunks)} chunks. {task_message_suffix}")

        if hasattr(reflection_interaction, 'reflection_indexed_in_vs'):
            try:
                if not attributes.instance_state(reflection_interaction).session:  # type: ignore
                    logger.warning(f"{log_prefix}: Interaction ID {reflection_interaction.id} detached. Merging.")
                    reflection_interaction = db_session.merge(reflection_interaction)  # type: ignore
                setattr(reflection_interaction, 'reflection_indexed_in_vs', True)
                setattr(reflection_interaction, 'last_modified_db',
                        datetime.datetime.now(datetime.timezone.utc)) # Use datetime directly
                db_session.commit()
                logger.info(f"{log_prefix}: Marked reflection ID {reflection_interaction.id} as indexed in SQLite.")
            except Exception as e_db_update:
                logger.error(
                    f"{log_prefix}: Failed to mark reflection ID {reflection_interaction.id} as indexed: {e_db_update}")
                db_session.rollback()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reflection Indexer Test CLI")
    parser.add_argument("--test", action="store_true", help="Run test initialization and optional search.")
    parser.add_argument("--search_query", type=str, default=None, help="Query string to search the reflection vector store.")
    cli_args = parser.parse_args()

    if cli_args.test:
        logger.remove() # Remove default loguru handler
        logger.add(sys.stderr, level="INFO") # Add a new one with INFO level for testing
        logger.info("--- Reflection Indexer Test Mode ---")

        # 1. Initialize Database (critical for SessionLocal and schema)
        db_session_for_test: Optional[Session] = None
        try:
            logger.info("Initializing database via database.init_db()...")
            init_db() # This function from database.py handles Alembic migrations etc.
            db_session_for_test = SessionLocal() # type: ignore
            if db_session_for_test is None:
                raise Exception("Failed to create a database session after init_db.")
            logger.info("Database initialization and session creation complete.")
        except Exception as e_db_init:
            logger.error(f"Failed to initialize database or create session: {e_db_init}")
            sys.exit(1)

        # 2. Initialize AIProvider (needed for embeddings)
        test_ai_provider: Optional[AIProvider] = None
        try:
            logger.info(f"Initializing AIProvider (Provider: {PROVIDER})...") # PROVIDER from config.py
            test_ai_provider = AIProvider(PROVIDER)
            if not test_ai_provider.embeddings:
                raise ValueError("AIProvider initialized, but embeddings are not available.")
            logger.info("AIProvider initialized successfully for embeddings.")
        except Exception as e_ai_provider:
            logger.error(f"Failed to initialize AIProvider: {e_ai_provider}")
            if db_session_for_test: db_session_for_test.close()
            sys.exit(1)

        # 3. Initialize Global Reflection Vector Store
        try:
            logger.info("Initializing global reflection vector store...")
            # initialize_global_reflection_vectorstore is synchronous
            initialize_global_reflection_vectorstore(test_ai_provider, db_session_for_test)
            logger.info("Global reflection vector store initialization process completed.")
        except Exception as e_vs_init:
            logger.error(f"Error during reflection vector store initialization: {e_vs_init}")
            if db_session_for_test: db_session_for_test.close()
            sys.exit(1)

        # 4. Get the vector store instance
        reflection_vs = get_global_reflection_vectorstore()
        if reflection_vs:
            logger.success("Successfully retrieved global reflection vector store instance.")
            try:
                collection_count = reflection_vs._collection.count() # type: ignore
                logger.info(f"Reflection Chroma collection contains {collection_count} items.")
            except Exception as e_count:
                logger.warning(f"Could not get count from Chroma collection: {e_count}")


            # Example: Add a dummy reflection to test indexing (optional)
            if collection_count == 0: # Only add if empty for testing
                logger.info("Adding a dummy reflection for testing indexing...")
                dummy_reflection_text = "This is a test reflection about AI metacognition and learning from past interactions. It explores the concept of self-improvement in artificial intelligence systems."
                dummy_interaction = Interaction(
                    session_id="test_session_reflection",
                    mode="chat",
                    input_type="reflection_result", # Critical: must be this type
                    user_input="[Test Reflection Trigger]",
                    llm_response=dummy_reflection_text,
                    timestamp=datetime.datetime.now(datetime.timezone.utc)
                )
                if db_session_for_test:
                    try:
                        db_session_for_test.add(dummy_interaction)
                        db_session_for_test.commit()
                        db_session_for_test.refresh(dummy_interaction) # Get ID
                        logger.info(f"Added dummy reflection with ID: {dummy_interaction.id}")
                        index_single_reflection(dummy_interaction, test_ai_provider, db_session_for_test)
                        logger.info("Attempted to index dummy reflection.")
                        # Re-check count
                        collection_count_after_add = reflection_vs._collection.count() # type: ignore
                        logger.info(f"Reflection Chroma collection NOW contains {collection_count_after_add} items.")
                    except Exception as e_dummy:
                        logger.error(f"Error adding/indexing dummy reflection: {e_dummy}")
                        if db_session_for_test: db_session_for_test.rollback()
                else:
                    logger.error("Cannot add dummy reflection: db_session_for_test is None.")


            # 5. Perform Search if query provided
            if cli_args.search_query:
                query = cli_args.search_query
                logger.info(f"Performing search with query: '{query}'")
                try:
                    search_results = reflection_vs.similarity_search_with_relevance_scores(query, k=3)
                    if search_results:
                        logger.info(f"Found {len(search_results)} results:")
                        for i, (doc, score) in enumerate(search_results):
                            print(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
                            print(f"  Metadata: {doc.metadata}")
                            print(f"  Content Snippet:\n    {doc.page_content[:300].replace(os.linesep, ' ')}...")
                    else:
                        logger.info("No results found for the query.")
                except Exception as e_search:
                    logger.error(f"Error during vector search: {e_search}")
            else:
                logger.info("No search query provided. Skipping search.")
        else:
            logger.error("Failed to get global reflection vector store instance after initialization attempt.")

        if db_session_for_test:
            db_session_for_test.close()
        logger.info("--- Reflection Indexer Test Mode Finished ---")
    else:
        print("Run with --test to initialize and optionally --search_query \"your query\" to search.")
