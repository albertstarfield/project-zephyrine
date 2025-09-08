#interaction_indexer.py

import os
import time
import threading
import json
from typing import Optional, List
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy import update
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil
from database import Interaction, SessionLocal
from cortex_backbone_provider import CortexEngine
from CortexConfiguration import *
import chromadb
from chromadb.config import Settings

# --- Globals for this module ---
global_interaction_vectorstore: Optional[Chroma] = None
_interaction_vs_init_lock = threading.Lock()
_interaction_vs_write_lock = threading.Lock()
_interaction_vs_initialized_event = threading.Event()

# --- Chroma DB Path ---
INTERACTION_VS_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_interaction_store")
INTERACTION_COLLECTION_NAME = "global_interaction_history"


def initialize_global_interaction_vectorstore(provider: CortexEngine):
    global global_interaction_vectorstore
    logger.info(">>> InteractionIndexer: Initializing global interaction vector store...")

    # Define consistent client settings to prevent the error in the first place.
    chroma_settings = Settings(anonymized_telemetry=False)

    # Allow one retry attempt if we need to clear the old DB
    for attempt in range(2):
        try:
            with _interaction_vs_init_lock:
                if _interaction_vs_initialized_event.is_set():
                    return

                os.makedirs(INTERACTION_VS_PERSIST_DIR, exist_ok=True)

                if os.path.exists(os.path.join(INTERACTION_VS_PERSIST_DIR, "chroma.sqlite3")):
                    logger.info(f"Loading existing persisted Interaction Chroma DB from: {INTERACTION_VS_PERSIST_DIR}")
                    global_interaction_vectorstore = Chroma(
                        collection_name=INTERACTION_COLLECTION_NAME,
                        persist_directory=INTERACTION_VS_PERSIST_DIR,
                        embedding_function=provider.embeddings,
                        client_settings=chroma_settings  # Consistent settings
                    )
                else:
                    logger.info(f"No persisted Interaction Chroma DB found. Rebuilding from main database...")
                    db_session = SessionLocal()
                    try:
                        # First, create the empty, persistent Chroma store
                        global_interaction_vectorstore = Chroma(
                            collection_name=INTERACTION_COLLECTION_NAME,
                            persist_directory=INTERACTION_VS_PERSIST_DIR,
                            embedding_function=provider.embeddings,
                            client_settings=chroma_settings # Consistent settings
                        )

                        interactions_with_backup = db_session.query(Interaction).filter(
                            Interaction.embedding_json.isnot(None)).all()

                        if interactions_with_backup:
                            logger.info(
                                f"Found {len(interactions_with_backup)} interactions with embedding backups to rebuild.")
                            texts = [f"User: {i.user_input or ''}\nAI: {i.llm_response or ''}" for i in
                                     interactions_with_backup]
                            embeddings = [json.loads(i.embedding_json) for i in interactions_with_backup]
                            metadatas = [{"interaction_id": i.id, "session_id": i.session_id, "timestamp": str(i.timestamp)}
                                         for i in interactions_with_backup]
                            ids = [f"int_{i.id}_chunk_0" for i in
                                   interactions_with_backup]

                            global_interaction_vectorstore._collection.add(
                                embeddings=embeddings,
                                documents=texts,
                                metadatas=metadatas,
                                ids=ids
                            )
                            logger.success(
                                f"Rebuilt and persisted Interaction Vector Store with {len(interactions_with_backup)} records.")
                        else:
                            logger.info("No interactions with backups found. Created new empty vector store.")

                    finally:
                        db_session.close()

                logger.success("✅ InteractionIndexer: Global interaction vector store initialized.")
                _interaction_vs_initialized_event.set()
                break  # Exit the loop on success

        except ValueError as e:
            if "An instance of Chroma already exists" in str(e) and attempt == 0:
                logger.warning(f"Chroma conflict detected: {e}. Deleting directory and retrying...")
                with _interaction_vs_init_lock: # Ensure exclusive access for deletion
                    if os.path.isdir(INTERACTION_VS_PERSIST_DIR):
                        try:
                            shutil.rmtree(INTERACTION_VS_PERSIST_DIR)
                            logger.info(f"Successfully deleted directory: {INTERACTION_VS_PERSIST_DIR}")
                        except Exception as rm_e:
                            logger.error(f"❌ Failed to delete Chroma directory {INTERACTION_VS_PERSIST_DIR}: {rm_e}")
                            raise rm_e # Re-raise if deletion fails
            else:
                logger.error(f"❌ InteractionIndexer: Failed to initialize vector store: {e}")
                global_interaction_vectorstore = None
                raise e # Re-raise the exception if it's not the specific conflict or on the final attempt
        except Exception as e:
            logger.error(f"❌ InteractionIndexer: Failed to initialize vector store: {e}")
            global_interaction_vectorstore = None
            raise  # Re-raise other critical errors


def get_global_interaction_vectorstore() -> Optional[Chroma]:
    if not _interaction_vs_initialized_event.wait(timeout=60):
        logger.error("Timeout waiting for interaction vector store initialization.")
        return None
    return global_interaction_vectorstore


class InteractionIndexer(threading.Thread):
    def __init__(self, stop_event: threading.Event, provider: CortexEngine):
        super().__init__(name="InteractionIndexerThread", daemon=True)
        self.stop_event = stop_event
        self.provider = provider
        self.embedding_model = provider.embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200,
            length_function=len,
        )

    def run(self):
        logger.info("🚀 Starting Interaction Indexer thread...")
        while not self.stop_event.is_set():
            try:
                if not _interaction_vs_initialized_event.wait(timeout=10):
                    logger.warning("InteractionIndexer: Timed out waiting for VS initialization. Retrying next cycle.")
                    continue

                db_session = SessionLocal()
                try:
                    self._index_new_interactions(db_session)
                finally:
                    db_session.close()
            except Exception as e:
                logger.error(f"Error in Interaction Indexer loop: {e}")

            self.stop_event.wait(300)  # Wait 5 minutes
        logger.info("🛑 Interaction Indexer thread has been shut down.")

    def _index_new_interactions(self, db_session: Session):
        if global_interaction_vectorstore is None or global_interaction_vectorstore._collection is None:
            logger.error("InteractionIndexer: Vector store or its collection is not available. Cannot index.")
            return

        unindexed_interactions = db_session.query(Interaction).filter(
            Interaction.is_indexed_for_rag == False
        ).limit(100).all()

        if not unindexed_interactions:
            return

        logger.info(f"InteractionIndexer: Found {len(unindexed_interactions)} new interactions to index.")

        for interaction in unindexed_interactions:
            try:
                content_parts = []
                interaction_type = interaction.input_type or "unknown"

                # Handle standard conversational turns
                if interaction_type in ['text', 'llm_response', 'image+text']:
                    if interaction.user_input:
                        content_parts.append(f"User: {interaction.user_input}")
                    if interaction.llm_response:
                        content_parts.append(f"Assistant: {interaction.llm_response}")
                    if interaction.emotion_context_analysis:
                        content_parts.append(f"Emotion Analysis: {interaction.emotion_context_analysis}")
                    if interaction.assistant_action_type:
                        action_params = interaction.assistant_action_params or "{}"
                        action_result = interaction.assistant_action_result or "No result."
                        content_parts.append(f"Action Taken: {interaction.assistant_action_type} with params {action_params} -> Result: {action_result}")
                # Handle log_info* and other log types
                elif interaction_type.startswith('log_'):
                    log_message = interaction.user_input or interaction.llm_response or "[Empty Log]"
                    content_parts.append(f"System Log ({interaction_type}): {log_message}")

                content = "\n\n".join(content_parts)
                if not content.strip():
                    interaction.is_indexed_for_rag = True
                    continue

                main_embedding = self.embedding_model.embed_query(content)
                interaction.embedding_json = json.dumps(main_embedding)

                chunks = self.text_splitter.split_text(content)
                if not chunks:
                    interaction.is_indexed_for_rag = True
                    continue

                chunk_embeddings = self.embedding_model.embed_documents(chunks)

                metadatas = [{
                    "interaction_id": interaction.id, "session_id": interaction.session_id,
                    "timestamp": str(interaction.timestamp),
                    "input_type": interaction.input_type or "unknown"
                } for _ in chunks]
                ids = [f"int_{interaction.id}_chunk_{i}" for i in range(len(chunks))]

                with _interaction_vs_write_lock:
                    global_interaction_vectorstore._collection.add(
                        embeddings=chunk_embeddings,
                        documents=chunks,
                        metadatas=metadatas,
                        ids=ids
                    )

                interaction.is_indexed_for_rag = True

            except Exception as e:
                logger.error(f"Failed to process interaction {interaction.id}: {e}")
                interaction.is_indexed_for_rag = True

        db_session.commit()
        logger.success(
            f"InteractionIndexer: Successfully processed a batch of {len(unindexed_interactions)} interactions.")