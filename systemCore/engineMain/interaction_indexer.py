# Replace the entire content of interaction_indexer.py with this corrected version:

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

from database import Interaction, SessionLocal
from ai_provider import AIProvider
from config import *
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


def initialize_global_interaction_vectorstore(provider: AIProvider):
    global global_interaction_vectorstore
    logger.info(">>> InteractionIndexer: Initializing global interaction vector store...")
    with _interaction_vs_init_lock:
        if _interaction_vs_initialized_event.is_set():
            return

        try:
            os.makedirs(INTERACTION_VS_PERSIST_DIR, exist_ok=True)

            # Check if a Chroma DB already exists at the path
            if os.path.exists(os.path.join(INTERACTION_VS_PERSIST_DIR, "chroma.sqlite3")):
                logger.info(f"Loading existing persisted Interaction Chroma DB from: {INTERACTION_VS_PERSIST_DIR}")
                global_interaction_vectorstore = Chroma(
                    collection_name=INTERACTION_COLLECTION_NAME,
                    persist_directory=INTERACTION_VS_PERSIST_DIR,
                    embedding_function=provider.embeddings,
                    client_settings=Settings(anonymized_telemetry=False)
                )
            else:
                # If no persistent store, rebuild it from the main SQLite DB backups
                logger.info(f"No persisted Interaction Chroma DB found. Rebuilding from main database...")
                db_session = SessionLocal()
                try:
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
                               interactions_with_backup]  # Assuming one backup embedding per interaction

                        global_interaction_vectorstore = Chroma.from_embeddings(
                            embeddings=embeddings,
                            embedding_function=provider.embeddings,
                            documents=texts,  # Pass original texts for context
                            metadatas=metadatas,
                            ids=ids,
                            collection_name=INTERACTION_COLLECTION_NAME,
                            persist_directory=INTERACTION_VS_PERSIST_DIR
                        )
                        logger.success(
                            f"Rebuilt and persisted Interaction Vector Store with {len(interactions_with_backup)} records.")
                    else:
                        logger.info("No interactions with backups found. Creating new empty vector store.")
                        global_interaction_vectorstore = Chroma(
                            collection_name=INTERACTION_COLLECTION_NAME,
                            persist_directory=INTERACTION_VS_PERSIST_DIR,
                            embedding_function=provider.embeddings
                        )
                finally:
                    db_session.close()

            logger.success("✅ InteractionIndexer: Global interaction vector store initialized.")
            _interaction_vs_initialized_event.set()
        except Exception as e:
            logger.error(f"❌ InteractionIndexer: Failed to initialize vector store: {e}")
            global_interaction_vectorstore = None


def get_global_interaction_vectorstore() -> Optional[Chroma]:
    if not _interaction_vs_initialized_event.wait(timeout=60):
        logger.error("Timeout waiting for interaction vector store initialization.")
        return None
    return global_interaction_vectorstore


class InteractionIndexer(threading.Thread):
    def __init__(self, stop_event: threading.Event, provider: AIProvider):
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
                content = f"User: {interaction.user_input or ''}\n\nAssistant: {interaction.llm_response or ''}"
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

                metadatas = [{"interaction_id": interaction.id, "session_id": interaction.session_id,
                              "timestamp": str(interaction.timestamp)} for _ in chunks]
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