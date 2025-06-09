# interaction_indexer.py
import os
import time
import threading
from typing import Optional, List, Any
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy import update
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Assuming these are accessible
from database import Interaction, SessionLocal
from ai_provider import AIProvider
from config import *

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
            global_interaction_vectorstore = Chroma(
                collection_name=INTERACTION_COLLECTION_NAME,
                persist_directory=INTERACTION_VS_PERSIST_DIR,
                embedding_function=provider.embeddings
            )
            logger.success("✅ InteractionIndexer: Global interaction vector store initialized.")
            _interaction_vs_initialized_event.set()
        except Exception as e:
            logger.error(f"❌ InteractionIndexer: Failed to initialize vector store: {e}")
            global_interaction_vectorstore = None


def get_global_interaction_vectorstore() -> Optional[Chroma]:
    _interaction_vs_initialized_event.wait(timeout=60)
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
        if not _interaction_vs_initialized_event.is_set():
            logger.warning("InteractionIndexer created, but the global vector store is not yet initialized.")

    def run(self):
        logger.info("🚀 Starting Interaction Indexer thread...")
        while not self.stop_event.is_set():
            try:
                # Wait for the vector store to be ready
                if not _interaction_vs_initialized_event.wait(timeout=10):
                    logger.warning("InteractionIndexer: Timed out waiting for VS initialization. Retrying next cycle.")
                    continue

                db_session = SessionLocal()
                if not db_session:
                    logger.error("InteractionIndexer: Could not create DB session.")
                    continue

                try:
                    self._index_new_interactions(db_session)
                finally:
                    db_session.close()

            except Exception as e:
                logger.error(f"Error in Interaction Indexer loop: {e}")

            # Wait for 5 minutes before the next cycle
            self.stop_event.wait(5 * 60)
        logger.info("🛑 Interaction Indexer thread has been shut down.")

    def _index_new_interactions(self, db_session: Session):
        if global_interaction_vectorstore is None:
            logger.error("InteractionIndexer: Global interaction vector store is None. Cannot index.")
            return

        unindexed_interactions = db_session.query(Interaction).filter(
            Interaction.is_indexed_for_rag == False
        ).all()

        if not unindexed_interactions:
            logger.info("InteractionIndexer: No new interactions to index.")
            return

        logger.info(f"InteractionIndexer: Found {len(unindexed_interactions)} new interactions to index.")
        texts_to_embed = []
        metadatas = []
        ids_to_update = []

        for interaction in unindexed_interactions:
            # Combine user input and AI response for a complete conversational turn context
            content = f"User: {interaction.user_input or ''}\n\nAssistant: {interaction.llm_response or ''}"
            chunks = self.text_splitter.split_text(content)

            for i, chunk in enumerate(chunks):
                texts_to_embed.append(chunk)
                metadatas.append({
                    "interaction_id": interaction.id,
                    "session_id": interaction.session_id,
                    "timestamp": str(interaction.timestamp),
                    "input_type": interaction.input_type,
                    "chunk_index": i
                })
            ids_to_update.append(interaction.id)

        if texts_to_embed:
            with _interaction_vs_write_lock:
                logger.info(f"InteractionIndexer: Adding {len(texts_to_embed)} new text chunks to the vector store.")
                global_interaction_vectorstore.add_texts(texts=texts_to_embed, metadatas=metadatas)

            # Update the database records to mark them as indexed
            db_session.execute(
                update(Interaction)
                .where(Interaction.id.in_(ids_to_update))
                .values(is_indexed_for_rag=True)
            )
            db_session.commit()
            logger.success(f"InteractionIndexer: Successfully indexed {len(unindexed_interactions)} interactions.")