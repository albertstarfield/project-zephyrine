# asyncbg_generate_contextTest.py
import asyncio
import os
import sys
import uuid
import time
from typing import Dict, Any, List

# Import Configurations and Database
import CortexConfiguration
from AdelaideAlbertCortex import CortexThoughts

# Import Cortex components directly to mirror exact production behavior
from cortex_backbone_provider import CortexEngine
from database import SessionLocal, init_db
from interaction_indexer import initialize_global_interaction_vectorstore, get_global_interaction_vectorstore
from loguru import logger
from langchain_core.documents import Document

async def run_asyncbg_context_test():
    print("=====================================================")
    print(" 🔄 Async Background Context Retrieval Test ")
    print("=====================================================")
    
    config_keys_to_dump = [
        "PROVIDER",
        "RAG_HISTORY_COUNT",
        "FUZZY_SEARCH_THRESHOLD_CONTEXT",
        "DATABASE_URL",
    ]
    for key in config_keys_to_dump:
        val = getattr(CortexConfiguration, key, "NOT_FOUND")
        print(f" - {key}: {val}")
    print("=====================================================\n")

    logger.info("Initializing Database...")
    init_db()

    logger.info("Initializing CortexEngine (Provider)...")
    provider = CortexEngine(CortexConfiguration.PROVIDER)

    db = SessionLocal()

    try:
        logger.info("Initializing Global Interaction Vectorstore...")
        await asyncio.to_thread(initialize_global_interaction_vectorstore, provider)

        logger.info("Initializing CortexThoughts (Brain)...")
        cortex_thoughts = CortexThoughts(provider)

        print("\n🧠 Async Background Memory Retrieval Test Primed.")
        print("Type 'exit' or 'quit' to stop.\n")

        test_session_id = "test_asyncbg_debugger_001"
        cortex_thoughts.current_session_id = test_session_id

        while True:
            user_input = input(">_ ")

            if user_input.strip().lower() in ["quit", "exit"]:
                break
            if not user_input.strip():
                continue

            # Emulating background_generate's temporal anchor
            temporal_anchor = cortex_thoughts._get_temporal_context_string()
            user_input_with_context = f"as context This is current time{temporal_anchor}\n\n{user_input}"

            print(f"\n[Searching Memory for: '{user_input}']")

            # Emulating the background_generate context gathering steps
            priority = 1 # ELP1 for external testing / foreground priority

            print("\n[1. Gathering RAG Retriever Context (Hybrid Vector + Fuzzy History)...]")
            wrapped_rag_res = await asyncio.to_thread(
                cortex_thoughts._get_rag_retriever_thread_wrapper,
                db,
                user_input_with_context,
                priority,
            )

            if wrapped_rag_res.get("status") == "success":
                (
                    url_ret_obj,
                    sess_hist_ret_obj,
                    refl_chunk_ret_obj,
                    sess_chat_rag_ids,
                ) = wrapped_rag_res.get("data")

                if sess_hist_ret_obj:
                    print("\n--- Session History Context (Hybrid Vector + Fuzzy) ---")
                    session_docs = await asyncio.to_thread(sess_hist_ret_obj.invoke, user_input_with_context)
                    if not session_docs:
                        print("No session history context found.")
                    for i, doc in enumerate(session_docs):
                        source_type = doc.metadata.get("source", "unknown")
                        print(f"Doc {i+1} (Source: {source_type}):\n{doc.page_content}\n---")

                if url_ret_obj:
                    print("\n--- URL Context ---")
                    url_docs = await asyncio.to_thread(url_ret_obj.invoke, user_input_with_context)
                    for i, doc in enumerate(url_docs):
                        print(f"Doc {i+1} (Source: {doc.metadata.get('source')}): {doc.page_content[:200]}...")

                if refl_chunk_ret_obj:
                    print("\n--- Reflection Context ---")
                    reflection_docs = await asyncio.to_thread(refl_chunk_ret_obj.invoke, user_input_with_context)
                    for i, doc in enumerate(reflection_docs):
                        print(f"Doc {i+1}: {doc.page_content[:200]}...")
            else:
                print(f"❌ RAG retrieval failed: {wrapped_rag_res.get('error_message')}")

            print("\n[2. Gathering File Index Context (Vector + Fuzzy Fallback)...]")
            file_ctx = await cortex_thoughts._get_vector_search_file_index_context(
                query=user_input_with_context,
                session_id_for_log=test_session_id,
                priority=priority
            )
            print("\n--- File Index Context ---")
            print(file_ctx if file_ctx else "No file index context found.")

            print("\n[3. Gathering Global Direct History...]")
            from database import get_global_recent_interactions
            global_hist = await asyncio.to_thread(get_global_recent_interactions, db, limit=5)
            direct_hist_str = cortex_thoughts._format_direct_history(global_hist)
            print("\n--- Direct History ---")
            print(direct_hist_str)

            print("\n[4. Gathering Session Log History...]")
            from database import get_recent_interactions
            log_entries = await asyncio.to_thread(
                get_recent_interactions,
                db,
                getattr(CortexConfiguration, "RAG_HISTORY_COUNT", 5) * 2,
                test_session_id,
                "chat",
                True,
            )
            log_ctx_str = cortex_thoughts._format_log_history(log_entries)
            print("\n--- Log History ---")
            print(log_ctx_str)

            print("\n==================================================\n")
    except KeyboardInterrupt:
        print("\nTest interrupted.")
    except Exception as e:
        logger.error(f"Error during memory retrieval: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
        print("Database session closed. Exiting.")


if __name__ == "__main__":
    asyncio.run(run_asyncbg_context_test())
