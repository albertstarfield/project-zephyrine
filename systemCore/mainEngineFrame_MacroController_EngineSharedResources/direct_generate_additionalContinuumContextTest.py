# direct_generate_additionalContinuumContextQualiaTest.py
import asyncio
import os
import sys

# Import Configurations and Database
import CortexConfiguration
from AdelaideAlbertCortex import CortexThoughts

# Import Cortex components directly to mirror exact production behavior
from cortex_backbone_provider import CortexEngine
from database import SessionLocal, init_db
from interaction_indexer import initialize_global_interaction_vectorstore
from loguru import logger


async def run_qualia_test():
    print("=====================================================")
    print(" ⚙️ CortexConfiguration Parameters Dump ")
    print("=====================================================")
    # Spitting out specific RAG and Model configurations
    config_keys_to_dump = [
        "PROVIDER",
        "MEMORY_SIZE",
        "RAG_HISTORY_COUNT",
        "FUZZY_SEARCH_THRESHOLD_CONTEXT",
        "DATABASE_URL",
        "LLAMA_CPP_N_CTX",
        "VECTOR_CALC_CHUNK_BATCH_TOKEN_SIZE",
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
        # We must initialize the vectorstore first so Chroma is loaded into memory
        await asyncio.to_thread(initialize_global_interaction_vectorstore, provider)

        logger.info("Initializing CortexThoughts (Brain)...")
        cortex_thoughts = CortexThoughts(provider)

        print("\n🧠 Qualia Memory Continuum Test Primed.")
        print("Type 'exit' or 'quit' to stop.\n")

        test_session_id = "test_qualia_debugger_001"

        while True:
            user_input = input(">_ ")

            if user_input.strip().lower() in ["quit", "exit"]:
                break
            if not user_input.strip():
                continue

            print("\n[Searching Memory...]")

            # Using the EXACT ELP1 fetcher from your architecture
            # This handles both Vector Embedding (Semantic) and Fuzzy (Exact Keyword)
            (
                context_str,
                vector_tokens,
            ) = await cortex_thoughts._get_direct_rag_context_elp1(
                db=db, user_input=user_input, session_id=test_session_id
            )

            print("\n================ QUALIA RECALL ===================")
            if context_str.strip():
                print(context_str)
            else:
                print("No relevant context found in memory.")
            print("==================================================\n")
            print(f"📊 Vector Token Usage: {vector_tokens} tokens\n")

    except KeyboardInterrupt:
        print("\nTest interrupted.")
    except Exception as e:
        logger.error(f"Error during memory retrieval: {e}")
    finally:
        db.close()
        print("Database session closed. Exiting.")


if __name__ == "__main__":
    asyncio.run(run_qualia_test())
