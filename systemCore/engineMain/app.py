# app.py
"""

Background Story...
Adelaide Zephyrine Charlotte: Modelling Metacognition for Artificial Quellia


This report/materialization explores the parallels and divergences between current Artificial Intelligence architectures, particularly Large Language Models (LLMs) augmented with external knowledge, and human cognitive processes. Drawing inspiration from the motivation to create a functional digital "clone" as a form of contribution and legacy ("I'm not sure how long will i live anymore. But I want to contribute to my family and friends, and people arounds me. So I decided to clone myself"), we examine analogies such as the "Low IQ LLM + High Crystallized Knowledge" model, the role of semantic embeddings versus simpler search methods, and the fundamental difference introduced by biological neuroplasticity. We contrast the static nature of current AI models with the dynamic adaptation of the human brain, including considerations of neurodiversity like Autism Spectrum Disorder (ASD). Furthermore, we investigate how awareness of limitations, analogous to Quantization-Aware Training in AI, enables strategic adaptation in humans (metacognition). Finally, we touch upon the philosophical distinctions regarding consciousness and embodiment, framing the goal of "cloning" within the realistic scope of replicating knowledge, decision patterns, and simulated experience rather than subjective selfhood.

1. Introduction
The desire to leave a lasting contribution often motivates explorations into the nature of intelligence and the potential for its replication or continuation. Framed by the poignant personal motivation –

 "I'm not sure how long will I live anymore. After I got diagnosed by something that I or We (family) have feared on my brain 2 years ago and have traumatic experience which we've seen person that suffering pain in a facility due to the day and night and then handling lost daughter), there's defective region on the MRI side on the left hemisphere, the medical staff doesn't seem to be confident if I would live for the next months, because the "mass" is quite big and because. But 2 years later it seems that I'm getting better, however it feels like i'm being supported by miracle right now that I do not know how it works, I'm not sure how sturdy is a miracle foundation and how long it will last but here I am. I want to contribute to my family and friends what they have been invested to me. I don't want my disappearance to be meaningless, and people arounds me. So I decided to clone myself at least from the cognitive side slowly built an actuator interface that can fly to the sky that I have learned, from Willy (for the basics of Machine Learning and AI and Spark of the project Zephy Racing against each other from modifying Alpaca-electron), Adri Stefanus (As the frontend & backend Human interface Developer that revamp the jump from the root project which is Alpaca-electron) into usable portable familiar AI webui), Adeela at High School (Observed on how She handle Chemistry and Recursive learning and self taught) (I wish I still had that crush feeling to have motivation at 2025 to propel myself in this hard-times that is ngl a very powerful force.) and Albert (Myself) love imagining stuff Physics for visualizations at Undergraduate and then Uki/Vito about (Task Tree of Thoughts decomposition) and Zhonghuang The Mathematichian (Observing on how he learn stuff and Unknown materials intuition building) at Post-graduate School." -Albert 2025

this report/materialization or delves into the comparison between contemporary AI systems and human cognition. We aim to understand the capabilities and inherent limitations of current AI by drawing analogies with human intelligence, while also acknowledging the profound differences rooted in biology and potentially philosophy. This exploration will cover the architecture of AI models augmented by external knowledge (akin to crystallized intelligence), the mechanisms of information retrieval (semantic embeddings), the critical role of neuroplasticity in biological systems, the implications of neurodiversity, the power of metacognitive awareness, and the philosophical boundaries relevant to the concept of creating a functional "digital clone" or legacy.

2. Modeling Intelligence: AI Analogies and Architectural Limits
A useful analogy posits current advanced AI systems, particularly those employing Retrieval-Augmented Generation (RAG), as possessing a core processing unit (the LLM) akin to fluid intelligence (or processing capability, potentially analogous to a fixed "IQ" score) combined with a vast, external, accessible knowledge base (the vector database) akin to crystallized intelligence. The core LLM, often static post-training, exhibits limitations in complex reasoning, synthesis, and novel problem-solving inherent to its architecture and parameter count.
The RAG mechanism bridges this gap by allowing the LLM to access relevant "crystallized knowledge." This is achieved not through direct vector ingestion by the LLM, but by using semantic vector embeddings (e.g., from models like mxbai-embed-large) to perform similarity searches. An input query is embedded, and this vector is used to find the most semantically relevant text chunks stored in the database (representing documents, past conversations, or experiences). This retrieved text is then provided as context to the LLM. This semantic approach is crucial, vastly outperforming simpler methods like fuzzy search, as it captures meaning, synonyms, and context rather than just surface-level textual similarity.
However, even with perfect knowledge retrieval, the system's ultimate capacity for complex reasoning, creativity, and nuanced understanding remains fundamentally constrained by the core LLM's static architecture. While iterative feedback loops, where the system learns from stored outcomes (errors, successes) via the database, can simulate adaptation and allow the system to tackle more complex execution tasks over time, they do not inherently increase the core model's single-turn reasoning power.
3. The Biological Counterpoint: Neuroplasticity, Neurodiversity, and Embodied Limits
The most significant divergence between current AI and human intelligence lies in neuroplasticity. Unlike the static nature of trained LLM parameters, the human brain physically reorganizes and adapts its structure and connectivity throughout life in response to learning and experience. This dynamic capability operates within the constraints of our species-specific neural architecture. This architecture, evolved over millennia, includes specialized regions (like language centers) that other species (e.g., cats, dolphins) lack, explaining why neuroplasticity alone doesn't enable them to acquire human language despite their own cognitive complexity and learning abilities. Their fundamental neural architecture sets different boundaries.
Furthermore, neurodevelopmental differences, such as those seen in Autism Spectrum Disorder (ASD), should not be misconstrued as "frozen parameters." ASD represents a different developmental trajectory and brain wiring, leading to distinct ways of processing information, social cues, and patterns. Crucially, the autistic brain retains neuroplasticity, allowing for learning, adaptation, and skill development throughout life. This neurodiversity can confer unique cognitive strengths, such as intense focus, exceptional pattern recognition, and novel perspectives, contributing significantly to insight and discovery. This highlights the inadequacy of simplistic metrics like IQ scores, which fail to capture the multifaceted nature of human capability and the potential inherent in diverse cognitive profiles. Biological systems also face limitations, such as those related to cellular regeneration caps like telomere shortening, which act as a finite resource analogous, perhaps, to SSD spare blocks, constraining long-term maintenance differently than hardware degradation.
4. Awareness, Adaptation, and Achievement: Metacognition as QAT
The concept of Quantization-Aware Training (QAT) in AI, where a model learns to perform optimally despite anticipated computational constraints, provides a compelling analogy for human metacognition. Being aware of one's own cognitive strengths, weaknesses, and thought processes allows humans to develop compensatory strategies. Recognizing a limitation (e.g., in memory or calculation speed) enables the use of tools (notes, calculators) and targeted efforts (practice, focused learning) to overcome or work around that constraint. This self-awareness, far from being solely a limitation, becomes a powerful driver for optimizing performance and achieving goals, allowing individuals, regardless of their scores on specific cognitive tests or their neurotype, to leverage their strengths and strategically navigate their challenges. Achievement is thus often a product of not just raw capability, but also of self-awareness, strategy, and perseverance.
5. Philosophical Considerations and the Goal of Digital Legacy
While functional similarities between advanced AI and human cognition can be drawn, profound philosophical questions remain. Current AI lacks subjective experience (qualia) – the "what it's like" to be aware. Human intelligence is deeply intertwined with biological embodiment, shaping our perceptions, motivations, emotions, and understanding through sensory experience and physiological needs. AI goals are externally programmed or derived from optimization functions, contrasting with potentially intrinsic human motivations. The question of whether AI achieves true "understanding" versus sophisticated pattern matching and prediction remains open. These factors suggest that creating a "clone" in the sense of replicating a conscious self is currently, and perhaps fundamentally, impossible.
However, interpreting the motivation as a desire to create a functional digital legacy frames the goal within achievable technological bounds. It is conceivable to build an AI system that encapsulates an individual's vast crystallized knowledge, mimics their decision-making patterns based on recorded data and interactions, simulates learning from experience via database feedback loops, and interacts with the world in a way that reflects their persona and expertise. This would constitute a powerful form of functional continuation and contribution.
6. Conclusion
Comparing AI like Amaryllis/Adelaide to human intelligence reveals both useful analogies and fundamental distinctions. The "Low IQ LLM + High Crystallized Knowledge" model highlights the power of RAG in augmenting static processing units but also underscores the limitations imposed by the core model's fixed architecture. Semantic embeddings are key to unlocking this external knowledge effectively. The crucial differentiator remains biological neuroplasticity, which allows continuous adaptation within evolved architectural constraints, a capability current AI lacks. Neurodiversity further illustrates that varied cognitive architectures can offer unique strengths. Metacognitive awareness of limitations, analogous to QAT, is a powerful tool for human adaptation and achievement. While replicating consciousness remains elusive, the goal of creating a digital legacy – capturing knowledge, function, and simulated experience – appears increasingly feasible, offering a potential path to fulfilling the desire to contribute beyond a biological lifespan, leveraging the strengths of AI augmentation while respecting the unique nature of human existence.
"""
import mimetypes
# --- Standard Library Imports ---
import os
import sys
import time
import json
import re
import asyncio
import threading # Used by asyncio.to_thread internally
import subprocess # Used in AgentTools (agent.py)
import base64 # Used for image handling
from io import BytesIO # Used for image handling
from typing import Any, Dict, List, Optional, Tuple, Union # Added Union
from operator import itemgetter # Used in runnables
import shlex # For safe command splitting in agent tools
import shutil # For copying directory trees in setup_assistant_proxy
import tempfile # For creating temporary files in setup_assistant_proxy
import uuid # For generating request/response IDs
import random
import traceback
#from quart import Quart, Response, request, g, jsonify, current_app # Use Quart imports
# --- Third-Party Library Imports ---
import requests # Used for URL fetching
from bs4 import BeautifulSoup # Used for URL parsing
from loguru import logger # Logging library
from PIL import Image # Used for image handling (optional validation/info)
# numpy import moved inside _find_existing_tot_result to make it optional
import threading
import datetime
import queue
import atexit # To signal shutdown
import datetime
import hashlib



import difflib
import contextlib # For ensuring driver quit
from urllib.parse import urlparse, parse_qs, quote_plus, urljoin

# --- Selenium Imports (add if not present) ---
try:
    from selenium import webdriver
    from selenium.webdriver.remote.webdriver import WebDriver # For type hints
    from selenium.webdriver.remote.webelement import WebElement # For type hints
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import (
        NoSuchElementException, TimeoutException, WebDriverException, InvalidSelectorException
    )
    # Using Chrome specific service/manager
    from selenium.webdriver.chrome.service import Service as ChromeService
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
    logger.info("✅ Selenium and WebDriver Manager imported.")
except ImportError as e:
    SELENIUM_AVAILABLE = False
    WebDriver = None # Define as None if import fails
    WebDriverException = Exception # Define base exception
    NoSuchElementException = Exception
    TimeoutException = TimeoutError
    logger.error(f"❌ Failed to import Selenium/WebDriverManager: {e}. Web scraping/download tools disabled.")
    logger.error("   Install dependencies: pip install selenium webdriver-manager requests beautifulsoup4")

# --- SQLAlchemy Imports ---
from sqlalchemy.orm import Session, sessionmaker # Import sessionmaker
from sqlalchemy import update, inspect as sql_inspect, desc

# --- Flask Imports ---
from flask import Flask, request, Response, g, jsonify # Use Flask imports

try:
    from shared_state import server_is_busy_event
except ImportError:
    logger.critical("Failed to import shared_state. Server busy signaling disabled.")
    # Create a dummy event if import fails to avoid crashing later code
    server_is_busy_event = threading.Event()

# --- Langchain Core Imports ---
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Langchain Community Imports ---
from langchain_community.vectorstores import Chroma # Use Chroma for in-memory history/URL RAG
from langchain.text_splitter import RecursiveCharacterTextSplitter
# --- ADD OLLAMA/FIREWORKS IMPORTS DIRECTLY FOR MULTI-MODEL ---
# Ollama
try:
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings
    logger.info("Using langchain_community imports for Ollama.")
except ImportError:
    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        logger.info("Using langchain_ollama imports.")
    except ImportError:
        logger.error("❌ Failed to import Ollama. Did you install 'langchain-ollama'?")
        ChatOllama = None
        OllamaEmbeddings = None
# Fireworks
try:
    from langchain_fireworks import ChatFireworks, FireworksEmbeddings
except ImportError:
     logger.warning("⚠️ Failed to import Fireworks. Did you install 'langchain-fireworks'? Fireworks provider disabled.")
     ChatFireworks = None
     FireworksEmbeddings = None
# --- END PROVIDER IMPORTS ---


# --- Fuzzy Search Imports ---
try:
    from thefuzz import process as fuzz_process, fuzz
    FUZZY_AVAILABLE = True
    logger.info("✅ thefuzz imported.")
except ImportError:
    FUZZY_AVAILABLE = False
    logger.warning("⚠️ thefuzz not installed. Fuzzy search RAG fallback disabled. Run 'pip install thefuzz python-Levenshtein'.")
    fuzz_process = None # Placeholder
    fuzz = None # Placeholder


# --- Local Imports with Error Handling ---
try:
    from ai_provider import AIProvider, ai_provider_instance as global_ai_provider_ref
    # Import database components needed in app.py
    
    from database import (
        init_db, add_interaction, get_recent_interactions, # <<< REMOVED get_db
        get_past_tot_interactions, Interaction, SessionLocal, AppleScriptAttempt, # Added AppleScriptAttempt if needed here
        get_global_recent_interactions, get_pending_tot_result, mark_tot_delivered,
        get_past_applescript_attempts, FileIndex, search_file_index # Added new DB function
    )
    # Import all config variables (prompts, settings, etc.)
    from config import * # Ensure this includes the SQLite DATABASE_URL and all prompts/models
    # Import Agent components
    # Make sure AmaryllisAgent and _start_agent_task are correctly defined/imported if used elsewhere
    from file_indexer import FileIndexer
    from agent import AmaryllisAgent, AgentTools, _start_agent_task # Keep Agent imports
except ImportError as e:
    print(f"Error importing local modules (database, config, agent, ai_provider): {e}")
    logger.exception("Import Error Traceback:") # Log traceback for import errors
    FileIndexer = None # Define as None if import fails
    FileIndex = None
    search_file_index = None
    sys.exit(1)

# --- NEW: Import the custom lock ---

from priority_lock import ELP0, ELP1 # Ensure these are imported
interruption_error_marker = "Worker task interrupted by higher priority request" # Define consistently

# --- End Local Imports ---

# Add the inspection code again *after* these imports
logger.debug("--- Inspecting Interaction Model Columns AFTER explicit import ---")
logger.debug(f"Columns found by SQLAlchemy: {[c.name for c in Interaction.__table__.columns]}")
if 'tot_delivered' in [c.name for c in Interaction.__table__.columns]: logger.debug("✅ 'tot_delivered' column IS present.")
else: logger.error("❌ 'tot_delivered' column IS STILL MISSING!")
logger.debug("-------------------------------------------------------------")


# Define these near the top, perhaps after imports or before app = Flask(...)

META_MODEL_NAME_STREAM = "Amaryllis-AdelaidexAlbert-MetacognitionArtificialQuellia-Stream"
META_MODEL_NAME_NONSTREAM = "Amaryllis-AdelaidexAlbert-MetacognitionArtificialQuellia"
META_MODEL_OWNER = "zephyrine-foundation"
META_MODEL_FAMILY = "zephyrine"
META_MODEL_PARAM_SIZE = "14.2B" # As requested
META_MODEL_QUANT_LEVEL = "fp16" # As requested
META_MODEL_FORMAT = "gguf" # Common format assumption for Ollama compatibility

# --- Constants for Streaming ---
LOG_QUEUE_TIMEOUT = 0.05 # How long generator waits for a log message (seconds)
PLACEHOLDER_MESSAGE = "(LoadingStreamingBase)(DoNotPanic)(WellBeRightBack)"
LOG_SINK_LEVEL = "DEBUG" # Minimum log level to forward to client
LOG_SINK_FORMAT = "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>" # Example format

# --- Assistant Proxy App Constants ---
ASSISTANT_PROXY_APP_NAME = "AdelaideHijackAppleBridge.app" # Your chosen app name
ASSISTANT_PROXY_DEST_PATH = f"/Applications/{ASSISTANT_PROXY_APP_NAME}"
# --- DEFINE SCRIPT_DIR HERE ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Get the directory containing app.py
# --- Use SCRIPT_DIR to define the source path ---
ASSISTANT_PROXY_SOURCE_PATH = os.path.join(SCRIPT_DIR, "AssistantProxy.applescript") # Assumes script is next to app.py
# --- NEW: Define Search Download Directory ---
SEARCH_DOWNLOAD_DIR = os.path.join(SCRIPT_DIR, "LiteratureReviewPool")
# --- END NEW ---


# --- Logger Configuration ---
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="DEBUG")
logger.info("📝 Logger configured")
logger.info(f"🐍 Running in directory: {os.getcwd()}")
logger.info(f"🔧 Script directory: {os.path.dirname(os.path.abspath(__file__))}")


# --- ADD DEBUGGING STEP ---
try:
    from database import Interaction # Make sure Interaction is imported
    logger.debug("--- Inspecting Interaction Model Columns BEFORE init_db ---")
    logger.debug(f"Columns found by SQLAlchemy: {[c.name for c in Interaction.__table__.columns]}")
    if 'tot_delivered' in [c.name for c in Interaction.__table__.columns]:
        logger.debug("✅ 'tot_delivered' column IS present in mapped model.")
    else:
        logger.error("❌ 'tot_delivered' column IS MISSING from mapped model BEFORE init_db!")
    logger.debug("---------------------------------------------------------")
except Exception as inspect_err:
    logger.error(f"Failed to inspect Interaction model: {inspect_err}")
# --- END DEBUGGING STEP ---

# --- Initialize Database ---
try:
    init_db()
    logger.success("✅ Database Initialized Successfully")
except Exception as e:
    logger.critical(f"🔥🔥 DATABASE INITIALIZATION FAILED: {e}")
    logger.exception("DB Init Traceback:") # Log full traceback
    sys.exit(1)

# === Global Indexer Thread Management ===

# === Global AI Instances ===



_indexer_thread: Optional[threading.Thread] = None
_indexer_stop_event = threading.Event()

def start_file_indexer():
    """Starts the background file indexer thread."""
    global _indexer_thread, ai_provider # <<< Need ai_provider here
    if not FileIndexer:
        logger.error("Cannot start file indexer: FileIndexer class not available (import failed?).")
        return
    if not ai_provider: # <<< Check if AIProvider initialized successfully
        logger.error("Cannot start file indexer: AIProvider (and embedding model) not available.")
        return

    # --- Get embedding model ---
    embedding_model = ai_provider.embeddings
    if not embedding_model:
        logger.error("Cannot start file indexer: Embedding model not found within AIProvider.")
        return
    # --- End get embedding model ---

    if _indexer_thread is None or not _indexer_thread.is_alive():
        logger.info("🚀 Starting background file indexer service...")
        try:
            # --- Pass embedding_model to FileIndexer ---
            indexer_instance = FileIndexer(
                stop_event=_indexer_stop_event,
                provider=ai_provider,
                server_busy_event=server_is_busy_event # <<< Pass the busy event
            )
            # --- End pass embedding_model ---
            _indexer_thread = threading.Thread(
                target=indexer_instance.run,
                name="FileIndexerThread",
                daemon=True
            )
            _indexer_thread.start()
            logger.success("✅ File indexer thread started successfully.")
        except Exception as e:
            logger.critical(f"🔥🔥 Failed to instantiate or start FileIndexer thread: {e}")
            logger.exception("Indexer Startup Traceback:")
    else:
        logger.warning("🤔 File indexer thread already running.")

# === NEW: Global Self-Reflection Thread Management ===
_reflector_thread: Optional[threading.Thread] = None
_reflector_stop_event = threading.Event()
_reflector_lock = threading.Lock() # Lock to prevent concurrent reflection cycles if one runs long

def run_self_reflection_loop():
    """
    Main loop for self-reflection. Continuously processes eligible interactions
    (including previous reflection results) in batches until none are found,
    then waits minimally before checking again.
    """
    global ai_provider, ai_chat # Need access to these instances
    thread_name = threading.current_thread().name
    logger.info(f"✅ {thread_name} started (Continuous Reflection Logic - Minimal Wait).")

    if not ai_provider or not ai_chat:
        logger.error(f"🛑 {thread_name}: AIProvider or AIChat not initialized. Cannot run reflection.")
        return

    # --- Configuration ---
    # How many interactions to process per DB query (from config)
    # REFLECTION_BATCH_SIZE = 5 (Example value if not imported)
    # Max age (optional, uncomment filter below if needed)
    # MAX_REFLECTION_AGE_DAYS = int(os.getenv("MAX_REFLECTION_AGE_DAYS", 7))

    # --- MODIFIED: Wait Times ---
    # How long to wait ONLY if NO work was found in a full active cycle
    IDLE_WAIT_SECONDS = 0.01 # Minimal wait to prevent pure busy-looping
    # How long to wait briefly between batches IF work IS being processed
    ACTIVE_CYCLE_PAUSE_SECONDS = 0.0 # No pause between batches when active
    # --- END MODIFICATION ---

    # Input types eligible for reflection
    reflection_eligible_input_types = ['text', 'reflection_result']
    logger.info(f"{thread_name}: Config - BatchSize={REFLECTION_BATCH_SIZE}, IdleWait={IDLE_WAIT_SECONDS}s, ActivePause={ACTIVE_CYCLE_PAUSE_SECONDS}s")
    logger.info(f"{thread_name}: Eligible Input Types: {reflection_eligible_input_types}")

    # --- Main Loop ---
    while not _reflector_stop_event.is_set():
        cycle_start_time = time.monotonic()
        total_processed_this_active_cycle = 0
        work_found_in_cycle = False # Track if any work was done

        logger.info(f"🤔 {thread_name}: Starting ACTIVE reflection cycle...")

        # --- Attempt to acquire lock ---
        if not _reflector_lock.acquire(blocking=False):
             logger.warning(f"{thread_name}: Previous reflection cycle lock held? Skipping.")
             # Short wait if lock held, then try again next outer loop iteration
             _reflector_stop_event.wait(timeout=1.0)
             continue

        db: Optional[Session] = None
        try:
            # --- Wait if Server Busy (Checks before starting the main work) ---
            # Use a flag to log only once per busy period start
            was_busy_waiting = False
            while server_is_busy_event.is_set():
                 if not was_busy_waiting:
                      logger.info(f"🚦 {thread_name}: Server busy, pausing reflection start...")
                      was_busy_waiting = True
                 if _reflector_stop_event.is_set(): break
                 # Check stop event frequently while waiting
                 if _reflector_stop_event.wait(timeout=0.5): break
            if was_busy_waiting:
                 wait_duration = time.monotonic() - cycle_start_time # Approx wait time
                 logger.info(f"🟢 {thread_name}: Server free after busy wait (~{wait_duration:.1f}s).")
            if _reflector_stop_event.is_set(): break # Exit main loop if stopped during wait

            # --- Create DB session for this active cycle ---
            try:
                db = SessionLocal()
                if not db: raise ValueError("Failed to create DB session.")
                logger.trace(f"{thread_name}: DB Session created for active cycle.")
            except Exception as db_err:
                 logger.error(f"{thread_name}: Failed to get DB session: {db_err}")
                 _reflector_lock.release() # Release lock if DB fails
                 _reflector_stop_event.wait(timeout=5) # Wait before retrying cycle
                 continue

            # --- Inner Loop: Keep processing batches as long as work is found ---
            while not _reflector_stop_event.is_set():
                batch_processed_count = 0
                interactions_to_reflect = []
                try:
                    # Query for the next batch of eligible interactions
                    logger.trace(f"{thread_name}: Querying DB for next reflection batch...")
                    query = db.query(Interaction).filter(
                        Interaction.reflection_completed == False,
                        Interaction.mode == 'chat',
                        Interaction.input_type.in_(reflection_eligible_input_types),
                        # Optional: Add age limit filter here
                        # Interaction.timestamp >= datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=MAX_REFLECTION_AGE_DAYS)
                    ).order_by(
                        Interaction.timestamp.asc() # Process oldest eligible first
                    ).limit(REFLECTION_BATCH_SIZE)

                    interactions_to_reflect = query.all()
                    logger.trace(f"{thread_name}: Query returned {len(interactions_to_reflect)} interactions.")

                except Exception as query_err:
                    logger.error(f"{thread_name}: Error querying interactions for reflection batch: {query_err}")
                    _reflector_stop_event.wait(timeout=5) # Wait a bit before exiting inner loop
                    break # Exit inner loop on query error, will retry after outer loop wait

                # --- Check if any work was found in this batch ---
                if not interactions_to_reflect:
                    logger.debug(f"{thread_name}: No more eligible interactions found in this batch/query.")
                    break # Exit the inner batch-processing loop

                # --- Process the found batch ---
                logger.info(f"{thread_name}: Found {len(interactions_to_reflect)} interaction(s) in batch. Processing...")
                work_found_in_cycle = True # Mark that we found work in this overall cycle

                for interaction in interactions_to_reflect:
                    if _reflector_stop_event.is_set():
                        logger.info(f"{thread_name}: Stop signal received during batch processing.")
                        break # Exit the 'for interaction' loop

                    # --- Wait if Server Busy (Check before each item) ---
                    item_was_busy_waiting = False
                    while server_is_busy_event.is_set():
                        if not item_was_busy_waiting:
                             logger.warning(f"{thread_name}: Server became busy during batch processing. Pausing...")
                             item_was_busy_waiting = True
                        if _reflector_stop_event.is_set(): break
                        if _reflector_stop_event.wait(timeout=0.5): break # Check stop frequently
                    if item_was_busy_waiting:
                         logger.info(f"{thread_name}: Server free, resuming batch processing.")
                    if _reflector_stop_event.is_set(): break # Check stop again after wait
                    # --- End Wait Logic ---

                    # Extract details for logging and processing
                    original_input = interaction.user_input or "[Original input missing]"
                    original_id = interaction.id
                    original_input_type = interaction.input_type
                    logger.info(f"{thread_name}: --> Triggering reflection task for Interaction ID {original_id} (Type: {original_input_type}) - Input: '{original_input[:60]}...'")

                    # Prepare parameters for background_generate
                    reflection_session_id = f"reflection_{uuid.uuid4()}" # Unique session for the reflection task log trail
                    task_launched = False
                    try:
                        # Frame the input clearly for the reflection context
                        reflection_input = f"[Self-Reflection on Interaction ID {original_id} (Type: {original_input_type})]: {original_input}"

                        # Run the async background_generate function using asyncio.run()
                        # This blocks the current (reflector) thread until background_generate completes
                        # (which is okay as the reflector is dedicated to this)
                        # background_generate itself uses asyncio internally but runs its core logic
                        # and saves state before returning.
                        asyncio.run(
                            ai_chat.background_generate(
                                db=db, # Pass the current session
                                user_input=reflection_input,
                                session_id=reflection_session_id,
                                classification="chat_complex", # Force complex for reflection
                                image_b64=None,
                                update_interaction_id=original_id # Tells bg_generate this is reflection
                            )
                        )
                        # If asyncio.run completes without exception, assume launch was successful
                        # (background_generate handles its own internal errors and saves state)
                        logger.info(f"{thread_name}: --> Background reflection task for ID {original_id} completed triggering.")
                        task_launched = True
                        batch_processed_count += 1

                    except Exception as trigger_err:
                        logger.error(f"{thread_name}: Failed to trigger/run background_generate for interaction ID {original_id}: {trigger_err}")
                        logger.exception(f"{thread_name} Trigger/Run Traceback:")
                        # If triggering fails, don't mark original as complete, let it retry next cycle

                    # --- Mark Original Interaction as Completed (if launch succeeded) ---
                    # This prevents it from being picked up again by the next query.
                    if task_launched:
                        try:
                            logger.debug(f"{thread_name}: Marking original interaction {original_id} as reflection_completed=True.")
                            stmt = update(Interaction).where(Interaction.id == original_id).values(
                                reflection_completed=True,
                                last_modified_db=datetime.datetime.now(datetime.timezone.utc)
                                )
                            db.execute(stmt)
                            db.commit() # Commit this specific update
                            logger.info(f"{thread_name}: Original interaction {original_id} marked complete.")
                        except Exception as update_err:
                            logger.error(f"{thread_name}: Failed to mark interaction {original_id} as reflected: {update_err}")
                            db.rollback() # Rollback failed update; it might get picked up again.

                    # --- No Pause Between Items in Batch (ACTIVE_CYCLE_PAUSE_SECONDS is 0) ---
                    # if not _reflector_stop_event.is_set():
                    #    time.sleep(ACTIVE_CYCLE_PAUSE_SECONDS) # Effectively time.sleep(0)

                # --- End of processing items in the current batch ---
                total_processed_this_active_cycle += batch_processed_count
                logger.info(f"{thread_name}: Finished processing batch ({batch_processed_count} items). Total this cycle: {total_processed_this_active_cycle}.")
                if _reflector_stop_event.is_set(): break # Check stop signal after processing batch

                # --- No Pause Between Batches (ACTIVE_CYCLE_PAUSE_SECONDS is 0) ---
                # if not _reflector_stop_event.is_set():
                #    time.sleep(ACTIVE_CYCLE_PAUSE_SECONDS) # Effectively time.sleep(0)

            # --- End of Inner Batch Processing Loop (exited because query returned empty or stop signal) ---

        except Exception as cycle_err:
            logger.error(f"💥 {thread_name}: Unhandled error during active reflection cycle: {cycle_err}")
            logger.exception(f"{thread_name} Cycle Traceback:")
            if db: # Rollback any potential partial changes
                try: db.rollback()
                except Exception as rb_err: logger.error(f"{thread_name}: Error during rollback: {rb_err}")

        finally:
            # --- Close DB Session for the Cycle ---
            if db:
                try: db.close(); logger.debug(f"{thread_name}: DB session closed for cycle.")
                except Exception as close_err: logger.error(f"{thread_name}: Error closing DB session: {close_err}")

            # --- Release Lock ---
            try:
                _reflector_lock.release()
                logger.trace(f"{thread_name}: Released cycle lock.")
            except (threading.ThreadError, RuntimeError) as lk_err:
                 logger.warning(f"{thread_name}: Lock release issue at end of cycle? {lk_err}")

            # --- Log Cycle Finish ---
            cycle_duration = time.monotonic() - cycle_start_time
            logger.info(f"{thread_name}: ACTIVE reflection cycle finished in {cycle_duration:.2f}s. Processed: {total_processed_this_active_cycle} interaction(s).")

            # --- Minimal Wait Before Next Cycle Check ---
            # Wait longer only if absolutely no work was found in the entire cycle
            wait_time_seconds = IDLE_WAIT_SECONDS if not work_found_in_cycle else ACTIVE_CYCLE_PAUSE_SECONDS # Use 0 if work was found
            if wait_time_seconds > 0:
                 logger.debug(f"{thread_name}: Waiting {wait_time_seconds:.2f} seconds before next cycle check...")
                 stopped = _reflector_stop_event.wait(timeout=wait_time_seconds)
                 if stopped:
                     logger.info(f"{thread_name}: Stop signal received during wait.")
                     break # Exit outer while loop
            else:
                 # If wait time is 0, immediately check stop event before looping again
                 if _reflector_stop_event.is_set():
                      logger.info(f"{thread_name}: Stop signal received.")
                      break

    # --- End of Outer While Loop ---
    logger.info(f"🛑 {thread_name}: Exiting.")


def start_self_reflector():
    """Starts the background self-reflection thread."""
    global _reflector_thread
    if not ENABLE_SELF_REFLECTION:
        logger.info("🤔 Self-reflection thread disabled via config.")
        return

    if _reflector_thread is None or not _reflector_thread.is_alive():
        logger.info("🚀 Starting background self-reflection service...")
        try:
            _reflector_stop_event.clear() # Ensure stop event is not set
            _reflector_thread = threading.Thread(
                target=run_self_reflection_loop,
                name="SelfReflectorThread",
                daemon=True
            )
            _reflector_thread.start()
            logger.success("✅ Self-reflection thread started successfully.")
        except Exception as e:
            logger.critical(f"🔥🔥 Failed to start SelfReflector thread: {e}")
            logger.exception("Reflector Startup Traceback:")
    else:
        logger.warning("🤔 Self-reflection thread already running.")

def stop_self_reflector():
    """Signals the self-reflection thread to stop."""
    global _reflector_thread
    if not ENABLE_SELF_REFLECTION: return # Don't try to stop if disabled

    if _reflector_thread and _reflector_thread.is_alive():
        logger.info("Signaling self-reflection thread to stop...")
        _reflector_stop_event.set()
        # Optional: Wait for thread to finish (might take time if in sleep)
        # _reflector_thread.join(timeout=10)
        # if _reflector_thread.is_alive(): logger.warning("Self-reflection thread did not stop within timeout.")
        logger.info("Stop signal sent to self-reflection thread.")
    else:
        logger.info("Self-reflection thread not running or already stopped.")

def stop_file_indexer():
    """Signals the file indexer thread to stop."""
    global _indexer_thread
    if _indexer_thread and _indexer_thread.is_alive():
        logger.info("Signaling file indexer thread to stop...")
        _indexer_stop_event.set()
        # Optional: Wait for thread to finish with a timeout
        # _indexer_thread.join(timeout=30) # Wait up to 30 seconds
        # if _indexer_thread.is_alive():
        #     logger.warning("File indexer thread did not stop within timeout.")
        logger.info("Stop signal sent to file indexer thread.")
    else:
        logger.info("File indexer thread not running or already stopped.")

# Register the stop function to run when the application exits
atexit.register(stop_file_indexer)
# Register the stop function for application exit
atexit.register(stop_self_reflector)


# --- Flask App Setup ---
app = Flask(__name__) # Use Flask app

# --- Request Context Functions for DB ---
@app.before_request
def setup_and_log_request():
    """Opens DB session and logs incoming request details."""
    # 0. Signal Busy Start
    if not server_is_busy_event.is_set(): # Avoid unnecessary locking if already set
        logger.trace("--> Request IN - Setting server_is_busy_event")
        server_is_busy_event.set()
    # --- If using a counter approach ---
    # g.request_count = getattr(g, 'request_count', 0) + 1
    # if g.request_count == 1:
    #    server_is_busy_event.set()
    # ---
    # 1. Open DB session
    try:
        g.db = SessionLocal()
        logger.trace("DB session opened for request.")
    except Exception as db_err:
        logger.error(f"!!! FAILED TO OPEN DB SESSION in before_request: {db_err}")
        g.db = None # Ensure g.db is None if opening failed

    # 2. Log Incoming Request
    try:
        headers = dict(request.headers)
        content_type = headers.get("Content-Type", "N/A")
        content_length = headers.get("Content-Length", "N/A")
        remote_addr = request.remote_addr or "Unknown"
        query_string = request.query_string.decode() if request.query_string else ""

        log_message = (
            f"--> REQ IN : {request.method} {request.path} "
            f"QS='{query_string}' "
            f"From={remote_addr} "
            f"Type={content_type} Len={content_length}"
        )
        logger.info(log_message)
        logger.debug(f"    REQ Headers: {json.dumps(headers, indent=2)}")
        # Body logging snippet (optional, use with caution) - Same as before
        # if request.content_length and request.content_length < 5000: # Only log small bodies
        #    try:
        #        body_snippet = request.get_data(as_text=True)[:500] # Read snippet
        #        logger.debug(f"    REQ Body Snippet: {body_snippet}...")
        #    except Exception as body_err: logger.warning(f"    REQ Body: Error reading snippet: {body_err}")
        # elif request.content_length: logger.debug(f"    REQ Body: Exists but not logging snippet (Length: {request.content_length}).")
        # else: logger.debug("    REQ Body: No body or zero length.")

    except Exception as log_err:
        logger.error(f"!!! Error during incoming request logging: {log_err}")
        # Continue processing the request anyway

@app.after_request
def log_and_clear_busy(response: Response) -> Response:
    """Logs details of the outgoing response AFTER the route handler."""
    # (Keep the exact same logic as the previous log_outgoing_response function)
    try:
        if request:
            log_message = (
                f"<-- RESP OUT: {request.method} {request.path} "
                f"Status={response.status_code} "
                f"Type={response.content_type} Len={response.content_length}"
            )
            logger.info(log_message)
            if (not response.is_streamed
                and response.mimetype == 'application/json'
                and response.content_length is not None
                and response.content_length < 10000):
                try:
                    data = response.get_data(as_text=True)
                    logger.debug(f"    RESP Body Snippet: {data[:500]}...")
                except Exception: logger.debug("    RESP Body: Could not get/decode JSON data for logging snippet.")
            elif response.is_streamed: logger.debug("    RESP Body: Streamed response (not logging snippet).")
            else: logger.debug(f"    RESP Body: Not logging snippet (Type: {response.mimetype}, Streamed: {response.is_streamed}).")
        else:
            logger.warning("!!! Response logging skipped: Request context not found.")
    except Exception as log_err:
        logger.error(f"!!! Error during outgoing response logging: {log_err}")
    finally:
        if server_is_busy_event.is_set():
            logger.trace("<-- Request OUT - Clearing server_is_busy_event")
            server_is_busy_event.clear()
        return response

@app.teardown_request
def teardown_request_db(exception=None): # Use your original function name if you prefer
    """Close the DB session after each request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()
        logger.trace("DB session closed for request.")
    if exception:
         # Log the exception that might have caused the teardown
         logger.error(f"Exception during request: {exception}")



def setup_assistant_proxy():
    """Reads AssistantProxy.applescript, compiles it, copies to /Applications, and attempts permission priming."""
    logger.info(f"Checking/Creating Assistant Proxy at {ASSISTANT_PROXY_DEST_PATH}...")

    # 1. Check if source AppleScript file exists
    if not os.path.isfile(ASSISTANT_PROXY_SOURCE_PATH):
        logger.critical(f"❌ Source AppleScript file not found at: {ASSISTANT_PROXY_SOURCE_PATH}")
        logger.critical("   Cannot create the Assistant Proxy application.")
        return False

    # Create a temporary directory for compilation
    with tempfile.TemporaryDirectory() as tmpdir:
        compiled_app_path_tmp = os.path.join(tmpdir, ASSISTANT_PROXY_APP_NAME)

        # 2. Compile the AppleScript from file into an Application bundle
        compile_cmd = ["osacompile", "-o", compiled_app_path_tmp, ASSISTANT_PROXY_SOURCE_PATH]
        logger.debug(f"Running osacompile: {' '.join(compile_cmd)}")
        try:
            process = subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
            logger.debug(f"osacompile stdout: {process.stdout}")
            if process.stderr: # Log stderr even on success, might contain warnings
                 logger.warning(f"osacompile stderr: {process.stderr}")
            logger.success(f"✅ Successfully compiled proxy app in temporary location: {compiled_app_path_tmp}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ osacompile failed (RC={e.returncode}): {e.stderr or e.stdout}")
            logger.error(f"   Check syntax in source file: {ASSISTANT_PROXY_SOURCE_PATH}")
            return False
        except FileNotFoundError:
            logger.error("❌ osacompile command not found. Is Xcode Command Line Tools installed?")
            return False
        except Exception as e:
            logger.error(f"❌ Error running osacompile: {e}")
            return False

        # 3. Copy the compiled .app to /Applications (requires sudo privileges)
        logger.info(f"Attempting to copy compiled app to {ASSISTANT_PROXY_DEST_PATH}...")
        try:
            if os.path.exists(ASSISTANT_PROXY_DEST_PATH):
                logger.warning(f"'{ASSISTANT_PROXY_DEST_PATH}' already exists. Removing old version (requires sudo).")
                # Use sudo directly since the script is assumed to run with sudo
                subprocess.run(["rm", "-rf", ASSISTANT_PROXY_DEST_PATH], check=True)

            # Use sudo directly for copy
            copy_cmd = ["cp", "-R", compiled_app_path_tmp, "/Applications/"]
            logger.debug(f"Running copy command: sudo {' '.join(copy_cmd)}")
            subprocess.run(copy_cmd, check=True) # Run without explicit sudo here, as parent script has it

            logger.success(f"✅ Successfully copied '{ASSISTANT_PROXY_APP_NAME}' to /Applications.")

            # --- 4. Attempt to Prime Permissions ---
            logger.info("Attempting to trigger initial permission prompts (may require user interaction)...")
            priming_action_details = {
                "actionType": "prime_permissions",
                "actionParamsJSON": "{}" # No specific params needed for priming
            }
            # Prepare osascript command to call the new handler
            params_json_str = priming_action_details["actionParamsJSON"]
            escaped_json_param = json.dumps(params_json_str) # Double encode
            applescript_command = f'''
            tell application "{ASSISTANT_PROXY_DEST_PATH}"
                handleAction given parameters:{{actionType:"prime_permissions", actionParamsJSON:{escaped_json_param}}}
            end tell
            '''
            osa_command = ["osascript", "-e", applescript_command]
            try:
                logger.debug(f"Running permission priming command: {osa_command}")
                # Run with a short timeout, don't check return code as errors are expected if permissions denied
                prime_process = subprocess.run(osa_command, capture_output=True, text=True, timeout=15, check=False)
                logger.info("Permission priming command sent.")
                if prime_process.stdout: logger.debug(f"Priming stdout: {prime_process.stdout.strip()}")
                if prime_process.stderr: logger.warning(f"Priming stderr: {prime_process.stderr.strip()}") # Stderr expected if prompts shown/denied
            except subprocess.TimeoutExpired:
                logger.warning("Permission priming script timed out (might be waiting for user input).")
            except Exception as prime_e:
                logger.warning(f"Failed to run priming script (this might be ok): {prime_e}")
            # --- End Priming ---

            # --- Final User Instructions ---
            print("-" * 60)
            print(f"IMPORTANT: Assistant Proxy Setup Complete!")
            print(f"'{ASSISTANT_PROXY_APP_NAME}' is now in /Applications.")
            print("\n>>> PERMISSION PROMPTS MAY HAVE APPEARED <<<")
            print("If macOS asked for permission to access Calendars, Contacts,")
            print("Reminders, etc., please ensure you clicked 'OK'/'Allow'.")
            print("\n>>> PLEASE MANUALLY CHECK/GRANT PERMISSIONS <<<")
            print("1. Open 'System Settings' > 'Privacy & Security'.")
            print("2. Check these sections for 'AssistantProxy' and enable it:")
            print("    - Full Disk Access (Recommended for file operations)")
            print("    - Automation (Allow control of Finder, System Events, etc.)")
            print("    - Calendars")
            print("    - Contacts")
            print("    - Reminders")
            print("    - Photos (If needed)")
            print("    - Accessibility (If needed)")
            print("\nFor Calendars, Contacts, Reminders: If AssistantProxy is not")
            print("listed yet, it will be added automatically after you allow")
            print("the first permission prompt triggered by an action.")
            print("For Full Disk Access/Automation: You may need to click '+'")
            print("to add '/Applications/AssistantProxy.app'.")
            print("-" * 60)
            # --- END Final User Instructions ---

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to copy/remove app in /Applications (RC={e.returncode}): {e.stderr or e.stdout}")
            logger.error("   This script needs to be run with sudo privileges.")
            return False
        except Exception as e:
            logger.error(f"❌ Error copying proxy app to /Applications: {e}")
            return False
    # Temporary directory tmpdir is automatically cleaned up



class TaskInterruptedException(Exception):
    """Custom exception raised when an ELP0 task is interrupted."""
    pass

# === AI Chat Logic (Amaryllis - SQLite RAG with Fuzzy Search) ===
class AIChat:
    """Handles Chat Mode interactions with RAG, ToT, Action Analysis, Multi-LLM routing, and VLM preprocessing."""

    def __init__(self, provider: AIProvider):
        self.provider = provider # AIProvider instance with multiple models
        self.vectorstore_url: Optional[Chroma] = None
        self.vectorstore_history: Optional[Chroma] = None # In-memory store for current request
        self.current_session_id: Optional[str] = None
        self.setup_prompts()

    def setup_prompts(self):
        """Initializes Langchain prompt templates."""
        logger.debug("Setting up AIChat prompt templates...")
        self.text_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", PROMPT_CHAT), # Expects various context keys
                ("human", "{input}")
            ]
        )
        self.visual_prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=PROMPT_VISUAL_CHAT),
                MessagesPlaceholder(variable_name="history_rag_messages"),
                HumanMessage(content="Image Description:\n{image_description}\n\nEmotion/Context Analysis: {emotion_analysis}\n\nUser Query: {input}"),
            ]
        )
        # Prompt specifically for getting a description from the VLM
        self.vlm_description_prompt = ChatPromptTemplate.from_template(
            "Describe the key elements, objects, people, and activities in the provided image accurately and concisely. Focus on factual observation."
        )
        self.input_classification_prompt = ChatPromptTemplate.from_template(PROMPT_COMPLEXITY_CLASSIFICATION)
        self.tot_prompt = ChatPromptTemplate.from_template(PROMPT_TREE_OF_THOUGHTS)
        self.emotion_analysis_prompt = ChatPromptTemplate.from_template(PROMPT_EMOTION_ANALYSIS)
        self.image_latex_prompt = ChatPromptTemplate.from_template(PROMPT_IMAGE_TO_LATEX)
        logger.debug("AIChat prompt templates setup complete.")

    async def _generate_file_search_query_async(self, db: Session, user_input_for_analysis: str, recent_direct_history_str: str, session_id: str) -> str:
        """
        Uses the default LLM to generate a concise search query for the file index.
        Removes <think> tags and cleans the output.
        """
        query_gen_id = f"fqgen-{uuid.uuid4()}"
        logger.info(f"{query_gen_id}: Generating dedicated file search query...")

        default_model = self.provider.get_model("default")
        if not default_model:
            logger.error(f"{query_gen_id}: Default model not available for file query generation. Falling back to user input.")
            # Log fallback
            add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning",
                            user_input="File Query Gen Fallback",
                            llm_response="Default model unavailable, using raw input for file search.")
            return user_input_for_analysis # Fallback to original input

        prompt_input = {
            "input": user_input_for_analysis,
            "recent_direct_history": recent_direct_history_str
        }

        chain = (
            ChatPromptTemplate.from_template(PROMPT_GENERATE_FILE_SEARCH_QUERY)
            | default_model
            | StrOutputParser()
        )

        query_gen_timing_data = {"session_id": session_id, "mode": "chat", "execution_time_ms": 0}
        generated_query_raw = ""
        try:
            generated_query_raw = await asyncio.to_thread(
                self._call_llm_with_timing, chain, prompt_input, query_gen_timing_data
            )
            logger.trace(f"{query_gen_id}: Raw generated query response: '{generated_query_raw}'")

            # --- Clean the output ---
            # 1. Remove <think> tags
            cleaned_query = re.sub(r'<think>.*?</think>', '', generated_query_raw, flags=re.DOTALL | re.IGNORECASE)
            # 2. Trim whitespace
            cleaned_query = cleaned_query.strip()
            # 3. Optional: Remove potential quotes if the model wraps the query
            cleaned_query = re.sub(r'^["\']|["\']$', '', cleaned_query)

            if not cleaned_query:
                 logger.warning(f"{query_gen_id}: LLM generated an empty search query. Falling back to user input.")
                 # Log empty generation
                 add_interaction(db, session_id=session_id, mode="chat", input_type="log_info",
                                 user_input="File Query Gen Result",
                                 llm_response="LLM generated empty query, using raw input for file search.")
                 return user_input_for_analysis # Fallback

            logger.info(f"{query_gen_id}: Generated file search query: '{cleaned_query}'")
            # Log successful generation
            add_interaction(db, session_id=session_id, mode="chat", input_type="log_debug",
                            user_input="File Query Gen Result",
                            llm_response=f"Generated query: '{cleaned_query}'. Raw: '{generated_query_raw[:100]}...'")
            return cleaned_query

        except Exception as e:
            logger.error(f"❌ {query_gen_id}: Error generating file search query: {e}")
            logger.exception(f"{query_gen_id}: Query Generation Traceback")
            # Log the error
            add_interaction(db, session_id=session_id, mode="chat", input_type="log_error",
                            user_input="File Query Gen Failed",
                            llm_response=f"Error: {e}. Raw Response: '{generated_query_raw[:100]}...'")
            # Fallback to original input on error
            return user_input_for_analysis
    # --- END NEW HELPER ---

    def _format_file_index_results(self, results: List[FileIndex]) -> str:
        """Formats FileIndex search results for the LLM prompt."""
        if not results:
            return "No relevant files found in the index."
        if not isinstance(results, list):
            logger.error(f"_format_file_index_results received non-list: {type(results)}")
            return "Invalid file index results provided."
        if not results: # Check again
            return "No relevant files found in the index."

        context_str = ""
        max_snippet_len = 300 # Max characters per snippet
        max_total_len = 2000 # Max total context length
        current_len = 0

        for i, record in enumerate(results):
            snippet = ""
            if record.index_status == 'indexed_text' and record.indexed_content:
                snippet = record.indexed_content[:max_snippet_len]
                if len(record.indexed_content) > max_snippet_len:
                    snippet += "..."
            elif record.processing_error:
                snippet = f"[Error accessing file: {record.processing_error}]"
            elif record.index_status == 'error_permission':
                 snippet = "[Error: Permission Denied]"
            elif record.index_status == 'skipped_size':
                 snippet = "[Content not indexed: File too large]"
            else:
                 snippet = "[Metadata indexed, no text content extracted]"

            entry = (f"--- File Result {i+1} ---\n"
                     f"Path: {record.file_path}\n"
                     f"Modified: {record.last_modified_os.strftime('%Y-%m-%d %H:%M') if record.last_modified_os else 'Unknown'}\n"
                     f"Status: {record.index_status}\n"
                     f"Content Snippet: {snippet}\n"
                     f"---\n")

            if current_len + len(entry) > max_total_len:
                context_str += "[File index context truncated due to length limit]...\n"
                break

            context_str += entry
            current_len += len(entry)

        return context_str if context_str else "No relevant files found in the index."

    def _run_search_and_download_sync(self, query: str, session_id: str, num_results: int, timeout: int, engines: List[str], download: bool, download_dir: str, dedup_mode: str, similarity_threshold: float):
        """
        Synchronous function to perform web scraping and downloading.
        Designed to be run in a separate thread via asyncio.to_thread.
        """
        search_logger = logger.bind(task="web_search", session=session_id)
        search_logger.info(f"Starting synchronous search task for query: '{query}'")

        if not SELENIUM_AVAILABLE:
            search_logger.error("Cannot perform search: Selenium/WebDriver is not available.")
            # Log failure to DB
            db = SessionLocal()
            try: add_interaction(db, session_id=session_id, mode="chat", input_type="log_error", user_input=f"Web Search Failed: {query}", llm_response="Selenium components missing.")
            finally: db.close()
            return # Exit if no Selenium

        # --- Engine Mapping (Internal) ---
        engine_map = {
            'ddg': self._scrape_duckduckgo, 'google': self._scrape_google,
            'searx': self._scrape_searx, 'sem': self._scrape_semantic_scholar,
            'scholar': self._scrape_google_scholar, 'base': self._scrape_base,
            'core': self._scrape_core, 'scigov': self._scrape_sciencegov,
            'baidu': self._scrape_baidu_scholar, 'refseek': self._scrape_refseek,
            'scidirect': self._scrape_sciencedirect, 'mdpi': self._scrape_mdpi,
            'tandf': self._scrape_tandf, 'ieee': self._scrape_ieee,
            'springer': self._scrape_springer
            # Add other implemented _scrape_ methods here
        }
        selected_engines = [e for e in engines if e in engine_map]
        if not selected_engines:
             search_logger.warning("No valid/implemented engines selected for search.")
             return # Nothing to do

        all_results = {}
        deduplicated_results = {}
        total_found_dedup = 0
        download_tasks = []
        download_success_count = 0

        # --- Execute Scrapers using WebDriver ---
        # Use the managed_webdriver context manager
        # Note: 'no_images' could be added as a parameter if needed
        with managed_webdriver(no_images=True) as driver:
            if driver is None:
                search_logger.error("WebDriver failed to initialize. Aborting search.")
                db = SessionLocal(); # Log failure to DB
                try: add_interaction(db, session_id=session_id, mode="chat", input_type="log_error", user_input=f"Web Search Failed: {query}", llm_response="WebDriver initialization failed.")
                finally: db.close()
                return # Exit if driver failed

            search_logger.info(f"WebDriver ready. Scraping engines: {selected_engines}")
            for engine_name in selected_engines:
                scraper_func = engine_map.get(engine_name)
                if not scraper_func: continue # Should not happen if selected_engines is filtered

                # Prepare args (adjust based on specific scraper needs)
                scraper_args = [driver, query, num_results, timeout]
                if engine_name in ['ddg', 'google', 'scholar']: scraper_args.append(1) # Add max_pages=1 for now
                # Add SearX instance handling if needed (requires config access or passing instances)
                # if engine_name == 'searx': scraper_args.insert(1, random_searx_instance)

                try:
                    search_logger.info(f"--- Scraping {engine_name.upper()} ---")
                    start_time = time.time()
                    # Call the internal scraper method
                    result_list = scraper_func(*scraper_args)
                    end_time = time.time()
                    search_logger.info(f"--- Finished {engine_name.upper()} in {end_time - start_time:.2f}s ({len(result_list or [])} results) ---")
                    all_results[engine_name] = result_list if result_list else []
                except Exception as exc:
                    search_logger.error(f"Error during scraping for {engine_name}: {exc}")
                    search_logger.exception("Scraper Traceback:")
                    all_results[engine_name] = []
                # Add a small delay between engines?
                time.sleep(random.uniform(0.5, 1.5))

        # --- Deduplication ---
        search_logger.info(f"Performing deduplication (Mode: {dedup_mode})...")
        # (Copy deduplication logic from search_cli.py main(), adapting variable names)
        deduplicated_results = {engine: [] for engine in all_results}
        total_found_dedup = 0
        engine_order = selected_engines # Process in the order they were run

        if dedup_mode == 'url':
            seen_urls = set()
            for engine in engine_order:
                if engine in all_results:
                    for res in all_results[engine]:
                        url = res.get('url')
                        if url and url.startswith('http') and url not in seen_urls:
                            deduplicated_results[engine].append(res); seen_urls.add(url); total_found_dedup += 1
        elif dedup_mode == 'title':
            seen_titles = []; seen_urls_for_title_dedup = set()
            for engine in engine_order:
                 if engine in all_results:
                    for res in all_results[engine]:
                        title = res.get('title', '').lower().strip(); url = res.get('url')
                        if not title or (url and url in seen_urls_for_title_dedup): continue
                        # Handle raw link special case from original cli if needed
                        is_duplicate = False; matcher = difflib.SequenceMatcher(None, "", title)
                        for seen_title in seen_titles:
                            matcher.set_seq1(seen_title)
                            if not seen_title or not title: continue
                            try:
                                if matcher.ratio() >= similarity_threshold: is_duplicate = True; break
                            except Exception as e: search_logger.warning(f"Error comparing titles: {e}")
                        if not is_duplicate:
                            deduplicated_results[engine].append(res); seen_titles.append(title)
                            if url: seen_urls_for_title_dedup.add(url)
                            total_found_dedup += 1
        else: # No deduplication
             deduplicated_results = all_results; total_found_dedup = sum(len(v) for v in all_results.values())
        search_logger.info(f"Deduplication complete. Found {total_found_dedup} unique results.")


        # --- Download Content ---
        if download:
            search_logger.info(f"Starting downloads (Saving to: {download_dir})...")
            urls_to_download = set()
            download_tasks = [] # List of (url, prefix) tuples

            for engine, results_list in deduplicated_results.items():
                 for i, res in enumerate(results_list):
                     main_url = res.get('url'); pdf_url = res.get('pdf_url')
                     prefix = sanitize_filename(res.get('title', f'result_{engine}_{i}')) or f'download_{engine}_{i}'

                     # Add main URL task if valid and not already added
                     if main_url and main_url.startswith('http') and main_url not in urls_to_download:
                         urls_to_download.add(main_url); download_tasks.append((main_url, prefix))
                     # Add PDF URL task if valid and not already added
                     if pdf_url and pdf_url.startswith('http') and pdf_url not in urls_to_download:
                         urls_to_download.add(pdf_url); download_tasks.append((pdf_url, f"{prefix}_pdf"))

            search_logger.info(f"Found {len(download_tasks)} unique URLs/PDFs to attempt download.")
            download_success_count = 0
            for i, (url, file_prefix) in enumerate(download_tasks):
                 search_logger.info(f"Downloading item {i+1}/{len(download_tasks)}: {url}")
                 # Call the synchronous download utility
                 if download_content_sync(url, download_dir, filename_prefix=file_prefix):
                     download_success_count += 1
                 # Add delay between downloads to be polite
                 time.sleep(random.uniform(1.0, 2.0))

            search_logger.info(f"Downloads finished ({download_success_count}/{len(download_tasks)} successful).")

        # --- Log Final Outcome ---
        outcome_summary = f"Web search completed. Found {total_found_dedup} unique results."
        if download: outcome_summary += f" Attempted {len(download_tasks)} downloads ({download_success_count} successful)."

        db = SessionLocal() # New session for final log
        try:
             add_interaction(db, session_id=session_id, mode="chat", input_type="log_info",
                             user_input=f"[Web Search Task Complete: {query[:100]}...]",
                             llm_response=outcome_summary
                            )
        finally: db.close()
        search_logger.success("Search and download task finished.")
    
    async def _trigger_web_search(self, db: Session, session_id: str, query: str) -> str:
        """
        Launches the internal _run_search_and_download_sync method in a separate thread
        to perform web search and download results asynchronously from the main flow.
        Returns an immediate confirmation message.
        """
        req_id = f"searchtrigger-{uuid.uuid4()}"
        logger.info(f"🚀 {req_id} Triggering internal background web search task for query: '{query}'")

        # --- Default settings for the search ---
        num_results_per_engine = 7 # Or get from config
        timeout_per_engine = 20    # Or get from config
        # Use all implemented engines by default
        # Note: Filter this list based on which _scrape_ methods you actually implemented!
        engines_to_use = ['ddg', 'google'] # Add other implemented keys: 'sem', 'scholar', 'base', 'core', 'scigov', 'baidu', 'refseek', 'scidirect', 'mdpi', 'tandf', 'ieee', 'springer'
        download = True # Always download for this integration
        download_dir_path = os.path.abspath(SEARCH_DOWNLOAD_DIR) # Use constant
        dedup_mode = 'url' # Default deduplication
        similarity_threshold = 0.8 # For title deduplication if used

        # Ensure download directory exists (synchronous check okay here before background task)
        try:
            os.makedirs(download_dir_path, exist_ok=True)
            logger.info(f"{req_id} Ensured download directory exists: {download_dir_path}")
        except OSError as e:
            logger.error(f"{req_id} Failed to create download directory '{download_dir_path}': {e}")
            # Log failure to DB
            add_interaction(db, session_id=session_id, mode="chat", input_type="log_error", user_input="Web Search Trigger Failed", llm_response=f"Cannot create download dir: {e}")
            return f"Error: Could not create the directory needed for search results ('{os.path.basename(download_dir_path)}')."

        # Log the initiation of the search action
        add_interaction(
            db, session_id=session_id, mode="chat", input_type="log_info",
            user_input="Web Search Action Triggered",
            llm_response=f"Query: '{query}'. Engines: {engines_to_use}. Results -> '{download_dir_path}'",
            assistant_action_type="search_web",
            assistant_action_params=json.dumps({"query": query, "engines": engines_to_use}),
            assistant_action_executed=True, # Mark as launched
            assistant_action_result="[Search process launched in background]"
        )
        db.flush() # Commit this log before returning

        # --- Schedule Background Task ---
        try:
            logger.info(f"{req_id} Scheduling internal search/download task in background thread...")
            # Get the current running event loop
            loop = asyncio.get_running_loop()
            # Schedule the SYNCHRONOUS function to run in the loop's default executor (ThreadPoolExecutor)
            # This prevents the blocking Selenium/requests code from stalling the main async loop
            loop.create_task(
                asyncio.to_thread(
                    self._run_search_and_download_sync, # Target synchronous function
                    # Pass arguments needed by the sync function
                    query, session_id, num_results_per_engine, timeout_per_engine,
                    engines_to_use, download, download_dir_path, dedup_mode, similarity_threshold
                )
            )
            logger.info(f"{req_id} Internal search/download task scheduled.")

            # --- Immediate Return ---
            return f"Okay, I've started a web search for '{query}' in the background. Relevant findings will be downloaded."

        except Exception as e:
            logger.error(f"{req_id} Error scheduling search task: {e}")
            logger.exception(f"{req_id} Scheduling Traceback:")
            # Log failure to DB
            add_interaction(db, session_id=session_id, mode="chat", input_type="log_error", user_input="Web Search Trigger Failed", llm_response=f"Failed to schedule background task: {e}")
            return f"Error: Failed to start the web search background process ({type(e).__name__})."
    


    def _check_for_captcha(self, driver: WebDriver):
        """Checks for common CAPTCHA indicators and pauses if found."""
        # Use specific logger
        captcha_logger = logger.bind(task="captcha_check")
        captcha_detected = False
        # Increase wait slightly?
        wait_time = 3
        try:
            # Check common iframe indicators first (less likely to raise immediate timeout)
            captcha_iframes = driver.find_elements(By.CSS_SELECTOR, "iframe[title*='captcha'], iframe[src*='hcaptcha'], iframe[src*='recaptcha']")
            if captcha_iframes: captcha_logger.warning("CAPTCHA iframe detected."); captcha_detected = True

            # Check specific site elements/URLs after brief wait
            body = WebDriverWait(driver, wait_time).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            if "google.com/sorry/" in driver.current_url: captcha_logger.warning("Google 'sorry' page detected."); captcha_detected = True
            if driver.find_elements(By.ID, "gs_captcha_f"): captcha_logger.warning("Google Scholar CAPTCHA form detected."); captcha_detected = True
            # Add other site-specific checks here if needed

        except TimeoutException:
             captcha_logger.debug(f"No CAPTCHA indicators found within {wait_time}s.")
             pass # No CAPTCHA found within timeout is normal
        except WebDriverException as e:
             captcha_logger.error(f"Error checking for CAPTCHA: {e}")
             # Don't pause if check fails, but log the error

        if captcha_detected:
            captcha_logger.critical("CAPTCHA DETECTED. Manual intervention required in browser window.")
            # This part is tricky for a background process. Ideally, it should signal failure.
            # For now, we'll just log and return True, assuming it cannot be solved automatically.
            # In a real unattended system, you'd likely use anti-captcha services or stop.
            # input("[?] CAPTCHA detected. Please solve it... ") # Cannot use input() in background
            return True
        return False

    def _extract_pdf_link(self, block_element: WebElement) -> str | None:
        """Attempts to find a direct PDF link within a result block element."""
        pdf_logger = logger.bind(task="pdf_extract")
        # Prioritize common direct PDF link patterns
        # Look for links ending in .pdf, containing /pdf/, or with specific text
        selectors = [
            'a[href$=".pdf"]',                 # Ends with .pdf
            'a[href*=".pdf?"]',                # Ends with .pdf?params...
            'a[href*="/pdf"]',                 # Contains /pdf/ path part
            'a[href*="/content/pdf"]',         # Common pattern
            'div.gs_ggsd a',                 # Google Scholar specific PDF link div
            'a.pdf-download-link',           # Example class name
            'a:contains("[PDF]")',           # Link containing text [PDF] (case-insensitive via JS usually)
            'a:contains("Download PDF")',      # Link containing text Download PDF
            'a:contains("Full text PDF")'     # Link containing text Full text PDF
        ]

        # Try selectors first
        for selector in selectors[:6]: # Prioritize direct href checks
            try:
                pdf_link_tag = block_element.find_element(By.CSS_SELECTOR, selector)
                pdf_href = pdf_link_tag.get_attribute('href')
                # Basic validation
                if pdf_href and pdf_href.startswith('http') and ('javascript:' not in pdf_href.lower()):
                    pdf_logger.debug(f"Found potential PDF link via selector '{selector}': {pdf_href}")
                    # Stronger check if it actually points to a PDF file type if possible
                    if pdf_href.lower().endswith('.pdf') or '.pdf?' in pdf_href.lower() or '/pdf' in pdf_href.lower():
                        return pdf_href
                    else:
                        pdf_logger.trace(f"Ignoring link from selector '{selector}' as it doesn't look like PDF: {pdf_href}")
            except NoSuchElementException:
                continue # Try next selector
            except InvalidSelectorException:
                pdf_logger.warning(f"Invalid PDF selector used: {selector}")
            except Exception as e:
                pdf_logger.warning(f"Error extracting PDF link via selector '{selector}': {e}")

        # Fallback: Check all links within the block by text content or path
        try:
            all_links = block_element.find_elements(By.TAG_NAME, 'a')
            for link in all_links:
                try:
                    link_text = link.text.lower().strip()
                    pdf_href = link.get_attribute('href')

                    if pdf_href and pdf_href.startswith('http') and ('javascript:' not in pdf_href.lower()):
                        # Check common PDF indicators in text or URL path
                        is_pdf_link = (
                            pdf_href.lower().endswith('.pdf') or
                            '.pdf?' in pdf_href.lower() or
                            '/pdf' in pdf_href.lower() or
                            '/download' in pdf_href.lower() or # Common download path
                            "[pdf]" in link_text or
                            "download pdf" in link_text or
                            "full text pdf" in link_text or
                            "view pdf" in link_text
                        )
                        if is_pdf_link:
                            pdf_logger.debug(f"Found potential PDF link via fallback check: {pdf_href}")
                            return pdf_href
                except Exception as inner_e:
                    pdf_logger.trace(f"Error checking individual link in fallback: {inner_e}")
                    continue # Skip this link if error occurs
        except Exception as e:
            pdf_logger.warning(f"Error during fallback PDF link check: {e}")

        return None # No PDF link found

    # --- Individual Scraper Methods ---

    def _scrape_duckduckgo(self, driver: WebDriver, query, num_results, timeout, max_pages=1):
        """Scrapes DuckDuckGo using Selenium, supporting pagination."""
        engine_name = "DuckDuckGo"
        scraper_logger = logger.bind(scraper=engine_name)
        scraper_logger.info(f"Starting search for '{query}' (Max Pages: {max_pages})...")
        results = []
        search_url = f"https://duckduckgo.com/?q={quote_plus(query)}&ia=web"
        processed_urls = set()

        for page_num in range(max_pages):
            scraper_logger.info(f"Processing page {page_num + 1}...")
            if page_num > 0: # Try to load more results
                try:
                    # DDG uses dynamically loaded results, wait for a known static element or timeout
                    WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.ID, "search_form_input_homepage"))) # Wait for search bar again?
                    more_results_button = driver.find_element(By.ID, "more-results")
                    # Scroll button into view and click using JavaScript
                    driver.execute_script("arguments[0].scrollIntoView(true);", more_results_button)
                    time.sleep(0.5) # Brief pause before click
                    driver.execute_script("arguments[0].click();", more_results_button)
                    time.sleep(1.5) # Wait for results to potentially load after click
                    scraper_logger.info(f"Clicked 'More results' for page {page_num + 1}.")
                except (NoSuchElementException, TimeoutException):
                    scraper_logger.info(f"No 'More results' button found or timed out. Stopping pagination.")
                    break
                except Exception as e:
                     scraper_logger.error(f"Error clicking 'More results': {e}. Stopping pagination.")
                     break
            else: # First page navigation
                try:
                    scraper_logger.info(f"Navigating to {search_url}")
                    driver.get(search_url)
                    # Wait for a stable element indicating results might be present
                    WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#links, .results--main")))
                    scraper_logger.info(f"Page loaded.")
                except TimeoutException: scraper_logger.error(f"Timed out waiting for page content. Aborting."); return results
                except WebDriverException as e: scraper_logger.error(f"Error navigating to {search_url}: {e}"); return results

            if self._check_for_captcha(driver): # Use self._
                 scraper_logger.error("CAPTCHA detected. Cannot proceed automatically.")
                 return results # Abort if CAPTCHA needed

            # --- Parse Results ---
            page_results_found = 0
            try:
                # Refresh result blocks search on each page/after load
                result_blocks = driver.find_elements(By.CSS_SELECTOR, "article[data-testid='result']")
                scraper_logger.info(f"Page {page_num + 1}: Found {len(result_blocks)} potential result blocks.")

                for block in result_blocks:
                    if len(results) >= num_results: break
                    title, url, snippet, pdf_url = None, None, None, None
                    try:
                        # Extract elements, handle potential NoSuchElementException for each part
                        title_tag = block.find_element(By.CSS_SELECTOR, "h2 a span")
                        link_tag = block.find_element(By.CSS_SELECTOR, "div[data-testid='result-extras-url'] a")
                        snippet_tag = block.find_element(By.CSS_SELECTOR, "div[data-testid='result-extras'] span")

                        url = link_tag.get_attribute('href')
                        title = title_tag.text.strip()
                        snippet = snippet_tag.text.strip()

                        if not url or not title or url in processed_urls: continue

                        pdf_url = self._extract_pdf_link(block) # Use self._

                    except NoSuchElementException:
                         scraper_logger.warning(f"Page {page_num + 1}: Skipping block, missing expected element.")
                         continue # Skip this block if essential parts missing
                    except Exception as e:
                         scraper_logger.error(f"Page {page_num + 1}: Error parsing result block: {e}. Skipping.")
                         continue

                    # Append valid result
                    result_data = {'title': title, 'url': url, 'snippet': snippet if snippet else "N/A"}
                    if pdf_url: result_data['pdf_url'] = pdf_url
                    results.append(result_data)
                    processed_urls.add(url)
                    page_results_found += 1

                scraper_logger.info(f"Page {page_num + 1}: Added {page_results_found} results this page.")
                if len(results) >= num_results: scraper_logger.info(f"Reached target results ({num_results})."); break
                # Check if 'more results' exists for pagination decision
                if page_num < max_pages - 1:
                    try: driver.find_element(By.ID, "more-results")
                    except NoSuchElementException: scraper_logger.info("No 'More results' button found for next page."); break

            except WebDriverException as e: scraper_logger.error(f"Error finding result blocks on page {page_num + 1}: {e}"); break

        scraper_logger.info(f"Finished scraping. Total results: {len(results)}")
        return results

    def _scrape_google(self, driver: WebDriver, query, num_results, timeout, max_pages=1):
        """Scrapes Google using Selenium. Supports pagination. Includes fallback."""
        engine_name = "Google"
        scraper_logger = logger.bind(scraper=engine_name)
        scraper_logger.info(f"Starting search for '{query}' (Max Pages: {max_pages})...")
        results = []
        search_url_base = "https://www.google.com/search"
        results_per_page = 10 # Google usually shows 10
        processed_urls = set()
        result_selectors = ["div.kvH3mc", "div.MjjYud", "div.g", "div.Gx5Zad.fP1Qef.xpd.EtOod.pkphOe"] # Common result block divs
        wait_container_selector = "#search" # Wait for main search container

        for page_num in range(max_pages):
            current_start = page_num * results_per_page
            search_url = f"{search_url_base}?q={quote_plus(query)}&num={results_per_page}&start={current_start}&hl=en" # Force English
            scraper_logger.info(f"Processing page {page_num + 1} (start={current_start})...")

            try:
                scraper_logger.info(f"Navigating to {search_url}")
                driver.get(search_url)
                WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_container_selector)))
                scraper_logger.info(f"Page loaded.")
            except TimeoutException: scraper_logger.error(f"Timed out waiting for page content ({wait_container_selector}). Aborting page."); break
            except WebDriverException as e: scraper_logger.error(f"Error navigating to {search_url}: {e}"); break

            if self._check_for_captcha(driver): # Use self._
                scraper_logger.error("CAPTCHA detected. Cannot proceed automatically.")
                return results # Abort if CAPTCHA needed

            # --- Parse Results ---
            page_results_found = 0
            result_blocks = []
            used_selector = "None"

            try:
                # Find result blocks using the list of selectors
                for selector in result_selectors:
                    try:
                        result_blocks = driver.find_elements(By.CSS_SELECTOR, selector)
                        if result_blocks:
                            used_selector = selector
                            scraper_logger.info(f"Page {page_num + 1}: Found {len(result_blocks)} potential blocks using '{used_selector}'.")
                            break # Use the first selector that yields results
                    except InvalidSelectorException:
                        scraper_logger.warning(f"Invalid selector '{selector}', skipping.")
                        continue

                if not result_blocks:
                    # Check for explicit "no results" message
                    page_text = driver.find_element(By.TAG_NAME, 'body').text
                    if "did not match any documents" in page_text or "No results found for" in page_text:
                        scraper_logger.info(f"Page {page_num + 1}: 'No results found' message detected.")
                    else:
                        scraper_logger.warning(f"Page {page_num + 1}: No result blocks found using any primary selector.")
                    # Don't try raw link extraction here, too noisy for Google
                    break # Stop pagination if no results found

                # Process found blocks
                for block in result_blocks:
                    if len(results) >= num_results: break
                    title, url, snippet, pdf_url = None, None, None, None
                    try:
                        # Extract link first
                        link_tag = block.find_element(By.CSS_SELECTOR, 'a[href]')
                        url = link_tag.get_attribute('href')
                        if not url or url.startswith('#') or "google.com" in urlparse(url).netloc: continue # Skip internal/invalid links

                        # Clean Google redirect URLs
                        if url.startswith('/url?q='):
                            try: url = parse_qs(urlparse(url).query)['q'][0]
                            except (KeyError, IndexError): pass # Keep original if parsing fails

                        if not url.startswith('http') or url in processed_urls: continue # Skip relative or duplicate URLs

                        # Extract title
                        try: h3_tag = block.find_element(By.CSS_SELECTOR, 'h3') ; title = h3_tag.text.strip()
                        except NoSuchElementException: title = "No Title Found"

                        # Extract snippet (try multiple common selectors)
                        try: snippet_div = block.find_element(By.CSS_SELECTOR, 'div.VwiC3b, div.Uroaid, div.s, div.gGQDAb, div[data-sncf="1"], span.aCOpRe span')
                        except NoSuchElementException:
                             try: # Fallback: get all text in block minus title
                                 all_text = block.text; snippet = all_text.replace(title, '').strip() if title != "No Title Found" else all_text
                             except Exception: snippet = None
                        else: snippet = snippet_div.text.strip() if snippet_div else None

                        if not title: continue # Skip if title is empty

                        pdf_url = self._extract_pdf_link(block) # Use self._

                    except NoSuchElementException:
                         # Sometimes blocks are just ads or featured snippets without standard links/titles
                         scraper_logger.trace(f"Page {page_num + 1}: Skipping block, missing core elements (likely not a standard result).")
                         continue
                    except Exception as e:
                         scraper_logger.error(f"Page {page_num + 1}: Error parsing block with selector '{used_selector}': {e}. Skipping.")
                         continue

                    # Append valid result
                    result_data = {'title': title, 'url': url, 'snippet': snippet if snippet else "N/A"}
                    if pdf_url: result_data['pdf_url'] = pdf_url
                    results.append(result_data)
                    processed_urls.add(url)
                    page_results_found += 1

                scraper_logger.info(f"Page {page_num + 1}: Added {page_results_found} structured results.")

                # --- Pagination Check ---
                if len(results) >= num_results: scraper_logger.info(f"Reached target results ({num_results})."); break
                if result_blocks and page_num < max_pages - 1:
                    try: driver.find_element(By.CSS_SELECTOR, 'a#pnnext, a[aria-label="Next page"]')
                    except NoSuchElementException: scraper_logger.info(f"Page {page_num + 1}: No 'Next' link found."); break

            except WebDriverException as e: scraper_logger.error(f"WebDriver error during parsing on page {page_num + 1}: {e}"); break

        scraper_logger.info(f"Finished scraping. Total results: {len(results)}")
        return results
    
    # --- Add other _scrape_... methods here, converted similarly ---
    # _scrape_searx, _scrape_semantic_scholar, _scrape_google_scholar, etc.
    # Remember to:
    #   - Add self parameter
    #   - Replace print with logger.bind(scraper=...).info/warning/error
    #   - Call helpers using self._check_for_captcha / self._extract_pdf_link
    #   - Adapt selectors and logic as needed based on the original scrapers.py
    #   - Return results list

    # Placeholder for remaining scrapers - IMPLEMENT THESE
    def _scrape_searx(self, driver: WebDriver, instance_url: str, query: str, num_results: int, timeout: int):
        """Scrapes a SearXNG instance using Selenium."""
        engine_name = "SearxNG" # More specific name
        scraper_logger = logger.bind(scraper=engine_name)
        scraper_logger.info(f"Starting search on instance '{instance_url}' for '{query}'...")
        results = []
        # Ensure instance URL is clean and build search URL
        search_url = f"{instance_url.rstrip('/')}/search?q={quote_plus(query)}"
        processed_urls = set()
        wait_selector = "#results, div.results-container" # Common containers

        try:
            scraper_logger.info(f"Navigating to {search_url}")
            driver.get(search_url)
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector)))
            scraper_logger.info(f"Page loaded.")
        except TimeoutException: scraper_logger.error(f"Timed out waiting for page content ({wait_selector}). Aborting."); return results
        except WebDriverException as e: scraper_logger.error(f"Error navigating to {search_url}: {e}"); return results

        # No automatic CAPTCHA handling for SearX usually needed, but keep the check just in case
        if self._check_for_captcha(driver):
             scraper_logger.error("CAPTCHA detected on SearX instance. Cannot proceed automatically.")
             return results

        try:
            # Common selectors across SearXNG themes
            result_blocks = driver.find_elements(By.CSS_SELECTOR, 'div.result, article.result, div.result-default')
            scraper_logger.info(f"Found {len(result_blocks)} potential result blocks.")

            for block in result_blocks:
                if len(results) >= num_results: break
                title, url, snippet, pdf_url = None, None, None, None
                try:
                    # Extract elements; SearXNG structure can vary slightly by theme
                    link_tag = block.find_element(By.CSS_SELECTOR, 'a[href]') # Usually the main link
                    title_tag = block.find_element(By.CSS_SELECTOR, 'h3 > a, h4 > a, h3, h4, .result-title a, .title a') # More title selectors
                    # Snippet selectors
                    try: snippet_tag = block.find_element(By.CSS_SELECTOR, 'p.description, p.content, div.snippet, div.description, p.result-content')
                    except NoSuchElementException: snippet_tag = None

                    url = link_tag.get_attribute('href')
                    title = title_tag.text.strip()
                    snippet = snippet_tag.text.strip() if snippet_tag else None

                    # Handle relative URLs sometimes found in SearXNG instances
                    if url and not urlparse(url).scheme: url = urljoin(instance_url, url)

                    if not url or not title or not url.startswith('http') or url in processed_urls: continue

                    pdf_url = self._extract_pdf_link(block)

                except NoSuchElementException:
                     scraper_logger.warning("Skipping block, missing expected elements (title/link).")
                     continue
                except Exception as e:
                     scraper_logger.error(f"Error parsing result block: {e}. Skipping.")
                     continue

                # Append valid result
                result_data = {'title': title, 'url': url, 'snippet': snippet if snippet else "N/A"}
                if pdf_url: result_data['pdf_url'] = pdf_url
                results.append(result_data)
                processed_urls.add(url)

        except WebDriverException as e: scraper_logger.error(f"Error finding result blocks: {e}")

        scraper_logger.info(f"Finished scraping. Total results: {len(results)}")
        return results

    def _scrape_semantic_scholar(self, driver: WebDriver, query: str, num_results: int, timeout: int):
        """Scrapes Semantic Scholar using Selenium."""
        engine_name = "SemanticScholar"
        scraper_logger = logger.bind(scraper=engine_name)
        scraper_logger.info(f"Starting search for '{query}'...")
        results = []
        search_url = f"https://www.semanticscholar.org/search?q={quote_plus(query)}&sort=relevance"
        processed_urls = set()
        wait_selector = "#main-content, div[data-test-id='search-result-list']" # Wait for main content area or result list

        try:
            scraper_logger.info(f"Navigating to {search_url}")
            driver.get(search_url)
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector)))
            scraper_logger.info(f"Page loaded.")
        except TimeoutException: scraper_logger.error(f"Timed out waiting for page content ({wait_selector}). Aborting."); return results
        except WebDriverException as e: scraper_logger.error(f"Error navigating to {search_url}: {e}"); return results

        if self._check_for_captcha(driver):
             scraper_logger.error("CAPTCHA detected. Cannot proceed automatically.")
             return results

        try:
            # Selector for result cards
            result_blocks = driver.find_elements(By.CSS_SELECTOR, 'div[data-test-id="search-result-card"], div.search-result--compact, div.search-result')
            scraper_logger.info(f"Found {len(result_blocks)} potential result blocks.")

            for block in result_blocks:
                if len(results) >= num_results: break
                title, url, snippet, pdf_url = None, None, None, None
                try:
                    # Extract title and link
                    title_link_tag = block.find_element(By.CSS_SELECTOR, 'a[data-test-id="title-link"], h3 > a, a[data-heap-id="result-title"]')
                    url = title_link_tag.get_attribute('href')
                    title = title_link_tag.text.strip()

                    # Extract snippet
                    try: snippet_tag = block.find_element(By.CSS_SELECTOR, 'span[data-test-id="text-truncator-abstract"], span.abstract-truncator, div.abstract')
                    except NoSuchElementException: snippet_tag = None
                    snippet = snippet_tag.text.strip() if snippet_tag else None

                    # Resolve relative URLs and check validity
                    if url and url.startswith('/'): url = urljoin("https://www.semanticscholar.org/", url)
                    if not url or not title or not url.startswith('http') or url in processed_urls: continue

                    pdf_url = self._extract_pdf_link(block)

                except NoSuchElementException:
                     scraper_logger.warning("Skipping block, missing expected elements (title/link).")
                     continue
                except Exception as e:
                     scraper_logger.error(f"Error parsing result block: {e}. Skipping.")
                     continue

                # Append valid result
                result_data = {'title': title, 'url': url, 'snippet': snippet if snippet else "N/A"}
                if pdf_url: result_data['pdf_url'] = pdf_url
                results.append(result_data)
                processed_urls.add(url)

        except WebDriverException as e: scraper_logger.error(f"Error finding result blocks: {e}")

        scraper_logger.info(f"Finished scraping. Total results: {len(results)}")
        return results

    def _scrape_google_scholar(self, driver: WebDriver, query: str, num_results: int, timeout: int, max_pages: int = 1):
        """Scrapes Google Scholar using Selenium. Highly unstable. Supports pagination."""
        engine_name = "GoogleScholar"
        scraper_logger = logger.bind(scraper=engine_name)
        scraper_logger.info(f"Starting search for '{query}' (Max Pages: {max_pages})...")
        results = []
        search_url_base = "https://scholar.google.com/scholar"
        results_per_page = 10
        processed_urls = set()
        wait_container_selector = "#gs_res_ccl_mid" # Container for results

        for page_num in range(max_pages):
            current_start = page_num * results_per_page
            search_url = f"{search_url_base}?hl=en&q={quote_plus(query)}&num={results_per_page}&start={current_start}"
            scraper_logger.info(f"Processing page {page_num + 1} (start={current_start})...")

            try:
                scraper_logger.info(f"Navigating to {search_url}")
                driver.get(search_url)
                WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_container_selector)))
                scraper_logger.info(f"Page loaded.")
            except TimeoutException: scraper_logger.error(f"Timed out waiting for page content ({wait_container_selector}). Aborting page."); break
            except WebDriverException as e: scraper_logger.error(f"Error navigating to {search_url}: {e}"); break

            if self._check_for_captcha(driver):
                scraper_logger.error("CAPTCHA detected. Cannot proceed automatically.")
                return results # Abort

            # --- Parse Results ---
            page_results_found = 0
            try:
                result_blocks = driver.find_elements(By.CSS_SELECTOR, 'div.gs_r.gs_or.gs_scl')
                scraper_logger.info(f"Page {page_num + 1}: Found {len(result_blocks)} potential result blocks.")

                if not result_blocks and page_num == 0: # Check for no results message on first page only
                     page_text = driver.find_element(By.TAG_NAME, 'body').text
                     if "did not match any articles" in page_text: scraper_logger.info("Page 1: 'No results found' message detected.")
                     else: scraper_logger.warning("Page 1: No result blocks found.")

                for block in result_blocks:
                    if len(results) >= num_results: break
                    title, url, snippet, pdf_url = None, None, None, None
                    try:
                        # Extract elements
                        title_link_tag = block.find_element(By.CSS_SELECTOR, 'h3.gs_rt a')
                        url = title_link_tag.get_attribute('href')
                        title = title_link_tag.text.strip()

                        try: snippet_tag = block.find_element(By.CSS_SELECTOR, 'div.gs_rs')
                        except NoSuchElementException: snippet_tag = None
                        snippet = snippet_tag.text.strip() if snippet_tag else None

                        if not url or not title or not url.startswith('http') or url in processed_urls: continue

                        pdf_url = self._extract_pdf_link(block)

                    except NoSuchElementException:
                         scraper_logger.warning("Skipping block, missing expected elements (title/link).")
                         continue
                    except Exception as e:
                         scraper_logger.error(f"Error parsing result block: {e}. Skipping.")
                         continue

                    # Append valid result
                    result_data = {'title': title, 'url': url, 'snippet': snippet if snippet else "N/A"}
                    if pdf_url: result_data['pdf_url'] = pdf_url
                    results.append(result_data)
                    processed_urls.add(url)
                    page_results_found += 1

                scraper_logger.info(f"Page {page_num + 1}: Added {page_results_found} results.")

                # --- Pagination Check ---
                if len(results) >= num_results: scraper_logger.info(f"Reached target results ({num_results})."); break
                if result_blocks and page_num < max_pages - 1: # Only check if we found results this page
                    try: driver.find_element(By.LINK_TEXT, 'Next')
                    except NoSuchElementException: scraper_logger.info(f"Page {page_num + 1}: No 'Next' link found."); break

            except WebDriverException as e: scraper_logger.error(f"Error finding/parsing result blocks on page {page_num + 1}: {e}"); break

        scraper_logger.info(f"Finished scraping. Total results: {len(results)}")
        return results

    def _scrape_base(self, driver: WebDriver, query: str, num_results: int, timeout: int):
        """Scrapes BASE (Bielefeld Academic Search Engine) using Selenium."""
        engine_name = "BASE"
        scraper_logger = logger.bind(scraper=engine_name)
        scraper_logger.info(f"Starting search for '{query}'...")
        results = []
        search_url = f"https://www.base-search.net/Search/Results?lookfor={quote_plus(query)}&limit={num_results}&sort=relevant"
        processed_urls = set()
        wait_selector = "#results" # Main results container

        try:
            scraper_logger.info(f"Navigating to {search_url}")
            driver.get(search_url)
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector)))
            scraper_logger.info(f"Page loaded.")
        except TimeoutException: scraper_logger.error(f"Timed out waiting for page content ({wait_selector}). Aborting."); return results
        except WebDriverException as e: scraper_logger.error(f"Error navigating to {search_url}: {e}"); return results

        if self._check_for_captcha(driver):
             scraper_logger.error("CAPTCHA detected. Cannot proceed automatically.")
             return results

        try:
            result_blocks = driver.find_elements(By.CSS_SELECTOR, 'div.record')
            scraper_logger.info(f"Found {len(result_blocks)} potential result blocks.")

            for block in result_blocks:
                if len(results) >= num_results: break
                title, url, snippet, pdf_url = None, None, None, None
                try:
                    # Extract elements
                    title_link_tag = block.find_element(By.CSS_SELECTOR, 'a.title')
                    url = title_link_tag.get_attribute('href')
                    title = title_link_tag.text.strip()

                    try: snippet_tag = block.find_element(By.CSS_SELECTOR, 'div.abstract')
                    except NoSuchElementException: snippet_tag = None
                    snippet = snippet_tag.text.strip() if snippet_tag else None

                    if not url or not title or not url.startswith('http') or url in processed_urls: continue

                    pdf_url = self._extract_pdf_link(block)

                except NoSuchElementException:
                     scraper_logger.warning("Skipping block, missing expected elements (title/link).")
                     continue
                except Exception as e:
                     scraper_logger.error(f"Error parsing result block: {e}. Skipping.")
                     continue

                # Append valid result
                result_data = {'title': title, 'url': url, 'snippet': snippet if snippet else "N/A"}
                if pdf_url: result_data['pdf_url'] = pdf_url
                results.append(result_data)
                processed_urls.add(url)

        except WebDriverException as e: scraper_logger.error(f"Error finding result blocks: {e}")

        scraper_logger.info(f"Finished scraping. Total results: {len(results)}")
        return results

    def _scrape_core(self, driver: WebDriver, query: str, num_results: int, timeout: int):
        """Scrapes CORE (core.ac.uk) using Selenium."""
        engine_name = "CORE"
        scraper_logger = logger.bind(scraper=engine_name)
        scraper_logger.info(f"Starting search for '{query}'...")
        results = []
        search_url = f"https://core.ac.uk/search?q={quote_plus(query)}"
        processed_urls = set()
        # Wait for results list or main content area
        wait_selector = "ul[class*='StyledList'], div.content, ul.results-list"

        try:
            scraper_logger.info(f"Navigating to {search_url}")
            driver.get(search_url)
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector)))
            scraper_logger.info(f"Page loaded.")
        except TimeoutException: scraper_logger.error(f"Timed out waiting for page content ({wait_selector}). Aborting."); return results
        except WebDriverException as e: scraper_logger.error(f"Error navigating to {search_url}: {e}"); return results

        if self._check_for_captcha(driver):
             scraper_logger.error("CAPTCHA detected. Cannot proceed automatically.")
             return results

        try:
            # Selectors for result items (can vary)
            result_blocks = driver.find_elements(By.CSS_SELECTOR, 'div.result-item, li.result-list-item, div[class*="styles__cardContainer"]')
            scraper_logger.info(f"Found {len(result_blocks)} potential result blocks.")

            for block in result_blocks:
                if len(results) >= num_results: break
                title, url, snippet, pdf_url = None, None, None, None
                try:
                    # Extract title and link
                    title_link_tag = block.find_element(By.CSS_SELECTOR, 'h3 > a, div[class*="title"] > a, a[data-testid="result-title"]')
                    url = title_link_tag.get_attribute('href')
                    title = title_link_tag.text.strip()

                    # Extract snippet
                    try: snippet_tag = block.find_element(By.CSS_SELECTOR, 'div.abstract, p.abstract, div[class*="abstract"]')
                    except NoSuchElementException: snippet_tag = None
                    snippet = snippet_tag.text.strip() if snippet_tag else None

                    # Resolve relative URLs and check validity
                    if url and not url.startswith('http'): url = urljoin(driver.current_url, url)
                    if not url or not title or not url.startswith('http') or url in processed_urls: continue

                    pdf_url = self._extract_pdf_link(block)

                except NoSuchElementException:
                     scraper_logger.warning("Skipping block, missing expected elements (title/link).")
                     continue
                except Exception as e:
                     scraper_logger.error(f"Error parsing result block: {e}. Skipping.")
                     continue

                # Append valid result
                result_data = {'title': title, 'url': url, 'snippet': snippet if snippet else "N/A"}
                if pdf_url: result_data['pdf_url'] = pdf_url
                results.append(result_data)
                processed_urls.add(url)

        except WebDriverException as e: scraper_logger.error(f"Error finding result blocks: {e}")

        scraper_logger.info(f"Finished scraping. Total results: {len(results)}")
        return results

    def _scrape_sciencegov(self, driver: WebDriver, query: str, num_results: int, timeout: int):
        """Scrapes Science.gov using Selenium."""
        engine_name = "ScienceGov"
        scraper_logger = logger.bind(scraper=engine_name)
        scraper_logger.info(f"Starting search for '{query}'...")
        results = []
        search_url = f"https://www.science.gov/scigov/desktop/en/results.html?q={quote_plus(query)}"
        processed_urls = set()
        wait_selector = "#resultsList" # Main results list ID

        try:
            scraper_logger.info(f"Navigating to {search_url}")
            driver.get(search_url)
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector)))
            scraper_logger.info(f"Page loaded.")
        except TimeoutException: scraper_logger.error(f"Timed out waiting for page content ({wait_selector}). Aborting."); return results
        except WebDriverException as e: scraper_logger.error(f"Error navigating to {search_url}: {e}"); return results

        if self._check_for_captcha(driver):
             scraper_logger.error("CAPTCHA detected. Cannot proceed automatically.")
             return results

        try:
            result_blocks = driver.find_elements(By.CSS_SELECTOR, 'div.result')
            scraper_logger.info(f"Found {len(result_blocks)} potential result blocks.")

            for block in result_blocks:
                if len(results) >= num_results: break
                title, url, snippet, pdf_url = None, None, None, None
                try:
                    # Extract elements
                    title_link_tag = block.find_element(By.CSS_SELECTOR, 'div.title > a')
                    url = title_link_tag.get_attribute('href')
                    title = title_link_tag.text.strip()

                    try: snippet_tag = block.find_element(By.CSS_SELECTOR, 'div.abstract')
                    except NoSuchElementException: snippet_tag = None
                    snippet = snippet_tag.text.strip() if snippet_tag else None

                    if not url or not title or not url.startswith('http') or url in processed_urls: continue

                    pdf_url = self._extract_pdf_link(block)

                except NoSuchElementException:
                     scraper_logger.warning("Skipping block, missing expected elements (title/link).")
                     continue
                except Exception as e:
                     scraper_logger.error(f"Error parsing result block: {e}. Skipping.")
                     continue

                # Append valid result
                result_data = {'title': title, 'url': url, 'snippet': snippet if snippet else "N/A"}
                if pdf_url: result_data['pdf_url'] = pdf_url
                results.append(result_data)
                processed_urls.add(url)

        except WebDriverException as e: scraper_logger.error(f"Error finding result blocks: {e}")

        scraper_logger.info(f"Finished scraping. Total results: {len(results)}")
        return results

    def _scrape_baidu_scholar(self, driver: WebDriver, query: str, num_results: int, timeout: int):
        """Scrapes Baidu Scholar (xueshu.baidu.com) using Selenium."""
        engine_name = "BaiduScholar"
        scraper_logger = logger.bind(scraper=engine_name)
        scraper_logger.info(f"Starting search for '{query}'...")
        results = []
        search_url = f"https://xueshu.baidu.com/s?wd={quote_plus(query)}&sc_f_para=sc_tasktype%3D%7BfirstSimpleSearch%7D" # Added para might help
        processed_urls = set()
        wait_selector = "#content_wrap" # Main content area ID

        try:
            scraper_logger.info(f"Navigating to {search_url}")
            driver.get(search_url)
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector)))
            scraper_logger.info(f"Page loaded.")
        except TimeoutException: scraper_logger.error(f"Timed out waiting for page content ({wait_selector}). Aborting."); return results
        except WebDriverException as e: scraper_logger.error(f"Error navigating to {search_url}: {e}"); return results

        if self._check_for_captcha(driver):
             scraper_logger.error("CAPTCHA detected. Cannot proceed automatically.")
             return results

        try:
            result_blocks = driver.find_elements(By.CSS_SELECTOR, 'div.result.sc_default_result')
            scraper_logger.info(f"Found {len(result_blocks)} potential result blocks.")

            for block in result_blocks:
                if len(results) >= num_results: break
                title, url, snippet, pdf_url = None, None, None, None
                try:
                    # Extract elements
                    title_link_tag = block.find_element(By.CSS_SELECTOR, 'h3 > a')
                    url = title_link_tag.get_attribute('href')
                    title = title_link_tag.text.strip()

                    try: snippet_tag = block.find_element(By.CSS_SELECTOR, 'div.c_abstract')
                    except NoSuchElementException: snippet_tag = None
                    snippet = snippet_tag.text.strip() if snippet_tag else None

                    if not url or not title or not url.startswith('http') or url in processed_urls: continue

                    pdf_url = self._extract_pdf_link(block)

                except NoSuchElementException:
                     scraper_logger.warning("Skipping block, missing expected elements (title/link).")
                     continue
                except Exception as e:
                     scraper_logger.error(f"Error parsing result block: {e}. Skipping.")
                     continue

                # Append valid result
                result_data = {'title': title, 'url': url, 'snippet': snippet if snippet else "N/A"}
                if pdf_url: result_data['pdf_url'] = pdf_url
                results.append(result_data)
                processed_urls.add(url)

        except WebDriverException as e: scraper_logger.error(f"Error finding result blocks: {e}")

        scraper_logger.info(f"Finished scraping. Total results: {len(results)}")
        return results

    def _scrape_refseek(self, driver: WebDriver, query: str, num_results: int, timeout: int):
        """Scrapes RefSeek (uses Google Custom Search Engine) using Selenium."""
        engine_name = "RefSeek"
        scraper_logger = logger.bind(scraper=engine_name)
        scraper_logger.info(f"Starting search for '{query}'...")
        results = []
        search_url = f"https://www.refseek.com/search?q={quote_plus(query)}"
        processed_urls = set()
        # Wait for the CSE results box to be visible
        wait_selector = "div.gsc-resultsbox-visible"

        try:
            scraper_logger.info(f"Navigating to {search_url}")
            driver.get(search_url)
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector)))
            scraper_logger.info(f"Page loaded.")
        except TimeoutException: scraper_logger.error(f"Timed out waiting for page content ({wait_selector}). Aborting."); return results
        except WebDriverException as e: scraper_logger.error(f"Error navigating to {search_url}: {e}"); return results

        if self._check_for_captcha(driver):
             scraper_logger.error("CAPTCHA detected. Cannot proceed automatically.")
             return results

        try:
            # Results are within the Google CSE structure
            result_blocks = driver.find_elements(By.CSS_SELECTOR, 'div.gsc-webResult.gsc-result')
            scraper_logger.info(f"Found {len(result_blocks)} potential result blocks.")

            for block in result_blocks:
                if len(results) >= num_results: break
                title, url, snippet, pdf_url = None, None, None, None
                try:
                    # Extract elements (CSE structure)
                    title_link_tag = block.find_element(By.CSS_SELECTOR, 'a.gs-title')
                    url = title_link_tag.get_attribute('href') # URL is direct here
                    title = title_link_tag.text.strip()

                    try: snippet_tag = block.find_element(By.CSS_SELECTOR, 'div.gs-bidi-start-align.gs-snippet')
                    except NoSuchElementException: snippet_tag = None
                    snippet = snippet_tag.text.strip() if snippet_tag else None

                    if not url or not title or not url.startswith('http') or url in processed_urls: continue

                    pdf_url = self._extract_pdf_link(block)

                except NoSuchElementException:
                     scraper_logger.warning("Skipping block, missing expected elements (title/link).")
                     continue
                except Exception as e:
                     scraper_logger.error(f"Error parsing result block: {e}. Skipping.")
                     continue

                # Append valid result
                result_data = {'title': title, 'url': url, 'snippet': snippet if snippet else "N/A"}
                if pdf_url: result_data['pdf_url'] = pdf_url
                results.append(result_data)
                processed_urls.add(url)

        except WebDriverException as e: scraper_logger.error(f"Error finding result blocks: {e}")

        scraper_logger.info(f"Finished scraping. Total results: {len(results)}")
        return results

    def _scrape_sciencedirect(self, driver: WebDriver, query: str, num_results: int, timeout: int):
        """Scrapes ScienceDirect (Elsevier) using Selenium."""
        engine_name = "ScienceDirect"
        scraper_logger = logger.bind(scraper=engine_name)
        scraper_logger.info(f"Starting search for '{query}'...")
        results = []
        search_url = f"https://www.sciencedirect.com/search?qs={quote_plus(query)}"
        processed_urls = set()
        wait_selector = "#results-list" # Wait for the results list container

        try:
            scraper_logger.info(f"Navigating to {search_url}")
            driver.get(search_url)
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector)))
            scraper_logger.info(f"Page loaded.")
        except TimeoutException: scraper_logger.error(f"Timed out waiting for page content ({wait_selector}). Aborting."); return results
        except WebDriverException as e: scraper_logger.error(f"Error navigating to {search_url}: {e}"); return results

        if self._check_for_captcha(driver):
             scraper_logger.error("CAPTCHA detected. Cannot proceed automatically.")
             return results

        try:
            result_blocks = driver.find_elements(By.CSS_SELECTOR, 'li.ResultItem')
            scraper_logger.info(f"Found {len(result_blocks)} potential result blocks.")

            for block in result_blocks:
                if len(results) >= num_results: break
                title, url, snippet, pdf_url = None, None, None, None
                try:
                    # Extract elements
                    link_tag = block.find_element(By.CSS_SELECTOR, 'a.result-list-title-link')
                    title_tag = link_tag.find_element(By.CSS_SELECTOR, 'span.title-text') # Title is inside link
                    url = link_tag.get_attribute('href')
                    title = title_tag.text.strip()

                    try: snippet_tag = block.find_element(By.CSS_SELECTOR, 'div.abstract-snippet-container div.snippet-text, div.SubType')
                    except NoSuchElementException: snippet_tag = None
                    snippet = snippet_tag.text.strip() if snippet_tag else None

                    if not url or not title or not url.startswith('http') or url in processed_urls: continue

                    pdf_url = self._extract_pdf_link(block)

                except NoSuchElementException:
                     scraper_logger.warning("Skipping block, missing expected elements (title/link).")
                     continue
                except Exception as e:
                     scraper_logger.error(f"Error parsing result block: {e}. Skipping.")
                     continue

                # Append valid result
                result_data = {'title': title, 'url': url, 'snippet': snippet if snippet else "N/A"}
                if pdf_url: result_data['pdf_url'] = pdf_url
                results.append(result_data)
                processed_urls.add(url)

        except WebDriverException as e: scraper_logger.error(f"Error finding result blocks: {e}")

        scraper_logger.info(f"Finished scraping. Total results: {len(results)}")
        return results

    def _scrape_mdpi(self, driver: WebDriver, query: str, num_results: int, timeout: int):
        """Scrapes MDPI using Selenium."""
        engine_name = "MDPI"
        scraper_logger = logger.bind(scraper=engine_name)
        scraper_logger.info(f"Starting search for '{query}'...")
        results = []
        search_url = f"https://www.mdpi.com/search?q={quote_plus(query)}"
        processed_urls = set()
        wait_selector = "div.article-items" # Container for article results

        try:
            scraper_logger.info(f"Navigating to {search_url}")
            driver.get(search_url)
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector)))
            scraper_logger.info(f"Page loaded.")
        except TimeoutException: scraper_logger.error(f"Timed out waiting for page content ({wait_selector}). Aborting."); return results
        except WebDriverException as e: scraper_logger.error(f"Error navigating to {search_url}: {e}"); return results

        if self._check_for_captcha(driver):
             scraper_logger.error("CAPTCHA detected. Cannot proceed automatically.")
             return results

        try:
            result_blocks = driver.find_elements(By.CSS_SELECTOR, 'article.article-item')
            scraper_logger.info(f"Found {len(result_blocks)} potential result blocks.")

            for block in result_blocks:
                if len(results) >= num_results: break
                title, url, snippet, pdf_url = None, None, None, None
                try:
                    # Extract elements
                    title_link_tag = block.find_element(By.CSS_SELECTOR, 'a.title-link')
                    url = title_link_tag.get_attribute('href')
                    title = title_link_tag.text.strip()

                    try: snippet_tag = block.find_element(By.CSS_SELECTOR, 'div.abstract-full, div.abstract-content')
                    except NoSuchElementException: snippet_tag = None
                    snippet = snippet_tag.text.strip() if snippet_tag else None

                    # Resolve relative URLs and check validity
                    if url and not url.startswith('http'): url = urljoin(driver.current_url, url)
                    if not url or not title or not url.startswith('http') or url in processed_urls: continue

                    pdf_url = self._extract_pdf_link(block)

                except NoSuchElementException:
                     scraper_logger.warning("Skipping block, missing expected elements (title/link).")
                     continue
                except Exception as e:
                     scraper_logger.error(f"Error parsing result block: {e}. Skipping.")
                     continue

                # Append valid result
                result_data = {'title': title, 'url': url, 'snippet': snippet if snippet else "N/A"}
                if pdf_url: result_data['pdf_url'] = pdf_url
                results.append(result_data)
                processed_urls.add(url)

        except WebDriverException as e: scraper_logger.error(f"Error finding result blocks: {e}")

        scraper_logger.info(f"Finished scraping. Total results: {len(results)}")
        return results

    def _scrape_tandf(self, driver: WebDriver, query: str, num_results: int, timeout: int):
        """Scrapes Taylor & Francis Online using Selenium."""
        engine_name = "T&F"
        scraper_logger = logger.bind(scraper=engine_name)
        scraper_logger.info(f"Starting search for '{query}'...")
        results = []
        search_url = f"https://www.tandfonline.com/action/doSearch?AllField={quote_plus(query)}"
        processed_urls = set()
        wait_selector = "div.search-results, div.results-list" # Container selectors

        try:
            scraper_logger.info(f"Navigating to {search_url}")
            driver.get(search_url)
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector)))
            scraper_logger.info(f"Page loaded.")
        except TimeoutException: scraper_logger.error(f"Timed out waiting for page content ({wait_selector}). Aborting."); return results
        except WebDriverException as e: scraper_logger.error(f"Error navigating to {search_url}: {e}"); return results

        if self._check_for_captcha(driver):
             scraper_logger.error("CAPTCHA detected. Cannot proceed automatically.")
             return results

        try:
            result_blocks = driver.find_elements(By.CSS_SELECTOR, 'div.searchResultItem, li.search-result')
            scraper_logger.info(f"Found {len(result_blocks)} potential result blocks.")

            for block in result_blocks:
                if len(results) >= num_results: break
                title, url, snippet, pdf_url = None, None, None, None
                try:
                    # Extract elements
                    title_link_tag = block.find_element(By.CSS_SELECTOR, 'a.hlFld-Title, span.hlFld-Title > a')
                    url = title_link_tag.get_attribute('href')
                    title = title_link_tag.text.strip()

                    try: snippet_tag = block.find_element(By.CSS_SELECTOR, 'div.abstractSection.hidden, div.search-result__snippet')
                    except NoSuchElementException: snippet_tag = None
                    snippet = snippet_tag.text.strip() if snippet_tag else None

                    # Resolve relative URLs and check validity
                    if url and not url.startswith('http'): url = urljoin(driver.current_url, url)
                    if not url or not title or not url.startswith('http') or url in processed_urls: continue

                    pdf_url = self._extract_pdf_link(block)

                except NoSuchElementException:
                     scraper_logger.warning("Skipping block, missing expected elements (title/link).")
                     continue
                except Exception as e:
                     scraper_logger.error(f"Error parsing result block: {e}. Skipping.")
                     continue

                # Append valid result
                result_data = {'title': title, 'url': url, 'snippet': snippet if snippet else "N/A"}
                if pdf_url: result_data['pdf_url'] = pdf_url
                results.append(result_data)
                processed_urls.add(url)

        except WebDriverException as e: scraper_logger.error(f"Error finding result blocks: {e}")

        scraper_logger.info(f"Finished scraping. Total results: {len(results)}")
        return results

    def _scrape_ieee(self, driver: WebDriver, query: str, num_results: int, timeout: int):
        """Scrapes IEEE Xplore using Selenium. Prone to breaking due to dynamic content."""
        engine_name = "IEEE"
        scraper_logger = logger.bind(scraper=engine_name)
        scraper_logger.info(f"Starting search for '{query}'...")
        results = []
        search_url = f"https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText={quote_plus(query)}"
        processed_urls = set()
        # Wait for main content area or results list (structure varies)
        wait_selector = "#xplMainContent, div.List-results-items, section[aria-label='search results']"

        try:
            scraper_logger.info(f"Navigating to {search_url}")
            driver.get(search_url)
            # Increase wait time slightly for IEEE as it can be slow
            WebDriverWait(driver, timeout + 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector)))
            scraper_logger.info(f"Page loaded.")
        except TimeoutException: scraper_logger.error(f"Timed out waiting for page content ({wait_selector}). Aborting."); return results
        except WebDriverException as e: scraper_logger.error(f"Error navigating to {search_url}: {e}"); return results

        if self._check_for_captcha(driver):
             scraper_logger.error("CAPTCHA detected. Cannot proceed automatically.")
             return results

        try:
            # Selectors for result items (can change frequently)
            result_blocks = driver.find_elements(By.CSS_SELECTOR, 'div.List-results-items, xpl-results-item')
            scraper_logger.info(f"Found {len(result_blocks)} potential result blocks.")

            for block in result_blocks:
                if len(results) >= num_results: break
                title, url, snippet, pdf_url = None, None, None, None
                try:
                    # Extract elements (selectors might need frequent updates)
                    title_link_tag = block.find_element(By.CSS_SELECTOR, 'h2 a, h3 a, a[data-artnum]')
                    url = title_link_tag.get_attribute('href')
                    title = title_link_tag.text.strip()

                    try: snippet_tag = block.find_element(By.CSS_SELECTOR, 'div.abstract span, span.text-body-sm, div.description')
                    except NoSuchElementException: snippet_tag = None
                    snippet = snippet_tag.text.strip() if snippet_tag else None

                    # Resolve relative URLs and check validity
                    if url and not url.startswith('http'): url = urljoin(driver.current_url, url)
                    if not url or not title or not url.startswith('http') or url in processed_urls: continue

                    pdf_url = self._extract_pdf_link(block)

                except NoSuchElementException:
                     scraper_logger.warning("Skipping block, missing expected elements (title/link).")
                     continue
                except Exception as e:
                     scraper_logger.error(f"Error parsing result block: {e}. Skipping.")
                     continue

                # Append valid result
                result_data = {'title': title, 'url': url, 'snippet': snippet if snippet else "N/A"}
                if pdf_url: result_data['pdf_url'] = pdf_url
                results.append(result_data)
                processed_urls.add(url)

        except WebDriverException as e: scraper_logger.error(f"Error finding result blocks: {e}")

        scraper_logger.info(f"Finished scraping. Total results: {len(results)}")
        return results

    def _scrape_springer(self, driver: WebDriver, query: str, num_results: int, timeout: int):
        """Scrapes SpringerLink using Selenium."""
        engine_name = "Springer"
        scraper_logger = logger.bind(scraper=engine_name)
        scraper_logger.info(f"Starting search for '{query}'...")
        results = []
        search_url = f"https://link.springer.com/search?query={quote_plus(query)}"
        processed_urls = set()
        wait_selector = "#results-list, ol.app-search-results-list" # Container for results

        try:
            scraper_logger.info(f"Navigating to {search_url}")
            driver.get(search_url)
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector)))
            scraper_logger.info(f"Page loaded.")
        except TimeoutException: scraper_logger.error(f"Timed out waiting for page content ({wait_selector}). Aborting."); return results
        except WebDriverException as e: scraper_logger.error(f"Error navigating to {search_url}: {e}"); return results

        if self._check_for_captcha(driver):
             scraper_logger.error("CAPTCHA detected. Cannot proceed automatically.")
             return results

        try:
            result_blocks = driver.find_elements(By.CSS_SELECTOR, 'li.results-list__item, article.app-search-results-item')
            scraper_logger.info(f"Found {len(result_blocks)} potential result blocks.")

            for block in result_blocks:
                if len(results) >= num_results: break
                title, url, snippet, pdf_url = None, None, None, None
                try:
                    # Extract elements
                    title_link_tag = block.find_element(By.CSS_SELECTOR, 'h2 a, a.app-card-title, a[data-test="title"]')
                    url = title_link_tag.get_attribute('href')
                    title = title_link_tag.text.strip()

                    try: snippet_tag = block.find_element(By.CSS_SELECTOR, 'p.app-card-snippet, p.snippet, div.content')
                    except NoSuchElementException: snippet_tag = None
                    snippet = snippet_tag.text.strip() if snippet_tag else None

                    # Resolve relative URLs and check validity
                    if url and not url.startswith('http'): url = urljoin(driver.current_url, url)
                    if not url or not title or not url.startswith('http') or url in processed_urls: continue

                    pdf_url = self._extract_pdf_link(block)

                except NoSuchElementException:
                     scraper_logger.warning("Skipping block, missing expected elements (title/link).")
                     continue
                except Exception as e:
                     scraper_logger.error(f"Error parsing result block: {e}. Skipping.")
                     continue

                # Append valid result
                result_data = {'title': title, 'url': url, 'snippet': snippet if snippet else "N/A"}
                if pdf_url: result_data['pdf_url'] = pdf_url
                results.append(result_data)
                processed_urls.add(url)

        except WebDriverException as e: scraper_logger.error(f"Error finding result blocks: {e}")

        scraper_logger.info(f"Finished scraping. Total results: {len(results)}")
        return results

    def _cleanup_llm_output(self, text: str) -> str:
        """Removes potential log lines, extra processing messages, think tags, and leaked analysis from LLM output."""
        if not isinstance(text, str):
            logger.trace(f"Cleanup received non-str type: {type(text)}, returning as is.")
            return text

        # Pattern to match typical log lines: [HH:MM:SS.ms LEVEL] Message
        log_prefix_pattern = r"^\s*\[\d{2}:\d{2}:\d{2}(\.\d{3,6})?\s+\w*\]\s+.*\n?"
        cleaned_text = re.sub(log_prefix_pattern, '', text, flags=re.MULTILINE)

        # Pattern to remove standalone "Processing complete." or "Log stream complete." lines
        processing_complete_pattern = r"^\s*(Processing complete|Log stream complete)\.?\s*\n?"
        cleaned_text = re.sub(processing_complete_pattern, '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)

        # --- ADDED: Pattern to remove leaked Emotion/User Analysis Preamble ---
        # Looks for lines starting with common analysis phrases up to where the actual response should start
        # This might need refinement based on variations in the LLM's preamble output
        analysis_preamble_pattern = r"^(?:The user(?:'s input|\s+expressed|\s+is asking)|Analysis:|Emotional Tone:|Intent:|Context:).*\n+"
        # Use re.DOTALL? No, process line by line likely safer with MULTILINE
        # Keep removing matches until none are found at the beginning of the string
        original_len = -1
        while len(cleaned_text) != original_len: # Loop until no more changes
            original_len = len(cleaned_text)
            cleaned_text = re.sub(analysis_preamble_pattern, '', cleaned_text.lstrip(), count=1, flags=re.IGNORECASE | re.MULTILINE)
            cleaned_text = cleaned_text.lstrip() # Remove leading space after removal

        # Optional: Remove "Draft Response:" lines if they leak
        draft_response_pattern = r"^\s*(?:Draft Response|Your Final, Refined Response).*?:?\s*\n?"
        cleaned_text = re.sub(draft_response_pattern, '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        # --- END ADDED ---


        # Remove think tags just in case
        cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)

        # Remove leading/trailing whitespace that might be left
        cleaned_text = cleaned_text.strip()

        if cleaned_text != text:
            logger.warning(f"Applied cleanup to LLM output. Original len: {len(text)}, Cleaned len: {len(cleaned_text)}")
            logger.debug(f"Original Text Starts: '{text[:150]}...'")
            logger.debug(f"Cleaned Text Starts: '{cleaned_text[:150]}...'")


        return cleaned_text

    def _get_rag_retriever(self, db: Session, user_input: str, priority: int = ELP0) -> Tuple[Optional[Any], Optional[Any], str]:
        """
        Creates/gets retrievers for URL and History RAG.
        History RAG embedding uses the specified priority.
        """
        log_prefix = f"RAGRetriever|ELP{priority}|{self.current_session_id or 'NoSession'}"
        logger.debug(f"{log_prefix} Preparing RAG retrievers...")

        url_retriever = None
        history_retriever = None
        history_ids_set = set()
        self.vectorstore_history = None # Clear previous request's history store

        # --- URL Retriever (Doesn't involve LLM calls) ---
        if hasattr(self, 'vectorstore_url') and self.vectorstore_url:
            try:
                # as_retriever is local, no priority needed here
                url_retriever = self.vectorstore_url.as_retriever(search_kwargs={"k": 4})
                logger.trace(f"{log_prefix} Using existing URL vector store retriever.")
            except Exception as e:
                logger.error(f"{log_prefix} Failed to get URL retriever: {e}")
                url_retriever = None
        else:
             logger.trace(f"{log_prefix} No URL vector store available.")

        # --- History Retriever (Involves Embedding with Priority) ---
        # Get recent interactions (DB call, no LLM)
        interactions_for_rag = get_recent_interactions(db, limit=RAG_HISTORY_COUNT * 2, session_id=self.current_session_id, mode="chat", include_logs=False)

        if interactions_for_rag:
            logger.debug(f"{log_prefix} Found {len(interactions_for_rag)} recent interactions for History RAG.")
            history_texts = []
            interactions_for_rag.reverse() # Oldest first
            for interaction in interactions_for_rag:
                text_to_embed = None
                if interaction.user_input and interaction.input_type == 'text':
                    text_to_embed = f"User: {interaction.user_input}"
                elif interaction.llm_response and interaction.input_type not in ['system', 'error', 'log_error', 'log_warning', 'log_info', 'log_debug']:
                     text_to_embed = f"AI: {interaction.llm_response}"

                if text_to_embed and interaction.id not in history_ids_set:
                     # Limit length of text going into embedding? Optional.
                     # text_to_embed = text_to_embed[:1000]
                     history_texts.append(text_to_embed)
                     history_ids_set.add(interaction.id)

            if history_texts:
                logger.debug(f"{log_prefix} Attempting to embed {len(history_texts)} history texts (Priority: ELP{priority}).")
                try:
                    if not self.provider.embeddings:
                         raise ValueError("Embeddings provider not initialized.")

                    # --- USE PRIORITY DURING EMBEDDING ---
                    # Chroma.from_texts calls the embedding model internally.
                    # We assume the embedding model wrapper now accepts 'priority'.
                    # If Chroma doesn't pass kwargs down, this needs adjustment
                    # in how embeddings are generated *before* Chroma creation.
                    # Assuming LlamaCppEmbeddingsWrapper handles priority:
                    logger.trace(f"{log_prefix} Calling Chroma.from_texts...")
                    self.vectorstore_history = Chroma.from_texts(
                        texts=history_texts,
                        embedding=self.provider.embeddings,
                        # How to pass priority here? Chroma might not support it directly.
                        # --> We need to rely on the embedding object handling it, maybe via **kwargs
                        # --> OR embed first, then create Chroma from vectors.
                        # Let's assume the wrapper handles it if called directly.
                        # **Modification Required if Wrapper Doesn't Handle via Implicit Call:**
                        # vectors = self.provider.embeddings.embed_documents(history_texts, priority=priority)
                        # self.vectorstore_history = Chroma.from_vectors(texts=history_texts, embedding=self.provider.embeddings, vectors=vectors)
                    )
                    # For now, assume implicit priority handling by the embedding object passed to Chroma
                    # --- END PRIORITY CONSIDERATION ---

                    # as_retriever is local, no priority needed
                    history_retriever = self.vectorstore_history.as_retriever(search_kwargs={"k": RAG_HISTORY_COUNT})
                    logger.debug(f"{log_prefix} Created temporary History retriever (embedded {len(history_texts)} items).")

                except TaskInterruptedException as tie:
                    logger.warning(f"🚦 {log_prefix} History RAG embedding INTERRUPTED: {tie}")
                    history_retriever = None # Indicate failure
                    self.vectorstore_history = None
                    # Re-raise to be caught by the caller (direct_generate)
                    raise tie
                except Exception as e:
                    logger.error(f"❌ {log_prefix} Failed history vector store creation: {e}")
                    logger.exception(f"{log_prefix} History RAG Traceback:")
                    history_retriever = None
                    self.vectorstore_history = None
            else:
                 logger.debug(f"{log_prefix} No suitable text interactions found for history RAG.")
                 history_retriever = None
        else:
             logger.debug(f"{log_prefix} No interactions found for history RAG.")
             history_retriever = None

        # Format history IDs used
        history_ids_str = ",".join(map(str, sorted(list(history_ids_set))))
        logger.trace(f"{log_prefix} Chat History IDs used for RAG: {history_ids_str}")
        return url_retriever, history_retriever, history_ids_str

    async def _correct_response(self, db: Session, session_id: str, original_input: str, context: Dict, draft_response: str) -> str:
        """
        Uses the corrector LLM (ELP0) to refine a draft response.
        Handles TaskInterruptedException by re-raising it.
        Cleans the output if successful, otherwise returns the cleaned draft on other errors.
        """
        # Unique ID for this specific correction attempt for tracing
        correction_id = f"corr-{uuid.uuid4()}"
        log_prefix = f"✍️ {correction_id}|ELP0" # Add ELP0 marker to log prefix
        logger.info(f"{log_prefix} Refining draft response for session {session_id}...")

        # Get the model configured for correction (using "router" key)
        corrector_model = self.provider.get_model("router")

        # Handle case where the corrector model itself isn't available
        if not corrector_model:
            logger.error(f"{log_prefix} Corrector model (key 'router') not available! Returning cleaned draft.")
            # Clean the original draft using the instance method before returning
            cleaned_draft = self._cleanup_llm_output(draft_response)
            # Log this fallback action
            try: # Use try-except for DB logging
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning",
                                user_input="Corrector Fallback",
                                llm_response="Corrector model unavailable, returned cleaned draft.")
            except Exception as db_err:
                logger.error(f"{log_prefix} Failed log corrector model unavailable: {db_err}")
            return cleaned_draft

        # Prepare the input dictionary for the corrector prompt template
        prompt_input = {
            "input": original_input,
            "context": context.get("url_context", "None."),
            "history_rag": context.get("history_rag", "None."),
            "file_index_context": context.get("file_index_context", "None."),
            "log_context": context.get("log_context", "None."),
            "recent_direct_history": context.get("recent_direct_history", "None."),
            "emotion_analysis": context.get("emotion_analysis", "N/A."),
            "draft_response": draft_response
        }

        # Define the Langchain chain for correction
        try:
            chain = (
                ChatPromptTemplate.from_template(PROMPT_CORRECTOR)
                | corrector_model
                | StrOutputParser()
            )
        except Exception as chain_setup_err:
             logger.error(f"{log_prefix} Failed to set up corrector chain: {chain_setup_err}")
             try: # Use try-except for DB logging
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_error",
                                user_input="Corrector Chain Setup Error",
                                llm_response=f"Failed: {chain_setup_err}")
             except Exception as db_err:
                logger.error(f"{log_prefix} Failed log corrector chain setup error: {db_err}")
             # Fallback to cleaned draft if chain setup fails
             cleaned_draft = self._cleanup_llm_output(draft_response)
             return cleaned_draft

        # Prepare timing data dictionary for the LLM call
        corrector_timing_data = {"session_id": session_id, "mode": "chat", "execution_time_ms": 0}

        try:
            # Execute the corrector chain (sync Langchain call) in a separate thread with ELP0
            logger.debug(f"{log_prefix} Calling corrector LLM...")
            refined_response_raw = await asyncio.to_thread(
                self._call_llm_with_timing, # Use the modified helper
                chain,
                prompt_input,
                corrector_timing_data,
                priority=ELP0 # Explicitly set ELP0 priority
            )
            logger.info(f"{log_prefix} Refinement LLM call complete. Raw length: {len(refined_response_raw)}")

            # Apply robust cleanup using the instance method
            final_response_cleaned = self._cleanup_llm_output(refined_response_raw)
            logger.info(f"{log_prefix} Cleaned response length: {len(final_response_cleaned)}")

            # Add detailed log comparing inputs/outputs (Consider sampling if responses are huge)
            log_message = (
                f"Refined draft.\n"
                f"Original Input Snippet: '{original_input[:100]}...'\n"
                f"Draft Response Snippet: '{draft_response[:100]}...'\n"
                f"Raw Corrector Output Snippet: '{refined_response_raw[:100]}...'\n"
                f"Cleaned Final Snippet: '{final_response_cleaned[:100]}...'"
            )
            try: # Use try-except for DB logging
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_debug",
                                user_input="Corrector Step Details",
                                llm_response=log_message[:4000]) # Limit log length
            except Exception as db_err:
                 logger.error(f"{log_prefix} Failed log corrector details: {db_err}")

            # Return the cleaned final response (successful path)
            return final_response_cleaned

        except TaskInterruptedException as tie:
            # Specific handling for interruption
            logger.warning(f"🚦 {log_prefix} Corrector step INTERRUPTED: {tie}")
            # Re-raise the exception to be handled by the calling function (e.g., background_generate)
            raise tie

        except Exception as e:
            # Handle other, non-interruption errors during the LLM call itself
            logger.error(f"❌ {log_prefix} Error during correction LLM call: {e}")
            logger.exception(f"{log_prefix} Corrector Execution Traceback:") # Log full traceback

            # Log the failure to the database
            try: # Use try-except for DB logging
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_error",
                                user_input="Corrector Step Failed",
                                llm_response=f"Correction failed: {e}")
            except Exception as db_err:
                 logger.error(f"{log_prefix} Failed log corrector step failure: {db_err}")

            # Fallback: Clean the original draft response if correction fails
            logger.warning(f"{log_prefix} Falling back to cleaned original draft due to corrector error.")
            cleaned_draft = self._cleanup_llm_output(draft_response)
            return cleaned_draft

        except Exception as e:
            # Handle errors during the LLM call itself
            logger.error(f"❌ {correction_id} Error during correction LLM call: {e}")
            logger.exception("Corrector Execution Traceback:") # Log full traceback
            # Log the failure to the database
            add_interaction(db, session_id=session_id, mode="chat", input_type="log_error",
                            user_input="Corrector Step Failed",
                            llm_response=f"Correction failed: {e}")

            # Fallback: Clean the original draft response if correction fails
            logger.warning(f"{correction_id} Falling back to cleaned original draft due to corrector error.")
            cleaned_draft = self._cleanup_llm_output(draft_response)
            return cleaned_draft


    async def _execute_assistant_action(self, db: Session, session_id: str, action_details: Dict[str, Any], triggering_interaction: Interaction) -> str:
        """
        Executes the specified action using LLM-generated AppleScript (macOS) or background search.
        Includes RAG, generation, execution, refinement loop, and fallbacks.
        V3: Added model logging and reinforced error passing for refinement.
        """
        action_type = action_details.get("action_type", "unknown")
        parameters = action_details.get("parameters", {})
        req_id = f"act-{uuid.uuid4()}"
        logger.info(f"🚀 {req_id} Handling assistant action: '{action_type}' with params: {parameters} (Trigger ID: {triggering_interaction.id if triggering_interaction else 'N/A'})")

        # Define Fallback Messages
        mac_exec_fallback = f"Okay, I tried to perform the macOS action '{action_type}', but couldn't get it to work after {AGENT_MAX_SCRIPT_RETRIES} attempts. The script kept having errors. You might need to do it manually or check system permissions."
        non_mac_fallback = f"Action '{action_type}' seems to be macOS-specific and cannot be performed on this OS ({sys.platform})."
        search_exec_fallback = f"Sorry, I encountered an error while trying to start the web search for '{parameters.get('query', 'that topic')}'. Please try again later."
        generation_fallback = f"Sorry, I had trouble figuring out the exact steps to perform the '{action_type}' action. Please try phrasing your request differently."

        exec_db = SessionLocal()

        try:
            # --- Handle Web Search (Non-AppleScript) ---
            if action_type == "search" and parameters.get("query"):
                logger.info(f"{req_id} Handling 'search' action type. Triggering background web search...")
                trigger_id_for_log = triggering_interaction.id if triggering_interaction else None
                add_interaction(exec_db,
                                session_id=session_id, mode="chat", input_type="log_info",
                                user_input=f"Triggering Web Search: {parameters['query']}",
                                assistant_action_type=action_type, assistant_action_params=json.dumps(parameters),
                                assistant_action_executed=True, assistant_action_result="[Search process launched]",
                            )
                exec_db.commit()
                confirmation_message = await self._trigger_web_search(exec_db, session_id, parameters["query"])
                if triggering_interaction:
                    triggering_interaction.assistant_action_executed = True
                    triggering_interaction.assistant_action_result = confirmation_message
                    # Safely merge triggering_interaction state back if needed (assuming it might be detached)
                    exec_db.merge(triggering_interaction)
                    exec_db.commit()
                return confirmation_message

            # --- Handle macOS AppleScript Actions ---
            elif sys.platform == 'darwin':
                logger.info(f"{req_id} Running on macOS. Attempting LLM-based AppleScript execution.")
                params_json = json.dumps(parameters, sort_keys=True)

                script_to_execute = None
                last_error_summary = "No previous errors."
                last_stderr = ""
                last_stdout = ""
                last_rc = 0

                for attempt in range(1, AGENT_MAX_SCRIPT_RETRIES + 1):
                    logger.info(f"{req_id} AppleScript Attempt {attempt}/{AGENT_MAX_SCRIPT_RETRIES} for '{action_type}'")

                    # --- 1. RAG: Get Past Attempts ---
                    past_attempts = await asyncio.to_thread(
                        get_past_applescript_attempts, exec_db, action_type, params_json, limit=5 # Fetch last 5 attempts for this specific action/params
                    )
                    past_attempts_context = self._format_applescript_rag_context(past_attempts)
                    logger.trace(f"Past attempts context for RAG:\n{past_attempts_context}")

                    # --- 2. LLM: Generate or Refine Script ---
                    script_llm = self.provider.get_model("code")
                    if not script_llm:
                        logger.error(f"{req_id} Code model not available for AppleScript generation.")
                        if triggering_interaction:
                            triggering_interaction.assistant_action_executed = False
                            triggering_interaction.assistant_action_result = generation_fallback + " (Code model unavailable)"
                            triggering_interaction.input_type = "log_error"
                            exec_db.merge(triggering_interaction)
                            exec_db.commit()
                        return generation_fallback

                    # --- Log the model being used ---
                    model_name_used = "Unknown Code Model"
                    if hasattr(script_llm, 'model'): # For Ollama
                        model_name_used = script_llm.model
                    elif hasattr(script_llm, 'model_name'): # Generic Langchain attribute
                        model_name_used = script_llm.model_name
                    logger.debug(f"{req_id} Using code model '{model_name_used}' for {'generation' if attempt == 1 else 'refinement'}.")
                    # --- End model logging ---

                    llm_prompt_template = None
                    llm_input = {}

                    if attempt == 1:
                        llm_prompt_template = ChatPromptTemplate.from_template(PROMPT_GENERATE_APPLESCRIPT)
                        llm_input = {
                            "action_type": action_type,
                            "parameters_json": params_json,
                            "past_attempts_context": past_attempts_context # Include RAG context
                        }
                    else:
                        # Ensure all error details are passed for refinement
                        llm_prompt_template = ChatPromptTemplate.from_template(PROMPT_REFINE_APPLESCRIPT)
                        llm_input = {
                            "action_type": action_type,
                            "parameters_json": params_json,
                            "failed_script": script_to_execute or "[Script Missing]",
                            "return_code": last_rc,
                            "stderr": last_stderr, # Pass the captured stderr
                            "stdout": last_stdout, # Pass the captured stdout
                            "error_summary": last_error_summary, # Pass the summary string
                            "past_attempts_context": past_attempts_context # Include RAG context
                        }
                    logger.debug(f"{req_id} Calling LLM with this submitted prompt... {llm_prompt_template} {script_llm}")
                    script_chain = llm_prompt_template | script_llm | StrOutputParser()
                    logger.debug(f"{req_id} Calling LLM...")
                    try:
                        # Add context about the attempt number to the logger
                        with logger.contextualize(applescript_attempt=attempt):
                            logger.debug(f"{req_id} Calling LLM with this submitted prompt llm_input... {llm_input}")
                            generated_script_raw = await asyncio.to_thread(script_chain.invoke, llm_input)
                        
                        script_to_execute = re.sub(r"^```(?:applescript)?\s*|```\s*$", "", generated_script_raw, flags=re.MULTILINE).strip()
                        if not script_to_execute:
                            logger.warning(f"{req_id} LLM returned empty script on attempt {attempt}.")
                            last_error_summary = "LLM generated an empty script."
                            if attempt == AGENT_MAX_SCRIPT_RETRIES:
                                if triggering_interaction:
                                    triggering_interaction.assistant_action_executed = False
                                    triggering_interaction.assistant_action_result = generation_fallback + f" (Empty script on final attempt {attempt})"
                                    triggering_interaction.input_type = "log_error"
                                    exec_db.merge(triggering_interaction)
                                    exec_db.commit()
                                return generation_fallback
                            continue
                        logger.info(f"{req_id} LLM {'generated' if attempt == 1 else 'refined'} script (length: {len(script_to_execute)}).")
                        logger.trace(f"Script attempt {attempt}:\n{script_to_execute}")

                    except Exception as gen_err:
                        logger.error(f"{req_id} Error calling LLM for script attempt {attempt}: {gen_err}")
                        last_error_summary = f"LLM call failed: {gen_err}"
                        if attempt == AGENT_MAX_SCRIPT_RETRIES:
                            if triggering_interaction:
                                triggering_interaction.assistant_action_executed = False
                                triggering_interaction.assistant_action_result = generation_fallback + f" (LLM error on final attempt {attempt}: {gen_err})"
                                triggering_interaction.input_type = "log_error"
                                exec_db.merge(triggering_interaction)
                                exec_db.commit()
                            return generation_fallback
                        continue

                    # --- 3. Execute Script ---
                    osa_command = ["osascript", "-e", script_to_execute]
                    logger.debug(f"{req_id} Running osascript command for attempt {attempt}...")
                    exec_start_time = time.monotonic()
                    process = await asyncio.to_thread(
                        subprocess.run,
                        osa_command, capture_output=True, text=True, timeout=90, check=False
                    )
                    exec_duration_ms = (time.monotonic() - exec_start_time) * 1000
                    stdout = process.stdout.strip(); stderr = process.stderr.strip(); rc = process.returncode
                    logger.info(f"{req_id} osascript attempt {attempt} finished in {exec_duration_ms:.0f}ms. RC={rc}.")
                    # Log full stdout/stderr only at DEBUG level to reduce noise otherwise
                    logger.debug(f"{req_id} Attempt {attempt} STDOUT:\n{stdout}")
                    logger.debug(f"{req_id} Attempt {attempt} STDERR:\n{stderr}")

                    # --- 4. Store Attempt Result (Crucial for RAG) ---
                    success = (rc == 0)
                    # Create error summary ONLY if failed
                    error_summary = f"RC={rc}. Stderr: {stderr}" if not success else None
                    attempt_record = AppleScriptAttempt(
                        session_id=session_id,
                        triggering_interaction_id=triggering_interaction.id if triggering_interaction else None,
                        action_type=action_type,
                        parameters_json=params_json,
                        attempt_number=attempt,
                        generated_script=script_to_execute,
                        execution_success=success,
                        execution_return_code=rc,
                        execution_stdout=stdout,
                        execution_stderr=stderr,
                        execution_duration_ms=exec_duration_ms,
                        error_summary=error_summary[:1000] if error_summary else None
                    )
                    exec_db.add(attempt_record)
                    exec_db.commit()
                    logger.debug(f"{req_id} Stored attempt {attempt} record ID {attempt_record.id}.")
                    # --- RAG data is now updated for the *next* loop iteration ---

                    # --- 5. Check Outcome ---
                    if success:
                        logger.success(f"{req_id} AppleScript execution successful on attempt {attempt} for '{action_type}'.")
                        if triggering_interaction:
                            triggering_interaction.assistant_action_executed = True
                            triggering_interaction.assistant_action_result = stdout or f"Action '{action_type}' completed successfully."
                            if hasattr(triggering_interaction, 'execution_time_ms'):
                                triggering_interaction.execution_time_ms = exec_duration_ms
                            else:
                                logger.warning("Interaction model missing 'execution_time_ms', skipping update.")
                            triggering_interaction.input_type = "text" # Reset from potential previous error state
                            exec_db.merge(triggering_interaction)
                            exec_db.commit()
                        return stdout or f"Action '{action_type}' completed."
                    else:
                        # --- VERBOSE FAILURE LOGGING (Already implemented in previous step) ---
                        logger.error(f"❌ {req_id} AppleScript Attempt {attempt} FAILED for action '{action_type}'.")
                        logger.error(f"  [FAIL Attempt {attempt}] Return Code: {rc}")
                        logger.error(f"  [FAIL Attempt {attempt}] Error Summary: {error_summary}") # Contains stderr
                        logger.error(f"  [FAIL Attempt {attempt}] Stderr:\n---\n{stderr}\n---")
                        logger.error(f"  [FAIL Attempt {attempt}] Stdout:\n---\n{stdout}\n---")
                        logger.error(f"  [FAIL Attempt {attempt}] Script Executed:\n--- Start Failed Script ---\n{script_to_execute}\n--- End Failed Script ---")
                        # --- End Verbose Logging ---

                        # Store details for next refinement attempt
                        last_error_summary = error_summary # Used in the next loop's prompt
                        last_stderr = stderr             # Used in the next loop's prompt
                        last_stdout = stdout             # Used in the next loop's prompt
                        last_rc = rc                     # Used in the next loop's prompt
                        # Loop continues...

                # --- End of Loop ---
                logger.error(f"{req_id} AppleScript execution failed after {AGENT_MAX_SCRIPT_RETRIES} attempts for '{action_type}'.")
                logger.error(f"  [FINAL FAIL] Last Error Summary: {last_error_summary}")
                logger.error(f"  [FINAL FAIL] Last RC: {last_rc}")
                logger.error(f"  [FINAL FAIL] Last Stderr:\n---\n{last_stderr}\n---")
                logger.error(f"  [FINAL FAIL] Last Stdout:\n---\n{last_stdout}\n---")
                logger.error(f"  [FINAL FAIL] Last Script Attempted (Attempt {AGENT_MAX_SCRIPT_RETRIES}):\n--- Start Final Failed Script ---\n{script_to_execute or '[Script Unavailable]'}\n--- End Final Failed Script ---")

                if triggering_interaction:
                    triggering_interaction.assistant_action_executed = True # It was attempted to exhaustion
                    triggering_interaction.assistant_action_result = f"Failed after {AGENT_MAX_SCRIPT_RETRIES} attempts. Last Error: {last_error_summary}"
                    triggering_interaction.input_type = "log_error"
                    exec_db.merge(triggering_interaction)
                    exec_db.commit()

                return mac_exec_fallback # Return fallback message after max retries

            # --- Handle Non-macOS platform ---
            else:
                logger.warning(f"{req_id} Action '{action_type}' skipped: Not web search and not on macOS. Platform: {sys.platform}")
                if triggering_interaction:
                    triggering_interaction.assistant_action_executed = False
                    triggering_interaction.assistant_action_result = non_mac_fallback
                    triggering_interaction.input_type = "log_warning"
                    exec_db.merge(triggering_interaction)
                    exec_db.commit()
                return non_mac_fallback

        except Exception as e:
            err_msg = f"Unexpected error during action execution for '{action_type}': {e}"
            logger.error(f"{req_id} {err_msg}")
            logger.exception(f"{req_id} Action Execution Traceback:")
            try:
                if triggering_interaction:
                    triggering_interaction.assistant_action_executed = True
                    triggering_interaction.assistant_action_result = err_msg[:1000]
                    triggering_interaction.input_type = "log_error"
                    exec_db.merge(triggering_interaction)
                    exec_db.commit()
            except Exception as log_err: logger.error(f"Failed to log final action execution error: {log_err}")
            return f"Sorry, I encountered an unexpected internal issue while trying the '{action_type}' action."
        finally:
            if exec_db: exec_db.close()

    def _format_applescript_rag_context(self, attempts: List[AppleScriptAttempt]) -> str:
        """Formats past attempts for the LLM prompt context."""
        if not attempts:
            return "None available."
        context_str = ""
        for i, attempt in enumerate(attempts):
            context_str += f"--- Attempt {i+1} ({attempt.timestamp.isoformat()}) ---\n"
            context_str += f"Script:\n```applescript\n{attempt.generated_script or '[Script Missing]'}\n```\n"
            context_str += f"Success: {attempt.execution_success}\n"
            if not attempt.execution_success:
                context_str += f"  RC: {attempt.execution_return_code}\n"
                context_str += f"  Error Summary: {attempt.error_summary}\n"
                # Optionally include short stderr/stdout snippets
                # context_str += f"  Stderr: {attempt.execution_stderr[:100]}...\n"
                # context_str += f"  Stdout: {attempt.execution_stdout[:100]}...\n"
            context_str += "---\n"
            if len(context_str) > 2000: # Limit context size
                context_str += "[Context truncated]...\n"
                break
        return context_str

    def _format_docs(self, docs: List[Any], source_type: str = "Context") -> str:
        """Helper to format retrieved Langchain Documents into a single string."""
        if not docs:
            logger.trace(f"_format_docs received empty list for {source_type}")
            return f"No relevant {source_type.lower()} found."
        if not isinstance(docs, list):
             logger.warning(f"_format_docs received non-list: {type(docs)}")
             return f"Invalid document list provided for {source_type}."
        if not docs: # Check again after type check
             return f"No relevant {source_type.lower()} found."

        if hasattr(docs[0], 'page_content'):
             return "\n\n".join(f"Source Chunk ({source_type}):\n{doc.page_content}" for doc in docs)
        else:
            logger.warning(f"Unrecognized doc type in _format_docs: {type(docs[0])}. Assuming Interaction list.")
            return self._format_interaction_list_to_string(docs)


    def _format_interaction_list_to_string(self, interactions: List[Interaction], include_type=False) -> str:
        """Formats a list of Interaction objects into a string for RAG/log context."""
        if not interactions:
            return "None found."
        if not isinstance(interactions, list):
             logger.error(f"_format_interaction_list_to_string received non-list: {type(interactions)}")
             return "Invalid interaction list provided."
        if not interactions:
             return "None found."

        str_parts = []
        sorted_interactions = sorted(interactions, key=lambda i: i.timestamp) # Oldest first

        for interaction in sorted_interactions:
            prefix, text = None, None
            if interaction.input_type == 'text' and interaction.user_input:
                prefix = "User:"
                text = interaction.user_input
            elif interaction.llm_response and interaction.input_type not in ['system', 'error', 'log_error', 'log_warning', 'log_info', 'log_debug']:
                prefix = "AI:"
                text = interaction.llm_response
            elif interaction.input_type.startswith('log_'):
                prefix = f"LOG ({interaction.input_type.split('_')[1].upper()}):"
                text = interaction.llm_response
            elif interaction.input_type == 'error':
                prefix = "LOG (ERROR):"
                text = interaction.llm_response
            elif interaction.input_type == 'system':
                prefix = "System:"
                text = interaction.user_input

            if prefix and text:
                entry = f"{prefix} {text}"
                if include_type:
                     entry = f"[{interaction.timestamp.strftime('%H:%M:%S')} {interaction.input_type}] {entry}"
                text_snippet = (entry[:250] + '...') if len(entry) > 250 else entry
                str_parts.append(text_snippet)

        return "\n---\n".join(str_parts) if str_parts else "None found."

    def _format_direct_history(self, interactions: List[Interaction]) -> str:
        """Formats a list of Interaction objects into a chronological string for the prompt."""
        if not interactions:
            return "No recent global conversation history available."
        if not isinstance(interactions, list):
             logger.error(f"_format_direct_history received non-list: {type(interactions)}")
             return "Invalid direct history list provided."
        if not interactions:
             return "No recent global conversation history available."

        history_str_parts = []
        for interaction in interactions: # Assumes sorted oldest first
            prefix, text = None, None
            if interaction.input_type == 'text' and interaction.user_input:
                prefix = "User:"
                text = interaction.user_input
            elif interaction.llm_response and interaction.input_type == 'llm_response':
                 prefix = "AI:"
                 text = interaction.llm_response

            if prefix and text:
                text_snippet = (text[:150] + '...') if len(text) > 150 else text
                history_str_parts.append(f"{prefix} {text_snippet}")

        if not history_str_parts:
             return "No textual conversation history available."

        return "\n".join(history_str_parts)

    def _format_log_history(self, interactions: List[Interaction]) -> str:
        """Formats a list of Interaction log objects into a string for the prompt."""
        if not interactions:
            return "No recent relevant logs found."
        if not isinstance(interactions, list):
             logger.error(f"_format_log_history received non-list: {type(interactions)}")
             return "Invalid log history list provided."
        if not interactions:
             return "No recent relevant logs found."

        log_str_parts = []
        sorted_interactions = sorted(interactions, key=lambda i: i.timestamp, reverse=True)

        for interaction in sorted_interactions:
            log_level = interaction.input_type.split('_')[-1].upper() if '_' in interaction.input_type else interaction.input_type.upper()
            log_message = interaction.llm_response or interaction.user_input or "[Log content missing]"
            timestamp_str = interaction.timestamp.strftime('%H:%M:%S')
            log_snippet = (log_message[:200] + '...') if len(log_message) > 200 else log_message
            log_str_parts.append(f"[{timestamp_str} {log_level}] {log_snippet}")

        return "\n".join(log_str_parts) if log_str_parts else "No recent relevant logs found."


    def _call_llm_with_timing(self, chain: Any, inputs: Any, interaction_data: Dict[str, Any], priority: int = ELP0): # Added priority parameter
        """
        Wrapper to call LLM chain, measure time, log, and handle priority/interruptions.
        Raises TaskInterruptedException if the underlying call is interrupted.
        """
        start_time = time.time()
        response = None # Initialize response
        try:
            logger.trace(f"🚀 Invoking chain {type(chain)} with inputs type: {type(inputs)} (Priority: ELP{priority})")
            # Pass priority down via config dictionary
            response = chain.invoke(inputs, config={'priority': priority})
            duration = (time.time() - start_time) * 1000
            logger.info(f"⏱️ LLM Inference (ELP{priority}) took {duration:.2f} ms")
            interaction_data['execution_time_ms'] = interaction_data.get('execution_time_ms', 0) + duration

            # --- Check for interruption marker in the response string ---
            # This assumes the StrOutputParser() returns the error string directly
            if isinstance(response, str) and interruption_error_marker in response:
                 logger.warning(f"🚦 Task Interrupted (Detected in _call_llm_with_timing). Raising TaskInterruptedException.")
                 raise TaskInterruptedException(response) # Raise specific exception

            return response

        except TaskInterruptedException:
             # Re-raise the specific exception if caught directly
             raise
        except Exception as e:
            # --- Check if the exception itself signals interruption ---
            if interruption_error_marker in str(e):
                 logger.warning(f"🚦 Task Interrupted (Detected via Exception in _call_llm_with_timing). Raising TaskInterruptedException.")
                 raise TaskInterruptedException(str(e)) # Raise specific exception
            # --- Handle other errors ---
            else:
                 log_err_msg = f"LLM Chain Error (ELP{priority}): {e}"
                 logger.error(f"❌ {log_err_msg}")
                 logger.exception("Traceback for LLM Chain error:")
                 session_id = interaction_data.get("session_id")
                 mode = interaction_data.get("mode", "chat")
                 try: # Log error to DB
                     temp_db = SessionLocal()
                     add_interaction(temp_db, session_id=session_id, mode=mode, input_type="log_error", llm_response=log_err_msg[:4000])
                     temp_db.close()
                 except Exception as db_log_err:
                      logger.error(f"Failed to log LLM error to DB: {db_log_err}")
                 duration = (time.time() - start_time) * 1000
                 interaction_data['execution_time_ms'] = interaction_data.get('execution_time_ms', 0) + duration
                 raise # Re-raise the original non-interruption error


    def _classify_input_complexity(self, db: Session, user_input: str, interaction_data: dict) -> str:
        """Classifies input as 'chat_simple', 'chat_complex', or 'agent_task' (synchronous)."""
        logger.info("🤔 Classifying input complexity...")
        history_summary = self._get_history_summary(db, MEMORY_SIZE)
        parser = JsonOutputParser()

        # --- FIX HERE: Get the appropriate model using get_model ---
        # Use the 'router' model for classification, or fallback to 'default'
        classification_model = self.provider.get_model("router")
        if not classification_model:
            logger.warning("Router model not found for classification, falling back to default.")
            classification_model = self.provider.get_model("default")

        if not classification_model:
            logger.error("❌ Default model also not found! Cannot perform input classification.")
            interaction_data['classification'] = "chat_simple" # Fallback classification
            interaction_data['classification_reason'] = "Classification failed: Required model not found."
            # Log error to DB
            try: add_interaction(db, session_id=interaction_data.get("session_id"), mode="chat", input_type="log_error", llm_response="Input classification failed: Model unavailable.")
            except Exception as db_err: logger.error(f"Failed log classification model error: {db_err}")
            return "chat_simple" # Return fallback

        # chain = (self.input_classification_prompt | self.provider.model | parser) # OLD LINE
        chain = (self.input_classification_prompt | classification_model | parser) # NEW LINE using fetched model
        # --- END FIX ---

        attempts = 0
        last_error = None
        while attempts < DEEP_THOUGHT_RETRY_ATTEMPTS:
            try:
                # Ensure input keys match the prompt template
                prompt_inputs_for_classification = {"input": user_input, "history_summary": history_summary}
                response_json = self._call_llm_with_timing(chain, prompt_inputs_for_classification, interaction_data)
                classification = response_json.get("classification", "chat_simple")
                reason = str(response_json.get("reason", "N/A"))
                if classification not in ["chat_simple", "chat_complex", "agent_task"]:
                    logger.warning(f"Classification LLM returned invalid category '{classification}', defaulting to chat_simple.")
                    classification = "chat_simple"
                interaction_data['classification'] = classification
                interaction_data['classification_reason'] = reason
                logger.info(f"✅ Input classified as: '{classification}'. Reason: {reason}")
                return classification
            except Exception as e:
                attempts += 1
                last_error = e
                logger.warning(f"⚠️ Error classifying input (Attempt {attempts}/{DEEP_THOUGHT_RETRY_ATTEMPTS}): {e}")
                if attempts < DEEP_THOUGHT_RETRY_ATTEMPTS:
                    time.sleep(0.5) # Use synchronous sleep here as this method is called sync

        # After retries
        logger.error(f"❌ Max retries ({DEEP_THOUGHT_RETRY_ATTEMPTS}) for input classification. Last error: {last_error}")
        interaction_data['classification'] = "chat_simple"
        interaction_data['classification_reason'] = f"Classification failed after retries: {last_error}"
        # Log error to DB
        try: add_interaction(db, session_id=interaction_data.get("session_id"), mode="chat", input_type="log_error", llm_response=f"Input classification failed after {attempts} attempts. Error: {last_error}")
        except Exception as db_err: logger.error(f"Failed log classification retry error: {db_err}")
        return "chat_simple"


    def _run_tree_of_thought(self, db: Session, input: str, rag_context_docs: List[Any], history_rag_interactions: List[Interaction], log_context_str: str, recent_direct_history_str: str, file_index_context_str: str, interaction_data: Dict[str, Any], triggering_interaction_id: int) -> str:
        """Runs Tree of Thoughts simulation (synchronous), includes direct history and logs."""
        user_input = input
        logger.warning(f"🌳 Running ToT for input: '{user_input[:50]}...' (Trigger ID: {triggering_interaction_id})")
        interaction_data['tot_analysis_requested'] = True
        rag_context_str = self._format_docs(rag_context_docs, source_type="URL")
        history_rag_str = self._format_interaction_list_to_string(history_rag_interactions) # Format Interaction list

        chain = (self.tot_prompt | self.provider.model | StrOutputParser())
        tot_result = "Error during ToT analysis."
        try:
            llm_result = self._call_llm_with_timing(
                chain,
                {
                    "input": user_input,
                    "context": rag_context_str,
                    "history_rag": history_rag_str,
                    "file_index_context": file_index_context_str, # Added here
                    "log_context": log_context_str,
                    "recent_direct_history": recent_direct_history_str
                },
                interaction_data
            )
            tot_result = llm_result
            logger.info(f"🌳 ToT analysis LLM call complete for Trigger ID: {triggering_interaction_id}.")

            if triggering_interaction_id:
                logger.debug(f"Attempting to save ToT result to original interaction ID: {triggering_interaction_id}")
                trigger_interaction = db.query(Interaction).filter(Interaction.id == triggering_interaction_id).first()
                if trigger_interaction:
                    trigger_interaction.tot_result = tot_result
                    trigger_interaction.tot_analysis_requested = True
                    trigger_interaction.tot_delivered = False
                    db.commit()
                    logger.success(f"✅ Saved ToT result to Interaction ID {triggering_interaction_id} (undelivered).")
                else:
                    logger.error(f"❌ Could not find original interaction {triggering_interaction_id} to save ToT result.")
                    add_interaction(db, session_id=interaction_data.get("session_id"), mode="chat", input_type="log_warning", llm_response=f"Orphaned ToT Result for input '{user_input[:50]}...': {tot_result[:200]}...")
            else:
                 logger.warning("No triggering interaction ID provided to save ToT result.")

            return tot_result
        except Exception as e:
            err_msg = f"Error during ToT generation (Trigger ID: {triggering_interaction_id}): {e}"
            logger.error(f"❌ {err_msg}")
            add_interaction(db, session_id=interaction_data.get("session_id"), mode="chat", input_type="log_error", llm_response=err_msg)
            if triggering_interaction_id:
                 trigger_interaction = db.query(Interaction).filter(Interaction.id == triggering_interaction_id).first()
                 if trigger_interaction:
                     trigger_interaction.tot_result = err_msg
                     trigger_interaction.tot_delivered = False
                     db.commit()
            return "Error during deep analysis."


    def _run_emotion_analysis(self, db: Session, user_input: str, interaction_data: dict) -> str:
        """Analyzes emotion/context (synchronous)."""
        logger.info("😊 Analyzing input emotion/context...")
        history_summary = self._get_history_summary(db, MEMORY_SIZE)
        chain = (self.emotion_analysis_prompt | self.provider.model | StrOutputParser())
        try:
            analysis = self._call_llm_with_timing(chain, {"input": user_input, "history_summary": history_summary}, interaction_data)
            logger.info(f"😊 Emotion/Context Analysis Result: {analysis}")
            interaction_data['emotion_context_analysis'] = analysis
            return analysis
        except Exception as e:
            err_msg = f"Error during emotion analysis: {e}"
            logger.error(f"❌ {err_msg}")
            interaction_data['emotion_context_analysis'] = err_msg
            add_interaction(db, session_id=interaction_data.get("session_id"), mode="chat", input_type="log_error", llm_response=err_msg)
            return "Could not analyze."


    def _get_history_summary(self, db: Session, limit: int) -> str:
        """Gets a simple string summary of recent chat interactions (synchronous)."""
        interactions = get_recent_interactions(db, limit=limit * 2, session_id=self.current_session_id, mode="chat", include_logs=False)
        if not interactions:
            return "No recent conversation history."
        summary = []
        interactions.reverse()
        processed_count = 0
        for interaction in interactions:
             prefix, text = None, None
             if interaction.llm_response and interaction.input_type != 'system': prefix, text = "AI:", interaction.llm_response
             elif interaction.user_input and interaction.input_type != 'system': prefix, text = "User:", interaction.user_input

             if text:
                text = (text[:150] + '...') if len(text) > 150 else text
                summary.append(f"{prefix} {text}")
                processed_count += 1
                if processed_count >= limit:
                    break
        return "\n".join(summary)


    async def _run_tot_in_background_wrapper(self, db_session_factory: Any, input: str, rag_context_docs: List[Any], history_rag_interactions: List[Interaction], log_context_str: str, recent_direct_history_str: str, file_index_context_str: str, triggering_interaction_id: int):
        """Async wrapper to run synchronous ToT logic with its own DB session."""
        logger.info(f"BG ToT Wrapper: Starting for trigger ID {triggering_interaction_id}")
        db = db_session_factory()
        bg_interaction_data = {'id': triggering_interaction_id, 'execution_time_ms': 0, 'session_id': self.current_session_id, 'mode': 'chat'}
        try:
            await asyncio.to_thread(
                self._run_tree_of_thought,
                db=db,
                input=input,
                rag_context_docs=rag_context_docs,
                history_rag_interactions=history_rag_interactions,
                log_context_str=log_context_str,
                recent_direct_history_str=recent_direct_history_str,
                file_index_context_str=file_index_context_str, # Passed here
                interaction_data=bg_interaction_data,
                triggering_interaction_id=triggering_interaction_id
            )
            logger.info(f"BG ToT Wrapper: Finished successfully for trigger ID {triggering_interaction_id}")
        except Exception as e:
            logger.error(f"BG ToT Wrapper: Error running ToT for trigger ID {triggering_interaction_id}: {e}")
        finally:
            if db:
                 db.close()


    def _analyze_assistant_action(self, db: Session, user_input: str, session_id: str, context: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Calls LLM (ELP0) to check if input implies a macOS action, extracts parameters.
        Uses a more robust method to find the JSON block, ignoring <think> tags during extraction.
        Handles TaskInterruptedException by re-raising.
        """
        log_prefix = f"🤔 ActionAnalyze|ELP0|{session_id}" # Add ELP0 marker
        logger.info(f"{log_prefix} Analyzing input for potential Assistant Action: '{user_input[:50]}...'")

        prompt_input = {
            "input": user_input,
            "history_summary": context.get("history_summary", "N/A"),
            "log_context": context.get("log_context", "N/A"),
            "recent_direct_history": context.get("recent_direct_history", "N/A")
        }

        # Get the appropriate model using get_model
        action_analysis_model = self.provider.get_model("router") # Or "default"
        if not action_analysis_model:
            logger.error(f"{log_prefix} Action analysis model (router/default) not available!")
            try: # Log error to DB
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_error",
                                user_input="Action Analysis Failed",
                                llm_response="Action analysis model unavailable.")
            except Exception as db_err:
                 logger.error(f"{log_prefix} Failed log action analysis model error: {db_err}")
            return None # Cannot proceed without a model

        # Define the chain using a StrOutputParser first to get the raw text
        analysis_chain = (
            ChatPromptTemplate.from_template(PROMPT_ASSISTANT_ACTION_ANALYSIS)
            | action_analysis_model
            | StrOutputParser() # Get raw string output first
        )

        action_timing_data = {"session_id": session_id, "mode": "chat", "execution_time_ms": 0}
        last_error = None
        raw_llm_response_full = "Error: Analysis LLM call failed." # Store the full response

        for attempt in range(DEEP_THOUGHT_RETRY_ATTEMPTS):
            logger.debug(f"{log_prefix} Assistant Action analysis attempt {attempt + 1}/{DEEP_THOUGHT_RETRY_ATTEMPTS}")
            analysis_result = None # Reset result for each attempt
            json_str_to_parse = None # Variable to hold the extracted JSON string

            try:
                # Call the LLM chain (outputs raw string) with ELP0 priority
                raw_llm_response_full = asyncio.to_thread(
                    self._call_llm_with_timing, analysis_chain, prompt_input, action_timing_data, priority=ELP0
                )
                logger.trace(f"{log_prefix} Raw LLM Analysis Response:\n{raw_llm_response_full}")

                # --- Robust JSON Extraction ---
                # 1. Try finding ```json ... ``` block first
                json_markdown_match = re.search(r"```json\s*(.*?)\s*```", raw_llm_response_full, re.DOTALL)
                if json_markdown_match:
                    json_str_to_parse = json_markdown_match.group(1).strip()
                    logger.trace(f"{log_prefix} Extracted JSON string from markdown block.")
                else:
                    # 2. Fallback: Find first '{' and last '}' in the *entire* response
                    json_start_index = raw_llm_response_full.find('{')
                    json_end_index = raw_llm_response_full.rfind('}')
                    if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
                        json_str_to_parse = raw_llm_response_full[json_start_index : json_end_index + 1]
                        logger.warning(f"{log_prefix} Extracted JSON using fallback find '{{...}}'. Might be less precise.")
                        # Log the raw response if fallback was used, as it might indicate non-compliance
                        try:
                           add_interaction(db, session_id=session_id, mode="chat", input_type="log_debug",
                                           user_input=f"Action Analysis Fallback Extraction (Attempt {attempt + 1})",
                                           llm_response=f"Used find {{...}} fallback. Raw: {raw_llm_response_full[:1000]}")
                        except Exception: pass
                    else:
                        # No plausible JSON block found
                        logger.warning(f"{log_prefix} Could not find JSON block in raw response (Attempt {attempt + 1}).")
                        last_error = ValueError("No JSON block found in LLM response")
                        # Log failure to DB
                        try:
                            add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning",
                                            user_input=f"Action Analysis No JSON Found (Attempt {attempt + 1})",
                                            llm_response=raw_llm_response_full[:4000])
                        except Exception as db_err:
                            logger.error(f"{log_prefix} Failed log no JSON found: {db_err}")
                        continue # Go to next retry attempt

                # --- Optional: Log the first <think> block without modifying json_str_to_parse ---
                think_match = re.search(r'<think>(.*?)</think>', raw_llm_response_full, re.DOTALL | re.IGNORECASE)
                if think_match:
                    thought_process = think_match.group(1).strip()
                    logger.debug(f"{log_prefix} Extracted thought process (for logging):\n{thought_process}")
                    # Log thought process separately if needed
                    # try: add_interaction(db, ..., llm_response=thought_process)
                    # except: pass
                # --- End Think Logging ---

                # Attempt to parse the extracted JSON string
                analysis_result = json.loads(json_str_to_parse)

                # Basic validation of the parsed JSON structure
                if isinstance(analysis_result, dict) and "action_type" in analysis_result and "parameters" in analysis_result:
                    action_type = analysis_result.get("action_type")
                    parameters = analysis_result.get("parameters", {})
                    explanation = analysis_result.get("explanation", "N/A")
                    logger.info(f"✅ {log_prefix} Assistant Action analysis successful: Type='{action_type}', Params={parameters}")

                    # Log success to DB (include raw response for context)
                    try:
                        add_interaction(db,
                                        session_id=session_id, mode="chat", input_type="log_info",
                                        user_input=f"Assistant Action Analysis OK for: {user_input[:100]}...",
                                        llm_response=f"Action Type: {action_type}, Explanation: {explanation}",
                                        # Store the raw response to see think tags etc. if needed
                                        assistant_action_analysis_json=raw_llm_response_full,
                                        assistant_action_type=action_type,
                                        assistant_action_params=json.dumps(parameters)
                                        )
                    except Exception as db_err:
                         logger.error(f"{log_prefix} Failed log action analysis success: {db_err}")

                    # Return result if action required, otherwise return None
                    if action_type != "no_action":
                        return analysis_result # Success, action needed
                    else:
                        logger.info(f"{log_prefix} Analysis determined 'no_action' required.")
                        return None # Success, no action needed
                else:
                    # Handle cases where JSON was parsed but structure is wrong
                    logger.warning(f"{log_prefix} Assistant Action analysis produced invalid JSON structure after parsing (Attempt {attempt + 1}): {analysis_result}")
                    last_error = ValueError("Invalid JSON structure after parsing")
                    try: # Log warning to DB
                        add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning",
                                        user_input=f"Action Analysis Invalid Structure (Attempt {attempt + 1})",
                                        llm_response=f"Parsed: {str(analysis_result)[:500]}. Raw: {raw_llm_response_full[:1000]}",
                                        assistant_action_analysis_json=raw_llm_response_full[:4000])
                    except Exception as db_err:
                         logger.error(f"{log_prefix} Failed log invalid structure: {db_err}")
                    # Continue to next retry

            except TaskInterruptedException as tie:
                 # Handle interruption specifically - DO NOT RETRY
                 logger.warning(f"🚦 {log_prefix} Action Analysis INTERRUPTED (Attempt {attempt + 1}): {tie}")
                 # Re-raise immediately to be caught by the calling function (background_generate)
                 raise tie
            except json.JSONDecodeError as json_e:
                 # Handle JSON parsing errors
                 logger.warning(f"⚠️ {log_prefix} Failed to parse extracted JSON string (Attempt {attempt + 1}): {json_e}")
                 last_error = json_e
                 try: # Log warning to DB
                     add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning",
                                     user_input=f"Action Analysis JSON Parse FAILED (Attempt {attempt + 1})",
                                     llm_response=f"Error: {json_e}. String was: {json_str_to_parse[:500]}. Raw: {raw_llm_response_full[:1000]}",
                                     assistant_action_analysis_json=raw_llm_response_full[:4000])
                 except Exception as db_err:
                      logger.error(f"{log_prefix} Failed log JSON parse failure: {db_err}")
                 # Continue to next retry
            except Exception as e:
                 # Handle other unexpected errors during analysis/parsing
                 logger.warning(f"⚠️ {log_prefix} Error during Assistant Action analysis attempt {attempt + 1}: {e}")
                 last_error = e
                 try: # Log warning to DB
                     add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning",
                                     user_input=f"Action Analysis FAILED (Attempt {attempt + 1})",
                                     llm_response=f"Error: {e}. Raw: {raw_llm_response_full[:1000]}",
                                     assistant_action_analysis_json=raw_llm_response_full[:4000])
                 except Exception as db_err:
                      logger.error(f"{log_prefix} Failed log general analysis failure: {db_err}")
                 # Continue to next retry

            # Wait before retrying if attempts remain and error wasn't interruption
            if attempt < DEEP_THOUGHT_RETRY_ATTEMPTS - 1:
                time.sleep(0.5 + attempt * 0.5) # Use asyncio sleep
            else:
                # Max retries reached for non-interruption errors
                logger.error(f"❌ {log_prefix} Max retries ({DEEP_THOUGHT_RETRY_ATTEMPTS}) reached for Assistant Action analysis. Last error: {last_error}")
                return None # Indicate failure after retries

        # This part should ideally not be reached if loop completes or breaks/returns
        logger.error(f"{log_prefix} Exited Assistant Action analysis loop unexpectedly after max retries.")
        return None



    def _generate_applescript_for_action(self, action_type: str, params: Dict[str, Any]) -> Optional[str]:
        """
        Generates a specific, predefined AppleScript code string based on the
        action_type and parameters. This is a deterministic mapping, not LLM/RAG based generation.
        Returns the AppleScript string if a match is found, otherwise None.
        Handles basic quoting for shell commands within AppleScript.
        V2: Added more examples based on potential action analysis categories.
        """
        # Ensure re and json are imported if not done globally in app.py
        import re
        import json # Used only for logging parameters in comments

        req_id = f"scriptgen-{uuid.uuid4()}" # For logging this specific generation attempt
        logger.debug(f"{req_id} Attempting to generate AppleScript for action '{action_type}' with params: {params}")

        # --- Helper for escaping AppleScript strings ---
        def escape_applescript_string(s: str) -> str:
            """Escapes double quotes and backslashes for AppleScript string literals."""
            if not isinstance(s, str): return ""
            return s.replace('\\', '\\\\').replace('"', '\\"')

        # --- Helper for quoting for shell script ---
        def quote_for_shell(s: str) -> str:
            """Uses shlex.quote for robust shell quoting (requires shlex import)."""
            import shlex
            if not isinstance(s, str): return "''"
            return shlex.quote(s)

        # --- Basic Script Structure ---
        script_lines = [
            'use AppleScript version "2.4"',
            'use scripting additions',
            '',
            f'-- Request ID: {req_id}', # Link script back to log
            f'-- Action: {action_type}',
            f'-- Parameters: {json.dumps(params)}',
            '',
            'try',
        ]
        success_result_code = f'return "Action \'{action_type}\' reported as completed."' # Default success return
        action_implemented = False # Flag to track if we found a match

        # --- Action Mapping Logic (Deterministic Rules) ---

        # --- Category: File/App Interaction ---
        # Example: Open a specific application, file, or URL target
        if params.get("target"):
            target = params["target"]
            escaped_target_log = escape_applescript_string(target) # For AS log string
            quoted_target_shell = quote_for_shell(target) # For shell command
            script_lines.append(f'  log "Action: Opening target: {escaped_target_log}"')
            script_lines.append(f'  do shell script "open " & {quoted_target_shell}')
            success_result_code = f'return "Attempted to open: {escape_applescript_string(target)}"'
            action_implemented = True

        # --- Category: Search ---
        # Example: Web Search
        elif action_type == "search" and params.get("query"):
            query = params["query"]
            escaped_query_log = escape_applescript_string(query)
            # Basic URL encoding might be needed here for robust search URLs
            # For simplicity, just using query directly in Google Search URL
            search_url = f'https://www.google.com/search?q={query}'
            quoted_url_shell = quote_for_shell(search_url)
            script_lines.append(f'  log "Action: Performing web search for: {escaped_query_log}"')
            script_lines.append(f'  do shell script "open " & {quoted_url_shell}')
            success_result_code = f'return "Opened web search for: {escaped_query_log}"'
            action_implemented = True

        # Example: Find Files (Basic using mdfind/spotlight)
        elif action_type == "search" and params.get("file_name"):
            file_name = params["file_name"]
            escaped_name_log = escape_applescript_string(file_name)
            quoted_name_shell = quote_for_shell(file_name)
            script_lines.append(f'  log "Action: Searching for file name containing: {escaped_name_log}"')
            # Use mdfind for Spotlight search - searches filenames and content
            shell_cmd = f'mdfind "kMDItemFSName == \'{file_name}\'c" || mdfind {quoted_name_shell}' # Try exact name then general
            script_lines.append(f'  set searchResults to do shell script "{shell_cmd}"')
            script_lines.append('  if searchResults is "" then')
            script_lines.append(f'    return "No files found containing name: {escaped_name_log}"')
            script_lines.append('  else')
            script_lines.append('    return "Files Found:\\n" & searchResults')
            script_lines.append('  end if')
            # Success result is handled within the script logic here
            success_result_code = None # Override default
            action_implemented = True

        # --- Category: Basics / System Info ---
        # Example: Check Disk Space
        elif action_type == "basics" and params.get("check_disk_space", False): # Check boolean flag
            script_lines.append('  log "Action: Checking available disk space on /."')
            awk_script = "'{print $4 \" available\"}'" # Note the quoting
            shell_cmd = f'df -h / | tail -n 1 | awk {awk_script}'
            # Escape double quotes within the shell command string for AppleScript
            escaped_shell_cmd = shell_cmd.replace('"', '\\"')
            script_lines.append(f'  set diskSpace to do shell script "{escaped_shell_cmd}"')
            success_result_code = 'return "Boot Volume Disk Space: " & diskSpace'
            action_implemented = True

        # Example: Get Current Volume
        elif action_type == "basics" and params.get("get_volume", False):
            script_lines.append('  log "Action: Getting current output volume level."')
            script_lines.append('  set volLevel to output volume of (get volume settings)')
            success_result_code = 'return "Current output volume: " & (volLevel as string) & "%"'
            action_implemented = True

        # Example: Basic Calculation (less ideal via AppleScript, better in Python)
        # Placeholder - prefer Python for calculations
        elif action_type == "basics" and params.get("calculate"):
             logger.warning("Calculation requested via AppleScript - Python is preferred.")
             success_result_code = 'return "Calculation via AppleScript not implemented. Perform in Python."'
             action_implemented = True # Treat as handled (by saying not implemented)


        # --- Category: Scheduling ---
        # Example: Open Calendar App
        elif action_type == "scheduling" and params.get("open_calendar", False):
             script_lines.append('  log "Action: Opening Calendar application."')
             script_lines.append('  tell application "Calendar"')
             script_lines.append('    activate') # Bring Calendar to front
             script_lines.append('  end tell')
             success_result_code = 'return "Opened Calendar application."'
             action_implemented = True

        # Example: Create a simple reminder (requires Reminders permission)
        elif action_type == "scheduling" and params.get("reminder_text"):
             reminder = params["reminder_text"]
             list_name = params.get("reminder_list", "Reminders") # Default list
             escaped_reminder = escape_applescript_string(reminder)
             escaped_list = escape_applescript_string(list_name)
             script_lines.append(f'  log "Action: Creating reminder \'{escaped_reminder}\' in list \'{escaped_list}\'."')
             script_lines.append('  tell application "Reminders"')
             script_lines.append('    -- Ensure the list exists, otherwise use default')
             script_lines.append(f'    if not (exists list "{escaped_list}") then')
             script_lines.append(f'      log "List \'{escaped_list}\' not found, using default Reminders list."')
             script_lines.append('      set targetList to list "Reminders"')
             script_lines.append('    else')
             script_lines.append(f'      set targetList to list "{escaped_list}"')
             script_lines.append('    end if')
             script_lines.append('    -- Create the reminder')
             script_lines.append(f'    make new reminder at end of targetList with properties {{name:"{escaped_reminder}"}}')
             script_lines.append('  end tell')
             success_result_code = f'return "Created reminder: {escaped_reminder}"'
             action_implemented = True


        # --- Category: Communication (Placeholders - Require more complex scripts/permissions) ---
        # Example: Placeholder for sending text (requires Messages access & complex contact lookup)
        elif action_type == "basics" and params.get("send_text_message"):
             contact = params.get("contact_name", "Unknown")
             message = params.get("message_body", "")
             logger.warning("Send text message via AppleScript requested - Placeholder only.")
             success_result_code = f'return "Placeholder: Would attempt to send \'{escape_applescript_string(message)}\' to {escape_applescript_string(contact)}."'
             action_implemented = True # Mark as "handled" by placeholder

        # --- Add more ELIF blocks for other desired actions ---
        # elif action_type == "..." and params.get("..."):
        #    ... script lines ...
        #    action_implemented = True


        # --- Finalize Script Assembly ---
        if action_implemented:
            if success_result_code: # Add the return line if one was set
                script_lines.append(f'  {success_result_code}')
            # Add standard error handling block
            script_lines.append('on error errMsg number errNum')
            script_lines.append(f'  log "AppleScript Error ({req_id}) for Action \'{action_type}\': " & errMsg & " (" & errNum & ")"')
            # Return an error message that includes the AppleScript error
            script_lines.append('  return "Error executing action \'' + action_type + '\': " & errMsg')
            script_lines.append('end try')

            # Join lines into final script string
            final_script = "\n".join(script_lines)
            logger.info(f"{req_id} Generated AppleScript for '{action_type}'. Length: {len(final_script)}")
            logger.trace(f"{req_id} Generated Script:\n---\n{final_script}\n---")
            return final_script
        else:
            # No matching action implementation found
            logger.warning(f"{req_id} No specific AppleScript implemented for action '{action_type}' with params {params}. Cannot execute directly.")
            return None # Signal that script generation failed
        
    


    # --- NEW HELPER: Translation ---
    # app.py -> Inside AIChat class

    # --- NEW HELPER: Translation ---
    async def _translate(self, text: str, target_lang: str, source_lang: str = "auto") -> str:
        """
        Translates text using the configured translator model (ELP0).
        Handles TaskInterruptedException by re-raising it. Returns original text on other errors.
        """
        log_prefix = f"🌐 Translate|ELP0" # Add ELP0 marker
        # Session ID might not be directly available here unless passed in, using generic log for now
        # log_prefix = f"🌐 Translate|ELP0|{self.current_session_id}" if self.current_session_id else "🌐 Translate|ELP0"

        # Get the translator model instance
        translator_model = self.provider.get_model("translator")
        if not translator_model:
            logger.error(f"{log_prefix}: Translator model not available, cannot translate.")
            # Silently return original text, assuming caller handles this possibility
            return text

        logger.debug(f"{log_prefix}: Translating from '{source_lang}' to '{target_lang}': '{text[:50]}...'")
        try:
            # Prepare the prompt for the translation model
            prompt = f"Translate the following text from {source_lang} to {target_lang}:\n\n{text}"

            # --- Invoke the translation model using the timing helper with ELP0 ---
            # Prepare dummy interaction data if needed by _call_llm_with_timing for logging session
            timing_data = {"session_id": self.current_session_id, "mode": "chat"}
            message_result = await asyncio.to_thread(
                self._call_llm_with_timing, # Use the modified helper
                translator_model,           # Pass the model directly (assuming it's callable like a chain)
                prompt,                     # Input is the prompt string
                timing_data,
                priority=ELP0               # Set ELP0 priority
            )
            # ---

            # Check if the result is a message object and extract content
            # (This logic handles different ways Langchain models might return results)
            translated_text = None
            if hasattr(message_result, 'content') and isinstance(message_result.content, str):
                translated_text = message_result.content
            elif isinstance(message_result, str): # Handle if model directly returns a string
                translated_text = message_result
            else:
                # Handle unexpected return types from the translation model
                logger.error(f"{log_prefix}: Translation model returned unexpected type: {type(message_result)}. Full result: {message_result}")
                # Attempt to log this issue to the database if possible
                try:
                    db_session = SessionLocal()
                    add_interaction(db_session, session_id=self.current_session_id, mode="chat", input_type="log_error",
                                    user_input="[Translation Type Error]",
                                    llm_response=f"Unexpected type: {type(message_result)}. Data: {str(message_result)[:500]}")
                    db_session.close()
                except Exception as db_err:
                     logger.error(f"{log_prefix} Failed log translation type error: {db_err}")
                return text # Return original text on unexpected type error

            # Clean the extracted translated text
            cleaned_text = translated_text.strip() if translated_text else ""

            logger.debug(f"{log_prefix}: Translation result: '{cleaned_text[:50]}...'")
            # Return the successfully translated and cleaned text
            return cleaned_text

        except TaskInterruptedException as tie:
            # Specific handling for interruption caught by _call_llm_with_timing
            logger.warning(f"🚦 {log_prefix}: Translation INTERRUPTED: {tie}")
            # Re-raise the exception to be handled by the calling function
            raise tie

        except Exception as e:
            # Handle any other exceptions during translation
            logger.error(f"{log_prefix}: Translation failed: {e}")
            logger.exception(f"{log_prefix} Translate Traceback:") # Add traceback log
            # Attempt to log this failure to the database
            try:
                db_session = SessionLocal()
                add_interaction(db_session, session_id=self.current_session_id, mode="chat", input_type="log_error",
                                user_input="[Translation Failed]",
                                llm_response=f"Error: {e}. Original text: {text[:200]}")
                db_session.close()
            except Exception as db_err:
                 logger.error(f"{log_prefix} Failed log translation failure: {db_err}")
            # Return the original text as a fallback on error
            return text

    # --- NEW HELPER: Routing ---
    async def _route_to_specialist(self, db: Session, session_id: str, user_input: str, context: Dict) -> Tuple[str, str, str]:
        """
        Uses router LLM (ELP0) for analysis and coder LLM (ELP0) for JSON extraction.
        Handles interruptions by re-raising TaskInterruptedException.
        Retries JSON extraction on other errors. Ensures a valid model key is returned.
        """
        log_prefix = f"🧠 Route|ELP0|{session_id}" # Add ELP0 marker
        logger.info(f"{log_prefix}: Routing request...")

        router_model = self.provider.get_model("router")
        extractor_model = self.provider.get_model("code") # Use coder model for extraction
        default_model_key = "general" # Define the fallback key

        if not router_model or not extractor_model:
            logger.error(f"{log_prefix} Router or JSON Extractor (Coder) model not available! Falling back to default.")
            try: # Log error to DB
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_error",
                                user_input="Router/Extractor Init Fail",
                                llm_response="Router or Extractor model unavailable.")
            except Exception as db_err:
                 logger.error(f"{log_prefix} Failed log router/extractor model unavailable: {db_err}")
            # Return default key, original input, and error reason
            return default_model_key, user_input, "Router/Extractor model unavailable."

        # --- 1. Call Router Model (ELP0) ---
        logger.debug(f"{log_prefix}: Preparing and calling Router model...")
        prompt_input_router = {
            "input": user_input,
            "pending_tot_result": context.get("pending_tot_result", "None."),
            "recent_direct_history": context.get("recent_direct_history", "None."),
            "context": context.get("url_context", "None."),
            "history_rag": context.get("history_rag", "None."),
            "file_index_context": context.get("file_index_context", "None."),
            "log_context": context.get("log_context", "None."),
            "emotion_analysis": context.get("emotion_context_analysis", "N/A.")
        }
        router_chain = (ChatPromptTemplate.from_template(PROMPT_ROUTER) | router_model | StrOutputParser())
        router_timing_data = {"session_id": session_id, "mode": "chat", "execution_time_ms": 0}
        raw_router_output = "Error: Router LLM call failed." # Default

        try:
            # Use _call_llm_with_timing with ELP0 priority
            raw_router_output = await asyncio.to_thread(
                self._call_llm_with_timing, router_chain, prompt_input_router, router_timing_data, priority=ELP0
            )
            logger.trace(f"{log_prefix} Router Raw Output:\n{raw_router_output}")
            # Log the raw output for debugging purposes
            try: # Log to DB
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_debug",
                                user_input="Router Raw Output", llm_response=raw_router_output[:4000])
            except Exception as db_err:
                 logger.error(f"{log_prefix} Failed log raw router output: {db_err}")

        except TaskInterruptedException as tie:
            # Handle interruption specifically
            logger.warning(f"🚦 {log_prefix} Router step INTERRUPTED: {tie}")
            # Re-raise to be caught by the calling function (e.g., background_generate)
            raise tie
        except Exception as e:
            # Handle other errors during the initial router call
            logger.error(f"❌ {log_prefix} Error during initial routing LLM call: {e}")
            try: # Log error to DB
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_error",
                                user_input="Router Analysis Failed",
                                llm_response=f"Routing failed (LLM call): {e}")
            except Exception as db_err:
                 logger.error(f"{log_prefix} Failed log router LLM error: {db_err}")
            # Fallback to default if router fails
            return default_model_key, user_input, f"Routing LLM error: {e}"

        # --- 2. Call Extractor Model to Get JSON (ELP0 with Retries) ---
        logger.info(f"{log_prefix} Extracting JSON from router output...")
        prompt_input_extractor = {"raw_llm_output": raw_router_output}
        extractor_parser = JsonOutputParser() # Assuming this parser doesn't need modification
        extractor_chain = (ChatPromptTemplate.from_template(PROMPT_EXTRACT_JSON) | extractor_model | extractor_parser)
        extractor_timing_data = {"session_id": session_id, "mode": "chat", "execution_time_ms": 0}
        routing_result = None
        last_error = None

        for attempt in range(DEEP_THOUGHT_RETRY_ATTEMPTS):
             logger.debug(f"{log_prefix} JSON Extraction attempt {attempt + 1}/{DEEP_THOUGHT_RETRY_ATTEMPTS}")
             try:
                 # Use _call_llm_with_timing with ELP0 priority inside the loop
                 routing_result = await asyncio.to_thread(
                     self._call_llm_with_timing, extractor_chain, prompt_input_extractor, extractor_timing_data, priority=ELP0
                 )

                 # --- Validate the extracted JSON structure ---
                 valid_model_keys = {"vlm", "latex", "math", "code", "general"}
                 if isinstance(routing_result, dict) and routing_result.get("chosen_model") and routing_result.get("refined_query"):
                    chosen_model = routing_result["chosen_model"]
                    refined_query = routing_result["refined_query"]
                    reasoning = routing_result.get("reasoning", "N/A")

                    # Check if the chosen model key is valid
                    if chosen_model in valid_model_keys:
                         logger.info(f"✅ {log_prefix} JSON Extraction successful. Router chose: '{chosen_model}'. Reason: {reasoning}")
                         try: # Log success to DB
                             add_interaction(db, session_id=session_id, mode="chat", input_type="log_info",
                                             user_input="Router Final Decision",
                                             llm_response=f"Chose: {chosen_model}, Query: '{refined_query[:100]}...', Reason: {reasoning}",
                                             assistant_action_analysis_json=json.dumps(routing_result))
                         except Exception as db_err:
                              logger.error(f"{log_prefix} Failed log router final decision: {db_err}")
                         return chosen_model, refined_query, reasoning # Success, exit function

                    else:
                         # Handle invalid model key from extractor
                         logger.warning(f"{log_prefix} Extractor returned invalid model key '{chosen_model}' on attempt {attempt + 1}. Full result: {routing_result}")
                         last_error = ValueError(f"Invalid model key '{chosen_model}' from extractor")
                         try: # Log warning to DB
                              add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning",
                                              user_input=f"JSON Extractor Invalid Model Key (Attempt {attempt + 1})",
                                              llm_response=f"Invalid key: {chosen_model}. Parsed: {str(routing_result)[:500]}",
                                              assistant_action_analysis_json=json.dumps(routing_result or {}))
                         except Exception as db_err:
                              logger.error(f"{log_prefix} Failed log extractor invalid key: {db_err}")
                         # Continue to the next retry attempt if not the last one

                 else:
                     # Handle invalid JSON structure (valid JSON, but wrong keys/types)
                     logger.warning(f"{log_prefix} Extractor produced valid JSON but invalid structure on attempt {attempt + 1}: {routing_result}. Retrying...")
                     last_error = ValueError("Invalid JSON structure from extractor")
                     try: # Log warning to DB
                          add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning",
                                          user_input=f"JSON Extractor Invalid Structure (Attempt {attempt + 1})",
                                          llm_response=f"Parsed: {str(routing_result)[:500]}",
                                          assistant_action_analysis_json=json.dumps(routing_result or {}))
                     except Exception as db_err:
                          logger.error(f"{log_prefix} Failed log extractor invalid structure: {db_err}")
                     # Continue to the next retry attempt

             except TaskInterruptedException as tie:
                  # Handle interruption specifically - DO NOT RETRY
                  logger.warning(f"🚦 {log_prefix} JSON Extractor step INTERRUPTED (Attempt {attempt + 1}): {tie}")
                  # Re-raise immediately to be caught by the calling function
                  raise tie
             except Exception as e:
                 # Handle other errors during extraction (e.g., JSONDecodeError from parser, LLM call errors)
                 logger.warning(f"⚠️ {log_prefix} Error during JSON Extraction attempt {attempt + 1}: {e}")
                 last_error = e
                 try: # Log warning to DB
                     add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning",
                                     user_input=f"JSON Extractor FAILED (Attempt {attempt + 1})",
                                     llm_response=f"Error: {e}. Raw Router Output: {raw_router_output[:500]}")
                 except Exception as db_err:
                      logger.error(f"{log_prefix} Failed log extractor failure: {db_err}")
                 # Continue to the next retry attempt if not the last one

             # Wait before retrying if attempts remain and error wasn't interruption
             if attempt < DEEP_THOUGHT_RETRY_ATTEMPTS - 1:
                 await asyncio.sleep(0.5 + attempt * 0.5) # Use asyncio sleep as we are in async function
             else:
                 # Max retries reached for non-interruption errors
                 logger.error(f"❌ {log_prefix} Max retries ({DEEP_THOUGHT_RETRY_ATTEMPTS}) reached for JSON extraction. Last error: {last_error}")
                 # Fallback after max retries below

        # --- Fallback if all extraction attempts fail (excluding interruptions) ---
        logger.error(f"❌ {log_prefix} JSON Extraction attempts failed. Falling back to default.")
        try: # Log error to DB
            add_interaction(db, session_id=session_id, mode="chat", input_type="log_error",
                            user_input="Router Analysis Failed (Extraction)",
                            llm_response=f"JSON Extraction failed after retries. Last Error: {last_error}")
        except Exception as db_err:
             logger.error(f"{log_prefix} Failed log extraction fallback: {db_err}")
        # Return default key, original input, and reason for failure
        return default_model_key, user_input, "Router JSON extraction failed after retries."

    # --- generate method ---
    # app.py -> Inside AIChat class

    # --- generate (Main Async Method - Fuzzy History RAG + Direct History + Log Context + Multi-LLM Routing + VLM Preprocessing) ---
    # app.py -> Inside AIChat class
    async def direct_generate(self, db: Session, user_input: str, session_id: str, vlm_description: Optional[str] = None) -> str:
        """
        Handles the fast-path direct response generation with ELP1 priority.
        Includes Direct History and History RAG (with ELP1 embedding).
        Uses PROMPT_DIRECT_GENERATE from config.py.
        Handles potential errors but does not retry interruptions for ELP1.
        """
        direct_req_id = f"dgen-{uuid.uuid4()}"
        log_prefix = f"⚡️ {direct_req_id}|ELP1" # Add priority marker
        logger.info(f"{log_prefix} Direct Generate START --> Session: {session_id} (with History RAG)")
        direct_start_time = time.monotonic()

        # Use the fast LLM
        fast_model = self.provider.get_model("general_fast")
        if not fast_model:
            logger.error(f"{log_prefix}: Fast model 'general_fast' not available!")
            try: # Log error to DB
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_error",
                                user_input="Direct Generate Failed", llm_response="Fast model unavailable.")
            except Exception as db_err:
                 logger.error(f"{log_prefix} Failed log fast model error: {db_err}")
            return "Error: Cannot generate quick response."

        # Prepare input for the LLM prompt
        input_for_llm = user_input
        if vlm_description:
            input_for_llm = f"[Image Description: {vlm_description}]\n\nUser Query: {user_input or '(Query related to image)'}"
            logger.debug(f"{log_prefix}: Using provided VLM description in input.")
        else:
             input_for_llm = user_input # Standard text input

        # --- Fetch Context Concurrently (Direct History + History RAG with ELP1 embedding) ---
        history_rag_str = "No relevant history snippets found."
        recent_direct_history_str = "No recent direct history available."
        history_retriever = None
        # --- CORRECTION: Use local variable ---
        local_history_ids_used = "" # Initialize local variable
        # --- END CORRECTION ---

        try:
            logger.debug(f"{log_prefix}: Fetching direct history and setting up history RAG (Embedding ELP1)...")

            # Setup History RAG (embedding runs with ELP1)
            # _get_rag_retriever returns url_retriever, history_retriever, history_ids_str
            _, history_retriever, local_history_ids_used = await asyncio.to_thread( # Assign to local var
                self._get_rag_retriever, db, input_for_llm, priority=ELP1
            )
            # --- CORRECTION: Removed line modifying non-existent interaction_data ---
            # interaction_data['rag_history_ids'] = history_ids_used # REMOVED THIS LINE
            # --- END CORRECTION ---

            # Retrieve History RAG Docs
            history_docs = []
            if history_retriever:
                 try:
                     history_docs = await asyncio.to_thread(
                         history_retriever.invoke, input_for_llm
                     )
                     logger.debug(f"{log_prefix}: Retrieved {len(history_docs)} history RAG docs.")
                     history_rag_str = self._format_docs(history_docs, source_type="History RAG")
                 except TaskInterruptedException as tie:
                      logger.warning(f"{log_prefix}: Interruption detected during RAG doc retrieval? {tie}")
                      history_rag_str = "[History RAG retrieval interrupted]"
                 except Exception as rag_err:
                      logger.warning(f"{log_prefix}: Error retrieving/formatting history RAG docs: {rag_err}")
                      history_rag_str = "Error retrieving history snippets."
            else:
                 logger.debug(f"{log_prefix}: History RAG retriever not available or no history.")

            # Fetch Direct History
            temp_global_history = await asyncio.to_thread(get_global_recent_interactions, db, limit=3)
            recent_direct_history_str = self._format_direct_history(temp_global_history)

        except TaskInterruptedException as tie_context:
            logger.warning(f"🚦 {log_prefix}: Context fetching INTERRUPTED during embedding: {tie_context}")
            history_rag_str = "[History embedding interrupted]"
            if not recent_direct_history_str: recent_direct_history_str = "No recent direct history available."
        except Exception as context_err:
            logger.error(f"{log_prefix}: Error fetching context for direct path: {context_err}")
            history_rag_str = "Error retrieving history snippets."
            recent_direct_history_str = "Error retrieving direct history."

        # --- Setup Prompt and Chain ---
        try:
            direct_prompt_template = ChatPromptTemplate.from_template(PROMPT_DIRECT_GENERATE)
        except Exception as prompt_err:
             logger.error(f"{log_prefix}: Error creating direct prompt template: {prompt_err}. Using fallback.")
             direct_prompt_template = ChatPromptTemplate.from_messages([
                 ("system", "Provide a direct answer using history."),
                 ("user", "History RAG:\n{history_rag}\n\nRecent Direct:\n{recent_direct_history}\n\nQuery:\n{input}")
             ])
        direct_chain = direct_prompt_template | fast_model | StrOutputParser()

        # --- Call LLM with ELP1 Priority ---
        direct_timing_data = {"session_id": session_id, "mode": "chat", "execution_time_ms": 0}
        raw_response = "[Direct generation failed]"
        final_response = raw_response
        llm_call_successful = False

        try:
            logger.debug(f"{log_prefix}: Calling fast model with ELP1 priority...")
            prompt_inputs = {
                "input": input_for_llm,
                "recent_direct_history": recent_direct_history_str,
                "history_rag": history_rag_str
            }
            raw_response = await asyncio.to_thread(
                self._call_llm_with_timing, direct_chain, prompt_inputs, direct_timing_data, priority=ELP1
            )
            llm_call_successful = True
            logger.info(f"{log_prefix}: Fast model call complete. Raw length: {len(raw_response)}")
            final_response = self._cleanup_llm_output(raw_response)

        except TaskInterruptedException as tie_llm:
            logger.error(f"🚦 {log_prefix}: Direct LLM call (ELP1) INTERRUPTED? {tie_llm}")
            final_response = f"[Error: Direct generation (ELP1) was unexpectedly interrupted]"
            # Log error
            try: add_interaction(...)
            except Exception: pass
        except Exception as e:
            logger.error(f"❌ {log_prefix}: Error during direct LLM call: {e}")
            logger.exception(f"{log_prefix} Direct LLM Traceback:")
            final_response = f"[Error generating direct response: {e}]"
            # Log error
            try: add_interaction(...)
            except Exception: pass

        # --- Final Logging and Return ---
        direct_duration = (time.monotonic() - direct_start_time) * 1000
        logger.info(f"{log_prefix} Direct Generate END. Duration: {direct_duration:.2f}ms. Resp len: {len(final_response)}")

        # Log the interaction
        try:
            # --- CORRECTION: Create log data dict here, using local variables ---
            log_data_for_db = {
                "session_id": session_id, "mode": "chat",
                "input_type": "image+text" if vlm_description else "text",
                "user_input": user_input,
                "llm_response": final_response,
                "execution_time_ms": direct_duration,
                "image_description": vlm_description,
                "classification": "direct_response",
                "rag_history_ids": local_history_ids_used, # Use the local variable here
                "tot_delivered": False,
                "assistant_action_executed": False,
                "reflection_completed": False
            }
            # --- END CORRECTION ---

            # Filter kwargs to match Interaction model columns before saving
            valid_keys = {c.name for c in Interaction.__table__.columns}
            db_kwargs = {k: v for k, v in log_data_for_db.items() if k in valid_keys} # Use log_data_for_db
            await asyncio.to_thread(add_interaction, db, **db_kwargs)
        except Exception as log_err:
            logger.error(f"❌ {log_prefix}: Failed to log direct interaction: {log_err}")
            logger.exception(f"{log_prefix} DB Log Traceback:")

        return final_response

    # --- generate (Main Async Method - V15 version) ---
    async def background_generate(self, db: Session, user_input: str, session_id: str = None,
                                 classification: str = "chat_simple", image_b64: Optional[str] = None,
                                 update_interaction_id: Optional[int] = None):
        """
        Handles text/image-based generation in background (ELP0), including RAG, Actions,
        ToT, Routing, Translation, Correction, Logging, and ToT spawning.
        Gracefully handles TaskInterruptedException raised by ELP0 sub-tasks. V21.1.
        """
        # --- Initialization ---
        request_id = f"gen-{uuid.uuid4()}"
        is_reflection_task = update_interaction_id is not None
        log_prefix = f"🔄 REFLECT {request_id}" if is_reflection_task else f"💬 BGEN {request_id}"
        if not session_id: session_id = f"session_{int(time.time())}" if not is_reflection_task else f"reflection_for_{update_interaction_id}"
        self.current_session_id = session_id

        logger.info(f"{log_prefix} Async Chat generate START --> Session: {session_id}, Initial Class: '{classification}', Input: '{user_input[:50]}...', Img: {'Y' if image_b64 else 'N'}, Priority: ELP0")
        request_start_time = time.monotonic()

        # Interaction data dictionary, populated throughout the process
        interaction_data = {
            "session_id": session_id, "mode": "chat", "input_type": "text",
            "user_input": user_input, "llm_response": "[Processing...]", # Placeholder until final response
            "execution_time_ms": 0,
            "classification": classification, "classification_reason": None,
            "rag_history_ids": None, "rag_source_url": None,
            "requires_deep_thought": False, "deep_thought_reason": None,
            "tot_analysis_requested": False, "tot_result": None, "tot_delivered": False,
            "emotion_context_analysis": None, "image_description": None,
            "assistant_action_analysis_json": None, "assistant_action_type": None,
            "assistant_action_params": None, "assistant_action_executed": False,
            "assistant_action_result": None,
            "image_data": image_b64[:20] + "..." if image_b64 else None # Indicate presence/size
        }
        if image_b64: interaction_data["input_type"] = "image+text"

        final_response = "Error: Processing failed unexpectedly." # Default final response
        saved_interaction: Optional[Interaction] = None # For the initial user interaction record
        interaction_to_update: Optional[Interaction] = None # For reflection tasks
        interrupted_flag = False # Track if TaskInterruptedException was caught

        # --- Input Validation ---
        if not user_input and not image_b64:
            logger.warning(f"{request_id} Empty input (no text or image).")
            try:
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning", user_input="[Empty Request Received]", llm_response="No text or image provided.")
            except Exception as log_err:
                 logger.error(f"Failed to log empty request: {log_err}")
            # This function runs in the background, returning isn't directly useful to user
            # Log and exit the task.
            return

        # --- Load existing interaction if updating (Reflection Task) ---
        if is_reflection_task:
            try:
                interaction_to_update = db.query(Interaction).filter(Interaction.id == update_interaction_id).first()
                if not interaction_to_update:
                    logger.error(f"{log_prefix}: CRITICAL - Cannot find original interaction {update_interaction_id} to update. Aborting reflection task.")
                    # Log this failure clearly
                    try:
                         add_interaction(db, session_id=session_id, mode="chat", input_type="error",
                                         user_input=f"[Reflection Aborted - Missing Original ID {update_interaction_id}]",
                                         llm_response=f"Could not find interaction {update_interaction_id} for reflection update.")
                    except Exception as log_err:
                         logger.error(f"Failed to log missing reflection target: {log_err}")
                    return # Exit the function gracefully
                logger.info(f"{log_prefix}: Found original interaction {update_interaction_id} to update.")
            except Exception as load_err:
                 logger.error(f"{log_prefix}: Error loading interaction {update_interaction_id} for update: {load_err}")
                 # Attempt to log this load failure
                 try:
                      add_interaction(db, session_id=session_id, mode="chat", input_type="error",
                                      user_input=f"[Reflection Aborted - Error loading Original ID {update_interaction_id}]",
                                      llm_response=f"DB Error: {load_err}")
                 except Exception as log_err:
                      logger.error(f"Failed to log reflection target load error: {log_err}")
                 return # Exit if loading fails

        # --- Main Processing Block (Wrapped for Interruption Handling) ---
        try:
            # --- 0. VLM Preprocessing (If Image Provided) ---
            # This assumes VLM preprocessing (like getting image_description)
            # happened *before* this background task was called, or that if
            # it's done here, the VLM call uses ELP0 and handles interruption.
            current_input_for_analysis = user_input
            if image_b64:
                logger.info(f"{log_prefix} Image provided, attempting to use VLM description...")
                # Fetch description from interaction_data if it was pre-processed
                # In a real scenario, might need to pass it explicitly or re-fetch from DB
                vlm_description = interaction_data.get('image_description')
                if not vlm_description:
                    # Attempt to get description from DB if this task was triggered by an image interaction
                    try:
                         img_interaction = db.query(Interaction).filter(
                             Interaction.session_id == session_id,
                             Interaction.input_type == 'image+text', # Or 'image' if logged separately
                             Interaction.user_input == user_input # Match on input? Risky. Better to pass ID.
                         ).order_by(desc(Interaction.timestamp)).first()
                         if img_interaction and img_interaction.image_description:
                             vlm_description = img_interaction.image_description
                             interaction_data['image_description'] = vlm_description # Store it
                             logger.info(f"{log_prefix} Found image description in previous interaction {img_interaction.id}.")
                         else:
                             logger.warning(f"{log_prefix} Image provided but no description found in interaction_data or recent history.")
                             # Optionally, could call VLM here with ELP0, but adds complexity.
                    except Exception as db_desc_err:
                         logger.error(f"{log_prefix} Error fetching image description from DB: {db_desc_err}")

                # Update the input used for analysis if description was found
                if vlm_description:
                     current_input_for_analysis = f"[Image Description: {vlm_description}]\n\nUser Query: {user_input or '(Query related to image)'}"

                # Handle image-only request case? Unlikely needed for background task.
                # If needed: if not user_input and vlm_description: final_response = vlm_description; raise StopIteration("Image only") # Use custom signal?

            # --- 1. Check for Pending ToT Result ---
            logger.debug(f"{log_prefix} Checking for pending ToT results for session {session_id}...")
            pending_tot_interaction = await asyncio.to_thread(get_pending_tot_result, db, session_id)
            pending_tot_str = "None."
            if pending_tot_interaction and pending_tot_interaction.tot_result:
                logger.info(f"{log_prefix} Injecting pending ToT result from Interaction ID: {pending_tot_interaction.id}")
                original_input_snippet = (pending_tot_interaction.user_input or "your previous query")[:50] + "..."
                pending_tot_str = ( f"Okay, regarding your previous query ('{original_input_snippet}'), "
                                    f"I've finished thinking about it:\n\n{pending_tot_interaction.tot_result}\n\n"
                                    f"Now, about your current request:" )

            # --- 2. Generate File Search Query (ELP0) ---
            # Ensure _generate_file_search_query_async uses ELP0 internally
            logger.debug(f"{log_prefix} Generating file search query (ELP0)...")
            temp_global_history = await asyncio.to_thread(get_global_recent_interactions, db, limit=5)
            temp_recent_direct_history_str = self._format_direct_history(temp_global_history)
            file_search_query = await self._generate_file_search_query_async(db, current_input_for_analysis, temp_recent_direct_history_str, session_id)


            # --- 3. CONCURRENT FETCHING & CONTEXT PREP (Embedding uses ELP0) ---
            logger.debug(f"{log_prefix} Starting concurrent context fetching (Embedding: ELP0)...")
            # Ensure _get_rag_retriever uses ELP0 internally for embedding & handles interruption
            url_retriever, history_retriever, history_ids_used = await asyncio.to_thread( self._get_rag_retriever, db, current_input_for_analysis )
            interaction_data['rag_history_ids'] = history_ids_used
            if hasattr(self, 'vectorstore_url') and self.vectorstore_url:
                interaction_data['rag_source_url'] = getattr(self.vectorstore_url, '_source_url', None)

            # Define async helper tasks
            async def retrieve_docs_async():
                 # RAG retrieval doesn't involve LLM calls, just vector similarity search
                 retrieve_docs_step = RunnableParallel( url_docs=(url_retriever if url_retriever else RunnableLambda(lambda x: [])), history_docs=(history_retriever if history_retriever else RunnableLambda(lambda x: [])) ); return await asyncio.to_thread(retrieve_docs_step.invoke, current_input_for_analysis)
            async def get_logs_async(): # DB Query
                 log_pool = await asyncio.to_thread(get_recent_interactions, db, RAG_HISTORY_COUNT*2, session_id, "chat", include_logs=True); return [i for i in log_pool if i.input_type.startswith('log_') or i.input_type == 'error']
            async def get_global_history_async(): # DB Query
                 return await asyncio.to_thread(get_global_recent_interactions, db, limit=10)
            async def run_emotion_analysis_async(): # LLM Call (Needs ELP0)
                 # Ensure _run_emotion_analysis uses _call_llm_with_timing(priority=ELP0) internally
                 await asyncio.to_thread(self._run_emotion_analysis, db, user_input, interaction_data)
            async def get_file_index_results_async(query_to_use: str): # DB Query
                if search_file_index and query_to_use: return await asyncio.to_thread(search_file_index, db, query_to_use, limit=5)
                else: return []

            # Execute concurrent tasks
            logger.debug(f"{log_prefix} Waiting for parallel fetches & analysis...")
            results = await asyncio.gather(
                retrieve_docs_async(), get_logs_async(), get_global_history_async(),
                run_emotion_analysis_async(), get_file_index_results_async(file_search_query),
                return_exceptions=True
            )
            logger.debug(f"{log_prefix} Parallel fetches & analysis completed.")

            # Process results, checking for exceptions (especially TaskInterruptedException)
            retrieved_docs = None; log_entries_for_context = []; global_direct_history = []; file_index_results = []
            for i, res in enumerate(results):
                 if isinstance(res, TaskInterruptedException):
                      logger.warning(f"🚦 Interruption caught during concurrent context fetch (Task {i})")
                      raise res # Re-raise interruption immediately
                 elif isinstance(res, BaseException):
                      logger.error(f"{log_prefix} Error in concurrent context task {i}: {res}")
                      # Assign default empty values for failed tasks
                      if i == 0: retrieved_docs = {'url_docs': [], 'history_docs': []}
                      elif i == 1: log_entries_for_context = []
                      elif i == 2: global_direct_history = []
                      # Emotion analysis modifies interaction_data directly
                      elif i == 4: file_index_results = []
                 else:
                      # Assign successful results
                      if i == 0: retrieved_docs = res
                      elif i == 1: log_entries_for_context = res
                      elif i == 2: global_direct_history = res
                      # Emotion analysis modifies interaction_data
                      elif i == 4: file_index_results = res

            # Fallback if retrieval failed badly
            if retrieved_docs is None: retrieved_docs = {'url_docs': [], 'history_docs': []}
            url_docs = retrieved_docs.get("url_docs", [])
            history_docs = retrieved_docs.get("history_docs", [])
            emotion_analysis_result = interaction_data.get('emotion_context_analysis', "Analysis Unavailable or Failed")

            emotion_log_str = str(emotion_analysis_result)[:50] if emotion_analysis_result is not None else "None"
            logger.debug(f"{log_prefix} Context Fetched - URL Docs: {len(url_docs)}, Hist Docs: {len(history_docs)}, FileIdx Docs: {len(file_index_results)}, Logs: {len(log_entries_for_context)}, Global Hist: {len(global_direct_history)}, Emotion: '{emotion_log_str}...'")

            # Format Context Strings
            logger.debug(f"{log_prefix} Formatting context strings...")
            url_context_str = self._format_docs(url_docs, source_type="URL Context")
            history_rag_str = self._format_docs(history_docs, source_type="History RAG")
            log_context_str = self._format_log_history(log_entries_for_context)
            recent_direct_history_str = self._format_direct_history(global_direct_history)
            history_summary_for_action = self._get_history_summary(db, MEMORY_SIZE) # Sync DB call is ok here
            file_index_context_str = self._format_file_index_results(file_index_results)

            # --- 4. Analyze for Assistant Action (ELP0) ---
            # Ensure _analyze_assistant_action uses ELP0 internally and handles interruptions
            logger.debug(f"{log_prefix} Analyzing for assistant action (ELP0)...")
            action_analysis_context = { "history_summary": history_summary_for_action, "log_context": log_context_str, "recent_direct_history": recent_direct_history_str }
            action_details = await asyncio.to_thread( self._analyze_assistant_action, db, current_input_for_analysis, session_id, action_analysis_context )

            action_type_detected = "no_action"
            if action_details:
                 action_type_detected = action_details.get("action_type", "no_action")
                 interaction_data['assistant_action_analysis_json'] = json.dumps(action_details)
                 interaction_data['assistant_action_type'] = action_type_detected
                 interaction_data['assistant_action_params'] = json.dumps(action_details.get("parameters", {}))
                 logger.info(f"{log_prefix} Assistant action analysis result: Type='{action_type_detected}'")
            else:
                 logger.info(f"{log_prefix} Assistant action analysis result: None (implies 'no_action')")
                 interaction_data['assistant_action_analysis_json'] = None; interaction_data['assistant_action_type'] = None; interaction_data['assistant_action_params'] = None

            # --- 5. SAVE INITIAL/PLACEHOLDER INTERACTION RECORD ---
            # Do not save if this is a reflection task (we update existing instead)
            if not is_reflection_task:
                logger.debug(f"{log_prefix} Saving initial interaction record (placeholder)...")
                try:
                    initial_save_data = interaction_data.copy() # Take snapshot
                    initial_save_data['llm_response'] = "[Action/Response Pending]"
                    initial_save_data['execution_time_ms'] = (time.monotonic() - request_start_time) * 1000
                    # Ensure all needed defaults are set before saving
                    initial_save_data.setdefault('classification_reason', None); initial_save_data.setdefault('rag_history_ids', history_ids_used or None);
                    initial_save_data.setdefault('rag_source_url', interaction_data.get('rag_source_url'));
                    initial_save_data.setdefault('requires_deep_thought', False); initial_save_data.setdefault('deep_thought_reason', None);
                    initial_save_data.setdefault('tot_analysis_requested', False); initial_save_data.setdefault('tot_result', None);
                    initial_save_data.setdefault('tot_delivered', False); initial_save_data.setdefault('emotion_context_analysis', emotion_analysis_result or 'N/A');
                    initial_save_data.setdefault('image_description', interaction_data.get('image_description'));
                    initial_save_data.setdefault('assistant_action_analysis_json', interaction_data.get('assistant_action_analysis_json'));
                    initial_save_data.setdefault('assistant_action_type', interaction_data.get('assistant_action_type'));
                    initial_save_data.setdefault('assistant_action_params', interaction_data.get('assistant_action_params'));
                    initial_save_data.setdefault('assistant_action_executed', False); initial_save_data.setdefault('assistant_action_result', None);
                    initial_save_data.setdefault('image_data', interaction_data.get('image_data'));

                    valid_keys = {c.name for c in Interaction.__table__.columns}
                    db_kwargs = {k: v for k, v in initial_save_data.items() if k in valid_keys}
                    # Call add_interaction synchronously using asyncio.to_thread
                    saved_interaction = await asyncio.to_thread(add_interaction, db, **db_kwargs)
                    if saved_interaction:
                        logger.info(f"{log_prefix} Saved initial interaction record ID {saved_interaction.id}")
                    else:
                         # add_interaction handles its own errors/rollback, but log failure here
                         logger.error(f"{log_prefix} CRITICAL: add_interaction failed to save initial record!")
                         # Should we halt processing if initial save fails? Depends on desired behavior.
                         # For now, we continue but saved_interaction will be None.
                except Exception as db_err:
                    logger.error(f"❌ {log_prefix} Failed to save initial interaction record: {db_err}")
                    logger.exception("Initial Save Traceback:")
                    saved_interaction = None # Ensure it's None if save failed

            # --- 6. Execute Action (ELP0 internally) OR Generate LLM Response (ELP0 pipeline) ---
            if action_details and action_type_detected != "no_action":
                # --- Execute Assistant Action ---
                logger.info(f"{log_prefix} Action '{action_type_detected}' required. Executing (ELP0)...")
                # Ensure _execute_assistant_action handles ELP0 for LLM calls AND interruptions
                target_interaction_for_action = saved_interaction if not is_reflection_task else interaction_to_update
                if target_interaction_for_action:
                    # _execute_assistant_action is async, await it directly
                    action_result = await self._execute_assistant_action( db, session_id, action_details, target_interaction_for_action )
                    final_response = action_result # Use result from executor (success msg or fallback)
                    # Status (e.g., assistant_action_executed) is updated within _execute_assistant_action
                else:
                     logger.error(f"{log_prefix} Cannot execute action: target interaction record missing (initial save failed or reflection target lost).")
                     final_response = "Error: Internal issue preventing action execution (missing record)."
                     # Update dict for final log save attempt later
                     interaction_data['llm_response'] = final_response
                     interaction_data['assistant_action_result'] = final_response # Log action result failure
                     interaction_data['assistant_action_executed'] = False # Explicitly mark as not executed

            else:
                # --- No Action Required - Proceed with LLM Generation Pipeline (All steps ELP0) ---
                logger.info(f"{log_prefix} No specific assistant action detected, proceeding with LLM generation (ELP0 pipeline).")

                # --- Route (ELP0) ---
                # _route_to_specialist handles ELP0 internally & raises TaskInterruptedException
                logger.debug(f"{log_prefix} Routing to specialist (ELP0)...")
                router_context = {
                    "pending_tot_result": pending_tot_str,
                    "recent_direct_history": recent_direct_history_str,
                    "url_context": url_context_str,
                    "history_rag": history_rag_str,
                    "file_index_context": file_index_context_str,
                    "log_context": log_context_str,
                    "emotion_analysis": emotion_analysis_result
                }
                # _route_to_specialist is async, await it
                chosen_model_key, refined_query, routing_reason = await self._route_to_specialist( db, session_id, current_input_for_analysis, router_context )
                logger.info(f"{log_prefix} Router chose '{chosen_model_key}'. Reason: {routing_reason}. Query: '{refined_query[:50]}...'")

                # --- Translate Input (ELP0) ---
                # _translate handles ELP0 internally & raises TaskInterruptedException
                specialist_input = refined_query; target_lang = "en"; requires_translation = False
                if chosen_model_key in ["math", "code"]:
                    translator = self.provider.get_model("translator")
                    if translator:
                        logger.warning(f"{log_prefix} Translating input for {chosen_model_key} (ELP0)...")
                        specialist_input = await self._translate(refined_query, target_lang="zh") # _translate is async
                        target_lang = "zh"
                        requires_translation = True
                    else: logger.error(f"{log_prefix} Translator unavailable for {chosen_model_key}.")

                # --- Get Specialist Model ---
                logger.debug(f"{log_prefix} Getting specialist model: {chosen_model_key}")
                specialist_model = self.provider.get_model(chosen_model_key)
                if not specialist_model:
                     logger.error(f"{log_prefix} Fatal: Specialist model '{chosen_model_key}' not found!")
                     raise ValueError(f"Specialist model '{chosen_model_key}' unavailable!")

                # --- Prepare and Call Specialist Chain (ELP0) ---
                specialist_chain_input_map = {
                    "input": refined_query, # Use the refined query
                    "emotion_analysis": emotion_analysis_result,
                    "context": url_context_str,
                    "history_rag": history_rag_str,
                    "file_index_context": file_index_context_str,
                    "log_context": log_context_str,
                    "recent_direct_history": recent_direct_history_str,
                    "pending_tot_result": pending_tot_str
                }
                specialist_chain = (
                     RunnableLambda(lambda x: specialist_chain_input_map) # Use lambda to inject map
                     | self.text_prompt_template # Use the standard chat template
                     | specialist_model
                     | StrOutputParser()
                )
                logger.info(f"{log_prefix} Calling Specialist Model '{chosen_model_key}' (ELP0)...")
                specialist_timing_data = {"session_id": session_id, "mode": "chat", "execution_time_ms": 0}
                # Execute the specialist call using the timing helper with ELP0
                draft_response = await asyncio.to_thread(
                    self._call_llm_with_timing, specialist_chain, {}, specialist_timing_data, priority=ELP0 # Pass empty dict if lambda handles input map
                )
                logger.info(f"{log_prefix} Specialist model call complete. Draft length: {len(draft_response)}")

                # --- Translate Output (ELP0) ---
                # _translate handles ELP0 internally & raises TaskInterruptedException
                if requires_translation:
                    translator = self.provider.get_model("translator")
                    if translator:
                        logger.warning(f"{log_prefix} Translating response back to English (ELP0)...")
                        draft_response = await self._translate(draft_response, target_lang="en", source_lang=target_lang) # _translate is async
                    else: logger.error(f"{log_prefix} Translator unavailable for back-translation.")

                # --- Call Corrector Model (ELP0) ---
                # _correct_response handles ELP0 internally & raises TaskInterruptedException
                logger.info(f"{log_prefix} Passing draft to corrector model (ELP0)...")
                corrector_context = {
                    "url_context": url_context_str,
                    "history_rag": history_rag_str,
                    "file_index_context": file_index_context_str,
                    "log_context": log_context_str,
                    "recent_direct_history": recent_direct_history_str,
                    "emotion_analysis": emotion_analysis_result
                }
                # _correct_response is async, await it
                llm_final_response = await self._correct_response( db, session_id, current_input_for_analysis, corrector_context, draft_response )
                logger.info(f"{log_prefix} Corrector model call complete. Final length: {len(llm_final_response)}")
                final_response = llm_final_response # Assign to final_response

            # --- 7. Post-Execution/Generation Steps (ToT Spawning uses ELP0 internally) ---
            logger.debug(f"{log_prefix} Post-execution/generation steps...")
            target_interaction_for_tot = saved_interaction if not is_reflection_task else interaction_to_update

            # Determine if ToT should be spawned based on initial classification AND if an action was NOT attempted/required
            action_attempted_or_required = (action_details and action_type_detected != "no_action")
            # Use initial classification stored in interaction_data
            should_spawn_tot = (interaction_data.get('classification') == "chat_complex") and not action_attempted_or_required

            # Update ToT flags on the target interaction record if it exists
            if target_interaction_for_tot:
                 try:
                      needs_commit = False
                      if target_interaction_for_tot.requires_deep_thought != should_spawn_tot: target_interaction_for_tot.requires_deep_thought = should_spawn_tot; needs_commit = True
                      # Use classification reason stored earlier
                      new_reason = interaction_data.get('classification_reason', 'N/A' if should_spawn_tot else None)
                      if target_interaction_for_tot.deep_thought_reason != new_reason: target_interaction_for_tot.deep_thought_reason = new_reason; needs_commit = True
                      if target_interaction_for_tot.tot_analysis_requested != should_spawn_tot: target_interaction_for_tot.tot_analysis_requested = should_spawn_tot; needs_commit = True
                      if needs_commit:
                          db.commit()
                          logger.debug(f"{log_prefix} Updated ToT flags on interaction {target_interaction_for_tot.id}.")
                 except Exception as tot_db_err:
                      logger.error(f"{log_prefix} Failed update ToT flags on DB {target_interaction_for_tot.id}: {tot_db_err}")
                      db.rollback()

            # Spawn Background ToT Task if needed AND target interaction exists
            if should_spawn_tot and target_interaction_for_tot:
                 logger.warning(f"⏳ {log_prefix} Spawning background ToT task (Trigger ID: {target_interaction_for_tot.id})...")
                 # Ensure _run_tot_in_background_wrapper uses ELP0 internally for its LLM call
                 tot_inputs_for_bg = {
                     "db_session_factory": SessionLocal, "input": user_input,
                     "rag_context_docs": url_docs,
                     "history_rag_interactions": history_docs, # Pass the retrieved docs/interactions
                     "log_context_str": log_context_str,
                     "recent_direct_history_str": recent_direct_history_str,
                     "file_index_context_str": file_index_context_str,
                     "triggering_interaction_id": target_interaction_for_tot.id,
                 }
                 # Create task - no await needed here, it runs in background
                 asyncio.create_task(self._run_tot_in_background_wrapper(**tot_inputs_for_bg))
                 logger.info(f"{log_prefix} Background ToT task scheduled.")
            elif should_spawn_tot and not target_interaction_for_tot:
                 logger.error(f"{log_prefix} Cannot spawn ToT: target interaction record missing.")

            # --- Mark Pending ToT as Delivered (if one was injected earlier) ---
            if pending_tot_interaction:
                logger.debug(f"{log_prefix} Marking pending ToT {pending_tot_interaction.id} as delivered...")
                # Run the synchronous DB update in a thread
                await asyncio.to_thread(mark_tot_delivered, db, pending_tot_interaction.id)

            # --- Final Return Value Preparation ---
            # Cleanup happens just before final save/update in finally block

        # --- Catch Interruption Exception ---
        except TaskInterruptedException as tie:
            logger.warning(f"🚦 {log_prefix} Task INTERRUPTED during execution: {tie}")
            interrupted_flag = True # Set flag to handle in finally block
            final_response = f"[Task Interrupted by Higher Priority Request: {tie}]" # Set final response message

        # --- Top-Level Exception Handling ---
        except Exception as e:
            logger.error(f"❌❌ {log_prefix} UNHANDLED exception in background_generate: {e}")
            logger.exception(f"{log_prefix} Traceback:")
            final_response = f"Error during background processing: {type(e).__name__} - {e}"
            # Update interaction_data for saving error state in finally block
            interaction_data['llm_response'] = final_response[:4000] # Store error
            interaction_data['input_type'] = 'error' # Mark as error state

        finally:
            # --- Final Save/Update Logic (Handles Normal, Error, Interruption) ---
            final_db_data_to_save = interaction_data.copy() # Use latest data
            # Apply final cleanup to the response before saving
            final_response = self._cleanup_llm_output(final_response)
            final_db_data_to_save['llm_response'] = final_response
            final_db_data_to_save['execution_time_ms'] = (time.monotonic() - request_start_time) * 1000

            # Determine the correct interaction object to update/reference
            target_interaction_for_final_update = saved_interaction if not is_reflection_task else interaction_to_update

            try:
                if interrupted_flag:
                    # --- Handle Interruption Save ---
                    logger.warning(f"{log_prefix}: Finalizing state for INTERRUPTED task.")
                    if is_reflection_task and interaction_to_update:
                        # Reset reflection status, add note
                        interaction_to_update.reflection_completed = False # Ensure it gets picked up again
                        # Append interruption note, avoiding excessive length
                        existing_resp = interaction_to_update.llm_response or ""
                        interrupt_note = f"\n\n--- Reflection Interrupted ({datetime.datetime.now(datetime.timezone.utc).isoformat()}) ---"
                        interaction_to_update.llm_response = (existing_resp + interrupt_note)[:Interaction.llm_response.type.length]
                        interaction_to_update.last_modified_db = datetime.datetime.now(datetime.timezone.utc)
                        db.commit()
                        logger.info(f"{log_prefix}: Marked original interaction {update_interaction_id} as reflection NOT completed (interrupted).")
                    elif not is_reflection_task and saved_interaction:
                         # Update standard task state to interrupted
                         saved_interaction.llm_response = final_response # Store interruption message
                         saved_interaction.classification = "task_failed_interrupted" # Specific classification
                         saved_interaction.execution_time_ms = final_db_data_to_save['execution_time_ms']
                         saved_interaction.input_type = 'log_warning' # Indicate non-fatal issue
                         db.commit()
                         logger.info(f"{log_prefix}: Updated interaction {saved_interaction.id} state to interrupted.")
                    else:
                         # Interrupted before initial save or during reflection with no target? Log it.
                         logger.warning(f"{log_prefix}: Task interrupted but no primary interaction record found/saved. Logging interruption event.")
                         add_interaction(db, session_id=session_id, mode="chat", input_type='log_warning',
                                         user_input=f"[Interrupted Background Task {request_id}]",
                                         llm_response=final_response)
                else:
                    # --- Handle Normal/Error Save ---
                    if is_reflection_task and interaction_to_update:
                        # Create NEW record for successful/failed reflection result
                        interaction_to_update.reflection_completed = True # Mark original complete
                        interaction_to_update.last_modified_db = datetime.datetime.now(datetime.timezone.utc)
                        db.commit() # Commit original update first

                        # Create and add new interaction record for the reflection result
                        new_interaction_data = {
                            "session_id": session_id,
                            "mode": "chat",
                            "input_type": "reflection_result" if final_db_data_to_save.get('input_type') != 'error' else 'error', # Use 'error' if reflection failed
                            "user_input": f"[Self-Reflection Result for Interaction ID {update_interaction_id}]",
                            "llm_response": final_response, # The reflection result or error message
                            "execution_time_ms": final_db_data_to_save.get('execution_time_ms', 0),
                            "classification": final_db_data_to_save.get('classification', 'reflection'), # Use 'error' if applicable
                            "reflection_completed": False, # New record isn't reflected yet
                            "tot_delivered": False,
                            "assistant_action_executed": False,
                        }
                        # Use synchronous add_interaction helper
                        await asyncio.to_thread(add_interaction, db, **new_interaction_data)
                        logger.info(f"{log_prefix}: Saved reflection result as new interaction.")

                    elif not is_reflection_task and saved_interaction:
                         # Update the existing standard interaction record with final results
                         logger.debug(f"{log_prefix}: Updating interaction {saved_interaction.id} with final results.")
                         saved_interaction.llm_response = final_response
                         saved_interaction.execution_time_ms = final_db_data_to_save['execution_time_ms']
                         saved_interaction.classification = final_db_data_to_save.get('classification', saved_interaction.classification)
                         saved_interaction.input_type = final_db_data_to_save.get('input_type', saved_interaction.input_type)
                         # Update other relevant fields based on the processing outcome
                         saved_interaction.rag_history_ids = final_db_data_to_save.get('rag_history_ids', saved_interaction.rag_history_ids)
                         saved_interaction.rag_source_url = final_db_data_to_save.get('rag_source_url', saved_interaction.rag_source_url)
                         saved_interaction.emotion_context_analysis = final_db_data_to_save.get('emotion_context_analysis', saved_interaction.emotion_context_analysis)
                         saved_interaction.image_description = final_db_data_to_save.get('image_description', saved_interaction.image_description)
                         saved_interaction.assistant_action_analysis_json = final_db_data_to_save.get('assistant_action_analysis_json', saved_interaction.assistant_action_analysis_json)
                         saved_interaction.assistant_action_type = final_db_data_to_save.get('assistant_action_type', saved_interaction.assistant_action_type)
                         saved_interaction.assistant_action_params = final_db_data_to_save.get('assistant_action_params', saved_interaction.assistant_action_params)
                         saved_interaction.assistant_action_executed = final_db_data_to_save.get('assistant_action_executed', saved_interaction.assistant_action_executed)
                         saved_interaction.assistant_action_result = final_db_data_to_save.get('assistant_action_result', saved_interaction.assistant_action_result)
                         # ToT flags should have been updated earlier if needed
                         db.commit()
                         logger.info(f"{log_prefix}: Updated interaction {saved_interaction.id} with final results.")
                    elif not is_reflection_task and not saved_interaction:
                        # Initial save failed, but task completed without error/interruption? Save final state as new.
                        logger.warning(f"{log_prefix}: Initial interaction save failed, saving final state as new record.")
                        valid_keys = {c.name for c in Interaction.__table__.columns}
                        db_kwargs = {k: v for k, v in final_db_data_to_save.items() if k in valid_keys}
                        await asyncio.to_thread(add_interaction, db, **db_kwargs) # Use sync helper in thread
                    else: # Should not happen (e.g., reflection task but no interaction_to_update)
                         logger.error(f"{log_prefix}: Final Save Logic Error - Reached unexpected state.")

            except Exception as final_save_err:
                 logger.error(f"❌ {log_prefix}: Failed final DB save/update: {final_save_err}")
                 logger.exception(f"{log_prefix} Final Save Traceback:")
                 try:
                     db.rollback() # Rollback potential partial changes
                 except Exception as rb_err:
                     logger.error(f"{log_prefix} Rollback after final save error FAILED: {rb_err}")

            # Log final status
            final_status = 'Interrupted' if interrupted_flag else ('Error' if final_db_data_to_save.get('input_type') == 'error' else 'Success')
            logger.info(f"{log_prefix} END. Final Status: {final_status}. Duration: {final_db_data_to_save['execution_time_ms']:.2f}ms")


    # --- reset Method ---
    def reset(self, db: Session, session_id: str = None):
        """Resets Chat mode state for the session."""
        logger.warning(f"🔄 Resetting Chat state. (Session: {session_id})")
        self.vectorstore_url = None
        self.vectorstore_history = None
        self.current_session_id = None
        logger.info("🧹 Chat URL Vectorstore and History context cleared.")
        try:
            add_interaction(db, session_id=session_id, mode="chat", input_type='system', user_input='Chat Session Reset Requested', llm_response='Chat state cleared.')
        except Exception as db_err:
            logger.error(f"Failed to log chat reset: {db_err}")
        return "Chat state cleared."


    # --- Image/URL Processing Methods (Synchronous, Corrected Syntax) ---
    async def _run_image_latex_analysis_stream(self, db: Session, session_id: str, image_content_part: Dict, user_input: str, interaction_data: dict):
        """
        Async generator for image analysis (LaTeX/TikZ).
        Yields progress updates and final token stream as SSE-formatted data chunks.
        Parses the full response afterwards and yields the structured result.

        Args:
            db: SQLAlchemy Session object.
            session_id: The current session ID.
            image_content_part: Dictionary representing the image data for the LLM.
            user_input: The original user text query accompanying the image.
            interaction_data: Dictionary holding data about the current interaction (used for logging context).

        Yields:
            str: SSE formatted strings containing status updates, token deltas, errors,
                 or the final parsed data structure.
        """
        stream_id = f"latex-stream-{uuid.uuid4()}"
        task_start_time = time.monotonic()
        logger.info(f"📸 {stream_id}: Starting STREAMING analysis for LaTeX/TikZ. Input: '{user_input[:50]}...'")

        # Yield initial status
        try:
            yield format_sse({"status": "Initializing LaTeX/Visual Model...", "stream_id": stream_id}, event_type="progress")
        except Exception as yield_err:
             logger.error(f"Error yielding initial progress for {stream_id}: {yield_err}")
             return # Stop if we can't even yield

        # --- Get Model ---
        latex_model = self.provider.get_model("latex")
        if not latex_model:
            error_msg = "LaTeX/Visual model (e.g., LatexMind) not configured."
            logger.error(f"❌ {stream_id}: {error_msg}")
            yield format_sse({"error": error_msg, "final": True, "stream_id": stream_id}, event_type="error")
            # Attempt to log error to DB (best effort)
            try: add_interaction(db, session_id=session_id, mode="chat", input_type="error", user_input="[Image LaTeX/TikZ Init Failed]", llm_response=error_msg)
            except Exception as db_log_err: logger.error(f"Failed to log LaTeX model config error: {db_log_err}")
            return # Stop generation

        # --- Prepare LLM Call ---
        # Combine image and the specific prompt from config.py
        messages = [HumanMessage(content=[image_content_part, {"type": "text", "text": PROMPT_IMAGE_TO_LATEX}])]
        # Ensure the chain uses the correct model instance
        chain = latex_model | StrOutputParser() # Assumes StrOutputParser works with stream

        yield format_sse({"status": "Sending request to LaTeX/Visual Model...", "stream_id": stream_id}, event_type="progress")
        logger.trace(f"{stream_id}: LaTeX/VLM input messages: {messages}")

        full_response_markdown = ""
        llm_call_start_time = time.monotonic()
        llm_execution_time_ms = 0
        final_status = "success" # Assume success unless error occurs

        try:
            # --- Stream LLM Response ---
            token_count = 0
            async for chunk in chain.astream(messages):
                # Ensure chunk is a string before processing
                if isinstance(chunk, str):
                    full_response_markdown += chunk
                    token_count += 1 # Approximate token count
                    # Yield token chunk (default 'data' event)
                    yield format_sse({"delta": chunk, "stream_id": stream_id})
                elif chunk is not None: # Log unexpected non-string chunks
                    logger.warning(f"{stream_id}: Received non-string chunk during stream: {type(chunk)} - {str(chunk)[:100]}")

            llm_execution_time_ms = (time.monotonic() - llm_call_start_time) * 1000
            logger.info(f"📄 {stream_id}: LaTeX/VLM Raw Stream Complete. Approx Tokens: {token_count}, Duration: {llm_execution_time_ms:.2f} ms. Final Length: {len(full_response_markdown)}")
            yield format_sse({"status": "LLM stream complete. Processing response...", "stream_id": stream_id}, event_type="progress")

            # --- Parse the Completed Markdown Response ---
            logger.debug(f"{stream_id}: Parsing full response...")
            description = full_response_markdown # Default
            latex_code = None
            tikz_code = None
            explanation = None

            # Regex to find ```latex ... ``` block
            latex_match = re.search(r"```latex\s*(.*?)\s*```", full_response_markdown, re.DOTALL)
            if latex_match:
                latex_code = latex_match.group(1).strip()
                logger.info(f"{stream_id}: Found LaTeX code block ({len(latex_code)} chars).")

            # Regex to find ```tikz ... ``` block
            tikz_match = re.search(r"```tikz\s*(.*?)\s*```", full_response_markdown, re.DOTALL)
            if tikz_match:
                tikz_code = tikz_match.group(1).strip()
                logger.info(f"{stream_id}: Found TikZ code block ({len(tikz_code)} chars).")

            # Attempt to extract text outside code blocks as description/explanation
            cleaned_response = full_response_markdown
            # Remove matched blocks to isolate remaining text
            if latex_match: cleaned_response = cleaned_response.replace(latex_match.group(0), "", 1)
            if tikz_match: cleaned_response = cleaned_response.replace(tikz_match.group(0), "", 1)
            cleaned_response = cleaned_response.strip() # Remove leading/trailing whitespace

            # Split based on potential headers (case-insensitive) - refine as needed based on model output
            parts = re.split(r'\n\s*(?:Explain|Explanation|Description)[:\s]*\n', cleaned_response, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1 :
                 description = parts[0].strip()
                 explanation = parts[1].strip()
                 logger.debug(f"{stream_id}: Split description and explanation.")
            else:
                 # Assume all remaining non-code text is description/explanation
                 description = cleaned_response
                 explanation = description # Set explanation to description if no clear split
                 logger.debug(f"{stream_id}: Using combined text as description/explanation.")
                 if not description and (latex_code or tikz_code):
                     description = "(Code generated, no separate description provided)" # Placeholder if only code exists

            logger.debug(f"{stream_id}: Final Parsed -> Desc:'{description[:50]}...', LaTeX:{latex_code is not None}, TikZ:{tikz_code is not None}, Explain:'{explanation[:50]}...'")

            # --- Yield Final Parsed Data ---
            # This structure can be captured by the calling route handler
            final_parsed_data = {
                "description": description,
                "latex_code": latex_code,
                "tikz_code": tikz_code,
                "explanation": explanation,
                "raw_response": full_response_markdown # Include raw for debugging if needed
            }
            yield format_sse({"status": "Parsing complete.", "parsed_data": final_parsed_data, "stream_id": stream_id}, event_type="final_parsed")

        except Exception as e:
             final_status = "error"
             error_msg = f"Error during LaTeX/Visual streaming or processing: {e}"
             logger.error(f"❌ {stream_id}: {error_msg}")
             logger.exception(f"{stream_id}: LaTeX/VLM Stream Traceback:")
             # Yield error information
             yield format_sse({"error": error_msg, "final": True, "stream_id": stream_id}, event_type="error")
             # Attempt to log error to DB
             try: add_interaction(db, session_id=session_id, mode="chat", input_type="error", user_input=f"[Image LaTeX/TikZ Failed Stream {stream_id}]", llm_response=f"{error_msg}\nRaw Response Snippet: {full_response_markdown[:500]}")
             except Exception as db_log_err: logger.error(f"Failed to log LaTeX stream error: {db_log_err}")

        finally:
             # --- Signal End of Stream ---
             total_duration_ms = (time.monotonic() - task_start_time) * 1000
             yield format_sse({"status": f"Stream ended ({final_status}).", "final": True, "stream_id": stream_id, "total_duration_ms": total_duration_ms}, event_type="end_stream")
             logger.info(f"🏁 {stream_id}: LaTeX/VLM analysis stream finished. Status: {final_status}, Duration: {total_duration_ms:.2f} ms")
             # --- DB Logging for Success ---
             # If the calling function saved an initial interaction record,
             # it would ideally update it here or after collecting the 'final_parsed' event.
             # Since this generator doesn't easily get the interaction_id back,
             # we log the final results separately here if successful.
             if final_status == "success":
                 try:
                     log_data = {
                         'session_id': session_id,
                         'mode': 'chat',
                         'input_type': 'latex_analysis_result', # Custom type
                         'user_input': f"[Result for Image LaTeX/TikZ {stream_id}]",
                         'llm_response': f"Description: {description[:200]}...\nExplanation: {explanation[:200]}...",
                         'image_description': description,
                         'latex_representation': latex_code,
                         # 'tikz_representation': tikz_code, # If DB column exists
                         'latex_explanation': explanation,
                         'execution_time_ms': llm_execution_time_ms # Log LLM time specifically
                     }
                     valid_keys = {c.name for c in Interaction.__table__.columns}
                     db_kwargs = {k: v for k, v in log_data.items() if k in valid_keys}
                     add_interaction(db, **db_kwargs)
                     logger.debug(f"{stream_id}: Logged successful LaTeX/TikZ analysis results to DB.")
                 except Exception as db_log_err:
                     logger.error(f"{stream_id}: Failed to log successful LaTeX/TikZ results: {db_log_err}")

    # --- (rest of AIChat class, including the modified generate method) ---

    def process_image(self, db: Session, image_b64: str, session_id: str = None):
        """Processes image, gets description/LaTeX, returns description for non-VLM flow."""
        logger.info(f"🖼️ Processing image for session: {session_id}")
        self.current_session_id = session_id
        # Log initial interaction attempt
        interaction_data = {
            "session_id": session_id, "mode": "chat", "input_type": "image",
            "user_input": "[Image Uploaded]", "image_data": "..." # Placeholder
        }
        self.vectorstore_url = None
        logger.info("🧹 Cleared URL context due to image upload.")

        # Use VLM to get description
        vlm = self.provider.get_model("vlm")
        if not vlm:
            desc = "Error: Visual model (VLM) not available for image description."
            logger.error(desc)
            interaction_data['llm_response'] = desc
            valid_keys = {c.name for c in Interaction.__table__.columns}
            db_kwargs = {k: v for k, v in interaction_data.items() if k in valid_keys}
            add_interaction(db, **db_kwargs)
            return desc, None # Return description and None for image content

        try:
            base64.b64decode(image_b64) # Validate base64
            image_uri = f"data:image/jpeg;base64,{image_b64}"
        except Exception as e:
            desc = f"Error: Invalid image data format. {e}"
            logger.error(desc)
            interaction_data['llm_response'] = desc
            valid_keys = {c.name for c in Interaction.__table__.columns}
            db_kwargs = {k: v for k, v in interaction_data.items() if k in valid_keys}
            add_interaction(db, **db_kwargs)
            return desc, None

        # Prepare VLM input
        image_content_part = {"type": "image_url", "image_url": {"url": image_uri}}
        # Use a simple description prompt
        vlm_messages = [HumanMessage(content=[image_content_part, {"type": "text", "text": "Describe this image concisely."}])]
        vlm_chain = vlm | StrOutputParser()
        vlm_timing_data = {"session_id": session_id, "mode": "chat", "execution_time_ms": 0}

        try:
            logger.info("Calling VLM for image description...")
            image_description = self._call_llm_with_timing(vlm_chain, vlm_messages, vlm_timing_data)
            logger.info(f"🖼️ VLM Description: {image_description[:200]}...")
            # Log this VLM interaction
            interaction_data['llm_response'] = f"[VLM Description: {image_description}]"
            interaction_data['image_description'] = image_description # Store description
            interaction_data['image_data'] = image_b64 # Store image data
            interaction_data['execution_time_ms'] = vlm_timing_data['execution_time_ms']
            valid_keys = {c.name for c in Interaction.__table__.columns}
            db_kwargs = {k: v for k, v in interaction_data.items() if k in valid_keys}
            add_interaction(db, **db_kwargs)
            # Return the description to be added to the user prompt for non-VLM models
            return image_description, image_content_part # Return description and original image content part

        except Exception as e:
            desc = f"Error getting description from VLM: {e}"
            logger.error(desc)
            interaction_data['llm_response'] = desc
            valid_keys = {c.name for c in Interaction.__table__.columns}
            db_kwargs = {k: v for k, v in interaction_data.items() if k in valid_keys}
            add_interaction(db, **db_kwargs)
            return desc, None


    def process_url(self, db: Session, url: str, session_id: str = None):
        """Extracts text from URL, creates vectorstore (synchronous)."""
        logger.info(f"🔗 Processing URL: {url} (Session: {session_id})")
        self.current_session_id = session_id
        interaction_data = {"session_id": session_id, "mode": "chat", "input_type": "url", "user_input": f"[URL Submitted: {url}]", "url_processed": url}
        start_time = time.time()
        result_msg = ""
        success = False
        try:
            text = self.extract_text_from_url(url)
            if not text or not text.strip():
                raise ValueError("No significant text extracted")
            self.create_vectorstore_for_url(text, url)
            if hasattr(self, 'vectorstore_url') and self.vectorstore_url:
                setattr(self.vectorstore_url, '_source_url', url)
                result_msg = f"Processed URL: {url}. Ready for questions."
                success = True
                logger.success(f"✅ URL processed.")
            else:
                result_msg = f"Failed to create vectorstore for URL: {url}"
                success = False
        except Exception as e:
            logger.error(f"❌ Failed to process URL {url}: {e}")
            result_msg = f"Error processing URL: {e}"
            success = False
        finally:
            duration = (time.time() - start_time) * 1000
            interaction_data['llm_response'] = result_msg
            interaction_data['execution_time_ms'] = duration
            valid_keys = {c.name for c in Interaction.__table__.columns}
            db_kwargs = {k: v for k, v in interaction_data.items() if k in valid_keys}
            add_interaction(db, **db_kwargs)
            return result_msg

    def extract_text_from_url(self, url):
        """Extracts text from URL content (synchronous)."""
        logger.debug(f"🌐 Fetching content from {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, timeout=30, headers=headers, allow_redirects=True)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '').lower()
            if 'html' not in content_type and 'xml' not in content_type and 'text/plain' not in content_type:
                 logger.warning(f"⚠️ URL content type ({content_type}) not standard text.")
                 try:
                     return response.text.strip()
                 except Exception as de:
                     logger.error(f"Could not decode: {de}")
                     return None
            soup = BeautifulSoup(response.content, "html.parser")
            unwanted = ["script", "style", "header", "footer", "nav", "aside", "form", "button", "select", "noscript", "svg", "canvas", "audio", "video", "iframe", "embed", "object"]
            for element in soup(unwanted):
                element.decompose()
            text = soup.get_text(separator=' ', strip=True)
            if not text:
                 logger.warning("🚫 BeautifulSoup extraction resulted in empty text.")
                 return None
            logger.debug(f"📄 Extracted ~{len(text)} characters from {url}")
            return text
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ HTTP Error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Parsing Error: {e}")
            logger.exception("Parser Traceback:")
            return None


    def create_vectorstore_for_url(self, text: str, url: str):
        """Creates in-memory Chroma vectorstore from text (synchronous)."""
        logger.info(f"🧠 Creating vectorstore for URL: {url}")
        if not self.provider.embeddings:
            logger.error("❌ Embeddings provider missing.")
            self.vectorstore_url = None
            raise ValueError("Embeddings needed")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNCK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = text_splitter.split_text(text)
        if not splits:
            logger.warning("⚠️ No text splits generated.")
            self.vectorstore_url = None
            return
        logger.debug(f"📊 Split into {len(splits)} chunks.")
        try:
            self.vectorstore_url = Chroma.from_texts(splits, self.provider.embeddings)
            logger.success("✅ Vectorstore created.")
        except Exception as e:
            logger.error(f"❌ Failed Chroma create: {e}")
            logger.exception("Chroma Traceback:")
            self.vectorstore_url = None


def sanitize_filename(name: str, max_length: int = 200, replacement_char: str = '_') -> str:
    """
    Cleans a string to be suitable for use as a filename.

    Removes potentially problematic characters, replaces whitespace,
    and truncates to a maximum length.
    """
    if not isinstance(name, str):
        name = str(name) # Attempt to convert non-strings

    # Remove leading/trailing whitespace
    name = name.strip()

    # Replace problematic characters with the replacement character
    # Characters to remove/replace include path separators, control chars, etc.
    # Keeping alphanumeric, hyphen, underscore, period.
    # Removing: / \ : * ? " < > | and control characters (0-31)
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', replacement_char, name)

    # Replace multiple consecutive replacement characters or spaces with a single one
    sanitized = re.sub(f'[{re.escape(replacement_char)}\s]+', replacement_char, sanitized)

    # Remove leading/trailing replacement characters that might result from substitutions
    sanitized = sanitized.strip(replacement_char)

    # Truncate to maximum length
    if len(sanitized) > max_length:
        # Try truncating at the last replacement char before max_length to avoid cutting words
        try:
            trunc_point = sanitized[:max_length].rindex(replacement_char)
            sanitized = sanitized[:trunc_point]
        except ValueError:
            # If no replacement char found, just truncate hard
            sanitized = sanitized[:max_length]
        # Ensure it doesn't end with the replacement char after truncation
        sanitized = sanitized.strip(replacement_char)

    # Handle empty string case after sanitization
    if not sanitized:
        return "sanitized_empty_name"

    return sanitized

def download_content_sync(url: str, download_dir: str, filename_prefix: str, timeout: int = 30) -> bool:
    """
    Downloads content from a URL synchronously and saves it to a directory.

    Args:
        url: The URL to download from.
        download_dir: The directory to save the file in.
        filename_prefix: A prefix (usually derived from title) for the filename.
        timeout: Request timeout in seconds.

    Returns:
        True if download and save were successful, False otherwise.
    """
    download_logger = logger.bind(task="download_sync", url=url)
    download_logger.info(f"Attempting download...")

    try:
        headers = {'User-Agent': get_random_user_agent()} # Use helper if available, or hardcode one
        # Use stream=True to handle potentially large files without loading all into memory
        with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as response:
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # --- Determine Filename ---
            content_type = response.headers.get('content-type', '').split(';')[0].strip()
            content_disposition = response.headers.get('content-disposition')
            final_filename = None

            # 1. Try Content-Disposition header
            if content_disposition:
                disp_parts = content_disposition.split(';')
                for part in disp_parts:
                    if part.strip().lower().startswith('filename='):
                        final_filename = part.split('=', 1)[1].strip().strip('"')
                        # Sanitize filename from header, using prefix as fallback base
                        final_filename = sanitize_filename(final_filename or filename_prefix)
                        download_logger.debug(f"Using filename from Content-Disposition: {final_filename}")
                        break

            # 2. If no Content-Disposition filename, generate from prefix and type/URL
            if not final_filename:
                # Guess extension from content-type
                extension = mimetypes.guess_extension(content_type) if content_type else None
                # If no extension from type, try getting from URL path
                if not extension:
                     try:
                         parsed_url = urlparse(url)
                         path_part = parsed_url.path
                         _, potential_ext = os.path.splitext(path_part)
                         if potential_ext and len(potential_ext) < 10: # Basic check for valid-looking extension
                             extension = potential_ext
                     except Exception: pass # Ignore errors parsing URL path extension

                # Fallback extension
                if not extension:
                    if 'html' in content_type: extension = '.html'
                    elif 'pdf' in content_type: extension = '.pdf'
                    elif 'xml' in content_type: extension = '.xml'
                    elif 'json' in content_type: extension = '.json'
                    elif 'plain' in content_type: extension = '.txt'
                    else: extension = '.download' # Generic fallback

                # Combine sanitized prefix and extension
                final_filename = f"{filename_prefix}{extension}"
                download_logger.debug(f"Generated filename: {final_filename} (Type: {content_type}, Ext: {extension})")

            # --- Save File ---
            save_path = os.path.join(download_dir, final_filename)
            download_logger.info(f"Saving content to: {save_path}")

            # Create directory if it doesn't exist (should already exist from _trigger_web_search)
            os.makedirs(download_dir, exist_ok=True)

            # Write content in chunks
            chunk_count = 0
            total_bytes = 0
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        chunk_count += 1
                        total_bytes += len(chunk)

            download_logger.success(f"Download complete. Saved {total_bytes} bytes in {chunk_count} chunks.")
            return True

    except requests.exceptions.Timeout:
        download_logger.error(f"Request timed out after {timeout} seconds.")
        return False
    except requests.exceptions.RequestException as e:
        download_logger.error(f"Download failed: {e}")
        return False
    except IOError as e:
        download_logger.error(f"File saving failed: {e}")
        # Attempt to clean up partially written file
        if 'save_path' in locals() and os.path.exists(save_path):
            try: os.remove(save_path); download_logger.warning("Removed partial file after save error.")
            except Exception as rm_err: download_logger.error(f"Failed to remove partial file: {rm_err}")
        return False
    except Exception as e:
        download_logger.error(f"An unexpected error occurred during download: {e}")
        logger.exception("Download Unexpected Error Traceback:") # Log full traceback for unexpected errors
        return False

# Helper to format SSE data (can be reused)
def format_sse(data: Dict[str, Any], event_type: Optional[str] = None) -> str:
    """Formats data as a Server-Sent Event string."""
    json_data = json.dumps(data)
    sse_string = f"data: {json_data}\n"
    if event_type:
        sse_string = f"event: {event_type}\n{sse_string}"
    return sse_string + "\n"


# --- OpenAI Response Formatting Helpers ---

def _create_openai_error_response(message: str, err_type="internal_error", code=None, status_code=500):
    """Creates an OpenAI-like error JSON response."""
    error_obj = {
        "message": message,
        "type": err_type,
        "param": None,
        "code": code,
    }
    return {"error": error_obj}, status_code

def _format_openai_chat_response(response_text: str, model_name: str = "Amaryllis-Adelaide-IdioticRecursiveLearner-LegacyMoEArch") -> Dict[str, Any]:
    """Formats a simple text response into OpenAI ChatCompletion structure."""
    resp_id = f"chatcmpl-{uuid.uuid4()}"
    timestamp = int(time.time())
    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": timestamp,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop", # Assume completion for non-streaming direct response
            }
        ],
        "usage": { # Placeholder token usage
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "system_fingerprint": None,
    }


GENERATION_DONE_SENTINEL = object()
def _stream_openai_chat_response_generator_flask(
    session_id: str,
    user_input: str,
    classification: str,
    model_name: str = "Amaryllis-Adelaide-LegacyMoEArch-IdioticRecursiveLearner-FlaskStream"
):
    """
    Generator for Flask: Runs generation logic in a background thread,
    streams logs live via a queue, handles errors and cleanup, and yields
    Server-Sent Events (SSE) formatted chunks. Does not use placeholders. V7 Final.
    """
    # Unique ID for this streaming request for easier log tracing
    resp_id = f"chatcmpl-{uuid.uuid4()}"
    # Standard SSE timestamp
    timestamp = int(time.time())
    logger.debug(f"FLASK_STREAM_LIVE_V7 {resp_id}: Starting generation for session {session_id}")

    # Thread-safe queue for communication between the background thread and this generator
    message_queue = queue.Queue()
    # Placeholder for the background thread object
    background_thread: Optional[threading.Thread] = None
    # Dictionary to store the final result tuple (text, finish_reason, error_obj) received from the queue
    final_result_data = {
        "text": "Error: Generation failed to return result.", # Default text if thread fails badly
        "finish_reason": "error", # Default reason
        "error": None # Store actual exception object if one occurs
    }

    # --- Helper function to format data into SSE chunk ---
    def yield_chunk(delta_content: Optional[str] = None, role: Optional[str] = None, finish_reason: Optional[str] = None):
        """Constructs a dictionary matching OpenAI's streaming chunk format and returns it as an SSE string."""
        delta = {}
        if role: delta["role"] = role
        if delta_content is not None: delta["content"] = delta_content
        # Construct the chunk payload
        chunk_payload = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": model_name, # The model name passed to the generator
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
        }
        # Format according to SSE standard: "data: {json_payload}\n\n"
        return f"data: {json.dumps(chunk_payload)}\n\n"

    # --- Target function that will run in the background thread ---
    def run_async_generate_in_thread(q: queue.Queue, sess_id: str, u_input: str, classi: str):
        """
        Runs the core async ai_chat.generate function within this dedicated thread.
        1. Sets up an event loop for asyncio operations.
        2. Adds a temporary Loguru sink filtering logs for this request and putting them on the queue `q`.
        3. Executes the main `ai_chat.generate` coroutine.
        4. Catches exceptions during generation.
        5. Puts the final result tuple ("RESULT", (text, reason, error)) onto the queue `q`.
        6. Puts the `GENERATION_DONE_SENTINEL` onto the queue `q`.
        7. Cleans up the Loguru sink and potentially the event loop.
        """
        # --- Thread-Specific Variables ---
        sink_id = None # Holds the ID of the temporary Loguru sink
        db_session: Optional[Session] = None # DB Session for the generate task
        temp_loop: Optional[asyncio.AbstractEventLoop] = None # Asyncio event loop for this thread
        # Unique ID for this specific request/thread used for filtering logs in the sink
        log_session_id = f"{sess_id}-{threading.get_ident()}"
        # Default outcome variables, modified by the generate task
        thread_final_text = "Error: Processing failed within background thread."
        thread_final_reason = "error"
        thread_final_error = None # Stores the actual exception object

        try:
            # --- Setup asyncio Event Loop for this thread ---
            try:
                # Attempt to get an existing loop for this thread
                temp_loop = asyncio.get_event_loop()
                if temp_loop.is_running():
                     logger.warning(f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Event loop already running.")
            except RuntimeError:
                # No loop exists, create and set a new one
                logger.debug(f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Creating new event loop.")
                temp_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(temp_loop)

            # --- Define Loguru Sink function ---
            def log_sink(message):
                """Called by Loguru for logs matching the filter; puts formatted log on queue."""
                record = message.record
                bound_session_id = record.get("extra", {}).get("request_session_id")
                if bound_session_id == log_session_id: # Filter check
                    log_entry = f"[{record['time'].strftime('%H:%M:%S.%f')[:-3]} {record['level'].name}] {record['message']}"
                    try:
                        q.put_nowait(("LOG", log_entry)) # Put log tuple on queue
                    except queue.Full: pass # Ignore if queue is full
                    except Exception as e: print(f"ERROR in log_sink putting to queue: {e}", file=sys.stderr) # Log sink errors

            # --- Add the Loguru Sink ---
            try:
                logger.debug(f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Adding loguru sink.")
                sink_id = logger.add(
                    log_sink,
                    level=LOG_SINK_LEVEL, # Minimum level to capture
                    format=LOG_SINK_FORMAT, # Apply standard format
                    filter=lambda record: record["extra"].get("request_session_id") == log_session_id, # Crucial filter
                    enqueue=False # Run synchronously
                )
                logger.debug(f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Sink {sink_id} added.")
            except Exception as sink_add_err:
                 logger.error(f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): CRITICAL - Failed add log sink: {sink_add_err}")
                 thread_final_text = "Error setting up internal logging."; thread_final_reason = "error"; thread_final_error = sink_add_err
                 raise sink_add_err # Raise to trigger finally block

            # --- Define the Inner Async Function for AI Logic ---
            async def run_generate_with_logging():
                """Creates DB session, runs ai_chat.direct_generate for streaming, handles outcome.""" # <<< Docstring updated
                nonlocal db_session, thread_final_text, thread_final_reason, thread_final_error
                try:
                    db_session = SessionLocal() # Create DB session for this task
                    # Add unique ID to loguru context for all logs within this block
                    with logger.contextualize(request_session_id=log_session_id):
                         logger.info(f"Async direct_generate task starting for streaming...") # <<< Log updated
                         # --- Execute the DIRECT AI logic ---
                         # FIX: Call direct_generate, not generate.
                         # Also, check parameters needed by direct_generate.
                         # direct_generate expects: db, user_input, session_id, vlm_description=None
                         # It does NOT expect classification ('classi').
                         # We need to handle potential VLM description if image was involved
                         # Note: image_b64 isn't directly available here, this might need restructure
                         # For now, assume no image for the streaming direct call for simplicity,
                         # OR pass image_b64 into the generator if needed.
                         # Let's assume no image for now in this specific stream pathway.
                         result = await ai_chat.direct_generate(
                             db=db_session,
                             user_input=u_input,
                             session_id=sess_id,
                             vlm_description=None # <<< Assuming no image needed for stream content
                         )
                         # --- END FIX ---

                         thread_final_text = result if result is not None else "Error: Generation returned None."

                         # Determine finish reason based on result content
                         if "internal error" in thread_final_text.lower() or (thread_final_text.startswith("Error:") and "encountered a system issue" not in thread_final_text.lower()):
                             thread_final_reason = "error"
                             logger.warning(f"Async direct_generate task completed with internal error: {thread_final_text[:100]}...")
                         else: # Includes success and handled action fallbacks
                             thread_final_reason = "stop"
                             logger.info(f"Async direct_generate task completed successfully. Result len: {len(thread_final_text)}. Starts: '{thread_final_text[:50]}...'")

                except Exception as e:
                    # Log exceptions from ai_chat.direct_generate
                    with logger.contextualize(request_session_id=log_session_id):
                        logger.error(f"Async direct_generate task EXCEPTION: {e}"); logger.exception("Async Direct Generate Traceback:") # <<< Log updated
                    thread_final_error = e # Store the exception object
                    thread_final_text = f"[Error during direct generation for streaming: {type(e).__name__} - {e}]"
                    thread_final_reason = "error"
                finally:
                    # Ensure DB session is closed
                    if db_session:
                        try: db_session.close(); logger.debug("Async direct_generate DB session closed.") # <<< Log updated
                        except Exception as ce: logger.error(f"Error closing async direct_generate DB session: {ce}") # <<< Log updated

            # --- Run the async function ---
            temp_loop.run_until_complete(run_generate_with_logging())
            # Outcome is now stored in thread_final_text/reason/error

        except Exception as thread_err:
            # Catch errors in the synchronous setup part of this thread function
            logger.error(f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Error in background thread setup/run: {thread_err}")
            logger.exception("Background Thread Traceback:")
            # Store the error details if not already set by an inner failure
            if thread_final_error is None:
                thread_final_error = thread_err
                thread_final_text = f"Background thread execution error: {thread_err}"
                thread_final_reason = "error"
        finally:
            # --- Reliable Cleanup and Signaling ---
            try:
                logger.debug(f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Putting final RESULT onto queue.")
                # Put the final outcome onto the queue for the main generator thread
                q.put(("RESULT", (thread_final_text, thread_final_reason, thread_final_error)))
            except Exception as put_err:
                 logger.error(f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): CRITICAL - FAILED to put RESULT on queue: {put_err}")
            try:
                logger.debug(f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Putting DONE sentinel onto queue.")
                # Always signal completion via the sentinel
                q.put(GENERATION_DONE_SENTINEL)
            except Exception as put_done_err:
                 logger.error(f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): CRITICAL - FAILED to put DONE sentinel on queue: {put_done_err}")

            # --- Remove Loguru Sink ---
            if sink_id is not None:
                try:
                    logger.remove(sink_id)
                    logger.debug(f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Log sink {sink_id} removed.")
                except ValueError: # Sink might have already been removed if adding failed
                     logger.warning(f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Log sink {sink_id} not found for removal.")
                except Exception as remove_err:
                    logger.error(f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Failed to remove log sink {sink_id}: {remove_err}")

            logger.info(f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Background thread function finished.")


    # --- Main Generator Logic (Runs in Flask Request Thread) ---
    try:
        # Start the background thread to run the generation logic
        logger.debug(f"FLASK_STREAM_LIVE {resp_id}: Starting background thread...")
        background_thread = threading.Thread(
            target=run_async_generate_in_thread,
            args=(message_queue, session_id, user_input, classification),
            daemon=True # Allows main Flask app to exit even if this thread hangs
        )
        background_thread.start()
        logger.debug(f"FLASK_STREAM_LIVE {resp_id}: Background thread started (ID: {background_thread.ident}).")

        # --- Initial SSE Output to Client ---
        yield yield_chunk(role="assistant", delta_content="<think>\n")
        time.sleep(0.05)
        yield yield_chunk(delta_content="Starting live processing...\n---\n")
        time.sleep(0.05)

        # --- Loop to Consume from Queue and Yield Logs/Results ---
        logs_streamed_count = 0
        processing_complete = False # Becomes True when DONE sentinel is received
        result_received = False # Becomes True when RESULT tuple is received

        while not processing_complete:
            try:
                # Wait for an item from the queue with a timeout
                queue_item = message_queue.get(timeout=LOG_QUEUE_TIMEOUT)

                # --- Check if item is the completion sentinel ---
                if queue_item is GENERATION_DONE_SENTINEL:
                    logger.debug(f"FLASK_STREAM_LIVE {resp_id}: Received DONE sentinel. Exiting queue loop.")
                    processing_complete = True
                    continue # Finish the loop

                # --- If not sentinel, expect a tuple (Type, Data) ---
                elif isinstance(queue_item, tuple) and len(queue_item) == 2:
                    message_type, message_data = queue_item

                    if message_type == "LOG":
                        # Yield the log string received from the background thread
                        yield yield_chunk(delta_content=message_data + "\n")
                        logs_streamed_count += 1
                    elif message_type == "RESULT":
                        # Store the final result tuple when received
                        final_result_data["text"], final_result_data["finish_reason"], final_result_data["error"] = message_data
                        result_received = True # Mark that we have the final data
                        logger.debug(f"FLASK_STREAM_LIVE {resp_id}: Received RESULT from queue. Reason: {final_result_data['finish_reason']}")
                        # Continue loop, waiting for the DONE sentinel
                    else:
                        # Log if an unexpected tuple type is received
                        logger.warning(f"FLASK_STREAM_LIVE {resp_id}: Received unexpected message type from queue: {message_type}")
                else:
                     # Log if item structure is incorrect
                     logger.error(f"FLASK_STREAM_LIVE {resp_id}: Received unexpected item structure from queue: {type(queue_item)}")

            except queue.Empty:
                # --- Queue was empty (timeout occurred) ---
                # Check if the background thread died before signaling completion
                if not processing_complete and not background_thread.is_alive():
                    logger.error(f"FLASK_STREAM_LIVE {resp_id}: Background thread died unexpectedly before DONE sentinel.")
                    # Ensure error state is set if not already received via RESULT
                    if not result_received: # If we never even got a result back
                         final_result_data["error"] = RuntimeError("Background thread died before sending result.")
                         final_result_data["finish_reason"] = "error"
                         final_result_data["text"] = "[Critical Error: Background processing failed prematurely]"
                    else: # We got a result, but not the DONE signal - treat as error
                         final_result_data["error"] = RuntimeError("Background thread died after sending result but before signaling completion.")
                         final_result_data["finish_reason"] = "error"
                    processing_complete = True # Exit loop because background task is gone
                # else: Thread is alive or DONE already received, just continue waiting

            except Exception as q_err:
                 # Handle potential errors getting items from the queue
                 logger.error(f"FLASK_STREAM_LIVE {resp_id}: Error getting from queue: {q_err}")
                 if final_result_data["error"] is None: # Don't overwrite specific errors
                      final_result_data["error"] = q_err
                      final_result_data["finish_reason"] = "error"
                 processing_complete = True # Exit loop on queue error

        # --- Post-Loop: Final Processing and Streaming ---
        logger.debug(f"FLASK_STREAM_LIVE {resp_id}: Exited queue loop. Total logs streamed: {logs_streamed_count}. Result received: {result_received}")

        # Yield the closing think tag and final status message
        yield yield_chunk(delta_content="\n---\nLog stream complete.\n</think>\n\n")

        # Retrieve the final text and reason determined by the background thread
        final_text = final_result_data["text"]
        final_reason = final_result_data["finish_reason"]

        # If an error object was stored (either from RESULT or loop error handling), ensure finish_reason reflects it
        if final_result_data["error"] is not None:
            final_reason = "error"
            # Log only if the error wasn't the one set for premature exit without result
            if not (isinstance(final_result_data["error"], RuntimeError) and "Background thread" in str(final_result_data["error"])):
                 logger.warning(f"FLASK_STREAM_LIVE {resp_id}: Final result marked as error due to exception: {final_result_data['error']}")

        # --- Apply Final Cleanup (Safety Net) ---
        cleaned_final_text = final_text
        if result_received and isinstance(final_text, str): # Only clean if we got a string result
             if ai_chat: # Ensure global instance exists
                 try:
                     # Call instance method for cleanup
                     cleaned_final_text = ai_chat._cleanup_llm_output(final_text)
                     if cleaned_final_text != final_text:
                          logger.warning(f"FLASK_STREAM_LIVE {resp_id}: Applied cleanup to final text in streamer.")
                 except Exception as cleanup_err:
                      # Log error during cleanup but proceed with uncleaned text
                      logger.error(f"FLASK_STREAM_LIVE {resp_id}: Error applying cleanup in streamer: {cleanup_err}")
                      cleaned_final_text = final_text
             else:
                  logger.error(f"FLASK_STREAM_LIVE {resp_id}: Global ai_chat instance not found for cleanup.")
        elif not result_received:
             logger.error(f"FLASK_STREAM_LIVE {resp_id}: Skipping cleanup as no valid result was received from background thread.")
             cleaned_final_text = final_text # Use whatever text was set (likely error message)

        # --- Stream the *cleaned* final content ---
        if cleaned_final_text:
            logger.info(f"FLASK_STREAM_LIVE {resp_id}: Streaming final cleaned content ({len(cleaned_final_text)} chars). Finish: {final_reason}")
            final_stream_target_tok_per_sec = 300; chunk_size = 5 # Stream speed/chunking
            for i in range(0, len(cleaned_final_text), chunk_size):
                 text_chunk = cleaned_final_text[i:i+chunk_size]
                 yield yield_chunk(delta_content=text_chunk)
                 # Simulate token generation speed
                 num_chars = len(text_chunk); target_delay = max(0.001, (num_chars / 4.0) / final_stream_target_tok_per_sec)
                 time.sleep(target_delay) # Use synchronous sleep
            logger.debug(f"FLASK_STREAM_LIVE {resp_id}: Finished streaming final content.")
        else:
             # Handle cases where generation results in empty text after cleanup
             logger.warning(f"FLASK_STREAM_LIVE {resp_id}: Final cleaned response text is empty or None. Finish: {final_reason}")
             # If text is empty but no actual error occurred, finish reason should be 'stop'
             if final_reason != "error":
                 final_reason = "stop"

        # --- Send the final chunk with the determined finish reason ---
        logger.debug(f"FLASK_STREAM_LIVE {resp_id}: Yielding final chunk. Finish reason: {final_reason}")
        yield yield_chunk(finish_reason=final_reason)
        # --- Send the SSE termination signal ---
        yield "data: [DONE]\n\n"
        logger.debug(f"FLASK_STREAM_LIVE {resp_id}: Finished streaming.")

    except GeneratorExit:
        # Handle client disconnecting while generator is active
        logger.warning(f"FLASK_STREAM_LIVE {resp_id}: Generator exited prematurely (client likely disconnected).")
        # Background thread is daemon, should exit eventually. Consider if explicit cleanup needed.
    except Exception as e:
        # Catch any unexpected errors in the main generator logic itself
        logger.error(f"FLASK_STREAM_LIVE {resp_id}: Unhandled error during Flask streaming orchestration: {e}")
        logger.exception("Streaming Orchestration Traceback:")
        # Attempt to yield a final error chunk to the client if possible
        try:
            error_delta = {"content": f"\n\n[STREAMING ORCHESTRATION ERROR: {e}]"}
            error_chunk = { "id": resp_id, "object": "chat.completion.chunk", "created": timestamp, "model": model_name, "choices": [{"index": 0, "delta": error_delta, "finish_reason": "error"}] }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as final_err:
            # Log error if even the error chunk fails
            logger.error(f"FLASK_STREAM_LIVE {resp_id}: Failed yield final error chunk: {final_err}")
    finally:
        # Final cleanup actions for the generator itself
        # Check if thread is still alive (shouldn't be if daemon=True and main thread ends, but good for debug)
        if background_thread and background_thread.is_alive():
             logger.warning(f"FLASK_STREAM_LIVE {resp_id}: Generator finished, but background thread {background_thread.ident} might still be running (daemon).")
        logger.debug(f"FLASK_STREAM_LIVE {resp_id}: Generator function finished.")


@contextlib.contextmanager
def managed_webdriver(no_images=False):
    """Context manager for initializing and quitting the WebDriver (synchronous)."""
    # Use specific logger
    wd_logger = logger.bind(task="webdriver")
    driver = None
    service = None
    if not SELENIUM_AVAILABLE:
         wd_logger.error("Selenium not available, cannot create WebDriver.")
         yield None # Yield None if Selenium couldn't be imported
         return

    try:
        wd_logger.info("Initializing WebDriver (Chrome)...")
        options = webdriver.ChromeOptions()
        # Try to make it appear less automated
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        # Headless option - uncomment if desired, but CAPTCHAs might be harder
        # options.add_argument("--headless")
        # options.add_argument("--window-size=1920,1080")
        options.add_argument("--log-level=3") # Reduce browser console noise

        if no_images:
            wd_logger.info("Disabling image loading.")
            # Preferences to disable images
            prefs = {"profile.managed_default_content_settings.images": 2}
            options.add_experimental_option("prefs", prefs)

        try:
             # Use webdriver-manager to automatically handle driver download/update
             wd_logger.debug("Installing/updating ChromeDriver via webdriver-manager...")
             service = ChromeService(ChromeDriverManager().install())
             wd_logger.debug("ChromeDriver service ready.")
             driver = webdriver.Chrome(service=service, options=options)
             # Set user agent after driver is created
             # driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": get_random_user_agent()}) # Needs helper
             wd_logger.success("WebDriver initialized successfully.")
             yield driver # Provide the driver instance to the 'with' block
        except Exception as setup_exc:
             wd_logger.error(f"WebDriver Initialization Failed: {setup_exc}")
             wd_logger.exception("WebDriver Setup Traceback:")
             # If setup fails, yield None so the caller can handle it
             yield None
             return # Exit context manager if setup failed

    finally:
        # This block executes when exiting the 'with' statement
        if driver:
            wd_logger.info("Shutting down WebDriver...")
            try:
                driver.quit()
                wd_logger.success("WebDriver shut down.")
            except Exception as quit_exc:
                wd_logger.error(f"Error shutting down WebDriver: {quit_exc}")
        # Service doesn't usually need explicit stopping if driver.quit() works
        # if service and service.process:
        #    service.stop()
        #    wd_logger.info("ChromeDriver service stopped.")
    
def get_random_user_agent():
    """Returns a random User-Agent string."""
    return random.choice(USER_AGENTS)

def _format_legacy_completion_response(response_text: str, model_name: str = META_MODEL_NAME_NONSTREAM) -> Dict[str, Any]:
    """Formats a simple text response into the legacy OpenAI Completion structure."""
    resp_id = f"cmpl-{uuid.uuid4()}" # Different prefix often used for legacy
    timestamp = int(time.time())
    return {
        "id": resp_id,
        "object": "text_completion", # Legacy object type
        "created": timestamp,
        "model": model_name, # Use the non-streaming meta model name
        "choices": [
            {
                "text": response_text, # The generated text
                "index": 0,
                "logprobs": None, # Not supported here
                "finish_reason": "stop", # Assume stop if successful
            }
        ],
        "usage": { # Placeholder token usage
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    }

# --- End Helpers ---


# === Global AI Instances ===
ai_agent: Optional[AmaryllisAgent] = None
ai_provider: Optional[AIProvider] = None # Defined globally
ai_chat: Optional[AIChat] = None # Define ai_chat globally too

try:
    ai_provider = AIProvider(PROVIDER) # <<< ai_provider is initialized here
    global_ai_provider_ref = ai_provider
    ai_chat = AIChat(ai_provider)
    AGENT_CWD = os.path.dirname(os.path.abspath(__file__))
    SUPPORTS_COMPUTER_USE = True # Or determine dynamically
    ai_agent = AmaryllisAgent(ai_provider, AGENT_CWD, SUPPORTS_COMPUTER_USE)
    logger.success("✅ AI Instances Initialized.")
except Exception as e:
    logger.critical(f"🔥🔥 Failed AI init: {e}")
    logger.exception("AI Init Traceback:")
    # Ensure ai_provider is None if init fails
    ai_provider = None # <<< Add this line
    sys.exit(1)

# === Flask Routes (Async) ===

@app.route("/", methods=["POST"])
async def handle_interaction():
    """Main endpoint to handle user interactions asynchronously (Quart)."""
    start_req = time.monotonic()
    # --- Get DB Session within route ---
    db: Session = SessionLocal()
    # --- End DB Session Get ---
    response_text = "An unexpected server error occurred."
    status_code = 500
    request_data = None
    mode_param = "chat" # Default mode if not specified

    try:
        # Use Quart's await request.get_json()
        request_data = await request.get_json()
        if not request_data:
            logger.warning("⚠️ Empty JSON payload.")
            # Use Quart's jsonify or Response
            # Returning plain text as per original design of this endpoint
            response_text = "Empty request payload."
            status_code = 400
            resp = Response(response_text, status=status_code, mimetype="text/plain; charset=utf-8")
            # Need to close DB session in this early return path
            if db: db.close()
            return resp

        prompt = request_data.get("prompt", "")
        image_b64 = request_data.get("image", "")
        url = request_data.get("url", "")
        reset = request_data.get("reset", False)
        session_id = request_data.get("session_id", f"session_{int(time.time())}")
        mode_param = request_data.get("mode", "chat").lower()
        if mode_param not in ["chat", "agent"]:
            logger.warning(f"Invalid mode '{mode_param}' received, defaulting to 'chat'.")
            mode_param = "chat"

        logger.info(f"🚀 Quart Custom Request: Session={session_id}, ReqMode={mode_param}, Reset={reset}, URL='{url}', Img={'Y' if image_b64 else 'N'}, Prompt='{prompt[:30]}...'")

        # --- Workflow Logic ---
        if reset:
            # Pass the created db session to reset methods
            ai_chat.reset(db, session_id)
            ai_agent.reset(db, session_id) # Agent reset might also need db session
            response_text = "Chat and Agent session contexts reset."
            status_code = 200
        elif url:
            # process_url is synchronous, run in thread
            response_text = await asyncio.to_thread(ai_chat.process_url, db, url, session_id)
            status_code = 200 if "Error" not in response_text else 500
        elif image_b64:
             if len(image_b64) % 4 != 0 or not all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in image_b64):
                 response_text = "Invalid image data format."
                 status_code = 400
                 logger.error(f"Invalid base64 image received for session {session_id}")
             else:
                 # process_image is synchronous, run in thread
                 # It now returns description, image_content_part tuple
                 # For this simple endpoint, we just return the description/error text
                 description_or_error, _ = await asyncio.to_thread(
                     ai_chat.process_image, db, image_b64, session_id
                 )
                 response_text = description_or_error
                 status_code = 200 if "Error" not in response_text else 500
        elif prompt:
            # --- Use AIChat.generate which contains all complex logic ---
            # 1. Classify complexity first (sync in thread)
            classification_data = {"session_id": session_id, "mode": "chat", "input_type": "classification", "user_input": prompt[:100]}
            input_classification = await asyncio.to_thread(
                ai_chat._classify_input_complexity, db, prompt, classification_data
            )
            classification_reason = classification_data.get('classification_reason', 'N/A')
            interaction_data = classification_data # Use this dict for logging later

            # 2. Decide based on classification (and mode_param if forced)
            if input_classification == "agent_task" or mode_param == "agent":
                 logger.info(f"🎯 Agent Task triggered (Reason: {classification_reason}). Starting background workflow.")
                 # Agent manages its own DB session in background task
                 initial_agent_interaction = add_interaction(
                     db, session_id=session_id, mode="agent", input_type="text",
                     user_input=prompt, classification=input_classification,
                     classification_reason=classification_reason
                 )
                 response_text = f"Okay, I'll work on that task ({initial_agent_interaction.id}) in the background."
                 status_code = 200
                 await _start_agent_task(ai_agent, initial_agent_interaction.id, prompt, session_id)
            else:
                 # Use the main generate function for chat_simple/chat_complex
                 logger.info(f"🗣️ Running Chat workflow via generate (Classification: {input_classification})...")
                 response_text = await ai_chat.generate(
                     db, prompt, session_id, classification=input_classification
                 )
                 status_code = 500 if "internal error" in response_text.lower() or "Error:" in response_text else 200
        else:
            logger.warning("⚠️ No action specified (no prompt, image, url, or reset).")
            response_text = "No action specified in request."
            status_code = 400

        # Create the final plain text response object
        resp = Response(response_text, status=status_code, mimetype="text/plain; charset=utf-8")

    except Exception as e:
        logger.exception("🔥🔥 Unhandled exception in custom request handler:")
        response_text = f"Internal Server Error: {e}"
        status_code = 500
        resp = Response(response_text, status=status_code, mimetype="text/plain; charset=utf-8")
        # Error logging to DB (uses a new session)
        try:
            sid_for_error = "unknown"
            input_for_error = "Unknown"
            captured_mode = mode_param # Use mode from request if available
            if request_data and isinstance(request_data, dict):
                sid_for_error = request_data.get("session_id", f"error_{int(time.time())}")
                input_summary = [f"{k}: {str(v)[:50]}..." for k, v in request_data.items()]
                input_for_error = "; ".join(input_summary) if input_summary else str(request_data)[:1000]
            elif hasattr(request, 'data') and request.data:
                 input_for_error = f"Raw Data: {(await request.get_data(as_text=True))[:1000]}" # Use await for Quart

            error_db = SessionLocal()
            try:
                err_interaction_data = {
                    "session_id": sid_for_error, "mode": captured_mode, "input_type": 'error',
                    "user_input": input_for_error[:2000], "llm_response": f"Handler Error: {e}"[:2000]
                }
                valid_keys = {c.name for c in Interaction.__table__.columns}
                db_kwargs = {k: v for k, v in err_interaction_data.items() if k in valid_keys}
                add_interaction(error_db, **db_kwargs)
            except Exception as db_log_err_inner:
                logger.error(f"❌ Failed to log handler error to DB within error block: {db_log_err_inner}")
            finally:
                 error_db.close()
        except Exception as db_err:
            logger.error(f"❌ Failed to create session or log handler error to DB: {db_err}")

    finally:
        # --- Close DB Session for this route ---
        if db:
            db.close()
            logger.debug("Custom route DB session closed.")
        # --- End DB Session Close ---
        duration_req = (time.monotonic() - start_req) * 1000
        logger.info(f"🏁 Quart Custom Request handled in {duration_req:.2f} ms. Status: {status_code}")

    return resp


# === NEW OpenAI Compatible Embeddings Route ===
# app.py -> Flask Routes Section

# === NEW/REVISED OpenAI Compatible Embeddings Route for Quart ===
@app.route("/v1/embeddings", methods=["POST"])
@app.route("/api/embed", methods=["POST"])
@app.route("/api/embeddings", methods=["POST"])
async def handle_openai_embeddings():
    """Handles requests mimicking OpenAI's embedding endpoint using Quart."""
    start_req = time.monotonic()
    request_id = f"req-emb-{uuid.uuid4()}" # Unique ID for this request
    logger.info(f"🚀 Quart OpenAI-Style Embedding Request ID: {request_id}")
    status_code = 500 # Default to error
    response_payload = "" # Initialize

    # --- Check Provider Initialization ---
    if not ai_provider or not ai_provider.embeddings or not ai_provider.EMBEDDINGS_MODEL_NAME:
        logger.error(f"{request_id}: Embeddings provider not initialized correctly.")
        resp_data, status_code = _create_openai_error_response("Embedding model not available.", err_type="server_error", status_code=500)
        response_payload = json.dumps(resp_data)
        return Response(response_payload, status=status_code, mimetype='application/json')

    # Use the configured embedding model name for the response
    model_name_to_return = f"{ai_provider.provider_name}/{ai_provider.EMBEDDINGS_MODEL_NAME}"

    # --- Get and Validate Request Data ---
    try:
        request_data = await request.get_json() # Use await for Quart
        if not request_data:
            logger.warning(f"{request_id}: Empty JSON payload.")
            resp_data, status_code = _create_openai_error_response("Request body is missing or invalid JSON.", err_type="invalid_request_error", status_code=400)
            response_payload = json.dumps(resp_data)
            return Response(response_payload, status=status_code, mimetype='application/json')

        input_data = request_data.get("input")
        model_requested = request_data.get("model") # Log requested model, but ignore it

        if model_requested:
            logger.warning(f"{request_id}: Request specified model '{model_requested}', but will use configured '{model_name_to_return}'.")

        if not input_data:
            logger.warning(f"{request_id}: 'input' field missing.")
            resp_data, status_code = _create_openai_error_response("'input' is a required property.", err_type="invalid_request_error", status_code=400)
            response_payload = json.dumps(resp_data)
            return Response(response_payload, status=status_code, mimetype='application/json')

        # --- Prepare Input Texts ---
        texts_to_embed = []
        if isinstance(input_data, list):
            if not all(isinstance(item, str) for item in input_data):
                logger.warning(f"{request_id}: 'input' array must contain only strings.")
                resp_data, status_code = _create_openai_error_response("If 'input' is an array, all elements must be strings.", err_type="invalid_request_error", status_code=400)
                response_payload = json.dumps(resp_data)
                return Response(response_payload, status=status_code, mimetype='application/json')
            texts_to_embed = input_data
        elif isinstance(input_data, str):
            texts_to_embed = [input_data]
        else:
            logger.warning(f"{request_id}: 'input' must be a string or an array of strings.")
            resp_data, status_code = _create_openai_error_response("'input' must be a string or an array of strings.", err_type="invalid_request_error", status_code=400)
            response_payload = json.dumps(resp_data)
            return Response(response_payload, status=status_code, mimetype='application/json')

        if not texts_to_embed:
             logger.warning(f"{request_id}: No valid text found in 'input'.")
             resp_data, status_code = _create_openai_error_response("No text provided in 'input'.", err_type="invalid_request_error", status_code=400)
             response_payload = json.dumps(resp_data)
             return Response(response_payload, status=status_code, mimetype='application/json')

        # --- Generate Embeddings ---
        embeddings_list = []
        total_tokens = 0 # Placeholder
        status_code = 200 # Assume success unless error occurs

        logger.debug(f"{request_id}: Embedding {len(texts_to_embed)} text(s) using {model_name_to_return}...")
        start_embed_time = time.monotonic()

        # Run embedding in a thread as it can be CPU intensive
        if len(texts_to_embed) == 1:
            # Use embed_query for single string
            embedding_vector = await asyncio.to_thread(ai_provider.embeddings.embed_query, texts_to_embed[0])
            embeddings_list = [embedding_vector]
        else:
            # Use embed_documents for list of strings
            embeddings_list = await asyncio.to_thread(ai_provider.embeddings.embed_documents, texts_to_embed)

        embed_duration = (time.monotonic() - start_embed_time) * 1000
        logger.info(f"{request_id}: Embedding generation took {embed_duration:.2f} ms.")

        # --- Prepare Response Body ---
        response_data_list = []
        for i, vector in enumerate(embeddings_list):
            response_data_list.append({
                "object": "embedding",
                "embedding": vector, # Should be List[float]
                "index": i,
            })

        # Estimate token usage (very rough estimate)
        try:
             total_tokens = sum(len(text) for text in texts_to_embed) // 4
        except Exception:
             total_tokens = 0

        final_response_body = {
            "object": "list",
            "data": response_data_list,
            "model": model_name_to_return, # Return the actual model used
            "usage": {
                "prompt_tokens": total_tokens, # Estimated input tokens
                "total_tokens": total_tokens,
            },
        }
        response_payload = json.dumps(final_response_body)

    except Exception as e:
        logger.exception(f"{request_id}: 🔥🔥 Error during embedding generation:")
        resp_data, status_code = _create_openai_error_response(f"Failed to generate embeddings: {e}", err_type="server_error", status_code=500)
        response_payload = json.dumps(resp_data)
        # Attempt to log error to DB
        try:
            # Need a DB session - create one temporarily for error logging
            error_db = SessionLocal()
            try:
                add_interaction(error_db, session_id=f"openai_emb_error_{request_id}", mode="embedding", input_type='error', user_input=f"Embedding Error. Request: {str(request_data)[:1000]}", llm_response=f"Handler Error: {e}"[:2000])
            finally:
                error_db.close()
        except Exception as db_err:
            logger.error(f"❌ Failed to log embedding endpoint error to DB: {db_err}")


    finally:
        duration_req = (time.monotonic() - start_req) * 1000
        logger.info(f"🏁 OpenAI-Style Embedding Request {request_id} handled in {duration_req:.2f} ms. Status: {status_code}")
        # No DB session from 'g' to close here as it wasn't used directly in this route

    # Return Quart Response object
    return Response(response_payload, status=status_code, mimetype='application/json')


# app.py -> Flask Routes Section

@app.route("/v1/completions", methods=["POST"])
@app.route("/api/generate", methods=["POST"])
def handle_legacy_completions():
    """Handles requests mimicking the legacy OpenAI /v1/completions endpoint."""
    endpoint_hit = request.path
    start_req = time.monotonic()
    request_id = f"req-legacy-{uuid.uuid4()}"
    logger.info(f"🚀 Flask Legacy Completion Request ID: {request_id} on Endpoint: {endpoint_hit}")

    db: Session = g.db
    response_payload = ""
    status_code = 500
    resp: Optional[Response] = None
    session_id: str = f"legacy_req_{request_id}_unassigned"
    request_data_for_log: str = "No request data processed"
    final_response_status_code = 500
    raw_request_data: Optional[Dict] = None

    try:
        # --- Get Request Data ---
        try:
            raw_request_data = request.get_json()
            if not raw_request_data: raise ValueError("Empty JSON payload.")
            try: request_data_for_log = json.dumps(raw_request_data)[:1000]
            except: request_data_for_log = str(raw_request_data)[:1000]
        except Exception as e:
            logger.warning(f"{request_id}: Failed to get/parse JSON body: {e}")
            try: request_data_for_log = request.get_data(as_text=True)[:1000]
            except Exception: request_data_for_log = "Could not read request body"
            resp_data, status_code = _create_openai_error_response(f"Request body is missing or invalid JSON: {e}", err_type="invalid_request_error", status_code=400)
            response_payload = json.dumps(resp_data); resp = Response(response_payload, status=status_code, mimetype='application/json'); final_response_status_code = status_code; return resp

        # --- Extract Legacy Parameters ---
        prompt = raw_request_data.get("prompt")
        stream = raw_request_data.get("stream", False)
        model_requested = raw_request_data.get("model") # Log, but likely ignored
        session_id = raw_request_data.get("session_id", f"legacy_req_{request_id}") # Allow session override

        logger.debug(f"{request_id}: Legacy Request parsed - SessionID={session_id}, Stream: {stream}, Model Requested: {model_requested}, Prompt Snippet: '{str(prompt)[:50]}...'")

        # --- Input Validation ---
        if prompt is None or not isinstance(prompt, str):
            logger.warning(f"{request_id}: 'prompt' field missing or not a string.")
            resp_data, status_code = _create_openai_error_response("The 'prompt' parameter is required and must be a string.", err_type="invalid_request_error", status_code=400)
            response_payload = json.dumps(resp_data); resp = Response(response_payload, status=status_code, mimetype='application/json'); final_response_status_code = status_code; return resp

        # --- Handle Stream Request (Not Implemented Here) ---
        if stream:
            logger.warning(f"{request_id}: Streaming is not currently implemented for the legacy /v1/completions endpoint. Ignoring stream=True.")
            # Optionally return an error:
            # resp_data, status_code = _create_openai_error_response("Streaming is not supported for this legacy endpoint.", err_type="invalid_request_error", status_code=400)
            # response_payload = json.dumps(resp_data); resp = Response(response_payload, status=status_code, mimetype='application/json'); final_response_status_code = status_code; return resp
            # Or just proceed with non-streaming... we'll proceed for now.

        # --- Call Core Generation Logic (Non-Streaming) ---
        response_text = ""
        status_code = 200
        logger.info(f"{request_id}: Proceeding with non-streaming AIChat.generate for legacy prompt...")
        try:
            # Use asyncio.run to call the async generate function
            # Pass the legacy prompt directly as user_input
            # Classification will be handled inside `generate`
            response_text = asyncio.run(
                ai_chat.generate(db, prompt, session_id)
            )

            if "internal error" in response_text.lower() or "Error:" in response_text or "Traceback" in response_text:
                status_code = 500; logger.warning(f"{request_id}: AIChat.generate potential error: {response_text[:200]}...")
            else: status_code = 200
            logger.debug(f"{request_id}: AIChat.generate completed. Status: {status_code}")

        except Exception as gen_err:
            logger.error(f"{request_id}: Error during asyncio.run(ai_chat.generate): {gen_err}")
            logger.exception("Traceback for legacy generate error:")
            response_text = f"Error during generation: {gen_err}"
            status_code = 500

        # --- Format and Return NON-STREAMING Legacy Response ---
        if status_code != 200:
            resp_data, status_code = _create_openai_error_response(response_text, status_code=status_code)
        else:
            # Use the helper to format the response correctly
            resp_data = _format_legacy_completion_response(response_text, model_name=META_MODEL_NAME_NONSTREAM)

        response_payload = json.dumps(resp_data)
        logger.debug(f"{request_id}: Returning non-streaming legacy JSON. Status: {status_code}")
        resp = Response(response_payload, status=status_code, mimetype='application/json')
        final_response_status_code = status_code

    except Exception as e:
        # --- Main Exception Handler ---
        logger.exception(f"{request_id}: 🔥🔥 Unhandled exception in Flask Legacy Completion endpoint:")
        resp_data, status_code = _create_openai_error_response(f"Internal server error: {e}", status_code=500)
        response_payload = json.dumps(resp_data)
        resp = Response(response_payload, status=status_code, mimetype='application/json')
        final_response_status_code = status_code
        try: # Log error to DB
            if 'db' in g: add_interaction(g.db, session_id=session_id, mode="completion", input_type='error', user_input=f"Legacy Endpoint Error. Request: {request_data_for_log}", llm_response=f"Handler Error ({type(e).__name__}): {e}"[:2000])
            else: logger.error(f"{request_id}: Cannot log error: DB session 'g.db' unavailable.")
        except Exception as db_err: logger.error(f"{request_id}: ❌ Failed log error to DB: {db_err}")

    finally:
        # DB session is closed by teardown_request
        duration_req = (time.monotonic() - start_req) * 1000
        logger.info(f"🏁 Flask Legacy Completion Request {request_id} handled in {duration_req:.2f} ms. Status: {final_response_status_code}")

    # --- Return Response ---
    if resp is None: # Safety check
        logger.error(f"{request_id}: Handler finished unexpectedly without response object.")
        resp_data, _ = _create_openai_error_response("Internal error: Handler finished without response.", status_code=500)
        resp = Response(json.dumps(resp_data), status=500, mimetype='application/json')

    return resp

@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/api/chat", methods=["POST"]) # <<< ADD THIS LINE
def handle_openai_chat_completion():
    """
    Handles requests mimicking OpenAI/Ollama's chat completion endpoint.

    Implements Dual Generate Logic:
    1. Calls `ai_chat.direct_generate` synchronously to get a fast initial (ELP1) response.
    2. Formats and returns/streams this initial response.
    3. Concurrently launches `ai_chat.background_generate` in a separate thread
       to perform deeper analysis (ELP0) without blocking the initial response.
    """
    start_req = time.monotonic()
    request_id = f"req-chat-{uuid.uuid4()}"
    logger.info(f"🚀 Flask OpenAI/Ollama Chat Request ID: {request_id} (Dual Generate Logic)")

    # --- Initialize variables ---
    db: Session = g.db # Use request-bound session from Flask's g
    response_payload: str = ""
    status_code: int = 500 # Default to error
    resp: Optional[Response] = None
    session_id: str = f"openai_req_{request_id}_unassigned"
    request_data_for_log: str = "No request data processed"
    final_response_status_code: int = 500
    raw_request_data: Optional[Dict] = None
    classification_used: str = "chat_simple" # Placeholder, classification happens in background

    try:
        # --- 1. Get and Validate Request Data ---
        try:
            raw_request_data = request.get_json()
            if not raw_request_data:
                raise ValueError("Empty JSON payload received.")
            # Log request snippet safely
            try:
                request_data_for_log = json.dumps(raw_request_data)[:1000]
            except Exception:
                request_data_for_log = str(raw_request_data)[:1000]
        except Exception as json_err:
            logger.warning(f"{request_id}: Failed to get/parse JSON body: {json_err}")
            try:
                request_data_for_log = request.get_data(as_text=True)[:1000]
            except Exception:
                request_data_for_log = "Could not read request body"
            # Prepare error response
            resp_data, status_code = _create_openai_error_response(
                f"Request body is missing or invalid JSON: {json_err}",
                err_type="invalid_request_error", status_code=400
            )
            response_payload = json.dumps(resp_data)
            resp = Response(response_payload, status=status_code, mimetype='application/json')
            final_response_status_code = status_code
            # Return early on parsing error
            return resp

        # --- 2. Extract Parameters ---
        messages = raw_request_data.get("messages", [])
        stream_requested = raw_request_data.get("stream", False)
        model_requested = raw_request_data.get("model") # Logged but usually ignored for routing
        # Use session_id from request or generate one
        session_id = raw_request_data.get("session_id", f"openai_req_{request_id}")

        logger.debug(f"{request_id}: Request parsed - SessionID={session_id}, Stream: {stream_requested}, Model Requested: {model_requested}")

        # --- 3. Validate 'messages' Structure ---
        if not messages or not isinstance(messages, list):
            logger.warning(f"{request_id}: 'messages' field missing or not a list.")
            resp_data, status_code = _create_openai_error_response(
                "'messages' is required and must be a list.",
                err_type="invalid_request_error", status_code=400
            )
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
            final_response_status_code = status_code
            return resp

        # --- 4. Parse Last User Message for Input and Image ---
        last_user_message = None
        user_input = ""
        image_b64 = None
        # Find the most recent message with role 'user'
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user_message = msg
                break

        if not last_user_message:
            logger.warning(f"{request_id}: No message with role 'user' found.")
            resp_data, status_code = _create_openai_error_response(
                "No message with role 'user' found in 'messages'.",
                err_type="invalid_request_error", status_code=400
            )
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
            final_response_status_code = status_code
            return resp

        # Process content (string or list for multimodal)
        content = last_user_message.get("content")
        if isinstance(content, str):
            user_input = content
        elif isinstance(content, list):
             # Iterate through content parts (text and image)
             for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        user_input += item.get("text", "") # Concatenate text parts
                    elif item_type == "image_url":
                        image_url_data = item.get("image_url", {}).get("url", "")
                        if image_url_data.startswith("data:image"):
                            # Extract and validate base64 data
                            try:
                                header, potential_b64 = image_url_data.split(",", 1)
                                # Basic validation (padding, characters)
                                if len(potential_b64) % 4 != 0 or not re.match(r'^[A-Za-z0-9+/=]+$', potential_b64):
                                    raise ValueError("Invalid base64 characters or padding")
                                # Decode to validate
                                base64.b64decode(potential_b64, validate=True)
                                image_b64 = potential_b64 # Store validated base64
                                logger.info(f"{request_id}: Extracted base64 image data from message.")
                            except Exception as img_err:
                                logger.warning(f"{request_id}: Invalid base64 image data found in message: {img_err}")
                                resp_data, status_code = _create_openai_error_response(
                                    f"Invalid image data provided: {img_err}",
                                    err_type="invalid_request_error", status_code=400
                                )
                                resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
                                final_response_status_code = status_code
                                return resp
                        else:
                            # Handle unsupported non-data URLs
                            logger.warning(f"{request_id}: Unsupported image_url format (only data:image supported): {image_url_data[:50]}...")
                            resp_data, status_code = _create_openai_error_response(
                                "Unsupported image_url format. Only data URIs (data:image/...) are supported.",
                                err_type="invalid_request_error", status_code=400
                            )
                            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
                            final_response_status_code = status_code
                            return resp
        else:
             # Handle invalid content type (not string or list)
             logger.warning(f"{request_id}: Invalid 'content' type in user message: {type(content)}")
             resp_data, status_code = _create_openai_error_response(
                 "Invalid user message 'content' type. Must be string or list.",
                 err_type="invalid_request_error", status_code=400
             )
             resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
             final_response_status_code = status_code
             return resp

        # Final check: Ensure we have either text or image input
        if not user_input and not image_b64:
             logger.warning(f"{request_id}: No usable text or image content found after parsing.")
             resp_data, status_code = _create_openai_error_response(
                 "No text or image content provided in the user message.",
                 err_type="invalid_request_error", status_code=400
             )
             resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
             final_response_status_code = status_code
             return resp

        # --- 5. Call DIRECT Generate Logic (Synchronous via asyncio.run with ELP1) ---
        direct_response_text = ""
        vlm_description_for_bg = None # Store VLM result for background task if needed
        status_code = 200 # Assume success initially for direct response

        logger.info(f"{request_id}: Calling AIChat.direct_generate (ELP1)...")
        try:
            # Handle image preprocessing FIRST if an image exists
            # This call runs synchronously within this request handler thread
            if image_b64:
                 logger.info(f"{request_id}: Preprocessing image for direct_generate (ELP1 implied)...")
                 # Assume process_image handles priority internally if needed
                 # Pass the request's DB session (g.db)
                 vlm_description_for_bg, _ = ai_chat.process_image(db, image_b64, session_id)
                 if vlm_description_for_bg and "Error:" in vlm_description_for_bg:
                      logger.error(f"{request_id}: VLM preprocessing failed for direct path: {vlm_description_for_bg}")
                      # Set status code, but still proceed to get text-only direct response
                      status_code = 500 # Indicate VLM error occurred
                      direct_response_text = asyncio.run(
                          ai_chat.direct_generate(db, user_input, session_id, vlm_description=None) # Call with text only
                      )
                 else:
                      # VLM success, include description in direct call
                      direct_response_text = asyncio.run(
                          ai_chat.direct_generate(db, user_input, session_id, vlm_description=vlm_description_for_bg)
                      )
            else:
                 # No image, just call direct_generate with text
                 direct_response_text = asyncio.run(
                     ai_chat.direct_generate(db, user_input, session_id, vlm_description=None)
                 )

            # Check if direct_generate itself indicated an internal error
            if status_code != 500: # Only override if VLM didn't already fail
                 if "internal error" in direct_response_text.lower() or direct_response_text.lower().startswith("error:"):
                    status_code = 500
                    logger.warning(f"{request_id}: AIChat.direct_generate returned an error: {direct_response_text[:200]}...")
                 else:
                    status_code = 200 # Explicitly confirm success
            logger.info(f"{request_id}: AIChat.direct_generate completed. Effective Status for Direct: {status_code}")

        except Exception as direct_gen_err:
            logger.error(f"{request_id}: Error during asyncio.run(ai_chat.direct_generate): {direct_gen_err}")
            logger.exception(f"{request_id} Traceback for direct_generate error:")
            direct_response_text = f"Error during initial response generation: {direct_gen_err}"
            status_code = 500

        # --- 6. LAUNCH BACKGROUND Generate Logic (in a separate thread - ELP0) ---
        logger.info(f"{request_id}: Preparing to launch background_generate task (ELP0)...")

        # Define the target function for the background thread
        def run_background_task():
            bg_db: Optional[Session] = None # Initialize as None
            bg_err: Optional[Exception] = None # Initialize error tracker

            try:
                # --- Attempt to create DB session ---
                try:
                    bg_db = SessionLocal() # Create a new session for this thread
                    if not bg_db: raise ValueError("SessionLocal() returned None")
                    logger.debug(f"[BG Task {request_id}] Background DB session created.")
                except Exception as session_err:
                     logger.error(f"[BG Task {request_id}] CRITICAL: Failed to create background DB session: {session_err}")
                     bg_err = session_err
                     return # Exit thread if DB session fails

                # --- Main Background Logic ---
                logger.info(f"[BG Task {request_id}] Background task started.")
                # Re-classify complexity within the background task's context
                bg_classification_data = {"session_id": session_id, "mode": "chat", "input_type": "classification", "user_input": user_input[:100]}
                # _classify_input_complexity runs synchronously but uses ELP0 internally via its helpers
                bg_classification = ai_chat._classify_input_complexity(bg_db, user_input, bg_classification_data)
                logger.info(f"[BG Task {request_id}] Background classification: {bg_classification}")

                # Run background_generate using asyncio.run within this thread
                # background_generate uses bg_db and handles ELP0/interruptions internally
                asyncio.run(
                    ai_chat.background_generate(
                        db=bg_db,
                        user_input=user_input,
                        session_id=session_id,
                        classification=bg_classification, # Use classification determined here
                        image_b64=image_b64, # Pass image again if present
                        # update_interaction_id is None here (not a reflection task)
                    )
                )
                logger.info(f"[BG Task {request_id}] Background task completed successfully.")

            except Exception as task_err:
                # Catch errors from classification or background_generate
                bg_err = task_err # Store the exception
                logger.error(f"[BG Task {request_id}] Error during background task execution: {bg_err}")
                logger.exception(f"[BG Task {request_id}] Background Task Execution Traceback:")
                # Log error to DB using bg_db (if it was created)
                if bg_db:
                    try:
                        add_interaction(bg_db, session_id=session_id, mode="chat", input_type="log_error",
                                        user_input=f"Background Task Error ({request_id})",
                                        llm_response=f"Error: {bg_err}"[:2000])
                        bg_db.commit() # Commit the error log
                    except Exception as db_log_err:
                         logger.error(f"[BG Task {request_id}] Failed log background execution error to DB: {db_log_err}")
                         if bg_db: bg_db.rollback()
                else:
                    logger.error(f"[BG Task {request_id}] Cannot log background execution error: DB session was not created.")
            finally:
                # Ensure background session is closed if it was created
                if bg_db:
                    try:
                        bg_db.close()
                        logger.debug(f"[BG Task {request_id}] Background DB session closed.")
                    except Exception as close_err:
                         logger.error(f"[BG Task {request_id}] Error closing background DB session: {close_err}")
                logger.info(f"[BG Task {request_id}] Background thread finished.")

        # Launch the background task in a daemon thread
        try:
            background_thread = threading.Thread(target=run_background_task, daemon=True)
            background_thread.start()
            logger.info(f"{request_id}: Launched background_generate in thread {background_thread.ident}.")
        except Exception as launch_err:
            logger.error(f"{request_id}: Failed to launch background thread: {launch_err}")
            # Log this failure? May affect background processing.
            try:
                 add_interaction(db, session_id=session_id, mode="chat", input_type="log_error",
                                 user_input="Background Task Launch Failed",
                                 llm_response=f"Error: {launch_err}")
            except Exception as db_err:
                 logger.error(f"{request_id}: Failed log background launch error: {db_err}")

        # --- 7. Format and Return/Stream the IMMEDIATE Response (from direct_generate) ---
        # Determine model ID based on stream preference for the immediate response
        model_id_used = META_MODEL_NAME_STREAM if stream_requested else META_MODEL_NAME_NONSTREAM

        if stream_requested:
             # Return a streaming response using the streaming generator helper
             # This generator streams the 'direct_response_text' and associated logs
             logger.info(f"{request_id}: Client requested stream. Creating stream generator for direct response.")
             generator = _stream_openai_chat_response_generator_flask(
                 session_id=session_id,
                 user_input=user_input, # Pass original user input for context if needed by generator
                 classification=classification_used, # Pass placeholder classification
                 model_name=model_id_used # Use STREAM meta name
             )
             # Set up Flask Response for streaming SSE
             resp = Response(generator, mimetype='text/event-stream')
             resp.headers['Content-Type'] = 'text/event-stream; charset=utf-8'
             resp.headers['Cache-Control'] = 'no-cache'
             resp.headers['Connection'] = 'keep-alive'
             # Set final status code to 200 as stream initiated successfully
             # (Errors during stream generation are handled within the generator)
             final_response_status_code = 200

        else:
             # Format and return the non-streaming JSON based on direct_response_text
             logger.debug(f"{request_id}: Formatting non-streaming JSON response based on direct_generate.")
             if status_code != 200:
                 # Format as OpenAI error object if direct_generate failed
                 resp_data, _ = _create_openai_error_response(direct_response_text, status_code=status_code)
             else:
                 # Format as standard OpenAI ChatCompletion object using the successful direct response
                 resp_data = _format_openai_chat_response(direct_response_text, model_name=model_id_used) # Use NONSTREAM meta name
             # Prepare Flask Response object
             response_payload = json.dumps(resp_data)
             resp = Response(response_payload, status=status_code, mimetype='application/json')
             final_response_status_code = status_code # Reflect status from direct generate

    except Exception as main_err:
        # Catch any truly unexpected errors in the main request handling logic
        logger.exception(f"{request_id}: 🔥🔥 UNHANDLED exception in main handler:")
        # Create a generic error response
        error_message = f"Internal server error: {type(main_err).__name__}"
        resp_data, status_code = _create_openai_error_response(error_message, status_code=500)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_response_status_code = status_code
        # Log this critical failure to DB if possible
        try:
            if 'db' in g and g.db:
                 add_interaction(g.db, session_id=session_id, mode="chat", input_type='error',
                                 user_input=f"Main Handler Error. Request: {request_data_for_log}",
                                 llm_response=error_message[:2000])
            else: logger.error(f"{request_id}: Cannot log main handler error: DB session 'g.db' unavailable.")
        except Exception as db_err: logger.error(f"{request_id}: ❌ Failed log main handler error to DB: {db_err}")

    finally:
        # This block ALWAYS runs after try/except
        duration_req = (time.monotonic() - start_req) * 1000
        # Log final status using the status code determined by the try/except blocks
        logger.info(f"🏁 Flask OpenAI/Ollama Chat Request {request_id} (Dual Generate) handled in {duration_req:.2f} ms. Final Status: {final_response_status_code}")
        # DB session g.db is closed automatically by the @app.teardown_request handler

    # --- Return Response ---
    # Ensure 'resp' is always assigned
    if resp is None:
        logger.error(f"{request_id}: Handler finished unexpectedly without response object assigned!")
        # Create a generic error response if 'resp' is somehow still None
        resp_data, status_code = _create_openai_error_response("Internal error: Handler finished without creating response.", status_code=500)
        resp = Response(json.dumps(resp_data), status=500, mimetype='application/json')
        # Log this critical failure
        try:
             if 'db' in g and g.db:
                 add_interaction(g.db, session_id=session_id, mode="chat", input_type='error',
                                 user_input=f"Handler Logic Error. Request: {request_data_for_log}",
                                 llm_response="Critical: No response object created.")
             else: logger.error(f"{request_id}: Cannot log 'no response' error: DB session 'g.db' unavailable.")
        except Exception as db_err: logger.error(f"{request_id}: ❌ Failed log 'no response' error to DB: {db_err}")

    return resp # Return the final Flask Response object

@app.route("/v1/models", methods=["GET"])
def handle_openai_models():
    """Handles requests mimicking OpenAI's models endpoint (Flask), using global constants."""
    logger.info("Received request for /v1/models")
    start_req = time.monotonic()
    status_code = 200
    # --- Use global constants ---
    model_list = [
        {
            "id": META_MODEL_NAME_STREAM, # Use Constant
            "object": "model",
            "created": int(time.time()),
            "owned_by": META_MODEL_OWNER, # Use Constant
            "permission": [],
            "root": META_MODEL_NAME_STREAM, # Use Constant
            "parent": None,
        },
        {
            "id": META_MODEL_NAME_NONSTREAM, # Use Constant
            "object": "model",
            "created": int(time.time()),
            "owned_by": META_MODEL_OWNER, # Use Constant
            "permission": [],
            "root": META_MODEL_NAME_NONSTREAM, # Use Constant
            "parent": None,
        },
        # Add more meta-models here if needed by referencing constants
    ]
    # --- End use global constants ---
    response_body = {
        "object": "list",
        "data": model_list,
    }
    response_payload = json.dumps(response_body, indent=2)
    duration_req = (time.monotonic() - start_req) * 1000
    logger.info(f"🏁 /v1/models request handled in {duration_req:.2f} ms. Status: {status_code}")
    return Response(response_payload, status=status_code, mimetype='application/json')


@app.route("/api/tags", methods=["GET", "HEAD"])
def handle_ollama_tags():
    """Handles requests mimicking Ollama's /api/tags endpoint, using global constants."""
    logger.info("Received request for /api/tags (Ollama Compatibility)")
    start_req = time.monotonic()
    status_code = 200

    # --- Use global constants ---
    ollama_models = [
        {
            "name": f"{META_MODEL_NAME_STREAM}:latest", # Use Constant
            "model": f"{META_MODEL_NAME_STREAM}:latest", # Use Constant
            "modified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "size": 0, # Placeholder size
            "digest": hashlib.sha256(META_MODEL_NAME_STREAM.encode()).hexdigest(), # Fake digest
            "details": {
                "parent_model": "",
                "format": META_MODEL_FORMAT,             # Use Constant
                "family": META_MODEL_FAMILY,             # Use Constant
                "families": [META_MODEL_FAMILY],         # Use Constant
                "parameter_size": META_MODEL_PARAM_SIZE, # Use Constant
                "quantization_level": META_MODEL_QUANT_LEVEL # Use Constant
            }
        },
        {
            "name": f"{META_MODEL_NAME_NONSTREAM}:latest", # Use Constant
            "model": f"{META_MODEL_NAME_NONSTREAM}:latest", # Use Constant
            "modified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "size": 0,
            "digest": hashlib.sha256(META_MODEL_NAME_NONSTREAM.encode()).hexdigest(),
            "details": {
                "parent_model": "",
                "format": META_MODEL_FORMAT,             # Use Constant
                "family": META_MODEL_FAMILY,             # Use Constant
                "families": [META_MODEL_FAMILY],         # Use Constant
                "parameter_size": META_MODEL_PARAM_SIZE, # Use Constant
                "quantization_level": META_MODEL_QUANT_LEVEL # Use Constant
            }
        },
        # Add more meta-models here if needed
    ]
    # --- End use global constants ---

    response_body = {
        "models": ollama_models
    }
    response = jsonify(response_body)
    response.status_code = status_code

    duration_req = (time.monotonic() - start_req) * 1000
    logger.info(f"🏁 /api/tags request handled in {duration_req:.2f} ms. Status: {status_code}")
    return response

# --- NEW: Dummy Handlers for Pretending this is Ollama Model Management ---

@app.route("/api/pull", methods=["POST"])
def handle_api_pull_dummy():
    logger.warning("⚠️ Received request for dummy endpoint: /api/pull (Not Implemented)")
    # Simulate Ollama's streaming progress (optional, but makes it look real)
    def generate_dummy_pull():
        yield json.dumps({"status": "pulling manifest"}) + "\n"
        time.sleep(0.5)
        yield json.dumps({"status": "verifying sha256:...", "total": 100, "completed": 50}) + "\n"
        time.sleep(0.5)
        yield json.dumps({"status": "success"}) + "\n"
    # Or just return an error directly:
    # return jsonify({"error": "Model pulling not implemented in this server"}), 501
    return Response(generate_dummy_pull(), mimetype='application/x-ndjson') # Mimic streaming

@app.route("/api/push", methods=["POST"])
def handle_api_push_dummy():
    logger.warning("⚠️ Received request for dummy endpoint: /api/push (Not Implemented)")
    return jsonify({"error": "Model pushing not implemented in this server"}), 501

@app.route("/api/show", methods=["POST"])
def handle_api_show_dummy():
    logger.warning("⚠️ Received request for dummy endpoint: /api/show (Not Implemented)")
    # You could try and fake a response based on your known meta-models if needed
    # For now, just return not implemented
    return jsonify({"error": "Showing model details not implemented in this server"}), 501

@app.route("/api/delete", methods=["DELETE"])
def handle_api_delete_dummy():
    logger.warning("⚠️ Received request for dummy endpoint: /api/delete (Not Implemented)")
    return jsonify({"status": "Model deletion not implemented"}), 501 # Ollama might return 200 OK even if no-op? Return 501 for clarity.

@app.route("/api/create", methods=["POST"])
def handle_api_create_dummy():
    logger.warning("⚠️ Received request for dummy endpoint: /api/create (Not Implemented)")
    return jsonify({"error": "Model creation from Modelfile not implemented"}), 501

@app.route("/api/copy", methods=["POST"])
def handle_api_copy_dummy():
    logger.warning("⚠️ Received request for dummy endpoint: /api/copy (Not Implemented)")
    return jsonify({"error": "Model copying not implemented"}), 501

@app.route("/api/blobs/<digest>", methods=["POST", "HEAD"])
def handle_api_blobs_dummy(digest: str):
    logger.warning(f"⚠️ Received request for dummy endpoint: /api/blobs/{digest} (Not Implemented)")
    if request.method == 'HEAD':
        # HEAD usually just checks existence, return 404
        return Response(status=404)
    else: # POST
        return jsonify({"error": "Blob creation/checking not implemented"}), 501


@app.route("/api/version", methods=["GET", "HEAD"])
def handle_api_version():
    logger.info("Received request for /api/version")
    version_string = "Adelaide-Zephyrine-Charlotte-MetacognitionArtificialQuellia-0.0.1" # As requested
    response_data = {"version": version_string}
    # For HEAD, Flask might implicitly handle sending only headers if body is small
    # or you could explicitly check request.method == 'HEAD' and return empty Response
    return jsonify(response_data), 200

@app.route("/api/ps", methods=["GET"])
def handle_api_ps_dummy():
    logger.warning("⚠️ Received request for dummy endpoint: /api/ps (Not Implemented)")
    # Return an empty list of running models, mimicking Ollama
    return jsonify({"models": []}), 200

@app.route("/v1/models/<path:model>", methods=["GET"])
def handle_openai_retrieve_model(model: str):
    """Handles requests mimicking OpenAI's retrieve model endpoint."""
    logger.info(f"Received request for /v1/models/{model}")
    start_req = time.monotonic()
    status_code = 404
    response_body = {"error": f"Model '{model}' not found."}

    # Check if the requested model matches one of our meta-models
    known_models = [META_MODEL_NAME_STREAM, META_MODEL_NAME_NONSTREAM]
    if model in known_models:
        status_code = 200
        response_body = {
                "id": model,
                "object": "model",
                "created": int(time.time()), # Placeholder timestamp
                "owned_by": META_MODEL_OWNER,
        }

    response_payload = json.dumps(response_body)
    duration_req = (time.monotonic() - start_req) * 1000
    logger.info(f"🏁 /v1/models/{model} request handled in {duration_req:.2f} ms. Status: {status_code}")
    return Response(response_payload, status=status_code, mimetype='application/json')

if __name__ != "__main__":
    # --- Remove Assistant Proxy Setup ---
    # logger.info("Attempting to set up Assistant Proxy...") # Removed
    # setup_assistant_proxy() # Removed
    logger.info("Assistant Proxy setup Arent needed anymore")
    logger.info("Impostoring as an Ordinary static noncapable non adaptive Ollama... at port 11434")

    # --- Start the file indexer ---
    # The start_file_indexer function now handles logging and errors internally
    start_file_indexer()
    start_self_reflector() # <<< ADD THIS LINE

# This block prevents direct execution and provides instructions
else:
    logger.error("This script should be run with an ASGI/WSGI server like Hypercorn or Gunicorn. To be Impostor of being ordinary Ollama")
    logger.error("Example: hypercorn app:app --bind 127.0.0.1:11434") #Impostor mode enabled
    sys.exit(1)

# === Main Execution (Requires ASGI Server) ===
if __name__ == "__main__":
    # This block won't run when Hypercorn imports 'app'
    logger.error("This script should be run with an ASGI/WSGI server like Hypercorn. This script should be run with an ASGI/WSGI server like Hypercorn or Gunicorn. To be Impostor of being ordinary Ollama")
    logger.error("Example: hypercorn app:app --bind 127.0.0.1:11434")
    sys.exit(1)
else:
    # This block runs when Hypercorn imports 'app'
    try:
        #logger.info("Attempting to set up Assistant Proxy...")
        #setup_assistant_proxy() # Call the setup function
        pass
    except Exception as setup_err:
         logger.error(f"🚨 Failed to run Assistant Proxy setup: {setup_err}")