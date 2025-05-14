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
from langchain_core.exceptions import OutputParserException

# --- Langchain Community Imports ---
#from langchain_community.vectorstores import Chroma # Use Chroma for in-memory history/URL RAG
from langchain_chroma import Chroma
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
    from file_indexer import (
        FileIndexer,
        initialize_global_file_index_vectorstore as init_file_vs_from_indexer,  # <<< ADD THIS IMPORT and ALIAS
        get_global_file_index_vectorstore  # You already had this for AIChat
    )
    from agent import AmaryllisAgent, AgentTools, _start_agent_task # Keep Agent imports
except ImportError as e:
    print(f"Error importing local modules (database, config, agent, ai_provider): {e}")
    logger.exception("Import Error Traceback:") # Log traceback for import errors
    FileIndexer = None # Define as None if import fails
    FileIndex = None
    search_file_index = None
    sys.exit(1)

from reflection_indexer import (
    initialize_global_reflection_vectorstore,
    index_single_reflection, # If you want AIChat to trigger indexing
    get_global_reflection_vectorstore
)

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

# --- End Local Imports ---

# Add the inspection code again *after* these imports
logger.debug("--- Inspecting Interaction Model Columns AFTER explicit import ---")
logger.debug(f"Columns found by SQLAlchemy: {[c.name for c in Interaction.__table__.columns]}")
if 'tot_delivered' in [c.name for c in Interaction.__table__.columns]: logger.debug("✅ 'tot_delivered' column IS present.")
else: logger.error("❌ 'tot_delivered' column IS STILL MISSING!")
logger.debug("-------------------------------------------------------------")


try:
    import tiktoken
    # Attempt to load the encoder once globally for AIChat if not already done by worker logic
    # Or load it on demand in the helper function.
    # For simplicity here, assume it's available or loaded in a helper.
    TIKTOKEN_AVAILABLE_APP = True
    # Try to get a common encoder
    try:
        cl100k_base_encoder_app = tiktoken.get_encoding("cl100k_base")
    except Exception:
        cl100k_base_encoder_app = tiktoken.encoding_for_model("gpt-4") # Fallback
except ImportError:
    logger.warning("tiktoken not available in app.py. Context truncation will be less accurate (char-based).")
    TIKTOKEN_AVAILABLE_APP = False
    cl100k_base_encoder_app = None

# Define these near the top, perhaps after imports or before app = Flask(...)

META_MODEL_NAME_STREAM = "Amaryllis-AdelaidexAlbert-MetacognitionArtificialQuellia-Stream"
META_MODEL_NAME_NONSTREAM = "Amaryllis-AdelaidexAlbert-MetacognitionArtificialQuellia"
META_MODEL_OWNER = "zephyrine-foundation"
TTS_MODEL_NAME_CLIENT_FACING = "Zephyloid-Alpha" # Client-facing TTS model name
ASR_MODEL_NAME_CLIENT_FACING = "Zephyloid-Whisper-Normal" # New constant for ASR
IMAGE_GEN_MODEL_NAME_CLIENT_FACING = "Zephyrine-InternalFlux-Imagination-Engine"
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


# --- Determine Python Executable ---
# This will be the Python interpreter that is currently running app.py
# When launched via launcher.py, this will be the venv Python.
APP_PYTHON_EXECUTABLE = sys.executable
logger.info(f"🐍 app.py is running with Python: {APP_PYTHON_EXECUTABLE}")
PYTHON_EXECUTABLE = APP_PYTHON_EXECUTABLE
# ---

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
    IDLE_WAIT_SECONDS = 5 # Minimal wait to prevent pure busy-looping
    # How long to wait briefly between batches IF work IS being processed
    ACTIVE_CYCLE_PAUSE_SECONDS = 0.0 # No pause between batches when active
    # --- END MODIFICATION ---

    # Input types eligible for reflection
    reflection_eligible_input_types = ['text', 'reflection_result', 'log_error', 'log_warning', '']
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
                        reflection_input = f"[Self-Reflection on Interaction ID {original_id} (Type: {original_input_type})] Make new Ideas or critically assessed this idea with new idea what if and then verify the answer and what can be done: {original_input}"

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

    @staticmethod
    def _construct_raw_chatml_prompt(
            system_content: Optional[str],
            history_turns: List[Dict[str, str]],  # e.g., [{"role": "user", "content": "..."}]
            current_turn_content: Optional[str] = None,  # Content for the current user/instruction turn
            current_turn_role: str = "user",  # Role for the current_turn_content
            prompt_for_assistant_response: bool = True  # Add "<|im_start|>assistant\n" at the end
    ) -> str:
        """
        Constructs a raw ChatML prompt string.
        History turns are processed in order.
        """
        prompt_parts = []

        if system_content and system_content.strip():
            prompt_parts.append(
                f"{CHATML_START_TOKEN}system{CHATML_NL}{system_content.strip()}{CHATML_END_TOKEN}{CHATML_NL}")

        for turn in history_turns:
            role = turn.get("role", "user").lower()
            content = str(turn.get("content", "")).strip()  # Ensure content is string
            if role not in ["user", "assistant", "system"]:  # System in history is rare but possible
                logger.warning(f"ChatML Constructor: Unknown role '{role}' in history. Skipping.")
                continue
            if content:  # Only add turns with actual content
                prompt_parts.append(f"{CHATML_START_TOKEN}{role}{CHATML_NL}{content}{CHATML_END_TOKEN}{CHATML_NL}")

        if current_turn_content and current_turn_content.strip():
            current_turn_role = current_turn_role.lower()
            if current_turn_role not in ["user", "system"]:  # Typically "user" or "system" for instructions
                logger.warning(
                    f"ChatML Constructor: Invalid role '{current_turn_role}' for current turn. Defaulting to 'user'.")
                current_turn_role = "user"
            prompt_parts.append(
                f"{CHATML_START_TOKEN}{current_turn_role}{CHATML_NL}{current_turn_content.strip()}{CHATML_END_TOKEN}{CHATML_NL}")

        if prompt_for_assistant_response:
            prompt_parts.append(f"{CHATML_START_TOKEN}assistant{CHATML_NL}")

        return "".join(prompt_parts)

    def _count_tokens(self, text: str) -> int:
        """Counts tokens using tiktoken if available, else estimates by characters."""
        if TIKTOKEN_AVAILABLE_APP and cl100k_base_encoder_app and text:
            try:
                return len(cl100k_base_encoder_app.encode(text))
            except Exception as e:
                logger.warning(f"Tiktoken counting error in AIChat: {e}. Falling back to char count.")
                return len(text) // 4  # Rough char to token estimate
        elif text:
            return len(text) // 4  # Rough char to token estimate
        return 0

    def _truncate_rag_context(self, context_str: str, max_tokens: int) -> str:
        """Truncates RAG context string to not exceed max_tokens."""
        if not context_str or max_tokens <= 0:
            return ""

        current_tokens = self._count_tokens(context_str)
        if current_tokens <= max_tokens:
            return context_str

        # Simple truncation by characters (more sophisticated truncation is possible)
        # Estimate characters per token (very rough, depends on tokenizer)
        avg_chars_per_token = 3.5  # Can be adjusted
        target_chars = int(max_tokens * avg_chars_per_token)

        if len(context_str) > target_chars:
            truncated_context = context_str[:target_chars]
            # Try to truncate at a natural boundary (e.g., end of a "Source Chunk")
            last_source_chunk_end = truncated_context.rfind("\n--- End Relevant Context ---")  # if you add this
            if last_source_chunk_end != -1:
                truncated_context = truncated_context[:last_source_chunk_end + len("\n--- End Relevant Context ---")]
            else:
                # Fallback to word boundary
                last_space = truncated_context.rfind(' ')
                if last_space != -1:
                    truncated_context = truncated_context[:last_space]

            logger.warning(
                f"Truncated RAG context from {current_tokens} tokens to approx. {self._count_tokens(truncated_context)} tokens (target: {max_tokens}).")
            return truncated_context + "\n[...RAG context truncated due to length...]"
        return context_str  # Should not be reached if current_tokens > max_tokens and char truncation applied

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

    async def _refine_direct_image_prompt_async(
            self,
            db: Session,
            session_id: str,
            user_image_request: str,  # The prompt from the /v1/images/generations request
            history_rag_str: str,
            recent_direct_history_str: str,
            priority: int = ELP1  # Default to ELP1 for user-facing requests
    ) -> Optional[str]:
        """
        Uses an LLM to refine a user's direct image request into a more detailed image generation prompt,
        considering some conversational context. Runs with the specified priority.
        Strips <think> tags programmatically.
        """
        req_id = f"refineimgprompt-{uuid.uuid4()}"
        log_prefix = f"🖌️ {req_id}|ELP{priority}"  # Include priority in log
        logger.info(
            f"{log_prefix} Refining direct image request for session {session_id}: '{user_image_request[:100]}...'")

        # Use a general-purpose model for this creative task
        refiner_model = self.provider.get_model("general")  # Or "router"
        if not refiner_model:
            logger.error(f"{log_prefix} Model for image prompt refinement ('general') not available.")
            try:
                add_interaction(db, session_id=session_id, mode="image_gen", input_type="log_error",
                                user_input="[ImgPromptRefine Failed - Model Unavailable]",
                                llm_response="Image prompt refinement model not configured.")
            except Exception as db_err:
                logger.error(f"Failed log img prompt refine model error: {db_err}")
            return user_image_request  # Fallback to original request if model unavailable

        prompt_inputs = {
            "original_user_input": user_image_request,
            "history_rag": history_rag_str,
            "recent_direct_history": recent_direct_history_str,
        }

        chain = (
                ChatPromptTemplate.from_template(PROMPT_REFINE_USER_IMAGE_REQUEST)  # Use the new prompt
                | refiner_model
                | StrOutputParser()
        )
        timing_data = {"session_id": session_id, "mode": "image_gen", "execution_time_ms": 0}
        refined_prompt_raw = None

        try:
            refined_prompt_raw = await asyncio.to_thread(
                self._call_llm_with_timing, chain, prompt_inputs, timing_data, priority=priority
            )
            logger.trace(
                f"{log_prefix}: LLM Raw Output for Image Prompt:\n```\n{refined_prompt_raw}\n```")  # Log full raw output

            if not refined_prompt_raw:
                logger.warning(f"{log_prefix}: LLM returned empty image prompt string.")
                return user_image_request

            # Step 1: Remove <think> tags
            prompt_after_think_removal = re.sub(r'<think>.*?</think>', '', refined_prompt_raw,
                                                flags=re.DOTALL | re.IGNORECASE)
            logger.trace(f"{log_prefix}: After <think> removal:\n```\n{prompt_after_think_removal}\n```")

            # Step 2: Remove preambles
            cleaned_prompt_intermediate = prompt_after_think_removal
            preambles = [
                r"^(image generation prompt:|here is the prompt:|sure, here's an image prompt:|okay, based on the context, here's an image prompt:|refined image prompt:)\s*",
                r"^(Okay, I've generated an image prompt based on.*)\n*"
            ]
            for i, preamble_pattern in enumerate(preambles):
                before_preamble_strip = cleaned_prompt_intermediate
                cleaned_prompt_intermediate = re.sub(preamble_pattern, "", cleaned_prompt_intermediate,
                                                     flags=re.IGNORECASE | re.MULTILINE).strip()
                if before_preamble_strip != cleaned_prompt_intermediate:
                    logger.trace(
                        f"{log_prefix}: After preamble strip {i + 1} ('{preamble_pattern}'):\n```\n{cleaned_prompt_intermediate}\n```")

            # Step 3: Remove "Image Generation Prompt:" line
            before_header_strip = cleaned_prompt_intermediate
            cleaned_prompt_intermediate = re.sub(r"^\s*Image Generation Prompt:\s*\n?", "", cleaned_prompt_intermediate,
                                                 flags=re.MULTILINE | re.IGNORECASE).strip()
            if before_header_strip != cleaned_prompt_intermediate:
                logger.trace(
                    f"{log_prefix}: After 'Image Generation Prompt:' header strip:\n```\n{cleaned_prompt_intermediate}\n```")

            # Step 4: Trim whitespace (already done by .strip() in preamble loop, but good for final)
            cleaned_prompt_intermediate = cleaned_prompt_intermediate.strip()
            # logger.trace(f"{log_prefix}: After final strip:\n```\n{cleaned_prompt_intermediate}\n```")

            # Step 5: Remove surrounding quotes
            before_quote_strip = cleaned_prompt_intermediate
            cleaned_prompt_intermediate = re.sub(r'^["\'](.*?)["\']$', r'\1', cleaned_prompt_intermediate)
            if before_quote_strip != cleaned_prompt_intermediate:
                logger.trace(f"{log_prefix}: After surrounding quote strip:\n```\n{cleaned_prompt_intermediate}\n```")

            # Step 6: Remove "Output only this:"
            before_output_only_strip = cleaned_prompt_intermediate
            cleaned_prompt = re.sub(r"\(Output only this\):?", "", cleaned_prompt_intermediate,
                                    flags=re.IGNORECASE).strip()
            if before_output_only_strip != cleaned_prompt:
                logger.trace(f"{log_prefix}: After '(Output only this):' strip:\n```\n{cleaned_prompt}\n```")

            if not cleaned_prompt:
                logger.warning(f"{log_prefix} LLM generated an empty image prompt after all cleaning steps.")
                # Log the raw and intermediate steps if this happens
                logger.debug(
                    f"{log_prefix} DEBUG: Raw='{refined_prompt_raw}', AfterThink='{prompt_after_think_removal}'")
                return user_image_request

            logger.info(f"{log_prefix} Final Refined Image Prompt: '{cleaned_prompt}'")

        except TaskInterruptedException as tie:
            logger.warning(f"🚦 {log_prefix} Image prompt refinement INTERRUPTED: {tie}")
            raise tie  # Propagate for the endpoint to handle
        except Exception as e:
            logger.error(f"❌ {log_prefix} Error refining direct image prompt: {e}")
            logger.exception(f"{log_prefix} ImgPromptRefine Traceback:")
            try:
                add_interaction(db, session_id=session_id, mode="image_gen", input_type="log_error",
                                user_input="[ImgPromptRefine Failed]",
                                llm_response=f"Error: {e}. Raw: {str(refined_prompt_raw)[:200]}")
            except Exception:
                pass
            return user_image_request  # Fallback to original on error

    async def _generate_image_generation_prompt_async(
        self,
        db: Session,
        session_id: str,
        original_user_input: str,
        current_thought_context: str, # Specific idea/ToT output to visualize
        history_rag_str: str,
        file_index_context_str: str,
        recent_direct_history_str: str,
        url_context_str: str,
        log_context_str: str
    ) -> Optional[str]:
        """
        Uses an LLM (e.g., 'general' or 'router') to generate a concise, creative
        image generation prompt based on comprehensive context. Strips <think> tags.
        Called with ELP0 priority.
        """
        req_id = f"imgpromptgen-{uuid.uuid4()}"
        log_prefix = f"🎨 {req_id}|ELP0"
        logger.info(f"{log_prefix} Generating image prompt for session {session_id} with rich context.")

        prompt_gen_model = self.provider.get_model("general") # Or "router"
        if not prompt_gen_model:
            logger.error(f"{log_prefix} Model for image prompt generation ('general') not available.")
            try:
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_error",
                                user_input="[ImgPromptGen Failed - Model Unavailable]",
                                llm_response="Image prompt generation model not configured.")
            except Exception as db_err: logger.error(f"Failed log img prompt gen model error: {db_err}")
            return None

        # Prepare the input dictionary for the prompt template
        prompt_inputs = {
            "original_user_input": original_user_input,
            "current_thought_context": current_thought_context,
            "history_rag": history_rag_str,
            "file_index_context": file_index_context_str,
            "recent_direct_history": recent_direct_history_str,
            "url_context": url_context_str,
            "log_context": log_context_str
        }

        chain = (
            ChatPromptTemplate.from_template(PROMPT_CREATE_IMAGE_PROMPT) # Uses the updated prompt from config
            | prompt_gen_model
            | StrOutputParser()
        )
        timing_data = {"session_id": session_id, "mode": "chat", "execution_time_ms": 0}
        generated_prompt_raw = None

        try:
            # Call LLM with ELP0 priority
            generated_prompt_raw = await asyncio.to_thread(
                self._call_llm_with_timing, chain, prompt_inputs, timing_data, priority=ELP0
            )

            if not generated_prompt_raw:
                logger.warning(f"{log_prefix} LLM returned empty image generation prompt string.")
                return None

            # --- Programmatic <think> tag removal and cleaning ---
            # 1. Remove <think> tags (case-insensitive, multiline)
            cleaned_prompt = re.sub(r'<think>.*?</think>', '', generated_prompt_raw, flags=re.DOTALL | re.IGNORECASE)
            # 2. Remove common LLM preamble/postamble
            preambles = [
                r"^(image generation prompt:|here is the prompt:|sure, here's an image prompt:|okay, based on the context, here's an image prompt:)\s*",
                r"^(Okay, I've generated an image prompt based on.*)\n*"
            ]
            for preamble_pattern in preambles:
                cleaned_prompt = re.sub(preamble_pattern, "", cleaned_prompt, flags=re.IGNORECASE | re.MULTILINE).strip()
            # 3. Remove any "Image Generation Prompt:" line if it somehow survived or was re-added by the model
            cleaned_prompt = re.sub(r"^\s*Image Generation Prompt:\s*\n?", "", cleaned_prompt, flags=re.MULTILINE | re.IGNORECASE).strip()
            # 4. Trim whitespace
            cleaned_prompt = cleaned_prompt.strip()
            # 5. Remove surrounding quotes if the model added them
            cleaned_prompt = re.sub(r'^["\'](.*?)["\']$', r'\1', cleaned_prompt)
            # 6. Remove any remaining "Output only this:" type instructions if they leak
            cleaned_prompt = re.sub(r"\(Output only this\):?", "", cleaned_prompt, flags=re.IGNORECASE).strip()


            if not cleaned_prompt:
                logger.warning(f"{log_prefix} LLM generated an empty image prompt after cleaning.")
                try: add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning", user_input="[ImgPromptGen Empty]", llm_response=f"Raw: {generated_prompt_raw[:200]}")
                except Exception: pass
                return None

            logger.info(f"{log_prefix} Generated image prompt: '{cleaned_prompt}' (Raw len: {len(generated_prompt_raw)}, Cleaned len: {len(cleaned_prompt)})")
            try: add_interaction(db, session_id=session_id, mode="chat", input_type="log_debug", user_input="[ImgPromptGen Success]", llm_response=f"Prompt: '{cleaned_prompt}'. Raw: {generated_prompt_raw[:200]}")
            except Exception: pass
            return cleaned_prompt

        except TaskInterruptedException as tie:
            logger.warning(f"🚦 {log_prefix} Image prompt generation INTERRUPTED: {tie}")
            raise tie
        except Exception as e:
            logger.error(f"❌ {log_prefix} Error generating image prompt: {e}")
            logger.exception(f"{log_prefix} ImgPromptGen Traceback:")
            try: add_interaction(db, session_id=session_id, mode="chat", input_type="log_error", user_input="[ImgPromptGen Failed]", llm_response=f"Error: {e}. Raw: {str(generated_prompt_raw)[:200]}")
            except Exception: pass
            return None

    # --- NEW HELPER: Describe Image with VLM (ELP0) ---
        # app.py -> AIChat class

    async def _describe_generated_image_async(self, db: Session, session_id: str, image_b64: str) -> Optional[str]:
        """
        Sends a base64 image (assumed PNG or similar VLM-compatible) to the VLM
        to get a textual description. Called with ELP0 priority.
        Uses PROMPT_VLM_DESCRIBE_GENERATED_IMAGE.
        """
        req_id = f"imgdesc-{uuid.uuid4()}"
        log_prefix = f"🖼️ {req_id}|ELP0"
        logger.info(f"{log_prefix} Requesting VLM description for generated image (session {session_id}).")

        vlm_model = self.provider.get_model("vlm")
        if not vlm_model:
            logger.error(f"{log_prefix} VLM model not available for image description.")
            try:
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_error",
                                user_input="[ImgDesc Failed - VLM Unavailable]",
                                llm_response="VLM model for description not configured.")
            except Exception as db_log_err:
                logger.error(f"Failed to log VLM unavailable error: {db_log_err}")
            return None

        try:
            image_uri = f"data:image/png;base64,{image_b64}" # Assumes PNG from imagination_worker
            image_content_part = {"type": "image_url", "image_url": {"url": image_uri}}

            # Use the correctly named prompt from config.py
            messages = [HumanMessage(content=[image_content_part, {"type": "text", "text": PROMPT_VLM_DESCRIBE_GENERATED_IMAGE}])]
            chain = vlm_model | StrOutputParser()
            timing_data = {"session_id": session_id, "mode": "chat", "execution_time_ms": 0}

            description = await asyncio.to_thread(
                self._call_llm_with_timing, chain, messages, timing_data, priority=ELP0
            )

            if not description:
                logger.warning(f"{log_prefix} VLM returned empty description for generated image.")
                try:
                    add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning",
                                    user_input="[ImgDesc VLM Empty Response]",
                                    llm_response="VLM returned an empty description for the generated image.")
                except Exception as db_log_err:
                    logger.error(f"Failed to log VLM empty response: {db_log_err}")
                return None

            cleaned_description = description.strip()
            logger.info(f"{log_prefix} VLM description received (first 100 chars): '{cleaned_description[:100]}...'")

            try:
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_debug",
                                user_input="[ImgDesc Success]",
                                llm_response=f"VLM Desc (generated img): {cleaned_description[:200]}")
            except Exception as db_log_err:
                 logger.error(f"Failed to log ImgDesc success: {db_log_err}")

            return cleaned_description

        except TaskInterruptedException as tie:
            logger.warning(f"🚦 {log_prefix} VLM image description INTERRUPTED: {tie}")
            try:
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning",
                                user_input="[ImgDesc Interrupted]",
                                llm_response=f"VLM image description task was interrupted: {tie}")
            except Exception as db_log_err:
                logger.error(f"Failed to log ImgDesc interruption: {db_log_err}")
            raise tie
        except Exception as e:
            logger.error(f"❌ {log_prefix} Error getting VLM description for generated image: {e}")
            logger.exception(f"{log_prefix} ImgDesc Traceback:")
            try:
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_error",
                                user_input="[ImgDesc Failed - VLM Error]",
                                llm_response=f"Error during VLM description of generated image: {e}")
            except Exception as db_log_err:
                logger.error(f"Failed to log VLM error: {db_log_err}")
            return None

    def _get_rag_retriever_thread_wrapper(self, db_session: Session, user_input_str: str, priority_val: int) -> Dict[
        str, Any]:
        """
        Synchronous wrapper for _get_rag_retriever to be run in asyncio.to_thread.
        Catches exceptions and returns a structured dictionary.
        """
        log_prefix = f"RAGThreadWrap|ELP{priority_val}|{self.current_session_id or 'NoSession'}"
        try:
            logger.debug(f"{log_prefix}: Executing _get_rag_retriever in thread...")
            # Call the actual synchronous _get_rag_retriever method
            result_tuple = self._get_rag_retriever(db_session, user_input_str, priority_val)
            logger.debug(
                f"{log_prefix}: _get_rag_retriever completed. Result tuple length: {len(result_tuple) if isinstance(result_tuple, tuple) else 'N/A'}")
            return {"status": "success", "data": result_tuple}
        except TaskInterruptedException as tie_wrapper:
            logger.warning(
                f"🚦 {log_prefix}: TaskInterruptedException caught: {tie_wrapper}. Returning interruption status.")
            return {"status": "interrupted", "error_message": str(tie_wrapper)}
        except Exception as e_wrapper:
            logger.error(f"❌ {log_prefix}: Exception caught: {e_wrapper}")
            logger.exception(f"{log_prefix} _get_rag_retriever_thread_wrapper Exception Details:")
            return {"status": "error", "error_message": str(e_wrapper)}

    def _get_rag_retriever(self, db: Session, user_input: str, priority: int = ELP0) -> Tuple[
        Optional[Any],  # url_retriever
        Optional[Any],  # session_history_retriever (for on-the-fly session chat)
        Optional[Any],  # reflection_chunks_retriever (for pre-indexed global reflections) # <<< THIS IS THE ONE
        str             # session_history_ids_str (IDs of chat turns embedded on-the-fly for session RAG)
    ]:
        log_prefix = f"RAGRetriever|ELP{priority}|{self.current_session_id or 'NoSession'}"
        logger.critical(
            f"@@@ ENTERING _get_rag_retriever for session {self.current_session_id}, priority ELP{priority} @@@")

        # Initialize all return variables to None or default
        url_retriever: Optional[Any] = None
        session_history_retriever: Optional[Any] = None
        # ===>>> POTENTIAL ISSUE HERE: Was reflection_chunks_retriever removed from an earlier version of this method?
        # If this was accidentally removed or commented out during refactoring,
        # and the return statement still expects 4 items, this would cause the error.
        # Let's assume based on the error that one of the intended return values is missing.
        # The type hint suggests it should be `reflection_chunks_retriever`.

        reflection_chunks_retriever: Optional[Any] = None # ENSURE THIS IS INITIALIZED
        session_history_ids_str: str = ""

        self.vectorstore_history: Optional[Chroma] = None # This is for session history, not global reflections
        session_history_ids_set = set()

        try:
            # 1. URL Retriever
            logger.debug(f"{log_prefix} Step 1: URL Retriever processing...")
            if hasattr(self, 'vectorstore_url') and self.vectorstore_url:
                # ... (url_retriever logic)
                try:
                    url_rag_k_val = RAG_URL_COUNT
                    url_retriever = self.vectorstore_url.as_retriever(search_kwargs={"k": url_rag_k_val})
                    logger.trace(f"{log_prefix} Using existing URL vector store retriever (k={url_rag_k_val}).")
                except Exception as e_url_vs:
                    logger.error(
                        f"{log_prefix} Failed to get URL retriever from existing self.vectorstore_url: {e_url_vs}")
                    url_retriever = None
            else:
                logger.trace(f"{log_prefix} No URL vector store (self.vectorstore_url) available.")
            logger.debug(f"{log_prefix} Step 1 complete. url_retriever type: {type(url_retriever)}")


            # 2. Session Chat History Retriever
            logger.debug(f"{log_prefix} Step 2: Session Chat History Retriever processing...")
            # ... (session_history_retriever logic) ...
            chat_interactions_for_rag = get_recent_interactions( # Synchronous DB call
                db,
                limit=RAG_HISTORY_COUNT * 2,
                session_id=self.current_session_id,
                mode="chat",
                include_logs=False
            )
            session_history_texts_to_embed = []
            if chat_interactions_for_rag:
                chat_interactions_for_rag.reverse()
                for interaction in chat_interactions_for_rag:
                    text_to_embed_content = None
                    if interaction.user_input and interaction.input_type == 'text':
                        text_to_embed_content = f"User (current session): {interaction.user_input}"
                    elif interaction.llm_response and interaction.input_type == 'llm_response':
                        if len(interaction.llm_response.strip()) > 10:
                            text_to_embed_content = f"AI (current session): {interaction.llm_response}"

                    if text_to_embed_content and interaction.id not in session_history_ids_set:
                        session_history_texts_to_embed.append(text_to_embed_content)
                        session_history_ids_set.add(interaction.id)
                logger.debug(
                    f"{log_prefix} Prepared {len(session_history_texts_to_embed)} text snippets from session chat for on-the-fly embedding.")
            else:
                logger.debug(f"{log_prefix} No recent session chat interactions found for RAG.")

            if session_history_texts_to_embed:
                logger.debug(
                    f"{log_prefix} Embedding {len(session_history_texts_to_embed)} session history texts (Embedding Priority ELP{priority})...")
                try:
                    if not self.provider.embeddings:
                        raise ValueError(
                            "Embeddings provider (self.provider.embeddings) not initialized for session history RAG.")
                    embedded_session_history_vectors = self.provider.embeddings.embed_documents(
                        session_history_texts_to_embed, priority=priority
                    )
                    self.vectorstore_history = Chroma.from_embeddings(
                        text_embeddings=embedded_session_history_vectors,
                        documents=session_history_texts_to_embed,
                        embedding=self.provider.embeddings
                    )
                    k_session_hist = RAG_HISTORY_COUNT // 2 if RAG_HISTORY_COUNT > 2 else RAG_HISTORY_COUNT
                    if k_session_hist < 1 and RAG_HISTORY_COUNT >= 1: k_session_hist = 1
                    session_history_retriever = self.vectorstore_history.as_retriever(
                        search_kwargs={"k": k_session_hist}
                    )
                    logger.debug(
                        f"{log_prefix} Created temporary Session History retriever (k={k_session_hist}) using on-the-fly embeddings.")
                except TaskInterruptedException as tie_hist:
                    logger.warning(f"🚦 {log_prefix} Session History RAG embedding INTERRUPTED: {tie_hist}")
                    session_history_retriever = None
                    self.vectorstore_history = None
                    raise tie_hist
                except Exception as e_hist_vs:
                    logger.error(f"❌ {log_prefix} Failed temporary session history vector store creation: {e_hist_vs}")
                    logger.exception(f"{log_prefix} Session History VS Creation Traceback:")
                    session_history_retriever = None
                    self.vectorstore_history = None
            else:
                logger.debug(f"{log_prefix} No suitable text found in session interactions for on-the-fly history RAG.")
                session_history_retriever = None
            logger.debug(
                f"{log_prefix} Step 2 complete. session_history_retriever type: {type(session_history_retriever)}")


            # 3. Global Reflection Chunks Retriever
            logger.debug(f"{log_prefix} Step 3: Global Reflection Chunks Retriever processing...")
            active_global_reflection_vs = get_global_reflection_vectorstore() # Synchronous call

            if active_global_reflection_vs:
                logger.debug(f"{log_prefix} Global reflection vector store is available. Creating retriever for it.")
                try:
                    # Ensure k_reflection_chunks is at least 1 if RAG_HISTORY_COUNT is positive
                    k_reflection_chunks = RAG_HISTORY_COUNT // 2
                    if RAG_HISTORY_COUNT > 0 and k_reflection_chunks < 1:
                        k_reflection_chunks = 1
                    elif RAG_HISTORY_COUNT == 0: # If RAG_HISTORY_COUNT is 0, k_reflection_chunks might be 0
                        k_reflection_chunks = 0 # or handle as an error/skip

                    if k_reflection_chunks > 0: # Only create retriever if k > 0
                        reflection_chunks_retriever = active_global_reflection_vs.as_retriever(
                            search_kwargs={"k": k_reflection_chunks}
                        )
                        logger.debug(
                            f"{log_prefix} Created retriever for global reflection chunks (k={k_reflection_chunks}).")
                    else:
                        logger.debug(f"{log_prefix} Skipping reflection retriever as k_reflection_chunks is 0.")
                        reflection_chunks_retriever = None

                except Exception as e_refl_retr:
                    logger.error(f"❌ {log_prefix} Failed to create retriever from global reflection VS: {e_refl_retr}")
                    reflection_chunks_retriever = None
            else:
                logger.debug(f"{log_prefix} Global reflection vector store not available or not initialized.")
                reflection_chunks_retriever = None
            logger.debug(
                f"{log_prefix} Step 3 complete. reflection_chunks_retriever type: {type(reflection_chunks_retriever)}")


            session_history_ids_str = ",".join(map(str, sorted(list(session_history_ids_set))))
            logger.trace(
                f"{log_prefix} Session Chat History Interaction IDs embedded on-the-fly for RAG: [{session_history_ids_str}]")

            final_log_msg = (
                f"{log_prefix} RAG retriever preparation complete. "
                f"URL Retr: {'Yes' if url_retriever else 'No'}, "
                f"SessionChat Retr: {'Yes' if session_history_retriever else 'No'}, "
                f"ReflectionChunk Retr: {'Yes' if reflection_chunks_retriever else 'No'}." # Added this
            )
            logger.info(final_log_msg)

            # === CRITICAL RETURN STATEMENT ===
            # Debug print just before returning
            logger.critical(f"@@@ _get_rag_retriever RETURNING (try block): "
                            f"url_type={type(url_retriever)}, "
                            f"session_hist_type={type(session_history_retriever)}, "
                            f"reflection_chunk_type={type(reflection_chunks_retriever)}, "  # Added type log
                            f"ids_str_type={type(session_history_ids_str)} @@@")
            # Ensure all 4 items are present in the tuple being returned.

            ret_val = (url_retriever, session_history_retriever, reflection_chunks_retriever, session_history_ids_str)
            logger.critical(
                f"!!!!! ABOUT TO RETURN {len(ret_val)} items from _get_rag_retriever. Types: {[type(x) for x in ret_val]} !!!!!")
            return url_retriever, session_history_retriever, reflection_chunks_retriever, session_history_ids_str

        except TaskInterruptedException as tie:
            logger.warning(
                f"🚦 {log_prefix} TaskInterruptedException caught within _get_rag_retriever: {tie}. Re-raising.")
            raise tie # This will be caught by the wrapper
        except Exception as e_outer:
            logger.error(f"❌❌ {log_prefix} UNHANDLED EXCEPTION in _get_rag_retriever: {e_outer}")
            logger.exception(f"{log_prefix} _get_rag_retriever Outer Exception Traceback:")
            logger.critical(
                "!!! _get_rag_retriever: Reached outer exception handler. Returning default 4-tuple of Nones/empty string.")
            # Ensure 4 items are returned even in this fallback
            return None, None, None, ""

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
                db = SessionLocal()  # Log failure to DB
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
        request_id = f"classify-{interaction_data.get('session_id', 'unknownsession')}-{uuid.uuid4()}"
        log_prefix = f"🤔 Classify|{request_id}"
        logger.info(f"{log_prefix} Classifying input complexity for: '{user_input[:50]}...'")

        history_summary = self._get_history_summary(db, MEMORY_SIZE)

        classification_model = self.provider.get_model("router")
        if not classification_model:
            logger.warning(f"{log_prefix} Router model not found for classification, falling back to default.")
            classification_model = self.provider.get_model("default")

        if not classification_model:
            logger.error(f"{log_prefix} ❌ Default model also not found! Cannot perform input classification.")
            # ... (existing fallback and DB logging) ...
            return "chat_simple"

        # --- MODIFICATION: Use StrOutputParser first, then manually parse JSON ---
        # The chain will first output the raw string from the LLM.
        # We will then attempt to extract JSON from this raw string.
        classification_chain_raw_output = (
                self.input_classification_prompt  # PROMPT_COMPLEXITY_CLASSIFICATION
                | classification_model
                | StrOutputParser()  # Get raw string output
        )

        json_parser = JsonOutputParser()  # We'll use this after cleaning

        attempts = 0
        last_error = None
        classification_reason_from_llm = "N/A"  # For logging if JSON parse fails

        while attempts < DEEP_THOUGHT_RETRY_ATTEMPTS:
            attempts += 1
            logger.debug(f"{log_prefix} Classification attempt {attempts}/{DEEP_THOUGHT_RETRY_ATTEMPTS}")
            raw_llm_response_text = ""  # Initialize
            try:
                prompt_inputs_for_classification = {"input": user_input, "history_summary": history_summary}

                # Call LLM to get raw string output
                raw_llm_response_text = self._call_llm_with_timing(
                    classification_chain_raw_output,
                    prompt_inputs_for_classification,
                    interaction_data  # For timing
                    # Priority ELP0 is default for _call_llm_with_timing
                )
                logger.trace(
                    f"{log_prefix} Raw LLM output for classification (Attempt {attempts}):\n{raw_llm_response_text}")

                # --- Step 1: Clean <think> tags (if any) ---
                text_after_think_removal = re.sub(r'<think>.*?</think>', '', raw_llm_response_text,
                                                  flags=re.DOTALL | re.IGNORECASE).strip()

                # --- Step 2: Robustly find and extract the JSON block ---
                # Try to find JSON within ```json ... ``` or standalone { ... }
                json_str_to_parse = None
                json_markdown_match = re.search(r"```json\s*(.*?)\s*```", text_after_think_removal, re.DOTALL)
                if json_markdown_match:
                    json_str_to_parse = json_markdown_match.group(1).strip()
                    logger.trace(f"{log_prefix} Extracted JSON from markdown block: {json_str_to_parse}")
                else:
                    # Fallback: Find the first '{' and last '}'
                    first_brace = text_after_think_removal.find('{')
                    last_brace = text_after_think_removal.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        json_str_to_parse = text_after_think_removal[first_brace: last_brace + 1].strip()
                        logger.trace(f"{log_prefix} Extracted JSON using find '{{...}}' fallback: {json_str_to_parse}")
                    else:
                        # If no clear JSON block, the raw response itself might be non-JSON (as in your error)
                        # or it might be malformed. Let the json_parser try the cleaned text_after_think_removal.
                        # If that also fails, the JSONDecodeError will be caught.
                        json_str_to_parse = text_after_think_removal  # Try parsing the whole cleaned string
                        logger.warning(
                            f"{log_prefix} No clear JSON block found, will attempt to parse: '{json_str_to_parse[:100]}...'")

                if not json_str_to_parse:  # If even after fallbacks, string is empty
                    logger.warning(
                        f"{log_prefix} After cleaning, string to parse for JSON is empty. Raw was: '{raw_llm_response_text}'")
                    last_error = ValueError("LLM response for classification was empty after cleaning.")
                    # Log this specific state to DB
                    try:
                        add_interaction(db, session_id=interaction_data.get("session_id"), mode="chat",
                                        input_type="log_warning",
                                        user_input=f"[Classify Empty after Clean Attempt {attempts}]",
                                        llm_response=f"Raw: {raw_llm_response_text[:500]}")
                    except Exception:
                        pass
                    if attempts < DEEP_THOUGHT_RETRY_ATTEMPTS:
                        time.sleep(0.5 + attempts * 0.5); continue
                    else:
                        break  # Max retries

                # --- Step 3: Parse the extracted JSON string ---
                parsed_json = json_parser.parse(json_str_to_parse)  # Use the Langchain parser's method

                classification = parsed_json.get("classification", "chat_simple")
                reason = str(parsed_json.get("reason", "N/A"))
                classification_reason_from_llm = reason  # Store for logging if needed

                if classification not in ["chat_simple", "chat_complex", "agent_task"]:
                    logger.warning(
                        f"{log_prefix} Classification LLM returned invalid category '{classification}', defaulting to chat_simple. Parsed from: '{json_str_to_parse}'")
                    classification = "chat_simple"

                interaction_data['classification'] = classification
                interaction_data['classification_reason'] = reason
                logger.info(f"{log_prefix} ✅ Input classified as: '{classification}'. Reason: {reason}")
                return classification  # Success

            except json.JSONDecodeError as json_e:  # Catch errors from json_parser.parse or direct json.loads
                last_error = json_e
                logger.warning(
                    f"⚠️ {log_prefix} Error parsing JSON for classification (Attempt {attempts}/{DEEP_THOUGHT_RETRY_ATTEMPTS}): {json_e}. String tried: '{json_str_to_parse if 'json_str_to_parse' in locals() else 'N/A'}'. Raw LLM output: '{raw_llm_response_text[:200]}...'")
                if attempts < DEEP_THOUGHT_RETRY_ATTEMPTS:
                    time.sleep(0.5 + attempts * 0.5)
                else:
                    break  # Max retries
            except Exception as e:  # Catch other errors like OutputParserException from Langchain if parse fails
                last_error = e
                logger.warning(
                    f"⚠️ {log_prefix} Error during classification processing (Attempt {attempts}/{DEEP_THOUGHT_RETRY_ATTEMPTS}): {e}. Raw LLM output: '{raw_llm_response_text[:200]}...'")
                if attempts < DEEP_THOUGHT_RETRY_ATTEMPTS:
                    time.sleep(0.5 + attempts * 0.5)
                else:
                    break  # Max retries

        # After retries or if loop broken
        logger.error(
            f"{log_prefix} ❌ Max retries ({attempts}/{DEEP_THOUGHT_RETRY_ATTEMPTS}) for input classification. Last error: {last_error}")
        interaction_data['classification'] = "chat_simple"  # Fallback
        interaction_data[
            'classification_reason'] = f"Classification failed after {attempts} retries. Last error: {last_error}. Last LLM reason attempt: '{classification_reason_from_llm}'"
        try:
            add_interaction(db, session_id=interaction_data.get("session_id"), mode="chat", input_type="log_error",
                            llm_response=f"Input classification failed after {attempts} attempts. Last LLM output: '{raw_llm_response_text[:500]}'. Error: {last_error}")
        except Exception as db_err:
            logger.error(f"Failed log classification retry error: {db_err}")
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
        """
        Analyzes emotion/context (synchronous).
        Updates interaction_data with the analysis result or error.
        Returns the analysis string or an error message.
        """
        request_id_suffix = str(uuid.uuid4())[:8]
        log_prefix = f"😊 EmotionAnalyze|{interaction_data.get('session_id', 'unknown')[:8]}-{request_id_suffix}"
        logger.info(f"{log_prefix} Analyzing input emotion/context for: '{user_input[:50]}...'")

        history_summary = self._get_history_summary(db, MEMORY_SIZE)  # MEMORY_SIZE from config

        # Determine which model role to use for emotion analysis
        emotion_model_role = "router"  # Configurable: could be "router" or a dedicated role
        emotion_model = self.provider.get_model(emotion_model_role)

        analysis_result_for_return = "Analysis unavailable."  # Default return

        if not emotion_model:
            error_msg = f"Emotion analysis model ('{emotion_model_role}') not available."
            logger.error(f"{log_prefix} {error_msg}")
            interaction_data['emotion_context_analysis'] = error_msg  # Update main interaction data
            analysis_result_for_return = f"Could not analyze emotion (model '{emotion_model_role}' unavailable)."
            try:
                add_interaction(db, session_id=interaction_data.get("session_id"), mode="chat",
                                input_type="log_error",
                                user_input="[Emotion Analysis Init Failed]",
                                llm_response=error_msg)
            except Exception as db_err:
                logger.error(f"{log_prefix} Failed to log emotion model unavailable error: {db_err}")
            return analysis_result_for_return

        try:
            # Construct the chain with the fetched model
            chain = (self.emotion_analysis_prompt | emotion_model | StrOutputParser())

            # _call_llm_with_timing mutates interaction_data for 'execution_time_ms'
            # It uses ELP0 by default unless overridden
            analysis = self._call_llm_with_timing(
                chain,
                {"input": user_input, "history_summary": history_summary},
                interaction_data  # Pass the main interaction_data for timing updates
            )

            # Clean up the analysis string
            cleaned_analysis = self._cleanup_llm_output(analysis)  # Use existing cleanup

            logger.info(f"{log_prefix} Emotion/Context Analysis Result: {cleaned_analysis[:200]}...")
            interaction_data['emotion_context_analysis'] = cleaned_analysis  # Update main interaction data
            analysis_result_for_return = cleaned_analysis

            # Log the successful analysis (optional, as it's stored in interaction_data)
            # try:
            #     add_interaction(db, session_id=interaction_data.get("session_id"), mode="chat",
            #                     input_type="log_debug", user_input="[Emotion Analysis Success]",
            #                     llm_response=cleaned_analysis[:500])
            # except Exception: pass

        except TaskInterruptedException as tie:
            error_msg = f"[Emotion Analysis Interrupted by higher priority task: {tie}]"
            logger.warning(f"🚦 {log_prefix} Emotion analysis INTERRUPTED: {tie}")
            interaction_data['emotion_context_analysis'] = error_msg
            analysis_result_for_return = error_msg
            try:
                add_interaction(db, session_id=interaction_data.get("session_id"), mode="chat",
                                input_type="log_warning",
                                user_input="[Emotion Analysis Interrupted]",
                                llm_response=str(tie)[:4000])
            except Exception as db_err:
                logger.error(f"{log_prefix} Failed log emotion analysis interruption: {db_err}")
            # For emotion analysis, we typically don't re-raise interruption to stop the whole background_generate,
            # just record that it was interrupted.
        except Exception as e:
            error_msg = f"Error during emotion analysis: {e}"
            logger.error(f"❌ {log_prefix} {error_msg}")
            logger.exception(f"{log_prefix} Emotion Analysis Traceback:")
            interaction_data['emotion_context_analysis'] = error_msg
            analysis_result_for_return = f"Could not analyze emotion (processing error: {type(e).__name__})."
            try:
                add_interaction(db, session_id=interaction_data.get("session_id"), mode="chat",
                                input_type="log_error",
                                user_input="[Emotion Analysis Failed]",
                                llm_response=error_msg[:4000])
            except Exception as db_err:
                logger.error(f"{log_prefix} Failed to log emotion analysis error: {db_err}")

        return analysis_result_for_return


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

    def _analyze_assistant_action(self, db: Session, user_input: str, session_id: str, context: Dict[str, str]) -> \
    Optional[Dict[str, Any]]:
        """
        Calls LLM (ELP0) to check if input implies a macOS action, extracts parameters.
        Uses a more robust method to find the JSON block, ignoring <think> tags during extraction.
        Handles TaskInterruptedException by re-raising.
        Retries JSON extraction on other errors. Ensures a valid model key is returned.
        """
        request_id_suffix = str(uuid.uuid4())[:8]  # Shorter suffix for log prefix
        log_prefix = f"🤔 ActionAnalyze|ELP0|{session_id[:8]}-{request_id_suffix}"
        logger.info(f"{log_prefix} Analyzing input for potential Assistant Action: '{user_input[:50]}...'")

        prompt_input = {
            "input": user_input,
            "history_summary": context.get("history_summary", "N/A"),
            "log_context": context.get("log_context", "N/A"),
            "recent_direct_history": context.get("recent_direct_history", "N/A")
        }

        action_analysis_model = self.provider.get_model("router")
        if not action_analysis_model:
            logger.warning(f"{log_prefix} Router model not found for action analysis, falling back to default.")
            action_analysis_model = self.provider.get_model("default")

        if not action_analysis_model:
            logger.error(f"{log_prefix} ❌ Action analysis model (router/default) not available!")
            try:
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_error",
                                user_input="[Action Analysis Failed - Model Unavailable]",
                                llm_response="Action analysis model unavailable.")
            except Exception as db_err:
                logger.error(f"{log_prefix} Failed log action analysis model error: {db_err}")
            return None

        # Chain to get raw string output first
        analysis_chain_raw_output = (
                ChatPromptTemplate.from_template(PROMPT_ASSISTANT_ACTION_ANALYSIS)
                | action_analysis_model
                | StrOutputParser()
        )

        json_parser = JsonOutputParser()  # For parsing the cleaned JSON string

        action_timing_data = {"session_id": session_id, "mode": "chat", "execution_time_ms": 0}
        last_error: Optional[Exception] = None
        raw_llm_response_full_for_logging = "Error: Analysis LLM call failed or did not produce parsable output."
        classification_reason_from_llm_if_any = "N/A (No JSON reason extracted)"

        for attempt in range(DEEP_THOUGHT_RETRY_ATTEMPTS):
            current_attempt_num = attempt + 1
            logger.debug(
                f"{log_prefix} Assistant Action analysis attempt {current_attempt_num}/{DEEP_THOUGHT_RETRY_ATTEMPTS}")

            raw_llm_response_text_current_attempt = ""  # Specific to this attempt
            json_str_to_parse_current_attempt: Optional[str] = None
            analysis_result_current_attempt: Optional[Dict[str, Any]] = None

            try:
                raw_llm_response_text_current_attempt = self._call_llm_with_timing(
                    analysis_chain_raw_output,
                    prompt_input,
                    action_timing_data,  # This dict is mutated by _call_llm_with_timing
                    priority=ELP0
                )
                raw_llm_response_full_for_logging = raw_llm_response_text_current_attempt  # Update for logging on error
                logger.trace(
                    f"{log_prefix} Raw LLM Analysis Response (Attempt {current_attempt_num}):\n{raw_llm_response_text_current_attempt}")

                text_after_think_removal = re.sub(r'<think>.*?</think>', '', raw_llm_response_text_current_attempt,
                                                  flags=re.DOTALL | re.IGNORECASE).strip()

                json_markdown_match = re.search(r"```json\s*(.*?)\s*```", text_after_think_removal, re.DOTALL)
                if json_markdown_match:
                    json_str_to_parse_current_attempt = json_markdown_match.group(1).strip()
                    logger.trace(
                        f"{log_prefix} Extracted JSON from markdown block: {json_str_to_parse_current_attempt}")
                else:
                    first_brace = text_after_think_removal.find('{')
                    last_brace = text_after_think_removal.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        json_str_to_parse_current_attempt = text_after_think_removal[
                                                            first_brace: last_brace + 1].strip()
                        logger.trace(
                            f"{log_prefix} Extracted JSON using find '{{...}}' fallback: {json_str_to_parse_current_attempt}")
                    else:
                        json_str_to_parse_current_attempt = text_after_think_removal
                        logger.warning(
                            f"{log_prefix} No clear JSON block found in (cleaned) '{text_after_think_removal[:100]}...', will attempt to parse.")

                if not json_str_to_parse_current_attempt or not json_str_to_parse_current_attempt.strip():
                    logger.warning(
                        f"{log_prefix} After cleaning, string to parse for JSON is empty. Raw was: '{raw_llm_response_text_current_attempt}'")
                    last_error = ValueError("LLM response for action analysis was empty after cleaning.")
                    try:
                        add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning",
                                        user_input=f"[Action Analysis Empty after Clean Attempt {current_attempt_num}]",
                                        llm_response=f"Raw: {raw_llm_response_text_current_attempt[:500]}")
                    except Exception:
                        pass
                    if current_attempt_num < DEEP_THOUGHT_RETRY_ATTEMPTS:
                        time.sleep(0.5 + attempt * 0.5); continue
                    else:
                        break

                analysis_result_current_attempt = json_parser.parse(json_str_to_parse_current_attempt)

                # Validate structure
                if isinstance(analysis_result_current_attempt,
                              dict) and "action_type" in analysis_result_current_attempt and "parameters" in analysis_result_current_attempt:
                    action_type = analysis_result_current_attempt.get("action_type")
                    parameters = analysis_result_current_attempt.get("parameters", {})
                    explanation = analysis_result_current_attempt.get("explanation", "N/A")
                    classification_reason_from_llm_if_any = explanation  # Store the latest good explanation

                    logger.info(
                        f"✅ {log_prefix} Assistant Action analysis successful (Attempt {current_attempt_num}): Type='{action_type}', Params={parameters}, Explanation='{explanation[:50]}...'")
                    try:
                        add_interaction(db,
                                        session_id=session_id, mode="chat", input_type="log_info",
                                        user_input=f"Assistant Action Analysis OK for: {user_input[:100]}...",
                                        llm_response=f"Action Type: {action_type}, Explanation: {explanation}",
                                        assistant_action_analysis_json=json.dumps(analysis_result_current_attempt),
                                        # Store parsed JSON
                                        assistant_action_type=action_type,
                                        assistant_action_params=json.dumps(parameters)
                                        )
                    except Exception as db_err:
                        logger.error(f"{log_prefix} Failed log action analysis success: {db_err}")

                    return analysis_result_current_attempt if action_type != "no_action" else None
                else:
                    logger.warning(
                        f"{log_prefix} Assistant Action analysis produced invalid JSON structure (Attempt {current_attempt_num}): {analysis_result_current_attempt}. String parsed: '{json_str_to_parse_current_attempt}'")
                    last_error = ValueError("Invalid JSON structure after parsing for action analysis")
                    try:
                        add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning",
                                        user_input=f"Action Analysis Invalid Structure (Attempt {current_attempt_num})",
                                        llm_response=f"Parsed: {str(analysis_result_current_attempt)[:500]}. From: '{json_str_to_parse_current_attempt[:200]}'. Raw: {raw_llm_response_text_current_attempt[:500]}",
                                        assistant_action_analysis_json=raw_llm_response_text_current_attempt[:4000])
                    except Exception as db_err:
                        logger.error(f"{log_prefix} Failed log invalid structure: {db_err}")
                    # Continue to next retry

            except TaskInterruptedException as tie:
                logger.warning(f"🚦 {log_prefix} Action Analysis INTERRUPTED (Attempt {current_attempt_num}): {tie}")
                raise tie
            except json.JSONDecodeError as json_e:
                last_error = json_e
                logger.warning(
                    f"⚠️ {log_prefix} Failed to parse JSON (Attempt {current_attempt_num}): {json_e}. String tried: '{json_str_to_parse_current_attempt if json_str_to_parse_current_attempt is not None else 'N/A'}'. Raw LLM: '{raw_llm_response_text_current_attempt[:200]}...'")
                try:
                    add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning",
                                    user_input=f"Action Analysis JSON Parse FAILED (Attempt {current_attempt_num})",
                                    llm_response=f"Error: {json_e}. Tried to parse: '{str(json_str_to_parse_current_attempt)[:500]}'. Raw LLM: {raw_llm_response_text_current_attempt[:1000]}",
                                    assistant_action_analysis_json=raw_llm_response_text_current_attempt[:4000])
                except Exception as db_err:
                    logger.error(f"{log_prefix} Failed log JSON parse failure: {db_err}")
            except Exception as e:
                last_error = e
                logger.warning(
                    f"⚠️ {log_prefix} Error during Action Analysis processing (Attempt {current_attempt_num}): {e}")
                # --- DIAGNOSTIC LOGS for 'got coroutine' type errors ---
                logger.warning(
                    f"   Type of raw_llm_response_text_current_attempt at this error point: {type(raw_llm_response_text_current_attempt)}")
                if asyncio.iscoroutine(raw_llm_response_text_current_attempt):
                    logger.warning(
                        f"   Content of raw_llm_response_text_current_attempt (it is a coroutine): {str(raw_llm_response_text_current_attempt)}")
                else:  # Log a snippet if it's not a coroutine, to see what it was
                    logger.warning(
                        f"   Content of raw_llm_response_text_current_attempt: {str(raw_llm_response_text_current_attempt)[:200]}...")
                # --- END DIAGNOSTIC LOGS ---
                try:
                    add_interaction(db, session_id=session_id, mode="chat", input_type="log_warning",
                                    user_input=f"Action Analysis FAILED (Attempt {current_attempt_num})",
                                    llm_response=f"Error: {e}. Raw (type {type(raw_llm_response_text_current_attempt)}): {str(raw_llm_response_text_current_attempt)[:1000]}",
                                    assistant_action_analysis_json=str(raw_llm_response_text_current_attempt)[:4000])
                except Exception as db_err:
                    logger.error(f"{log_prefix} Failed log general analysis failure: {db_err}")

            # Wait before retrying if not last attempt and not interrupted
            if current_attempt_num < DEEP_THOUGHT_RETRY_ATTEMPTS:
                time.sleep(0.5 + attempt * 0.5)  # Correct: synchronous sleep
            else:  # Max retries reached
                logger.error(
                    f"❌ {log_prefix} Max retries ({DEEP_THOUGHT_RETRY_ATTEMPTS}) reached for Assistant Action analysis. Last error: {last_error}")
                logger.error(f"   Type of last_error: {type(last_error)}")
                # Log final failure to DB, using the last raw response we captured
                try:
                    add_interaction(db, session_id=session_id, mode="chat", input_type="log_error",
                                    user_input=f"Action Analysis Max Retries for: {user_input[:100]}...",
                                    llm_response=f"Max retries. Last Error: {last_error}. Last LLM output captured: {raw_llm_response_full_for_logging[:500]}",
                                    assistant_action_analysis_json=raw_llm_response_full_for_logging[:4000])
                except Exception as db_final_err:
                    logger.error(f"{log_prefix} Failed to log max_retries error to DB: {db_final_err}")
                return None

                # Fallback if loop finishes due to max retries without returning successfully
        logger.error(
            f"{log_prefix} Exited Assistant Action analysis loop after {DEEP_THOUGHT_RETRY_ATTEMPTS} attempts without success. Last error: {last_error}")
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
    async def _route_to_specialist(self, db: Session, session_id: str, user_input: str, context: Dict) -> Tuple[
        str, str, str]:
        log_prefix = f"🧠 Route|ELP0|{session_id}"
        logger.info(f"{log_prefix}: Routing request with direct JSON output expected from router...")

        # --- Use the "router" model, which should be a general instruction-following model ---
        router_model = self.provider.get_model("router")
        default_model_key = "general"

        if not router_model:
            # ... (error handling as before) ...
            return default_model_key, user_input, "Router model unavailable."

        prompt_input_router = {  # Inputs for PROMPT_ROUTER
            "input": user_input,
            "pending_tot_result": context.get("pending_tot_result", "None."),
            "recent_direct_history": context.get("recent_direct_history", "None."),
            "context": context.get("url_context", "None."),  # URL context
            "history_rag": context.get("history_rag", "None."),
            "file_index_context": context.get("file_index_context", "None."),
            "log_context": context.get("log_context", "None."),
            "emotion_analysis": context.get("emotion_context_analysis", "N/A."),
            "imagined_image_vlm_description": context.get("imagined_image_vlm_description", "None.")
        }

        # Expect direct JSON output from the router model
        router_chain = (
                ChatPromptTemplate.from_template(PROMPT_ROUTER)
                | router_model
                | JsonOutputParser()  # Parse JSON directly from router output
        )

        router_timing_data = {"session_id": session_id, "mode": "chat", "execution_time_ms": 0}
        routing_result_json: Optional[Dict] = None
        last_error = None

        for attempt in range(DEEP_THOUGHT_RETRY_ATTEMPTS):
            logger.debug(f"{log_prefix} Router direct JSON attempt {attempt + 1}/{DEEP_THOUGHT_RETRY_ATTEMPTS}")
            try:
                routing_result_json = await asyncio.to_thread(
                    self._call_llm_with_timing, router_chain, prompt_input_router,
                    router_timing_data, priority=ELP0
                )
                # _call_llm_with_timing returns the parsed JSON if JsonOutputParser succeeds

                if isinstance(routing_result_json, dict) and routing_result_json.get(
                        "chosen_model") and routing_result_json.get("refined_query"):
                    chosen_model = routing_result_json["chosen_model"]
                    refined_query = routing_result_json["refined_query"]
                    reasoning = routing_result_json.get("reasoning", "N/A")
                    valid_model_keys = {"vlm", "latex", "math", "code", "general"}
                    if chosen_model in valid_model_keys:
                        logger.info(
                            f"✅ {log_prefix} Router chose: '{chosen_model}'. Reason: {reasoning}. Query: '{refined_query[:50]}...'")
                        # ... (DB logging of success) ...
                        return chosen_model, refined_query, reasoning
                    else:
                        last_error = ValueError(f"Router returned invalid model key '{chosen_model}'")
                        logger.warning(f"{log_prefix} {last_error}. Full result: {routing_result_json}")
                        # ... (DB logging of warning) ...
                else:
                    last_error = ValueError("Router output not the expected JSON dict structure.")
                    logger.warning(f"{log_prefix} {last_error}. Received: {routing_result_json}")
                    # ... (DB logging of warning) ...

            except TaskInterruptedException as tie:
                raise tie
            except Exception as e:  # Catches OutputParserException if JSON is malformed, or other errors
                last_error = e
                logger.warning(f"⚠️ {log_prefix} Error during router direct JSON attempt {attempt + 1}: {e}")
                # The raw text that failed parsing would be inside 'e' if it's OutputParserException
                if isinstance(e, OutputParserException):
                    logger.warning(f"   LLM Output that failed JSON parsing: {e.llm_output}")
                # ... (DB logging of error) ...

            if attempt < DEEP_THOUGHT_RETRY_ATTEMPTS - 1:
                await asyncio.sleep(0.5 + attempt * 0.5)
            else:
                logger.error(f"❌ {log_prefix} Max retries for router. Last error: {last_error}")
                # ... (DB logging of final failure) ...
                return default_model_key, user_input, f"Router failed after retries: {last_error}"

        # Fallback if loop finishes unexpectedly (shouldn't happen)
        return default_model_key, user_input, "Router unexpected exit after retries."

    # --- generate method ---
    # app.py -> Inside AIChat class

    # --- generate (Main Async Method - Fuzzy History RAG + Direct History + Log Context + Multi-LLM Routing + VLM Preprocessing) ---
    # app.py -> Inside AIChat class

    async def direct_generate(self, db: Session, user_input: str, session_id: str,
                              vlm_description: Optional[str] = None,
                              image_b64: Optional[str] = None) -> str:
        """
        Handles the fast-path direct response generation with ELP1 priority.
        Constructs a raw ChatML prompt including RAG from URLs, conversational history (on-the-fly),
        and global self-reflection chunks (pre-indexed). Sends it for completion.
        Uses _get_rag_retriever_thread_wrapper for robust RAG context fetching and applies token truncation.
        """
        direct_req_id = f"dgen-raw_chatml-{uuid.uuid4()}"
        log_prefix = f"⚡️ {direct_req_id}|ELP1"
        logger.info(
            f"{log_prefix} Direct Generate (RAW CHATML MODE) START --> Session: {session_id}, Input: '{user_input[:50]}...', VLM Desc: {'Yes' if vlm_description else 'No'}, Image b64: {'Yes' if image_b64 else 'No'}")
        direct_start_time = time.monotonic()
        self.current_session_id = session_id

        # --- Get Fast Model ---
        fast_model = self.provider.get_model("general_fast")
        if not fast_model:
            error_msg = "Fast model 'general_fast' for direct response is not configured."
            logger.error(f"{log_prefix}: {error_msg}")
            try:
                add_interaction(db, session_id=session_id, mode="chat", input_type="log_error",
                                user_input="[Direct Gen Failed - Model Unavailable]",
                                llm_response=error_msg)
                db.commit()
            except Exception as db_log_err:
                logger.error(f"{log_prefix} Failed to log fast model unavailable error: {db_log_err}")
                if db: db.rollback()
            return f"Error: Cannot generate quick response ({error_msg})."

        # --- Prepare Content for ChatML Construction ---
        system_prompt_content_base = PROMPT_DIRECT_GENERATE_SYSTEM_CONTENT  # From config.py
        historical_turns_for_chatml: List[Dict[str, str]] = []
        rag_context_block_for_system_prompt = "No relevant RAG context found."
        session_chat_rag_ids_used_str = ""  # Populated by RAG wrapper
        final_system_prompt_content = system_prompt_content_base  # Start with base
        final_response_text = "[Direct generation failed (process error) - LLM call not reached or failed]"  # Default error

        url_retriever_obj: Optional[Any] = None
        session_hist_retriever_obj: Optional[Any] = None
        reflection_chunk_retriever_obj: Optional[Any] = None

        try:
            # --- RAG Context Fetching ---
            logger.debug(f"{log_prefix} Fetching RAG retrievers via thread wrapper (ELP1)...")
            rag_query_input = user_input
            if not rag_query_input and vlm_description:
                rag_query_input = f"Regarding the image described as: {vlm_description}"
            elif not rag_query_input and not vlm_description:
                rag_query_input = ""  # Empty query if no text/vlm

            wrapped_rag_result = await asyncio.to_thread(
                self._get_rag_retriever_thread_wrapper, db, rag_query_input, ELP1
            )

            if wrapped_rag_result.get("status") == "success":
                url_retriever_obj, session_hist_retriever_obj, reflection_chunk_retriever_obj, session_chat_rag_ids_used_str = \
                wrapped_rag_result["data"]
            elif wrapped_rag_result.get("status") == "interrupted":
                raise TaskInterruptedException(wrapped_rag_result.get("error_message", "RAG interrupted"))
            else:  # Error
                error_msg = wrapped_rag_result.get("error_message", "Unknown RAG error")
                logger.error(f"{log_prefix}: RAG retrieval failed: {error_msg}")
                final_system_prompt_content = f"{system_prompt_content_base}\n\n[System Note: Error RAG: {error_msg}]"
                # Retrievers remain None

            all_retrieved_rag_docs: List[Any] = []
            if wrapped_rag_result.get("status") == "success":
                if url_retriever_obj:
                    try:
                        docs = await asyncio.to_thread(url_retriever_obj.invoke,
                                                       rag_query_input); all_retrieved_rag_docs.extend(docs or [])
                    except Exception as e:
                        logger.warning(f"{log_prefix} URL RAG invoke error: {e}")
                if session_hist_retriever_obj:
                    try:
                        docs = await asyncio.to_thread(session_hist_retriever_obj.invoke,
                                                       rag_query_input); all_retrieved_rag_docs.extend(docs or [])
                    except Exception as e:
                        logger.warning(f"{log_prefix} Session RAG invoke error: {e}")
                if reflection_chunk_retriever_obj:
                    try:
                        docs = await asyncio.to_thread(reflection_chunk_retriever_obj.invoke,
                                                       rag_query_input); all_retrieved_rag_docs.extend(docs or [])
                    except Exception as e:
                        logger.warning(f"{log_prefix} Reflection RAG invoke error: {e}")

            # --- Token-aware Truncation of Formatted RAG Context ---
            if all_retrieved_rag_docs:
                untruncated_rag_block = self._format_docs(all_retrieved_rag_docs, "Combined RAG Context")

                # Determine current user input for token counting (including VLM)
                current_full_input_for_tokens = user_input
                if vlm_description:
                    current_full_input_for_tokens = f"[Image Description: {vlm_description.strip()}]{CHATML_NL}{CHATML_NL}User Query: {user_input.strip() or '(Query about image)'}"
                else:
                    current_full_input_for_tokens = user_input.strip()

                # Fetch direct history for accurate fixed token count
                direct_hist_interactions_tc = get_global_recent_interactions(db, limit=3)  # Sync for token count
                temp_hist_turns_tc: List[Dict[str, str]] = []
                for item_tc in direct_hist_interactions_tc:  # Renamed to avoid clash
                    r, c = (None, None)
                    if item_tc.input_type == 'text' and item_tc.user_input:
                        r, c = "user", item_tc.user_input
                    elif item_tc.llm_response and item_tc.input_type == 'llm_response':
                        r, c = "assistant", item_tc.llm_response
                    if r and c: temp_hist_turns_tc.append({"role": r, "content": c.strip()})

                base_prompt_tokens = self._count_tokens(system_prompt_content_base)
                user_input_tokens = self._count_tokens(current_full_input_for_tokens)
                history_turns_text_tc = "\n".join(t["content"] for t in temp_hist_turns_tc)
                history_turns_tokens = self._count_tokens(history_turns_text_tc)

                BUFFER_TOKENS_FOR_RESPONSE = 512  # From config or defined here
                # LLAMA_CPP_N_CTX for the 'general_fast' model. If worker uses dynamic for it, this is an estimate.
                # For direct_generate, we target LLAMA_CPP_N_CTX as defined in config.py.
                MODEL_CONTEXT_WINDOW = LLAMA_CPP_N_CTX
                fixed_parts_total_tokens = base_prompt_tokens + user_input_tokens + history_turns_tokens

                max_rag_tokens_budget = MODEL_CONTEXT_WINDOW - fixed_parts_total_tokens - BUFFER_TOKENS_FOR_RESPONSE
                if max_rag_tokens_budget < 100: max_rag_tokens_budget = max(0, max_rag_tokens_budget)

                logger.debug(
                    f"{log_prefix} Token Budget for RAG: MaxBudget={max_rag_tokens_budget} (ModelCtx={MODEL_CONTEXT_WINDOW}, FixedParts={fixed_parts_total_tokens})")

                if max_rag_tokens_budget > 0:
                    rag_context_block_for_system_prompt = self._truncate_rag_context(untruncated_rag_block,
                                                                                     max_rag_tokens_budget)
                else:
                    rag_context_block_for_system_prompt = "[RAG Context Skipped: Insufficient Token Budget]"
            else:
                rag_context_block_for_system_prompt = "No relevant RAG context found."

            if rag_context_block_for_system_prompt not in ["No relevant RAG context found.",
                                                           "[RAG Context Skipped: Insufficient Token Budget]"]:
                final_system_prompt_content += (
                    f"\n\n--- Relevant Context from History, Reflections & URLs (RAG) ---\n"
                    f"{rag_context_block_for_system_prompt}\n"
                    f"--- End Relevant Context ---"
                )

            # Actual direct history for ChatML turns (not RAG source material)
            direct_history_interactions = await asyncio.to_thread(get_global_recent_interactions, db, limit=3)
            for interaction_dh in direct_history_interactions:  # Renamed loop variable
                role_dh, content_dh = None, None
                if interaction_dh.input_type == 'text' and interaction_dh.user_input:
                    role_dh, content_dh = "user", interaction_dh.user_input
                elif interaction_dh.llm_response and interaction_dh.input_type == 'llm_response':
                    role_dh, content_dh = "assistant", interaction_dh.llm_response
                if role_dh and content_dh:
                    is_current_input_repeat = (interaction_dh == direct_history_interactions[
                        -1] and role_dh == "user" and content_dh.strip() == user_input.strip())
                    if not is_current_input_repeat:
                        historical_turns_for_chatml.append({"role": role_dh, "content": content_dh.strip()})
            if historical_turns_for_chatml: logger.debug(
                f"{log_prefix} Added {len(historical_turns_for_chatml)} direct history turns to ChatML.")

        except TaskInterruptedException as tie_context:  # From RAG wrapper or other await asyncio.to_thread calls
            logger.warning(f"🚦 {log_prefix}: Context fetching INTERRUPTED: {tie_context}")
            final_system_prompt_content = f"{system_prompt_content_base}\n\n[System Note: Context fetching interrupted. Limited info.]"
            historical_turns_for_chatml.append({"role": "system", "content": f"[RAG Interrupted: {tie_context}]"})
            raise  # Propagate to main try-except
        except Exception as context_err:  # Other errors during context prep
            logger.error(f"{log_prefix}: Error during context prep: {context_err}")
            logger.exception(f"{log_prefix} Context Prep Traceback:")
            final_system_prompt_content = f"{system_prompt_content_base}\n\n[System Note: Error preparing RAG. Base instructions only.]"
            historical_turns_for_chatml.append({"role": "system", "content": "[Error RAG Context]"})
            # Do not re-raise general errors, try to proceed

        # --- Construct final prompt ---
        current_user_turn_for_chatml = user_input  # Default
        if vlm_description:
            current_user_turn_for_chatml = f"[Image Description: {vlm_description.strip()}]{CHATML_NL}{CHATML_NL}User Query: {user_input.strip() or '(Query on image)'}"
        else:
            current_user_turn_for_chatml = user_input.strip()
        if not current_user_turn_for_chatml: current_user_turn_for_chatml = "(User provided no text)"

        raw_chatml_prompt_string = self._construct_raw_chatml_prompt(
            system_content=final_system_prompt_content,
            history_turns=historical_turns_for_chatml,
            current_turn_content=current_user_turn_for_chatml,
            current_turn_role="user",
            prompt_for_assistant_response=True
        )
        prompt_tokens_final_est = self._count_tokens(raw_chatml_prompt_string)
        logger.trace(
            f"{log_prefix}: Final Raw ChatML Prompt (len {len(raw_chatml_prompt_string)}, est. tokens {prompt_tokens_final_est}):\n==PROMPT START==\n{raw_chatml_prompt_string[:1500]}...\n==PROMPT END PREVIEW=="
        )
        if prompt_tokens_final_est > LLAMA_CPP_N_CTX:  # Compare to config's general context size
            logger.error(
                f"{log_prefix}: CRITICAL - Assembled prompt ({prompt_tokens_final_est} tokens) for 'general_fast' model may exceed its effective context window ({LLAMA_CPP_N_CTX}). Worker might fail.")

        # --- Call LLM with ELP1 ---
        try:
            logger.debug(f"{log_prefix}: Calling 'general_fast' model with raw ChatML (ELP1)...")
            raw_llm_response = await asyncio.to_thread(
                fast_model._call,  # Synchronous LlamaCppChatWrapper._call
                messages=raw_chatml_prompt_string,
                stop=[CHATML_END_TOKEN],  # Ensure stop token from config
                priority=ELP1,
                # kwargs like temperature, max_tokens (for generation) are passed from fast_model.model_kwargs
                # or can be added here if needed, e.g. max_tokens = BUFFER_FOR_RESPONSE_DIRECT
            )
            logger.info(
                f"{log_prefix}: 'general_fast' (ELP1) call complete. Raw len: {len(raw_llm_response if isinstance(raw_llm_response, str) else str(raw_llm_response))}")
            final_response_text = self._cleanup_llm_output(raw_llm_response)

        except TaskInterruptedException as tie_llm:
            logger.error(f"🚦 {log_prefix}: Direct LLM call (ELP1) INTERRUPTED: {tie_llm}")
            final_response_text = f"[Error: Direct response (ELP1) interrupted. Try again.]"
            # DB log in finally, re-raise to be caught by Flask route handler's main try-except
            raise
        except Exception as e_llm:
            logger.error(f"❌ {log_prefix}: Error during direct LLM call (ELP1): {e_llm}")
            logger.exception(f"{log_prefix} Direct LLM Call Traceback (ELP1):")
            final_response_text = f"[Error generating direct response (ELP1): {type(e_llm).__name__} - {e_llm}]"
            # DB log in finally

        # --- Final DB Log ---
        direct_duration_ms = (time.monotonic() - direct_start_time) * 1000
        logger.info(
            f"{log_prefix} Direct Generate END. Duration: {direct_duration_ms:.2f}ms. Final Response len: {len(final_response_text)}")

        try:
            interaction_log_data = {
                "session_id": session_id, "mode": "chat",
                "input_type": "image+text" if vlm_description else "text",
                "user_input": user_input, "llm_response": final_response_text,
                "execution_time_ms": direct_duration_ms,
                "image_description": vlm_description,  # Desc of user-provided image
                "classification": "direct_response_raw_chatml_elp1",
                "rag_history_ids": session_chat_rag_ids_used_str,  # From RAG wrapper
                "rag_source_url": self.vectorstore_url._source_url if hasattr(self,
                                                                              'vectorstore_url') and self.vectorstore_url and hasattr(
                    self.vectorstore_url, '_source_url') else None,
                "image_data": image_b64[:20] + "..." if image_b64 else None,  # User-provided image_b64 snippet
                "requires_deep_thought": False, "deep_thought_reason": None,
                "tot_analysis_requested": False, "tot_result": None, "tot_delivered": False,
                "emotion_context_analysis": None,  # Not run in this direct path
                "assistant_action_analysis_json": None, "assistant_action_type": None,
                "assistant_action_params": None, "assistant_action_executed": False,
                "assistant_action_result": None,
                "imagined_image_prompt": None, "imagined_image_b64": None,  # No AI imagination in direct path
                "imagined_image_vlm_description": None, "reflection_completed": False
            }
            valid_db_keys = {c.name for c in Interaction.__table__.columns}
            db_kwargs_for_log = {k: v for k, v in interaction_log_data.items() if k in valid_db_keys}
            add_interaction(db, **db_kwargs_for_log)
            db.commit()
        except Exception as log_err_final:
            logger.error(f"❌ {log_prefix}: Failed to log final direct_generate interaction: {log_err_final}")
            if db: db.rollback()

        return final_response_text

    async def _get_vector_search_file_index_context(self, query: str, priority: int = ELP0) -> str:
        """
        Performs a vector similarity search on the global file index vector store
        and formats the results. Runs embedding for the query with specified priority.
        """
        log_prefix = f"🔍 FileVecSearch|ELP{priority}|{self.current_session_id or 'NoSession'}"
        logger.debug(f"{log_prefix} Performing vector search on file index for query: '{query[:50]}...'")

        global_file_vs = get_global_file_index_vectorstore()  # Synchronous call to get the store

        if not global_file_vs:
            logger.warning(f"{log_prefix} Global file index vector store not available. Cannot perform vector search.")
            return "No vector file index available for search."

        if not self.provider.embeddings:
            logger.error(f"{log_prefix} Embeddings provider not available for vector search query.")
            return "Embeddings provider missing, cannot perform vector file search."

        if not query:
            logger.debug(f"{log_prefix} Empty query for vector search. Skipping.")
            return "No specific query provided for vector file search."

        try:
            # Chroma's similarity_search will internally use the embedding function
            # (which is our LlamaCppEmbeddingsWrapper) to embed the query.
            # We need to ensure that the LlamaCppEmbeddingsWrapper.embed_query
            # method correctly handles the 'priority' kwarg.
            # If `similarity_search` does not pass down arbitrary kwargs to the
            # embedding function, this becomes more complex.
            #
            # Current LlamaCppEmbeddingsWrapper.embed_query is designed to take priority.
            # Chroma's `similarity_search` method typically takes the query string directly.
            # Let's assume for now that Chroma calls `embed_query` on its configured
            # embedding function.
            #
            # A more explicit way if Chroma doesn't pass priority:
            # query_embedding = await asyncio.to_thread(self.provider.embeddings.embed_query, query, priority=priority)
            # search_results_docs = await asyncio.to_thread(
            #     global_file_vs.similarity_search_by_vector,
            #     embedding=query_embedding,
            #     k=RAG_FILE_INDEX_COUNT # Use the count from config
            # )
            # For simplicity and assuming embed_query in the wrapper is correctly prioritized:

            logger.debug(
                f"{log_prefix} Calling similarity_search (k={RAG_FILE_INDEX_COUNT}) with priority {priority} implicitly through embeddings object...")
            # The `priority` argument is implicitly handled by the `LlamaCppEmbeddingsWrapper`
            # when `similarity_search` calls its `embed_query` method.
            # We must ensure `LlamaCppEmbeddingsWrapper.embed_query` correctly passes priority to `_embed_texts`.

            # Run the blocking similarity_search in a thread
            # The LlamaCppEmbeddingsWrapper.embed_query will be called with the specified priority
            # when Chroma needs to embed the query.
            search_results_docs = await asyncio.to_thread(
                global_file_vs.similarity_search,
                query,
                k=RAG_FILE_INDEX_COUNT,
                # How to pass priority to the underlying embedding function of Chroma?
                # Chroma's API for similarity_search doesn't directly take a 'priority' kwarg
                # to pass to the embedding function.
                # This implies that the LlamaCppEmbeddingsWrapper *must* handle priority
                # globally or via a context, which is not ideal.
                #
                # A better approach for prioritized query embedding with Chroma:
                # 1. Embed the query explicitly with priority.
                # 2. Use similarity_search_by_vector.
            )

            # === Revised approach for explicit priority in query embedding ===
            if not hasattr(self.provider.embeddings, 'embed_query'):
                logger.error(f"{log_prefix} Embeddings object does not have 'embed_query' method.")
                return "Vector search failed: Embeddings misconfigured."

            # Explicitly embed the query with priority
            # embed_query in LlamaCppEmbeddingsWrapper should accept priority
            query_vector = await asyncio.to_thread(
                self.provider.embeddings.embed_query, query, priority=priority  # Pass priority here
            )

            if not query_vector:
                logger.error(f"{log_prefix} Failed to embed query for vector search.")
                return "Vector search failed: Could not embed query."

            # Perform search using the pre-computed vector
            search_results_docs = await asyncio.to_thread(
                global_file_vs.similarity_search_by_vector,
                embedding=query_vector,
                k=RAG_FILE_INDEX_COUNT
            )
            # === End revised approach ===

            if not search_results_docs:
                logger.debug(f"{log_prefix} No results found from vector file search for query '{query[:50]}...'")
                return "No relevant file content found via vector search for the query."

            logger.info(f"{log_prefix} Found {len(search_results_docs)} results from vector file search.")

            # Format results (similar to _format_file_index_results but from Langchain Documents)
            context_parts = []
            max_snippet_len = 300  # Characters per snippet
            max_total_chars = 2000  # Max total characters for this context block

            for i, doc in enumerate(search_results_docs):
                if not hasattr(doc, 'page_content') or not hasattr(doc, 'metadata'):
                    logger.warning(f"{log_prefix} Skipping malformed document in vector search results: {doc}")
                    continue

                content = doc.page_content
                metadata = doc.metadata

                file_path = metadata.get("source", "Unknown path")
                file_name = metadata.get("file_name",
                                         os.path.basename(file_path) if file_path != "Unknown path" else "Unknown file")
                last_mod = metadata.get("last_modified", "Unknown date")

                snippet = content[:max_snippet_len]
                if len(content) > max_snippet_len:
                    snippet += "..."

                entry = (
                    f"--- Vector File Result {i + 1} (Score: {doc.metadata.get('relevance_score', 'N/A') if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) else 'N/A'}) ---\n"  # Chroma often puts score in metadata
                    f"File: {file_name}\n"
                    f"Path Hint: ...{file_path[-70:]}\n"  # Show end of path
                    f"Modified: {last_mod}\n"
                    f"Content Snippet: {snippet}\n"
                    f"---\n"
                )
                if sum(len(p) for p in context_parts) + len(entry) > max_total_chars:
                    context_parts.append("[Vector file search context truncated due to length]...\n")
                    break
                context_parts.append(entry)

            return "".join(context_parts) if context_parts else "No relevant file content found via vector search."

        except TaskInterruptedException as tie:
            logger.warning(f"🚦 {log_prefix} Vector file search INTERRUPTED: {tie}")
            # Re-raise to be handled by the caller (e.g., asyncio.gather)
            raise
        except Exception as e:
            logger.error(f"❌ {log_prefix} Error during vector file search: {e}")
            logger.exception(f"{log_prefix} Vector File Search Traceback:")
            return f"Error performing vector file search: {type(e).__name__}"

    async def background_generate(self, db: Session, user_input: str, session_id: str = None,
                                  classification: str = "chat_simple", image_b64: Optional[str] = None,
                                  update_interaction_id: Optional[int] = None):
        """
        Handles comprehensive background processing (ELP0) for chat interactions.
        Includes multi-source RAG, action analysis/execution, multi-LLM routing,
        translation, response correction, and ToT spawning, with token-aware context management.
        If update_interaction_id is provided, this is a reflection task.
        """
        request_id = f"gen-{uuid.uuid4()}"
        is_reflection_task = update_interaction_id is not None
        log_prefix = f"🔄 REFLECT {request_id}" if is_reflection_task else f"💬 BGEN {request_id}"

        if not session_id:
            session_id = f"session_{int(time.time())}" if not is_reflection_task else f"reflection_on_{update_interaction_id}_{str(uuid.uuid4())[:8]}"
        self.current_session_id = session_id

        logger.info(
            f"{log_prefix} Async Background Generate (ELP0 Pipeline) START --> Session: {session_id}, "
            f"Initial Class: '{classification}', Input: '{user_input[:50]}...', Img: {'Yes' if image_b64 else 'N'}, "
            f"Reflection Target ID: {update_interaction_id if is_reflection_task else 'N/A'}"
        )
        request_start_time = time.monotonic()

        interaction_data = {
            "session_id": session_id, "mode": "chat", "input_type": "text",
            "user_input": user_input, "llm_response": "[Processing background task...]",
            "execution_time_ms": 0, "classification": classification, "classification_reason": None,
            "rag_history_ids": None, "rag_source_url": None, "requires_deep_thought": False,
            "deep_thought_reason": None, "tot_analysis_requested": False, "tot_result": None,
            "tot_delivered": False, "emotion_context_analysis": None, "image_description": None,
            "assistant_action_analysis_json": None, "assistant_action_type": None,
            "assistant_action_params": None, "assistant_action_executed": False,
            "assistant_action_result": None, "image_data": image_b64[:20] + "..." if image_b64 else None,
            "imagined_image_prompt": None, "imagined_image_b64": None,
            "imagined_image_vlm_description": None, "reflection_completed": False
        }
        if image_b64: interaction_data["input_type"] = "image+text"

        final_response_text = "Error: Background processing failed unexpectedly."
        saved_initial_interaction: Optional[Interaction] = None
        original_interaction_to_update_for_reflection: Optional[Interaction] = None
        interrupted_flag = False

        if not user_input and not image_b64:
            logger.warning(f"{log_prefix} Empty input for background generate.");
            return

        if is_reflection_task:
            try:
                original_interaction_to_update_for_reflection = db.query(Interaction).filter(
                    Interaction.id == update_interaction_id).first()
                if not original_interaction_to_update_for_reflection:
                    logger.error(
                        f"{log_prefix}: CRITICAL - Cannot find original interaction ID {update_interaction_id}. Aborting reflection.");
                    return
            except Exception as load_err:
                logger.error(f"{log_prefix}: Error loading original interaction {update_interaction_id}: {load_err}");
                return

        try:
            current_input_for_llm_analysis = user_input
            if image_b64:
                logger.info(f"{log_prefix} User image provided, VLM description (ELP0)...")
                vlm_model_user = self.provider.get_model("vlm")
                if vlm_model_user:
                    try:
                        user_img_uri = f"data:image/png;base64,{image_b64}"
                        user_img_content_part = {"type": "image_url", "image_url": {"url": user_img_uri}}
                        vlm_user_prompt = "Describe this image concisely, focusing on key elements relevant to a conversation."
                        vlm_user_messages = [
                            HumanMessage(content=[user_img_content_part, {"type": "text", "text": vlm_user_prompt}])]
                        vlm_user_chain = vlm_model_user | StrOutputParser()
                        vlm_user_timing_data = {"session_id": session_id, "mode": "chat"}
                        desc = await asyncio.to_thread(self._call_llm_with_timing, vlm_user_chain, vlm_user_messages,
                                                       vlm_user_timing_data, ELP0)
                        interaction_data['image_description'] = desc
                        current_input_for_llm_analysis = f"[Image Desc (User): {desc}]\n\nUser Query: {user_input or '(Query related to image)'}"
                        logger.info(f"{log_prefix} VLM desc for user image: '{str(desc)[:100]}...'")
                    except TaskInterruptedException:
                        raise
                    except Exception as e:
                        interaction_data['image_description'] = f"[VLM Error: {e}]"; logger.error(
                            f"VLM user image error: {e}")
                else:
                    interaction_data['image_description'] = "[VLM Model Unavailable]"

            pending_tot_interaction_obj = await asyncio.to_thread(get_pending_tot_result, db, session_id)
            pending_tot_context_str = "None."
            if pending_tot_interaction_obj and pending_tot_interaction_obj.tot_result:
                pending_tot_context_str = f"Pending ToT Result: {pending_tot_interaction_obj.tot_result}"

            temp_global_hist_kw = await asyncio.to_thread(get_global_recent_interactions, db, limit=5)
            temp_direct_hist_kw = self._format_direct_history(temp_global_hist_kw)
            keyword_file_search_query = await self._generate_file_search_query_async(db, current_input_for_llm_analysis,
                                                                                     temp_direct_hist_kw, session_id)

            logger.debug(f"{log_prefix} Starting concurrent context fetching (All ELP0 operations)...")
            wrapped_rag_res_bg = await asyncio.to_thread(self._get_rag_retriever_thread_wrapper, db,
                                                         current_input_for_llm_analysis, ELP0)

            url_ret_obj_bg, sess_hist_ret_obj_bg, refl_chunk_ret_obj_bg, sess_chat_rag_ids_bg = None, None, None, ""
            if wrapped_rag_res_bg.get("status") == "success":
                url_ret_obj_bg, sess_hist_ret_obj_bg, refl_chunk_ret_obj_bg, sess_chat_rag_ids_bg = wrapped_rag_res_bg[
                    "data"]
            elif wrapped_rag_res_bg.get("status") == "interrupted":
                raise TaskInterruptedException(wrapped_rag_res_bg.get("error_message"))
            else:
                raise RuntimeError(f"RAG retrieval failed: {wrapped_rag_res_bg.get('error_message')}")
            interaction_data['rag_history_ids'] = sess_chat_rag_ids_bg
            if hasattr(self, 'vectorstore_url') and self.vectorstore_url: interaction_data['rag_source_url'] = getattr(
                self.vectorstore_url, '_source_url', None)

            async def retrieve_separate_rag_docs_concurrently_bg_local():
                url_docs, session_docs, reflection_docs = [], [], []
                q_rag = current_input_for_llm_analysis
                if url_ret_obj_bg:
                    try:
                        u_docs = await asyncio.to_thread(url_ret_obj_bg.invoke, q_rag); url_docs.extend(u_docs or [])
                    except Exception as e:
                        logger.warning(f"{log_prefix} BG URL RAG invoke error: {e}")
                if sess_hist_ret_obj_bg:
                    try:
                        s_docs = await asyncio.to_thread(sess_hist_ret_obj_bg.invoke, q_rag); session_docs.extend(
                            s_docs or [])
                    except Exception as e:
                        logger.warning(f"{log_prefix} BG Session RAG invoke error: {e}")
                if refl_chunk_ret_obj_bg:
                    try:
                        r_docs = await asyncio.to_thread(refl_chunk_ret_obj_bg.invoke, q_rag); reflection_docs.extend(
                            r_docs or [])
                    except Exception as e:
                        logger.warning(f"{log_prefix} BG Reflection RAG invoke error: {e}")
                return {"url_docs_bg": url_docs, "session_hist_docs_bg": session_docs,
                        "reflection_chunk_docs_bg": reflection_docs}

            gathered_results = await asyncio.gather(
                retrieve_separate_rag_docs_concurrently_bg_local(),
                asyncio.to_thread(get_recent_interactions, db, RAG_HISTORY_COUNT * 2, session_id, "chat", True),
                asyncio.to_thread(get_global_recent_interactions, db, limit=10),
                asyncio.to_thread(self._run_emotion_analysis, db, user_input, interaction_data),
                asyncio.to_thread(search_file_index, db, keyword_file_search_query,
                                  RAG_FILE_INDEX_COUNT) if search_file_index and keyword_file_search_query else asyncio.sleep(
                    0, result=[]),
                self._get_vector_search_file_index_context(current_input_for_llm_analysis, ELP0),
                return_exceptions=True
            )

            retrieved_rag_dict, log_entries, global_hist, _, kw_file_res, vec_file_ctx_str = [None] * 6  # Default unpack
            for i, res_item in enumerate(gathered_results):
                if isinstance(res_item, TaskInterruptedException):
                    raise res_item
                elif isinstance(res_item, BaseException):
                    logger.error(f"{log_prefix} Error in BG concurrent task {i}: {res_item}")
                    if i == 0:
                        retrieved_rag_dict = {"url_docs_bg": [], "session_hist_docs_bg": [],
                                              "reflection_chunk_docs_bg": []}
                    elif i == 1:
                        log_entries = []
                    elif i == 2:
                        global_hist = []
                    elif i == 4:
                        kw_file_res = []
                    elif i == 5:
                        vec_file_ctx_str = f"[Vector File Search Error: {type(res_item).__name__}]"
                else:
                    if i == 0:
                        retrieved_rag_dict = res_item
                    elif i == 1:
                        log_entries = res_item
                    elif i == 2:
                        global_hist = res_item
                    elif i == 4:
                        kw_file_res = res_item
                    elif i == 5:
                        vec_file_ctx_str = res_item

            emotion_ctx_str = interaction_data.get('emotion_context_analysis', "N/A")
            url_docs_list = retrieved_rag_dict.get("url_docs_bg", [])
            session_hist_docs_list = retrieved_rag_dict.get("session_hist_docs_bg", [])
            reflection_docs_list = retrieved_rag_dict.get("reflection_chunk_docs_bg", [])

            url_ctx_untruncated = self._format_docs(url_docs_list, "URL Context")
            sess_refl_rag_untruncated = self._format_docs(session_hist_docs_list + reflection_docs_list,
                                                          "Session/Reflection RAG")
            kw_file_ctx_untruncated = self._format_file_index_results(kw_file_res)
            log_ctx_str_prompt = self._format_log_history(log_entries)
            direct_hist_str_prompt = self._format_direct_history(global_hist)
            hist_summary_action = self._get_history_summary(db, MEMORY_SIZE)

            # --- TOKEN BUDGETING FOR ALL LLM PROMPTS IN BACKGROUND_GENERATE ---
            # Base this on the router prompt as it's often complex and includes many context placeholders
            # Using a general LLAMA_CPP_N_CTX from config as the model's actual context window.
            # If different models chosen by router have different context windows, this becomes more complex.
            # The llama_worker's dynamic n_ctx is based on its *input prompt*, not a fixed model property visible here easily.
            MODEL_CONTEXT_WINDOW_BUDGET = LLAMA_CPP_N_CTX  # Use the configured general context size
            ROUTER_PROMPT_TOKEN_ESTIMATE = self._count_tokens(PROMPT_ROUTER)  # Estimate fixed part of router prompt
            CURRENT_INPUT_TOKEN_ESTIMATE = self._count_tokens(current_input_for_llm_analysis)
            STATIC_CONTEXT_TOKEN_ESTIMATE = self._count_tokens(log_ctx_str_prompt) + \
                                            self._count_tokens(direct_hist_str_prompt) + \
                                            self._count_tokens(pending_tot_context_str) + \
                                            self._count_tokens(emotion_ctx_str) + \
                                            self._count_tokens(interaction_data.get('imagined_image_vlm_description',
                                                                                    ''))  # Context for imagined image

            BUFFER_FOR_LLM_RESPONSE_BG = 1024  # More generous buffer for complex background tasks

            total_fixed_tokens_for_prompts = ROUTER_PROMPT_TOKEN_ESTIMATE + CURRENT_INPUT_TOKEN_ESTIMATE + STATIC_CONTEXT_TOKEN_ESTIMATE
            max_tokens_for_all_dynamic_rag_file = MODEL_CONTEXT_WINDOW_BUDGET - total_fixed_tokens_for_prompts - BUFFER_FOR_LLM_RESPONSE_BG

            if max_tokens_for_all_dynamic_rag_file < 200:
                logger.warning(
                    f"{log_prefix} Low token budget for all dynamic context ({max_tokens_for_all_dynamic_rag_file}). Will be heavily truncated.")
                max_tokens_for_all_dynamic_rag_file = max(0, max_tokens_for_all_dynamic_rag_file)

            logger.debug(
                f"{log_prefix} BG Token Budget: MaxForAllDynamic={max_tokens_for_all_dynamic_rag_file} (ModelCtx={MODEL_CONTEXT_WINDOW_BUDGET}, FixedEst={total_fixed_tokens_for_prompts})")

            # Combine all RAG/File contexts then truncate ONCE
            combined_dynamic_untruncated_for_specialist_corrector_tot = (
                f"URL Context:\n{url_ctx_untruncated}\n\n"
                f"Session/Reflection RAG:\n{sess_refl_rag_untruncated}\n\n"
                f"File Keyword Search Context:\n{kw_file_ctx_untruncated}\n\n"
                f"File Vector Search Context:\n{vec_file_ctx_str}"
            ).strip()

            if max_tokens_for_all_dynamic_rag_file > 0:
                truncated_main_dynamic_context_block = self._truncate_rag_context(
                    combined_dynamic_untruncated_for_specialist_corrector_tot,
                    max_tokens_for_all_dynamic_rag_file
                )
            else:
                truncated_main_dynamic_context_block = "[All RAG/File Context Skipped Due to Token Budget]"

            # For router prompt, which takes separate context fields, we need to truncate them individually
            # while respecting the overall dynamic budget. A simple division of budget:
            budget_per_router_src = max_tokens_for_all_dynamic_rag_file // 3 if max_tokens_for_all_dynamic_rag_file > 0 else 0
            url_ctx_for_router = self._truncate_rag_context(url_ctx_untruncated, budget_per_router_src)
            sess_refl_rag_for_router = self._truncate_rag_context(sess_refl_rag_untruncated, budget_per_router_src)
            combined_file_ctx_for_router = self._truncate_rag_context(
                (
                    f"File Keyword Context:\n{kw_file_ctx_untruncated}\n\nFile Vector Context:\n{vec_file_ctx_str}").strip(),
                budget_per_router_src
            )

            logger.trace(
                f"{log_prefix} BG Truncated Main Dynamic Context for Specialist/Corrector/ToT:\n{truncated_main_dynamic_context_block[:300]}...")

            # --- Action Analysis ---
            action_analysis_payload = {"history_summary": hist_summary_action, "log_context": log_ctx_str_prompt,
                                       "recent_direct_history": direct_hist_str_prompt,
                                       "file_index_context": combined_file_ctx_for_router}
            action_details = await asyncio.to_thread(self._analyze_assistant_action, db, current_input_for_llm_analysis,
                                                     session_id, action_analysis_payload)
            detected_action_type = "no_action"
            if action_details:
                detected_action_type = action_details.get("action_type", "no_action")
                interaction_data.update({
                    'assistant_action_analysis_json': json.dumps(action_details),
                    'assistant_action_type': detected_action_type,
                    'assistant_action_params': json.dumps(action_details.get("parameters", {}))
                })

            if not is_reflection_task:
                initial_save_data = interaction_data.copy()
                initial_save_data['llm_response'] = "[BG Pending...]"
                initial_save_data['execution_time_ms'] = (time.monotonic() - request_start_time) * 1000
                valid_keys = {c.name for c in Interaction.__table__.columns}
                db_kwargs = {k: v for k, v in initial_save_data.items() if k in valid_keys}
                saved_initial_interaction = await asyncio.to_thread(add_interaction, db, **db_kwargs)

            imagined_image_vlm_description_for_llm_context = interaction_data.get(
                'image_description')  # From user image
            if action_details and detected_action_type == "imagine":
                interaction_data['assistant_action_executed'] = True
                idea_to_viz = action_details.get("parameters", {}).get("idea_to_visualize", user_input)
                img_prompt = await self._generate_image_generation_prompt_async(db, session_id, user_input, idea_to_viz,
                                                                                sess_refl_rag_for_router,
                                                                                combined_file_ctx_for_router,
                                                                                direct_hist_str_prompt,
                                                                                url_ctx_for_router, log_ctx_str_prompt)
                interaction_data['imagined_image_prompt'] = img_prompt
                if img_prompt:
                    img_data_list, img_err = await self.provider.generate_image_async(img_prompt, None, ELP0)
                    if img_err:
                        final_response_text = f"Imagine Error: {img_err}"; interaction_data[
                            'assistant_action_result'] = final_response_text;
                    elif img_data_list and img_data_list[0].get("b64_json"):
                        interaction_data['imagined_image_b64'] = img_data_list[0]["b64_json"]
                        desc = await self._describe_generated_image_async(db, session_id, img_data_list[0]["b64_json"])
                        interaction_data['imagined_image_vlm_description'] = desc
                        imagined_image_vlm_description_for_llm_context = desc  # For THIS turn's context
                        final_response_text = f"I've imagined that: {desc or 'Generated an image.'}"
                        interaction_data['assistant_action_result'] = "Imagine success."
                    else:
                        final_response_text = "Imagine Error: No image data."; interaction_data[
                            'assistant_action_result'] = final_response_text
                else:
                    final_response_text = "Imagine Error: Prompt gen failed."; interaction_data[
                        'assistant_action_result'] = final_response_text

            elif action_details and detected_action_type != "no_action":
                target_inter_for_action = saved_initial_interaction if not is_reflection_task else original_interaction_to_update_for_reflection
                if target_inter_for_action:
                    final_response_text = await self._execute_assistant_action(db, session_id, action_details,
                                                                               target_inter_for_action)
                else:
                    final_response_text = "Error: Missing interaction context for action."
            else:  # No action, standard LLM response
                router_payload_for_llm = {
                    "pending_tot_result": pending_tot_context_str,
                    "recent_direct_history": direct_hist_str_prompt,
                    "context": url_ctx_for_router,  # Individually truncated for router
                    "history_rag": sess_refl_rag_for_router,  # Individually truncated for router
                    "file_index_context": combined_file_ctx_for_router,  # Individually truncated for router
                    "log_context": log_ctx_str_prompt,
                    "emotion_analysis": emotion_ctx_str,
                    "imagined_image_vlm_description": imagined_image_vlm_description_for_llm_context or "None."
                    # From this turn's imagine, or user's image desc
                }
                chosen_model, refined_q, route_reason = await self._route_to_specialist(db, session_id,
                                                                                        current_input_for_llm_analysis,
                                                                                        router_payload_for_llm)
                interaction_data['classification_reason'] = f"Routed to {chosen_model}: {route_reason}"

                specialist_input_trans = refined_q
                if chosen_model in ["math", "code"] and self.provider.get_model("translator"):
                    specialist_input_trans = await self._translate(refined_q, "zh")

                specialist_model_llm = self.provider.get_model(chosen_model)
                if not specialist_model_llm: raise ValueError(f"Specialist '{chosen_model}' unavailable.")

                # Use the globally truncated combined RAG/File context for specialist and corrector
                specialist_payload_final = {
                    "input": specialist_input_trans, "emotion_analysis": emotion_ctx_str,
                    "context": url_ctx_untruncated,
                    # Send the original untruncated, but it will be part of overall string for the main prompt
                    "history_rag": sess_refl_rag_untruncated,
                    "file_index_context": truncated_main_dynamic_context_block,  # MAIN TRUNCATED BLOCK
                    "log_context": log_ctx_str_prompt,
                    "recent_direct_history": direct_hist_str_prompt,
                    "pending_tot_result": pending_tot_context_str,
                    "imagined_image_vlm_description": imagined_image_vlm_description_for_llm_context or interaction_data.get(
                        'image_description', 'None.')
                }
                # Ensure self.text_prompt_template uses these keys
                spec_chain = (RunnableLambda(lambda
                                                 x: specialist_payload_final) | self.text_prompt_template | specialist_model_llm | StrOutputParser())
                timing_spec = {"session_id": session_id, "mode": "chat"}
                draft_resp = await asyncio.to_thread(self._call_llm_with_timing, spec_chain, {}, timing_spec, ELP0)

                if chosen_model in ["math", "code"] and self.provider.get_model("translator") and (
                        "zh" in specialist_input_trans or any("\u4e00" <= char <= "\u9fff" for char in draft_resp)):
                    draft_resp = await self._translate(draft_resp, "en", "zh")

                final_response_text = await self._correct_response(db, session_id, current_input_for_llm_analysis,
                                                                   specialist_payload_final,
                                                                   draft_resp)  # Corrector gets same rich context

            # --- ToT Spawning ---
            classification_for_tot = interaction_data.get('classification', 'chat_simple')
            if is_reflection_task: classification_for_tot = "chat_complex"
            should_spawn_tot_bg = (classification_for_tot == "chat_complex")
            target_inter_for_tot_flags = saved_initial_interaction if not is_reflection_task else original_interaction_to_update_for_reflection

            if target_inter_for_tot_flags:
                try:
                    # ... (Update ToT flags as before)
                    needs_commit = False
                    if target_inter_for_tot_flags.requires_deep_thought != should_spawn_tot_bg: target_inter_for_tot_flags.requires_deep_thought = should_spawn_tot_bg; needs_commit = True
                    dt_reason = interaction_data.get('classification_reason',
                                                     'ToT spawned') if should_spawn_tot_bg else None
                    if target_inter_for_tot_flags.deep_thought_reason != dt_reason: target_inter_for_tot_flags.deep_thought_reason = dt_reason; needs_commit = True
                    if target_inter_for_tot_flags.tot_analysis_requested != should_spawn_tot_bg: target_inter_for_tot_flags.tot_analysis_requested = should_spawn_tot_bg; needs_commit = True
                    if needs_commit: db.commit()
                except Exception as e:
                    logger.error(f"ToT DB flag update error: {e}"); db.rollback()

            if should_spawn_tot_bg and target_inter_for_tot_flags:
                logger.warning(f"⏳ {log_prefix} Spawning ToT for Trigger ID: {target_inter_for_tot_flags.id}...")
                imagined_ctx_for_tot = imagined_image_vlm_description_for_llm_context or interaction_data.get(
                    'image_description') or "None."
                tot_payload = {
                    "db_session_factory": SessionLocal, "input": user_input,
                    "rag_context_docs": url_docs_list,  # Pass the lists of actual Document objects
                    "history_rag_interactions": session_hist_docs_list + reflection_docs_list,
                    "log_context_str": log_ctx_str_prompt,
                    "recent_direct_history_str": direct_hist_str_prompt,
                    "file_index_context_str": truncated_main_dynamic_context_block,  # Pass the globally truncated block
                    "imagined_image_context_str": imagined_ctx_for_tot,
                    "triggering_interaction_id": target_inter_for_tot_flags.id,
                }
                asyncio.create_task(self._run_tot_in_background_wrapper_v2(**tot_payload))

            if pending_tot_interaction_obj:
                await asyncio.to_thread(mark_tot_delivered, db, pending_tot_interaction_obj.id)

        except TaskInterruptedException as tie:
            logger.warning(f"🚦 {log_prefix} Background Generate Task INTERRUPTED: {tie}")
            interrupted_flag = True;
            final_response_text = f"[Background task interrupted: {tie}]"
            interaction_data.update({'llm_response': final_response_text, 'classification': "task_failed_interrupted",
                                     'input_type': 'log_warning'})
        except Exception as e_bg_gen:
            logger.error(f"❌❌ {log_prefix} UNHANDLED exception in background_generate: {e_bg_gen}")
            logger.exception(f"{log_prefix} Background Generate Main Traceback:")
            final_response_text = f"Error during background processing: {type(e_bg_gen).__name__} - {e_bg_gen}"
            interaction_data.update({'llm_response': final_response_text[:4000], 'input_type': 'error'})
        finally:
            # --- Final DB Update ---
            final_db_data = interaction_data.copy()
            final_response_text_cleaned = self._cleanup_llm_output(final_response_text)
            final_db_data['llm_response'] = final_response_text_cleaned
            final_db_data['execution_time_ms'] = (time.monotonic() - request_start_time) * 1000
            if final_db_data.get('imagined_image_b64'):  # Truncate b64 for DB log
                b64 = final_db_data['imagined_image_b64']
                if len(b64) > 1000000: final_db_data['imagined_image_b64'] = b64[:100] + f"...[trunc_{len(b64)}]"
            try:
                if interrupted_flag:
                    if is_reflection_task and original_interaction_to_update_for_reflection:
                        original_interaction_to_update_for_reflection.reflection_completed = False
                        original_interaction_to_update_for_reflection.llm_response = (
                                                                                                 original_interaction_to_update_for_reflection.llm_response or "") + f"\n--- Reflection Interrupted ---"
                        db.commit()
                    elif not is_reflection_task and saved_initial_interaction:
                        for k, v in final_db_data.items():
                            if hasattr(saved_initial_interaction, k): setattr(saved_initial_interaction, k, v)
                        db.commit()
                    else:
                        add_interaction(db, **final_db_data); db.commit()
                else:  # Not interrupted
                    if is_reflection_task and original_interaction_to_update_for_reflection:
                        original_interaction_to_update_for_reflection.reflection_completed = True;
                        db.commit()
                        new_refl_output_data = final_db_data.copy()
                        new_refl_output_data.update({
                            'input_type': "reflection_result" if final_db_data.get(
                                'input_type') != 'error' else 'error',
                            'user_input': f"[Refl.Result for Orig.ID {update_interaction_id}]",
                            'reflection_completed': False
                        })
                        valid_keys_refl = {c.name for c in Interaction.__table__.columns}
                        db_kwargs_refl = {k: v for k, v in new_refl_output_data.items() if k in valid_keys_refl}
                        refl_rec = await asyncio.to_thread(add_interaction, db, **db_kwargs_refl)
                        if refl_rec and self.provider and self.provider.embeddings:
                            await asyncio.to_thread(index_single_reflection, refl_rec, self.provider, db, ELP0)
                    elif not is_reflection_task and saved_initial_interaction:
                        for k, v in final_db_data.items():
                            if hasattr(saved_initial_interaction, k): setattr(saved_initial_interaction, k, v)
                        db.commit()
                    elif not is_reflection_task and not saved_initial_interaction:  # Initial save failed
                        valid_keys = {c.name for c in Interaction.__table__.columns}
                        db_kwargs_final = {k: v for k, v in final_db_data.items() if k in valid_keys}
                        await asyncio.to_thread(add_interaction, db, **db_kwargs_final)
            except Exception as final_db_err:
                logger.error(f"❌ {log_prefix}: CRITICAL error during final DB save: {final_db_err}");
                db.rollback()

            final_status = 'Interrupted' if interrupted_flag else (
                'Error' if final_db_data.get('input_type') == 'error' else 'Success')
            logger.info(
                f"{log_prefix} Async Background Generate END. Status: {final_status}. Duration: {final_db_data['execution_time_ms']:.2f}ms")

    async def _run_tot_in_background_wrapper_v2(self, db_session_factory: Any, input: str,
                                                rag_context_docs: List[Any],
                                                history_rag_interactions: List[Interaction], log_context_str: str,
                                                recent_direct_history_str: str, file_index_context_str: str,
                                                triggering_interaction_id: int,
                                                imagined_image_context_str: str):  # Added imagined_image_context_str
        """Async wrapper to run synchronous ToT logic with its own DB session. V2 passes imagined context."""
        logger.info(f"BG ToT Wrapper V2: Starting for trigger ID {triggering_interaction_id}")
        db = db_session_factory()
        # Ensure interaction_data is correctly structured for _run_tree_of_thought
        bg_interaction_data = {
            'id': triggering_interaction_id,  # Placeholder, _run_tree_of_thought might not use it
            'execution_time_ms': 0,
            'session_id': self.current_session_id,  # Assuming self.current_session_id is accessible
            'mode': 'chat',
            # Add other fields if _call_llm_with_timing or other helpers inside _run_tree_of_thought need them
        }
        try:
            await asyncio.to_thread(
                self._run_tree_of_thought_v2,  # Call a V2 of ToT
                db=db,
                input=input,
                rag_context_docs=rag_context_docs,
                history_rag_interactions=history_rag_interactions,
                log_context_str=log_context_str,
                recent_direct_history_str=recent_direct_history_str,
                file_index_context_str=file_index_context_str,
                imagined_image_context_str=imagined_image_context_str,  # Pass new context
                interaction_data=bg_interaction_data,
                triggering_interaction_id=triggering_interaction_id
            )
            logger.info(f"BG ToT Wrapper V2: Finished successfully for trigger ID {triggering_interaction_id}")
        except Exception as e:
            logger.error(f"BG ToT Wrapper V2: Error running ToT for trigger ID {triggering_interaction_id}: {e}")
            logger.exception("BG ToT Wrapper V2 Traceback:")
        finally:
            if db:
                db.close()

    def _run_tree_of_thought_v2(self, db: Session, input: str,
                                rag_context_docs: List[Any],  # These are URL docs (List[Document])
                                history_rag_interactions: List[Any],  # THIS IS List[Document] from history RAG
                                log_context_str: str,
                                recent_direct_history_str: str,
                                file_index_context_str: str,
                                imagined_image_context_str: str,
                                interaction_data: Dict[str, Any],
                                triggering_interaction_id: int) -> str:
        """Runs Tree of Thoughts simulation (synchronous), includes direct history, logs, and imagined context."""
        user_input = input
        logger.warning(
            f"🌳 Running ToT V2 for input: '{user_input[:50]}...' (Trigger ID: {triggering_interaction_id})")
        # interaction_data['tot_analysis_requested'] = True # This dict is local to this call, not the main one from background_generate

        # rag_context_docs are the URL RAG results (List[Document])
        url_rag_context_str = self._format_docs(rag_context_docs, source_type="URL Document Context")

        # history_rag_interactions are the History RAG results (List[Document])
        # Use _format_docs for these, not _format_interaction_list_to_string
        history_rag_context_str = self._format_docs(history_rag_interactions,
                                                    source_type="Retrieved History/Reflection Context")  # <<< CORRECTED HERE

        tot_model = self.provider.get_model("general")
        if not tot_model:
            # ... (error handling as before) ...
            logger.error("ToT model ('general') not available for ToT V2 execution.")
            add_interaction(db, session_id=interaction_data.get("session_id"), mode="chat", input_type="log_error",
                            user_input=f"[ToT V2 Failed - Model Unavailable ID: {triggering_interaction_id}]",
                            llm_response="ToT model unavailable.")
            if triggering_interaction_id:
                trigger_interaction = db.query(Interaction).filter(Interaction.id == triggering_interaction_id).first()
                if trigger_interaction: trigger_interaction.tot_result = "Error: ToT model unavailable."; trigger_interaction.tot_delivered = False; db.commit()
            return "Error during deep analysis: Model unavailable."

        # The prompt PROMPT_TREE_OF_THOUGHTS_V2 expects 'context' (for URL RAG) and 'history_rag' (for History/Reflection RAG)
        chain = (ChatPromptTemplate.from_template(PROMPT_TREE_OF_THOUGHTS_V2) | tot_model | StrOutputParser())
        tot_result = "Error during ToT analysis (V2)."
        try:
            # Ensure the keys in the dict match the variables in PROMPT_TREE_OF_THOUGHTS_V2
            llm_input_for_tot = {
                "input": user_input,
                "context": url_rag_context_str,  # For URL docs
                "history_rag": history_rag_context_str,  # For history/reflection RAG docs
                "file_index_context": file_index_context_str,
                "log_context": log_context_str,
                "recent_direct_history": recent_direct_history_str,  # This is already a string
                "imagined_image_context": imagined_image_context_str
            }

            llm_result = self._call_llm_with_timing(
                chain,
                llm_input_for_tot,
                interaction_data,
                priority=ELP0
            )
            tot_result = llm_result
            # ... (rest of the method as before: logging, saving ToT result to DB) ...
            logger.info(f"🌳 ToT analysis V2 LLM call complete for Trigger ID: {triggering_interaction_id}.")
            if triggering_interaction_id:
                trigger_interaction = db.query(Interaction).filter(Interaction.id == triggering_interaction_id).first()
                if trigger_interaction:
                    trigger_interaction.tot_result = tot_result
                    trigger_interaction.tot_analysis_requested = True
                    trigger_interaction.tot_delivered = False
                    db.commit()
                    logger.success(
                        f"✅ Saved ToT V2 result to Interaction ID {triggering_interaction_id} (undelivered).")
                else:
                    # ... (log orphaned ToT result) ...
                    logger.error(
                        f"❌ Could not find original interaction {triggering_interaction_id} to save ToT V2 result.")
                    add_interaction(db, session_id=interaction_data.get("session_id"), mode="chat",
                                    input_type="log_warning",
                                    llm_response=f"Orphaned ToT V2 Result for input '{user_input[:50]}...': {tot_result[:200]}...")
            return tot_result
        except TaskInterruptedException as tie:
            # ... (handle interruption as before) ...
            logger.warning(f"🚦 ToT V2 for Trigger ID {triggering_interaction_id} INTERRUPTED: {tie}")
            if triggering_interaction_id:
                trigger_interaction = db.query(Interaction).filter(Interaction.id == triggering_interaction_id).first()
                if trigger_interaction:
                    if trigger_interaction.tot_result is None or "Error" in trigger_interaction.tot_result:
                        trigger_interaction.tot_result = "[ToT Analysis Interrupted]"
                        trigger_interaction.tot_delivered = False
                    db.commit()
            raise tie
        except Exception as e:
            # ... (handle other exceptions as before) ...
            err_msg = f"Error during ToT V2 generation (Trigger ID: {triggering_interaction_id}): {e}"
            logger.error(f"❌ {err_msg}")
            add_interaction(db, session_id=interaction_data.get("session_id"), mode="chat", input_type="log_error",
                            llm_response=err_msg)
            if triggering_interaction_id:
                trigger_interaction = db.query(Interaction).filter(Interaction.id == triggering_interaction_id).first()
                if trigger_interaction:
                    trigger_interaction.tot_result = err_msg
                    trigger_interaction.tot_delivered = False
                    db.commit()
            return "Error during deep analysis (V2)."

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
        classification: str,  # Note: direct_generate might not use this for its own logic.
        image_b64: Optional[str],  # <<< Added image_b64 as parameter
        model_name: str = "Amaryllis-Adelaide-LegacyMoEArch-IdioticRecursiveLearner-FlaskStream"
):
    """
    Generator for Flask: Runs ai_chat.direct_generate in a background thread for ELP1 streaming.
    Streams logs live via a queue, handles errors and cleanup, and yields
    Server-Sent Events (SSE) formatted chunks.
    """
    resp_id = f"chatcmpl-{uuid.uuid4()}"
    timestamp = int(time.time())
    logger.debug(
        f"FLASK_STREAM_LIVE {resp_id}: Starting generation for session {session_id}, input: '{user_input[:50]}...'")

    message_queue = queue.Queue()
    background_thread: Optional[threading.Thread] = None
    final_result_data = {
        "text": "Error: Generation failed to return result from background thread.",
        "finish_reason": "error",
        "error": None  # Store actual exception object if one occurs
    }
    sink_id_holder = [None]  # Use a list to pass sink_id by reference to inner func

    def yield_chunk(delta_content: Optional[str] = None, role: Optional[str] = None,
                    finish_reason: Optional[str] = None):
        delta = {}
        if role: delta["role"] = role
        if delta_content is not None: delta["content"] = delta_content
        chunk_payload = {
            "id": resp_id, "object": "chat.completion.chunk", "created": timestamp,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
        }
        return f"data: {json.dumps(chunk_payload)}\n\n"

    def run_async_generate_in_thread(
            q: queue.Queue,
            sess_id: str,
            u_input: str,
            classi_param_ignored: str,  # Classification is not directly used by direct_generate
            img_b64_param_for_thread: Optional[str]  # <<< This is the new parameter
    ):
        """
        Target function for the background thread.
        Runs ai_chat.direct_generate, handles logging sink, and puts results/sentinel on queue.
        """
        nonlocal sink_id_holder  # To modify sink_id in the outer scope
        db_session: Optional[Session] = None
        temp_loop: Optional[asyncio.AbstractEventLoop] = None
        log_session_id = f"{sess_id}-{threading.get_ident()}"

        # Default outcomes for this thread
        thread_final_text_val = "Error: Processing failed within background thread (initial)."
        thread_final_reason_val = "error"
        thread_final_error_obj = None

        try:
            try:
                temp_loop = asyncio.get_event_loop()
                if temp_loop.is_running():
                    logger.warning(
                        f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Event loop already running in this thread.")
            except RuntimeError:
                logger.debug(
                    f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Creating new event loop for this thread.")
                temp_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(temp_loop)

            def log_sink(message):
                record = message.record
                bound_req_session_id = record.get("extra", {}).get("request_session_id")
                if bound_req_session_id == log_session_id:
                    log_entry = f"[{record['time'].strftime('%H:%M:%S.%f')[:-3]} {record['level'].name}] {record['message']}"
                    try:
                        q.put_nowait(("LOG", log_entry))
                    except queue.Full:
                        pass
                    except Exception as e_log_put:
                        print(f"ERROR in log_sink putting LOG to queue: {e_log_put}", file=sys.stderr)

            try:
                # Ensure ai_chat is not None before proceeding
                if ai_chat is None:
                    raise RuntimeError("Global ai_chat instance is not initialized.")

                # LOG_SINK_LEVEL and LOG_SINK_FORMAT should be available from config or defined
                sink_id_holder[0] = logger.add(log_sink, level=LOG_SINK_LEVEL, format=LOG_SINK_FORMAT,
                                               filter=lambda record: record["extra"].get(
                                                   "request_session_id") == log_session_id, enqueue=False)
                logger.debug(
                    f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Log sink {sink_id_holder[0]} added.")
            except Exception as sink_add_err:
                logger.error(
                    f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): CRITICAL - Failed to add Loguru sink: {sink_add_err}")
                thread_final_error_obj = sink_add_err
                thread_final_text_val = f"Error setting up internal logging: {sink_add_err}"
                raise sink_add_err  # Propagate to outer try-except in this function

            async def run_generate_with_logging_inner():
                nonlocal db_session, thread_final_text_val, thread_final_reason_val, thread_final_error_obj
                try:
                    db_session = SessionLocal()  # New DB session for this async task within the thread
                    if not db_session: raise RuntimeError("Failed to create DB session for direct_generate.")

                    with logger.contextualize(request_session_id=log_session_id):  # For logs within direct_generate
                        logger.info(f"Async direct_generate task starting for streaming (ELP1)...")
                        # ai_chat should be the global instance initialized in app.py
                        result_text = await ai_chat.direct_generate(
                            db=db_session,
                            user_input=u_input,
                            session_id=sess_id,
                            vlm_description=None,  # Assuming no VLM desc for this streaming ELP1 path for simplicity,
                            # if user sends image, it would be pre-processed before calling stream generator.
                            image_b64=img_b64_param_for_thread  # Pass the image_b64
                        )
                        thread_final_text_val = result_text if result_text is not None else "Error: Generation returned None."

                        if "interrupted" in thread_final_text_val.lower() or isinstance(thread_final_error_obj,
                                                                                        TaskInterruptedException):
                            thread_final_reason_val = "error"  # Or a specific "interrupted" status
                            logger.warning(f"Async direct_generate task INTERRUPTED or returned interruption message.")
                        elif "internal error" in thread_final_text_val.lower() or (
                                thread_final_text_val.lower().startswith(
                                        "error:") and "encountered a system issue" not in thread_final_text_val.lower()):
                            thread_final_reason_val = "error"
                            logger.warning(f"Async direct_generate task completed with internal error indication.")
                        else:
                            thread_final_reason_val = "stop"
                            logger.info(f"Async direct_generate task completed successfully.")

                except TaskInterruptedException as tie_direct_inner:
                    with logger.contextualize(request_session_id=log_session_id):
                        logger.error(
                            f"Async direct_generate task INTERRUPTED by ELP1 TaskInterruptedException: {tie_direct_inner}")
                    thread_final_error_obj = tie_direct_inner
                    thread_final_text_val = f"[Critical Error: ELP1 processing was interrupted by another high-priority task: {tie_direct_inner}]"
                    thread_final_reason_val = "error"
                except Exception as e_direct_inner:
                    with logger.contextualize(request_session_id=log_session_id):
                        logger.error(f"Async direct_generate task EXCEPTION: {e_direct_inner}")
                        logger.exception("Async Direct Generate (Inner) Traceback:")
                    thread_final_error_obj = e_direct_inner
                    thread_final_text_val = f"[Error during direct generation for streaming: {type(e_direct_inner).__name__} - {e_direct_inner}]"
                    thread_final_reason_val = "error"
                finally:
                    if db_session:
                        try:
                            db_session.close(); logger.debug(
                                f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): DB session closed for direct_generate.")
                        except Exception as ce:
                            logger.error(f"Error closing DB session for direct_generate: {ce}")

            temp_loop.run_until_complete(run_generate_with_logging_inner())

        except Exception as outer_thread_err:
            logger.error(
                f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Error in outer background thread function: {outer_thread_err}")
            if thread_final_error_obj is None:  # Only set if not already set by inner errors
                thread_final_error_obj = outer_thread_err
                thread_final_text_val = f"Background thread execution error (outer): {outer_thread_err}"
                thread_final_reason_val = "error"
        finally:
            logger.debug(
                f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): FINALLY block in run_async_generate_in_thread. Text: '{thread_final_text_val[:50]}', Reason: {thread_final_reason_val}")
            try:
                q.put(("RESULT", (thread_final_text_val, thread_final_reason_val, thread_final_error_obj)))
                logger.debug(f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Put RESULT on queue.")
            except Exception as put_result_err:
                logger.error(
                    f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): CRITICAL - FAILED to put RESULT on queue: {put_result_err}")
            try:
                q.put(GENERATION_DONE_SENTINEL)
                logger.debug(f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Put DONE sentinel on queue.")
            except Exception as put_done_err:
                logger.error(
                    f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): CRITICAL - FAILED to put DONE sentinel: {put_done_err}")

            current_sink_id = sink_id_holder[0]
            if current_sink_id is not None:
                try:
                    logger.remove(current_sink_id); logger.debug(
                        f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Log sink {current_sink_id} removed.")
                except Exception as remove_err:
                    logger.error(f"Failed remove log sink {current_sink_id}: {remove_err}")
            logger.info(
                f"FLASK_STREAM_LIVE {resp_id} (Thread {log_session_id}): Background thread function fully finished.")

    # --- Main Generator Logic (Runs in Flask Request Thread) ---
    try:
        logger.debug(f"FLASK_STREAM_LIVE {resp_id}: Starting background thread for ELP1 streaming logic...")
        background_thread = threading.Thread(
            target=run_async_generate_in_thread,
            args=(
                message_queue,
                session_id,
                user_input,
                classification,  # Pass along
                image_b64  # <<< Pass the image_b64 received by the generator
            ),
            daemon=True
        )
        background_thread.start()
        logger.debug(f"FLASK_STREAM_LIVE {resp_id}: Background thread started (ID: {background_thread.ident}).")

        yield yield_chunk(role="assistant", delta_content="<think>\n")
        time.sleep(0.01)  # Minimal sleep
        yield yield_chunk(delta_content="Starting live processing...\n---\n")
        time.sleep(0.01)

        logs_streamed_count = 0
        processing_complete = False
        result_received = False

        while not processing_complete:
            try:
                queue_item = message_queue.get(timeout=LOG_QUEUE_TIMEOUT)
                if queue_item is GENERATION_DONE_SENTINEL:
                    logger.debug(f"FLASK_STREAM_LIVE {resp_id}: Received DONE sentinel.")
                    processing_complete = True;
                    continue
                elif isinstance(queue_item, tuple) and len(queue_item) == 2:
                    message_type, message_data = queue_item
                    if message_type == "LOG":
                        yield yield_chunk(delta_content=message_data + "\n")
                        logs_streamed_count += 1
                    elif message_type == "RESULT":
                        final_result_data["text"], final_result_data["finish_reason"], final_result_data[
                            "error"] = message_data
                        result_received = True
                        logger.debug(
                            f"FLASK_STREAM_LIVE {resp_id}: Received RESULT from queue. Reason: {final_result_data['finish_reason']}")
                    else:
                        logger.warning(f"Unexpected message type from queue: {message_type}")
                else:
                    logger.error(f"Unexpected item structure from queue: {type(queue_item)}")
            except queue.Empty:
                if not processing_complete and not background_thread.is_alive():
                    logger.error(
                        f"FLASK_STREAM_LIVE {resp_id}: Background thread died unexpectedly before DONE sentinel.")
                    if not result_received:
                        final_result_data["error"] = RuntimeError("Background thread died before sending result.")
                        final_result_data["finish_reason"] = "error"
                        final_result_data["text"] = "[Critical Error: Background processing failed prematurely]"
                    else:  # Result received, but not DONE
                        final_result_data["error"] = RuntimeError(
                            "Background thread died after result, before completion signal.")
                        final_result_data["finish_reason"] = "error"
                    processing_complete = True
            except Exception as q_err:
                logger.error(f"FLASK_STREAM_LIVE {resp_id}: Error getting from queue: {q_err}")
                if final_result_data["error"] is None:
                    final_result_data["error"] = q_err;
                    final_result_data["finish_reason"] = "error"
                processing_complete = True

        logger.debug(
            f"FLASK_STREAM_LIVE {resp_id}: Exited queue loop. Logs: {logs_streamed_count}. Result recv: {result_received}")
        yield yield_chunk(delta_content="\n---\nLog stream complete.\n</think>\n\n")

        final_text_to_stream = final_result_data["text"]
        final_reason_to_send = final_result_data["finish_reason"]
        if final_result_data["error"] is not None:  # If any error was captured
            final_reason_to_send = "error"
            logger.warning(f"FLASK_STREAM_LIVE {resp_id}: Final result error: {final_result_data['error']}")

        cleaned_final_text_for_stream = final_text_to_stream
        if result_received and isinstance(final_text_to_stream, str):
            if ai_chat:  # Check if ai_chat instance exists
                try:
                    cleaned_final_text_for_stream = ai_chat._cleanup_llm_output(final_text_to_stream)
                except Exception as e_clean:
                    logger.error(f"Streamer cleanup error: {e_clean}")
            else:
                logger.error(f"ai_chat instance not found for final cleanup in streamer.")
        elif not result_received:
            logger.error(f"FLASK_STREAM_LIVE {resp_id}: No valid result received. Streaming default error text.")

        if cleaned_final_text_for_stream:
            logger.info(
                f"FLASK_STREAM_LIVE {resp_id}: Streaming final content ({len(cleaned_final_text_for_stream)} chars). Finish: {final_reason_to_send}")
            # Streaming character by character or small chunks for live effect
            # This part can be adjusted for desired streaming speed/behavior
            words = cleaned_final_text_for_stream.split(' ')
            for word_idx, word in enumerate(words):
                delta_to_send = word + (' ' if word_idx < len(words) - 1 else '')
                yield yield_chunk(delta_content=delta_to_send)
                # Simulate token speed - adjust sleep time as needed
                # Very short sleep for responsiveness, can be removed if causing too much overhead
                time.sleep(0.001)  # Use await asyncio.sleep for async generator
        else:
            logger.warning(
                f"FLASK_STREAM_LIVE {resp_id}: Final cleaned response text is empty. Finish: {final_reason_to_send}")
            if final_reason_to_send != "error": final_reason_to_send = "stop"  # Empty but not error means stop

        yield yield_chunk(finish_reason=final_reason_to_send)
        yield "data: [DONE]\n\n"
        logger.debug(f"FLASK_STREAM_LIVE {resp_id}: Finished streaming response.")

    except GeneratorExit:
        logger.warning(f"FLASK_STREAM_LIVE {resp_id}: Generator exited (client disconnected).")
    except Exception as e_gen_main:
        logger.error(f"FLASK_STREAM_LIVE {resp_id}: Unhandled error in streaming generator main: {e_gen_main}")
        logger.exception("Streaming Orchestration Main Traceback:")
        try:
            err_delta = {"content": f"\n\n[STREAMING ORCHESTRATION ERROR: {e_gen_main}]"}
            err_chunk_payload = {"id": resp_id, "object": "chat.completion.chunk", "created": timestamp,
                                 "model": model_name,
                                 "choices": [{"index": 0, "delta": err_delta, "finish_reason": "error"}]}
            yield f"data: {json.dumps(err_chunk_payload)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e_final_err:
            logger.error(f"Failed yield final error chunk: {e_final_err}")
    finally:
        if background_thread and background_thread.is_alive():
            logger.warning(
                f"FLASK_STREAM_LIVE {resp_id}: Generator finished, but BG thread {background_thread.ident} might still be daemonized.")
        logger.debug(f"FLASK_STREAM_LIVE {resp_id}: Generator function fully finished.")


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


def _execute_audio_worker_with_priority(
    worker_command: list[str],
    request_data: Dict[str, Any],
    priority: int, # ELP0 or ELP1
    worker_cwd: str,
    timeout: int = 120 # Timeout for the worker process
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Executes the audio_worker.py script with specified priority,
    sends request_data via stdin, and returns parsed JSON response or error.

    Args:
        worker_command: List representing the command and its arguments.
        request_data: Dictionary to be sent as JSON to the worker.
        priority: ELP0 or ELP1 for acquiring the shared resource lock.
        worker_cwd: The working directory for the worker script.
        timeout: Timeout for the worker process in seconds.

    Returns:
        A tuple: (parsed_json_response, error_message_string).
                 One of them will be None.
    """
    # --- Access the global priority lock (from ai_provider or a global var) ---
    # This assumes ai_provider and its lock are initialized and accessible.
    # If not, this part needs to be adapted.
    shared_priority_lock: Optional[PriorityQuotaLock] = getattr(ai_provider, '_priority_quota_lock', None)
    # ---

    request_id = request_data.get("request_id", "audio-worker-unknown") # Get request_id if passed
    log_prefix = f"AudioExec|ELP{priority}|{request_id}"
    logger.debug(f"{log_prefix}: Attempting to execute audio worker.")

    if not shared_priority_lock:
        logger.error(f"{log_prefix}: Shared PriorityQuotaLock not available/initialized! Cannot run audio worker with priority.")
        return None, "Shared resource lock not available."

    lock_acquired = False
    worker_process = None
    start_lock_wait = time.monotonic()
    logger.debug(f"{log_prefix}: Acquiring shared resource lock (Priority: ELP{priority})...")

    # Acquire the lock. Audio worker doesn't typically have a process to register
    # for interruption in the same way as the llama_worker (it's shorter lived per request).
    # If audio worker *did* have long-running internal processes that ELP1 should kill,
    # then set_holder_process would be needed for ELP0 audio tasks.
    # For TTS, we assume the worker itself is the unit to be managed by communicate timeout.
    lock_acquired = shared_priority_lock.acquire(priority=priority, timeout=None) # No specific timeout for lock acquire itself
    lock_wait_duration = time.monotonic() - start_lock_wait

    if lock_acquired:
        logger.info(f"{log_prefix}: Lock acquired (waited {lock_wait_duration:.2f}s). Starting audio worker.")
        try:
            start_time = time.monotonic()
            worker_process = subprocess.Popen(
                worker_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=worker_cwd
            )

            # If this ELP0 audio task could be long and needs to be interruptible by ELP1 LLM calls:
            # if priority == ELP0:
            # shared_priority_lock.set_holder_process(worker_process) # Then lock needs to know this proc

            input_json = json.dumps(request_data)
            logger.debug(f"{log_prefix}: Sending input JSON (len={len(input_json)}) to audio worker stdin...")
            stdout_data, stderr_data = "", ""

            try:
                stdout_data, stderr_data = worker_process.communicate(input=input_json, timeout=timeout)
                logger.debug(f"{log_prefix}: Audio worker communicate() finished.")
            except subprocess.TimeoutExpired:
                logger.error(f"{log_prefix}: Audio worker process timed out after {timeout}s.")
                worker_process.kill()
                stdout_data, stderr_data = worker_process.communicate() # Try to get final output
                logger.error(f"{log_prefix}: Worker timed out. Stderr: {stderr_data.strip()}")
                return None, "Audio worker process timed out."
            except BrokenPipeError: # This could happen if ELP0 audio worker was interrupted by ELP1 LLM
                logger.warning(f"{log_prefix}: Broken pipe with audio worker. Likely interrupted.")
                worker_process.wait(timeout=0.5) # Ensure state update
                return None, "Audio worker task interrupted by higher priority request."
            except Exception as comm_err:
                 logger.error(f"{log_prefix}: Error communicating with audio worker: {comm_err}")
                 if worker_process and worker_process.poll() is None:
                      try: worker_process.kill(); worker_process.communicate()
                      except: pass
                 return None, f"Communication error with audio worker: {comm_err}"

            exit_code = worker_process.returncode
            duration = time.monotonic() - start_time
            logger.info(f"{log_prefix}: Audio worker finished. Exit Code: {exit_code}, Duration: {duration:.2f}s")

            if stderr_data:
                log_level = "ERROR" if exit_code != 0 else "DEBUG"
                logger.log(log_level, f"{log_prefix}: Audio Worker STDERR:\n-------\n{stderr_data.strip()}\n-------")

            if exit_code == 0:
                if not stdout_data:
                    logger.error(f"{log_prefix}: Audio worker exited cleanly but no stdout.")
                    return None, "Audio worker produced no output."
                try:
                    parsed_json = json.loads(stdout_data)
                    logger.debug(f"{log_prefix}: Parsed audio worker JSON response successfully.")
                    # Check for internal error reported by the worker
                    if isinstance(parsed_json, dict) and "error" in parsed_json:
                        logger.error(f"{log_prefix}: Audio worker reported internal error: {parsed_json['error']}")
                        return None, f"Audio worker error: {parsed_json['error']}"
                    return parsed_json, None # Success
                except json.JSONDecodeError as json_err:
                    logger.error(f"{log_prefix}: Failed to decode audio worker stdout JSON: {json_err}")
                    logger.error(f"{log_prefix}: Raw stdout from worker:\n{stdout_data[:1000]}...")
                    return None, f"Failed to decode audio worker response: {json_err}"
            else:
                err_msg = f"Audio worker process failed (exit code {exit_code})."
                logger.error(f"{log_prefix}: {err_msg}")
                return None, err_msg

        except Exception as e:
            logger.error(f"{log_prefix}: Unexpected error managing audio worker: {e}")
            logger.exception(f"{log_prefix} Audio Worker Management Traceback:")
            if worker_process and worker_process.poll() is None:
                try: worker_process.kill(); worker_process.communicate()
                except: pass
            return None, f"Error managing audio worker: {e}"
        finally:
            logger.info(f"{log_prefix}: Releasing shared resource lock.")
            shared_priority_lock.release()
    else:
        logger.error(f"{log_prefix}: FAILED to acquire shared resource lock for audio worker.")
        return None, "Failed to acquire execution lock for audio worker."

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
        request_data = request.get_json()
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
    1. Calls `ai_chat.direct_generate` (via streaming generator or direct call)
       to get a fast initial (ELP1) response.
    2. Formats and returns/streams this initial response.
    3. Concurrently launches `ai_chat.background_generate` in a separate thread
       to perform deeper analysis (ELP0) without blocking the initial response.
    """
    start_req_time_main_handler = time.monotonic()  # Renamed to avoid conflict
    request_id = f"req-chat-{uuid.uuid4()}"
    logger.info(f"🚀 Flask OpenAI/Ollama Chat Request ID: {request_id} (Dual Generate Logic)")

    db: Session = g.db  # Use request-bound session from Flask's g
    response_payload_str: str = ""  # Renamed to avoid conflict
    status_code_val: int = 500
    resp_obj: Optional[Response] = None  # Renamed to avoid conflict
    session_id_for_logs: str = f"openai_req_{request_id}_unassigned"  # Renamed
    raw_request_data_dict: Optional[Dict] = None  # Renamed
    request_data_log_snippet: str = "No request data processed"  # Renamed

    # Variables to be extracted from request
    user_input_from_req: str = ""
    image_b64_from_req: Optional[str] = None
    stream_requested_by_client: bool = False

    # This will be passed to background_generate; direct_generate might use a simpler classification
    classification_for_background = "chat_simple"

    try:
        # --- 1. Get and Validate Request Data ---
        try:
            raw_request_data_dict = request.get_json()
            if not raw_request_data_dict:
                raise ValueError("Empty JSON payload received.")
            try:
                request_data_log_snippet = json.dumps(raw_request_data_dict)[:1000]
            except:
                request_data_log_snippet = str(raw_request_data_dict)[:1000]
        except Exception as json_err:
            logger.warning(f"{request_id}: Failed to get/parse JSON body: {json_err}")
            try:
                request_data_log_snippet = request.get_data(as_text=True)[:1000]
            except:
                request_data_log_snippet = "Could not read request body"

            resp_data_err, status_code_val = _create_openai_error_response(
                f"Request body is missing or invalid JSON: {json_err}",
                err_type="invalid_request_error", status_code=400)
            resp_obj = Response(json.dumps(resp_data_err), status=status_code_val, mimetype='application/json')
            return resp_obj  # Early return

        # --- 2. Extract Parameters ---
        messages_from_req = raw_request_data_dict.get("messages", [])
        stream_requested_by_client = raw_request_data_dict.get("stream", False)
        model_requested_by_client = raw_request_data_dict.get("model")
        session_id_for_logs = raw_request_data_dict.get("session_id", f"openai_req_{request_id}")
        if ai_chat: ai_chat.current_session_id = session_id_for_logs  # Set session for AIChat instance

        logger.debug(
            f"{request_id}: Request parsed - SessionID={session_id_for_logs}, Stream: {stream_requested_by_client}, ModelReq: {model_requested_by_client}")

        if not messages_from_req or not isinstance(messages_from_req, list):
            raise ValueError("'messages' is required and must be a list.")

        # --- 4. Parse Last User Message for Input and Image ---
        last_user_msg_obj = None
        for msg_item in reversed(messages_from_req):  # Renamed loop var
            if isinstance(msg_item, dict) and msg_item.get("role") == "user":
                last_user_msg_obj = msg_item;
                break

        if not last_user_msg_obj: raise ValueError("No message with role 'user' found.")

        content_from_user_msg = last_user_msg_obj.get("content")
        if isinstance(content_from_user_msg, str):
            user_input_from_req = content_from_user_msg
        elif isinstance(content_from_user_msg, list):
            for item_part in content_from_user_msg:  # Renamed loop var
                if isinstance(item_part, dict):
                    item_type = item_part.get("type")
                    if item_type == "text":
                        user_input_from_req += item_part.get("text", "")
                    elif item_type == "image_url":
                        img_url = item_part.get("image_url", {}).get("url", "")
                        if img_url.startswith("data:image"):
                            try:
                                _, potential_b64 = img_url.split(",", 1)
                                if len(potential_b64) % 4 != 0 or not re.match(r'^[A-Za-z0-9+/=]+$', potential_b64):
                                    raise ValueError("Invalid base64 characters or padding")
                                base64.b64decode(potential_b64, validate=True)
                                image_b64_from_req = potential_b64
                            except Exception as img_err:
                                raise ValueError(f"Invalid image data: {img_err}")
                        else:
                            raise ValueError("Unsupported image_url format. Only data URIs allowed.")
        else:
            raise ValueError("Invalid user message 'content' type.")

        if not user_input_from_req and not image_b64_from_req:
            raise ValueError("No text or image content provided in user message.")

        # --- Call AIProvider to classify complexity for background task planning ---
        # This runs synchronously here to determine if background_generate needs "chat_complex"
        # Note: direct_generate (for ELP1 streaming) might do its own simpler/no classification.
        logger.info(f"{request_id}: Classifying input for background task planning (ELP0 context)...")
        classification_data_for_bg = {"session_id": session_id_for_logs, "mode": "chat", "input_type": "classification"}
        # _classify_input_complexity is synchronous and handles its own ELP0 for LLM calls
        classification_for_background = ai_chat._classify_input_complexity(db, user_input_from_req,
                                                                           classification_data_for_bg)
        logger.info(
            f"{request_id}: Input classified for background task as: '{classification_for_background}'. Reason: {classification_data_for_bg.get('classification_reason', 'N/A')}")

        # --- 5. Call DIRECT Generate Logic (ELP1) ---
        # This part handles the immediate response to the client (either streaming or non-streaming)
        direct_response_text_val = ""
        vlm_desc_for_bg_and_direct: Optional[str] = None  # VLM desc of user's image
        status_code_val = 200  # Assume success for direct path unless error

        logger.info(f"{request_id}: Preparing for AIChat.direct_generate (ELP1 path)...")
        try:
            if image_b64_from_req:  # If user sent an image
                logger.info(f"{request_id}: Preprocessing user-provided image for direct_generate (ELP1 context)...")
                # ai_chat.process_image is synchronous but might call VLM (ELP0)
                # For ELP1 path, VLM desc should ideally be fast or direct_generate robust to its absence.
                # Let's assume process_image is quick enough or handles VLM with ELP0 gracefully.
                vlm_desc_for_bg_and_direct, _ = ai_chat.process_image(db, image_b64_from_req, session_id_for_logs)
                if vlm_desc_for_bg_and_direct and "Error:" in vlm_desc_for_bg_and_direct:
                    logger.error(
                        f"{request_id}: VLM preprocessing for direct path failed: {vlm_desc_for_bg_and_direct}")
                    # Don't set status_code_val to 500 here yet, let direct_generate try text-only
                    # And log this problem to the DB for the user-provided image.
                    add_interaction(db, session_id=session_id_for_logs, mode="chat", input_type="log_error",
                                    user_input="[VLM Preprocessing Error for User Image - ELP1 Path]",
                                    llm_response=vlm_desc_for_bg_and_direct)

            # Call direct_generate (which is async, so run it if not streaming, or it's called by streamer)
            # The streaming path calls direct_generate inside its thread via asyncio.run
            # The non-streaming path calls it here via asyncio.run
            if not stream_requested_by_client:
                logger.info(f"{request_id}: Non-streaming path. Calling direct_generate now...")
                direct_response_text_val = asyncio.run(
                    ai_chat.direct_generate(
                        db,
                        user_input_from_req,
                        session_id_for_logs,
                        vlm_description=vlm_desc_for_bg_and_direct,  # Pass VLM desc of user image
                        image_b64=image_b64_from_req  # Pass user image b64 for logging within direct_generate
                    )
                )
                if "interrupted" in direct_response_text_val.lower() or \
                        (
                                "Error:" in direct_response_text_val and "interrupted" not in direct_response_text_val.lower()) or \
                        "internal error" in direct_response_text_val.lower():  # Check for error strings
                    status_code_val = 503 if "interrupted" in direct_response_text_val.lower() else 500
                logger.info(f"{request_id}: Non-streaming direct_generate completed. Status: {status_code_val}")

        except TaskInterruptedException as tie_direct:
            logger.error(f"🚦 {request_id}: Direct generation path (ELP1) INTERRUPTED: {tie_direct}")
            direct_response_text_val = f"[Critical Error: ELP1 Processing Interrupted by Higher Priority Task: {tie_direct}]"
            status_code_val = 503  # Service Unavailable due to interruption
        except Exception as direct_gen_err:
            logger.error(f"{request_id}: Error during direct_generate call/setup: {direct_gen_err}")
            logger.exception(f"{request_id} Traceback for direct_generate error:")
            direct_response_text_val = f"Error during initial response generation: {direct_gen_err}"
            status_code_val = 500

        # --- 6. LAUNCH BACKGROUND Generate Logic (ELP0) in a separate thread ---
        logger.info(f"{request_id}: Preparing to launch background_generate task (ELP0) in new thread...")

        # Define the target function for the background thread
        # This function needs to create its own asyncio event loop and DB session
        def run_background_task_with_new_loop(
                user_input_bg: str, session_id_bg: str, classification_bg: str,
                image_b64_bg: Optional[str], vlm_desc_user_img_bg: Optional[str]):
            bg_log_prefix_thread = f"[BG Task {request_id}]"
            bg_db_session: Optional[Session] = None
            loop = None
            try:
                logger.info(f"{bg_log_prefix_thread} Background thread started for session {session_id_bg}.")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                bg_db_session = SessionLocal()  # New DB session for this thread
                if not bg_db_session:
                    logger.error(f"{bg_log_prefix_thread} Failed to create DB session for background task.")
                    return

                # The actual background_generate call
                loop.run_until_complete(
                    ai_chat.background_generate(
                        db=bg_db_session,
                        user_input=user_input_bg,  # Use the original user input
                        session_id=session_id_bg,
                        classification=classification_bg,  # Use classification determined earlier
                        image_b64=image_b64_bg,  # Pass user's original image b64 if any
                        # vlm_description for user's image is passed implicitly via current_input_for_llm_analysis within background_generate if image_b64_bg is present
                    )
                )
                logger.info(f"{bg_log_prefix_thread} Background generate task completed.")
            except Exception as task_err:
                logger.error(f"{bg_log_prefix_thread} Error during background task: {task_err}")
                logger.exception(f"{bg_log_prefix_thread} Background Task Execution Traceback:")
                if bg_db_session:
                    try:
                        add_interaction(bg_db_session, session_id=session_id_bg, mode="chat", input_type="log_error",
                                        user_input=f"Background Task Error ({request_id})",
                                        llm_response=f"Error: {task_err}"[:2000])
                        bg_db_session.commit()
                    except Exception as db_log_err_bg:
                        logger.error(f"{bg_log_prefix_thread} Failed to log BG error: {db_log_err_bg}")
            finally:
                if bg_db_session:
                    try:
                        bg_db_session.close()
                    except:
                        pass  # Ignore close errors
                if loop:
                    try:
                        loop.close()
                    except:
                        pass  # Ignore loop close errors if already closed or in use
                logger.info(f"{bg_log_prefix_thread} Background thread finished and cleaned up.")

        try:
            background_thread_obj = threading.Thread(  # Renamed to avoid conflict
                target=run_background_task_with_new_loop,
                args=(user_input_from_req, session_id_for_logs, classification_for_background,
                      image_b64_from_req, vlm_desc_for_bg_and_direct),  # Pass necessary args
                daemon=True
            )
            background_thread_obj.start()
            logger.info(f"{request_id}: Launched background_generate in thread {background_thread_obj.ident}.")
        except Exception as launch_err:
            logger.error(f"{request_id}: Failed to launch background thread: {launch_err}")
            # Log this failure as it impacts deeper processing.
            add_interaction(db, session_id=session_id_for_logs, mode="chat", input_type="log_error",
                            user_input="Background Task Launch Failed", llm_response=f"Error: {launch_err}")

        # --- 7. Format and Return/Stream the IMMEDIATE Response (from direct_generate) ---
        model_id_for_response = META_MODEL_NAME_STREAM if stream_requested_by_client else META_MODEL_NAME_NONSTREAM

        if stream_requested_by_client:
            logger.info(f"{request_id}: Client requested stream. Creating SSE generator for direct response.")
            # The direct_response_text_val for streaming will be generated inside the _stream... generator now
            # The classification_for_background is what the BG task will use. The streamer might use a simpler one or none.
            sse_generator = _stream_openai_chat_response_generator_flask(
                session_id=session_id_for_logs,
                user_input=user_input_from_req,
                classification="direct_stream_elp1",  # Indication for the streamer's internal logic
                image_b64=image_b64_from_req,  # Pass user's image for direct_generate inside streamer
                model_name=model_id_for_response
            )
            resp_obj = Response(sse_generator, mimetype='text/event-stream')
            resp_obj.headers['Content-Type'] = 'text/event-stream; charset=utf-8'
            resp_obj.headers['Cache-Control'] = 'no-cache'
            resp_obj.headers['Connection'] = 'keep-alive'
            status_code_val = 200  # Stream initiated successfully
        else:  # Non-streaming path
            logger.debug(
                f"{request_id}: Formatting non-streaming JSON (direct_response_text_val: '{direct_response_text_val[:50]}...')")
            if status_code_val != 200:  # Error occurred during direct_generate
                resp_data_err, _ = _create_openai_error_response(direct_response_text_val, status_code=status_code_val)
                response_payload_str = json.dumps(resp_data_err)
            else:
                resp_data_ok = _format_openai_chat_response(direct_response_text_val, model_name=model_id_for_response)
                response_payload_str = json.dumps(resp_data_ok)
            resp_obj = Response(response_payload_str, status=status_code_val, mimetype='application/json')

        final_response_status_code = status_code_val  # Log the status of the immediate response

    except ValueError as ve:  # Catch input validation errors
        logger.warning(f"{request_id}: Invalid request: {ve}")
        resp_data_err, status_code_val = _create_openai_error_response(str(ve), err_type="invalid_request_error",
                                                                       status_code=400)
        resp_obj = Response(json.dumps(resp_data_err), status=status_code_val, mimetype='application/json')
        final_response_status_code = status_code_val
    except TaskInterruptedException as tie_main:  # Catch if direct_generate or RAG prep raised it
        logger.error(f"🚦 {request_id}: Main handler caught TaskInterruptedException (ELP1 path): {tie_main}")
        resp_data_err, status_code_val = _create_openai_error_response(
            f"Processing interrupted by a higher priority task: {tie_main}",
            err_type="server_error", code="task_interrupted", status_code=503)
        resp_obj = Response(json.dumps(resp_data_err), status=status_code_val, mimetype='application/json')
        final_response_status_code = status_code_val
    except Exception as main_err:
        logger.exception(f"{request_id}: 🔥🔥 UNHANDLED exception in main handler:")
        err_msg_main = f"Internal server error: {type(main_err).__name__}"
        resp_data_err, status_code_val = _create_openai_error_response(err_msg_main, status_code=500)
        resp_obj = Response(json.dumps(resp_data_err), status=status_code_val, mimetype='application/json')
        final_response_status_code = status_code_val
        try:
            if db: add_interaction(db, session_id=session_id_for_logs, mode="chat", input_type='error',
                                   user_input=f"Main Handler Error. Req: {request_data_log_snippet}",
                                   llm_response=err_msg_main[:2000]); db.commit()
        except Exception as db_err_log:
            logger.error(f"{request_id}: ❌ Failed log main handler error: {db_err_log}")

    finally:
        duration_req_main = (time.monotonic() - start_req_time_main_handler) * 1000
        logger.info(
            f"🏁 OpenAI Chat Request {request_id} (DualGen) handled in {duration_req_main:.2f} ms. Final HTTP Status: {final_response_status_code}")
        # DB session g.db is closed automatically by @app.teardown_request

    if resp_obj is None:  # Safety: Should always have a response object by now
        logger.error(f"{request_id}: Handler finished, but 'resp_obj' is None! Fallback error.")
        resp_data_err, status_code_val = _create_openai_error_response(
            "Internal error: Handler did not produce response.", status_code=500)
        resp_obj = Response(json.dumps(resp_data_err), status=500, mimetype='application/json')

    return resp_obj

@app.route("/v1/models", methods=["GET"])
def handle_openai_models():
    """
    Handles requests mimicking OpenAI's models endpoint.
    Lists available models, including chat and TTS.
    """
    logger.info("Received request for /v1/models")
    start_req = time.monotonic()
    status_code = 200

    model_list = [
        {
            "id": META_MODEL_NAME_STREAM,
            "object": "model",
            "created": int(time.time()), # Placeholder timestamp
            "owned_by": META_MODEL_OWNER,
            "permission": [], "root": META_MODEL_NAME_STREAM, "parent": None,
        },
        {
            "id": META_MODEL_NAME_NONSTREAM,
            "object": "model",
            "created": int(time.time()),
            "owned_by": META_MODEL_OWNER,
            "permission": [], "root": META_MODEL_NAME_NONSTREAM, "parent": None,
        },
        # --- NEW: Add TTS Model Entry ---
        {
            "id": TTS_MODEL_NAME_CLIENT_FACING, # Use the constant
            "object": "model", # Standard object type
            "created": int(time.time()), # Placeholder timestamp
            "owned_by": META_MODEL_OWNER, # Your foundation name
            "permission": [], # Standard permissions array
            "root": TTS_MODEL_NAME_CLIENT_FACING, # Root is itself
            "parent": None, # No parent model
            # Optionally, add a 'capabilities' or 'description' field if useful,
            # though OpenAI's TTS model listing is very basic.
            # "description": "Text-to-Speech model with Zephyrine Persona."
        },
        {
            "id": ASR_MODEL_NAME_CLIENT_FACING,  # Use the constant
            "object": "model",
            "created": int(time.time()),
            "owned_by": META_MODEL_OWNER,  # Your foundation name
            "permission": [],
            "root": ASR_MODEL_NAME_CLIENT_FACING,
            "parent": None,
            # "description": "Speech-to-Text model based on Whisper." # Optional
        },
        {
            "id": IMAGE_GEN_MODEL_NAME_CLIENT_FACING,  # Use the constant
            "object": "model",
            "created": int(time.time()),
            "owned_by": META_MODEL_OWNER,  # Your foundation name
            "permission": [],
            "root": IMAGE_GEN_MODEL_NAME_CLIENT_FACING,
            "parent": None,
            # "description": "Image Generation Model (Internal Use Only)." # Optional
        }
        # --- END NEW TTS Model Entry ---
    ]

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

# === NEW: OpenAI Compatible TTS Endpoint (Stub) ===
@app.route("/v1/audio/speech", methods=["POST"])
def handle_openai_tts():
    """
    Handles requests mimicking OpenAI's Text-to-Speech endpoint.
    Expects model "Zephyloid-Alpha", uses audio_worker.py with ELP1 priority.
    """
    start_req = time.monotonic()
    request_id = f"req-tts-{uuid.uuid4()}" # Unique ID for this TTS request
    logger.info(f"🚀 Flask OpenAI-Style TTS Request ID: {request_id} (Worker ELP1)")

    # --- Initialize variables ---
    db: Session = g.db # Use request-bound session from Flask's g, set by @app.before_request
    session_id: str = f"tts_req_default_{request_id}" # Default if not parsed
    raw_request_data: Optional[Dict] = None
    input_text: Optional[str] = None
    model_requested: Optional[str] = None
    voice_requested: Optional[str] = None
    response_format_requested: Optional[str] = "mp3" # Default format

    final_response_status_code: int = 500 # For logging actual outcome
    resp: Optional[Response] = None
    request_data_snippet_for_log: str = "No request data processed"

    try:
        # --- 1. Get and Validate Request JSON Body ---
        try:
            raw_request_data = request.get_json()
            if not raw_request_data:
                raise ValueError("Empty JSON payload received.")
            # Safely create a snippet for logging, even if full dump fails
            try: request_data_snippet_for_log = json.dumps(raw_request_data)[:1000]
            except: request_data_snippet_for_log = str(raw_request_data)[:1000]
        except Exception as json_err:
            logger.warning(f"{request_id}: Failed to get/parse JSON body: {json_err}")
            try: request_data_snippet_for_log = request.get_data(as_text=True)[:1000]
            except: request_data_snippet_for_log = "Could not read request body"

            resp_data, status_code = _create_openai_error_response(
                f"Request body is missing or invalid JSON: {json_err}",
                err_type="invalid_request_error", status_code=400
            )
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
            final_response_status_code = status_code
            # Return early on critical parsing error
            return resp

        # --- 2. Extract Expected Parameters ---
        input_text = raw_request_data.get("input")
        model_requested = raw_request_data.get("model")
        voice_requested = raw_request_data.get("voice")
        # Update session_id if provided in the request, else keep default
        session_id = raw_request_data.get("session_id", session_id)
        response_format_requested = raw_request_data.get("response_format", "mp3").lower()
        # speed = raw_request_data.get("speed", 1.0) # SCLParser handles speed via tags in input_text

        logger.debug(
            f"{request_id}: TTS Request Parsed - SessionID: {session_id}, Input: '{str(input_text)[:50]}...', "
            f"Client Model Req: {model_requested}, Internal Voice/Speaker: {voice_requested}, "
            f"Format: {response_format_requested}"
        )

        # --- 3. Validate Core Parameters ---
        if not input_text or not isinstance(input_text, str):
            raise ValueError("'input' field is required and must be a string.")
        if not model_requested or not isinstance(model_requested, str):
            # This is the client-facing model name, e.g., "Zephyloid-Alpha"
            raise ValueError("'model' field (e.g., 'Zephyloid-Alpha') is required.")
        if not voice_requested or not isinstance(voice_requested, str):
            # This will be the actual MeloTTS speaker ID like "EN-US"
            raise ValueError("'voice' field (MeloTTS speaker ID, e.g., EN-US) is required.")

        # --- 4. Validate Requested Model Name ---
        if model_requested != TTS_MODEL_NAME_CLIENT_FACING:
            logger.warning(f"{request_id}: Invalid TTS model requested '{model_requested}'. Expected '{TTS_MODEL_NAME_CLIENT_FACING}'.")
            resp_data, status_code = _create_openai_error_response(
                f"Invalid model. This endpoint only supports the '{TTS_MODEL_NAME_CLIENT_FACING}' model for TTS.",
                err_type="invalid_request_error", code="model_not_found", status_code=404
            )
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
            final_response_status_code = status_code
            # Log duration before early return for invalid model
            # (This specific logging moved to finally block for consistency)
            return resp

        # --- 5. Determine Internal Melo Language from Voice Parameter ---
        melo_language = "EN" # Default language
        try:
            lang_part = voice_requested.split('-')[0].upper()
            # Define supported languages by your MeloTTS setup in audio_worker.py
            supported_melo_langs = ["EN", "ZH", "JP", "ES", "FR", "KR", "DE"] # Example
            if lang_part in supported_melo_langs:
                melo_language = lang_part
            else:
                logger.warning(f"{request_id}: Could not infer a supported language from voice '{voice_requested}'. Defaulting to {melo_language}.")
        except Exception: # Catch potential errors like voice_requested not being a string or empty
            logger.warning(f"{request_id}: Error parsing language from voice '{voice_requested}'. Defaulting to {melo_language}.")
        logger.debug(f"{request_id}: Using Internal Melo Language: {melo_language} (for voice: {voice_requested})")

        # --- 6. Prepare and Execute Audio Worker ---
        audio_worker_script = os.path.join(SCRIPT_DIR, "audio_worker.py")
        if not os.path.exists(audio_worker_script):
            logger.error(f"{request_id}: audio_worker.py not found at {audio_worker_script}")
            # This is a server configuration error
            raise FileNotFoundError(f"Audio worker script missing at {audio_worker_script}")

        # Define a temporary directory for any files the worker might create (e.g., for MP3 conversion)
        temp_audio_dir = os.path.join(SCRIPT_DIR, "temp_audio_worker_files")
        os.makedirs(temp_audio_dir, exist_ok=True)

        worker_command = [
            APP_PYTHON_EXECUTABLE, # Path to python in venv
            audio_worker_script,
            "--model-lang", melo_language,
            "--device", "auto", # Or could be configurable via app config
            "--temp-dir", temp_audio_dir
            # Worker will use defaults for output-file if not in test mode
        ]

        worker_request_data = {
            "input": input_text,
            "voice": voice_requested, # The SCLParser/MeloTTS speaker ID
            "response_format": response_format_requested,
            "request_id": request_id # Pass request_id for better worker logging continuity
            # SCL settings are embedded in the input_text itself by the client
        }

        logger.info(f"{request_id}: Executing audio worker with ELP1 priority...")
        # Call the helper function to execute the worker
        parsed_response_from_worker, error_string_from_worker = _execute_audio_worker_with_priority(
            worker_command=worker_command,
            request_data=worker_request_data,
            priority=ELP1, # CRITICAL: Use ELP1 for user-facing TTS
            worker_cwd=SCRIPT_DIR, # Worker can run from the app's directory
            timeout=60 # Adjust timeout as needed for TTS generation (e.g., 60 seconds)
        )

        # --- 7. Process Worker Response ---
        if error_string_from_worker:
            # An error occurred during worker execution or communication
            logger.error(f"{request_id}: Audio worker execution failed: {error_string_from_worker}")
            resp_data, status_code = _create_openai_error_response(
                f"Audio generation failed: {error_string_from_worker}",
                err_type="server_error", status_code=500
            )
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
            final_response_status_code = status_code
        elif parsed_response_from_worker and "result" in parsed_response_from_worker and "audio_base64" in parsed_response_from_worker["result"]:
            # Successful response from worker
            audio_info = parsed_response_from_worker["result"]
            audio_b64_data = audio_info["audio_base64"]
            actual_audio_format = audio_info.get("format", "mp3") # Get the format worker actually produced
            response_mime_type = audio_info.get("mime_type", f"audio/{actual_audio_format}")

            logger.info(f"{request_id}: Audio successfully generated by worker. Format: {actual_audio_format}, Length (b64): {len(audio_b64_data)}")
            try:
                audio_bytes = base64.b64decode(audio_b64_data)
                # Return the raw audio bytes with the correct MIME type
                resp = Response(audio_bytes, status=200, mimetype=response_mime_type)
                final_response_status_code = 200
            except Exception as decode_err:
                logger.error(f"{request_id}: Failed to decode base64 audio from worker: {decode_err}")
                resp_data, status_code = _create_openai_error_response(
                    "Failed to decode audio data received from worker.",
                    err_type="server_error", status_code=500
                )
                resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
                final_response_status_code = status_code
        else:
            # Worker returned a response, but it was invalid or incomplete
            logger.error(f"{request_id}: Audio worker returned invalid or incomplete response: {parsed_response_from_worker}")
            resp_data, status_code = _create_openai_error_response(
                "Audio worker returned an invalid response structure.",
                err_type="server_error", status_code=500
            )
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
            final_response_status_code = status_code

    except ValueError as ve: # Catches our explicit ValueErrors for bad client input
        logger.warning(f"{request_id}: Invalid TTS request parameters: {ve}")
        resp_data, status_code = _create_openai_error_response(
            str(ve), err_type="invalid_request_error", status_code=400
        )
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_response_status_code = status_code
    except FileNotFoundError as fnf_err: # For missing worker script
        logger.error(f"{request_id}: Server configuration error: {fnf_err}")
        resp_data, status_code = _create_openai_error_response(
            f"Server configuration error: {fnf_err}",
            err_type="server_error", status_code=500
        )
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_response_status_code = status_code
    except Exception as main_handler_err: # Catches other unexpected errors in this handler
        logger.exception(f"{request_id}: 🔥🔥 Unhandled exception in TTS endpoint main handler:")
        error_message = f"Internal server error in TTS endpoint: {type(main_handler_err).__name__}"
        resp_data, status_code = _create_openai_error_response(error_message, status_code=500)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_response_status_code = status_code
        # Log this critical failure to DB if possible
        try:
            if 'db' in g and g.db: # Check if request DB session exists
                 add_interaction(g.db, session_id=session_id, mode="tts", input_type='error',
                                 user_input=f"TTS Handler Error. Request: {request_data_snippet_for_log}",
                                 llm_response=error_message[:2000])
            else: logger.error(f"{request_id}: Cannot log TTS handler error: DB session 'g.db' unavailable.")
        except Exception as db_err_log:
             logger.error(f"{request_id}: ❌ Failed log TTS handler error to DB: {db_err_log}")

    finally:
        # This block ALWAYS runs after try/except
        duration_req = (time.monotonic() - start_req) * 1000
        # Log final status using the status code determined by the try/except blocks
        logger.info(f"🏁 OpenAI-Style TTS Request {request_id} handled in {duration_req:.2f} ms. Final Status: {final_response_status_code}")
        # Flask's g.db is closed automatically by the @app.teardown_request handler

    # --- Return Response ---
    # Ensure 'resp' is always assigned before returning
    if resp is None:
        logger.error(f"{request_id}: TTS Handler logic flaw - response object 'resp' was not assigned!")
        # Create a generic error response if 'resp' is somehow still None
        resp_data, status_code = _create_openai_error_response(
            "Internal error: TTS Handler failed to produce a response object.",
            err_type="server_error", status_code=500
        )
        resp = Response(json.dumps(resp_data), status=500, mimetype='application/json')
        final_response_status_code = status_code # Update for final log if it changed here
        # Log this critical internal failure if possible
        try:
             if 'db' in g and g.db:
                 add_interaction(g.db, session_id=session_id, mode="tts", input_type='error',
                                 user_input=f"TTS Handler No Resp Object. Req: {request_data_snippet_for_log}",
                                 llm_response="Critical: No response object created by handler.")
        except Exception as db_err_log_final:
             logger.error(f"{request_id}: Failed to log 'no response object' error to DB: {db_err_log_final}")

    return resp # Return the final Flask Response object

@app.route("/v1/audio/transcriptions", methods=["POST"])
def handle_openai_asr_transcriptions():
    """
    Handles requests mimicking OpenAI's Audio Transcriptions endpoint.
    Currently a STUB - Acknowledges request but does not perform ASR.
    Expected model: "Zephyloid-Whisper-Normal"
    """
    start_req = time.monotonic()
    request_id = f"req-asr-{uuid.uuid4()}"
    logger.info(f"🚀 Flask OpenAI-Style ASR Request ID: {request_id} (STUB)")

    final_response_status_code: int = 500
    resp: Optional[Response] = None
    session_id_for_log: str = f"asr_req_default_{request_id}" # For logging in case of early failure

    try:
        # OpenAI ASR endpoint uses multipart/form-data
        if not request.content_type or not request.content_type.startswith('multipart/form-data'):
            logger.warning(f"{request_id}: Invalid content type: {request.content_type}. Expected multipart/form-data.")
            resp_data, status_code = _create_openai_error_response(
                "Invalid content type. Must be multipart/form-data.",
                err_type="invalid_request_error", status_code=400
            )
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
            final_response_status_code = status_code
            return resp

        # --- Extract Parameters from Form Data ---
        file_storage = request.files.get('file')
        model_requested = request.form.get('model')
        language_requested = request.form.get('language') # Optional (ISO 639-1 code)
        prompt_requested = request.form.get('prompt') # Optional context prompt
        response_format_requested = request.form.get('response_format', 'json').lower() # json, text, srt, verbose_json, vtt
        temperature_requested = request.form.get('temperature', type=float) # Optional

        # --- Get session_id if provided in form data (less common for this endpoint) ---
        session_id_for_log = request.form.get("session_id", session_id_for_log)

        logger.debug(
            f"{request_id}: ASR Request Parsed - File: {'Present' if file_storage else 'Missing'}, "
            f"Model: {model_requested}, Language: {language_requested}, Prompt: {str(prompt_requested)[:30]}..., "
            f"Format: {response_format_requested}, Temp: {temperature_requested}"
        )

        # --- Validate Core Parameters ---
        if not file_storage:
            raise ValueError("'file' field (audio data) is required.")
        if not model_requested:
            raise ValueError("'model' field (e.g., 'Zephyloid-Whisper-Normal') is required.")

        # --- Validate Requested Model Name ---
        if model_requested != ASR_MODEL_NAME_CLIENT_FACING:
            logger.warning(f"{request_id}: Invalid ASR model requested '{model_requested}'. Expected '{ASR_MODEL_NAME_CLIENT_FACING}'.")
            resp_data, status_code = _create_openai_error_response(
                f"Invalid model. This endpoint only supports the '{ASR_MODEL_NAME_CLIENT_FACING}' model for ASR.",
                err_type="invalid_request_error", code="model_not_found", status_code=404
            )
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
            final_response_status_code = status_code
            return resp

        # --- STUB IMPLEMENTATION ---
        # In a real implementation:
        # 1. Save `file_storage.stream` to a temporary file.
        # 2. Prepare command for `whisper_worker.py` with the temp file path and other options.
        # 3. Call a helper like `_execute_whisper_worker_with_priority(..., priority=ELP1, ...)`
        # 4. Get the transcribed text (or structured JSON) from the worker.
        # 5. Format the response according to `response_format_requested`.
        # 6. Delete the temporary file.

        supported_response_formats = ["json", "text", "srt", "verbose_json", "vtt"]
        if response_format_requested not in supported_response_formats:
            logger.warning(f"{request_id}: Unsupported ASR response_format: {response_format_requested}")
            resp_data, status_code = _create_openai_error_response(
                f"Unsupported response_format: '{response_format_requested}'. Supported: {', '.join(supported_response_formats)}.",
                err_type="invalid_request_error", status_code=400
            )
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
            final_response_status_code = status_code
            return resp

        logger.warning(f"{request_id}: ASR transcription is a STUB. Returning 501 Not Implemented.")
        resp_data, status_code = _create_openai_error_response(
            "ASR endpoint is currently a stub and does not perform transcription.",
            err_type="server_error", code="stub_not_implemented", status_code=501 # 501 Not Implemented
        )
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_response_status_code = status_code
        # --- END STUB ---

    except ValueError as ve: # Catch our explicit ValueErrors for bad input
        logger.warning(f"{request_id}: Invalid ASR request: {ve}")
        resp_data, status_code = _create_openai_error_response(
            str(ve), err_type="invalid_request_error", status_code=400
        )
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_response_status_code = status_code
    except Exception as e:
        logger.exception(f"{request_id}: 🔥🔥 Unhandled exception in ASR endpoint STUB:")
        error_message = f"Internal server error in ASR stub: {type(e).__name__}"
        resp_data, status_code = _create_openai_error_response(error_message, status_code=500)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_response_status_code = status_code
        # Log this critical failure to DB if possible
        try:
            if 'db' in g and g.db: # Check if request DB session exists
                 add_interaction(g.db, session_id=session_id_for_log, mode="asr", input_type='error',
                                 user_input=f"ASR Handler Error. File: {file_storage.filename if file_storage else 'N/A'}",
                                 llm_response=error_message[:2000])
            else: logger.error(f"{request_id}: Cannot log ASR handler error: DB session 'g.db' unavailable.")
        except Exception as db_err_log:
             logger.error(f"{request_id}: ❌ Failed log ASR handler error to DB: {db_err_log}")

    finally:
        duration_req = (time.monotonic() - start_req) * 1000
        logger.info(f"🏁 OpenAI-Style ASR Request {request_id} STUB handled in {duration_req:.2f} ms. Status: {final_response_status_code}")
        # g.db is closed by teardown_request

    if resp is None: # Safety net
        logger.error(f"{request_id}: ASR Handler STUB finished unexpectedly without response object!")
        resp_data, status_code = _create_openai_error_response("Internal error: ASR Handler STUB no response.", status_code=500)
        resp = Response(json.dumps(resp_data), status=500, mimetype='application/json')
        # final_response_status_code = status_code # Already captured or will be by default
        try:
             if 'db' in g and g.db:
                 add_interaction(g.db, session_id=session_id_for_log, mode="asr", input_type='error',
                                 user_input=f"ASR Handler No Resp. File: {file_storage.filename if file_storage else 'N/A'}",
                                 llm_response="Critical: No response object created by ASR handler stub.")
        except Exception as db_err_log_final:
             logger.error(f"{request_id}: Failed log 'no response object' ASR error to DB: {db_err_log_final}")

    return resp

# === NEW: OpenAI Compatible Image Generation Endpoint (Stub) ===
@app.route("/v1/images/generations", methods=["POST"])
async def handle_openai_image_generations():  # Route is async
    start_req = time.monotonic()
    request_id = f"req-img-gen-{uuid.uuid4()}"
    logger.info(f"🚀 OpenAI-Style Image Generation Request ID: {request_id} (ELP1 Priority)")

    db: Session = g.db
    final_response_status_code: int = 500
    resp: Optional[Response] = None
    session_id_for_log: str = f"img_gen_req_default_{request_id}"
    raw_request_data: Optional[Dict] = None
    request_data_snippet_for_log: str = "No request data processed"

    try:
        try:
            raw_request_data = request.get_json()
            if not raw_request_data: raise ValueError("Empty JSON payload received.")
            try:
                request_data_snippet_for_log = json.dumps(raw_request_data)[:1000]
            except:
                request_data_snippet_for_log = str(raw_request_data)[:1000]
        except Exception as json_err:
            logger.warning(f"{request_id}: Failed to get/parse JSON body: {json_err}")
            try:
                request_data_snippet_for_log = request.get_data(as_text=True)[:1000]
            except:
                request_data_snippet_for_log = "Could not read request body"
            resp_data, status_code_val = _create_openai_error_response(
                f"Request body is missing or invalid JSON: {json_err}", err_type="invalid_request_error",
                status_code=400)
            resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
            final_response_status_code = status_code_val
            return resp

        prompt_from_user = raw_request_data.get("prompt")
        model_requested = raw_request_data.get("model")

        # --- MODIFICATION FOR n_images ---
        # Default to 2 images for this direct ELP1 endpoint if 'n' is not specified by the client.
        # Client can still override by sending "n": 1 or "n": X.
        n_images_requested_by_client = raw_request_data.get("n")
        if n_images_requested_by_client is None:
            n_images = 2  # Default to 2 for ELP1 direct generation
            logger.info(
                f"{request_id}: 'n' not specified by client, defaulting to {n_images} for ELP1 image generation.")
        else:
            try:
                n_images = int(n_images_requested_by_client)
                if n_images < 1:
                    logger.warning(
                        f"{request_id}: Client requested 'n={n_images_requested_by_client}', which is invalid. Defaulting to 1.")
                    n_images = 1
                # You might want to add a server-side cap on 'n' here, e.g., if n_images > MAX_IMAGES_PER_REQUEST: n_images = MAX_IMAGES_PER_REQUEST
            except ValueError:
                logger.warning(
                    f"{request_id}: Client sent invalid value for 'n': '{n_images_requested_by_client}'. Defaulting to 2.")
                n_images = 2
        # --- END MODIFICATION FOR n_images ---

        size_requested_str = raw_request_data.get("size", IMAGE_GEN_DEFAULT_SIZE)
        response_format_requested = raw_request_data.get("response_format", "b64_json").lower()
        quality_requested = raw_request_data.get("quality", "standard")
        style_requested = raw_request_data.get("style", "vivid")
        user_provided_id = raw_request_data.get("user")

        session_id_for_log = raw_request_data.get("session_id", session_id_for_log)
        if ai_chat:
            ai_chat.current_session_id = session_id_for_log
        else:
            logger.error(f"{request_id}: ai_chat instance not available. Cannot set session ID for helpers.")
            resp_data, status_code_val = _create_openai_error_response("Server AI component not ready.",
                                                                       err_type="server_error", status_code=503)
            resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
            final_response_status_code = status_code_val
            return resp

        logger.debug(
            f"{request_id}: Image Gen Request Parsed - Prompt: '{str(prompt_from_user)[:50]}...', "
            f"ModelReq: {model_requested}, N (final): {n_images}, Size: {size_requested_str}, RespFormat: {response_format_requested}, "  # Log final n_images
            f"Quality: {quality_requested}, Style: {style_requested}, User: {user_provided_id}"
        )

        if not prompt_from_user or not isinstance(prompt_from_user, str):
            raise ValueError("'prompt' field is required and must be a string.")
        if not model_requested or model_requested != IMAGE_GEN_MODEL_NAME_CLIENT_FACING:
            logger.warning(
                f"{request_id}: Invalid Image Gen model '{model_requested}'. Expected '{IMAGE_GEN_MODEL_NAME_CLIENT_FACING}'.")
            resp_data, status_code_val = _create_openai_error_response(
                f"Invalid model. This endpoint only supports the '{IMAGE_GEN_MODEL_NAME_CLIENT_FACING}' model for image generation.",
                err_type="invalid_request_error", code="model_not_found", status_code=404)
            resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
            final_response_status_code = status_code_val
            return resp

        if response_format_requested not in ["b64_json", "url"]:
            logger.warning(
                f"{request_id}: Invalid 'response_format': {response_format_requested}. Defaulting to 'b64_json'.")
            response_format_requested = "b64_json"

        logger.info(f"{request_id}: Refining user prompt for image generation (ELP1)...")
        url_retriever_obj, history_retriever_obj, _ = await asyncio.to_thread(
            ai_chat._get_rag_retriever, db, prompt_from_user, priority=ELP1
        )
        retrieved_history_docs_for_refine = []
        if history_retriever_obj:
            retrieved_history_docs_for_refine = await asyncio.to_thread(history_retriever_obj.invoke, prompt_from_user)
        history_rag_str_for_refine = ai_chat._format_docs(retrieved_history_docs_for_refine, source_type="History RAG")
        direct_hist_interactions = await asyncio.to_thread(get_global_recent_interactions, db, limit=3)
        recent_direct_history_str_for_refine = ai_chat._format_direct_history(direct_hist_interactions)

        refined_prompt = await ai_chat._refine_direct_image_prompt_async(
            db=db, session_id=session_id_for_log, user_image_request=prompt_from_user,
            history_rag_str=history_rag_str_for_refine, recent_direct_history_str=recent_direct_history_str_for_refine,
            priority=ELP1
        )
        if not refined_prompt:
            logger.warning(f"{request_id}: Prompt refinement yielded empty or failed. Using original prompt.")
            refined_prompt = prompt_from_user
        elif refined_prompt == prompt_from_user:
            logger.info(f"{request_id}: Prompt refinement did not change the prompt. Using original.")
        else:
            logger.info(f"{request_id}: Using refined prompt for generation: '{refined_prompt}'")
        try:
            add_interaction(db, session_id=session_id_for_log, mode="image_gen", input_type="text_prompt_to_worker",
                            user_input=prompt_from_user, llm_response=refined_prompt)
            db.commit()
        except Exception as db_log_err:
            logger.error(f"{request_id}: Failed to log refined prompt: {db_log_err}")
            if db: db.rollback()

        logger.info(
            f"{request_id}: Requesting {n_images} image(s) from AIProvider with ELP1 priority. Refined Prompt: '{refined_prompt[:100]}...'")
        all_generated_image_data_from_provider = []
        error_occurred_during_generation_loop = False
        final_error_message_from_loop = None

        for i in range(n_images):  # Loop n_images times
            if error_occurred_during_generation_loop: break
            logger.info(f"{request_id}: Generating image {i + 1}/{n_images}...")
            list_of_image_dicts, gen_error_msg = await ai_provider.generate_image_async(
                prompt=refined_prompt, image_base64=None, priority=ELP1
            )
            if gen_error_msg:
                logger.error(f"{request_id}: Image generation failed for image attempt {i + 1}: {gen_error_msg}")
                final_error_message_from_loop = gen_error_msg
                if interruption_error_marker in gen_error_msg: raise TaskInterruptedException(gen_error_msg)
                error_occurred_during_generation_loop = True
                break
            elif list_of_image_dicts and isinstance(list_of_image_dicts, list) and list_of_image_dicts:
                all_generated_image_data_from_provider.append(list_of_image_dicts[0])
                logger.info(f"{request_id}: Image {i + 1}/{n_images} data received from provider.")
            else:
                logger.warning(
                    f"{request_id}: Image generation for image attempt {i + 1} returned no data and no error.")
                final_error_message_from_loop = "Image generation worker returned no data for an image."
                error_occurred_during_generation_loop = True
                break

        if error_occurred_during_generation_loop or not all_generated_image_data_from_provider:
            resp_data, status_code_val = _create_openai_error_response(
                final_error_message_from_loop or "Image generation failed to produce any results.",
                err_type="server_error", status_code=500)
            resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
            final_response_status_code = status_code_val
            return resp

        response_data_items_for_client = []
        for img_data_dict_from_worker in all_generated_image_data_from_provider:
            png_b64_data = img_data_dict_from_worker.get("b64_json")
            if png_b64_data:
                if response_format_requested == "b64_json":
                    response_data_items_for_client.append({"b64_json": png_b64_data})
                elif response_format_requested == "url":
                    logger.warning(f"{request_id}: Returning data URI for 'url' format request.")
                    response_data_items_for_client.append(
                        {"url": f"data:image/png;base64,{png_b64_data}"})  # Provide data URI
            else:
                logger.error(
                    f"{request_id}: Worker returned image data item without 'b64_json' (PNG). Skipping item: {img_data_dict_from_worker}")

        if not response_data_items_for_client:
            logger.error(
                f"{request_id}: No valid image data (PNG b64_json) found after processing worker response for {n_images} images.")
            resp_data, status_code_val = _create_openai_error_response(
                "Failed to obtain valid image data from generation worker.", err_type="server_error", status_code=500)
            resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
            final_response_status_code = status_code_val
            return resp

        openai_response_body = {"created": int(time.time()), "data": response_data_items_for_client}
        response_payload = json.dumps(openai_response_body)
        resp = Response(response_payload, status=200, mimetype='application/json')
        final_response_status_code = 200

    except ValueError as ve:
        logger.warning(f"{request_id}: Invalid Image Gen request: {ve}")
        resp_data, status_code_val = _create_openai_error_response(str(ve), err_type="invalid_request_error",
                                                                   status_code=400)
        resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
        final_response_status_code = status_code_val
    except TaskInterruptedException as tie:
        logger.warning(f"🚦 {request_id}: Image Generation request (ELP1) INTERRUPTED: {tie}")
        resp_data, status_code_val = _create_openai_error_response(f"Image generation task was interrupted: {tie}",
                                                                   err_type="server_error", code="task_interrupted",
                                                                   status_code=503)
        resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
        final_response_status_code = status_code_val
    except Exception as e:
        logger.exception(f"{request_id}: 🔥🔥 Unhandled exception in Image Gen endpoint:")
        error_message = f"Internal server error in Image Gen endpoint: {type(e).__name__} - {str(e)}"
        resp_data, status_code_val = _create_openai_error_response(error_message, err_type="server_error",
                                                                   status_code=500)
        resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
        final_response_status_code = status_code_val
        try:
            if db:
                add_interaction(db, session_id=session_id_for_log, mode="image_gen", input_type='error',
                                user_input=f"Image Gen Handler Error. Request: {request_data_snippet_for_log}",
                                llm_response=error_message[:2000]); db.commit()
            else:
                logger.error(f"{request_id}: Cannot log Image Gen handler error: DB session unavailable.")
        except Exception as db_err_log:
            logger.error(f"{request_id}: ❌ Failed log Image Gen handler error to DB: {db_err_log}")
            if db: db.rollback()
    finally:
        duration_req = (time.monotonic() - start_req) * 1000
        logger.info(
            f"🏁 OpenAI-Style Image Gen Request {request_id} handled in {duration_req:.2f} ms. Final HTTP Status: {final_response_status_code}")

    if resp is None:
        logger.error(f"{request_id}: Image Gen Handler logic flaw - response object 'resp' was not assigned!")
        resp_data, status_code_val = _create_openai_error_response(
            "Internal error: Image Gen Handler failed to produce a response object.", err_type="server_error",
            status_code=500)
        resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
        try:
            if db: add_interaction(db, session_id=session_id_for_log, mode="image_gen", input_type='error',
                                   user_input=f"Image Gen Handler No Resp. Req: {request_data_snippet_for_log}",
                                   llm_response="Critical: No response object created by Image Gen handler."); db.commit()
        except Exception as db_err_log_final:
            logger.error(f"{request_id}: Failed to log 'no response object' Image Gen error to DB: {db_err_log_final}")
            if db: db.rollback()
    return resp


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


try:
    ai_provider = AIProvider(PROVIDER)
    global_ai_provider_ref = ai_provider
    ai_chat = AIChat(ai_provider)
    AGENT_CWD = os.path.dirname(os.path.abspath(__file__))
    SUPPORTS_COMPUTER_USE = True
    ai_agent = AmaryllisAgent(ai_provider, AGENT_CWD, SUPPORTS_COMPUTER_USE)
    logger.success("✅ AI Instances Initialized.")
except Exception as e:
    logger.critical(f"🔥🔥 Failed AI init: {e}")
    logger.exception("AI Init Traceback:")
    ai_provider = None
    sys.exit(1)

# Define startup_tasks (as you had it)
async def startup_tasks():
    logger.info("APP.PY: >>> Entered startup_tasks (async). <<<")
    task_start_time = time.monotonic()

    if ENABLE_FILE_INDEXER:
        logger.info("APP.PY: startup_tasks: Attempting to initialize global FileIndex vector store...")
        if ai_provider and ai_provider.embeddings:
            init_vs_start_time = time.monotonic()
            logger.info(
                "APP.PY: startup_tasks: >>> CALLING await init_file_vs_from_indexer(ai_provider). This will block here. <<<")
            await init_file_vs_from_indexer(ai_provider)  # This is initialize_global_file_index_vectorstore
            init_vs_duration = time.monotonic() - init_vs_start_time
            logger.info(
                f"APP.PY: startup_tasks: >>> init_file_vs_from_indexer(ai_provider) HAS COMPLETED. Duration: {init_vs_duration:.2f}s <<<")
        else:
            logger.error("APP.PY: startup_tasks: CRITICAL - AIProvider or embeddings None. Cannot init FileIndex VS.")
    else:
        logger.info("APP.PY: startup_tasks: File Indexer and its Vector Store are DISABLED by config.")

    if ENABLE_SELF_REFLECTION:
        logger.info("APP.PY: startup_tasks: Attempting to initialize global Reflection vector store...")
        if ai_provider and ai_provider.embeddings:
            init_refl_vs_start_time = time.monotonic()
            logger.info(
                "APP.PY: startup_tasks: >>> CALLING await asyncio.to_thread(initialize_global_reflection_vectorstore, ...). This will block here. <<<")
            temp_db_session_for_init = SessionLocal()  # type: ignore
            try:
                await asyncio.to_thread(initialize_global_reflection_vectorstore, ai_provider, temp_db_session_for_init)
            finally:
                temp_db_session_for_init.close()
            init_refl_vs_duration = time.monotonic() - init_refl_vs_start_time
            logger.info(
                f"APP.PY: startup_tasks: >>> initialize_global_reflection_vectorstore HAS COMPLETED. Duration: {init_refl_vs_duration:.2f}s <<<")
        else:
            logger.error("APP.PY: startup_tasks: CRITICAL - AIProvider or embeddings None. Cannot init Reflection VS.")
    else:
        logger.info("APP.PY: startup_tasks: Self Reflection and its Vector Store are DISABLED by config.")

    task_duration = time.monotonic() - task_start_time
    logger.info(f"APP.PY: >>> Exiting startup_tasks (async). Total Duration: {task_duration:.2f}s <<<")


# --- Main Execution Control ---

if __name__ == "__main__":
    # This block executes if app.py is run directly (e.g., python app.py)
    # This is NOT the typical way to run a Flask/Quart app for production or development with Hypercorn.
    # Hypercorn imports 'app' as a module.
    logger.error("This script (app.py) is designed to be run with an ASGI/WSGI server like Hypercorn.")
    logger.error("Example: hypercorn app:app --bind 127.0.0.1:11434")
    logger.error(
        "If you are trying to run standalone for testing limited functionality, some features might not work as expected, especially background tasks and full request handling.")

    # Optionally, you could add some minimal direct execution logic here for specific tests,
    # but it's generally better to test through the server.
    # For instance, you could try to initialize just the DB and AI provider for basic checks:
    # print("Attempting direct initialization for basic checks (not a full server run)...")
    # try:
    #     init_db() # Assuming this is globally defined
    #     if ai_provider is None: # Check if it was initialized globally by module import
    #          print("AI Provider not initialized globally. This path needs review for direct run.")
    #     # You might try calling startup_tasks here for testing, but it's tricky without the server's loop context
    #     # asyncio.run(startup_tasks()) # This might work depending on context
    #     print("Basic initializations attempted. For full functionality, use Hypercorn.")
    # except Exception as e:
    #     print(f"Error during direct initialization attempt: {e}")

    sys.exit(1)  # Exit because this isn't the intended way to run
else:
    # This block executes when app.py is imported as a module by a server (e.g., Hypercorn).
    logger.info("----------------------------------------------------------------------")
    logger.info(">>> APP.PY: MODULE IMPORTED BY SERVER (e.g., Hypercorn worker process) <<<")
    logger.info("----------------------------------------------------------------------")

    # Global instances ai_provider, ai_chat, ai_agent should be initialized above this block.
    if ai_provider is None or ai_chat is None or ai_agent is None:
        logger.critical("APP.PY: 🔥🔥 AI components NOT INITIALIZED GLOBALLY. Startup will likely fail.")
    else:
        logger.success("APP.PY: ✅ Core AI components appear initialized globally.")

    logger.info("APP.PY: 🚀 Preparing to run asynchronous startup_tasks...")
    startup_tasks_completed_successfully = False
    startup_tasks_start_time = time.monotonic()

    try:
        logger.debug("APP.PY: Setting up asyncio event loop for startup_tasks...")
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop.is_closed():
                logger.warning("APP.PY: Default asyncio event loop was closed. Creating new one.")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            logger.info("APP.PY: No current asyncio event loop. Creating new one.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        logger.info(
            "APP.PY: >>> CALLING loop.run_until_complete(startup_tasks()). This will block until startup_tasks finishes. <<<")
        loop.run_until_complete(startup_tasks())  # This should block
        startup_tasks_duration = time.monotonic() - startup_tasks_start_time
        logger.info(
            f"APP.PY: >>> loop.run_until_complete(startup_tasks()) HAS COMPLETED. Duration: {startup_tasks_duration:.2f}s <<<")

        # Check the event status from file_indexer directly after startup_tasks
        # This requires _file_index_vs_initialized_event to be accessible or a getter.
        # For simplicity, we'll rely on the logs from initialize_global_file_index_vectorstore itself.
        # from .file_indexer import _file_index_vs_initialized_event # Avoid direct import of private-like var
        # logger.info(f"APP.PY: Post startup_tasks, _file_index_vs_initialized_event.is_set(): {_file_index_vs_initialized_event.is_set()}")

        logger.success("APP.PY: ✅ Asynchronous startup_tasks (Vector Store Initializations) reported completion.")
        startup_tasks_completed_successfully = True

    except Exception as su_err:
        startup_tasks_duration = time.monotonic() - startup_tasks_start_time
        logger.error(
            f"APP.PY: 🚨🚨 CRITICAL FAILURE during loop.run_until_complete(startup_tasks()) after {startup_tasks_duration:.2f}s: {su_err} 🚨🚨")
        logger.exception("APP.PY: Startup Tasks Execution Traceback:")
        startup_tasks_completed_successfully = False  # Ensure it's false

    if startup_tasks_completed_successfully:
        logger.info("APP.PY: 🚀 Proceeding to start background services (File Indexer, Self Reflector)...")
        if ENABLE_FILE_INDEXER:
            logger.info("APP.PY: Starting File Indexer...")
            start_file_indexer()
        else:
            logger.info("APP.PY: File Indexer is DISABLED by config. Not starting.")

        if ENABLE_SELF_REFLECTION:
            logger.info("APP.PY: Starting Self Reflector...")
            start_self_reflector()
        else:
            logger.info("APP.PY: Self Reflector is DISABLED by config. Not starting.")
    else:
        logger.error(
            "APP.PY: Background services (File Indexer, Self Reflector) NOT started due to startup_tasks failure or incompletion.")

    logger.info("--------------------------------------------------------------------")
    logger.info("APP.PY: ✅ Zephyrine EngineMain module-level initializations complete.")
    logger.info(f"   Application (app) is now considered ready by this worker process.")
    logger.info("   Waiting for server to route requests...")
    logger.info("--------------------------------------------------------------------")