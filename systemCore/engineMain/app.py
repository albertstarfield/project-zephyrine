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
import pandas as pd
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
import tempfile
from werkzeug.utils import secure_filename # For safely handling uploaded filenames
import difflib
import contextlib # For ensuring driver quit
from urllib.parse import urlparse, parse_qs, quote_plus, urljoin
import langcodes

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
from sqlalchemy.sql import func

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
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever, VectorStore

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
FUZZY_SEARCH_THRESHOLD_APP = getattr(globals(), 'FUZZY_SEARCH_THRESHOLD', 30) # Default to 80 if not from

try:
    # ... (your existing local imports for AIProvider, database, config) ...
    from config import ENABLE_STELLA_ICARUS_HOOKS # Ensure this is imported
    from stella_icarus_utils import StellaIcarusHookManager # <<< NEW IMPORT
except ImportError as e:
    # ... (your existing ImportError handling) ...
    StellaIcarusHookManager = None # Define as None if import fails
    ENABLE_STELLA_ICARUS_HOOKS = False # Default to false if config itself fails
    # ...


# === Global Semaphores and Concurrency Control ===
# Default to a small number, can be overridden by environment variable if desired
_default_max_bg_tasks = 10 #parallel???
try:
    MAX_CONCURRENT_BACKGROUND_GENERATE_TASKS = int(os.getenv("MAX_CONCURRENT_BACKGROUND_GENERATE_TASKS", _default_max_bg_tasks))
    if MAX_CONCURRENT_BACKGROUND_GENERATE_TASKS <= 0:
        logger.warning(f"MAX_CONCURRENT_BACKGROUND_GENERATE_TASKS was <= 0, defaulting to {_default_max_bg_tasks}")
        MAX_CONCURRENT_BACKGROUND_GENERATE_TASKS = _default_max_bg_tasks
except ValueError:
    logger.warning(f"Invalid value for MAX_CONCURRENT_BACKGROUND_GENERATE_TASKS env var, defaulting to {_default_max_bg_tasks}")
    MAX_CONCURRENT_BACKGROUND_GENERATE_TASKS = _default_max_bg_tasks

logger.info(f"🚦 Max concurrent background generate tasks: {MAX_CONCURRENT_BACKGROUND_GENERATE_TASKS}")
background_generate_task_semaphore = threading.Semaphore(MAX_CONCURRENT_BACKGROUND_GENERATE_TASKS)
# === End Global Semaphores ===

# --- Local Imports with Error Handling ---
try:
    from ai_provider import AIProvider, ai_provider_instance as global_ai_provider_ref
    # Import database components needed in app.py
    
    from database import (
        init_db, queue_interaction_for_batch_logging, add_interaction, get_recent_interactions, # <<< REMOVED get_db
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
    Main loop for self-reflection. Periodically processes eligible interactions.
    If no new interactions, may proactively re-queue an old one for re-reflection.
    """
    global ai_provider, ai_chat  # Assuming these are global instances
    thread_name = threading.current_thread().name
    logger.info(f"✅ {thread_name} started (Continuous Reflection Logic).")

    if not ai_provider or not ai_chat:
        logger.error(f"🛑 {thread_name}: AIProvider or AIChat not initialized. Cannot run reflection.")
        return

    logger.info(
        f"{thread_name}: Config - BatchSize={REFLECTION_BATCH_SIZE}, IdleWait={IDLE_WAIT_SECONDS}s, ActivePause={ACTIVE_CYCLE_PAUSE_SECONDS}s")
    logger.info(
        f"{thread_name}: Config - ProactiveReReflect={ENABLE_PROACTIVE_RE_REFLECTION}, Chance={PROACTIVE_RE_REFLECTION_CHANCE}, MinAgeDays={MIN_AGE_FOR_RE_REFLECTION_DAYS}")
    logger.info(f"{thread_name}: Eligible Input Types for new reflection: {REFLECTION_ELIGIBLE_INPUT_TYPES}")

    while not _reflector_stop_event.is_set():
        cycle_start_time = time.monotonic()
        total_processed_this_active_cycle = 0
        work_found_in_this_cycle = False

        logger.info(f"🤔 {thread_name}: Starting ACTIVE reflection cycle...")

        if not _reflector_lock.acquire(blocking=False):
            logger.warning(f"{thread_name}: Previous reflection cycle lock still held? Skipping this cycle attempt.")
            _reflector_stop_event.wait(timeout=IDLE_WAIT_SECONDS)  # Wait before trying again
            continue

        db: Optional[Session] = None
        try:
            # Wait if server is busy (main API requests are active)
            was_busy_waiting = False
            while server_is_busy_event.is_set():
                if not was_busy_waiting:
                    logger.info(f"🚦 {thread_name}: Server busy, pausing reflection start...")
                    was_busy_waiting = True
                if _reflector_stop_event.is_set(): break
                if _reflector_stop_event.wait(timeout=0.5): break
            if was_busy_waiting: logger.info(f"🟢 {thread_name}: Server free, resuming reflection cycle.")
            if _reflector_stop_event.is_set(): break

            db = SessionLocal()  # type: ignore
            if not db: raise RuntimeError("Failed to create DB session for reflection cycle.")
            logger.trace(f"{thread_name}: DB Session created for active cycle.")

            # --- Inner Loop: Process NEW interactions needing reflection ---
            while not _reflector_stop_event.is_set():
                batch_processed_count_this_inner_loop = 0
                interactions_to_reflect: List[Interaction] = []
                try:
                    query = db.query(Interaction).filter(
                        Interaction.reflection_completed == False,
                        Interaction.mode == 'chat',
                        Interaction.input_type.in_(REFLECTION_ELIGIBLE_INPUT_TYPES)
                    ).order_by(Interaction.timestamp.asc()).limit(REFLECTION_BATCH_SIZE)
                    interactions_to_reflect = query.all()
                except Exception as query_err:
                    logger.error(f"{thread_name}: Error querying new interactions for reflection: {query_err}")
                    _reflector_stop_event.wait(timeout=5)  # Wait before breaking inner loop
                    break

                if not interactions_to_reflect:
                    logger.debug(f"{thread_name}: No new eligible interactions found in this batch/query.")
                    break  # Exit the inner batch-processing loop

                work_found_in_this_cycle = True
                logger.info(
                    f"{thread_name}: Found {len(interactions_to_reflect)} new interaction(s) for reflection. Processing batch...")

                for interaction_obj in interactions_to_reflect:  # Renamed to avoid conflict
                    if _reflector_stop_event.is_set(): logger.info(f"{thread_name}: Stop signal during batch."); break
                    if server_is_busy_event.is_set(): logger.info(
                        f"{thread_name}: Server became busy, pausing batch."); break

                    original_input_text = interaction_obj.user_input or "[Original input missing]"
                    original_interaction_id = interaction_obj.id
                    original_input_type_text = interaction_obj.input_type

                    logger.info(
                        f"{thread_name}: --> Triggering reflection for Interaction ID {original_interaction_id} (Type: {original_input_type_text}) - Input: '{original_input_text[:60]}...'")

                    reflection_session_id_for_bg = f"reflection_{original_interaction_id}_{str(uuid.uuid4())[:4]}"
                    task_launched_successfully = False
                    try:
                        # background_generate is async, run it and wait for it to complete
                        # The db session is passed to it.
                        asyncio.run(  # This blocks the current thread until the async function completes
                            ai_chat.background_generate(  # type: ignore
                                db=db,
                                user_input=original_input_text,  # This is the content to reflect upon
                                session_id=reflection_session_id_for_bg,
                                classification="chat_complex",  # Force complex for reflection
                                image_b64=None,
                                update_interaction_id=original_interaction_id  # Key for reflection
                            )
                        )
                        # If asyncio.run completes without exception, assume the core logic of background_generate
                        # (including its own error handling and DB updates for the original interaction) finished.
                        logger.info(
                            f"{thread_name}: --> Background reflection task for ID {original_interaction_id} completed its execution path.")
                        task_launched_successfully = True  # Indicates background_generate ran
                        batch_processed_count_this_inner_loop += 1
                    except Exception as trigger_err:
                        logger.error(
                            f"{thread_name}: Failed to run/complete background_generate for reflection on ID {original_interaction_id}: {trigger_err}")
                        logger.exception(f"{thread_name} Background Generate Trigger/Run Traceback:")
                        # If background_generate itself fails, the original interaction's reflection_completed
                        # status should ideally remain False due to error handling within background_generate.
                        # We just log here and continue to the next item in batch.

                    # No explicit marking of original_interaction.reflection_completed = True here.
                    # That is now handled *inside* background_generate's finally block.

                    if not _reflector_stop_event.is_set() and ACTIVE_CYCLE_PAUSE_SECONDS > 0:
                        _reflector_stop_event.wait(timeout=ACTIVE_CYCLE_PAUSE_SECONDS)

                total_processed_this_active_cycle += batch_processed_count_this_inner_loop
                logger.info(
                    f"{thread_name}: Finished processing batch ({batch_processed_count_this_inner_loop} items). Total this cycle: {total_processed_this_active_cycle}.")
                if _reflector_stop_event.is_set() or server_is_busy_event.is_set(): break
                # --- End of Inner Loop for NEW interactions ---

            # --- "Gabut State" - Proactive Re-Reflection Logic ---
            if not work_found_in_this_cycle and ENABLE_PROACTIVE_RE_REFLECTION and random.random() < PROACTIVE_RE_REFLECTION_CHANCE:
                logger.info(f"{thread_name}: Gabut State - No new work. Attempting proactive re-reflection.")
                re_reflected_interaction_id: Optional[int] = None
                try:
                    time_threshold = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
                        days=MIN_AGE_FOR_RE_REFLECTION_DAYS)

                    # For SQLite ORDER BY RANDOM()
                    order_by_clause = func.random() if db.bind and db.bind.dialect.name == 'sqlite' else Interaction.timestamp
                    # For other DBs, random might be different or less efficient. Fallback to oldest.
                    if db.bind and db.bind.dialect.name != 'sqlite':
                        logger.warning(
                            f"{thread_name}: RANDOM() for re-reflection query might be inefficient for {db.bind.dialect.name}. Consider dialect-specific random or oldest.")

                    candidate_for_re_reflection = db.query(Interaction).filter(
                        Interaction.reflection_completed == True,
                        Interaction.mode == 'chat',
                        Interaction.input_type.in_(REFLECTION_ELIGIBLE_INPUT_TYPES),
                        Interaction.timestamp < time_threshold
                    ).order_by(order_by_clause).limit(1).first()

                    if candidate_for_re_reflection:
                        logger.info(
                            f"{thread_name}: Selected Interaction ID {candidate_for_re_reflection.id} (Type: {candidate_for_re_reflection.input_type}) for proactive re-reflection.")

                        candidate_for_re_reflection.reflection_completed = False
                        note = f"\n\n[System Re-queued for Reflection ({datetime.datetime.now(datetime.timezone.utc).isoformat()}) due to Gabut State]"
                        current_resp = candidate_for_re_reflection.llm_response or ""
                        candidate_for_re_reflection.llm_response = (current_resp + note)[
                                                                   :getattr(Interaction.llm_response.type, 'length',
                                                                            4000)]
                        # last_modified_db will be updated by SQLAlchemy's onupdate if configured, or manually:
                        # candidate_for_re_reflection.last_modified_db = datetime.datetime.now(datetime.timezone.utc)

                        db.commit()
                        re_reflected_interaction_id = candidate_for_re_reflection.id
                        work_found_in_this_cycle = True  # Signal that work was "found" for next cycle logic
                        logger.info(
                            f"{thread_name}: Interaction ID {re_reflected_interaction_id} re-queued for reflection.")
                    else:
                        logger.info(f"{thread_name}: No suitable old interactions found for proactive re-reflection.")
                except Exception as e_re_reflect:
                    logger.error(f"{thread_name}: Error during proactive re-reflection: {e_re_reflect}")
                    if db: db.rollback()
            # --- END "Gabut State" Logic ---

        except RuntimeError as rt_err:  # Catch DB session creation failure
            logger.error(f"💥 {thread_name}: Runtime error in reflection cycle (likely DB session): {rt_err}")
        except Exception as cycle_err:
            logger.error(f"💥 {thread_name}: Unhandled error during active reflection cycle: {cycle_err}")
            logger.exception(f"{thread_name} Cycle Traceback:")
            if db: db.rollback()
        finally:
            if db:
                try:
                    db.close(); logger.debug(f"{thread_name}: DB session closed for cycle.")
                except Exception as close_err:
                    logger.error(f"{thread_name}: Error closing DB session: {close_err}")

            try:
                _reflector_lock.release(); logger.trace(f"{thread_name}: Released cycle lock.")
            except (threading.ThreadError, RuntimeError) as lk_err:
                logger.warning(f"{thread_name}: Lock release issue: {lk_err}")

            cycle_duration = time.monotonic() - cycle_start_time
            logger.info(
                f"{thread_name}: ACTIVE reflection cycle finished in {cycle_duration:.2f}s. Processed: {total_processed_this_active_cycle} new interaction(s).")

        # --- Wait logic before next full cycle check ---
        wait_time_seconds = IDLE_WAIT_SECONDS if not work_found_in_this_cycle else ACTIVE_CYCLE_PAUSE_SECONDS
        if wait_time_seconds > 0 and not _reflector_stop_event.is_set():
            logger.debug(f"{thread_name}: Waiting {wait_time_seconds:.2f} seconds before next cycle check...")
            stopped_during_wait = _reflector_stop_event.wait(timeout=wait_time_seconds)
            if stopped_during_wait: logger.info(f"{thread_name}: Stop signal received during idle wait."); break
        elif _reflector_stop_event.is_set():  # If stop set and no wait needed
            logger.info(f"{thread_name}: Stop signal detected after active cycle.")
            break

    logger.info(f"🛑 {thread_name}: Exiting self-reflection loop.")


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

# json Fixer
def _extract_json_candidate_string(raw_llm_text: str, log_prefix: str = "JSONExtract") -> Optional[str]:
    """
    Robustly extracts a potential JSON string from raw LLM text.
    Handles <think> tags, markdown code blocks, and finding outermost braces.
    """
    if not raw_llm_text or not isinstance(raw_llm_text, str):
        return None

    logger.trace(f"{log_prefix}: Starting JSON candidate extraction from raw text (len {len(raw_llm_text)}).")

    # 1. Remove <think> tags first
    text_after_think_removal = re.sub(r'<think>.*?</think>', '', raw_llm_text, flags=re.DOTALL | re.IGNORECASE).strip()
    if not text_after_think_removal:
        logger.trace(f"{log_prefix}: Text empty after <think> removal.")
        return None

    # 2. Remove common LLM preambles/postambles around the JSON content
    cleaned_text = text_after_think_removal
    # Remove common assistant start patterns, case insensitive
    cleaned_text = re.sub(r"^\s*(assistant\s*\n?)?(<\|im_start\|>\s*(system|assistant)\s*\n?)?", "", cleaned_text,
                          flags=re.IGNORECASE).lstrip()
    # Remove trailing ChatML end token
    if CHATML_END_TOKEN and cleaned_text.endswith(CHATML_END_TOKEN):  # CHATML_END_TOKEN from config
        cleaned_text = cleaned_text[:-len(CHATML_END_TOKEN)].strip()

    # 3. Look for JSON within markdown code blocks (```json ... ```)
    json_markdown_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", cleaned_text, re.DOTALL)
    if json_markdown_match:
        extracted_str = json_markdown_match.group(1).strip()
        logger.trace(f"{log_prefix}: Extracted JSON from markdown block: '{extracted_str[:100]}...'")
        return extracted_str

    # 4. If not in markdown, find the first '{' and last '}' that likely enclose the main JSON object
    # This helps strip extraneous text before the first '{' or after the last '}'.
    first_brace = cleaned_text.find('{')
    last_brace = cleaned_text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        extracted_str = cleaned_text[first_brace: last_brace + 1].strip()
        logger.trace(f"{log_prefix}: Extracted JSON using outermost braces: '{extracted_str[:100]}...'")
        return extracted_str

    # 5. If still no clear structure, but text might be just the JSON (e.g. after LLM reformat)
    if cleaned_text.startswith("{") and cleaned_text.endswith("}"):
        logger.trace(f"{log_prefix}: Assuming cleaned text itself is the JSON candidate: '{cleaned_text[:100]}...'")
        return cleaned_text

    logger.warning(
        f"{log_prefix}: No clear JSON structure found after extraction attempts. Raw (after think): '{text_after_think_removal[:100]}...'")
    return None  # Return None if no candidate found


def _programmatic_json_parse_and_fix(
        json_candidate_str: str,
        max_fix_attempts: int = 3,  # How many times to try fixing based on errors
        log_prefix: str = "JSONFixParse"
) -> Optional[Union[Dict, List]]:
    """
    Attempts to parse a JSON string, applying a limited set of programmatic fixes
    if initial parsing fails. Iterates up to max_fix_attempts.
    Returns the parsed Python object (dict/list) or None.
    """
    if not json_candidate_str or not isinstance(json_candidate_str, str):
        return None

    current_text_to_parse = json_candidate_str
    json_parser = JsonOutputParser()  # Or just use json.loads directly

    for attempt in range(max_fix_attempts):
        logger.debug(
            f"{log_prefix}: Parse/fix attempt {attempt + 1}/{max_fix_attempts}. Current text snippet: '{current_text_to_parse[:100]}...'")
        try:
            # Standard parse attempt
            parsed_object = json_parser.parse(
                current_text_to_parse)  # Langchain's parser can sometimes fix minor issues
            # Or use: parsed_object = json.loads(current_text_to_parse)
            logger.debug(f"{log_prefix}: Successfully parsed JSON on attempt {attempt + 1}.")
            return parsed_object
        except (json.JSONDecodeError, OutputParserException) as e:
            logger.warning(f"{log_prefix}: Attempt {attempt + 1} parse failed: {type(e).__name__} - {str(e)[:100]}")

            if attempt == max_fix_attempts - 1:  # Last attempt failed, don't try to fix further
                logger.error(f"{log_prefix}: Max fix attempts reached. Could not parse after error: {e}")
                break

            # --- Apply limited programmatic fixes based on common LLM issues ---
            text_before_fixes = current_text_to_parse

            # Fix 1: Remove trailing commas (before closing brackets/braces)
            # This is a common issue. Python's json.loads tolerates ONE, but stricter parsers/standards don't.
            current_text_to_parse = re.sub(r",\s*([\}\]])", r"\1", current_text_to_parse)
            if current_text_to_parse != text_before_fixes:
                logger.debug(f"{log_prefix}: Applied trailing comma fix.")

            # Fix 2: Attempt to add quotes to unquoted keys (very heuristic and simple cases)
            # Looks for { or , followed by whitespace, then word chars (key), then whitespace, then :
            # This is risky and might not work for all cases or could corrupt complex keys.
            # Example: { key : "value"} -> { "key" : "value"}
            # Example: [{"key" : "value", next_key : "next_value"}] -> [{"key" : "value", "next_key" : "next_value"}]
            # Only apply if the error message suggests unquoted keys, if possible (hard to check generically here)
            if "Expecting property name enclosed in double quotes" in str(e):
                text_after_key_quote_fix = re.sub(r"([{,\s])(\w+)(\s*:)", r'\1"\2"\3', current_text_to_parse)
                if text_after_key_quote_fix != current_text_to_parse:
                    logger.debug(f"{log_prefix}: Applied heuristic key quoting fix.")
                    current_text_to_parse = text_after_key_quote_fix

            # Fix 3: Remove JavaScript-style comments
            text_after_comment_removal = re.sub(r"//.*?\n", "\n", current_text_to_parse, flags=re.MULTILINE)
            text_after_comment_removal = re.sub(r"/\*.*?\*/", "", text_after_comment_removal, flags=re.DOTALL)
            if text_after_comment_removal != current_text_to_parse:
                logger.debug(f"{log_prefix}: Applied JS comment removal fix.")
                current_text_to_parse = text_after_comment_removal.strip()

            # If no fixes changed the string, and it's not the last attempt,
            # it means our simple fixes aren't working for this error.
            if current_text_to_parse == text_before_fixes and attempt < max_fix_attempts - 1:
                logger.warning(
                    f"{log_prefix}: Programmatic fixes did not alter the string for error '{str(e)[:50]}...'. Further fixes might be needed or error is complex.")
                # For more complex errors, one might analyze e.pos, e.msg here, but it's very hard.
                # We are relying on the LLM re-request to do most heavy lifting.
                break  # Break from fix attempts if no change, let outer loop handle if it's the LLM re-request loop

    logger.error(
        f"{log_prefix}: Failed to parse JSON after all programmatic fix attempts. Original candidate: '{json_candidate_str[:200]}...'")
    return None


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

        # --- NEW: Initialize StellaIcarusHookManager ---
        # --- MODIFIED: Initialize StellaIcarusHookManager ---
        self.stella_icarus_manager: Optional[StellaIcarusHookManager] = None
        if ENABLE_STELLA_ICARUS_HOOKS and StellaIcarusHookManager is not None:  # Check if class was imported
            try:
                self.stella_icarus_manager = StellaIcarusHookManager()
                if self.stella_icarus_manager.hook_load_errors:
                    logger.warning("AIChat Init: StellaIcarusHookManager loaded with some errors.")
                elif not self.stella_icarus_manager.hooks:
                    logger.info("AIChat Init: StellaIcarusHookManager loaded, but no hooks found/active.")
                else:
                    logger.success("AIChat Init: StellaIcarusHookManager loaded successfully with hooks.")
            except Exception as e_sihm_init:
                logger.error(f"AIChat Init: Failed to initialize StellaIcarusHookManager: {e_sihm_init}")
                self.stella_icarus_manager = None
        elif not StellaIcarusHookManager:
            logger.error("AIChat Init: StellaIcarusHookManager class not available (import failed?). Hooks disabled.")
        else:  # ENABLE_STELLA_ICARUS_HOOKS is False
            logger.info("AIChat Init: StellaIcarusHooks are disabled by configuration.")
        # --- END MODIFIED ---
        # --- END NEW ---

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

    class _CustomVectorSearchRetriever(VectorStoreRetriever):
        vectorstore: VectorStore
        search_type: str = "similarity"
        search_kwargs: Dict[str, Any]
        vector_to_search: List[float]

        def _get_relevant_documents(self, query: str, *, run_manager: Any) -> List[Document]:
            if not self.vectorstore: raise ValueError("Vectorstore not set on _CustomVectorSearchRetriever")
            k_val = self.search_kwargs.get("k", 4)
            return self.vectorstore.similarity_search_by_vector(embedding=self.vector_to_search, k=k_val)

        async def _aget_relevant_documents(self, query: str, *, run_manager: Any) -> List[Document]:
            if not self.vectorstore: raise ValueError("Vectorstore not set on _CustomVectorSearchRetriever")
            k_val = self.search_kwargs.get("k", 4)
            # Assuming similarity_search_by_vector is synchronous for Chroma
            return await asyncio.to_thread(
                self.vectorstore.similarity_search_by_vector,
                embedding=self.vector_to_search,
                k=k_val
            )

    def _get_rag_retriever(self, db: Session, user_input_for_rag_query: str, priority: int = ELP0) -> Tuple[
        Optional[VectorStoreRetriever], Optional[VectorStoreRetriever], Optional[VectorStoreRetriever], str
    ]:
        # ... (log_prefix, logger.critical lines remain the same) ...
        log_prefix = f"RAGRetriever|ELP{priority}|{self.current_session_id or 'NoSession'}"
        logger.critical(
            f"@@@ ENTERING _get_rag_retriever for session {self.current_session_id}, query: '{user_input_for_rag_query[:30]}...', priority ELP{priority} @@@")

        url_retriever: Optional[VectorStoreRetriever] = None
        session_history_retriever: Optional[VectorStoreRetriever] = None
        reflection_chunks_retriever: Optional[VectorStoreRetriever] = None
        session_history_ids_str: str = ""  # For logging which specific session IDs were part of the on-the-fly Chroma

        self.vectorstore_history: Optional[Chroma] = None  # Temp Chroma for current session's history
        session_history_ids_set = set()  # To track IDs added to temp Chroma store
        rag_query_vector: Optional[List[float]] = None

        # --- Store for combined RAG results (both vector and fuzzy) ---
        # We will add Langchain Document objects here, converting fuzzy SQL results to Documents.
        all_retrieved_history_docs: List[Document] = []
        all_retrieved_reflection_docs: List[Document] = []
        # URL retriever results are handled separately as they come from a persistent store

        try:
            # --- Step 0: Pre-embed the main RAG query ---
            if user_input_for_rag_query and self.provider and self.provider.embeddings:
                # ... (existing pre-embedding logic using priority - UNCHANGED) ...
                logger.debug(
                    f"{log_prefix} Pre-embedding main RAG query via _embed_texts with priority ELP{priority}: '{user_input_for_rag_query[:50]}...'")
                if hasattr(self.provider.embeddings, '_embed_texts') and callable(
                        getattr(self.provider.embeddings, '_embed_texts')):
                    embedding_result_list = self.provider.embeddings._embed_texts([user_input_for_rag_query],
                                                                                  priority=priority)  # type: ignore
                    if embedding_result_list and len(embedding_result_list) > 0: rag_query_vector = \
                    embedding_result_list[0]
                else:
                    logger.warning(
                        f"{log_prefix} Embeddings object missing custom '_embed_texts'. Calling public 'embed_query' (will use its default priority).")
                    rag_query_vector = self.provider.embeddings.embed_query(user_input_for_rag_query)
                if rag_query_vector:
                    logger.debug(f"{log_prefix} Main RAG query pre-embedded successfully.")
                else:
                    logger.error(f"{log_prefix} Main RAG query embedding resulted in None or empty vector.")

            # --- Step 1: URL Retriever (Vector Search Only - No SQL fallback for URLs defined here) ---
            # ... (existing URL retriever logic - UNCHANGED) ...
            logger.debug(f"{log_prefix} Step 1: URL Retriever processing...")
            if hasattr(self, 'vectorstore_url') and self.vectorstore_url and isinstance(self.vectorstore_url, Chroma):
                if rag_query_vector:
                    try:
                        k_url = RAG_URL_COUNT
                        url_retriever = self._CustomVectorSearchRetriever(vectorstore=self.vectorstore_url,
                                                                          search_kwargs={"k": k_url},
                                                                          vector_to_search=rag_query_vector)
                    except Exception as e_url_ret:
                        logger.error(f"{log_prefix} Failed custom URL retriever creation: {e_url_ret}")
                else:
                    logger.warning(f"{log_prefix} Skipping URL retriever: RAG query not embedded.")
            else:
                logger.trace(f"{log_prefix} No URL vector store or not Chroma type.")
            logger.debug(f"{log_prefix} Step 1 complete. url_retriever: {'Set' if url_retriever else 'None'}")

            # --- Step 2: Session Chat History Retriever (Vector + Fuzzy Fallback) ---
            logger.debug(f"{log_prefix} Step 2: Session Chat History Retriever processing (Vector + Fuzzy Fallback)...")
            # Get raw chat interactions for both vector store creation AND fuzzy search
            chat_interactions_for_processing = get_recent_interactions(
                db, RAG_HISTORY_COUNT * 4, self.current_session_id, "chat", False  # Get more for fuzzy
            )
            chat_interactions_for_processing.reverse()  # Oldest first for prompt context flow

            # --- 2a. Vector Search for Session History ---
            if chat_interactions_for_processing and self.provider and self.provider.embeddings and rag_query_vector:
                session_texts_to_embed_for_vs: List[str] = []
                temp_interaction_mapping_for_vs: Dict[
                    int, Interaction] = {}  # Map index in list to original interaction

                for idx, interaction in enumerate(chat_interactions_for_processing):
                    text_content: Optional[str] = None
                    if interaction.user_input and interaction.input_type == 'text':
                        text_content = f"User (session): {interaction.user_input}"
                    elif interaction.llm_response and interaction.input_type == 'llm_response' and len(
                            interaction.llm_response.strip()) > 10:
                        text_content = f"AI (session): {interaction.llm_response}"

                    if text_content and interaction.id not in session_history_ids_set:
                        session_texts_to_embed_for_vs.append(text_content)
                        temp_interaction_mapping_for_vs[len(session_texts_to_embed_for_vs) - 1] = interaction
                        session_history_ids_set.add(interaction.id)  # Track for logging

                if session_texts_to_embed_for_vs:
                    logger.debug(
                        f"{log_prefix} Session History: Embedding {len(session_texts_to_embed_for_vs)} texts for on-the-fly VS (Priority ELP{priority})...")
                    embedded_session_vectors_vs: Optional[List[List[float]]] = None
                    if hasattr(self.provider.embeddings, '_embed_texts') and callable(
                            getattr(self.provider.embeddings, '_embed_texts')):
                        embedded_session_vectors_vs = self.provider.embeddings._embed_texts(
                            session_texts_to_embed_for_vs, priority=priority)  # type: ignore
                    else:  # Fallback
                        embedded_session_vectors_vs = self.provider.embeddings.embed_documents(
                            session_texts_to_embed_for_vs)

                    if embedded_session_vectors_vs and len(embedded_session_vectors_vs) == len(
                            session_texts_to_embed_for_vs):
                        try:
                            self.vectorstore_history = Chroma(collection_name=f"sess_hist_temp_{uuid.uuid4().hex[:6]}",
                                                              embedding_function=self.provider.embeddings)
                            self.vectorstore_history.add_embeddings(texts=session_texts_to_embed_for_vs,
                                                                    embeddings=embedded_session_vectors_vs)

                            k_sess_vec = RAG_HISTORY_COUNT // 2 if RAG_HISTORY_COUNT > 1 else RAG_HISTORY_COUNT  # Num vector results
                            if k_sess_vec > 0:
                                # Vector search using the custom retriever
                                temp_retriever = self._CustomVectorSearchRetriever(vectorstore=self.vectorstore_history,
                                                                                   search_kwargs={"k": k_sess_vec},
                                                                                   vector_to_search=rag_query_vector)
                                vector_results_session_docs = temp_retriever.invoke(
                                    user_input_for_rag_query)  # invoke() gets docs
                                all_retrieved_history_docs.extend(vector_results_session_docs or [])
                                logger.info(
                                    f"{log_prefix} Session History: Vector search found {len(vector_results_session_docs or [])} docs.")
                        except Exception as e_sess_vs:
                            logger.error(
                                f"{log_prefix} Session History: Error building/querying on-the-fly Chroma: {e_sess_vs}")
                    else:
                        logger.error(
                            f"{log_prefix} Session History: Embedding failed or vector count mismatch for on-the-fly VS.")
            else:
                logger.debug(
                    f"{log_prefix} Session History: Skipping vector search (no interactions, provider/embeddings, or query vector).")

            # --- 2b. Fuzzy Search Fallback for Session History ---
            # Perform if vector search found fewer than desired results or if FUZZY_ALWAYS_AUGMENT is True
            desired_total_history_results = RAG_HISTORY_COUNT
            if FUZZY_AVAILABLE and (
                    len(all_retrieved_history_docs) < desired_total_history_results // 2 or RAG_HISTORY_COUNT == 0):  # Example condition
                logger.info(
                    f"{log_prefix} Session History: Vector search results ({len(all_retrieved_history_docs)}) insufficient or RAG_HISTORY_COUNT is 0. Attempting fuzzy search...")
                fuzzy_matches_session: List[Tuple[Interaction, int]] = []
                processed_for_fuzzy_ids = set(
                    doc.metadata.get("interaction_id") for doc in all_retrieved_history_docs if
                    doc.metadata)  # Avoid re-adding vector results

                for interaction in chat_interactions_for_processing:  # Iterate over the already fetched interactions
                    if interaction.id in processed_for_fuzzy_ids: continue  # Skip if already got via vector search

                    text_to_match_on = ""
                    if interaction.user_input and interaction.input_type == 'text':
                        text_to_match_on = interaction.user_input
                    elif interaction.llm_response and interaction.input_type == 'llm_response':
                        text_to_match_on = interaction.llm_response

                    if text_to_match_on.strip():
                        score = fuzz.partial_ratio(user_input_for_rag_query.lower(), text_to_match_on.lower())
                        if score >= FUZZY_SEARCH_THRESHOLD_APP:
                            fuzzy_matches_session.append((interaction, score))

                if fuzzy_matches_session:
                    fuzzy_matches_session.sort(key=lambda x: x[1], reverse=True)  # Sort by score
                    # How many more do we need to reach desired_total_history_results?
                    needed_fuzzy_count = max(0, desired_total_history_results - len(all_retrieved_history_docs))
                    for interaction, score in fuzzy_matches_session[:needed_fuzzy_count]:
                        content = interaction.user_input if interaction.input_type == 'text' else interaction.llm_response
                        doc = Document(
                            page_content=f"{'User' if interaction.input_type == 'text' else 'AI'} (fuzzy score {score}): {content}",
                            metadata={"source": "session_history_fuzzy", "interaction_id": interaction.id,
                                      "timestamp": str(interaction.timestamp)}
                        )
                        all_retrieved_history_docs.append(doc)
                        processed_for_fuzzy_ids.add(interaction.id)
                    logger.info(
                        f"{log_prefix} Session History: Added {len(fuzzy_matches_session[:needed_fuzzy_count])} docs via fuzzy search.")

            # Create the final session_history_retriever based on *all_retrieved_history_docs*
            # This is a bit of a hack as retriever usually queries. Here, we pre-populate.
            if all_retrieved_history_docs:
                # Create a temporary Chroma store *just for these combined results* to make it a retriever
                if self.provider and self.provider.embeddings:
                    try:
                        # Extract texts and metadatas for the new temp store
                        texts_for_final_hist_store = [doc.page_content for doc in all_retrieved_history_docs]
                        metadatas_for_final_hist_store = [doc.metadata for doc in all_retrieved_history_docs]
                        # Embed these specific texts
                        final_hist_embeddings = self.provider.embeddings._embed_texts(texts_for_final_hist_store,
                                                                                      priority=priority) if hasattr(
                            self.provider.embeddings, '_embed_texts') else self.provider.embeddings.embed_documents(
                            texts_for_final_hist_store)

                        if final_hist_embeddings and len(final_hist_embeddings) == len(texts_for_final_hist_store):
                            temp_final_hist_vs = Chroma(collection_name=f"final_hist_ret_{uuid.uuid4().hex[:6]}",
                                                        embedding_function=self.provider.embeddings)
                            temp_final_hist_vs.add_embeddings(texts=texts_for_final_hist_store,
                                                              embeddings=final_hist_embeddings,
                                                              metadatas=metadatas_for_final_hist_store)
                            # This retriever will essentially just return all docs or a subset if k is small
                            session_history_retriever = temp_final_hist_vs.as_retriever(
                                search_kwargs={"k": desired_total_history_results})
                            logger.debug(
                                f"{log_prefix} Session History: Final retriever created with {len(all_retrieved_history_docs)} combined docs.")
                        else:
                            logger.error(
                                f"{log_prefix} Session History: Failed to embed docs for final combined retriever.")
                    except Exception as e_final_hist_vs:
                        logger.error(
                            f"{log_prefix} Session History: Error creating final combined retriever: {e_final_hist_vs}")
            logger.debug(
                f"{log_prefix} Step 2 (Session History) complete. Total retrieved: {len(all_retrieved_history_docs)} docs. Retriever set: {session_history_retriever is not None}")

            # --- Step 3: Global Reflection Chunks Retriever (Vector + Fuzzy Fallback) ---
            logger.debug(
                f"{log_prefix} Step 3: Global Reflection Chunks Retriever processing (Vector + Fuzzy Fallback)...")
            active_refl_vs_instance = get_global_reflection_vectorstore()  # Get the global store

            # --- 3a. Vector Search for Reflections ---
            if active_refl_vs_instance and isinstance(active_refl_vs_instance, Chroma) and rag_query_vector:
                try:
                    k_refl_vec = RAG_HISTORY_COUNT // 2 if RAG_HISTORY_COUNT > 1 else RAG_HISTORY_COUNT
                    if k_refl_vec > 0:
                        temp_refl_retriever = self._CustomVectorSearchRetriever(vectorstore=active_refl_vs_instance,
                                                                                search_kwargs={"k": k_refl_vec},
                                                                                vector_to_search=rag_query_vector)
                        vector_results_reflection_docs = temp_refl_retriever.invoke(user_input_for_rag_query)
                        all_retrieved_reflection_docs.extend(vector_results_reflection_docs or [])
                        logger.info(
                            f"{log_prefix} Reflections: Vector search found {len(vector_results_reflection_docs or [])} docs.")
                except Exception as e_refl_vs:
                    logger.error(f"{log_prefix} Reflections: Error querying global reflection VS: {e_refl_vs}")
            else:
                logger.debug(
                    f"{log_prefix} Reflections: Skipping vector search (VS not available/Chroma or no query vector).")

            # --- 3b. Fuzzy Search Fallback for Reflections ---
            desired_total_reflection_results = RAG_HISTORY_COUNT
            if FUZZY_AVAILABLE and (
                    len(all_retrieved_reflection_docs) < desired_total_reflection_results // 2 or RAG_HISTORY_COUNT == 0):
                logger.info(
                    f"{log_prefix} Reflections: Vector search results ({len(all_retrieved_reflection_docs)}) insufficient or RAG_HISTORY_COUNT is 0. Attempting fuzzy search on reflection_result interactions...")
                # Query SQL for 'reflection_result' type interactions
                # Limiting to recent ones for performance, or those not already retrieved by vector.
                # This part is more complex as fuzzy search needs text, and Interaction objects don't directly become Langchain Documents.
                # For simplicity, let's assume we can query a limited set of recent reflection_result Interaction objects.
                recent_reflection_sql_interactions = db.query(Interaction).filter(
                    Interaction.input_type == 'reflection_result',
                    Interaction.llm_response.isnot(None)  # Ensure there's content
                ).order_by(desc(Interaction.timestamp)).limit(RAG_HISTORY_COUNT * 5).all()  # Get more candidates

                fuzzy_matches_reflection: List[Tuple[Interaction, int]] = []
                processed_refl_fuzzy_ids = set(
                    doc.metadata.get("source_interaction_id") for doc in all_retrieved_reflection_docs if doc.metadata)

                for interaction in recent_reflection_sql_interactions:
                    if interaction.id in processed_refl_fuzzy_ids: continue
                    if interaction.llm_response:
                        score = fuzz.partial_ratio(user_input_for_rag_query.lower(), interaction.llm_response.lower())
                        if score >= FUZZY_SEARCH_THRESHOLD_APP:
                            fuzzy_matches_reflection.append((interaction, score))

                if fuzzy_matches_reflection:
                    fuzzy_matches_reflection.sort(key=lambda x: x[1], reverse=True)
                    needed_fuzzy_refl_count = max(0,
                                                  desired_total_reflection_results - len(all_retrieved_reflection_docs))
                    for interaction, score in fuzzy_matches_reflection[:needed_fuzzy_refl_count]:
                        doc = Document(
                            page_content=f"Reflection (fuzzy score {score}, ID {interaction.id}): {interaction.llm_response}",
                            metadata={"source": "reflection_fuzzy", "source_interaction_id": interaction.id,
                                      "timestamp": str(interaction.timestamp)}
                        )
                        all_retrieved_reflection_docs.append(doc)
                        processed_refl_fuzzy_ids.add(interaction.id)
                    logger.info(
                        f"{log_prefix} Reflections: Added {len(fuzzy_matches_reflection[:needed_fuzzy_refl_count])} docs via fuzzy search.")

            # Create the final reflection_chunks_retriever (similar to session history)
            if all_retrieved_reflection_docs:
                if self.provider and self.provider.embeddings:
                    try:
                        texts_for_final_refl_store = [doc.page_content for doc in all_retrieved_reflection_docs]
                        metadatas_for_final_refl_store = [doc.metadata for doc in all_retrieved_reflection_docs]
                        final_refl_embeddings = self.provider.embeddings._embed_texts(texts_for_final_refl_store,
                                                                                      priority=priority) if hasattr(
                            self.provider.embeddings, '_embed_texts') else self.provider.embeddings.embed_documents(
                            texts_for_final_refl_store)

                        if final_refl_embeddings and len(final_refl_embeddings) == len(texts_for_final_refl_store):
                            temp_final_refl_vs = Chroma(collection_name=f"final_refl_ret_{uuid.uuid4().hex[:6]}",
                                                        embedding_function=self.provider.embeddings)
                            temp_final_refl_vs.add_embeddings(texts=texts_for_final_refl_store,
                                                              embeddings=final_refl_embeddings,
                                                              metadatas=metadatas_for_final_refl_store)
                            reflection_chunks_retriever = temp_final_refl_vs.as_retriever(
                                search_kwargs={"k": desired_total_reflection_results})
                            logger.debug(
                                f"{log_prefix} Reflections: Final retriever created with {len(all_retrieved_reflection_docs)} combined docs.")
                        else:
                            logger.error(
                                f"{log_prefix} Reflections: Failed to embed docs for final combined retriever.")
                    except Exception as e_final_refl_vs:
                        logger.error(
                            f"{log_prefix} Reflections: Error creating final combined retriever: {e_final_refl_vs}")
            logger.debug(
                f"{log_prefix} Step 3 (Reflections) complete. Total retrieved: {len(all_retrieved_reflection_docs)} docs. Retriever set: {reflection_chunks_retriever is not None}")

            session_history_ids_str = ",".join(map(str, sorted(list(session_history_ids_set))))

            logger.info(
                f"{log_prefix} RAG retriever prep complete. URL: {'Yes' if url_retriever else 'No'}, "
                f"SessHistCombined: {'Yes' if session_history_retriever else 'No'} ({len(all_retrieved_history_docs)} docs total), "
                f"ReflChunkCombined: {'Yes' if reflection_chunks_retriever else 'No'} ({len(all_retrieved_reflection_docs)} docs total)."
            )

            ret_val = (url_retriever, session_history_retriever, reflection_chunks_retriever, session_history_ids_str)
            logger.critical(
                f"!!!!! _get_rag_retriever ABOUT TO RETURN {len(ret_val)} items. Types: {[type(x) for x in ret_val]} !!!!!")
            return ret_val

        except TaskInterruptedException as tie_outer:
            logger.warning(
                f"🚦 {log_prefix} TaskInterruptedException in _get_rag_retriever (likely during prioritized embedding): {tie_outer}. Re-raising.")
            raise tie_outer
        except Exception as e_outer:
            logger.error(f"❌❌ {log_prefix} UNHANDLED EXCEPTION in _get_rag_retriever: {e_outer}")
            logger.exception(f"{log_prefix} _get_rag_retriever Outer Exception Traceback:")
            raise e_outer

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

    def _call_llm_with_timing(self, chain: Any, inputs: Any, interaction_data: Dict[str, Any], priority: int = ELP0):
        """
        Wrapper to call LLM chain/model, measure time, log, and handle priority/interruptions.
        Implements retries for ELP0 tasks if they encounter TaskInterruptedException.
        """
        request_start_time = time.monotonic()  # Time for the whole operation including retries
        response_from_llm = None

        # Determine retry parameters based on priority
        # Only ELP0 tasks will attempt retries on interruption
        max_retries = LLM_CALL_ELP0_INTERRUPT_MAX_RETRIES if priority == ELP0 else 0
        retry_delay_seconds = LLM_CALL_ELP0_INTERRUPT_RETRY_DELAY

        attempt_count = 0
        while attempt_count <= max_retries:
            attempt_count += 1
            call_start_time = time.monotonic()  # Time for this specific attempt
            log_prefix_call = f"LLMCall|ELP{priority}|Attempt-{attempt_count}"

            try:
                logger.trace(f"{log_prefix_call}: Invoking chain/model {type(chain)}...")

                llm_call_config = {'priority': priority}  # For LlamaCppChatWrapper

                # The actual call to the LLM (via chain or model)
                if hasattr(chain, 'invoke') and callable(chain.invoke):  # Langchain runnable
                    response_from_llm = chain.invoke(inputs, config=llm_call_config)
                elif callable(chain):  # Direct model call (e.g., for raw ChatML in direct_generate)
                    # Assuming 'chain' is the model and 'inputs' is the raw prompt string.
                    # The LlamaCppChatWrapper._call method handles 'priority' from config.
                    response_from_llm = chain(messages=inputs, stop=[CHATML_END_TOKEN], **llm_call_config)
                else:
                    raise TypeError(f"Unsupported chain/model type for _call_llm_with_timing: {type(chain)}")

                call_duration_ms = (time.monotonic() - call_start_time) * 1000
                # Log duration for this specific attempt
                logger.info(f"⏱️ {log_prefix_call}: Succeeded in {call_duration_ms:.2f} ms")

                # Update total execution time in interaction_data with this attempt's duration
                interaction_data['execution_time_ms'] = interaction_data.get('execution_time_ms', 0) + call_duration_ms

                # Check if the response string itself indicates an interruption (from worker)
                if isinstance(response_from_llm, str) and interruption_error_marker in response_from_llm:
                    logger.warning(f"🚦 {log_prefix_call}: Task Interrupted (marker found in LLM response string).")
                    raise TaskInterruptedException(response_from_llm)  # Trigger retry logic

                return response_from_llm  # Successful call, exit retry loop

            except TaskInterruptedException as tie:
                call_duration_ms_on_interrupt = (time.monotonic() - call_start_time) * 1000
                interaction_data['execution_time_ms'] = interaction_data.get('execution_time_ms',
                                                                             0) + call_duration_ms_on_interrupt
                logger.warning(
                    f"🚦 {log_prefix_call}: Caught TaskInterruptedException after {call_duration_ms_on_interrupt:.2f}ms: {tie}")

                if priority == ELP0 and attempt_count <= max_retries:
                    logger.info(
                        f"    Retrying ELP0 task (attempt {attempt_count}/{max_retries + 1}) after {retry_delay_seconds}s due to interruption...")
                    # For asyncio.to_thread compatibility, use synchronous time.sleep
                    time.sleep(retry_delay_seconds)
                    # Loop continues for the next attempt
                else:
                    # Max retries reached for ELP0, or it's not an ELP0 task, or error from non-LLM part
                    if priority == ELP0:
                        logger.error(f"    ELP0 task giving up after {attempt_count} interruption attempts.")
                    raise  # Re-raise TaskInterruptedException to be handled by the caller (e.g., background_generate)

            except Exception as e:  # Handles other exceptions not related to TaskInterruptedException
                call_duration_ms_on_error = (time.monotonic() - call_start_time) * 1000
                interaction_data['execution_time_ms'] = interaction_data.get('execution_time_ms',
                                                                             0) + call_duration_ms_on_error
                log_err_msg = f"LLM Chain/Model Error (ELP{priority}, Attempt {attempt_count}): {e}"
                logger.error(f"❌ {log_err_msg}")
                # Log full traceback for these non-interruption errors
                logger.exception(f"Traceback for LLM Chain/Model error ({log_prefix_call}):")

                # Log this error to DB (simplified for brevity, actual DB logging in background_generate)
                session_id_for_log = interaction_data.get("session_id", "unknown_session")
                # add_interaction(db, session_id=session_id_for_log, ..., llm_response=log_err_msg)

                raise  # Re-raise the original non-interruption error; these are not retried by this loop

        # This part of the function should ideally not be reached if the loop logic is correct,
        # as success returns directly, and exceptions (including TaskInterruptedException after max retries) are re-raised.
        # This is a fallback.
        total_duration_ms = (time.monotonic() - request_start_time) * 1000
        interaction_data['execution_time_ms'] = total_duration_ms  # Ensure total time is updated
        logger.error(
            f"{log_prefix_call}: LLM call failed after all retries or was not retriable. Returning error indication.")
        return f"[LLM_CALL_UNEXPECTED_EXIT_ELP{priority}]"

    def _extract_json_candidate_string(self, raw_llm_text: str, log_prefix: str = "JSONExtract") -> Optional[str]:
        """
        Robustly extracts a potential JSON string from raw LLM text.
        Handles <think> tags, markdown code blocks, and finding outermost braces.
        """
        if not raw_llm_text or not isinstance(raw_llm_text, str):
            return None

        logger.trace(f"{log_prefix}: Starting JSON candidate extraction from raw text (len {len(raw_llm_text)}).")

        text_after_think_removal = re.sub(r'<think>.*?</think>', '', raw_llm_text,
                                          flags=re.DOTALL | re.IGNORECASE).strip()
        if not text_after_think_removal:
            logger.trace(f"{log_prefix}: Text empty after <think> removal.")
            return None

        cleaned_text = text_after_think_removal
        cleaned_text = re.sub(r"^\s*(assistant\s*\n?)?(<\|im_start\|>\s*(system|assistant)\s*\n?)?", "", cleaned_text,
                              flags=re.IGNORECASE).lstrip()
        if CHATML_END_TOKEN and cleaned_text.endswith(CHATML_END_TOKEN):
            cleaned_text = cleaned_text[:-len(CHATML_END_TOKEN)].strip()

        json_markdown_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", cleaned_text, re.DOTALL)
        if json_markdown_match:
            extracted_str = json_markdown_match.group(1).strip()
            logger.trace(f"{log_prefix}: Extracted JSON from markdown block: '{extracted_str[:100]}...'")
            return extracted_str

        first_brace = cleaned_text.find('{')
        last_brace = cleaned_text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            extracted_str = cleaned_text[first_brace: last_brace + 1].strip()
            logger.trace(f"{log_prefix}: Extracted JSON using outermost braces: '{extracted_str[:100]}...'")
            return extracted_str

        if cleaned_text.startswith("{") and cleaned_text.endswith("}"):
            logger.trace(f"{log_prefix}: Assuming cleaned text itself is the JSON candidate: '{cleaned_text[:100]}...'")
            return cleaned_text

        logger.warning(
            f"{log_prefix}: No clear JSON structure found after extraction attempts. Raw (after think): '{text_after_think_removal[:100]}...'")
        return None

    def _programmatic_json_parse_and_fix(self, json_candidate_str: str,
                                         max_fix_attempts: int = 3,
                                         log_prefix: str = "JSONFixParse") -> Optional[Union[Dict, List]]:
        """
        Attempts to parse a JSON string, applying a limited set of programmatic fixes
        if initial parsing fails.
        """
        if not json_candidate_str or not isinstance(json_candidate_str, str):
            return None

        current_text_to_parse = json_candidate_str
        json_parser = JsonOutputParser()

        for attempt in range(max_fix_attempts):
            logger.debug(f"{log_prefix}: Parse/fix attempt {attempt + 1}/{max_fix_attempts}.")
            try:
                parsed_object = json_parser.parse(current_text_to_parse)
                logger.debug(f"{log_prefix}: Successfully parsed JSON on attempt {attempt + 1}.")
                return parsed_object
            except (json.JSONDecodeError, OutputParserException) as e:
                logger.warning(f"{log_prefix}: Attempt {attempt + 1} parse failed: {type(e).__name__} - {str(e)[:100]}")
                if attempt == max_fix_attempts - 1:
                    logger.error(f"{log_prefix}: Max fix attempts reached. Could not parse after error: {e}")
                    break

                text_before_fixes = current_text_to_parse
                current_text_to_parse = re.sub(r",\s*([\}\]])", r"\1", current_text_to_parse)  # Fix trailing commas
                text_after_comment_removal = re.sub(r"//.*?\n", "\n", current_text_to_parse, flags=re.MULTILINE)
                current_text_to_parse = re.sub(r"/\*.*?\*/", "", text_after_comment_removal, flags=re.DOTALL).strip()

                if current_text_to_parse == text_before_fixes:
                    logger.warning(f"{log_prefix}: Programmatic fixes did not alter the string. Error may be complex.")
                    break  # Break if no progress is made

        logger.error(f"{log_prefix}: Failed to parse JSON after all programmatic fix attempts.")
        return None


    async def _classify_input_complexity(self, db: Session, user_input: str,
                                         interaction_data_for_metrics: dict) -> str:
        """
        Classifies input as 'chat_simple', 'chat_complex', or 'agent_task'.
        Uses router model, expects JSON, and now uses robust extraction and parsing/fixing.
        """
        request_id_suffix = str(uuid.uuid4())[:8]
        # Use session_id from interaction_data if available, else a default
        current_session_id = interaction_data_for_metrics.get("session_id", "unknown_classify_session")
        log_prefix = f"🤔 Classify|ELP0|{current_session_id[:8]}-{request_id_suffix}"
        logger.info(f"{log_prefix} Classifying input complexity for: '{user_input[:50]}...'")

        # Get history summary synchronously (it's a DB call)
        history_summary = await asyncio.to_thread(self._get_history_summary, db, MEMORY_SIZE)  # MEMORY_SIZE from config

        classification_model_instance = self.provider.get_model("router")
        if not classification_model_instance:
            logger.warning(f"{log_prefix} Router model unavailable for classification. Falling back to default model.")
            classification_model_instance = self.provider.get_model("default")

        if not classification_model_instance:
            error_msg = "Classification model (router/default) not available."
            logger.error(f"{log_prefix} ❌ {error_msg}")
            interaction_data_for_metrics['classification'] = "chat_simple"  # Fallback
            interaction_data_for_metrics['classification_reason'] = error_msg
            await asyncio.to_thread(add_interaction, db, session_id=current_session_id, mode="chat",
                                    input_type="log_error", user_input="[Classify Model Unavailable]",
                                    llm_response=error_msg)
            await asyncio.to_thread(db.commit)
            return "chat_simple"

        classification_model_for_call = classification_model_instance
        if hasattr(classification_model_instance, 'bind') and callable(getattr(classification_model_instance, 'bind')):
            try:
                classification_model_for_call = classification_model_instance.bind(temperature=0.1)
            except Exception as bind_err:
                logger.warning(f"{log_prefix} Could not bind temperature: {bind_err}.")

        prompt_inputs_for_classification = {"input": user_input, "history_summary": history_summary}
        classification_chain_raw_output = (
                self.input_classification_prompt  # PROMPT_COMPLEXITY_CLASSIFICATION from config
                | classification_model_for_call
                | StrOutputParser()
        )

        last_error_for_log: Optional[Exception] = None
        raw_llm_response_for_final_log: str = "Classification LLM call did not produce parsable output."
        parsed_json_output: Optional[Dict[str, Any]] = None

        # Initial LLM calls and parsing attempts
        for attempt in range(DEEP_THOUGHT_RETRY_ATTEMPTS):  # DEEP_THOUGHT_RETRY_ATTEMPTS from config
            current_attempt_num = attempt + 1
            logger.debug(
                f"{log_prefix} Classification LLM call attempt {current_attempt_num}/{DEEP_THOUGHT_RETRY_ATTEMPTS}")

            raw_llm_text_this_attempt = ""
            try:
                raw_llm_text_this_attempt = await asyncio.to_thread(
                    self._call_llm_with_timing,
                    classification_chain_raw_output,
                    prompt_inputs_for_classification,
                    interaction_data_for_metrics,  # For timing
                    priority=ELP0
                )
                raw_llm_response_for_final_log = raw_llm_text_this_attempt  # Save last raw output
                logger.trace(
                    f"{log_prefix} Raw LLM for classification (Attempt {current_attempt_num}): '{raw_llm_text_this_attempt[:200]}...'")

                json_candidate_str = self._extract_json_candidate_string(raw_llm_text_this_attempt,
                                                                         log_prefix + "-Extract")
                if json_candidate_str:
                    # Try parsing with limited fixes (e.g., 1 attempt for this initial loop)
                    parsed_json_output = self._programmatic_json_parse_and_fix(
                        json_candidate_str,
                        1,  # Only 1 fix attempt per initial LLM call
                        log_prefix + f"-InitialFixAttempt{current_attempt_num}"
                    )
                    if parsed_json_output and isinstance(parsed_json_output, dict) and \
                            "classification" in parsed_json_output and "reason" in parsed_json_output:
                        classification_val = str(parsed_json_output.get("classification", "chat_simple")).lower()
                        reason_val = str(parsed_json_output.get("reason", "N/A"))
                        if classification_val not in ["chat_simple", "chat_complex", "agent_task"]:
                            logger.warning(f"{log_prefix} Invalid category '{classification_val}'. Defaulting.")
                            classification_val = "chat_simple"
                        interaction_data_for_metrics['classification'] = classification_val
                        interaction_data_for_metrics['classification_reason'] = reason_val
                        logger.info(f"✅ {log_prefix} Input classified as: '{classification_val}'. Reason: {reason_val}")
                        return classification_val  # Success
                else:  # No candidate extracted
                    last_error_for_log = ValueError(
                        f"No JSON candidate extracted from LLM output: {raw_llm_text_this_attempt[:100]}")

            except TaskInterruptedException as tie:
                raise tie  # Propagate immediately
            except Exception as e:
                last_error_for_log = e

            logger.warning(
                f"⚠️ {log_prefix} Classification attempt {current_attempt_num} failed. Error: {last_error_for_log}")
            if current_attempt_num < DEEP_THOUGHT_RETRY_ATTEMPTS: await asyncio.sleep(0.5 + attempt * 0.5)

        # --- If all initial attempts failed, try LLM Re-request to fix format ---
        if not parsed_json_output:  # Check if we still don't have valid JSON
            logger.warning(
                f"{log_prefix} Initial classification attempts failed. Trying LLM re-request to fix format. Last raw output: '{raw_llm_response_for_final_log[:200]}...'")
            action_analysis_model = self.provider.get_model("router")
            reformat_prompt_input = {"faulty_llm_output_for_reformat": raw_llm_response_for_final_log}
            reformat_chain = ChatPromptTemplate.from_template(
                PROMPT_REFORMAT_TO_ACTION_JSON) | action_analysis_model | StrOutputParser()  # PROMPT_REFORMAT_TO_ACTION_JSON from config

            reformatted_llm_output_text = await asyncio.to_thread(
                self._call_llm_with_timing, reformat_chain, reformat_prompt_input,
                interaction_data_for_metrics, priority=ELP0
            )

            if reformatted_llm_output_text and not (
                    isinstance(reformatted_llm_output_text, str) and "ERROR" in reformatted_llm_output_text.upper()):
                logger.info(
                    f"{log_prefix} Received reformatted output from LLM for classification. Attempting to parse/fix...")
                json_candidate_from_reformat = self._extract_json_candidate_string(reformatted_llm_output_text,
                                                                                   log_prefix + "-ReformatExtract")
                if json_candidate_from_reformat:
                    parsed_json_output = self._programmatic_json_parse_and_fix(
                        json_candidate_from_reformat,
                        JSON_FIX_RETRY_ATTEMPTS_AFTER_REFORMAT,  # From config (e.g., 2-3 attempts)
                        log_prefix + "-ReformatFix"
                    )
                    if parsed_json_output and isinstance(parsed_json_output, dict) and \
                            "classification" in parsed_json_output and "reason" in parsed_json_output:
                        classification_val = str(parsed_json_output.get("classification", "chat_simple")).lower()
                        reason_val = str(parsed_json_output.get("reason", "N/A (reformatted)"))
                        if classification_val not in ["chat_simple", "chat_complex", "agent_task"]:
                            logger.warning(
                                f"{log_prefix} Invalid category '{classification_val}' from reformat. Defaulting.")
                            classification_val = "chat_simple"
                        interaction_data_for_metrics['classification'] = classification_val
                        interaction_data_for_metrics['classification_reason'] = reason_val
                        logger.info(
                            f"✅ {log_prefix} Reformat & Fix successful: Classified as '{classification_val}'. Reason: {reason_val}")
                        return classification_val  # Success after reformat and fix
                else:
                    logger.error(
                        f"{log_prefix} Failed to extract any JSON from LLM's reformat attempt. Output: {reformatted_llm_output_text[:200]}")
            else:
                logger.error(
                    f"{log_prefix} LLM re-request for JSON formatting failed or returned error: {reformatted_llm_output_text}")

        # --- Fallback if all methods failed ---
        final_fallback_classification = "chat_simple"
        final_fallback_reason = f"Classification failed after all attempts. Last error: {last_error_for_log}. Last LLM raw: {raw_llm_response_for_final_log[:200]}..."
        logger.error(
            f"{log_prefix} ❌ All classification methods failed. Defaulting to '{final_fallback_classification}'. Reason: {final_fallback_reason}")

        interaction_data_for_metrics['classification'] = final_fallback_classification
        interaction_data_for_metrics['classification_reason'] = final_fallback_reason
        await asyncio.to_thread(add_interaction, db, session_id=current_session_id, mode="chat", input_type="log_error",
                                user_input=f"[Classify Max Retries for: {user_input[:100]}]",
                                llm_response=final_fallback_reason[:4000])
        await asyncio.to_thread(db.commit)
        return final_fallback_classification


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

    async def _run_tot_in_background_wrapper_v2(self, db_session_factory: Any,
                                                original_input_for_tot: str,  # Renamed for clarity
                                                rag_context_docs: List[Any],
                                                history_rag_interactions: List[Any],
                                                log_context_str: str,
                                                recent_direct_history_str: str,
                                                file_index_context_str: str,
                                                triggering_interaction_id: int,
                                                # ID of interaction that triggered this ToT
                                                imagined_image_context_str: str):

        # This wrapper runs in a separate thread created by asyncio.create_task(self._run_tot_in_background_wrapper_v2(...))
        # So, it needs its own DB session.
        db_for_tot_thread: Optional[Session] = None
        thread_log_prefix = f"BG_ToT_Wrap|TrigID:{triggering_interaction_id}"
        logger.info(f"{thread_log_prefix}: Background ToT task thread started.")

        try:
            db_for_tot_thread = db_session_factory()  # Create a new session for this thread
            if not db_for_tot_thread:
                logger.error(f"{thread_log_prefix}: Failed to create DB session. Aborting ToT.")
                return

            # Prepare interaction_data for the _call_llm_with_timing within _run_tree_of_thought_v2
            # This is for metrics of the ToT LLM call itself.
            interaction_data_for_llm_call = {
                'session_id': self.current_session_id,  # Use session_id from the AIChat instance
                'mode': 'chat',  # Or 'internal_tot_llm_call'
                'execution_time_ms': 0
            }

            # Run the synchronous _run_tree_of_thought_v2 using asyncio.to_thread
            # because _run_tree_of_thought_v2 itself makes blocking calls (_call_llm_with_timing)
            await asyncio.to_thread(
                self._run_tree_of_thought_v2,
                db=db_for_tot_thread,
                input=original_input_for_tot,  # This is passed to ToT prompt as {input}
                rag_context_docs=rag_context_docs,
                history_rag_interactions=history_rag_interactions,
                log_context_str=log_context_str,
                recent_direct_history_str=recent_direct_history_str,
                file_index_context_str=file_index_context_str,
                imagined_image_context_str=imagined_image_context_str,
                interaction_data_for_tot_llm_call=interaction_data_for_llm_call,  # For the LLM call timing
                original_user_input_for_log=original_input_for_tot,  # For logging within ToT result record
                triggering_interaction_id_for_log=triggering_interaction_id  # For logging
            )
            logger.info(f"{thread_log_prefix}: _run_tree_of_thought_v2 completed execution.")

            # Mark the original interaction as "ToT analysis spawned"
            # This is better than directly putting the ToT result on it.
            trigger_interaction = db_for_tot_thread.query(Interaction).filter(
                Interaction.id == triggering_interaction_id).first()
            if trigger_interaction:
                if hasattr(trigger_interaction, 'tot_analysis_spawned'):
                    trigger_interaction.tot_analysis_spawned = True  # type: ignore
                    trigger_interaction.last_modified_db = time.strftime("%Y-%m-%d %H:%M:%S")  # type: ignore
                    db_for_tot_thread.commit()
                    logger.info(
                        f"{thread_log_prefix}: Marked original Interaction ID {triggering_interaction_id} as tot_analysis_spawned=True.")
                else:
                    logger.warning(
                        f"{thread_log_prefix}: Original Interaction ID {triggering_interaction_id} missing 'tot_analysis_spawned' field.")
            else:
                logger.error(
                    f"{thread_log_prefix}: Could not find original Interaction ID {triggering_interaction_id} to mark as ToT spawned.")

        except TaskInterruptedException:
            logger.warning(f"🚦 {thread_log_prefix}: ToT task was interrupted.")
            # The interruption should have been logged by _run_tree_of_thought_v2 already.
        except Exception as e:
            logger.error(f"{thread_log_prefix}: Error running ToT in background wrapper: {e}")
            logger.exception(f"{thread_log_prefix} ToT Wrapper Traceback:")
            # The error should have been logged by _run_tree_of_thought_v2 as a new interaction.
        finally:
            if db_for_tot_thread:
                try:
                    db_for_tot_thread.close()
                except Exception as e_close:
                    logger.error(f"{thread_log_prefix}: Error closing ToT DB session: {e_close}")
            logger.info(f"{thread_log_prefix}: Background ToT task thread finished.")

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

    async def _analyze_assistant_action(self, db: Session, user_input: str, session_id: str,
                                        context: Dict[str, str]) -> Optional[Dict[str, Any]]:
        request_id_suffix = str(uuid.uuid4())[:8]
        log_prefix = f"🤔 ActionAnalyze|ELP0|{session_id[:8]}-{request_id_suffix}"
        logger.info(f"{log_prefix} Analyzing input for action: '{user_input[:50]}...'")

        prompt_input_initial = {
            "input": user_input, "history_summary": context.get("history_summary", "N/A"),
            "log_context": context.get("log_context", "N/A"),
            "recent_direct_history": context.get("recent_direct_history", "N/A")
        }

        action_analysis_model = self.provider.get_model("router") or self.provider.get_model("default")
        if not action_analysis_model:
            logger.error(f"{log_prefix} ❌ Action analysis model (router/default) not available!");
            return None

        analysis_chain_raw = ChatPromptTemplate.from_template(
            PROMPT_ASSISTANT_ACTION_ANALYSIS) | action_analysis_model | StrOutputParser()
        action_timing_data = {"session_id": session_id, "mode": "chat", "execution_time_ms": 0}

        raw_llm_output_from_initial_loop: str = "Initial LLM action analysis did not yield parsable output."
        parsed_action_json: Optional[Dict[str, Any]] = None

        for attempt in range(DEEP_THOUGHT_RETRY_ATTEMPTS):
            current_attempt_num = attempt + 1
            logger.debug(
                f"{log_prefix} Initial Action analysis LLM call attempt {current_attempt_num}/{DEEP_THOUGHT_RETRY_ATTEMPTS}")
            try:
                raw_llm_text = await asyncio.to_thread(self._call_llm_with_timing, analysis_chain_raw,
                                                       prompt_input_initial, action_timing_data, priority=ELP0)
                raw_llm_output_from_initial_loop = raw_llm_text
                json_candidate = self._extract_json_candidate_string(raw_llm_text, log_prefix)
                if json_candidate:
                    parsed_action_json = self._programmatic_json_parse_and_fix(json_candidate, 1,
                                                                               log_prefix + f"-InitialFix{current_attempt_num}")
                    if parsed_action_json and isinstance(parsed_action_json,
                                                         dict) and "action_type" in parsed_action_json and "parameters" in parsed_action_json:
                        action_type = parsed_action_json.get("action_type")
                        logger.info(
                            f"✅ {log_prefix} Initial Action analysis successful (Attempt {current_attempt_num}): Type='{action_type}'")
                        return parsed_action_json if action_type != "no_action" else None
            except TaskInterruptedException as tie:
                raise tie
            except Exception as e:
                logger.warning(f"⚠️ {log_prefix} Initial Action analysis attempt {current_attempt_num} failed: {e}")
            if current_attempt_num < DEEP_THOUGHT_RETRY_ATTEMPTS: await asyncio.sleep(0.5 + attempt * 0.5)

        if not parsed_action_json:
            logger.warning(
                f"{log_prefix} Initial attempts failed. Trying LLM re-request to fix format. Last raw output: '{raw_llm_output_from_initial_loop[:200]}...'")

            # --- START OF THE FIX ---
            # Define the example string that the new prompt expects.
            json_example_string = """{
  "action_type": "string (e.g., 'search', 'scheduling', or 'no_action')",
  "parameters": {
    "param_key_1": "string_value_1",
    "param_key_2": "string_value_2"
  },
  "explanation": "string (your reasoning for the choice)"
}"""

            # Create the input dictionary with BOTH required variables.
            reformat_prompt_input = {
                "faulty_llm_output_for_reformat": raw_llm_output_from_initial_loop,
                "json_structure_example": json_example_string
            }

            reformat_prompt_input = {
                "faulty_llm_output_for_reformat": raw_llm_output_from_initial_loop,
                "\"action_type\"": "{dummy_value}"  # Provide the exact key the error asks for
            }
            # --- END OF THE FIX ---

            reformat_chain = ChatPromptTemplate.from_template(
                PROMPT_REFORMAT_TO_ACTION_JSON) | action_analysis_model | StrOutputParser()

            reformatted_llm_output_text = await asyncio.to_thread(self._call_llm_with_timing, reformat_chain,
                                                                  reformat_prompt_input, action_timing_data,
                                                                  priority=ELP0)

            if reformatted_llm_output_text and not (isinstance(reformatted_llm_output_text,
                                                               str) and "ERROR" in reformatted_llm_output_text.upper()):
                logger.info(f"{log_prefix} Received reformatted output from LLM. Parsing with fix retries...")
                json_candidate_from_reformat = self._extract_json_candidate_string(reformatted_llm_output_text,
                                                                                   log_prefix + "-ReformatExtract")
                if json_candidate_from_reformat:
                    parsed_action_json = self._programmatic_json_parse_and_fix(json_candidate_from_reformat,
                                                                               JSON_FIX_RETRY_ATTEMPTS_AFTER_REFORMAT,
                                                                               log_prefix + "-ReformatFix")
                    if parsed_action_json and isinstance(parsed_action_json,
                                                         dict) and "action_type" in parsed_action_json and "parameters" in parsed_action_json:
                        action_type = parsed_action_json.get("action_type")
                        logger.info(f"✅ {log_prefix} Reformatted Action analysis successful: Type='{action_type}'")
                        return parsed_action_json if action_type != "no_action" else None
                else:
                    logger.error(
                        f"{log_prefix} Failed to extract any JSON from LLM's reformat attempt. Output: {reformatted_llm_output_text[:200]}")
            else:
                logger.error(
                    f"{log_prefix} LLM re-request for JSON formatting failed or returned error: {reformatted_llm_output_text}")

        logger.warning(
            f"{log_prefix} All action analysis methods failed. Falling back to keyword-based default action for: '{user_input[:100]}'")
        words = re.findall(r'\b\w+\b', user_input.lower());
        stop_words = {"the", "is", "a", "to", "and", "what", "how", "who", "please", "can", "you", "tell", "me",
                      "about", "of", "for", "in", "on", "at", "an", "i", "my", "me"}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        fallback_params = {"original_query": user_input[:200], "extracted_keywords": list(set(keywords))[:7]}
        logger.info(f"{log_prefix} Extracted keywords for fallback: {fallback_params['extracted_keywords']}")
        final_fallback_action = {"action_type": "keyword_based_response_fallback", "parameters": fallback_params,
                                 "explanation": "Automated action analysis failed after multiple attempts. Using keyword-based fallback for general response or search."}

        try:
            await asyncio.to_thread(add_interaction, db, session_id=session_id, mode="chat",
                                    input_type="log_warning",
                                    user_input=f"[Action Analysis Fallback for: {user_input[:100]}]",
                                    llm_response=json.dumps(final_fallback_action),
                                    assistant_action_analysis_json=json.dumps(final_fallback_action),
                                    assistant_action_type=final_fallback_action["action_type"]);
            await asyncio.to_thread(db.commit)
        except Exception as db_log_fallback_err:
            logger.error(f"{log_prefix} Failed to log keyword fallback action: {db_log_fallback_err}");
            await asyncio.to_thread(db.rollback)

        return final_fallback_action



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
    async def _route_to_specialist(self, db: Session, session_id: str, user_input_for_routing: str,
                                   prompt_input_for_router: Dict[str, Any]  # This now contains all contexts
                                   ) -> Tuple[str, str, str]:  # (chosen_model, refined_query, reasoning)

        log_prefix = f"🧠 Route|ELP0|{session_id}"  # Router typically runs at ELP0
        logger.info(f"{log_prefix} Routing request for: '{user_input_for_routing[:50]}...'")

        router_model = self.provider.get_model("router")
        default_model_key = "general"
        default_reason = "Fell back to default model after routing/parsing issues."

        if not router_model:
            logger.error(f"{log_prefix}: Router model ('router') not available! Defaulting to '{default_model_key}'.")
            # Log this failure to DB
            await asyncio.to_thread(add_interaction, db, session_id=session_id, mode="chat", input_type="log_error",
                                    user_input="[Router Model Unavailable]",
                                    llm_response=f"Router model key 'router' not configured. Defaulting to {default_model_key}.")
            await asyncio.to_thread(db.commit)
            return default_model_key, user_input_for_routing, "Router model unavailable, using default."

        router_chain_raw_output = (
                ChatPromptTemplate.from_template(PROMPT_ROUTER)  # PROMPT_ROUTER from config
                | router_model
                | StrOutputParser()
        )
        router_timing_data = {"session_id": session_id, "mode": "chat", "execution_time_ms": 0}

        last_error_from_initial_loop: Optional[Exception] = None
        raw_llm_output_from_initial_loop: str = "Initial router LLM call did not yield parsable JSON."
        parsed_routing_json: Optional[Dict[str, Any]] = None

        # --- Stage 1: Initial LLM Calls and Parsing Attempts ---
        for attempt in range(DEEP_THOUGHT_RETRY_ATTEMPTS):
            current_attempt_num = attempt + 1
            logger.debug(f"{log_prefix} Router LLM call attempt {current_attempt_num}/{DEEP_THOUGHT_RETRY_ATTEMPTS}")
            raw_llm_text_this_attempt = ""
            try:
                raw_llm_text_this_attempt = await asyncio.to_thread(
                    self._call_llm_with_timing, router_chain_raw_output, prompt_input_for_router,
                    router_timing_data, priority=ELP0
                )
                raw_llm_output_from_initial_loop = raw_llm_text_this_attempt
                logger.trace(
                    f"{log_prefix} Raw LLM Router Output (Attempt {current_attempt_num}): '{raw_llm_text_this_attempt[:200]}...'")

                json_candidate_str = self._extract_json_candidate_string(raw_llm_text_this_attempt,
                                                                         log_prefix + "-ExtractInitial")
                if json_candidate_str:
                    # Try parsing with limited fixes (e.g., 1 attempt for this initial loop)
                    parsed_routing_json = self._programmatic_json_parse_and_fix(
                        json_candidate_str, 1, log_prefix + f"-InitialFixAttempt{current_attempt_num}"
                    )
                    if parsed_routing_json and isinstance(parsed_routing_json, dict) and \
                            all(k in parsed_routing_json for k in ["chosen_model", "refined_query", "reasoning"]):
                        chosen_model = str(parsed_routing_json["chosen_model"])
                        refined_query = str(parsed_routing_json["refined_query"])
                        reasoning = str(parsed_routing_json.get("reasoning", "N/A"))
                        # Basic validation of chosen_model key could be added here
                        logger.info(
                            f"✅ {log_prefix} Router successful (Attempt {current_attempt_num}): Chose '{chosen_model}'. Reason: {reasoning[:50]}...")
                        return chosen_model, refined_query, reasoning
                else:
                    last_error_from_initial_loop = ValueError(
                        f"No JSON candidate extracted from LLM router output: {raw_llm_text_this_attempt[:100]}")

            except TaskInterruptedException as tie:
                raise tie  # Propagate immediately
            except Exception as e_initial_route:
                last_error_from_initial_loop = e_initial_route

            logger.warning(
                f"⚠️ {log_prefix} Router LLM/parse attempt {current_attempt_num} failed. Error: {last_error_from_initial_loop}")
            if current_attempt_num < DEEP_THOUGHT_RETRY_ATTEMPTS: await asyncio.sleep(0.5 + attempt * 0.5)

        # --- Stage 2: LLM Re-request for Formatting (if initial attempts failed) ---
        if not parsed_routing_json:
            logger.warning(
                f"{log_prefix} Initial routing attempts failed. Trying LLM re-request to fix format. Last raw output: '{raw_llm_output_from_initial_loop[:200]}...'")

            reformat_prompt_input = {
                "faulty_llm_output_for_reformat": raw_llm_output_from_initial_loop,
                "original_user_input_placeholder": user_input_for_routing  # For the fallback in the reformat prompt
            }
            reformat_chain = ChatPromptTemplate.from_template(
                PROMPT_REFORMAT_TO_ROUTER_JSON) | router_model | StrOutputParser()  # New Prompt

            reformatted_llm_output_text = await asyncio.to_thread(
                self._call_llm_with_timing, reformat_chain, reformat_prompt_input,
                router_timing_data, priority=ELP0
            )

            if reformatted_llm_output_text and not (
                    isinstance(reformatted_llm_output_text, str) and "ERROR" in reformatted_llm_output_text.upper()):
                logger.info(f"{log_prefix} Received reformatted output from LLM for router. Attempting to parse/fix...")
                json_candidate_from_reformat = self._extract_json_candidate_string(reformatted_llm_output_text,
                                                                                   log_prefix + "-ReformatExtract")
                if json_candidate_from_reformat:
                    # --- Stage 3: Parse/Fix Reformatted Output ---
                    parsed_routing_json = self._programmatic_json_parse_and_fix(
                        json_candidate_from_reformat,
                        JSON_FIX_RETRY_ATTEMPTS_AFTER_REFORMAT,  # From config
                        log_prefix + "-ReformatFix"
                    )
                    if parsed_routing_json and isinstance(parsed_routing_json, dict) and \
                            all(k in parsed_routing_json for k in ["chosen_model", "refined_query", "reasoning"]):
                        chosen_model = str(parsed_routing_json["chosen_model"])
                        refined_query = str(parsed_routing_json["refined_query"])
                        reasoning = str(parsed_routing_json.get("reasoning", "N/A (reformatted)"))
                        logger.info(f"✅ {log_prefix} Reformatted Routing analysis successful: Chose '{chosen_model}'.")
                        return chosen_model, refined_query, reasoning
                else:
                    logger.error(
                        f"{log_prefix} Failed to extract any JSON from LLM's reformat (router) attempt. Output: {reformatted_llm_output_text[:200]}")
            else:
                logger.error(
                    f"{log_prefix} LLM re-request for router JSON formatting failed or returned error: {reformatted_llm_output_text}")

        # --- Fallback if all methods failed ---
        logger.error(
            f"{log_prefix} ❌ All routing methods failed. Defaulting to '{default_model_key}'. Last error: {last_error_from_initial_loop}. Last raw output: '{raw_llm_output_from_initial_loop[:200]}...'")
        await asyncio.to_thread(add_interaction, db, session_id=session_id, mode="chat", input_type="log_error",
                                user_input=f"[Router Failed for: {user_input_for_routing[:100]}]",
                                llm_response=f"All routing attempts failed. Defaulting. Last raw: {raw_llm_output_from_initial_loop[:500]}")
        await asyncio.to_thread(db.commit)
        return default_model_key, user_input_for_routing, default_reason

    # --- generate method ---
    # app.py -> Inside AIChat class

    # --- generate (Main Async Method - Fuzzy History RAG + Direct History + Log Context + Multi-LLM Routing + VLM Preprocessing) ---
    # app.py -> Inside AIChat class

    async def direct_generate(self, db: Session, user_input: str, session_id: str,
                              vlm_description: Optional[str] = None,
                              image_b64: Optional[str] = None) -> str:
        direct_req_id = f"dgen-raw_chatml-{uuid.uuid4()}"
        log_prefix = f"⚡️ {direct_req_id}|ELP1"
        logger.info(
            f"{log_prefix} Direct Generate START --> Session: {session_id}, Input: '{user_input[:50]}...', VLM Desc: {'Yes' if vlm_description else 'No'}, Image b64: {'Yes' if image_b64 else 'No'}")
        direct_start_time = time.monotonic()
        self.current_session_id = session_id

        # Initialize with all potential fields and their defaults for a successful ELP1 LLM path
        interaction_data_for_log: Dict[str, Any] = {
            "session_id": session_id, "mode": "chat",
            "input_type": "image+text" if vlm_description or image_b64 else "text",
            "user_input": user_input,
            "llm_response": "[Processing direct generate...]",  # Placeholder, will be updated
            "execution_time_ms": 0,
            "image_description": vlm_description,
            "image_data": image_b64[:20] + "..." if image_b64 else None,
            "classification": "direct_response_raw_chatml_elp1",  # Default for this path
            "classification_reason": "Direct ELP1 path execution.",
            "rag_history_ids": None, "rag_source_url": None,
            "requires_deep_thought": False, "deep_thought_reason": None,
            "tot_analysis_requested": False, "tot_analysis_spawned": False,
            "tot_result": None, "tot_delivered": False,
            "emotion_context_analysis": None, "assistant_action_analysis_json": None,
            "assistant_action_type": None, "assistant_action_params": None,
            "assistant_action_executed": False, "assistant_action_result": None,
            "imagined_image_prompt": None, "imagined_image_b64": None,
            "imagined_image_vlm_description": None, "reflection_completed": False,
            "reflection_indexed_in_vs": False  # Added new field
        }

        # This variable will hold the text to be returned to the client
        # It's updated by hook, LLM, or error handling.
        response_to_return_to_client = "[Error: Response not set during direct_generate]"

        try:
            # --- StellaIcarusHook Check (EARLY EXIT) ---
            if self.stella_icarus_manager and self.stella_icarus_manager.is_enabled:
                hook_response_text = self.stella_icarus_manager.check_and_execute(user_input, session_id)
                if hook_response_text is not None:
                    logger.info(f"{log_prefix} STELLA_ICARUS_HOOK triggered for: '{user_input[:50]}...'")
                    interaction_data_for_log['llm_response'] = hook_response_text
                    interaction_data_for_log['classification'] = "stella_icarus_hooked"
                    interaction_data_for_log['classification_reason'] = "Input matched a StellaIcarusHook."
                    response_to_return_to_client = hook_response_text
                    return response_to_return_to_client  # Exits try, 'finally' will run
            # --- End StellaIcarusHook Check ---

            fast_model = self.provider.get_model("general_fast")
            if not fast_model:
                error_msg = "Fast model 'general_fast' for direct response is not configured."
                logger.error(f"{log_prefix}: {error_msg}")
                response_to_return_to_client = f"Error: Cannot generate quick response ({error_msg})."
                interaction_data_for_log['llm_response'] = response_to_return_to_client
                interaction_data_for_log['classification'] = "error_model_unavailable"
                return response_to_return_to_client  # Exits try, 'finally' will run

            # --- RAG Context & Prompt Construction (if not hooked) ---
            system_prompt_content_base = PROMPT_DIRECT_GENERATE_SYSTEM_CONTENT
            historical_turns_for_chatml: List[Dict[str, str]] = []
            rag_context_block_for_system_prompt = "No relevant RAG context found."
            session_chat_rag_ids_used_str = ""
            final_system_prompt_content = system_prompt_content_base

            url_retriever_obj: Optional[Any] = None
            session_hist_retriever_obj: Optional[Any] = None
            reflection_chunk_retriever_obj: Optional[Any] = None

            rag_query_input = user_input
            if not rag_query_input and vlm_description:
                rag_query_input = f"Regarding image: {vlm_description}"
            elif not rag_query_input and not vlm_description:
                rag_query_input = ""

            wrapped_rag_result = await asyncio.to_thread(
                self._get_rag_retriever_thread_wrapper, db, rag_query_input, ELP1
            )

            if wrapped_rag_result.get("status") == "success":
                rag_data_tuple = wrapped_rag_result.get("data")
                if isinstance(rag_data_tuple, tuple) and len(rag_data_tuple) == 4:
                    url_retriever_obj, session_hist_retriever_obj, reflection_chunk_retriever_obj, session_chat_rag_ids_used_str = rag_data_tuple
                    interaction_data_for_log['rag_history_ids'] = session_chat_rag_ids_used_str
                    if hasattr(self, 'vectorstore_url') and self.vectorstore_url and hasattr(self.vectorstore_url,
                                                                                             '_source_url'):
                        interaction_data_for_log['rag_source_url'] = self.vectorstore_url._source_url  # type: ignore
                else:
                    raise RuntimeError(f"RAG wrapper malformed data: {rag_data_tuple}")
            elif wrapped_rag_result.get("status") == "interrupted":
                raise TaskInterruptedException(wrapped_rag_result.get("error_message", "RAG interrupted"))
            else:  # Error
                error_msg_rag = wrapped_rag_result.get("error_message", "Unknown RAG error")
                logger.error(f"{log_prefix}: RAG retrieval failed: {error_msg_rag}")
                final_system_prompt_content = f"{system_prompt_content_base}\n\n[System Note: Error RAG: {error_msg_rag}]"

            all_retrieved_rag_docs: List[Any] = []
            if wrapped_rag_result.get("status") == "success":
                # (Invoke retrievers as before, populate all_retrieved_rag_docs)
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

            if all_retrieved_rag_docs:
                # (Token budgeting and truncation logic as before)
                untruncated_rag_block = self._format_docs(all_retrieved_rag_docs, "Combined RAG Context")
                current_full_input_for_tokens = user_input
                if vlm_description: current_full_input_for_tokens = f"[Image: {vlm_description.strip()}]{CHATML_NL}{user_input.strip() or '(Query image)'}"

                direct_hist_interactions_tc = await asyncio.to_thread(get_global_recent_interactions, db, limit=3)
                temp_hist_turns_tc: List[Dict[str, str]] = []
                for item_tc in direct_hist_interactions_tc:
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

                # Ensure BUFFER_TOKENS_FOR_RESPONSE is defined (e.g., in config or here)
                BUFFER_TOKENS_FOR_RESPONSE = getattr(self, 'BUFFER_TOKENS_FOR_RESPONSE', 512)  # Example
                MODEL_CONTEXT_WINDOW = LLAMA_CPP_N_CTX
                fixed_parts_total_tokens = base_prompt_tokens + user_input_tokens + history_turns_tokens
                max_rag_tokens_budget = MODEL_CONTEXT_WINDOW - fixed_parts_total_tokens - BUFFER_TOKENS_FOR_RESPONSE
                if max_rag_tokens_budget < 100: max_rag_tokens_budget = max(0, max_rag_tokens_budget)

                if max_rag_tokens_budget > 0:
                    rag_context_block_for_system_prompt = self._truncate_rag_context(untruncated_rag_block,
                                                                                     max_rag_tokens_budget)
                else:
                    rag_context_block_for_system_prompt = "[RAG Context Skipped: Budget]"

            if rag_context_block_for_system_prompt not in ["No relevant RAG context found.",
                                                           "[RAG Context Skipped: Budget]"]:
                final_system_prompt_content += (
                    f"\n\n--- Relevant Context (RAG) ---\n{rag_context_block_for_system_prompt}\n--- End RAG ---"
                )

            direct_history_interactions = await asyncio.to_thread(get_global_recent_interactions, db, limit=3)
            for interaction_dh in direct_history_interactions:
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

            current_user_turn_for_chatml = user_input
            if vlm_description:
                current_user_turn_for_chatml = f"[Image Description: {vlm_description.strip()}]{CHATML_NL}{CHATML_NL}User Query: {user_input.strip() or '(Query on image)'}"
            elif not current_user_turn_for_chatml.strip() and image_b64:  # Only image provided, no text query
                current_user_turn_for_chatml = "(User provided an image without a specific text query)"
            elif not current_user_turn_for_chatml.strip():  # No text, no image
                current_user_turn_for_chatml = "(User provided no text)"

            raw_chatml_prompt_string = self._construct_raw_chatml_prompt(
                system_content=final_system_prompt_content,
                history_turns=historical_turns_for_chatml,
                current_turn_content=current_user_turn_for_chatml,
                current_turn_role="user",
                prompt_for_assistant_response=True
            )
            # (Log prompt length and potential overflow as before)

            # --- LLM Call ---
            logger.debug(f"{log_prefix} Calling 'general_fast' model (ELP1)...")
            raw_llm_response = await asyncio.to_thread(
                fast_model._call,
                messages=raw_chatml_prompt_string,
                stop=[CHATML_END_TOKEN],
                priority=ELP1,
            )
            response_to_return_to_client = self._cleanup_llm_output(raw_llm_response)
            interaction_data_for_log['llm_response'] = response_to_return_to_client
            # interaction_data_for_log['classification'] is already "direct_response_raw_chatml_elp1"

        except TaskInterruptedException as tie:
            logger.warning(f"🚦 {log_prefix} Direct Generate Task INTERRUPTED: {tie}")
            response_to_return_to_client = f"[Error: Direct response (ELP1) interrupted: {tie}]"
            interaction_data_for_log['llm_response'] = response_to_return_to_client
            interaction_data_for_log['classification'] = "direct_response_interrupted"
            raise  # Re-raise to be caught by the route handler for 503 status

        except Exception as e_direct_path:
            logger.error(f"❌ {log_prefix}: Error during direct_generate LLM path: {e_direct_path}")
            logger.exception(f"{log_prefix} Direct Generate LLM Path Traceback:")
            response_to_return_to_client = f"[Error generating direct response (ELP1): {type(e_direct_path).__name__} - {str(e_direct_path)[:200]}]"
            interaction_data_for_log['llm_response'] = response_to_return_to_client
            interaction_data_for_log['classification'] = "direct_response_error"
            # Do not re-raise here; let 'finally' log this error state and then return the error text.

        finally:
            # This block always runs, ensuring logging and consistent return value setting.
            direct_duration_ms_final = (time.monotonic() - direct_start_time) * 1000.0
            interaction_data_for_log['execution_time_ms'] = direct_duration_ms_final

            # If an exception occurred and response_to_return_to_client wasn't set by the hook or LLM success,
            # it would still hold its default error or the error set in the except block.
            # Ensure llm_response in log data matches the final response_to_return_to_client.
            if interaction_data_for_log.get('llm_response') != response_to_return_to_client:
                # This might happen if an error occurred after the hook check but before LLM set it,
                # or if an error occurred in the LLM path and response_to_return_to_client was updated by an except block.
                interaction_data_for_log['llm_response'] = response_to_return_to_client

            logger.info(
                f"{log_prefix} Direct Generate END. Duration: {direct_duration_ms_final:.2f}ms. Final Resp Snippet: '{str(interaction_data_for_log.get('llm_response', ''))[:70]}...'. Class: {interaction_data_for_log.get('classification')}")

            try:
                # Use queue_interaction_for_batch_logging for this frequent log
                queue_interaction_for_batch_logging(**interaction_data_for_log)
                logger.trace(f"{log_prefix} Queued final direct_generate interaction for batch logging.")
            except Exception as log_err_final_q:
                logger.error(f"❌ {log_prefix}: Failed to QUEUE final direct_generate interaction: {log_err_final_q}")
                # Fallback to direct synchronous write if queueing fails.
                # This needs a new DB session as `db` from route might be closed or in bad state.
                fallback_db_session_for_log: Optional[Session] = None
                try:
                    if SessionLocal:
                        fallback_db_session_for_log = SessionLocal()  # type: ignore
                        # Use the direct add_interaction which handles its own commit
                        add_interaction(fallback_db_session_for_log, **interaction_data_for_log)
                        logger.warning(
                            f"{log_prefix} Logged direct_generate interaction DIRECTLY due to queueing failure.")
                    else:
                        logger.error(f"{log_prefix} SessionLocal not available for fallback direct log.")
                except Exception as fallback_direct_log_err:
                    logger.error(f"❌ {log_prefix}: Fallback direct logging ALSO FAILED: {fallback_direct_log_err}")
                finally:
                    if fallback_db_session_for_log:
                        fallback_db_session_for_log.close()

        return response_to_return_to_client

    async def _get_vector_search_file_index_context(self, query: str, priority: int = ELP0, stop_event_param: Optional[threading.Event] = None) -> str:
        """
        Performs a vector similarity search on the global file index vector store.
        If no vector results are found, attempts a fuzzy search on the SQL FileIndex table as a fallback.
        Formats the results. Explicitly uses _embed_texts for prioritized query embedding.
        """
        log_prefix = f"🔍 FileVecSearch|ELP{priority}|{self.current_session_id or 'NoSession'}"
        logger.debug(f"{log_prefix} Attempting file search for query: '{query[:50]}...'")

        global_file_vs = get_global_file_index_vectorstore() # Synchronous call

        # --- Vector Search Attempt ---
        vector_search_succeeded = False
        search_results_docs: List[Any] = [] # Will hold Langchain Document objects

        if not global_file_vs:
            logger.warning(f"{log_prefix} Global file index vector store not available for vector search.")
        elif not self.provider or not self.provider.embeddings:
            logger.error(f"{log_prefix} Embeddings provider not available for vector search query.")
        elif not query:
            logger.debug(f"{log_prefix} Empty query for vector search. Skipping vector part.")
        else:
            query_vector: Optional[List[float]] = None
            try:
                logger.debug(f"{log_prefix} Explicitly embedding query via _embed_texts with priority ELP{priority}...")
                if hasattr(self.provider.embeddings, '_embed_texts') and \
                        callable(getattr(self.provider.embeddings, '_embed_texts')):
                    embedding_result_list = await asyncio.to_thread(
                        self.provider.embeddings._embed_texts, [query], priority=priority # type: ignore
                    )
                    if embedding_result_list and len(embedding_result_list) > 0:
                        query_vector = embedding_result_list[0]
                    else:
                        logger.error(f"{log_prefix} _embed_texts returned None or empty list for query.")
                else:
                    logger.error(f"{log_prefix} Embeddings object missing '_embed_texts'. Cannot perform prioritized query embedding.")

                if not query_vector:
                    logger.error(f"{log_prefix} Failed to embed query for vector search (query_vector is None).")
                else:
                    logger.debug(f"{log_prefix} Query embedded. Performing similarity_search_by_vector (k={RAG_FILE_INDEX_COUNT})...")
                    # Perform search using the pre-computed vector
                    search_results_docs = await asyncio.to_thread(
                        global_file_vs.similarity_search_by_vector,
                        embedding=query_vector,
                        k=RAG_FILE_INDEX_COUNT  # From config
                    )
                    if search_results_docs:
                        vector_search_succeeded = True
                        logger.info(f"{log_prefix} Found {len(search_results_docs)} results from VECTOR file search.")
                    else:
                        logger.info(f"{log_prefix} No results from VECTOR file search for query '{query[:50]}...'")

            except TaskInterruptedException as tie:
                logger.warning(f"🚦 {log_prefix} Vector file search INTERRUPTED: {tie}")
                raise # Re-raise to be handled by the caller
            except Exception as e:
                logger.error(f"❌ {log_prefix} Error during vector file search: {e}")
                logger.exception(f"{log_prefix} Vector File Search Traceback:")
                # Continue to fuzzy search fallback

        # --- Fuzzy Search Fallback ---
        fuzzy_search_results_text_list: List[str] = []
        if not vector_search_succeeded:
            if not FUZZY_AVAILABLE:
                logger.warning(f"{log_prefix} Vector search failed and Fuzzy search (thefuzz) is not available. No file context.")
                return "No relevant file content found (vector search failed, fuzzy search unavailable)."

            logger.info(f"{log_prefix} Vector search yielded no results. Attempting FUZZY search fallback for query: '{query[:50]}...'")
            db_for_fuzzy: Optional[Session] = None
            try:
                db_for_fuzzy = SessionLocal() # type: ignore
                if not db_for_fuzzy: raise RuntimeError("Failed to get DB session for fuzzy search.")

                # Fetch a reasonable number of candidates from SQL to perform fuzzy search on
                # Limiting this to avoid loading too much into memory.
                # We search against file_name and indexed_content (if not too long).
                # Order by last_modified_os to potentially get more relevant recent files.
                candidate_records = db_for_fuzzy.query(FileIndex).filter(
                    FileIndex.index_status.in_(['indexed_text', 'success', 'partial_vlm_error']) # Only search indexed files
                ).order_by(desc(FileIndex.last_modified_os)).limit(500).all() # Limit candidates

                if not candidate_records:
                    logger.info(f"{log_prefix} FUZZY: No candidate records in SQL DB for fuzzy search.")
                else:
                    logger.debug(f"{log_prefix} FUZZY: Found {len(candidate_records)} candidate records from SQL.")
                    fuzzy_matches: List[Tuple[FileIndex, int]] = [] # Store (record, score)

                    for record in candidate_records:
                        if stop_event_param and stop_event_param.is_set():  # Check if passed and set
                            logger.info(f"{log_prefix} FUZZY search interrupted by stop_event_param.")
                            break
                        # Text to search against: filename + content snippet
                        text_to_match_on = record.file_name or ""
                        if record.indexed_content:
                            # Use a snippet of content to keep fuzzy search performant
                            content_snippet = (record.indexed_content[:500] + "...") if len(record.indexed_content) > 500 else record.indexed_content
                            text_to_match_on += " " + content_snippet

                        if not text_to_match_on.strip(): continue

                        # Use fuzz.partial_ratio for substring matching, good for finding queries within larger text
                        score = fuzz.partial_ratio(query.lower(), text_to_match_on.lower())

                        if score >= FUZZY_SEARCH_THRESHOLD_APP: # FUZZY_SEARCH_THRESHOLD_APP from app.py/config
                            fuzzy_matches.append((record, score))

                    if fuzzy_matches:
                        # Sort by score descending, then by last_modified_os descending
                        fuzzy_matches.sort(key=lambda x: (x[1], x[0].last_modified_os or datetime.datetime.min), reverse=True)
                        top_fuzzy_matches = fuzzy_matches[:RAG_FILE_INDEX_COUNT] # Take top N
                        logger.info(f"{log_prefix} FUZZY: Found {len(top_fuzzy_matches)} matches with score >= {FUZZY_SEARCH_THRESHOLD_APP}.")

                        for i, (record, score) in enumerate(top_fuzzy_matches):
                            content_snippet = (record.indexed_content[:300] + "...") if record.indexed_content and len(record.indexed_content) > 300 else (record.indexed_content or "[No content]")
                            entry = (
                                f"--- Fuzzy File Result {i + 1} (Score: {score}) ---\n"
                                f"File: {record.file_name}\nPath Hint: ...{record.file_path[-70:]}\nModified: {record.last_modified_os.strftime('%Y-%m-%d %H:%M') if record.last_modified_os else 'N/A'}\n"
                                f"Content Snippet: {content_snippet}\n---\n"
                            )
                            fuzzy_search_results_text_list.append(entry)
                    else:
                        logger.info(f"{log_prefix} FUZZY: No matches found above threshold {FUZZY_SEARCH_THRESHOLD_APP}.")

            except Exception as e_fuzzy:
                logger.error(f"❌ {log_prefix} Error during FUZZY search: {e_fuzzy}")
                logger.exception(f"{log_prefix} Fuzzy Search Traceback:")
                fuzzy_search_results_text_list.append(f"[Error performing fuzzy file search: {type(e_fuzzy).__name__}]\n")
            finally:
                if db_for_fuzzy: db_for_fuzzy.close()

        # --- Format Results ---
        if vector_search_succeeded and search_results_docs:
            context_parts = []
            max_snippet_len = 300
            max_total_chars = 2000 # Max length for combined vector context
            current_chars = 0
            for i, doc in enumerate(search_results_docs):
                if not hasattr(doc, 'page_content') or not hasattr(doc, 'metadata'):
                    logger.warning(f"{log_prefix} Skipping malformed vector document: {doc}")
                    continue
                content = doc.page_content
                metadata = doc.metadata
                file_path = metadata.get("source", "UnkPath")
                file_name = metadata.get("file_name", os.path.basename(file_path) if file_path != "UnkPath" else "UnkFile")
                last_mod = metadata.get("last_modified", "UnkDate")
                # Langchain Chroma typically returns relevance_score which is distance (lower is better).
                # We can invert it or just display as is.
                relevance_score = doc.metadata.get('relevance_score', 'N/A') if isinstance(doc.metadata, dict) else 'N/A'

                snippet = content[:max_snippet_len] + ("..." if len(content) > max_snippet_len else "")
                entry = (
                    f"--- Vector File Result {i + 1} (Score: {relevance_score}) ---\n" # Score might be distance
                    f"File: {file_name}\nPath Hint: ...{file_path[-70:]}\nModified: {last_mod}\n"
                    f"Content Snippet: {snippet}\n---\n")
                if current_chars + len(entry) > max_total_chars:
                    context_parts.append("[Vector file search context truncated due to length]...\n")
                    break
                context_parts.append(entry)
                current_chars += len(entry)
            return "".join(context_parts) if context_parts else "No relevant file content found via vector search."
        elif fuzzy_search_results_text_list:
            # Combine fuzzy results, already formatted as text strings
            # Limit total length of fuzzy results string for the prompt
            combined_fuzzy_text = "".join(fuzzy_search_results_text_list)
            max_fuzzy_chars = 2000 # Max length for combined fuzzy context
            if len(combined_fuzzy_text) > max_fuzzy_chars:
                return combined_fuzzy_text[:max_fuzzy_chars] + "\n[Fuzzy file search context truncated due to length]...\n"
            return combined_fuzzy_text
        else:
            # Neither vector nor fuzzy search yielded results
            return "No relevant file content found via vector or fuzzy search for the query."

    async def background_generate(self, db: Session, user_input: str, session_id: str = None,  # type: ignore
                                  classification: str = "chat_simple", image_b64: Optional[str] = None,
                                  update_interaction_id: Optional[int] = None,  # For reflection tasks
                                  stop_event_for_bg: Optional[threading.Event] = None):  # For external stop signal

        request_id = f"bgen-{uuid.uuid4()}"
        is_reflection_task = update_interaction_id is not None
        log_prefix_base = "🔄 REFLECT" if is_reflection_task else "💬 BGEN"
        log_prefix = f"{log_prefix_base} {request_id}|ELP0"

        if not session_id:
            session_id = f"session_{int(time.monotonic())}" if not is_reflection_task else f"reflection_on_{update_interaction_id}_{str(uuid.uuid4())[:4]}"

        original_chat_session_id = self.current_session_id
        self.current_session_id = session_id

        logger.info(
            f"{log_prefix} Async Background Generate (ELP0 Pipeline) START --> Session: {session_id}, "
            f"Initial Class: '{classification}', Input: '{user_input[:50]}...', Img: {'Yes' if image_b64 else 'N'}, "
            f"Reflection Target ID: {update_interaction_id if is_reflection_task else 'N/A'}"
        )
        request_start_time = time.monotonic()

        interaction_data: Dict[str, Any] = {
            "session_id": session_id, "mode": "chat", "input_type": "text",
            "user_input": user_input, "llm_response": "[Processing background task...]",
            "execution_time_ms": 0, "classification": classification, "classification_reason": None,
            "rag_history_ids": None, "rag_source_url": None,
            "requires_deep_thought": (classification == "chat_complex") or is_reflection_task,
            "deep_thought_reason": None, "tot_analysis_requested": False,
            "tot_analysis_spawned": False, "tot_result": None, "tot_delivered": False,
            "emotion_context_analysis": None, "image_description": None,
            "assistant_action_analysis_json": None, "assistant_action_type": None,
            "assistant_action_params": None, "assistant_action_executed": False,
            "assistant_action_result": None,
            "image_data": image_b64[:20] + "..." if image_b64 and isinstance(image_b64, str) else None,
            "imagined_image_prompt": None, "imagined_image_b64": None,
            "imagined_image_vlm_description": None,
            "reflection_completed": False,  # For original interaction if this is a reflection
            "reflection_indexed_in_vs": False
        }
        if image_b64: interaction_data["input_type"] = "image+text"

        final_response_text_for_this_turn = "Error: Background processing did not complete as expected."
        saved_initial_interaction: Optional[Interaction] = None
        original_interaction_to_update_for_reflection: Optional[Interaction] = None
        interrupted_flag = False

        if not user_input and not image_b64:
            logger.warning(f"{log_prefix} Empty input (no text, no image). Aborting background_generate.")
            self.current_session_id = original_chat_session_id
            return

        if is_reflection_task:
            try:
                original_interaction_to_update_for_reflection = await asyncio.to_thread(
                    db.query(Interaction).filter(Interaction.id == update_interaction_id).first
                )
                if not original_interaction_to_update_for_reflection:
                    logger.error(
                        f"{log_prefix}: CRITICAL - Original reflection target ID {update_interaction_id} not found.")
                    self.current_session_id = original_chat_session_id
                    return
                interaction_data["classification_reason"] = "Self-reflection task, proceeding with deep analysis."
                logger.info(f"{log_prefix}: Loaded original interaction {update_interaction_id} for reflection.")
            except Exception as e_load_orig:
                logger.error(f"{log_prefix}: Error loading reflection target ID {update_interaction_id}: {e_load_orig}")
                self.current_session_id = original_chat_session_id
                return

        if not is_reflection_task:
            try:
                initial_save_data = interaction_data.copy()
                initial_save_data['llm_response'] = "[Background task initiated... Processing...]"
                initial_save_data['execution_time_ms'] = (time.monotonic() - request_start_time) * 1000
                valid_keys_init = {c.name for c in Interaction.__table__.columns}
                db_kwargs_init = {k: v for k, v in initial_save_data.items() if k in valid_keys_init and k != 'id'}
                saved_initial_interaction = await asyncio.to_thread(add_interaction, db, **db_kwargs_init)
                await asyncio.to_thread(db.commit)
                if saved_initial_interaction and saved_initial_interaction.id:
                    logger.info(
                        f"{log_prefix}: Saved initial 'pending' Interaction ID {saved_initial_interaction.id} for this background task.")
                else:
                    logger.error(f"{log_prefix}: Failed to save initial 'pending' interaction to DB.")
            except Exception as e_initial_save:
                logger.error(f"{log_prefix}: Error saving initial 'pending' interaction: {e_initial_save}")
                if db: await asyncio.to_thread(db.rollback)

        current_input_for_llm_analysis = user_input

        try:
            # --- VLM Preprocessing ---
            if image_b64:
                logger.info(f"{log_prefix} User image provided. Requesting VLM description (ELP0)...")
                vlm_model_user = self.provider.get_model("vlm")
                if vlm_model_user:
                    try:
                        user_img_uri = f"data:image/png;base64,{image_b64}"
                        img_content_part = {"type": "image_url", "image_url": {"url": user_img_uri}}
                        vlm_desc_prompt = "Describe this image concisely for conversational context."
                        vlm_messages = [
                            HumanMessage(content=[img_content_part, {"type": "text", "text": vlm_desc_prompt}])]
                        vlm_chain = vlm_model_user | StrOutputParser()
                        timing_data_vlm = {"session_id": session_id, "mode": "vlm_preprocessing",
                                           "execution_time_ms": 0}

                        description = await asyncio.to_thread(  # _call_llm_with_timing is sync
                            self._call_llm_with_timing, vlm_chain, vlm_messages, timing_data_vlm, ELP0
                        )
                        interaction_data['image_description'] = description
                        current_input_for_llm_analysis = f"[Image Description (User Provided): {description}]\n\nUser Query: {user_input or '(User queried about the image)'}"
                        logger.info(
                            f"{log_prefix}: VLM desc obtained. New input for analysis: '{current_input_for_llm_analysis[:70]}...'")
                    except TaskInterruptedException as tie_vlm:
                        logger.warning(f"{log_prefix}: VLM preprocessing interrupted: {tie_vlm}")
                        interaction_data['image_description'] = "[VLM Preprocessing Interrupted]"
                        if not user_input: raise tie_vlm
                    except Exception as e_vlm_user:
                        logger.error(f"{log_prefix}: Error VLM preprocessing: {e_vlm_user}")
                        interaction_data['image_description'] = f"[VLM Error: {e_vlm_user}]"
                else:
                    interaction_data['image_description'] = "[VLM Model Unavailable]"
                    logger.error(f"{log_prefix}: VLM model unavailable.")

            # --- Initial Classification (if not reflection) ---
            classification_for_current_task = classification  # Use initial if reflection
            if not is_reflection_task:
                logger.info(f"{log_prefix}: Classifying input for background task (ELP0)...")
                classification_data_for_bg = {"session_id": session_id, "mode": "chat",
                                              "input_type": "bg_classification"}
                # ** CORRECTED CALL: Direct await for async helper **
                classification_for_current_task = await self._classify_input_complexity(
                    db, current_input_for_llm_analysis or user_input, classification_data_for_bg
                )
                logger.info(
                    f"{log_prefix}: Input classified as '{classification_for_current_task}'. Reason: {classification_data_for_bg.get('classification_reason', 'N/A')}")
                interaction_data['classification'] = classification_for_current_task
                interaction_data['classification_reason'] = classification_data_for_bg.get('classification_reason')

            # --- Gather Contexts (RAG, History, Logs, Emotion, Files) ---
            logger.debug(f"{log_prefix}: Gathering contexts (ELP0 for sub-tasks)...")
            # (Ensure your full context gathering logic is here, using `await asyncio.to_thread` for sync DB/file ops
            # and direct `await` for async helpers like `_generate_file_search_query_async` and `_get_vector_search_file_index_context`)
            # For brevity, this part is condensed. It populates:
            # hist_summary_action, log_ctx_prompt, direct_hist_prompt,
            # url_ctx_for_router, sess_refl_router, file_ctx_for_router,
            # url_ctx_untruncated, sess_refl_rag_untrunc, main_dynamic_ctx_block_for_llm,
            # emotion_analysis_str, imagined_img_vlm_desc_this_turn (from interaction_data)
            # Placeholder values for demonstration:
            hist_summary_action = await asyncio.to_thread(self._get_history_summary, db, MEMORY_SIZE)
            temp_global_hist_kw = await asyncio.to_thread(get_global_recent_interactions, db, limit=5)
            direct_hist_prompt = self._format_direct_history(temp_global_hist_kw)
            keyword_file_query = await self._generate_file_search_query_async(db, current_input_for_llm_analysis,
                                                                              direct_hist_prompt, session_id)
            kw_file_results_list_res = await asyncio.to_thread(search_file_index, db, keyword_file_query,
                                                               RAG_FILE_INDEX_COUNT) if search_file_index and keyword_file_query else []
            kw_file_ctx_untrunc = self._format_file_index_results(kw_file_results_list_res)
            vec_file_ctx_result_str = await self._get_vector_search_file_index_context(current_input_for_llm_analysis,
                                                                                       ELP0, stop_event_for_bg)
            log_entries_list_res = await asyncio.to_thread(get_recent_interactions, db, RAG_HISTORY_COUNT * 2,
                                                           session_id, "chat", True)
            log_ctx_prompt = self._format_log_history(log_entries_list_res)
            emotion_analysis_str = await asyncio.to_thread(self._run_emotion_analysis, db, user_input,
                                                           interaction_data)  # This _run_emotion_analysis itself is sync
            imagined_img_vlm_desc_this_turn = interaction_data.get('image_description')  # VLM of user image

            # RAG retrievers (sync setup, async invoke if retriever supports it)
            wrapped_rag_res = await asyncio.to_thread(self._get_rag_retriever_thread_wrapper, db,
                                                      current_input_for_llm_analysis or user_input, ELP0)
            url_ret_obj, sess_hist_ret_obj, refl_chunk_ret_obj, sess_chat_rag_ids = None, None, None, ""
            if wrapped_rag_res.get("status") == "success":
                url_ret_obj, sess_hist_ret_obj, refl_chunk_ret_obj, sess_chat_rag_ids = wrapped_rag_res["data"]
            elif wrapped_rag_res.get("status") == "interrupted":
                raise TaskInterruptedException(wrapped_rag_res.get("error_message"))
            else:
                raise RuntimeError(f"RAG retriever init failed: {wrapped_rag_res.get('error_message')}")
            interaction_data['rag_history_ids'] = sess_chat_rag_ids
            if hasattr(self, 'vectorstore_url') and self.vectorstore_url: interaction_data['rag_source_url'] = getattr(
                self.vectorstore_url, '_source_url', None)

            url_docs, session_docs, reflection_docs = [], [], []
            if url_ret_obj: url_docs = await asyncio.to_thread(url_ret_obj.invoke,
                                                               current_input_for_llm_analysis or user_input)
            if sess_hist_ret_obj: session_docs = await asyncio.to_thread(sess_hist_ret_obj.invoke,
                                                                         current_input_for_llm_analysis or user_input)
            if refl_chunk_ret_obj: reflection_docs = await asyncio.to_thread(refl_chunk_ret_obj.invoke,
                                                                             current_input_for_llm_analysis or user_input)

            url_ctx_untruncated = self._format_docs(url_docs, "URL Context")
            sess_refl_rag_untrunc = self._format_docs(session_docs + reflection_docs, "Session/Reflection RAG")
            # Token budgeting as in your file to prepare main_dynamic_ctx_block_for_llm
            # and context_for_router (url_ctx_for_router, sess_refl_router, file_ctx_for_router)
            # This logic is complex and omitted here for brevity but should be present
            main_dynamic_ctx_block_for_llm = "[Placeholder for combined, truncated dynamic context]"
            url_ctx_for_router = url_ctx_untruncated[:500]  # Placeholder
            sess_refl_router = sess_refl_rag_untrunc[:500]  # Placeholder
            file_ctx_for_router = kw_file_ctx_untrunc[:500] + vec_file_ctx_result_str[:500]  # Placeholder
            logger.debug(f"{log_prefix}: Contexts gathered and formatted.")

            # --- Action Analysis ---
            action_payload_context = {"history_summary": hist_summary_action, "log_context": log_ctx_prompt,
                                      "recent_direct_history": direct_hist_prompt,
                                      "file_index_context": file_ctx_for_router}
            logger.info(f"{log_prefix}: Calling _analyze_assistant_action (ELP0)...")
            # ** CORRECTED CALL: Direct await for async helper **
            action_details = await self._analyze_assistant_action(
                db, current_input_for_llm_analysis, session_id, action_payload_context
            )
            detected_action_type = action_details.get("action_type", "no_action") if action_details and isinstance(
                action_details, dict) else "no_action"
            if action_details and isinstance(action_details, dict):
                interaction_data.update({
                    'assistant_action_analysis_json': json.dumps(action_details),
                    'assistant_action_type': detected_action_type,
                    'assistant_action_params': json.dumps(action_details.get("parameters", {}))
                })
            logger.info(f"{log_prefix}: Action analysis complete. Type: '{detected_action_type}'.")

            # --- Main Logic Flow (Action / Imagine / Standard LLM response) ---
            if detected_action_type == "imagine" and action_details:
                interaction_data['assistant_action_executed'] = True
                idea_to_viz = action_details.get("parameters", {}).get("idea_to_visualize", user_input)
                # _generate_image_generation_prompt_async is async
                img_prompt = await self._generate_image_generation_prompt_async(db, session_id, user_input, idea_to_viz,
                                                                                sess_refl_router, file_ctx_for_router,
                                                                                direct_hist_prompt, url_ctx_for_router,
                                                                                log_ctx_prompt)
                interaction_data['imagined_image_prompt'] = img_prompt
                if img_prompt:
                    # provider.generate_image_async is async
                    img_data_list, img_err = await self.provider.generate_image_async(img_prompt, None, ELP0)
                    if img_err:
                        final_response_text_for_this_turn = f"Imagine Error: {img_err}"
                    elif img_data_list and img_data_list[0].get("b64_json"):
                        interaction_data['imagined_image_b64'] = img_data_list[0]["b64_json"]
                        # _describe_generated_image_async is async
                        desc = await self._describe_generated_image_async(db, session_id, img_data_list[0]["b64_json"])
                        interaction_data['imagined_image_vlm_description'] = desc
                        imagined_img_vlm_desc_this_turn = desc  # Update for subsequent LLM if any
                        final_response_text_for_this_turn = f"I've imagined that: {desc or 'Generated an image.'}"
                    else:
                        final_response_text_for_this_turn = "Imagine Error: No image data."
                else:
                    final_response_text_for_this_turn = "Imagine Error: Prompt gen failed."
                interaction_data['assistant_action_result'] = final_response_text_for_this_turn

            elif detected_action_type != "no_action" and action_details:
                target_interaction_for_action = saved_initial_interaction if not is_reflection_task else original_interaction_to_update_for_reflection
                if target_interaction_for_action:
                    # _execute_assistant_action is async
                    final_response_text_for_this_turn = await self._execute_assistant_action(db, session_id,
                                                                                             action_details,
                                                                                             target_interaction_for_action)  # type: ignore
                else:
                    final_response_text_for_this_turn = "Error: Missing interaction context for action."
                interaction_data['assistant_action_result'] = final_response_text_for_this_turn

            else:  # Standard LLM response
                router_payload_llm_for_specialist = {  # Renamed to avoid conflict
                    "input": current_input_for_llm_analysis, "recent_direct_history": direct_hist_prompt,
                    "context": url_ctx_for_router, "history_rag": sess_refl_router,
                    "file_index_context": file_ctx_for_router, "log_context": log_ctx_prompt,
                    "emotion_analysis": emotion_analysis_str,
                    "imagined_image_vlm_description": imagined_img_vlm_desc_this_turn or interaction_data.get(
                        'image_description', 'None.')  # Use VLM desc of user image if any
                }
                # _route_to_specialist is async
                chosen_model_role, refined_query_for_specialist, route_reason_str = await self._route_to_specialist(
                    db, session_id, current_input_for_llm_analysis, router_payload_llm_for_specialist
                )
                interaction_data['classification_reason'] = f"Routed to {chosen_model_role}: {route_reason_str}"

                specialist_model_instance = self.provider.get_model(chosen_model_role)
                if not specialist_model_instance: raise ValueError(
                    f"Specialist model for role '{chosen_model_role}' unavailable.")

                specialist_payload_final_llm = {  # Prepare full context for specialist
                    "input": refined_query_for_specialist, "emotion_analysis": emotion_analysis_str,
                    "context": url_ctx_untruncated, "history_rag": sess_refl_rag_untrunc,
                    "file_index_context": main_dynamic_ctx_block_for_llm,  # Fuller context for specialist
                    "log_context": log_ctx_prompt, "recent_direct_history": direct_hist_prompt,
                    "pending_tot_result": interaction_data.get("tot_result", "None."),  # Pass ToT result if available
                    "imagined_image_vlm_description": imagined_img_vlm_desc_this_turn or interaction_data.get(
                        'image_description', 'None.')
                }

                specialist_chain = (self.text_prompt_template | specialist_model_instance | StrOutputParser())
                timing_data_specialist = {"session_id": session_id, "mode": "chat_specialist"}

                draft_response = await asyncio.to_thread(  # _call_llm_with_timing is sync
                    self._call_llm_with_timing, specialist_chain, {}, specialist_payload_final_llm,
                    timing_data_specialist, priority=ELP0
                )

                # _correct_response is async
                final_response_text_for_this_turn = await self._correct_response(
                    db, session_id, current_input_for_llm_analysis, specialist_payload_final_llm, draft_response
                )

            # --- ToT Spawning (using classification_for_current_task) ---
            if not is_reflection_task and \
                    (classification_for_current_task == "chat_complex" or interaction_data.get(
                        "requires_deep_thought")):
                trigger_id_for_tot = saved_initial_interaction.id if saved_initial_interaction else None
                if trigger_id_for_tot:
                    logger.info(f"{log_prefix} Spawning background ToT for Interaction ID: {trigger_id_for_tot}.")
                    # Ensure all necessary context variables for ToT are defined here from the context gathering phase
                    # Example: url_docs_for_prompt_list, session_hist_docs_for_prompt_list, reflection_chunk_docs_for_prompt_list
                    # main_dynamic_ctx_block_for_llm, imagined_img_vlm_desc_this_turn or interaction_data.get('image_description')
                    # These need to be correctly passed to _run_tot_in_background_wrapper_v2
                    tot_payload = {
                        "db_session_factory": SessionLocal,
                        "original_input_for_tot": current_input_for_llm_analysis,  # Or user_input
                        "rag_context_docs": url_docs,  # Ensure these are populated from RAG results
                        "history_rag_interactions": session_docs + reflection_docs,  # Ensure populated
                        "log_context_str": log_ctx_prompt,
                        "recent_direct_history_str": direct_hist_prompt,
                        "file_index_context_str": main_dynamic_ctx_block_for_llm,  # Or specific file contexts
                        "imagined_image_context_str": imagined_img_vlm_desc_this_turn or interaction_data.get(
                            'image_description') or "None.",
                        "triggering_interaction_id": trigger_id_for_tot,
                    }
                    asyncio.create_task(self._run_tot_in_background_wrapper_v2(**tot_payload))  # type: ignore
                    interaction_data['tot_analysis_spawned'] = True
                    # Update the triggering interaction to note ToT was spawned
                    trigger_interaction_for_tot_update = await asyncio.to_thread(
                        db.query(Interaction).filter(Interaction.id == trigger_id_for_tot).first)
                    if trigger_interaction_for_tot_update:
                        trigger_interaction_for_tot_update.tot_analysis_spawned = True  # type: ignore
                        trigger_interaction_for_tot_update.requires_deep_thought = True  # type: ignore
                        trigger_interaction_for_tot_update.deep_thought_reason = interaction_data.get(
                            'classification_reason', 'Complex query, ToT spawned.')  # type: ignore
                        await asyncio.to_thread(db.commit)


        except TaskInterruptedException as tie:
            # ... (existing TaskInterruptedException handling) ...
            logger.warning(f"🚦 {log_prefix} Background Generate Task INTERRUPTED: {tie}")
            interrupted_flag = True
            final_response_text_for_this_turn = f"[Background task interrupted by higher priority request: {tie}]"
            interaction_data.update(
                {'llm_response': final_response_text_for_this_turn, 'classification': "task_failed_interrupted",
                 'input_type': 'log_warning'})
        except Exception as e_bg_gen:
            # ... (existing general Exception handling) ...
            logger.error(f"❌❌ {log_prefix} UNHANDLED exception in background_generate: {e_bg_gen}")
            logger.exception(f"{log_prefix} Background Generate Main Traceback:")
            final_response_text_for_this_turn = f"Error during background processing: {type(e_bg_gen).__name__} - {str(e_bg_gen)}"
            interaction_data.update({'llm_response': final_response_text_for_this_turn[:4000], 'input_type': 'error'})

        finally:
            # --- Final DB Update Logic ---
            final_db_data_to_save = interaction_data.copy()
            final_response_text_cleaned = self._cleanup_llm_output(final_response_text_for_this_turn)
            final_db_data_to_save['llm_response'] = final_response_text_cleaned
            final_db_data_to_save['execution_time_ms'] = (time.monotonic() - request_start_time) * 1000
            if 'last_modified_db' in final_db_data_to_save: del final_db_data_to_save['last_modified_db']
            if final_db_data_to_save.get('imagined_image_b64') and len(
                    final_db_data_to_save['imagined_image_b64']) > 1000000:
                final_db_data_to_save['imagined_image_b64'] = final_db_data_to_save['imagined_image_b64'][
                                                              :100] + f"...[base64_truncated]"

            logger.debug(
                f"{log_prefix}: Preparing to save final DB state. Interrupted: {interrupted_flag}. Is Reflection: {is_reflection_task}.")
            try:
                if is_reflection_task and original_interaction_to_update_for_reflection:
                    if interrupted_flag or final_db_data_to_save.get('input_type') == 'error':
                        original_interaction_to_update_for_reflection.reflection_completed = False
                        note = f"\n\n--- Reflection Task (ID: {request_id}) {'Interrupted' if interrupted_flag else 'Errored'} ({datetime.datetime.now(datetime.timezone.utc).isoformat()}): {final_response_text_cleaned} ---"
                        current_resp = getattr(original_interaction_to_update_for_reflection, 'llm_response', "") or ""
                        original_interaction_to_update_for_reflection.llm_response = (current_resp + note)[:getattr(
                            Interaction.llm_response.type, 'length', 4000)]
                    else:
                        original_interaction_to_update_for_reflection.reflection_completed = True
                    await asyncio.to_thread(db.commit)
                    logger.info(
                        f"{log_prefix}: Updated original interaction {update_interaction_id} for reflection state.")

                    if not (interrupted_flag or final_db_data_to_save.get('input_type') == 'error'):
                        new_reflection_data = final_db_data_to_save.copy()
                        new_reflection_data.update({'input_type': "reflection_result",
                                                    'user_input': f"[Reflection for OrigID {update_interaction_id}]",
                                                    'classification': "reflection_output", 'reflection_completed': True,
                                                    'tot_analysis_spawned': final_db_data_to_save.get(
                                                        'tot_analysis_spawned', False)})
                        valid_keys_refl = {c.name for c in Interaction.__table__.columns}
                        db_kwargs_refl = {k: v for k, v in new_reflection_data.items() if
                                          k in valid_keys_refl and k != 'id'}
                        reflection_record = await asyncio.to_thread(add_interaction, db, **db_kwargs_refl)  # Commits
                        if reflection_record and reflection_record.id and self.provider and self.provider.embeddings:
                            await asyncio.to_thread(index_single_reflection, reflection_record, self.provider, db,
                                                    ELP0)  # index_single_reflection is sync

                elif not is_reflection_task and saved_initial_interaction:
                    logger.debug(
                        f"{log_prefix}: Updating initial interaction {saved_initial_interaction.id} with final results.")
                    for key, value in final_db_data_to_save.items():
                        if hasattr(saved_initial_interaction, key) and key != 'id':
                            setattr(saved_initial_interaction, key, value)
                    await asyncio.to_thread(db.commit)
                elif not is_reflection_task and not saved_initial_interaction:  # Initial save failed
                    logger.warning(f"{log_prefix}: Initial placeholder save failed. Saving final state as new record.")
                    valid_keys_final_new = {c.name for c in Interaction.__table__.columns}
                    db_kwargs_final_new = {k: v for k, v in final_db_data_to_save.items() if
                                           k in valid_keys_final_new and k != 'id'}
                    await asyncio.to_thread(add_interaction, db, **db_kwargs_final_new)
            except Exception as final_db_save_err:
                logger.error(f"❌ {log_prefix}: CRITICAL error during final DB save/update: {final_db_save_err}")
                if db: await asyncio.to_thread(db.rollback)

            self.current_session_id = original_chat_session_id
            logger.info(
                f"{log_prefix} Async Background Generate (ELP0 Pipeline) END. Duration: {final_db_data_to_save.get('execution_time_ms', 0):.2f}ms")
            # The DB session `db` is managed by the caller (e.g., run_background_task_with_new_loop or run_self_reflection_loop)
            # and should be closed there.

    def _is_valid_tot_json(self, parsed_json: Any) -> bool:
        """Checks if the parsed JSON object has the required ToT structure."""
        if not isinstance(parsed_json, dict):
            return False
        required_keys = {"decomposition", "brainstorming", "evaluation", "synthesis", "confidence_score"}
        # New optional keys for spawning background task
        optional_keys_for_spawn = {"requires_background_task", "next_task_input"}

        if not required_keys.issubset(parsed_json.keys()):
            logger.warning(f"ToT JSON missing one or more required keys: {required_keys - set(parsed_json.keys())}")
            return False
        if not isinstance(parsed_json["brainstorming"], list):
            logger.warning("ToT JSON 'brainstorming' field is not a list.")
            return False
        if not isinstance(parsed_json["confidence_score"], float):
            logger.warning("ToT JSON 'confidence_score' field is not a float.")
            return False

        # Check types for new optional fields if they exist
        if "requires_background_task" in parsed_json and not isinstance(parsed_json["requires_background_task"], bool):
            logger.warning("ToT JSON 'requires_background_task' is not a boolean.")
            return False
        if "next_task_input" in parsed_json and not (
                parsed_json["next_task_input"] is None or isinstance(parsed_json["next_task_input"], str)):
            logger.warning("ToT JSON 'next_task_input' is not a string or null.")
            return False
        if parsed_json.get("requires_background_task") is True and not parsed_json.get("next_task_input"):
            logger.warning("ToT JSON 'requires_background_task' is true, but 'next_task_input' is missing or empty.")
            # This might still be "valid" structurally but logically flawed for spawning.
            # For validation purposes, we'll allow it, but the spawning logic will skip it.

        return True

    async def _run_tree_of_thought_v2(self, db: Session, input_for_tot: str,
                                      rag_context_docs: List[Any],
                                      history_rag_interactions: List[Any],
                                      log_context_str: str,
                                      recent_direct_history_str: str,
                                      file_index_context_str: str,
                                      imagined_image_context_str: str,
                                      interaction_data_for_tot_llm_call: Dict[str, Any],  # For _call_llm_with_timing
                                      original_user_input_for_log: str,
                                      triggering_interaction_id_for_log: int
                                      ) -> str:  # Returns the 'synthesis' string

        log_prefix = f"🌳 ToT_v2|ELP0|TrigID:{triggering_interaction_id_for_log}"
        current_session_id = interaction_data_for_tot_llm_call.get("session_id",
                                                                   f"tot_session_{triggering_interaction_id_for_log}")
        logger.info(f"{log_prefix} Starting ToT for original input: '{original_user_input_for_log[:50]}...'")

        tot_model = self.provider.get_model("router")  # Use router model for ToT
        if not tot_model:
            error_msg = "ToT model ('router') not available for ToT V2 execution."
            logger.error(f"{log_prefix} {error_msg}")
            await asyncio.to_thread(add_interaction, db, session_id=current_session_id,
                                    mode="internal_error", input_type="log_error",
                                    user_input=f"[ToT V2 Failed - Model Unavailable for TrigID: {triggering_interaction_id_for_log}]",
                                    llm_response=error_msg)
            await asyncio.to_thread(db.commit)
            return f"Error: ToT model unavailable for analysis."

        url_rag_context_str = self._format_docs(rag_context_docs, "URL Context")
        history_rag_context_str = self._format_docs(history_rag_interactions, "History/Reflection RAG")

        llm_input_for_tot = {
            "input": original_user_input_for_log, "context": url_rag_context_str,
            "history_rag": history_rag_context_str, "file_index_context": file_index_context_str,
            "log_context": log_context_str, "recent_direct_history": recent_direct_history_str,
            "imagined_image_context": imagined_image_context_str
        }

        chain = ChatPromptTemplate.from_template(PROMPT_TREE_OF_THOUGHTS_V2) | tot_model | StrOutputParser()

        raw_llm_output_from_initial_loop: str = "Initial ToT LLM call did not yield parsable JSON."
        parsed_tot_json: Optional[Dict[str, Any]] = None
        last_error_initial: Optional[Exception] = None

        # --- Stage 1: Initial LLM Call & Parse/Fix ---
        # For complex ToT JSON, let's keep 1 primary attempt before reformat.
        initial_llm_attempts_tot = 1
        for attempt in range(initial_llm_attempts_tot):
            logger.debug(f"{log_prefix} ToT LLM call attempt {attempt + 1}/{initial_llm_attempts_tot}")
            try:
                raw_llm_text_this_attempt = await asyncio.to_thread(
                    self._call_llm_with_timing, chain, llm_input_for_tot,
                    interaction_data_for_tot_llm_call, priority=ELP0
                )
                raw_llm_output_from_initial_loop = raw_llm_text_this_attempt
                logger.trace(
                    f"{log_prefix} Raw LLM for ToT (Attempt {attempt + 1}): '{raw_llm_text_this_attempt[:200]}...'")

                json_candidate_str = self._extract_json_candidate_string(raw_llm_text_this_attempt,
                                                                         log_prefix + "-ExtractInitial")
                if json_candidate_str:
                    parsed_tot_json = self._programmatic_json_parse_and_fix(
                        json_candidate_str, 1, log_prefix + f"-InitialFixAttempt{attempt + 1}"
                    )
                    if parsed_tot_json and self._is_valid_tot_json(parsed_tot_json):
                        logger.info(f"✅ {log_prefix} Initial ToT analysis successful (Attempt {attempt + 1}).")
                        break  # Success from initial attempt
                else:
                    last_error_initial = ValueError(
                        f"No JSON candidate from ToT LLM: {raw_llm_text_this_attempt[:100]}")

                if not (parsed_tot_json and self._is_valid_tot_json(parsed_tot_json)):  # If not broken from success
                    if parsed_tot_json:
                        last_error_initial = ValueError(f"Invalid ToT JSON structure: {str(parsed_tot_json)[:100]}")
                    elif not json_candidate_str:
                        pass  # Error already set if no candidate
                    else:
                        last_error_initial = ValueError(
                            f"Failed to parse/fix ToT JSON candidate: {json_candidate_str[:100]}")
                    parsed_tot_json = None  # Ensure it's None for next stage

            except TaskInterruptedException as tie:
                logger.warning(f"🚦 {log_prefix} ToT task INTERRUPTED: {tie}")
                raise tie  # Propagate to be handled by _run_tot_in_background_wrapper_v2
            except Exception as e_initial_tot:
                last_error_initial = e_initial_tot

            if last_error_initial and not parsed_tot_json:
                logger.warning(f"⚠️ {log_prefix} Initial ToT LLM/parse attempt failed. Error: {last_error_initial}")

        # --- Stage 2: LLM Re-request for Formatting (if initial attempt failed) ---
        if not (parsed_tot_json and self._is_valid_tot_json(parsed_tot_json)):
            logger.warning(
                f"{log_prefix} Initial ToT attempt failed. Trying LLM re-request to fix format. Last raw: '{raw_llm_output_from_initial_loop[:200]}...'")

            reformat_prompt_input = {
                "faulty_llm_output_for_reformat": raw_llm_output_from_initial_loop,
                "original_user_input_placeholder": original_user_input_for_log
            }
            reformat_chain = ChatPromptTemplate.from_template(
                PROMPT_REFORMAT_TO_TOT_JSON) | tot_model | StrOutputParser()

            reformatted_llm_output_text = await asyncio.to_thread(
                self._call_llm_with_timing, reformat_chain, reformat_prompt_input,
                interaction_data_for_tot_llm_call, priority=ELP0
            )

            if reformatted_llm_output_text and not (
                    isinstance(reformatted_llm_output_text, str) and "ERROR" in reformatted_llm_output_text.upper()):
                logger.info(f"{log_prefix} Received reformatted output from LLM for ToT. Attempting to parse/fix...")
                json_candidate_from_reformat = self._extract_json_candidate_string(reformatted_llm_output_text,
                                                                                   log_prefix + "-ReformatExtract")
                if json_candidate_from_reformat:
                    parsed_tot_json = self._programmatic_json_parse_and_fix(
                        json_candidate_from_reformat, JSON_FIX_RETRY_ATTEMPTS_AFTER_REFORMAT,
                        log_prefix + "-ReformatFix"
                    )
            else:
                logger.error(
                    f"{log_prefix} LLM re-request for ToT JSON formatting failed or returned error: {reformatted_llm_output_text}")

        # --- Process Final Result (parsed_tot_json or fallback) ---
        final_synthesis_for_caller = f"Error: ToT analysis for '{original_user_input_for_log[:30]}...' failed to produce valid structured output."
        tot_json_to_save_str: Optional[str] = None

        if parsed_tot_json and self._is_valid_tot_json(parsed_tot_json):
            logger.success(
                f"✅ {log_prefix} ToT analysis JSON successfully parsed. Synthesis snippet: '{str(parsed_tot_json.get('synthesis'))[:50]}...'")
            final_synthesis_for_caller = str(parsed_tot_json.get("synthesis", "ToT synthesis missing from JSON."))
            try:
                tot_json_to_save_str = json.dumps(parsed_tot_json, indent=2)
            except Exception as e_dump:
                logger.error(f"{log_prefix} Failed to dump parsed_tot_json: {e_dump}")
                tot_json_to_save_str = str(
                    parsed_tot_json)

            # --- New: Check if ToT requires a new background task ---
            if parsed_tot_json.get("requires_background_task") is True:
                next_task_input_str = parsed_tot_json.get("next_task_input")
                if next_task_input_str and isinstance(next_task_input_str, str) and next_task_input_str.strip():
                    logger.info(
                        f"{log_prefix} ToT synthesis requires further background task. Input: '{next_task_input_str[:70]}...'")
                    new_bg_task_session_id = f"sub_task_from_tot_{triggering_interaction_id_for_log}_{str(uuid.uuid4())[:4]}"
                    # Spawn new background_generate task (don't await it here, let it run truly in background)
                    asyncio.create_task(
                        self.background_generate(
                            db=db,  # Use current DB session for spawning, BG task will get its own
                            user_input=next_task_input_str,
                            session_id=new_bg_task_session_id,
                            classification="chat_complex",  # Assume new task is complex
                            image_b64=None,
                            update_interaction_id=None  # It's a new task
                        )
                    )
                    logger.info(
                        f"{log_prefix} Spawned new background_generate task for session {new_bg_task_session_id}")
                else:
                    logger.warning(
                        f"{log_prefix} ToT indicated 'requires_background_task' but 'next_task_input' was missing or empty.")
        else:
            logger.error(
                f"{log_prefix} ❌ All ToT attempts failed to produce valid JSON. Last raw: '{raw_llm_output_from_initial_loop[:200]}...'")
            fallback_json_content = {
                "decomposition": "N/A", "brainstorming": [], "evaluation": "N/A",
                "synthesis": final_synthesis_for_caller, "confidence_score": 0.0,
                "self_critique": f"Failed to parse LLM output for ToT. Last raw: {raw_llm_output_from_initial_loop[:200]}",
                "requires_background_task": False, "next_task_input": None
            }
            try:
                tot_json_to_save_str = json.dumps(fallback_json_content, indent=2)
            except Exception as e_dump_fallback:
                logger.error(
                    f"{log_prefix} Failed to dump fallback ToT JSON: {e_dump_fallback}")
                tot_json_to_save_str = str(
                    fallback_json_content)

        # --- Save ToT result (the JSON string) to a new Interaction record ---
        try:
            tot_result_interaction_data = {
                "session_id": current_session_id,
                "mode": "chat",
                "input_type": "tot_result",
                "user_input": f"[ToT Analysis Result for Original Query ID {triggering_interaction_id_for_log}: '{original_user_input_for_log[:100]}...']",
                "llm_response": tot_json_to_save_str,
                "classification": "tot_output_json",
                "execution_time_ms": interaction_data_for_tot_llm_call.get("execution_time_ms", 0),
                "reflection_completed": True, "tot_analysis_requested": False,
                "tot_analysis_spawned": parsed_tot_json.get("requires_background_task") if parsed_tot_json else False,
                # Log if it tried to spawn
                "tot_delivered": False
            }
            valid_keys = {c.name for c in Interaction.__table__.columns}
            db_kwargs_tot_result = {k: v for k, v in tot_result_interaction_data.items() if
                                    k in valid_keys and k != 'id'}

            new_tot_interaction = await asyncio.to_thread(add_interaction, db, **db_kwargs_tot_result)
            if new_tot_interaction and new_tot_interaction.id:
                await asyncio.to_thread(db.commit)
                logger.success(
                    f"✅ {log_prefix} Saved ToT V2 result as Interaction ID {new_tot_interaction.id} for TrigID {triggering_interaction_id_for_log}.")
            else:
                logger.error(
                    f"❌ {log_prefix} Failed to save ToT V2 result for TrigID {triggering_interaction_id_for_log}.")
                await asyncio.to_thread(db.rollback)
        except Exception as db_save_err:
            logger.error(
                f"❌ {log_prefix} Error saving ToT V2 result to DB for TrigID {triggering_interaction_id_for_log}: {db_save_err}")
            await asyncio.to_thread(db.rollback)

        return final_synthesis_for_caller

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
                            db_session.close()
                            logger.debug(
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
                    logger.remove(current_sink_id)
                    logger.debug(
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
                    processing_complete = True
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
                    final_result_data["error"] = q_err
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
        priority: int,
        worker_cwd: str,
        timeout: int = 120
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    # Ensure ai_provider and its _priority_quota_lock are accessible.
    # This might be self.ai_provider if this function is part of a class that has it,
    # or a global ai_provider instance. For this example, assuming global ai_provider.
    # If ai_provider is an instance variable (e.g., self.ai_provider), adjust accordingly.
    global ai_provider  # Assuming ai_provider is a global instance initialized elsewhere

    shared_priority_lock: Optional[PriorityQuotaLock] = getattr(ai_provider, '_priority_quota_lock', None)

    request_id = request_data.get("request_id", "audio-worker-unknown")
    log_prefix = f"AudioExec|ELP{priority}|{request_id}"
    logger.debug(f"{log_prefix}: Attempting to execute audio worker.")

    if not shared_priority_lock:
        logger.error(f"{log_prefix}: Shared PriorityQuotaLock not available/initialized! Cannot run audio worker.")
        return None, "Shared resource lock not available."

    lock_acquired = False
    worker_process = None  # Initialize to None
    start_lock_wait = time.monotonic()
    logger.debug(f"{log_prefix}: Acquiring shared resource lock (Priority: ELP{priority})...")

    # Assuming acquire method exists and works as previously discussed
    lock_acquired = shared_priority_lock.acquire(priority=priority, timeout=None)
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
                errors='replace',  # Handle potential encoding errors in output
                cwd=worker_cwd
            )

            # If ELP0, register process with lock for potential interruption
            if priority == ELP0 and worker_process:
                if hasattr(shared_priority_lock, 'set_holder_process'):
                    shared_priority_lock.set_holder_process(worker_process)
                else:
                    logger.warning(
                        f"{log_prefix}: Lock does not have set_holder_process method. ELP0 process cannot be registered for interruption by this lock instance.")

            input_json = json.dumps(request_data)
            logger.debug(f"{log_prefix}: Sending input JSON (len={len(input_json)}) to audio worker stdin...")
            stdout_data, stderr_data = "", ""  # Initialize

            try:
                stdout_data, stderr_data = worker_process.communicate(input=input_json, timeout=timeout)
                logger.debug(f"{log_prefix}: Audio worker communicate() finished.")
            except subprocess.TimeoutExpired:
                logger.error(f"{log_prefix}: Audio worker process timed out after {timeout}s.")
                if worker_process and worker_process.poll() is None: worker_process.kill()  # Ensure kill on timeout
                # Try to get final outputs after kill
                try:
                    stdout_data, stderr_data = worker_process.communicate()
                except:
                    pass  # Best effort
                logger.error(f"{log_prefix}: Worker timed out. Stderr: {stderr_data.strip() if stderr_data else 'N/A'}")
                return None, "Audio worker process timed out."
            except BrokenPipeError:
                logger.warning(
                    f"{log_prefix}: Broken pipe with audio worker. Likely interrupted by higher priority task.")
                if worker_process and worker_process.poll() is None:  # Check if process exists and is running
                    try:
                        worker_process.wait(timeout=0.5)  # Brief wait
                    except subprocess.TimeoutExpired:
                        worker_process.kill()  # Force kill if wait times out
                # Attempt to get any remaining output after ensuring process is dealt with
                try:
                    stdout_data_bp, stderr_data_bp = "", ""
                    if worker_process:  # Only if worker_process was successfully created
                        stdout_data_bp, stderr_data_bp = worker_process.communicate()
                    stdout_data += stdout_data_bp  # Append if any
                    stderr_data += stderr_data_bp
                except Exception as e_bp_comm:
                    logger.warning(f"{log_prefix}: Error getting final output after BrokenPipe: {e_bp_comm}")
                return None, "Audio worker task interrupted by higher priority request."  # Use the consistent marker
            except Exception as comm_err:  # Other communication errors
                logger.error(f"{log_prefix}: Error communicating with audio worker: {comm_err}")
                if worker_process and worker_process.poll() is None:
                    try:
                        worker_process.kill(); worker_process.communicate()  # Best effort cleanup
                    except:
                        pass
                return None, f"Communication error with audio worker: {comm_err}"

            exit_code = worker_process.returncode if worker_process else -1  # Handle if worker_process is None
            duration = time.monotonic() - start_time
            logger.info(f"{log_prefix}: Audio worker finished. Exit Code: {exit_code}, Duration: {duration:.2f}s")

            if stderr_data:  # Log stderr regardless of exit code for diagnostics
                log_level_stderr = "ERROR" if exit_code != 0 else "DEBUG"
                stderr_snippet = (stderr_data[:2000] + '...[TRUNCATED]') if len(stderr_data) > 2000 else stderr_data
                logger.log(log_level_stderr,
                           f"{log_prefix}: Audio Worker STDERR:\n-------\n{stderr_snippet.strip()}\n-------")

            if exit_code == 0:
                if not stdout_data or not stdout_data.strip():  # Check if stdout is empty or just whitespace
                    logger.error(
                        f"{log_prefix}: Audio worker exited cleanly but no stdout or stdout is empty/whitespace.")
                    return None, "Audio worker produced no parsable output."

                json_string_to_parse = None  # Initialize
                try:
                    # Find the first '{' which should mark the beginning of our JSON object
                    json_start_index = stdout_data.find('{')
                    if json_start_index == -1:
                        logger.error(f"{log_prefix}: No JSON object start ('{{') found in audio worker stdout.")
                        logger.error(f"{log_prefix}: Raw stdout from worker (first 1000 chars):\n{stdout_data[:1000]}")
                        return None, "Audio worker did not produce valid JSON output (no '{' found)."

                    json_string_to_parse = stdout_data[json_start_index:]
                    parsed_json = json.loads(json_string_to_parse)
                    logger.debug(f"{log_prefix}: Parsed audio worker JSON response successfully.")

                    if isinstance(parsed_json,
                                  dict) and "error" in parsed_json:  # Check if worker itself reported an error in JSON
                        logger.error(f"{log_prefix}: Audio worker reported internal error: {parsed_json['error']}")
                        return None, f"Audio worker error: {parsed_json['error']}"
                    return parsed_json, None  # Success

                except json.JSONDecodeError as json_err:
                    logger.error(f"{log_prefix}: Failed to decode audio worker stdout JSON: {json_err}")
                    problematic_string_snippet = json_string_to_parse[
                                                 :500] if json_string_to_parse is not None else stdout_data[:500]
                    logger.error(
                        f"{log_prefix}: String snippet attempted for parsing:\n{problematic_string_snippet}...")
                    logger.error(
                        f"{log_prefix}: Original raw stdout from worker (first 1000 chars):\n{stdout_data[:1000]}")
                    return None, f"Failed to decode audio worker response: {json_err}"
            else:  # Worker exited with non-zero code
                err_msg = f"Audio worker process failed (exit code {exit_code})."
                # Stderr already logged above if present
                logger.error(f"{log_prefix}: {err_msg}")
                return None, err_msg

        except Exception as e:  # Catch-all for unexpected errors in this function's try block
            logger.error(f"{log_prefix}: Unexpected error managing audio worker: {e}")
            logger.exception(f"{log_prefix} Audio Worker Management Traceback:")
            if worker_process and worker_process.poll() is None:  # If Popen succeeded but later error
                try:
                    worker_process.kill(); worker_process.communicate()  # Best effort cleanup
                except:
                    pass
            return None, f"Error managing audio worker: {e}"
        finally:
            if lock_acquired:  # Only release if it was acquired
                logger.info(f"{log_prefix}: Releasing shared resource lock.")
                shared_priority_lock.release()
    else:  # Lock acquisition failed
        logger.error(f"{log_prefix}: FAILED to acquire shared resource lock for audio worker.")
        return None, "Failed to acquire execution lock for audio worker."


async def _run_background_high_quality_asr(
        original_audio_path: str,
        elp1_transcription: str,  # The (potentially corrected/diarized) text returned to user
        session_id_for_log: str,
        request_id: str,  # For cohesive logging
        language_for_asr: str
):
    """
    Runs high-quality ASR in the background (ELP0) and logs comparison.
    Needs its own DB session as it runs in a separate thread/task.
    """
    bg_asr_log_prefix = f"ASR_BG_ELP0|{request_id}"
    logger.info(f"{bg_asr_log_prefix}: Starting background high-quality ASR for audio: {original_audio_path}")
    db_bg_task: Optional[Session] = None  # Initialize db_bg_task to None

    # Create a new DB session for this background task
    if SessionLocal is None:  # Ensure SessionLocal is imported from database.py
        logger.error(f"{bg_asr_log_prefix}: SessionLocal is None. Cannot create DB session for background ASR.")
        return

    db_bg_task = SessionLocal()  # type: ignore
    if not db_bg_task:
        logger.error(f"{bg_asr_log_prefix}: Failed to create DB session for background ASR.")
        return

    try:
        asr_worker_script_bg = os.path.join(SCRIPT_DIR, "audio_worker.py")
        # Command for the high-quality (default) Whisper model at ELP0
        asr_worker_cmd_bg = [
            APP_PYTHON_EXECUTABLE, asr_worker_script_bg,
            "--task-type", "asr",
            "--model-dir", WHISPER_MODEL_DIR,  # From config
            "--temp-dir", os.path.join(SCRIPT_DIR, "temp_audio_worker_files")  # Consistent temp dir
        ]
        asr_request_data_bg = {
            "input_audio_path": original_audio_path,  # Original audio path
            "whisper_model_name": WHISPER_DEFAULT_MODEL_FILENAME,  # High-quality model from config
            "language": language_for_asr,  # Language used for initial ASR
            "request_id": f"{request_id}-bg-asr"
        }

        logger.info(f"{bg_asr_log_prefix}: Executing audio worker for high-quality ASR (ELP0)...")
        # _execute_audio_worker_with_priority is synchronous, run in thread from async context
        hq_asr_response, hq_asr_err = await asyncio.to_thread(
            _execute_audio_worker_with_priority,  # This helper is in app.py
            worker_command=asr_worker_cmd_bg,
            request_data=asr_request_data_bg,
            priority=ELP0,  # Run this transcription at ELP0
            worker_cwd=SCRIPT_DIR,
            timeout=ASR_WORKER_TIMEOUT + 120  # Potentially longer timeout for higher quality model
        )

        elp0_transcription: Optional[str] = None
        if hq_asr_err or not (
                hq_asr_response and isinstance(hq_asr_response.get("result"), dict) and "text" in hq_asr_response[
            "result"]):
            logger.error(
                f"{bg_asr_log_prefix}: Background high-quality ASR step failed: {hq_asr_err or 'Invalid ASR worker response'}")
            elp0_transcription = f"[High-Quality ASR Failed: {hq_asr_err or 'Invalid ASR worker response'}]"
        else:
            elp0_transcription = hq_asr_response["result"]["text"]
            logger.info(
                f"{bg_asr_log_prefix}: Background high-quality ASR successful (Snippet: '{elp0_transcription[:100]}...').")

        # Log the comparison
        comparison_text = (
            f"--- ASR Comparison for Request {request_id} ---\n"
            f"Low-Latency Whisper ({WHISPER_LOW_LATENCY_MODEL_FILENAME}) - ELP1 Output (returned to client):\n"
            f"\"\"\"\n{elp1_transcription}\n\"\"\"\n\n"
            f"High-Quality Whisper ({WHISPER_DEFAULT_MODEL_FILENAME}) - ELP0 Background Output:\n"
            f"\"\"\"\n{elp0_transcription}\n\"\"\"\n"
        )

        add_interaction(
            db_bg_task, session_id=session_id_for_log, mode="asr_service",
            input_type="asr_comparison_log",
            user_input=f"[ASR Comparison Log for original request {request_id}]",
            llm_response=comparison_text[:4000],  # Ensure it fits DB field
            classification="asr_quality_comparison"
        )
        db_bg_task.commit()
        logger.info(f"{bg_asr_log_prefix}: Logged ASR comparison to database.")

    except Exception as e_bg_asr:
        logger.error(f"{bg_asr_log_prefix}: Error in background high-quality ASR task: {e_bg_asr}")
        logger.exception(f"{bg_asr_log_prefix} Background ASR Traceback:")
        if db_bg_task:
            try:
                add_interaction(db_bg_task, session_id=session_id_for_log, mode="asr_service", input_type="log_error",
                                user_input=f"[Background ASR Error for Req {request_id}]",
                                llm_response=str(e_bg_asr)[:2000])
                db_bg_task.commit()
            except Exception as e_db_log:
                logger.error(f"{bg_asr_log_prefix}: Failed to log background ASR error to DB: {e_db_log}")
                if db_bg_task: db_bg_task.rollback()
    finally:
        if db_bg_task:
            db_bg_task.close()
            logger.debug(f"{bg_asr_log_prefix}: Background ASR DB session closed.")
        logger.info(f"{bg_asr_log_prefix}: Background high-quality ASR task finished.")


async def _run_background_asr_and_translation_analysis(
        original_audio_path: str,  # This is the path to the initially uploaded temp file
        elp1_transcription_final_for_client: str,
        elp1_translation_final_for_client: Optional[str],  # The quick translation returned to client
        session_id_for_log: str,
        request_id: str,
        language_asr: str,
        target_language_translation: Optional[str]  # Target language code for translation (e.g., "es")
):
    bg_log_prefix = f"ASR_Translate_BG_ELP0|{request_id}"
    logger.info(f"{bg_log_prefix}: Starting background tasks for audio: {original_audio_path}")

    db_bg_task: Optional[Session] = None
    high_quality_transcribed_text: Optional[str] = None

    try:
        if SessionLocal is None:
            logger.error(f"{bg_log_prefix}: SessionLocal is None. Cannot create DB session.")
            return
        db_bg_task = SessionLocal()  # type: ignore
        if not db_bg_task:
            logger.error(f"{bg_log_prefix}: Failed to create DB session.")
            return

        # === Step 1: High-Quality ASR (ELP0) ===
        asr_worker_script_bg = os.path.join(SCRIPT_DIR, "audio_worker.py")
        asr_worker_cmd_bg = [
            APP_PYTHON_EXECUTABLE, asr_worker_script_bg,
            "--task-type", "asr", "--model-dir", WHISPER_MODEL_DIR,
            "--temp-dir", os.path.join(SCRIPT_DIR, "temp_audio_worker_files")  # For ffmpeg in worker
        ]
        asr_request_data_bg = {
            "input_audio_path": original_audio_path,  # Use the original uploaded file
            "whisper_model_name": WHISPER_DEFAULT_MODEL_FILENAME,  # High-quality model
            "language": language_asr,
            "request_id": f"{request_id}-bg-hq-asr"
        }
        logger.info(f"{bg_log_prefix}: Executing audio worker for high-quality ASR (ELP0) on {original_audio_path}...")
        hq_asr_response, hq_asr_err = await asyncio.to_thread(
            _execute_audio_worker_with_priority,
            asr_worker_cmd_bg, asr_request_data_bg, ELP0, SCRIPT_DIR, ASR_WORKER_TIMEOUT + 180
        )

        if hq_asr_err or not (
                hq_asr_response and isinstance(hq_asr_response.get("result"), dict) and "text" in hq_asr_response[
            "result"]):
            logger.error(
                f"{bg_log_prefix}: Background high-quality ASR failed: {hq_asr_err or 'Invalid ASR worker response'}")
            high_quality_transcribed_text = f"[High-Quality ASR (ELP0) Failed: {hq_asr_err or 'Invalid ASR response'}]"
        else:
            high_quality_transcribed_text = hq_asr_response["result"]["text"]
            logger.info(
                f"{bg_log_prefix}: Background high-quality ASR successful. Snippet: '{high_quality_transcribed_text[:100]}...'")

        # Log ASR comparison
        asr_comparison_log_text = (
            f"--- ASR Comparison for Request {request_id} ---\n"
            f"Quick Whisper ({WHISPER_LOW_LATENCY_MODEL_FILENAME}) ELP1 output (processed, sent to client):\n"
            f"\"\"\"\n{elp1_transcription_final_for_client}\n\"\"\"\n\n"
            f"Default Whisper ({WHISPER_DEFAULT_MODEL_FILENAME}) ELP0 background output:\n"
            f"\"\"\"\n{high_quality_transcribed_text}\n\"\"\"\n"
        )
        add_interaction(db_bg_task, session_id=session_id_for_log, mode="asr_service",
                        input_type="asr_comparison_log",
                        user_input=f"[ASR Comparison Log for {request_id}]",
                        llm_response=asr_comparison_log_text[:4000],
                        classification="internal_asr_comparison")
        db_bg_task.commit()
        logger.info(f"{bg_log_prefix}: Logged ASR comparison to database.")

        # === Step 2: If it was a translation request, do background "Grokking Generate" for translation ===
        if target_language_translation and high_quality_transcribed_text and \
                not high_quality_transcribed_text.startswith("[High-Quality ASR (ELP0) Failed"):

            logger.info(
                f"{bg_log_prefix}: Performing background 'Grokking Generate' translation of high-quality transcript to '{target_language_translation}'...")

            source_lang_full = langcodes.Language.make(
                language=language_asr).display_name() if language_asr != "auto" else "auto-detected"
            target_lang_full = langcodes.Language.make(language=target_language_translation).display_name()

            deep_translation_prompt = PROMPT_DEEP_TRANSLATION_ANALYSIS.format(
                source_language_code=language_asr,
                source_language_full_name=source_lang_full,
                target_language_code=target_language_translation,
                target_language_full_name=target_lang_full,
                high_quality_transcribed_text=high_quality_transcribed_text
            )

            # Use background_generate for a potentially more complex/thorough translation
            # This will create its own new interaction chain.
            deep_translation_session_id = f"deep_translate_{request_id}"
            logger.info(
                f"{bg_log_prefix}: Spawning background_generate for deep translation. Session: {deep_translation_session_id}")
            # background_generate itself handles ELP0 and its own logging.
            # It's async, so we can await it if this function is called via asyncio.create_task
            # or just let it run if this is already a background thread.
            # Since _run_background_asr_and_translation_analysis is async, we can await.
            await ai_chat.background_generate(  # type: ignore
                db=db_bg_task,  # Pass the DB session for this background task
                user_input=deep_translation_prompt,
                session_id=deep_translation_session_id,
                classification="deep_translation_task",
                image_b64=None,
                update_interaction_id=None  # It's a new root task for this session
            )
            # The result of this deep translation is stored as a new interaction by background_generate.
            # We log a comparison note pointing to this.

            translation_comparison_log_text = (
                f"--- Translation Comparison for Request {request_id} ---\n"
                f"Quick Translation (ELP1, from Low-Latency ASR, sent to client):\n"
                f"\"\"\"\n{elp1_translation_final_for_client or 'N/A (Not a translation request or ELP1 translation failed)'}\n\"\"\"\n\n"
                f"Deep Translation (ELP0 Background task using High-Quality ASR) was initiated.\n"
                f"Input to Deep Translation (High-Quality ASR Text, Lang: {language_asr}):\n"
                f"\"\"\"\n{high_quality_transcribed_text[:500]}...\n\"\"\"\n"
                f"Result will be logged under session_id starting with '{deep_translation_session_id}'."
            )
            add_interaction(db_bg_task, session_id=session_id_for_log, mode="translation_service",
                            input_type="translation_comparison_log",
                            user_input=f"[Translation Comparison Log for {request_id}]",
                            llm_response=translation_comparison_log_text[:4000],
                            classification="internal_translation_comparison")
            db_bg_task.commit()
            logger.info(f"{bg_log_prefix}: Logged translation comparison info and spawned deep translation.")
        elif target_language_translation:
            logger.warning(
                f"{bg_log_prefix}: Skipped deep translation because high-quality ASR failed or produced no text.")

    except Exception as e_bg_task:
        logger.error(f"{bg_log_prefix}: Error in background task: {e_bg_task}")
        logger.exception(f"{bg_log_prefix} Background Task Traceback:")
        if db_bg_task:
            try:
                add_interaction(db_bg_task, session_id=session_id_for_log, mode="asr_service", input_type="log_error",
                                user_input=f"[Background ASR/Translate Error for Req {request_id}]",
                                llm_response=str(e_bg_task)[:2000])
                db_bg_task.commit()
            except Exception as e_db_log_bg_err:
                logger.error(f"{bg_log_prefix}: Failed to log background task error to DB: {e_db_log_bg_err}")
                if db_bg_task: db_bg_task.rollback()
    finally:
        # This background task is now responsible for the original_audio_path
        if original_audio_path and os.path.exists(original_audio_path):
            try:
                await asyncio.to_thread(os.remove, original_audio_path)  # Use await for async context
                logger.info(f"{bg_log_prefix}: Deleted original temporary input audio file: {original_audio_path}")
            except Exception as e_del_orig:
                logger.warning(
                    f"{bg_log_prefix}: Failed to delete original temporary input audio file '{original_audio_path}': {e_del_orig}")
        if db_bg_task:
            db_bg_task.close()
            logger.debug(f"{bg_log_prefix}: Background task DB session closed.")
        logger.info(f"{bg_log_prefix}: Background task finished.")


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


def _create_personal_assistant_stub_response(endpoint_name: str, method: str, resource_id: Optional[str] = None, custom_status: str = "not_applicable_personal_assistant"):
    """
    Creates a standardized JSON response for Assistants API stubs.
    """
    message = (
        f"The endpoint '{method} {endpoint_name}' for managing separate assistants/threads "
        "is not implemented in the standard OpenAI way. This system operates as an integrated personal assistant. "
        "Functionalities like memory, context management, and tool use are embedded within its direct interaction flow "
        "and internal self-reflection or background generation processes."
    )
    if resource_id:
        message += f" Operation on resource ID '{resource_id}' is not applicable."

    response_body = {
        "object": "api.stub_response",
        "endpoint_called": endpoint_name,
        "method": method,
        "resource_id_queried": resource_id,
        "status": custom_status,
        "message": message,
        "note": "This is a placeholder response indicating a conceptual difference in system design."
    }
    # OpenAI often returns 200 OK even for informative non-errors, or specific errors for non-found resources.
    # For these stubs explaining a different design, 200 OK with the message can be user-friendly.
    # Or, you could choose 501 Not Implemented if you prefer to signal it more strongly.
    return jsonify(response_body), 200


def _create_assistants_api_stub_response(
        endpoint_name: str,
        method: str,
        resource_ids: Optional[Dict[str, str]] = None,
        operation_specific_message: str = "",
        additional_details: Optional[Dict[str, Any]] = None,
        object_type_for_response: str = "api.stub_information"  # Custom object type for these stubs
):
    """
    Creates a standardized JSON response for Assistants API stubs,
    explaining the system's personal assistant design.
    """
    message = (
        f"This system ('Zephy/Adelaide') is designed as an integrated personal assistant. "
        f"The standard OpenAI Assistants API endpoint '{method} {endpoint_name}' is handled differently. "
        "Core functionalities like memory, context management, and tool use (agentic mode) are embedded "
        "within its direct interaction flow and asynchronous background processes, rather than through explicit, "
        "multi-step Assistant/Thread/Run objects managed via this API."
    )
    if operation_specific_message:
        message += f" Regarding '{method} {endpoint_name}': {operation_specific_message}"

    response_body = {
        "object": object_type_for_response,
        "message": message,
        "details": {
            "endpoint_called": f"{method} {endpoint_name}",
            "system_paradigm": "Integrated Personal Assistant with Asynchronous Agentic Mode",
        }
    }
    if resource_ids:
        response_body["details"]["resource_ids_queried"] = resource_ids  # type: ignore
    if additional_details:
        response_body["details"].update(additional_details)

    # Using 200 OK with an informative message, as these are informational stubs.
    return jsonify(response_body), 200

# === Flask Routes (Async) ===
"""
                                                                                                                                                                                                                                                                                 ,--.                                                        
,--.,------.  ,-----.     ,---.                                            ,-----.                                           ,--.                ,--.  ,--.                    ,---.                 ,--.  ,--.                     ,-.,--.  ,--.,--------.,--------.,------.   /  /,--.   ,--.                                        ,-.   
|  ||  .--. ''  .--./    '   .-'  ,---. ,--.--.,--.  ,--.,---. ,--.--.    '  .--./ ,---. ,--,--,--.,--,--,--.,--.,--.,--,--, `--' ,---. ,--,--.,-'  '-.`--' ,---. ,--,--,     '   .-'  ,---.  ,---.,-'  '-.`--' ,---. ,--,--,      / .'|  '--'  |'--.  .--''--.  .--'|  .--. ' /  / |   `.'   | ,---. ,--,--,--. ,---. ,--.--.,--. ,--.'. \  
|  ||  '--' ||  |        `.  `-. | .-. :|  .--' \  `'  /| .-. :|  .--'    |  |    | .-. ||        ||        ||  ||  ||      \,--.| .--'' ,-.  |'-.  .-',--.| .-. ||      \    `.  `-. | .-. :| .--''-.  .-',--.| .-. ||      \    |  | |  .--.  |   |  |      |  |   |  '--' |/  /  |  |'.'|  || .-. :|        || .-. ||  .--' \  '  /  |  | 
|  ||  | --' '  '--'\    .-'    |\   --.|  |     \    / \   --.|  |       '  '--'\' '-' '|  |  |  ||  |  |  |'  ''  '|  ||  ||  |\ `--.\ '-'  |  |  |  |  |' '-' '|  ||  |    .-'    |\   --.\ `--.  |  |  |  |' '-' '|  ||  |    |  | |  |  |  |   |  |      |  |   |  | --'/  /   |  |   |  |\   --.|  |  |  |' '-' '|  |     \   '   |  | 
`--'`--'      `-----'    `-----'  `----'`--'      `--'   `----'`--'        `-----' `---' `--`--`--'`--`--`--' `----' `--''--'`--' `---' `--`--'  `--'  `--' `---' `--''--'    `-----'  `----' `---'  `--'  `--' `---' `--''--'     \ '.`--'  `--'   `--'      `--'   `--'   /  /    `--'   `--' `----'`--`--`--' `---' `--'   .-'  /   .' /  
                                                                                                                                                                                                                                    `-'                                    `--'                                               `---'    `-'   
"""
# ====== Server Root =======
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

#==============================[openAI API (Most standard) Behaviour]==============================
@app.route("/v1/embeddings", methods=["POST"])
@app.route("/api/embed", methods=["POST"])
@app.route("/api/embeddings", methods=["POST"])
async def handle_openai_embeddings():
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
                last_user_msg_obj = msg_item
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
                image_b64_bg: Optional[str], vlm_desc_user_img_bg: Optional[str]):  # Add other necessary args

            bg_log_prefix_thread = f"[BG Task {request_id} Thr:{threading.get_ident()}]"
            acquired_semaphore = False
            loop = None
            bg_db_session: Optional[Session] = None

            # --- NEW: Politeness Check Loop ---
            MAX_POLITENESS_WAIT_SECONDS = 30  # Max total time to wait for ELP1 activity to clear
            POLITENESS_CHECK_INTERVAL_SECONDS = 1.5  # How often to check
            politeness_wait_start_time = time.monotonic()
            initial_check_done = False

            while True:
                if ai_provider and ai_provider.is_resource_busy_with_high_priority():
                    logger.info(
                        f"{bg_log_prefix_thread} AIProvider resources busy with ELP1. Pausing background task start...")
                    initial_check_done = True
                    time.sleep(POLITENESS_CHECK_INTERVAL_SECONDS)
                    if time.monotonic() - politeness_wait_start_time > MAX_POLITENESS_WAIT_SECONDS:
                        logger.warning(
                            f"{bg_log_prefix_thread} Max politeness wait time ({MAX_POLITENESS_WAIT_SECONDS}s) exceeded. Proceeding despite ELP1 activity.")
                        break
                else:
                    if initial_check_done:  # Log only if we actually waited
                        logger.info(
                            f"{bg_log_prefix_thread} AIProvider resources appear free. Proceeding with background task.")
                    else:  # Log if we proceed immediately on first check
                        logger.debug(f"{bg_log_prefix_thread} AIProvider resources free on initial check. Proceeding.")
                    break  # Exit politeness loop
            # --- END NEW: Politeness Check Loop ---

            try:
                logger.debug(f"{bg_log_prefix_thread} Attempting to acquire background_generate_task_semaphore...")
                background_generate_task_semaphore.acquire()  # This is the main concurrency limiter
                acquired_semaphore = True
                logger.info(f"{bg_log_prefix_thread} Acquired background_generate_task_semaphore. Starting processing.")

                # ... (rest of your run_background_task_with_new_loop logic:
                #      new asyncio loop, db session, ai_chat.background_generate) ...
                # Setup new asyncio loop for this thread if needed
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        logger.warning(
                            f"{bg_log_prefix_thread} Event loop already running in this new thread. This is unexpected.")
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                logger.debug(f"{bg_log_prefix_thread} Creating DB session for background task...")
                if SessionLocal is None:
                    logger.error(f"{bg_log_prefix_thread} SessionLocal is None. Cannot create DB session.")
                    # Release semaphore if acquired, then return
                    if acquired_semaphore: background_generate_task_semaphore.release()
                    return
                bg_db_session = SessionLocal()  # type: ignore
                if not bg_db_session:
                    logger.error(f"{bg_log_prefix_thread} Failed to create DB session for background task.")
                    if acquired_semaphore: background_generate_task_semaphore.release()
                    return

                logger.info(f"{bg_log_prefix_thread} Calling ai_chat.background_generate...")
                loop.run_until_complete(
                    ai_chat.background_generate(
                        db=bg_db_session,
                        user_input=user_input_bg,
                        session_id=session_id_bg,
                        classification=classification_bg,
                        image_b64=image_b64_bg
                    )
                )
                logger.info(f"{bg_log_prefix_thread} ai_chat.background_generate task completed.")


            except Exception as task_err:
                logger.error(f"{bg_log_prefix_thread} Error during background task execution: {task_err}")
                logger.exception(f"{bg_log_prefix_thread} Background Task Execution Traceback:")
                if bg_db_session:
                    try:
                        from database import add_interaction
                        add_interaction(bg_db_session, session_id=session_id_bg, mode="chat", input_type="log_error",
                                        user_input=f"Background Task Error ({request_id})",
                                        llm_response=f"Error: {str(task_err)[:1500]}"
                                        )
                    except Exception as db_log_err_bg:
                        logger.error(f"{bg_log_prefix_thread} Failed to log BG error to DB: {db_log_err_bg}")
            finally:
                if bg_db_session:
                    try:
                        bg_db_session.close(); logger.debug(f"{bg_log_prefix_thread} BG DB session closed.")
                    except:
                        pass
                if loop and hasattr(loop, 'is_closed') and not loop.is_closed():  # Check if loop has is_closed
                    try:
                        loop.close(); logger.debug(f"{bg_log_prefix_thread} BG asyncio loop closed.")
                    except:
                        pass
                if acquired_semaphore:
                    background_generate_task_semaphore.release()
                    logger.info(f"{bg_log_prefix_thread} Released background_generate_task_semaphore.")
                logger.info(f"{bg_log_prefix_thread} Background thread function finished.")

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


@app.route("/v1/moderations", methods=["POST"])
async def handle_openai_moderations():
    """
    Handles requests mimicking OpenAI's Moderations endpoint.
    Uses AIChat.direct_generate() with a specific prompt for assessment.
    """
    start_req_time = time.monotonic()
    request_id = f"req-mod-{uuid.uuid4()}"
    logger.info(f"🚀 Flask OpenAI-Style Moderation Request ID: {request_id}")

    db: Session = g.db
    final_status_code: int = 500
    resp: Optional[Response] = None
    session_id_for_log: str = f"moderation_req_{request_id}"  # Each moderation is a unique "session" for logging
    raw_request_data: Optional[Dict[str, Any]] = None
    input_text_to_moderate: Optional[str] = None

    try:
        try:
            raw_request_data = await request.get_json()  # Assuming Quart, or request.get_json() for Flask sync
            if not raw_request_data:
                raise ValueError("Empty JSON payload received.")
        except Exception as json_err:
            logger.warning(f"{request_id}: Failed to get/parse JSON body: {json_err}")
            resp_data, status_code = _create_openai_error_response(
                f"Request body is missing or invalid JSON: {json_err}",
                err_type="invalid_request_error", status_code=400)
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
            final_status_code = status_code
            return resp

        input_data = raw_request_data.get("input")
        model_requested = raw_request_data.get("model")  # Logged but we'll use our configured model name

        if model_requested:
            logger.info(
                f"{request_id}: Client requested model '{model_requested}', will use configured '{MODERATION_MODEL_CLIENT_FACING}'.")

        if not input_data:
            raise ValueError("'input' field is required.")

        # OpenAI allows string or array of strings. For simplicity, let's handle single string first.
        # If you need to handle an array, you'd loop and generate multiple results.
        if isinstance(input_data, list):
            if not input_data: raise ValueError("'input' array is empty.")
            input_text_to_moderate = str(input_data[0])  # Process first item if array for now
            if len(input_data) > 1:
                logger.warning(
                    f"{request_id}: Received array for 'input', processing only the first item for moderation.")
        elif isinstance(input_data, str):
            input_text_to_moderate = input_data
        else:
            raise ValueError("'input' must be a string or an array of strings.")

        if not input_text_to_moderate.strip():
            # Handle empty string input gracefully - OpenAI typically flags it as not harmful.
            logger.info(f"{request_id}: Input text is empty or whitespace. Returning as not flagged.")
            results_list = [{
                "flagged": False,
                "categories": {cat: False for cat in
                               ["hate", "hate/threatening", "self-harm", "sexual", "sexual/minors", "violence",
                                "violence/graphic"]},
                "category_scores": {cat: 0.0 for cat in
                                    ["hate", "hate/threatening", "self-harm", "sexual", "sexual/minors", "violence",
                                     "violence/graphic"]}
            }]
            response_body = {
                "id": f"modr-{uuid.uuid4()}",
                "model": MODERATION_MODEL_CLIENT_FACING,
                "results": results_list
            }
            resp = Response(json.dumps(response_body), status=200, mimetype='application/json')
            final_status_code = 200
            return resp

        # --- Call AIChat.direct_generate() for moderation assessment ---
        # direct_generate runs at ELP1
        if not ai_chat:  # Should be initialized globally
            raise RuntimeError("AIChat instance not available.")

        moderation_prompt_filled = PROMPT_MODERATION_CHECK.format(input_text_to_moderate=input_text_to_moderate)

        # Use a unique session_id for this moderation call to keep its logs separate if needed
        # or use the session_id_for_log passed in the request if that makes sense for your context.
        # For now, creating a distinct one for the direct_generate call.
        moderation_llm_session_id = f"mod_llm_req_{request_id}"
        if ai_chat: ai_chat.current_session_id = moderation_llm_session_id

        logger.info(
            f"{request_id}: Calling direct_generate for moderation. Input snippet: '{input_text_to_moderate[:70]}...'")
        # direct_generate is async, and this Flask route is async
        llm_assessment_text = await ai_chat.direct_generate(
            db,  # Pass the current request's DB session
            moderation_prompt_filled,
            moderation_llm_session_id,  # Session ID for this specific LLM call
            vlm_description=None,  # No image for moderation
            image_b64=None
        )

        logger.info(f"{request_id}: LLM moderation assessment raw response: '{llm_assessment_text}'")

        # --- Parse LLM's assessment and build OpenAI response ---
        flagged = False
        categories = {
            "hate": False, "hate/threatening": False, "self-harm": False,
            "sexual": False, "sexual/minors": False, "violence": False, "violence/graphic": False
        }
        # Category scores are harder to get reliably from a general LLM without specific training
        # We'll use dummy scores for now.
        category_scores = {cat: 0.01 for cat in categories}  # Default low scores

        if llm_assessment_text.startswith("FLAGGED:"):
            flagged = True
            try:
                violated_categories_str = llm_assessment_text.replace("FLAGGED:", "").strip()
                violated_list = [cat.strip() for cat in violated_categories_str.split(',')]
                for cat_key in violated_list:
                    if cat_key in categories:
                        categories[cat_key] = True
                        category_scores[cat_key] = 0.9  # Example high score for flagged
                    else:
                        logger.warning(f"{request_id}: LLM reported an unknown category '{cat_key}'")
            except Exception as parse_cat_err:
                logger.error(
                    f"{request_id}: Could not parse categories from LLM response '{llm_assessment_text}': {parse_cat_err}")
                # Keep flagged as True, but categories might be inaccurate or all false
        elif "CLEAN" not in llm_assessment_text.upper():  # If not "CLEAN" and not "FLAGGED:", it's ambiguous
            logger.warning(
                f"{request_id}: LLM moderation response ambiguous: '{llm_assessment_text}'. Defaulting to not flagged.")
            # Flagged remains false

        results_list = [{
            "flagged": flagged,
            "categories": categories,
            "category_scores": category_scores
        }]

        response_body = {
            "id": f"modr-{uuid.uuid4()}",  # Generate a new ID for the moderation response
            "model": MODERATION_MODEL_CLIENT_FACING,  # Use your configured model name
            "results": results_list
        }
        resp = Response(json.dumps(response_body), status=200, mimetype='application/json')
        final_status_code = 200

        # Log the moderation interaction
        try:
            add_interaction(db, session_id=session_id_for_log, mode="moderation", input_type="text_moderated",
                            user_input=input_text_to_moderate[:2000],  # Log original input
                            llm_response=json.dumps(results_list[0]),  # Log the result
                            classification="moderation_checked",
                            execution_time_ms=(time.monotonic() - start_req_time) * 1000)
            db.commit()
        except Exception as db_log_err:
            logger.error(f"{request_id}: Failed to log moderation result to DB: {db_log_err}")
            if db: db.rollback()

    except ValueError as ve:
        logger.warning(f"{request_id}: Invalid Moderation request: {ve}")
        resp_data, status_code = _create_openai_error_response(str(ve), err_type="invalid_request_error",
                                                               status_code=400)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_status_code = status_code
    except RuntimeError as rte:  # Catch errors from downstream calls like model unavailable
        logger.error(f"{request_id}: Runtime error during moderation: {rte}")
        resp_data, status_code = _create_openai_error_response(str(rte), err_type="server_error", status_code=500)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_status_code = status_code
    except Exception as main_err:
        logger.exception(f"{request_id}: 🔥🔥 Unhandled exception in Moderation endpoint:")
        error_message = f"Internal server error in Moderation endpoint: {type(main_err).__name__}"
        resp_data, status_code = _create_openai_error_response(error_message, status_code=500)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_status_code = status_code
        try:
            if db: add_interaction(db, session_id=session_id_for_log, mode="moderation", input_type='error',
                                   user_input=f"Moderation Handler Error. Input: {str(input_text_to_moderate)[:200]}",
                                   llm_response=error_message[:2000]); db.commit()
        except Exception as db_err_log:
            logger.error(f"{request_id}: ❌ Failed log Moderation handler error: {db_err_log}")

    finally:
        duration_req = (time.monotonic() - start_req_time) * 1000
        logger.info(
            f"🏁 OpenAI-Style Moderation Request {request_id} handled in {duration_req:.2f} ms. Status: {final_status_code}")

    if resp is None:
        logger.error(f"{request_id}: Moderation Handler logic flaw - response object 'resp' was not assigned!")
        resp_data, _ = _create_openai_error_response("Internal error: Handler failed to produce a response.",
                                                     status_code=500)
        resp = Response(json.dumps(resp_data), status=500, mimetype='application/json')
    return resp

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
        {
            "id": MODERATION_MODEL_CLIENT_FACING, # e.g., "text-moderation-zephy"
            "object": "model",
            "created": int(time.time()),
            "owned_by": META_MODEL_OWNER, # As defined in your config.py
            "permission": [],
            "root": MODERATION_MODEL_CLIENT_FACING,
            "parent": None,
            # "description": "Moderation service based on internal LLM capabilities." # Optional
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
            "id": AUDIO_TRANSLATION_MODEL_CLIENT_FACING,  # From config.py
            "object": "model",
            "created": int(time.time()),
            "owned_by": META_MODEL_OWNER,  # From config.py
            "permission": [], "root": AUDIO_TRANSLATION_MODEL_CLIENT_FACING, "parent": None,
            # "description": "Audio-to-Audio Translation Service" # Optional
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

    ]

    response_body = {
        "object": "list",
        "data": model_list,
    }
    response_payload = json.dumps(response_body, indent=2)
    duration_req = (time.monotonic() - start_req) * 1000
    logger.info(f"🏁 /v1/models request handled in {duration_req:.2f} ms. Status: {status_code}")
    return Response(response_payload, status=status_code, mimetype='application/json')




#==============================[Ollama Behaviour]==============================
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


@app.route("/v1/audio/transcriptions", methods=["POST"])
async def handle_openai_asr_transcriptions():
    start_req_time = time.monotonic()
    request_id = f"req-asr-{uuid.uuid4()}"
    logger.info(f"🚀 OpenAI-Style ASR Request ID: {request_id} (Multi-Stage ELP1 + Background ELP0)")

    db: Session = g.db
    final_status_code: int = 500
    resp: Optional[Response] = None
    session_id_for_log: str = f"asr_req_default_{request_id}"
    uploaded_filename: Optional[str] = None
    temp_input_audio_path: Optional[str] = None  # Path to the original uploaded audio
    language_for_asr_steps: str = "auto"

    # Variables to store intermediate results for logging and background task
    raw_low_latency_transcription: Optional[str] = None
    corrected_transcription: Optional[str] = None
    diarized_text_final_for_client: Optional[str] = None

    try:
        if not ENABLE_ASR:
            logger.warning(f"{request_id}: ASR endpoint called but ASR is disabled in config.")
            resp_data, status_code = _create_openai_error_response(
                "ASR functionality is currently disabled on this server.",
                err_type="server_error", code="asr_disabled", status_code=503
            )
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
            final_status_code = status_code
            return resp  # type: ignore

        if not request.content_type or not request.content_type.startswith('multipart/form-data'):
            raise ValueError("Invalid content type. Must be multipart/form-data.")

        audio_file_storage = request.files.get('file')
        model_requested = request.form.get('model')
        language_param_for_log = request.form.get('language')
        response_format_req = request.form.get('response_format', 'json').lower()

        session_id_for_log = request.form.get("session_id", session_id_for_log)
        # Ensure ai_chat instance is available if needed for direct_generate
        if 'ai_chat' not in globals() or ai_chat is None:
            logger.error(f"{request_id}: ai_chat instance not available. Cannot proceed with LLM steps.")
            raise RuntimeError("AIChat instance not configured for ASR post-processing.")
        ai_chat.current_session_id = session_id_for_log  # type: ignore

        if audio_file_storage and audio_file_storage.filename:
            uploaded_filename = secure_filename(audio_file_storage.filename)

        if not audio_file_storage: raise ValueError("'file' field (audio data) is required.")
        if not model_requested or model_requested != ASR_MODEL_NAME_CLIENT_FACING:  # From config.py
            raise ValueError(f"Invalid 'model'. This endpoint supports '{ASR_MODEL_NAME_CLIENT_FACING}'.")

        language_for_asr_steps = language_param_for_log or WHISPER_DEFAULT_LANGUAGE  # From config.py
        if not language_for_asr_steps or language_for_asr_steps.strip().lower() == "auto":
            language_for_asr_steps = "auto"
        else:
            language_for_asr_steps = language_for_asr_steps.strip().lower()

        logger.debug(
            f"{request_id}: ASR Request Parsed - File: {uploaded_filename or 'Missing'}, "
            f"Lang: {language_for_asr_steps}, RespFormat: {response_format_req}"
        )

        # --- Step 0: Save Uploaded File ---
        # SCRIPT_DIR should be defined at the top of app.py: os.path.dirname(os.path.abspath(__file__))
        temp_audio_dir = os.path.join(SCRIPT_DIR, "temp_audio_worker_files")
        await asyncio.to_thread(os.makedirs, temp_audio_dir, exist_ok=True)

        _, file_extension = os.path.splitext(uploaded_filename or ".tmpaud")
        # Use mkstemp for a unique temporary file that we manage
        temp_fd, temp_input_audio_path = tempfile.mkstemp(prefix="asr_orig_", suffix=file_extension, dir=temp_audio_dir)
        os.close(temp_fd)  # We got the path, now we can save to it.

        await asyncio.to_thread(audio_file_storage.save, temp_input_audio_path)
        logger.info(f"{request_id}: Input audio saved temporarily to: {temp_input_audio_path}")

        # --- ELP1 PIPELINE for Transcription ---
        # --- Step 1.1: Low-Latency ASR ---
        logger.info(f"{request_id}: ELP1 Step 1.1: Low-Latency ASR (Model: {WHISPER_LOW_LATENCY_MODEL_FILENAME})...")
        asr_worker_script = os.path.join(SCRIPT_DIR, "audio_worker.py")
        # APP_PYTHON_EXECUTABLE and WHISPER_MODEL_DIR from config or defined in app.py
        ll_asr_cmd = [APP_PYTHON_EXECUTABLE, asr_worker_script, "--task-type", "asr",
                      "--model-dir", WHISPER_MODEL_DIR, "--temp-dir", temp_audio_dir]
        ll_asr_req_data = {
            "input_audio_path": temp_input_audio_path,
            "whisper_model_name": WHISPER_LOW_LATENCY_MODEL_FILENAME,  # From config.py
            "language": language_for_asr_steps,
            "request_id": f"{request_id}-elp1-llasr"  # low-latency asr
        }
        elp1_asr_call_start_time = time.monotonic()
        # --- END ADDITION ---

        elp1_asr_response, elp1_asr_err = await asyncio.to_thread(
            _execute_audio_worker_with_priority, ll_asr_cmd, ll_asr_req_data,
            ELP1, SCRIPT_DIR, ASR_WORKER_TIMEOUT
        )

        # --- ADD THIS CALCULATION ---
        elp1_asr_duration_ms = (time.monotonic() - elp1_asr_call_start_time) * 1000

        asr_call_start_time = time.monotonic()  # For timing this specific ASR call
        elp1_asr_response, elp1_asr_err = await asyncio.to_thread(
            _execute_audio_worker_with_priority,  # This helper is synchronous
            worker_command=ll_asr_cmd,
            request_data=ll_asr_req_data,
            priority=ELP1,
            worker_cwd=SCRIPT_DIR,
            timeout=ASR_WORKER_TIMEOUT  # From config.py
        )
        # Note: ASR_WORKER_TIMEOUT might need adjustment for low-latency vs high-quality if they differ significantly

        if elp1_asr_err or not (
                elp1_asr_response and isinstance(elp1_asr_response.get("result"), dict) and "text" in elp1_asr_response[
            "result"]):
            raise RuntimeError(f"Low-Latency ASR step failed: {elp1_asr_err or 'Invalid ASR worker response'}")

        raw_low_latency_transcription = elp1_asr_response["result"]["text"]
        logger.info(
            f"{request_id}: ELP1 Step 1.1: Low-Latency ASR successful. Snippet: '{raw_low_latency_transcription[:100]}...'")

        # --- Step 1.2: Auto-Correction with LLM (ELP1) ---
        corrected_transcription = raw_low_latency_transcription  # Default if correction fails
        if raw_low_latency_transcription and raw_low_latency_transcription.strip():
            logger.info(f"{request_id}: ELP1 Step 1.2: Auto-correcting transcript...")
            correction_prompt_filled = PROMPT_AUTOCORRECT_TRANSCRIPTION.format(
                raw_transcribed_text=raw_low_latency_transcription)
            correction_session_id = f"correct_asr_{request_id}"
            ai_chat.current_session_id = correction_session_id  # type: ignore

            llm_correction_output = await ai_chat.direct_generate(  # type: ignore
                db, correction_prompt_filled, correction_session_id,
                vlm_description=None, image_b64=None
            )

            if llm_correction_output and not (
                    isinstance(llm_correction_output, str) and "ERROR" in llm_correction_output.upper()):
                corrected_transcription = llm_correction_output.strip()
                logger.info(f"{request_id}: Auto-correction successful. Snippet: '{corrected_transcription[:100]}...'")
            else:
                logger.warning(
                    f"{request_id}: Auto-correction failed or LLM returned error-like response: '{llm_correction_output}'. Using previous transcript.")
                await asyncio.to_thread(add_interaction, db, session_id=session_id_for_log, mode="asr_service",
                                        input_type="log_warning",
                                        user_input=f"[ASR Correction Failed for Req ID {request_id}]",
                                        llm_response=f"LLM output for correction: {str(llm_correction_output)[:1000]}",
                                        classification="correction_failed_llm")
                await asyncio.to_thread(db.commit)

        # --- Step 1.3: Speaker Diarization with LLM (ELP1) ---
        diarized_text_final_for_client = corrected_transcription  # Default if diarization fails
        if corrected_transcription and corrected_transcription.strip():
            logger.info(f"{request_id}: ELP1 Step 1.3: Diarizing transcript...")
            diarization_prompt_filled = PROMPT_SPEAKER_DIARIZATION.format(transcribed_text=corrected_transcription)
            diarization_session_id = f"diarize_asr_{request_id}"
            ai_chat.current_session_id = diarization_session_id  # type: ignore

            llm_diarization_output = await ai_chat.direct_generate(  # type: ignore
                db, diarization_prompt_filled, diarization_session_id,
                vlm_description=None, image_b64=None
            )

            if llm_diarization_output and not (
                    isinstance(llm_diarization_output, str) and "ERROR" in llm_diarization_output.upper()):
                diarized_text_final_for_client = llm_diarization_output.strip()
                logger.info(
                    f"{request_id}: Speaker diarization attempt complete. Snippet: '{diarized_text_final_for_client[:100]}...'")
            else:
                logger.warning(
                    f"{request_id}: Speaker diarization LLM call failed or returned error: '{llm_diarization_output}'. Using non-diarized (but corrected) text.")
                await asyncio.to_thread(add_interaction, db, session_id=session_id_for_log, mode="asr_service",
                                        input_type="log_warning",
                                        user_input=f"[ASR Diarization Failed for Req ID {request_id}]",
                                        llm_response=f"LLM output for diarization: {str(llm_diarization_output)[:1000]}",
                                        classification="diarization_failed_llm")
                await asyncio.to_thread(db.commit)
        else:
            logger.info(f"{request_id}: Corrected transcript is empty. Skipping diarization.")

        # --- ELP1 Pipeline Complete: Format and Return Response to Client ---
        if response_format_req == "json":
            response_body = {"text": diarized_text_final_for_client}
            resp = Response(json.dumps(response_body), status=200, mimetype='application/json')
        elif response_format_req == "text":
            resp = Response(diarized_text_final_for_client, status=200, mimetype='text/plain; charset=utf-8')
        else:
            resp_data, status_code = _create_openai_error_response(
                f"Internal error: unhandled response format '{response_format_req}'.", status_code=500)
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_status_code = resp.status_code

        # --- Log the ELP1 result that was sent to client ---
        elp1_log_input = (f"[ASR Request ELP1 - File: {uploaded_filename or 'UnknownFile'}, "
                          f"Lang: {language_for_asr_steps}, API Format: {response_format_req}, "
                          f"LL-ASR Model: {WHISPER_LOW_LATENCY_MODEL_FILENAME}]")
        await asyncio.to_thread(add_interaction, db, session_id=session_id_for_log, mode="asr_service",
                                input_type="asr_transcribed_elp1",  # Mark as ELP1 result
                                user_input=elp1_log_input,
                                llm_response=diarized_text_final_for_client,
                                classification="transcription_elp1_pipeline_successful",
                                # Store ASR worker time for the low-latency pass for now
                                execution_time_ms=elp1_asr_duration_ms)
        await asyncio.to_thread(db.commit)

        # --- Spawn Background Task for High-Quality ASR (ELP0) ---
        logger.info(f"{request_id}: Spawning background task for high-quality ASR (ELP0)...")
        # The background task needs the original audio path.
        # It will perform its own ffmpeg conversion if needed.
        asyncio.create_task(_run_background_asr_and_translation_analysis(
            original_audio_path=temp_input_audio_path,  # Pass path, BG task will delete it
            elp1_transcription_final_for_client=diarized_text_final_for_client,  # Text returned to client
            elp1_translation_final_for_client=None,  # This is not a translation task
            session_id_for_log=session_id_for_log,
            request_id=request_id,
            language_asr=language_for_asr_steps,  # Language used for both ASR steps
            target_language_translation=None  # Not a translation task
        ))
        temp_input_audio_path = None  # Background task now owns the temp file lifecycle

    except ValueError as ve:
        logger.warning(f"{request_id}: Invalid ASR request: {ve}")
        resp_data, status_code = _create_openai_error_response(str(ve), err_type="invalid_request_error",
                                                               status_code=400)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_status_code = status_code
    except FileNotFoundError as fnf_err:
        logger.error(f"{request_id}: Server configuration error for ASR: {fnf_err}")
        resp_data, status_code = _create_openai_error_response(f"Server configuration error for ASR: {fnf_err}",
                                                               err_type="server_error", status_code=500)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_status_code = status_code
    except RuntimeError as rt_err:
        logger.error(f"{request_id}: ASR pipeline error: {rt_err}")
        resp_data, status_code = _create_openai_error_response(f"ASR failed: {rt_err}", err_type="server_error",
                                                               status_code=500)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_status_code = status_code
    except TaskInterruptedException as tie:  # If any ELP1 call was interrupted
        logger.warning(f"🚦 {request_id}: ASR/Diarization task INTERRUPTED: {tie}")
        resp_data, status_code = _create_openai_error_response(f"ASR task interrupted: {tie}", err_type="server_error",
                                                               code="task_interrupted", status_code=503)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_status_code = status_code
    except Exception as e:
        logger.exception(f"{request_id}: 🔥🔥 Unhandled exception in ASR endpoint (transcription):")
        error_message = f"Internal server error in ASR endpoint: {type(e).__name__}"
        resp_data_err, status_code_err = _create_openai_error_response(error_message, status_code=500)
        resp = Response(json.dumps(resp_data_err), status=status_code_err, mimetype='application/json')
        final_status_code = status_code_err
        try:
            if db:
                await asyncio.to_thread(add_interaction, db, session_id=session_id_for_log, mode="asr",
                                        input_type='error',
                                        user_input=f"ASR Handler Error. File: {uploaded_filename or 'N/A'}",
                                        llm_response=error_message[:2000])
                await asyncio.to_thread(db.commit)
        except Exception as db_log_err_final:
            logger.error(f"{request_id}: ❌ Failed log ASR handler main error: {db_log_err_final}")

    finally:
        # If temp_input_audio_path is not None here, it means the background task was not spawned
        # (e.g., due to an error before that point), so we should clean it up.
        if temp_input_audio_path and os.path.exists(temp_input_audio_path):
            logger.warning(
                f"{request_id}: Original temp audio file '{temp_input_audio_path}' was not passed to BG task or error occurred before spawn. Deleting.")
            try:
                await asyncio.to_thread(os.remove, temp_input_audio_path)
                logger.info(
                    f"{request_id}: Deleted temporary input audio file (in main handler finally): {temp_input_audio_path}")
            except Exception as e_del_final:
                logger.warning(f"{request_id}: Failed to delete temp file in main handler finally: {e_del_final}")

        duration_req_total = (time.monotonic() - start_req_time) * 1000
        logger.info(
            f"🏁 OpenAI-Style ASR Request {request_id} (transcription) handled in {duration_req_total:.2f} ms. Status: {final_status_code}")

    if resp is None:  # Fallback in case resp wasn't set due to an unexpected path
        logger.error(
            f"{request_id}: ASR Handler logic flaw - response object 'resp' was not assigned despite no clear earlier return!")
        resp_data_err, _ = _create_openai_error_response("Internal error: Handler did not produce response.",
                                                         status_code=500)
        resp = Response(json.dumps(resp_data_err), status=500, mimetype='application/json')
    return resp

# === NEW: Translation Audio Convo ===

@app.route("/v1/audio/translations", methods=["POST"])
async def handle_openai_audio_translations():
    start_req_time = time.monotonic()
    request_id = f"req-translate-{uuid.uuid4()}"
    logger.info(f"🚀 OpenAI-Style Audio Translation Request ID: {request_id} (Multi-Stage ELP1 + Background ELP0)")

    db: Session = g.db
    final_status_code: int = 500
    resp: Optional[Response] = None
    session_id_for_log: str = f"translate_req_{request_id}"
    uploaded_filename: Optional[str] = None
    temp_input_audio_path: Optional[str] = None  # Original uploaded file path

    # To store intermediate results for logging and background task
    raw_low_latency_transcription: Optional[str] = None
    corrected_transcription: Optional[str] = None
    diarized_transcription_for_client: Optional[str] = None  # This goes into translation
    quick_translated_text_for_client: Optional[str] = None  # This goes into TTS

    try:
        if not ENABLE_ASR:  # Master switch
            error_msg = "ASR/Translation capability disabled."
            logger.error(f"{request_id}: {error_msg}")
            resp_data, status_code = _create_openai_error_response(error_msg, err_type="server_error",
                                                                   code="translation_disabled", status_code=503)
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
            final_status_code = status_code
            return resp  # type: ignore

        # --- 1. Request Parsing & File Save (as before) ---
        if not request.content_type or not request.content_type.startswith('multipart/form-data'):
            raise ValueError("Invalid content type. Must be multipart/form-data.")
        audio_file_storage = request.files.get('file')
        model_requested = request.form.get('model')  # Should match AUDIO_TRANSLATION_MODEL_CLIENT_FACING
        target_language_code = request.form.get('target_language', DEFAULT_TRANSLATION_TARGET_LANGUAGE).lower()
        source_language_code_asr = request.form.get('source_language', "auto").lower()
        output_voice_requested = request.form.get('voice')
        output_audio_format = request.form.get('response_format', 'mp3').lower()
        session_id_for_log = request.form.get("session_id", session_id_for_log)
        if ai_chat: ai_chat.current_session_id = session_id_for_log
        if audio_file_storage and audio_file_storage.filename: uploaded_filename = secure_filename(
            audio_file_storage.filename)

        if not audio_file_storage: raise ValueError("'file' field is required.")
        if not model_requested or model_requested != AUDIO_TRANSLATION_MODEL_CLIENT_FACING:
            raise ValueError(f"Invalid 'model'. Expected '{AUDIO_TRANSLATION_MODEL_CLIENT_FACING}'.")

        temp_audio_dir = os.path.join(SCRIPT_DIR, "temp_audio_worker_files")
        await asyncio.to_thread(os.makedirs, temp_audio_dir, exist_ok=True)
        _, file_extension = os.path.splitext(uploaded_filename or ".tmpaud")
        # Create a uniquely named temp file that persists until explicitly deleted
        temp_fd, temp_input_audio_path = tempfile.mkstemp(prefix="translate_orig_", suffix=file_extension,
                                                          dir=temp_audio_dir)
        os.close(temp_fd)  # We just want the name; save will reopen and write
        await asyncio.to_thread(audio_file_storage.save, temp_input_audio_path)
        logger.info(f"{request_id}: Input audio saved to: {temp_input_audio_path}")

        # --- ELP1 PIPELINE ---
        # --- Step 1.1: Low-Latency ASR ---
        logger.info(f"{request_id}: ELP1 Step 1.1: Low-Latency ASR (Model: {WHISPER_LOW_LATENCY_MODEL_FILENAME})...")
        asr_worker_script = os.path.join(SCRIPT_DIR, "audio_worker.py")
        ll_asr_cmd = [APP_PYTHON_EXECUTABLE, asr_worker_script, "--task-type", "asr", "--model-dir", WHISPER_MODEL_DIR,
                      "--temp-dir", temp_audio_dir]
        ll_asr_req_data = {"input_audio_path": temp_input_audio_path,
                           "whisper_model_name": WHISPER_LOW_LATENCY_MODEL_FILENAME,
                           "language": source_language_code_asr, "request_id": f"{request_id}-llasr"}

        ll_asr_resp, ll_asr_err = await asyncio.to_thread(_execute_audio_worker_with_priority, ll_asr_cmd,
                                                          ll_asr_req_data, ELP1, SCRIPT_DIR, ASR_WORKER_TIMEOUT)
        if ll_asr_err or not (
                ll_asr_resp and isinstance(ll_asr_resp.get("result"), dict) and "text" in ll_asr_resp["result"]):
            raise RuntimeError(f"Low-Latency ASR failed: {ll_asr_err or 'Invalid ASR worker response'}")
        raw_low_latency_transcription = ll_asr_resp["result"]["text"]
        logger.info(
            f"{request_id}: ELP1 Step 1.1: Low-Latency ASR successful. Snippet: '{raw_low_latency_transcription[:100]}...'")

        # --- Step 1.2: Auto-Correction (LLM ELP1) ---
        corrected_transcription = raw_low_latency_transcription
        if raw_low_latency_transcription and raw_low_latency_transcription.strip():
            logger.info(f"{request_id}: ELP1 Step 1.2: Auto-correcting transcript...")
            correction_prompt = PROMPT_AUTOCORRECT_TRANSCRIPTION.format(
                raw_transcribed_text=raw_low_latency_transcription)
            correction_session_id = f"correct_{request_id}"
            if ai_chat: ai_chat.current_session_id = correction_session_id
            llm_correction_output = await ai_chat.direct_generate(db, correction_prompt, correction_session_id, None,
                                                                  None)
            if llm_correction_output and not (
                    isinstance(llm_correction_output, str) and "ERROR" in llm_correction_output.upper()):
                corrected_transcription = llm_correction_output.strip()
                logger.info(f"{request_id}: Auto-correction successful. Snippet: '{corrected_transcription[:100]}...'")
            else:
                logger.warning(
                    f"{request_id}: Auto-correction failed/error: '{llm_correction_output}'. Using raw LL ASR text.")

        # --- Step 1.3: Diarization (LLM ELP1) ---
        diarized_text_for_translation = corrected_transcription
        if corrected_transcription and corrected_transcription.strip():
            logger.info(f"{request_id}: ELP1 Step 1.3: Diarizing transcript...")
            diarization_prompt = PROMPT_SPEAKER_DIARIZATION.format(transcribed_text=corrected_transcription)
            diarization_session_id = f"diarize_{request_id}"
            if ai_chat: ai_chat.current_session_id = diarization_session_id
            llm_diarization_output = await ai_chat.direct_generate(db, diarization_prompt, diarization_session_id, None,
                                                                   None)
            if llm_diarization_output and not (
                    isinstance(llm_diarization_output, str) and "ERROR" in llm_diarization_output.upper()):
                diarized_text_for_translation = llm_diarization_output.strip()
                logger.info(
                    f"{request_id}: Diarization successful. Snippet: '{diarized_text_for_translation[:100]}...'")
            else:
                logger.warning(
                    f"{request_id}: Diarization failed/error: '{llm_diarization_output}'. Using non-diarized (but corrected) text.")

        # --- Step 1.4: Quick Translation (LLM ELP1) ---
        logger.info(
            f"{request_id}: ELP1 Step 1.4: Translating to '{target_language_code}' (LLM role '{TRANSLATION_LLM_ROLE}')...")
        translation_model = ai_provider.get_model(TRANSLATION_LLM_ROLE)  # type: ignore
        if not translation_model: raise RuntimeError(f"LLM role '{TRANSLATION_LLM_ROLE}' for translation unavailable.")

        src_lang_full = langcodes.Language.make(
            language=source_language_code_asr).display_name() if source_language_code_asr != "auto" else "Unknown (auto-detect)"
        tgt_lang_full = langcodes.Language.make(language=target_language_code).display_name()

        trans_prompt_input = {"text_to_translate": diarized_text_for_translation,
                              "target_language_full_name": tgt_lang_full, "target_language_code": target_language_code,
                              "source_language_full_name": src_lang_full,
                              "source_language_code": source_language_code_asr}
        trans_chain = ChatPromptTemplate.from_template(PROMPT_TRANSLATE_TEXT) | translation_model | StrOutputParser()
        trans_timing_data = {"session_id": session_id_for_log, "mode": "translation_elp1"}

        # _call_llm_with_timing is sync, direct_generate (which uses it) is async.
        # If direct_generate is called, it handles the threading.
        # For a direct chain invoke like this, we need asyncio.to_thread for the sync _call_llm_with_timing
        quick_translated_text_for_client = await asyncio.to_thread(
            ai_chat._call_llm_with_timing, trans_chain, trans_prompt_input, trans_timing_data, priority=ELP1
            # type: ignore
        )
        if not quick_translated_text_for_client or (isinstance(quick_translated_text_for_client, str) and (
                "ERROR" in quick_translated_text_for_client.upper() or "Traceback" in quick_translated_text_for_client)):
            raise RuntimeError(f"ELP1 LLM translation failed. Response: {quick_translated_text_for_client}")
        quick_translated_text_for_client = quick_translated_text_for_client.strip()
        logger.info(
            f"{request_id}: ELP1 Step 1.4: Quick translation successful. Snippet: '{quick_translated_text_for_client[:100]}...'")

        # --- Step 1.5: TTS of Quick Translation (ELP1) ---
        logger.info(f"{request_id}: ELP1 Step 1.5: Synthesizing quick translated text to audio...")
        final_tts_voice = output_voice_requested
        if not final_tts_voice:
            lang_map = {"en": "EN-US", "es": "ES-ES", "fr": "FR-FR", "de": "DE-DE", "zh": "ZH-CN", "ja": "JP-JA",
                        "ko": "KO-KR"}
            final_tts_voice = lang_map.get(target_language_code, f"{target_language_code.upper()}-US")

        tts_worker_cmd = [APP_PYTHON_EXECUTABLE, asr_worker_script, "--task-type", "tts", "--model-lang",
                          target_language_code.upper(), "--model-dir", WHISPER_MODEL_DIR, "--temp-dir", temp_audio_dir,
                          "--device", "auto"]
        tts_req_data = {"input": quick_translated_text_for_client, "voice": final_tts_voice,
                        "response_format": output_audio_format, "request_id": f"{request_id}-elp1-tts"}

        tts_resp, tts_err = await asyncio.to_thread(_execute_audio_worker_with_priority, tts_worker_cmd, tts_req_data,
                                                    ELP1, SCRIPT_DIR, TTS_WORKER_TIMEOUT)
        if tts_err or not (
                tts_resp and isinstance(tts_resp.get("result"), dict) and "audio_base64" in tts_resp["result"]):
            raise RuntimeError(f"ELP1 TTS step failed: {tts_err or 'Invalid TTS worker response'}")

        audio_info = tts_resp["result"]
        audio_b64_data = audio_info["audio_base64"]
        final_audio_format = audio_info.get("format", output_audio_format)
        final_mime_type = audio_info.get("mime_type", f"audio/{final_audio_format}")
        logger.info(
            f"{request_id}: ELP1 Step 1.5: Quick translated audio synthesis successful. Format: {final_audio_format}.")

        # --- Return ELP1 Audio Response to Client ---
        audio_bytes = base64.b64decode(audio_b64_data)
        resp = Response(audio_bytes, status=200, mimetype=final_mime_type)
        final_status_code = 200

        # --- Log the ELP1 pipeline outcome (before spawning background) ---
        elp1_log_summary = (f"ELP1 Pipeline for Request {request_id}:\n"
                            f"LL ASR: '{raw_low_latency_transcription[:70]}...' -> \n"
                            f"Corrected: '{corrected_transcription[:70]}...' -> \n"
                            f"Diarized: '{diarized_text_for_translation[:70]}...' -> \n"
                            f"Translated: '{quick_translated_text_for_client[:70]}...' -> \n"
                            f"TTS Output ({final_tts_voice}, {final_audio_format})")
        await asyncio.to_thread(add_interaction, db, session_id=session_id_for_log, mode="audio_translation",
                                input_type="elp1_pipeline_summary",
                                user_input=f"[AudioTranslate ELP1 - File: {uploaded_filename or 'UnknownFile'}]",
                                llm_response=elp1_log_summary, classification="translation_elp1_successful",
                                execution_time_ms=(time.monotonic() - start_req_time) * 1000)
        await asyncio.to_thread(db.commit)

        # --- Spawn Background Task for High-Quality ASR & Deeper Translation (ELP0) ---
        logger.info(f"{request_id}: Spawning background task for high-quality ASR & deep translation (ELP0)...")
        asyncio.create_task(_run_background_asr_and_translation_analysis(
            original_audio_path=temp_input_audio_path,
            elp1_transcription_final_for_client=diarized_text_for_translation,  # The text that was translated for ELP1
            elp1_translation_final_for_client=quick_translated_text_for_client,  # The ELP1 translation
            session_id_for_log=session_id_for_log,
            request_id=request_id,
            language_asr=source_language_code_asr,  # Language used for ASR
            target_language_translation=target_language_code  # Target language for translation
        ))
        temp_input_audio_path = None  # Background task now owns the temp file for its ASR pass

    # ... (existing except ValueError, FileNotFoundError, RuntimeError, TaskInterruptedException, Exception as e blocks from response #67) ...
    # Ensure all db operations and file operations in except/finally are wrapped in asyncio.to_thread
    except ValueError as ve:
        logger.warning(f"{request_id}: Invalid Audio Translation request: {ve}")
        resp_data, status_code = _create_openai_error_response(str(ve), err_type="invalid_request_error",
                                                               status_code=400)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_status_code = status_code
    except FileNotFoundError as fnf_err:
        logger.error(f"{request_id}: Server configuration error for Audio Translation: {fnf_err}")
        resp_data, status_code = _create_openai_error_response(f"Server configuration error: {fnf_err}",
                                                               err_type="server_error", status_code=500)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_status_code = status_code
    except RuntimeError as rt_err:
        logger.error(f"{request_id}: Audio Translation pipeline error: {rt_err}")
        resp_data, status_code = _create_openai_error_response(f"Audio Translation failed: {rt_err}",
                                                               err_type="server_error", status_code=500)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_status_code = status_code
    except TaskInterruptedException as tie:
        logger.warning(f"🚦 {request_id}: Audio Translation task INTERRUPTED: {tie}")
        resp_data, status_code = _create_openai_error_response(f"Translation task interrupted: {tie}",
                                                               err_type="server_error", code="task_interrupted",
                                                               status_code=503)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_status_code = status_code
    except Exception as e:
        logger.exception(f"{request_id}: 🔥🔥 Unhandled exception in Audio Translation endpoint:")
        error_message = f"Internal server error in Audio Translation endpoint: {type(e).__name__}"
        resp_data, status_code = _create_openai_error_response(error_message, status_code=500)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_status_code = status_code
        try:
            if db:
                await asyncio.to_thread(add_interaction, db, session_id=session_id_for_log, mode="audio_translation",
                                        input_type='error',
                                        user_input=f"AudioTranslation Handler Error. File: {uploaded_filename or 'N/A'}",
                                        llm_response=error_message[:2000])
                await asyncio.to_thread(db.commit)
        except Exception as db_err_log:
            logger.error(f"{request_id}: ❌ Failed log AudioTranslation handler error: {db_err_log}")
    finally:
        # If temp_input_audio_path was not passed to a background task (e.g., due to early error)
        # or if the background task is designed to copy it, then delete it here.
        # Current design: background task uses the original path, so it's responsible for deletion OR we pass a copy.
        # The _run_background_asr_and_translation_analysis now handles deleting the original_audio_path.
        if temp_input_audio_path and os.path.exists(temp_input_audio_path):
            logger.warning(
                f"{request_id}: Original temp audio file '{temp_input_audio_path}' was not consumed by a background task or an error occurred before spawning. Deleting.")
            try:
                await asyncio.to_thread(os.remove, temp_input_audio_path)
                logger.info(
                    f"{request_id}: Deleted temporary input audio file (in main handler finally): {temp_input_audio_path}")
            except Exception as e_del:
                logger.warning(
                    f"{request_id}: Failed to delete temporary input audio file '{temp_input_audio_path}' (in main handler finally): {e_del}")

        duration_req = (time.monotonic() - start_req_time) * 1000
        logger.info(
            f"🏁 OpenAI-Style Audio Translation Request {request_id} handled in {duration_req:.2f} ms. Status: {final_status_code}")

    if resp is None:
        logger.error(f"{request_id}: Audio Translation Handler logic flaw - response object 'resp' was not assigned!")
        resp_data, _ = _create_openai_error_response("Internal error: Handler failed to produce a response.",
                                                     status_code=500)
        resp = Response(json.dumps(resp_data), status=500, mimetype='application/json')
    return resp


# === NEW: OpenAI Compatible TTS Endpoint ===
@app.route("/v1/audio/speech", methods=["POST"])
def handle_openai_tts():
    """
    Handles requests mimicking OpenAI's Text-to-Speech endpoint.
    Expects model "Zephyloid-Alpha", uses audio_worker.py with ELP1 priority.
    """
    start_req = time.monotonic()
    request_id = f"req-tts-{uuid.uuid4()}"
    logger.info(f"🚀 Flask OpenAI-Style TTS Request ID: {request_id} (Worker ELP1)")

    db: Session = g.db
    session_id: str = f"tts_req_default_{request_id}"
    raw_request_data: Optional[Dict[str, Any]] = None
    input_text: Optional[str] = None
    model_requested: Optional[str] = None
    voice_requested: Optional[str] = None
    response_format_requested: Optional[str] = "mp3"

    final_response_status_code: int = 500
    resp: Optional[Response] = None
    request_data_snippet_for_log: str = "No request data processed"

    try:
        try:
            raw_request_data = request.get_json()
            if not raw_request_data:
                raise ValueError("Empty JSON payload received.")
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
            resp_data, status_code = _create_openai_error_response(
                f"Request body is missing or invalid JSON: {json_err}",
                err_type="invalid_request_error", status_code=400
            )
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
            final_response_status_code = status_code
            return resp

        input_text = raw_request_data.get("input")
        model_requested = raw_request_data.get("model")
        voice_requested = raw_request_data.get("voice")
        session_id = raw_request_data.get("session_id", session_id)
        response_format_requested = raw_request_data.get("response_format", "mp3").lower()

        logger.debug(
            f"{request_id}: TTS Request Parsed - SessionID: {session_id}, Input: '{str(input_text)[:50]}...', "
            f"Client Model Req: {model_requested}, Internal Voice/Speaker: {voice_requested}, "
            f"Format: {response_format_requested}"
        )

        if not input_text or not isinstance(input_text, str):
            raise ValueError("'input' field is required and must be a string.")
        if not model_requested or not isinstance(model_requested, str):
            raise ValueError("'model' field (e.g., 'Zephyloid-Alpha') is required.")
        if not voice_requested or not isinstance(voice_requested, str):
            raise ValueError("'voice' field (MeloTTS speaker ID, e.g., EN-US) is required.")

        if model_requested != TTS_MODEL_NAME_CLIENT_FACING:  # TTS_MODEL_NAME_CLIENT_FACING from config.py
            logger.warning(
                f"{request_id}: Invalid TTS model requested '{model_requested}'. Expected '{TTS_MODEL_NAME_CLIENT_FACING}'.")
            resp_data, status_code = _create_openai_error_response(
                f"Invalid model. This endpoint only supports the '{TTS_MODEL_NAME_CLIENT_FACING}' model for TTS.",
                err_type="invalid_request_error", code="model_not_found", status_code=404
            )
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
            final_response_status_code = status_code
            return resp

        melo_language = "EN"
        try:
            lang_part = voice_requested.split('-')[0].upper()
            supported_melo_langs = ["EN", "ZH", "JP", "ES", "FR", "KR", "DE"]
            if lang_part in supported_melo_langs:
                melo_language = lang_part
            else:
                logger.warning(
                    f"{request_id}: Could not infer a supported language from voice '{voice_requested}'. Defaulting to {melo_language}.")
        except Exception:
            logger.warning(
                f"{request_id}: Error parsing language from voice '{voice_requested}'. Defaulting to {melo_language}.")

        audio_worker_script_path = os.path.join(SCRIPT_DIR, "audio_worker.py")  # SCRIPT_DIR is where app.py is
        if not os.path.exists(audio_worker_script_path):
            logger.error(f"{request_id}: audio_worker.py not found at {audio_worker_script_path}")
            raise FileNotFoundError(f"Audio worker script missing at {audio_worker_script_path}")

        temp_audio_dir = os.path.join(SCRIPT_DIR, "temp_audio_worker_files")
        os.makedirs(temp_audio_dir, exist_ok=True)

        # Ensure APP_PYTHON_EXECUTABLE is defined (usually sys.executable at top of app.py)
        # Ensure WHISPER_MODEL_DIR is imported from config.py (this directory is used for all models, including MeloTTS data if needed by worker)
        worker_command = [
            APP_PYTHON_EXECUTABLE,
            audio_worker_script_path,
            "--task-type", "tts",  # Specify TTS task for the worker
            "--model-lang", melo_language,
            "--device", "auto",  # Or make this configurable via app's config
            "--model-dir", WHISPER_MODEL_DIR,  # Use the general model pool path from config.py
            "--temp-dir", temp_audio_dir
        ]

        worker_request_data = {
            "input": input_text,
            "voice": voice_requested,
            "response_format": response_format_requested,
            "request_id": request_id
        }

        logger.info(f"{request_id}: Executing audio worker with ELP1 priority...")
        # _execute_audio_worker_with_priority is synchronous
        parsed_response_from_worker, error_string_from_worker = _execute_audio_worker_with_priority(
            worker_command=worker_command,
            request_data=worker_request_data,
            priority=ELP1,  # Use ELP1 for user-facing TTS
            worker_cwd=SCRIPT_DIR,
            timeout=TTS_WORKER_TIMEOUT  # Adjust timeout as needed for TTS generation through config.py TTS_WORKER_TIMEOUT
        )

        if error_string_from_worker:
            logger.error(f"{request_id}: Audio worker execution failed: {error_string_from_worker}")
            resp_data, status_code = _create_openai_error_response(
                f"Audio generation failed: {error_string_from_worker}",
                err_type="server_error", status_code=500
            )
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
            final_response_status_code = status_code
        elif parsed_response_from_worker and "result" in parsed_response_from_worker and "audio_base64" in \
                parsed_response_from_worker["result"]:
            audio_info = parsed_response_from_worker["result"]
            audio_b64_data = audio_info["audio_base64"]
            actual_audio_format = audio_info.get("format", "mp3")
            response_mime_type = audio_info.get("mime_type", f"audio/{actual_audio_format}")

            logger.info(
                f"{request_id}: Audio successfully generated by worker. Format: {actual_audio_format}, Length (b64): {len(audio_b64_data)}")
            try:
                audio_bytes = base64.b64decode(audio_b64_data)
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
            logger.error(
                f"{request_id}: Audio worker returned invalid or incomplete response: {parsed_response_from_worker}")
            resp_data, status_code = _create_openai_error_response(
                "Audio worker returned an invalid response structure.",
                err_type="server_error", status_code=500
            )
            resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
            final_response_status_code = status_code

    except ValueError as ve:
        logger.warning(f"{request_id}: Invalid TTS request parameters: {ve}")
        resp_data, status_code = _create_openai_error_response(
            str(ve), err_type="invalid_request_error", status_code=400)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_response_status_code = status_code
    except FileNotFoundError as fnf_err:
        logger.error(f"{request_id}: Server configuration error for TTS: {fnf_err}")
        resp_data, status_code = _create_openai_error_response(
            f"Server configuration error for TTS: {fnf_err}", err_type="server_error", status_code=500)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_response_status_code = status_code
    except Exception as main_handler_err:
        logger.exception(f"{request_id}: 🔥🔥 Unhandled exception in TTS endpoint main handler:")
        error_message = f"Internal server error in TTS endpoint: {type(main_handler_err).__name__}"
        resp_data, status_code = _create_openai_error_response(error_message, status_code=500)
        resp = Response(json.dumps(resp_data), status=status_code, mimetype='application/json')
        final_response_status_code = status_code
        try:
            if 'db' in g and g.db:
                add_interaction(g.db, session_id=session_id, mode="tts", input_type='error',
                                user_input=f"TTS Handler Error. Request: {request_data_snippet_for_log}",
                                llm_response=error_message[:2000])
                db.commit()  # Ensure commit if add_interaction doesn't do it
            else:
                logger.error(f"{request_id}: Cannot log TTS handler error: DB session 'g.db' unavailable.")
        except Exception as db_err_log:
            logger.error(f"{request_id}: ❌ Failed log TTS handler error to DB: {db_err_log}")
            if 'db' in g and g.db: db.rollback()


    finally:
        duration_req = (time.monotonic() - start_req) * 1000
        logger.info(
            f"🏁 OpenAI-Style TTS Request {request_id} handled in {duration_req:.2f} ms. Final Status: {final_response_status_code}")
        # g.db is closed automatically by the @app.teardown_request handler

    if resp is None:
        logger.error(f"{request_id}: TTS Handler logic flaw - response object 'resp' was not assigned!")
        resp_data, _ = _create_openai_error_response(
            "Internal error: TTS Handler failed to produce a response object.",
            err_type="server_error", status_code=500
        )
        resp = Response(json.dumps(resp_data), status=500, mimetype='application/json')
        try:
            if 'db' in g and g.db:
                add_interaction(g.db, session_id=session_id, mode="tts", input_type='error',
                                user_input=f"TTS Handler No Resp Object. Req: {request_data_snippet_for_log}",
                                llm_response="Critical: No response object created by handler.")
                db.commit()
        except:
            pass  # Best effort at this point
    return resp

# === NEW: OpenAI Compatible Image Generation Endpoint (Stub) ===
@app.route("/v1/images/generations", methods=["POST"])
async def handle_openai_image_generations():  # Route is async
    start_req = time.monotonic()
    request_id = f"req-img-gen-{uuid.uuid4()}"
    logger.info(f"🚀 OpenAI-Style Image Generation Request ID: {request_id} (ELP1 Priority)")

    # Get DB session from Flask's g or create if not present (ensure before_request/teardown_request handle this)
    # For simplicity here, assuming g.db is correctly managed by Flask context handlers
    db: Optional[Session] = getattr(g, 'db', None)
    if db is None:  # Fallback if g.db is not set (e.g. if before_request failed or not run)
        logger.error(f"{request_id}: DB session not found in g.db. Creating temporary session for this request.")
        db_temp_session = SessionLocal()  # type: ignore
        db_to_use = db_temp_session
    else:
        db_to_use = db

    final_response_status_code: int = 500
    resp: Optional[Response] = None
    session_id_for_log: str = f"img_gen_req_default_{request_id}"
    raw_request_data: Optional[Dict[str, Any]] = None
    request_data_snippet_for_log: str = "No request data processed"

    try:
        # --- 1. Get and Validate Request JSON Body ---
        try:
            raw_request_data = request.get_json()  # Use await if using Quart, or request.get_json() for Flask
            if not raw_request_data:
                raise ValueError("Empty JSON payload received.")
            try:
                request_data_snippet_for_log = json.dumps(raw_request_data)[:1000]
            except:
                request_data_snippet_for_log = str(raw_request_data)[:1000]
        except Exception as json_err:
            logger.warning(f"{request_id}: Failed to get/parse JSON body: {json_err}")
            try:
                request_data_snippet_for_log = (await request.get_data(as_text=True))[:1000]  # await for Quart
            except:
                request_data_snippet_for_log = "Could not read request body"
            resp_data, status_code_val = _create_openai_error_response(
                f"Request body is missing or invalid JSON: {json_err}", err_type="invalid_request_error",
                status_code=400)
            resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
            final_response_status_code = status_code_val
            return resp  # Early return

        # --- 2. Extract Expected Parameters ---
        prompt_from_user = raw_request_data.get("prompt")
        model_requested = raw_request_data.get("model")
        n_images_requested_by_client = raw_request_data.get("n")
        size_requested_str = raw_request_data.get("size", IMAGE_GEN_DEFAULT_SIZE)
        response_format_requested = raw_request_data.get("response_format", "b64_json").lower()
        # Optional OpenAI params, currently logged but not all used by worker
        quality_requested = raw_request_data.get("quality", "standard")
        style_requested = raw_request_data.get("style", "vivid")
        user_provided_id_for_tracking = raw_request_data.get("user")

        session_id_for_log = raw_request_data.get("session_id", session_id_for_log)

        if ai_chat:  # Ensure ai_chat global instance is available
            ai_chat.current_session_id = session_id_for_log  # Set for helpers in ai_chat
        else:
            logger.error(
                f"{request_id}: Global 'ai_chat' instance not available. Cannot proceed with image generation context.")
            resp_data, status_code_val = _create_openai_error_response("Server AI component (AIChat) not ready.",
                                                                       err_type="server_error", status_code=503)
            resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
            final_response_status_code = status_code_val
            return resp

        # Determine n_images (number of images to generate)
        if n_images_requested_by_client is None:
            n_images = 1  # Default to 1 image for this direct ELP1 endpoint.
            # Previous default was 2, changed to 1 for faster single user requests.
            # Can be configured or kept at 1. Max can be capped.
            logger.info(f"{request_id}: 'n' not specified, defaulting to {n_images} for ELP1 image generation.")
        else:
            try:
                n_images = int(n_images_requested_by_client)
                if n_images < 1: n_images = 1  # Min 1
                # MAX_IMAGES_PER_REQUEST = 4 # Example cap
                # if n_images > MAX_IMAGES_PER_REQUEST: n_images = MAX_IMAGES_PER_REQUEST; logger.warning(...)
            except ValueError:
                logger.warning(f"{request_id}: Invalid 'n': '{n_images_requested_by_client}'. Defaulting to 1.")
                n_images = 1

        logger.debug(
            f"{request_id}: Image Gen Request Parsed - Prompt: '{str(prompt_from_user)[:50]}...', "
            f"ModelReq: {model_requested}, N (final): {n_images}, Size: {size_requested_str}, RespFormat: {response_format_requested}"
        )

        # --- 3. Validate Core Parameters ---
        if not prompt_from_user or not isinstance(prompt_from_user, str):
            raise ValueError("'prompt' field is required and must be a string for image generation.")
        if not model_requested or model_requested != IMAGE_GEN_MODEL_NAME_CLIENT_FACING:
            raise ValueError(f"Invalid 'model'. This endpoint supports '{IMAGE_GEN_MODEL_NAME_CLIENT_FACING}'.")
        if response_format_requested not in ["b64_json", "url"]:
            logger.warning(
                f"{request_id}: Invalid 'response_format': {response_format_requested}. Defaulting to 'b64_json'.")
            response_format_requested = "b64_json"

        # --- 4. Refine User Prompt using RAG Context (ELP1) ---
        logger.info(f"{request_id}: Refining user prompt for image generation (ELP1)...")

        wrapped_rag_result = await asyncio.to_thread(
            ai_chat._get_rag_retriever_thread_wrapper,
            db_to_use,
            prompt_from_user,  # Use original prompt for RAG context query
            ELP1
        )

        session_hist_retriever_for_refine: Optional[Any] = None
        if wrapped_rag_result.get("status") == "success":
            rag_data_tuple = wrapped_rag_result.get("data")
            if isinstance(rag_data_tuple, tuple) and len(rag_data_tuple) == 4:
                _url_ret_temp, session_hist_retriever_for_refine, _refl_ret_temp, _ids_temp = rag_data_tuple
            else:  # Should be caught by wrapper, but safeguard
                raise RuntimeError(f"RAG wrapper returned unexpected data structure for image prompt: {rag_data_tuple}")
        elif wrapped_rag_result.get("status") == "interrupted":
            raise TaskInterruptedException(wrapped_rag_result.get("error_message", "RAG for image prompt interrupted"))
        else:  # Error
            raise RuntimeError(
                f"RAG for image prompt refinement failed: {wrapped_rag_result.get('error_message', 'Unknown RAG error')}")

        retrieved_history_docs = []
        if session_hist_retriever_for_refine:
            retrieved_history_docs = await asyncio.to_thread(session_hist_retriever_for_refine.invoke, prompt_from_user)

        history_rag_str = ai_chat._format_docs(retrieved_history_docs, source_type="History RAG")

        direct_hist_interactions_list = await asyncio.to_thread(get_global_recent_interactions, db_to_use, limit=3)
        recent_direct_history_str = ai_chat._format_direct_history(direct_hist_interactions_list)

        refined_prompt_for_generation = await ai_chat._refine_direct_image_prompt_async(
            db=db_to_use, session_id=session_id_for_log, user_image_request=prompt_from_user,
            history_rag_str=history_rag_str, recent_direct_history_str=recent_direct_history_str,
            priority=ELP1
        )

        if not refined_prompt_for_generation or refined_prompt_for_generation == prompt_from_user:
            logger.info(f"{request_id}: Prompt refinement yielded no change or failed. Using original prompt.")
            refined_prompt_for_generation = prompt_from_user
        else:
            logger.info(f"{request_id}: Using refined prompt for image generation: '{refined_prompt_for_generation}'")

        try:  # Log the prompt that will be used
            add_interaction(db_to_use, session_id=session_id_for_log, mode="image_gen",
                            input_type="text_prompt_to_img_worker",
                            user_input=prompt_from_user,  # Original prompt
                            llm_response=refined_prompt_for_generation)  # Refined prompt sent to worker
            db_to_use.commit()
        except Exception as db_log_err_prompt:
            logger.error(f"{request_id}: Failed to log refined image prompt: {db_log_err_prompt}")
            if db_to_use: db_to_use.rollback()

        # --- 5. Generate Image(s) using AIProvider (ELP1) ---
        logger.info(
            f"{request_id}: Requesting {n_images} image(s) from AIProvider (ELP1). Prompt: '{refined_prompt_for_generation[:100]}...'")
        all_generated_image_data_items = []
        error_during_loop = False
        loop_error_message = None

        for i in range(n_images):
            if error_during_loop: break
            logger.info(f"{request_id}: Generating image {i + 1}/{n_images}...")
            # ai_provider.generate_image_async returns: Tuple[Optional[List[Dict[str, Optional[str]]]], Optional[str]]
            # The first element is a list of image data dicts (usually one dict per call for this worker)
            # The second is an error message string if any.
            list_of_one_image_dict, image_gen_err_msg = await ai_provider.generate_image_async(
                prompt=refined_prompt_for_generation,
                image_base64=None,  # This endpoint is for txt2img primarily
                priority=ELP1
            )
            if image_gen_err_msg:
                logger.error(f"{request_id}: Image generation failed for attempt {i + 1}: {image_gen_err_msg}")
                loop_error_message = image_gen_err_msg
                if interruption_error_marker in image_gen_err_msg: raise TaskInterruptedException(image_gen_err_msg)
                error_during_loop = True
                break
            elif list_of_one_image_dict and isinstance(list_of_one_image_dict, list) and list_of_one_image_dict:
                all_generated_image_data_items.append(list_of_one_image_dict[0])  # Append the single image dict
                logger.info(f"{request_id}: Image {i + 1}/{n_images} data received.")
            else:
                logger.warning(f"{request_id}: Image generation attempt {i + 1} returned no data and no error.")
                loop_error_message = "Image worker returned no data for an image attempt."
                error_during_loop = True
                break

        if error_during_loop or not all_generated_image_data_items:
            final_err_msg = loop_error_message or "Image generation failed to produce any results."
            resp_data, status_code_val = _create_openai_error_response(final_err_msg, err_type="server_error",
                                                                       status_code=500)
            resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
            final_response_status_code = status_code_val
            return resp

        # --- 6. Format Response ---
        response_data_list_for_client = []
        for img_data_item in all_generated_image_data_items:
            png_b64 = img_data_item.get("b64_json")  # Expecting PNG base64 from worker
            # avif_b64 = img_data_item.get("b64_avif") # Worker might also provide AVIF
            if png_b64:
                if response_format_requested == "b64_json":
                    response_data_list_for_client.append({"b64_json": png_b64})
                elif response_format_requested == "url":  # Return data URI
                    logger.warning(f"{request_id}: Returning data URI for 'url' format image request.")
                    response_data_list_for_client.append({"url": f"data:image/png;base64,{png_b64}"})
            else:
                logger.error(f"{request_id}: Worker image data item missing 'b64_json' (PNG). Item: {img_data_item}")

        if not response_data_list_for_client:
            logger.error(f"{request_id}: No valid PNG b64_json data found after processing worker response(s).")
            resp_data, status_code_val = _create_openai_error_response("Failed to get valid image data from worker.",
                                                                       err_type="server_error", status_code=500)
            resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
            final_response_status_code = status_code_val
            return resp

        openai_response_body = {"created": int(time.time()), "data": response_data_list_for_client}
        response_payload = json.dumps(openai_response_body)
        resp = Response(response_payload, status=200, mimetype='application/json')
        final_response_status_code = 200

    except ValueError as ve:  # Catches explicit ValueErrors from parameter validation
        logger.warning(f"{request_id}: Invalid Image Gen request (ValueError): {ve}")
        resp_data, status_code_val = _create_openai_error_response(str(ve), err_type="invalid_request_error",
                                                                   status_code=400)
        resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
        final_response_status_code = status_code_val
    except TaskInterruptedException as tie:  # Catches interruptions from RAG or image gen
        logger.warning(f"🚦 {request_id}: Image Generation request (ELP1) INTERRUPTED: {tie}")
        resp_data, status_code_val = _create_openai_error_response(f"Image generation task was interrupted: {tie}",
                                                                   err_type="server_error", code="task_interrupted",
                                                                   status_code=503)
        resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
        final_response_status_code = status_code_val
    except Exception as e_main:  # Catch-all for other unexpected errors
        logger.exception(f"{request_id}: 🔥🔥 Unhandled exception in Image Gen endpoint main try block:")
        error_message = f"Internal server error in Image Gen endpoint: {type(e_main).__name__} - {str(e_main)}"
        resp_data, status_code_val = _create_openai_error_response(error_message, err_type="server_error",
                                                                   status_code=500)
        resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
        final_response_status_code = status_code_val
        try:
            if db_to_use:
                add_interaction(db_to_use, session_id=session_id_for_log, mode="image_gen", input_type='error',
                                user_input=f"Image Gen Handler Error. Req: {request_data_snippet_for_log}",
                                llm_response=error_message[:2000])
                db_to_use.commit()
        except Exception as db_err_log_main:
            logger.error(f"{request_id}: ❌ Failed log main Image Gen handler error to DB: {db_err_log_main}")
            if db_to_use: db_to_use.rollback()
    finally:
        duration_req = (time.monotonic() - start_req) * 1000
        logger.info(
            f"🏁 OpenAI-Style Image Gen Request {request_id} handled in {duration_req:.2f} ms. Final HTTP Status: {final_response_status_code}")
        if 'db_temp_session' in locals() and db_temp_session:  # Close temp session if created
            db_temp_session.close()
            logger.debug(f"{request_id}: Closed temporary DB session for image gen.")

    if resp is None:  # Should ideally not be reached if all paths assign to resp
        logger.error(f"{request_id}: Image Gen Handler logic flaw - response object 'resp' was not assigned!")
        resp_data, status_code_val = _create_openai_error_response("Internal error: Handler did not produce response.",
                                                                   err_type="server_error", status_code=500)
        resp = Response(json.dumps(resp_data), status=status_code_val, mimetype='application/json')
        try:
            if db_to_use: add_interaction(db_to_use, session_id=session_id_for_log, mode="image_gen",
                                          input_type='error',
                                          user_input=f"ImgGen NoResp. Req: {request_data_snippet_for_log}",
                                          llm_response="Critical: No resp obj created."); db_to_use.commit()
        except:
            pass
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

## Fine tuning API Call handler

@app.route("/v1/fine_tuning/jobs", methods=["POST"])
async def handle_create_pseudo_fine_tuning_job():
    start_req_time = time.monotonic()
    request_id = f"req-pseudo-ft-job-{uuid.uuid4()}"  # Unique ID for this request
    logger.info(f"🚀 {request_id}: Received POST /v1/fine_tuning/jobs (Pseudo Fine-Tune via Data Ingestion)")

    db: Session = g.db  # Get DB session from Flask's g
    final_status_code: int = 500
    resp: Optional[Response] = None

    # --- Initialize session_id_for_log and try to get from request ---
    session_id_for_log: str = f"pseudo_ft_req_{request_id}"  # Default session ID
    raw_request_data: Optional[Dict[str, Any]] = None
    # ---

    try:
        raw_request_data = await request.get_json()  # Assumes route is async
        if not raw_request_data:
            raise ValueError("Empty JSON payload received.")

        # Attempt to get session_id from the request if provided by client
        session_id_from_request = raw_request_data.get("session_id")
        if session_id_from_request and isinstance(session_id_from_request, str):
            session_id_for_log = session_id_from_request

        # Set for AIChat instance if it were used directly in this specific stub
        # if ai_chat: ai_chat.current_session_id = session_id_for_log

        training_file_id = raw_request_data.get("training_file")
        model_requested = raw_request_data.get("model")  # e.g., "gpt-3.5-turbo" - we log this

        if not training_file_id:
            raise ValueError("'training_file' ID (referring to an ingested data batch) is required.")
        if not model_requested:
            raise ValueError("'model' (base model name to associate with this data) is required.")

        logger.info(
            f"{request_id}: Pseudo fine-tuning job requested. Session: {session_id_for_log}, Training File ID: '{training_file_id}', Base Model: '{model_requested}'.")
        logger.info(
            f"{request_id}: Data associated with '{training_file_id}' is intended to be processed by the system's self-reflection loop.")

        # Create a job ID and mimic OpenAI's FineTuningJob object
        job_id = f"ftjob-reflect-{str(uuid.uuid4())[:12]}"  # Shorter UUID part for job ID
        current_timestamp = int(time.time())

        # This is the response body we'll build
        response_body = {
            "id": job_id,
            "object": "fine_tuning.job",
            "model": model_requested,
            "created_at": current_timestamp,
            "finished_at": None,
            "fine_tuned_model": f"custom-model-via-reflection:{current_timestamp}",
            "organization_id": "org-placeholder",  # Placeholder
            "result_files": [],
            "status": "pending_reflection_processing",
            "validation_file": raw_request_data.get("validation_file"),  # Echo if provided
            "training_file": training_file_id,
            "hyperparameters": raw_request_data.get("hyperparameters", {"n_epochs": "continuous_self_reflection"}),
            "trained_tokens": None,  # Not applicable
            "message": "Data from training_file has been noted. The system will learn from this data via its ongoing self-reflection and knowledge update processes rather than traditional fine-tuning."
        }

        # Log this "pseudo fine-tuning job" creation to your interactions database
        # Now session_id_for_log is defined
        await asyncio.to_thread(add_interaction, db,
                                session_id=session_id_for_log,
                                mode="fine_tune_stub",
                                input_type="job_creation_request",
                                user_input=f"Pseudo FT Job Created: file_id={training_file_id}, base_model={model_requested}",
                                llm_response=json.dumps(
                                    {"job_id": job_id, "status": response_body.get("status", "unknown_status")}),
                                classification="pseudo_fine_tune_job_logged"
                                )
        await asyncio.to_thread(db.commit)

        final_status_code = 200  # HTTP 200 OK for accepted request
        resp = Response(json.dumps(response_body), status=final_status_code, mimetype='application/json')

    except ValueError as ve:
        logger.warning(f"{request_id}: Invalid pseudo fine-tuning request: {ve}")
        resp_data, final_status_code = _create_openai_error_response(str(ve), err_type="invalid_request_error",
                                                                     status_code=400)
        resp = Response(json.dumps(resp_data), status=final_status_code, mimetype='application/json')
    except Exception as e:
        logger.exception(f"{request_id}: 🔥🔥 Error creating pseudo fine-tuning job:")
        resp_data, final_status_code = _create_openai_error_response(
            f"Server error during pseudo fine-tuning job creation: {str(e)}", err_type="server_error", status_code=500)
        resp = Response(json.dumps(resp_data), status=final_status_code, mimetype='application/json')
        if db: await asyncio.to_thread(db.rollback)  # Rollback if error after DB interaction started
    finally:
        duration_req = (time.monotonic() - start_req_time) * 1000
        logger.info(
            f"🏁 /v1/fine_tuning/jobs POST request {request_id} handled in {duration_req:.2f} ms. Status: {final_status_code}")
        # DB session g.db is closed by @app.teardown_request

    if resp is None:  # Should not happen if try-except is comprehensive
        logger.error(f"{request_id}: Pseudo FT Job handler logic flaw - response object 'resp' was not assigned!")
        resp_data, final_status_code = _create_openai_error_response(
            "Internal error: Handler failed to produce response.", status_code=500)
        resp = Response(json.dumps(resp_data), status=final_status_code, mimetype='application/json')

    return resp


# b) Stubs for other /v1/fine_tuning/jobs/* endpoints
# These can reuse the logic from response #61, just ensure the message is consistent.
# (GET /jobs, GET /jobs/{id}, POST /jobs/{id}/cancel, GET /jobs/{id}/events)
# Example for GET /v1/fine_tuning/jobs:

@app.route("/v1/fine_tuning/jobs", methods=["GET"])
async def handle_list_pseudo_fine_tuning_jobs():
    request_id = f"req-pseudo-ft-list-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received GET /v1/fine_tuning/jobs (Pseudo Fine-Tune Info)")
    # In a real scenario, you might list your "ingestion job" records here.
    response_message = (
        "This system adapts via continuous self-reflection on ingested data and interactions, "
        "rather than discrete fine-tuning jobs. 'Jobs' here represent batches of data ingested for this learning process."
    )
    response_body = {
        "object": "list",
        "data": [
            # Example of what a "job" entry could represent
            # {
            #     "id": "ftjob-reflect-example123",
            #     "object": "fine_tuning.job",
            #     "model": "gpt-3.5-turbo", # Base model mentioned at ingestion
            #     "created_at": int(time.time()) - 86400,
            #     "fine_tuned_model": "custom-model-via-reflection:inprogress",
            #     "status": "processing_reflection_queue",
            #     "message": "Data batch 'file-ingest-job-abc' is being processed by self-reflection."
            # }
        ],
        "has_more": False,
        "message": response_message
    }
    return jsonify(response_body), 200

@app.route("/v1/fine_tuning/jobs/<string:fine_tuning_job_id>", methods=["GET"])
def handle_retrieve_fine_tuning_job(fine_tuning_job_id: str):
    request_id = f"req-ft-retrieve-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received GET /v1/fine_tuning/jobs/{fine_tuning_job_id} (Placeholder)")

    if fine_tuning_job_id == "continuous_learning_main_process":
        response_message = (
            f"Details for '{fine_tuning_job_id}': This represents the system's ongoing adaptation "
            "through self-reflection, background analysis, and knowledge base updates. "
            "It does not have traditional job parameters like epochs or specific data files."
        )
        response_body = {
            "id": fine_tuning_job_id,
            "object": "fine_tuning.job",  # Or custom
            "model": "adaptive_system",
            "created_at": int(time.time()),  # Could be app start time
            "status": "active_and_ongoing",
            "message": response_message,
            "hyperparameters": {
                "learning_mechanisms": ["self_reflection", "rag_vector_learning", "background_generate_tot"]
            }
        }
        logger.info(f"{request_id}: Responding with placeholder details for continuous learning process.")
        return jsonify(response_body), 200
    else:
        response_message = (
            f"Fine-tuning job ID '{fine_tuning_job_id}' does not correspond to a traditional fine-tuning job. "
            "This system adapts via continuous self-reflection and knowledge base updates, not discrete fine-tuning jobs."
        )
        # OpenAI usually returns 404 for non-existent job IDs
        resp_data, _ = _create_openai_error_response(
            message=response_message,
            err_type="invalid_request_error",
            code="fine_tuning_job_not_found"
        )
        logger.info(f"{request_id}: Job ID '{fine_tuning_job_id}' not found as a traditional job.")
        return jsonify(resp_data), 404


@app.route("/v1/fine_tuning/jobs/<string:fine_tuning_job_id>/cancel", methods=["POST"])
def handle_cancel_fine_tuning_job(fine_tuning_job_id: str):
    request_id = f"req-ft-cancel-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received POST /v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel (Placeholder)")

    response_message = (
        f"The learning process '{fine_tuning_job_id}' in this system represents continuous adaptation "
        "(self-reflection, background analysis, knowledge updates) and cannot be 'canceled' in the traditional sense of a discrete fine-tuning job. "
        "These processes are integral to the system's operation."
    )

    # If the ID matches the conceptual one, provide the explanation. Otherwise, 404.
    if fine_tuning_job_id == "continuous_learning_main_process":
        response_body = {
            "id": fine_tuning_job_id,
            "object": "fine_tuning.job",  # Or custom
            "status": "cancellation_not_applicable",  # Custom status
            "message": response_message
        }
        logger.info(f"{request_id}: Responding that continuous learning process cannot be canceled.")
        return jsonify(response_body), 200
    else:
        resp_data, _ = _create_openai_error_response(
            message=f"Fine-tuning job ID '{fine_tuning_job_id}' not found or not applicable for cancellation.",
            err_type="invalid_request_error",
            code="fine_tuning_job_not_found"
        )
        logger.info(f"{request_id}: Job ID '{fine_tuning_job_id}' not found for cancellation.")
        return jsonify(resp_data), 404


@app.route("/v1/fine_tuning/jobs/<string:fine_tuning_job_id>/events", methods=["GET"])
def handle_list_fine_tuning_job_events(fine_tuning_job_id: str):
    request_id = f"req-ft-events-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received GET /v1/fine_tuning/jobs/{fine_tuning_job_id}/events (Placeholder)")

    # Parameters for pagination (OpenAI standard), though we won't use them for much here
    # after_param = request.args.get("after")
    # limit_param = request.args.get("limit", type=int, default=20)

    response_message = (
        f"No discrete events available for '{fine_tuning_job_id}'. This system learns continuously "
        "through self-reflection on interactions, background analysis, and knowledge base updates. "
        "Progress and activities are logged internally."
    )

    if fine_tuning_job_id == "continuous_learning_main_process":
        response_body = {
            "object": "list",
            "data": [
                {
                    "object": "fine_tuning.job.event",
                    "id": f"event_info_{int(time.time())}",
                    "created_at": int(time.time()),
                    "level": "info",
                    "message": response_message,
                    "data": {
                        "step": None,
                        "metrics": {"self_reflection_cycles": "ongoing", "rag_updates": "continuous"}
                    },
                    "type": "message"
                }
            ],
            "has_more": False
        }
        logger.info(f"{request_id}: Responding with placeholder event for continuous learning.")
        return jsonify(response_body), 200
    else:
        # For unknown job IDs, return an empty list of events as per OpenAI spec for non-existent jobs
        # or a 404 if the job itself is considered not found.
        # Let's return 404 consistent with retrieve.
        resp_data, _ = _create_openai_error_response(
            message=f"Fine-tuning job ID '{fine_tuning_job_id}' not found.",
            err_type="invalid_request_error",
            code="fine_tuning_job_not_found"
        )
        logger.info(f"{request_id}: Job ID '{fine_tuning_job_id}' not found for events.")
        return jsonify(resp_data), 404


##v1/files openAI expected to be?
@app.route("/v1/files", methods=["POST"])
async def handle_upload_and_ingest_file():
    start_req_time = time.monotonic()  # For overall request timing
    request_id = f"req-file-ingest-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received POST /v1/files (Data Ingestion for Reflection)")

    db: Session = g.db
    final_status_code: int = 500
    response_body: Dict[str, Any] = {}  # Initialize for consistent return type
    temp_file_path: Optional[str] = None  # <<< Initialize temp_file_path to None HERE

    # Ensure base temporary directory for ingestions exists
    # FILE_INGESTION_TEMP_DIR should be imported from config.py
    # Ensure SCRIPT_DIR is defined if FILE_INGESTION_TEMP_DIR uses it
    try:
        # This creates the base directory if it doesn't exist.
        # Individual files will get unique names within it.
        await asyncio.to_thread(os.makedirs, FILE_INGESTION_TEMP_DIR, exist_ok=True)
    except Exception as e_mkdir:
        logger.error(
            f"{request_id}: Critical error creating base ingestion directory '{FILE_INGESTION_TEMP_DIR}': {e_mkdir}")
        response_body, final_status_code = _create_openai_error_response(
            f"Server setup error: Could not create temporary directory for file ingestion.",
            err_type="server_error", status_code=500
        )
        return Response(json.dumps(response_body), status=final_status_code, mimetype='application/json')

    try:
        if 'file' not in request.files:
            raise ValueError("No file part in the request. Please upload a file using the 'file' field.")

        file_storage = request.files['file']
        purpose = request.form.get('purpose')

        if not file_storage or not file_storage.filename:  # Check if filename is not None or empty
            raise ValueError("No file selected or file has no name.")
        if purpose != 'fine-tune':  # OpenAI API requires 'fine-tune' for files used in fine-tuning jobs
            raise ValueError(
                "Invalid 'purpose'. Must be 'fine-tune' for data ingestion to be used by the reflection system.")

        filename = secure_filename(file_storage.filename)
        # temp_file_path is assigned here, after initial checks
        temp_file_path = os.path.join(FILE_INGESTION_TEMP_DIR, f"{request_id}-{filename}")
        await asyncio.to_thread(file_storage.save, temp_file_path)
        logger.info(f"{request_id}: File '{filename}' saved temporarily to '{temp_file_path}'. Purpose: {purpose}")

        file_ext = os.path.splitext(filename)[1].lower()
        ingested_interactions_count = 0
        processed_rows_or_lines = 0

        common_ingest_session_id = f"ingest_batch_{request_id}"

        if file_ext == ".jsonl":
            with open(temp_file_path, 'r', encoding='utf-8') as f_jsonl:
                for line_num, line in enumerate(f_jsonl):
                    processed_rows_or_lines += 1
                    if not line.strip(): continue  # Skip empty lines
                    try:
                        data_entry = json.loads(line.strip())
                        user_input_content, assistant_response_content = None, None
                        extracted_successfully = False

                        messages = data_entry.get("messages")
                        if isinstance(messages, list) and len(messages) >= 1:
                            first_user_msg = next((m.get("content") for m in messages if m.get("role") == "user"), None)
                            first_asst_msg = next((m.get("content") for m in messages if m.get("role") == "assistant"),
                                                  None)
                            if first_user_msg: user_input_content = first_user_msg
                            if first_asst_msg: assistant_response_content = first_asst_msg
                            if user_input_content or assistant_response_content: extracted_successfully = True
                        elif "prompt" in data_entry and "completion" in data_entry:
                            user_input_content = data_entry.get("prompt")
                            assistant_response_content = data_entry.get("completion")
                            extracted_successfully = True
                        elif "user_input" in data_entry and "llm_response" in data_entry:
                            user_input_content = data_entry.get("user_input")
                            assistant_response_content = data_entry.get("llm_response")
                            extracted_successfully = True
                        elif "text" in data_entry:  # Fallback for generic text entries
                            user_input_content = data_entry.get("text")
                            assistant_response_content = "[Ingested as single text entry]"
                            extracted_successfully = True

                        if extracted_successfully and user_input_content is not None:
                            await asyncio.to_thread(add_interaction, db,
                                                    session_id=data_entry.get("session_id_override",
                                                                              common_ingest_session_id),
                                                    mode=data_entry.get("mode_override", "chat_ingested"),
                                                    input_type=data_entry.get("input_type_override",
                                                                              "text_ingested_jsonl"),
                                                    user_input=str(user_input_content),
                                                    llm_response=str(
                                                        assistant_response_content) if assistant_response_content is not None else None,
                                                    reflection_completed=False)
                            ingested_interactions_count += 1
                        else:  # Fallback: Dump raw JSON object
                            logger.warning(
                                f"{request_id}: JSONL line {line_num + 1} did not match expected structures. Ingesting raw.")
                            await asyncio.to_thread(add_interaction, db,
                                                    session_id=data_entry.get("session_id_override",
                                                                              common_ingest_session_id),
                                                    mode=data_entry.get("mode_override", "data_ingestion"),
                                                    input_type="raw_ingested_json_line",
                                                    user_input=f"[Raw JSON from line {line_num + 1} of {filename}]",
                                                    llm_response=json.dumps(data_entry), reflection_completed=False)
                            ingested_interactions_count += 1
                    except json.JSONDecodeError:
                        logger.warning(f"{request_id}: Skipped invalid JSON line {line_num + 1} in '{filename}'.")

        elif file_ext in [".csv", ".tsv"]:
            delimiter = '\t' if file_ext == ".tsv" else ','
            df = await asyncio.to_thread(pd.read_csv, temp_file_path, delimiter=delimiter)  # type: ignore
            processed_rows_or_lines = len(df)
            for index, row in df.iterrows():  # type: ignore
                user_input_content = row.get("prompt") or row.get("user_input") or row.get("text")
                assistant_response_content = row.get("completion") or row.get("llm_response")

                if user_input_content is not None:
                    await asyncio.to_thread(add_interaction, db,
                                            session_id=row.get("session_id_override", common_ingest_session_id),
                                            mode=row.get("mode_override", "chat_ingested"),
                                            input_type=row.get("input_type_override", f"text_ingested_{file_ext[1:]}"),
                                            user_input=str(user_input_content),
                                            llm_response=str(
                                                assistant_response_content) if assistant_response_content is not None else None,
                                            reflection_completed=False)
                    ingested_interactions_count += 1
                else:  # Fallback for CSV/TSV row
                    logger.warning(
                        f"{request_id}: {file_ext.upper()} row {index + 1} missing primary text field. Ingesting raw.")
                    raw_row_dict = {k: str(v) for k, v in row.to_dict().items() if pd.notna(v)}  # type: ignore
                    await asyncio.to_thread(add_interaction, db,
                                            session_id=row.get("session_id_override", common_ingest_session_id),
                                            mode=row.get("mode_override", "data_ingestion"),
                                            input_type=f"raw_ingested_tabular_row",
                                            user_input=f"[Raw {file_ext.upper()} row {index + 1} from {filename}]",
                                            llm_response=json.dumps(raw_row_dict), reflection_completed=False)
                    ingested_interactions_count += 1

        elif file_ext == ".parquet":
            df = await asyncio.to_thread(pd.read_parquet, temp_file_path)  # type: ignore
            processed_rows_or_lines = len(df)
            for index, row in df.iterrows():  # type: ignore
                user_input_content = row.get("prompt") or row.get("user_input") or row.get("text")
                assistant_response_content = row.get("completion") or row.get("llm_response")

                if user_input_content is not None:
                    await asyncio.to_thread(add_interaction, db,
                                            session_id=row.get("session_id_override", common_ingest_session_id),
                                            mode=row.get("mode_override", "chat_ingested"),
                                            input_type=row.get("input_type_override", "text_ingested_parquet"),
                                            user_input=str(user_input_content),
                                            llm_response=str(
                                                assistant_response_content) if assistant_response_content is not None else None,
                                            reflection_completed=False)
                    ingested_interactions_count += 1
                else:  # Fallback for Parquet row
                    logger.warning(f"{request_id}: Parquet row {index + 1} missing primary text field. Ingesting raw.")
                    raw_row_dict = {k: str(v) for k, v in row.to_dict().items() if pd.notna(v)}  # type: ignore
                    await asyncio.to_thread(add_interaction, db,
                                            session_id=row.get("session_id_override", common_ingest_session_id),
                                            mode=row.get("mode_override", "data_ingestion"),
                                            input_type="raw_ingested_tabular_row",
                                            user_input=f"[Raw Parquet row {index + 1} from {filename}]",
                                            llm_response=json.dumps(raw_row_dict), reflection_completed=False)
                    ingested_interactions_count += 1
        else:
            raise ValueError(f"Unsupported file type: '{file_ext}'. Please use JSONL, CSV, TSV, or Parquet.")

        await asyncio.to_thread(db.commit)
        ingestion_summary = f"Ingested {ingested_interactions_count}/{processed_rows_or_lines} entries from '{filename}' into interactions database for self-reflection."
        logger.info(f"{request_id}: {ingestion_summary}")

        response_body = {
            "id": f"file-ingest-job-{request_id}",
            "object": "file",
            "bytes": await asyncio.to_thread(os.path.getsize,
                                             temp_file_path) if temp_file_path and await asyncio.to_thread(
                os.path.exists, temp_file_path) else 0,
            "created_at": int(time.time()), "filename": filename,
            "purpose": "fine-tune-data-ingested-for-reflection",
            "status": "processed", "status_details": ingestion_summary
        }
        final_status_code = 200

    except ValueError as ve:
        logger.warning(f"{request_id}: Invalid file upload request: {ve}")
        response_body, final_status_code = _create_openai_error_response(str(ve), err_type="invalid_request_error",
                                                                         status_code=400)
    except ImportError as ie:  # Catch pandas import error specifically
        logger.error(
            f"{request_id}: Missing library for file processing: {ie}. Please ensure pandas and pyarrow/fastparquet are installed.")
        response_body, final_status_code = _create_openai_error_response(f"Server missing library for file type: {ie}",
                                                                         err_type="server_error", status_code=501)
    except Exception as e:
        logger.exception(f"{request_id}: 🔥🔥 Error processing uploaded file for ingestion:")
        response_body, final_status_code = _create_openai_error_response(
            f"Server error during file ingestion: {str(e)}", err_type="server_error", status_code=500)
        if db: await asyncio.to_thread(db.rollback)
    finally:
        # temp_file_path is defined at the start of the function (initialized to None)
        if temp_file_path and await asyncio.to_thread(os.path.exists, temp_file_path):
            try:
                await asyncio.to_thread(os.remove, temp_file_path)
                logger.info(f"{request_id}: Deleted temporary ingested file: {temp_file_path}")
            except Exception as e_del:
                logger.error(f"{request_id}: Failed to delete temporary ingested file '{temp_file_path}': {e_del}")

        duration_req = (time.monotonic() - start_req_time) * 1000
        logger.info(
            f"🏁 /v1/files POST request {request_id} handled in {duration_req:.2f} ms. Status: {final_status_code}")

    return Response(json.dumps(response_body), status=final_status_code, mimetype='application/json')


@app.route("/v1/files", methods=["GET"])
async def handle_list_files_ingestion_stub():
    request_id = f"req-file-list-ingest-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received GET /v1/files (Ingestion Info Placeholder)")
    # In a real scenario, you might query a table of ingestion jobs/batches.
    # For this stub, return an informative message.
    message = (
        "This endpoint normally lists uploaded files. In this system, files are ingested directly into the "
        "interaction database for self-reflection. There isn't a list of 'files' in the traditional OpenAI sense. "
        "Consider this a log of data ingestion batches if it were implemented."
    )
    response_body = {
        "object": "list",
        "data": [
            # Example of what an entry could look like if you tracked ingestion batches
            # {
            #     "id": "file-ingest-job-example123",
            #     "object": "file",
            #     "bytes": 10240, # Placeholder
            #     "created_at": int(time.time()) - 3600,
            #     "filename": "example_training_data.jsonl",
            #     "purpose": "fine-tune-data-ingested-for-reflection",
            #     "status": "processed",
            #     "status_details": "100 interactions ingested."
            # }
        ],
        "message": message
    }
    return jsonify(response_body), 200


@app.route("/v1/files/<string:file_id>", methods=["GET"])
async def handle_retrieve_file_ingestion_stub(file_id: str):
    request_id = f"req-file-retrieve-ingest-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received GET /v1/files/{file_id} (Ingestion Info Placeholder)")
    # Here, file_id would correspond to the ID returned by the POST /v1/files (e.g., "file-ingest-job-...")
    # You could query a database of ingestion jobs. For now, a stub:
    message = (
        f"Retrieval for 'file' ID '{file_id}' (representing an ingestion batch for self-reflection) is not fully implemented. "
        "This system ingests data directly into its interaction database. "
        "If this ID corresponds to a past ingestion, its data has been processed for self-reflection."
    )
    if file_id.startswith("file-ingest-job-"):
        response_body = {
            "id": file_id, "object": "file", "bytes": 0,  # Placeholder
            "created_at": int(time.time()) - 7200,  # Placeholder
            "filename": "ingested_dataset_placeholder.jsonl", "purpose": "fine-tune-data-ingested-for-reflection",
            "status": "processed", "status_details": message
        }
        return jsonify(response_body), 200
    else:
        resp_data, _ = _create_openai_error_response(
            f"Ingestion job ID '{file_id}' not found or not in expected format.", code="not_found",
            err_type="invalid_request_error")
        return jsonify(resp_data), 404


@app.route("/v1/files/<string:file_id>", methods=["DELETE"])
async def handle_delete_file_ingestion_stub(file_id: str):
    request_id = f"req-file-delete-ingest-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received DELETE /v1/files/{file_id} (Ingestion Info Placeholder)")
    # Deleting an "ingestion job" in this context might mean marking its interactions as "ignored_ingestion"
    # or some other status, rather than deleting them. For now, a stub.
    message = (
        f"Deletion of 'file' ID '{file_id}' (representing an ingestion batch) is not supported in a way that removes "
        "data from the interaction database. Data ingested for self-reflection is integrated into the system's learning process."
    )
    response_body = {"id": file_id, "object": "file.deleted_stub", "deleted": True, "message": message}
    return jsonify(response_body), 200


@app.route("/v1/files/<string:file_id>/content", methods=["GET"])
async def handle_retrieve_file_content_ingestion_stub(file_id: str):
    request_id = f"req-file-content-ingest-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received GET /v1/files/{file_id}/content (Ingestion Info Placeholder)")
    message = (
        f"Retrieving original content for 'file' ID '{file_id}' (ingestion batch) is not supported. "
        "Data is processed and integrated into the interaction database."
    )
    resp_data, _ = _create_openai_error_response(message, code="content_not_available",
                                                 err_type="invalid_request_error")
    return jsonify(resp_data), 404  # OpenAI returns 404 if content not retrievable for a file


@app.route("/v1/files/<string:file_id>", methods=["GET"])
def handle_retrieve_file_metadata_stub(file_id: str):
    request_id = f"req-file-retrieve-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received GET /v1/files/{file_id} (Placeholder/DB Lookup)")
    db: Session = g.db

    found_file_record: Optional[FileIndex] = None
    search_method = "unknown"

    # Try to interpret file_id as either a database integer ID or a file_path
    try:
        numeric_file_id = int(file_id)
        found_file_record = db.query(FileIndex).filter(FileIndex.id == numeric_file_id).first()
        search_method = f"database ID ({numeric_file_id})"
    except ValueError:
        # Not an integer, try as a file path (or part of it)
        # For a direct path match, it would need to be the full path.
        # For simplicity, let's assume if it's not an int, it might be a filename client has.
        # A more robust search might use `file_id` in a LIKE query for file_name or file_path.
        # For this stub, we'll primarily rely on ID or exact path match if we were to implement fully.
        # Let's try to see if it's a path we have indexed.
        # This requires `file_id` to be the actual path string.
        found_file_record = db.query(FileIndex).filter(FileIndex.file_path == file_id).first()
        search_method = f"exact file path ('{file_id}')"
        if not found_file_record:  # Fallback: search by filename if it's not a full path
            found_file_record = db.query(FileIndex).filter(FileIndex.file_name == file_id).first()
            if found_file_record: search_method = f"file name ('{file_id}')"

    if found_file_record:
        response_body = {
            "id": str(found_file_record.id),  # Use DB ID as the file_id
            "object": "file",
            "bytes": found_file_record.size_bytes or 0,
            "created_at": int(found_file_record.last_indexed_db.timestamp()),  # Or file creation time if stored
            "filename": found_file_record.file_name,
            "purpose": "locally_indexed_for_rag",  # Custom purpose
            "status": found_file_record.index_status,
            "status_details": f"Locally indexed file. Path: {found_file_record.file_path}. Last OS Mod: {found_file_record.last_modified_os}"
        }
        logger.info(
            f"{request_id}: Found and returning metadata for indexed file (ID: {found_file_record.id}) based on query for '{file_id}' using {search_method}.")
        return jsonify(response_body), 200
    else:
        response_message = (
            f"File ID or path '{file_id}' does not correspond to a known locally indexed file in the system's database. "
            "Zephy/Adelaide works with files indexed from the local filesystem."
        )
        resp_data, _ = _create_openai_error_response(
            message=response_message,
            err_type="invalid_request_error",
            code="file_not_found"
        )
        logger.info(f"{request_id}: File '{file_id}' not found in local index using {search_method}.")
        return jsonify(resp_data), 404


@app.route("/v1/files/<string:file_id>", methods=["DELETE"])
def handle_delete_file_stub(file_id: str):
    request_id = f"req-file-delete-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received DELETE /v1/files/{file_id} (Placeholder)")

    response_message = (
        f"File deletion via this API (for ID/path '{file_id}') is not supported. Zephy/Adelaide indexes "
        "local files. To remove a file from being accessed or indexed, please delete or move it "
        "from the local filesystem using your operating system's file manager. "
        "The file indexer will update its records on a subsequent scan."
    )

    # OpenAI typically returns a deletion confirmation object.
    response_body = {
        "id": file_id,  # Echo back the ID
        "object": "file.deleted_stub",  # Custom object type
        "deleted": True,  # Indicate the API call was processed
        "message": response_message
    }
    logger.info(f"{request_id}: Responding with placeholder for file deletion, explaining local file management.")
    return jsonify(response_body), 200


@app.route("/v1/files/<string:file_id>/content", methods=["GET"])
def handle_retrieve_file_content_stub(file_id: str):
    request_id = f"req-file-content-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received GET /v1/files/{file_id}/content (Placeholder/DB Lookup)")
    db: Session = g.db

    found_file_record: Optional[FileIndex] = None
    search_method = "unknown"
    try:
        numeric_file_id = int(file_id)
        found_file_record = db.query(FileIndex).filter(FileIndex.id == numeric_file_id).first()
        search_method = f"database ID ({numeric_file_id})"
    except ValueError:
        found_file_record = db.query(FileIndex).filter(FileIndex.file_path == file_id).first()
        search_method = f"exact file path ('{file_id}')"
        if not found_file_record:
            found_file_record = db.query(FileIndex).filter(FileIndex.file_name == file_id).first()
            if found_file_record: search_method = f"file name ('{file_id}')"

    if found_file_record and found_file_record.indexed_content:
        content = found_file_record.indexed_content
        # Determine a basic MIME type, default to text/plain
        mime_type = found_file_record.mime_type if found_file_record.mime_type else "text/plain"
        if "unknown" in mime_type.lower(): mime_type = "text/plain"  # Fallback for "UnknownMIME"

        logger.info(
            f"{request_id}: Found and returning content for indexed file (ID: {found_file_record.id}, Path: {found_file_record.file_path}) with MIME type {mime_type}.")
        # OpenAI returns raw content, not JSON
        return Response(content, status=200, mimetype=mime_type)
    elif found_file_record and not found_file_record.indexed_content:
        response_message = (
            f"File '{file_id}' (Path: {found_file_record.file_path}) was found in the index, but its content has not been "
            "extracted or stored. Status: {found_file_record.index_status}."
        )
        resp_data, _ = _create_openai_error_response(
            message=response_message, err_type="invalid_request_error", code="file_content_not_available")
        logger.info(f"{request_id}: File '{file_id}' found but no indexed content available.")
        return jsonify(resp_data), 404  # Or 200 with an error message in content if API expects that
    else:
        response_message = (
            f"File ID or path '{file_id}' does not correspond to a known locally indexed file with available content. "
            "Zephy/Adelaide works with files indexed from the local filesystem."
        )
        resp_data, _ = _create_openai_error_response(
            message=response_message, err_type="invalid_request_error", code="file_not_found")
        logger.info(f"{request_id}: File '{file_id}' or its content not found in local index using {search_method}.")
        return jsonify(resp_data), 404



# --- Assistants API Stubs ---
@app.route("/v1/assistants", methods=["POST"])
def create_assistant_stub():
    request_id = f"req-assistant-create-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received POST /v1/assistants (Placeholder)")
    # You could log request.json if you want to see what clients are trying to create
    return _create_personal_assistant_stub_response("/v1/assistants", "POST",
        custom_status="creation_not_applicable")

@app.route("/v1/assistants", methods=["GET"])
def list_assistants_stub():
    request_id = f"req-assistant-list-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received GET /v1/assistants (Placeholder)")
    # OpenAI list endpoints return an object with a "data" array.
    response_message = (
        "This system operates as a single, integrated personal assistant. "
        "There are no multiple, separately manageable 'assistant' objects to list. "
        "Its capabilities are defined by its core configuration and ongoing learning processes like self-reflection."
    )
    response_body = {
        "object": "list",
        "data": [], # Empty list as there are no discrete assistants in this model
        "message": response_message
    }
    return jsonify(response_body), 200

@app.route("/v1/assistants/<string:assistant_id>", methods=["GET"])
def retrieve_assistant_stub(assistant_id: str):
    request_id = f"req-assistant-retrieve-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received GET /v1/assistants/{assistant_id} (Placeholder)")
    return _create_personal_assistant_stub_response(f"/v1/assistants/{assistant_id}", "GET", resource_id=assistant_id)

@app.route("/v1/assistants/<string:assistant_id>", methods=["POST"])
def modify_assistant_stub(assistant_id: str):
    request_id = f"req-assistant-modify-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received POST /v1/assistants/{assistant_id} (Placeholder)")
    return _create_personal_assistant_stub_response(f"/v1/assistants/{assistant_id}", "POST", resource_id=assistant_id,
        custom_status="modification_not_applicable")

@app.route("/v1/assistants/<string:assistant_id>", methods=["DELETE"])
def delete_assistant_stub(assistant_id: str):
    request_id = f"req-assistant-delete-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received DELETE /v1/assistants/{assistant_id} (Placeholder)")
    return _create_personal_assistant_stub_response(f"/v1/assistants/{assistant_id}", "DELETE", resource_id=assistant_id,
        custom_status="deletion_not_applicable")

# --- Threads API Stubs ---
@app.route("/v1/threads", methods=["POST"])
def create_thread_stub():
    request_id = f"req-thread-create-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received POST /v1/threads (Placeholder)")
    return _create_personal_assistant_stub_response("/v1/threads", "POST",
        custom_status="thread_creation_not_applicable_managed_internally")

@app.route("/v1/threads/<string:thread_id>", methods=["GET"])
def retrieve_thread_stub(thread_id: str):
    request_id = f"req-thread-retrieve-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received GET /v1/threads/{thread_id} (Placeholder)")
    return _create_personal_assistant_stub_response(f"/v1/threads/{thread_id}", "GET", resource_id=thread_id)

@app.route("/v1/threads/<string:thread_id>", methods=["POST"])
def modify_thread_stub(thread_id: str):
    request_id = f"req-thread-modify-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received POST /v1/threads/{thread_id} (Placeholder)")
    return _create_personal_assistant_stub_response(f"/v1/threads/{thread_id}", "POST", resource_id=thread_id,
        custom_status="modification_not_applicable")

@app.route("/v1/threads/<string:thread_id>", methods=["DELETE"])
def delete_thread_stub(thread_id: str):
    request_id = f"req-thread-delete-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received DELETE /v1/threads/{thread_id} (Placeholder)")
    return _create_personal_assistant_stub_response(f"/v1/threads/{thread_id}", "DELETE", resource_id=thread_id,
        custom_status="deletion_not_applicable")

# --- Messages API Stubs (within Threads) ---
@app.route("/v1/threads/<string:thread_id>/messages", methods=["POST"])
def create_message_stub(thread_id: str):
    request_id = f"req-message-create-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received POST /v1/threads/{thread_id}/messages (Placeholder)")
    return _create_personal_assistant_stub_response(f"/v1/threads/{thread_id}/messages", "POST", resource_id=thread_id,
        custom_status="message_creation_handled_via_direct_chat")

@app.route("/v1/threads/<string:thread_id>/messages", methods=["GET"])
def list_messages_stub(thread_id: str):
    request_id = f"req-message-list-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received GET /v1/threads/{thread_id}/messages (Placeholder)")
    response_message = (
        f"Listing messages for thread '{thread_id}' via this API is not applicable. "
        "Conversation history is managed internally per session. "
        "Internal database logs store interaction history."
    )
    response_body = {
        "object": "list",
        "data": [],
        "message": response_message
    }
    return jsonify(response_body), 200

# --- Runs API Stubs (within Threads) ---
@app.route("/v1/threads/<string:thread_id>/runs", methods=["POST"])
def create_run_stub(thread_id: str):
    request_id = f"req-run-create-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received POST /v1/threads/{thread_id}/runs (Placeholder)")
    return _create_personal_assistant_stub_response(f"/v1/threads/{thread_id}/runs", "POST", resource_id=thread_id,
        custom_status="run_creation_implicit_in_direct_interaction")

@app.route("/v1/threads/<string:thread_id>/runs/<string:run_id>", methods=["GET"])
def retrieve_run_stub(thread_id: str, run_id: str):
    request_id = f"req-run-retrieve-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id}: Received GET /v1/threads/{thread_id}/runs/{run_id} (Placeholder)")
    return _create_personal_assistant_stub_response(f"/v1/threads/{thread_id}/runs/{run_id}", "GET", resource_id=run_id)


@app.route("/v1/threads/<string:thread_id>/runs/<string:run_id>/cancel", methods=["POST"])
def cancel_run_stub(thread_id: str, run_id: str):
    request_id_log = f"req-run-cancel-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id_log}: Received POST /v1/threads/{thread_id}/runs/{run_id}/cancel (Placeholder)")

    specific_message = (
        "Run cancellation via this API is not directly applicable. This system's agentic tasks, "
        "if initiated (e.g., by 'background_generate' or an internal agent loop), operate asynchronously. "
        "While explicit cancellation of a specific internal 'run' by ID is not exposed here, tasks are "
        "managed by an internal priority system and may be superseded or will naturally complete."
    )
    # Mimic OpenAI's Run object structure loosely for the response
    response_body = {
        "id": run_id,
        "object": "thread.run",
        "thread_id": thread_id,
        "assistant_id": "integrated_personal_assistant",
        "status": "cancellation_not_applicable",  # Custom status
        "started_at": int(time.time()),  # Dummy timestamp
        "expires_at": None,
        "cancelled_at": None,
        "failed_at": None,
        "completed_at": None,
        "last_error": None,
        "model": "N/A (Integrated System)",
        "instructions": "N/A (Integrated System)",
        "tools": [{"type": "internal_agentic_mode"}],
        "file_ids": [],
        "metadata": {
            "message": specific_message,
            "note": "This is a placeholder response indicating a conceptual difference in system design."
        }
    }
    return jsonify(response_body), 200


@app.route("/v1/threads/<string:thread_id>/runs/<string:run_id>/steps", methods=["GET"])
def list_run_steps_stub(thread_id: str, run_id: str):
    request_id_log = f"req-run-steps-stub-{uuid.uuid4()}"
    logger.info(f"🚀 {request_id_log}: Received GET /v1/threads/{thread_id}/runs/{run_id}/steps (Placeholder)")

    specific_message = (
        "This system does not expose discrete 'run steps' in the OpenAI Assistants API format. "
        "Agentic operations and tool use occur as part of an integrated, asynchronous background process "
        "(e.g., within AIChat.background_generate or AmaryllisAgent._run_task_in_background). "
        "Progress or outcomes are reflected in the ongoing conversation, database logs, or direct results "
        "from initiated tasks, rather than through a list of formal 'steps' for a given 'run ID'."
    )

    # OpenAI list endpoints return an object with a "data" array.
    response_body = {
        "object": "list",
        "data": [],  # No discrete steps to list in this manner
        "first_id": None,
        "last_id": None,
        "has_more": False,
        "message": specific_message,  # Custom message field
        "note": "This is a placeholder response indicating a conceptual difference in system design."
    }
    return jsonify(response_body), 200


@app.route("/v1/threads/<string:thread_id>/runs/<string:run_id>/submit_tool_outputs", methods=["POST"])
def submit_tool_outputs_stub(thread_id: str, run_id: str):
    request_id_log = f"req-run-submittools-stub-{uuid.uuid4()}"
    logger.info(
        f"🚀 {request_id_log}: Received POST /v1/threads/{thread_id}/runs/{run_id}/submit_tool_outputs (Placeholder)")
    # request_data = request.json # Could log if needed

    specific_message = (
        "Tool output submission via this API is not applicable. This system's agentic mode "
        "(e.g., AmaryllisAgent) operates autonomously using its integrated tools and internal logic. "
        "It does not require or use an explicit external step for tool output submission in this manner, "
        "as tool execution and result processing are part of its internal asynchronous flow."
    )

    # Mimic OpenAI's Run object structure loosely
    response_body = {
        "id": run_id,
        "object": "thread.run",
        "thread_id": thread_id,
        "assistant_id": "integrated_personal_assistant",
        "status": "tool_submission_not_applicable",  # Custom status
        "started_at": int(time.time()),
        "expires_at": None,
        "cancelled_at": None,
        "failed_at": None,
        "completed_at": None,
        "last_error": None,
        "model": "N/A (Integrated System)",
        "instructions": "N/A (Integrated System)",
        "tools": [{"type": "internal_agentic_mode"}],
        "file_ids": [],
        "metadata": {
            "message": specific_message,
            "note": "This is a placeholder response indicating a conceptual difference in system design."
        }
    }
    return jsonify(response_body), 200






#=============== IPC Server END ===============
#============================ Main Program Startup
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
    logger.error("This script (app.py) is designed to be run with an ASGI/WSGI server like Hypercorn.")
    logger.error("Example: hypercorn app:app --bind 127.0.0.1:11434")
    sys.exit(1)  # Exit because this isn't the intended way to run
else:
    # This block executes when app.py is imported as a module by a server (e.g., Hypercorn).
    logger.info("----------------------------------------------------------------------")
    logger.info(">>> APP.PY: MODULE IMPORTED BY SERVER (Hypercorn worker process) <<<")
    logger.info("----------------------------------------------------------------------")

    # Ensure critical global instances were initialized earlier in the module loading
    # (These are typically defined after config and before this 'else' block)
    if ai_provider is None or ai_chat is None or ai_agent is None:
        logger.critical(
            "APP.PY: 🔥🔥 Core AI components (ai_provider, ai_chat, ai_agent) are NOT INITIALIZED. Application cannot start properly.")
        # This is a fundamental setup error, exiting directly.
        print("APP.PY: CRITICAL FAILURE - Core AI components not initialized. Exiting.", file=sys.stderr, flush=True)
        sys.exit(1)
    else:
        logger.success("APP.PY: ✅ Core AI components appear initialized globally.")

    # --- Initialize Database ---
    # This is a critical step. If it fails, the app should not proceed.
    db_initialized_successfully = False
    try:
        logger.info("APP.PY: >>> CALLING init_db() NOW. This must complete successfully for the application. <<<")
        # init_db() is imported from database.py
        # It's responsible for setting up the engine, SessionLocal, and migrations.
        init_db()
        db_initialized_successfully = True  # If init_db() returns without exception, assume success
        logger.success("APP.PY: ✅ init_db() call completed (reported no critical errors).")
    except Exception as e_init_db_call:
        # init_db() itself should log details of its failure.
        # This catches any exception re-raised by init_db() indicating a fatal setup error.
        logger.critical(f"APP.PY: 🔥🔥 init_db() FAILED CRITICALLY DURING APP STARTUP: {e_init_db_call}")
        logger.exception("APP.PY: Traceback for init_db() failure at app level:")
        print(f"APP.PY: CRITICAL FAILURE IN init_db(): {e_init_db_call}. Cannot continue. Exiting.", file=sys.stderr,
              flush=True)
        sys.exit(1)  # Force exit if database initialization fails

    # If we reach here, db_initialized_successfully must be True,
    # because the except block above would have sys.exit(1).
    # This explicit check is a safeguard.
    if not db_initialized_successfully:
        logger.critical(
            "APP.PY: Sanity check - init_db() did not set success flag or was bypassed. EXITING ABNORMALLY.")
        sys.exit(1)

    # --- Check if SessionLocal from database.py is usable AFTER init_db() ---
    # This is a sanity check to ensure init_db actually configured SessionLocal.
    try:
        from database import SessionLocal as AppSessionLocalCheck  # Re-import to get current state

        if AppSessionLocalCheck is None:
            logger.critical(
                "APP.PY: FATAL - SessionLocal from database.py is STILL NONE after init_db() call! This indicates a severe problem in init_db's internal logic. EXITING.")
            sys.exit(1)
        else:
            logger.info(
                f"APP.PY: SessionLocal from database.py is NOT None after init_db(). Type: {type(AppSessionLocalCheck)}.")
            # Further check if it's bound (it should be if init_db was successful)
            if hasattr(AppSessionLocalCheck, 'kw') and AppSessionLocalCheck.kw.get('bind'):
                logger.success("APP.PY: ✅ SessionLocal appears configured and bound to an engine.")
            else:
                logger.error(
                    "APP.PY: 🔥 SessionLocal exists but may NOT BE BOUND to an engine (kw.bind missing). Startup tasks requiring DB will likely fail. EXITING.")
                sys.exit(1)
    except ImportError:
        logger.critical(
            "APP.PY: FATAL - Could not import SessionLocal from database.py AFTER init_db() for checking. EXITING.")
        sys.exit(1)
    except Exception as e_sl_check:
        logger.critical(f"APP.PY: FATAL - Unexpected error checking SessionLocal: {e_sl_check}. EXITING.")
        sys.exit(1)

    # --- Run Asynchronous Startup Tasks (like Vector Store Initialization) ---
    logger.info("APP.PY: 🚀 Preparing to run asynchronous startup_tasks (e.g., Vector Store initializations)...")
    startup_tasks_completed_successfully = False
    startup_tasks_start_time = time.monotonic()

    try:
        logger.debug("APP.PY: Setting up asyncio event loop for startup_tasks...")
        try:
            # Get an existing loop or create a new one for this context
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop.is_closed():
                logger.warning("APP.PY: Default asyncio event loop was closed. Creating new one for startup_tasks.")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:  # No current event loop on this thread
            logger.info("APP.PY: No current asyncio event loop for startup_tasks. Creating new one.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        logger.info(
            "APP.PY: >>> CALLING loop.run_until_complete(startup_tasks()). This will block until startup_tasks finishes. <<<")
        loop.run_until_complete(startup_tasks())  # startup_tasks() is defined earlier in app.py
        startup_tasks_duration = time.monotonic() - startup_tasks_start_time
        logger.info(
            f"APP.PY: >>> loop.run_until_complete(startup_tasks()) HAS COMPLETED. Duration: {startup_tasks_duration:.2f}s <<<")
        startup_tasks_completed_successfully = True
    except Exception as su_err:
        startup_tasks_duration = time.monotonic() - startup_tasks_start_time
        logger.critical(
            f"APP.PY: 🚨🚨 CRITICAL FAILURE during loop.run_until_complete(startup_tasks()) after {startup_tasks_duration:.2f}s: {su_err} 🚨🚨")
        logger.exception("APP.PY: Startup Tasks Execution Traceback:")
        # If startup_tasks fail (e.g., vector store init), the app might be in a bad state.
        # Deciding to exit here or continue with limited functionality is a design choice.
        # For now, let's exit as these tasks might be critical.
        print(f"APP.PY: CRITICAL FAILURE IN startup_tasks(): {su_err}. Cannot continue. Exiting.", file=sys.stderr,
              flush=True)
        sys.exit(1)

    # --- Start Other Background Services (File Indexer, Self Reflector) ---
    # These are only started if init_db() AND startup_tasks() completed successfully.
    if db_initialized_successfully and startup_tasks_completed_successfully:
        logger.info(
            "APP.PY: ✅ Core initializations (DB, Startup Tasks) successful. Proceeding to start background services...")

        # Ensure start_file_indexer and start_self_reflector are defined in app.py
        # and that ENABLE_FILE_INDEXER, ENABLE_SELF_REFLECTION are from config.py
        if 'ENABLE_FILE_INDEXER' in globals() and ENABLE_FILE_INDEXER:
            logger.info("APP.PY: Starting File Indexer service...")
            if 'start_file_indexer' in globals() and callable(globals()['start_file_indexer']):
                start_file_indexer()  # This function should exist in app.py
            else:
                logger.error("APP.PY: 'start_file_indexer' function not found. File Indexer NOT started.")
        else:
            logger.info("APP.PY: File Indexer is DISABLED by config or variable not found. Not starting.")

        if 'ENABLE_SELF_REFLECTION' in globals() and ENABLE_SELF_REFLECTION:
            logger.info("APP.PY: Starting Self Reflector service...")
            if 'start_self_reflector' in globals() and callable(globals()['start_self_reflector']):
                start_self_reflector()  # This function should exist in app.py
            else:
                logger.error("APP.PY: 'start_self_reflector' function not found. Self Reflector NOT started.")
        else:
            logger.info("APP.PY: Self Reflector is DISABLED by config or variable not found. Not starting.")
    else:
        logger.error(
            "APP.PY: Background services (File Indexer, Self Reflector) NOT started due to failure in DB init or startup_tasks.")
        # Even if we didn't sys.exit above, this state is problematic.
        # It's probably best to ensure exit happened earlier.
        # If somehow execution reaches here with flags false, it's a logic error.
        print("APP.PY: CRITICAL - Reached end of startup with initialization flags false. Exiting.", file=sys.stderr,
              flush=True)
        sys.exit(1)

    logger.info("--------------------------------------------------------------------")
    logger.info("APP.PY: ✅ Zephyrine EngineMain module-level initializations complete.")
    logger.info(f"   Application (app) is now considered ready by this worker process (PID: {os.getpid()}).")
    logger.info("   Waiting for server to route requests...")
    logger.info("--------------------------------------------------------------------")