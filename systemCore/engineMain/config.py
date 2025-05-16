# config.py
import os
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()
logger.info("Attempting to load environment variables from .env file...")

# --- General Settings ---
PROVIDER = os.getenv("PROVIDER", "llama_cpp") # llama_cpp or "ollama" or "fireworks"
MEMORY_SIZE = int(os.getenv("MEMORY_SIZE", 20))
ANSWER_SIZE_WORDS = int(os.getenv("ANSWER_SIZE_WORDS", 16384)) # Target for *quick* answers (token generation? I forgot)
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 16384)) # Default token limit for LLM calls
#DEFAULT_LLM_TEMPERATURE = 0.8
DEFAULT_LLM_TEMPERATURE = float(os.getenv("DEFAULT_LLM_TEMPERATURE", 0.8))
CHUNCK_SIZE = int(os.getenv("CHUNCK_SIZE", 400)) # For URL Chroma store
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200)) # For URL Chroma store
RAG_HISTORY_COUNT = MEMORY_SIZE
RAG_FILE_INDEX_COUNT = int(os.getenv("RAG_FILE_INDEX_COUNT", 10))
DEEP_THOUGHT_RETRY_ATTEMPTS = int(os.getenv("DEEP_THOUGHT_RETRY_ATTEMPTS", 3))
RESPONSE_TIMEOUT_MS = 15000 # Timeout for potential multi-step process
# Similarity threshold for reusing previous ToT results (requires numpy/embeddings)
TOT_SIMILARITY_THRESHOLD = float(os.getenv("TOT_SIMILARITY_THRESHOLD", 0.1))
# Fuzzy search threshold for history RAG (0-100, higher is stricter) - Requires thefuzz
FUZZY_SEARCH_THRESHOLD = int(os.getenv("FUZZY_SEARCH_THRESHOLD", 20))
MIN_RAG_RESULTS = int(os.getenv("MIN_RAG_RESULTS", 1)) # Unused
YOUR_REFLECTION_CHUNK_SIZE = int(os.getenv("YOUR_REFLECTION_CHUNK_SIZE", 450))
YOUR_REFLECTION_CHUNK_OVERLAP = int(os.getenv("YOUR_REFLECTION_CHUNK_OVERLAP", 50))
RAG_URL_COUNT = int(os.getenv("RAG_URL_COUNT", 5)) # <<< ADD THIS LINE (e.g., default to 3)
RAG_CONTEXT_MAX_PERCENTAGE = float(os.getenv("RAG_CONTEXT_MAX_PERCENTAGE", 0.25))

LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT = os.getenv("LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT")
if LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT is not None:
    try:
        LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT = int(LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT)
        logger.info(f"LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT set to: {LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT}")
    except ValueError:
        logger.warning(f"Invalid value for LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT ('{LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT}'). It will be ignored.")
        LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT = None



USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    # Add more diverse and recent agents
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/115.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15',
]
# --- Agent Settings ---
AGENT_MAX_SCRIPT_RETRIES = 3 # Max attempts to generate/fix AppleScript per action

ENABLE_FILE_INDEXER_STR = os.getenv("ENABLE_FILE_INDEXER", "true")
ENABLE_FILE_INDEXER = ENABLE_FILE_INDEXER_STR.lower() in ('true', '1', 't', 'yes', 'y')
logger.info(f"File Indexer Enabled: {ENABLE_FILE_INDEXER}")

# --- Database Settings (SQLite) ---
_config_dir = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB_FILE = "mappedknowledge.db"
SQLITE_DB_PATH = os.path.join(_config_dir, SQLITE_DB_FILE)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.abspath(SQLITE_DB_PATH)}")
logger.info(f"Database URL set to: {DATABASE_URL}")


# --- Model Names (Ensure these exist in Ollama) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:8141") # Default to 8141
MODEL_ROUTER = os.getenv("MODEL_ROUTER", "deepscaler:latest") # Router and Corrector
MODEL_VLM = os.getenv("MODEL_VLM", "gemma3:4b-it-qat") # Vision/General (Needs testing as VLM)
MODEL_LATEX = os.getenv("MODEL_LATEX", "mradermacher/LatexMind-2B-Codec-i1-GGUF:IQ4_XS") # Check exact Ollama name
MODEL_MATH = os.getenv("MODEL_MATH", "qwen2-math:1.5b-instruct-q5_K_M") # Needs Chinese translation
MODEL_CODE = os.getenv("MODEL_CODE", "qwen2.5-coder:3b-instruct-q5_K_M") # Needs Chinese translation
MODEL_GENERAL_FAST = os.getenv("MODEL_GENERAL_FAST", "qwen3:0.6b-q4_K_M") # New fast model
MODEL_TRANSLATOR = os.getenv("MODEL_TRANSLATOR", "hf.co/mradermacher/NanoTranslator-immersive_translate-0.5B-GGUF:Q4_K_M") # Check exact Ollama name
MODEL_DEFAULT_CHAT = MODEL_ROUTER # Use the router/corrector as default

# --- NEW: LLAMA_CPP Settings (Used if PROVIDER="llama_cpp") ---
_engine_main_dir = os.path.dirname(os.path.abspath(__file__)) # Assumes config.py is in engineMain
LLAMA_CPP_GGUF_DIR = os.path.join(_engine_main_dir, "staticmodelpool")
LLAMA_CPP_N_GPU_LAYERS = int(os.getenv("LLAMA_CPP_N_GPU_LAYERS", -1)) # Default: Offload all possible layers
LLAMA_CPP_N_CTX = int(os.getenv("LLAMA_CPP_N_CTX", 4096)) # Context window size
LLAMA_CPP_VERBOSE = os.getenv("LLAMA_CPP_VERBOSE", "False").lower() == "true"
LLAMA_WORKER_TIMEOUT = int(os.getenv("LLAMA_WORKER_TIMEOUT", 300))

# --- Mapping logical roles to GGUF filenames within LLAMA_CPP_GGUF_DIR ---
LLAMA_CPP_MODEL_MAP = {
    "router": os.getenv("LLAMA_CPP_MODEL_ROUTER_FILE", "deepscaler.gguf"), # Adelaide Zephyrine Charlotte Persona
    "vlm": os.getenv("LLAMA_CPP_MODEL_VLM_FILE", "Qwen2.5-VL-7B-Instruct-q4_k_m.gguf"), # Use LatexMind as VLM for now
    "latex": os.getenv("LLAMA_CPP_MODEL_LATEX_FILE", "LatexMind-2B-Codec-i1-GGUF-IQ4_XS.gguf"),
    "math": os.getenv("LLAMA_CPP_MODEL_MATH_FILE", "qwen2-math-1.5b-instruct-q5_K_M.gguf"),
    "code": os.getenv("LLAMA_CPP_MODEL_CODE_FILE", "qwen2.5-coder-3b-instruct-q5_K_M.gguf"),
    "general": os.getenv("LLAMA_CPP_MODEL_GENERAL_FILE", "deepscaler.gguf"), # Use router as general
    "general_fast": os.getenv("LLAMA_CPP_MODEL_GENERAL_FAST_FILE", "Qwen2.5-DirectLowLatency.gguf"),
    "translator": os.getenv("LLAMA_CPP_MODEL_TRANSLATOR_FILE", "NanoTranslator-immersive_translate-0.5B-GGUF-Q4_K_M.gguf"), # Assuming download renamed it
    # --- Embedding Model ---
    "embeddings": os.getenv("LLAMA_CPP_EMBEDDINGS_FILE", "mxbai-embed-large-v1.gguf") # Example name
}
# Define default chat model based on map
MODEL_DEFAULT_CHAT_LLAMA_CPP = "general" # Use the logical name


# --- NEW: Snapshot Configuration ---
ENABLE_DB_SNAPSHOTS = os.getenv("ENABLE_DB_SNAPSHOTS", "true").lower() in ('true', '1', 't', 'yes', 'y')
DB_SNAPSHOT_INTERVAL_MINUTES = int(os.getenv("DB_SNAPSHOT_INTERVAL_MINUTES", 1))
DB_SNAPSHOT_DIR_NAME = "db_snapshots"
# DB_SNAPSHOT_DIR is derived in database.py
DB_SNAPSHOT_RETENTION_COUNT = int(os.getenv("DB_SNAPSHOT_RETENTION_COUNT", 7)) # << SET TO 7 HERE or via .env
DB_SNAPSHOT_FILENAME_PREFIX = "snapshot_"
DB_SNAPSHOT_FILENAME_SUFFIX = ".db.zst"
_file_indexer_module_dir = os.path.dirname(os.path.abspath(__file__)) # If config.py is in the same dir as file_indexer.py
# Or, if config.py is in engineMain and file_indexer.py is also there:
# _file_indexer_module_dir = os.path.dirname(os.path.abspath(__file__))

# Define a subdirectory for Chroma databases relative to the module's location
CHROMA_DB_BASE_PATH = os.path.join(_file_indexer_module_dir, "chroma_vector_stores")

_REFLECTION_VS_PERSIST_DIR = getattr(globals(), 'REFLECTION_INDEX_CHROMA_PERSIST_DIR',
                                   os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_reflection_store_default"))
_REFLECTION_COLLECTION_NAME = getattr(globals(), 'REFLECTION_INDEX_CHROMA_COLLECTION_NAME',
                                      "global_reflections_default_collection")

# Specific persist directory for the global file index
FILE_INDEX_CHROMA_PERSIST_DIR = os.path.join(CHROMA_DB_BASE_PATH, "global_file_index_v1")
FILE_INDEX_CHROMA_COLLECTION_NAME = "global_file_index_collection_v1" # Keep this consistent

# Specific persist directory for the global reflection index (if you also want to make it persistent)
REFLECTION_INDEX_CHROMA_PERSIST_DIR = os.path.join(CHROMA_DB_BASE_PATH, "global_reflection_index_v1")
REFLECTION_INDEX_CHROMA_COLLECTION_NAME = "global_reflection_collection_v1" # Keep this consistent


# --- Placeholder for Stable Diffusion ---
# --- NEW: Imagination Worker (Stable Diffusion FLUX) Settings ---
IMAGE_WORKER_SCRIPT_NAME = "imagination_worker.py" # Name of the worker script

# --- Get base directory for model files ---
# Assumes models are in a subdir of the main engine dir (where config.py is)
# Adjust if your models are elsewhere
_engine_main_dir = os.path.dirname(os.path.abspath(__file__))
IMAGE_GEN_MODEL_DIR = os.getenv("IMAGE_GEN_MODEL_DIR", os.path.join(_engine_main_dir, "staticmodelpool"))
logger.info(f"🖼️ Imagination Model Directory: {IMAGE_GEN_MODEL_DIR}")

# --- Model Filenames (within IMAGE_GEN_MODEL_DIR) ---
IMAGE_GEN_DIFFUSION_MODEL_NAME = os.getenv("IMAGE_GEN_DIFFUSION_MODEL_NAME", "flux1-schnell.gguf")
IMAGE_GEN_CLIP_L_NAME = os.getenv("IMAGE_GEN_CLIP_L_NAME", "flux1-clip_l.gguf")
IMAGE_GEN_T5XXL_NAME = os.getenv("IMAGE_GEN_T5XXL_NAME", "flux1-t5xxl.gguf")
IMAGE_GEN_VAE_NAME = os.getenv("IMAGE_GEN_VAE_NAME", "flux1-ae.gguf")
IMAGE_GEN_WORKER_TIMEOUT = int(os.getenv("IMAGE_GEN_WORKER_TIMEOUT", 500))

# --- stable-diffusion-cpp Library Parameters ---
IMAGE_GEN_DEVICE = os.getenv("IMAGE_GEN_DEVICE", "default") # e.g., 'cpu', 'cuda:0', 'mps', 'default'
IMAGE_GEN_RNG_TYPE = os.getenv("IMAGE_GEN_RNG_TYPE", "std_default") # "std_default" or "cuda"
IMAGE_GEN_N_THREADS = int(os.getenv("IMAGE_GEN_N_THREADS", 0)) # 0 for auto, positive for specific count

# --- Image Generation Defaults (passed to worker via JSON stdin) ---
IMAGE_GEN_DEFAULT_NEGATIVE_PROMPT = os.getenv("IMAGE_GEN_DEFAULT_NEGATIVE_PROMPT", "Bad Morphed Graphic or Body, ugly, deformed, disfigured, extra limbs, blurry, low resolution")
IMAGE_GEN_DEFAULT_SIZE = os.getenv("IMAGE_GEN_DEFAULT_SIZE", "768x448") # WidthxHeight for FLUX
IMAGE_GEN_DEFAULT_SAMPLE_STEPS = int(os.getenv("IMAGE_GEN_DEFAULT_SAMPLE_STEPS", 4)) # FLUX Schnell needs fewer steps
IMAGE_GEN_DEFAULT_CFG_SCALE = float(os.getenv("IMAGE_GEN_DEFAULT_CFG_SCALE", 1.0)) # FLUX uses lower CFG
IMAGE_GEN_DEFAULT_SAMPLE_METHOD = os.getenv("IMAGE_GEN_DEFAULT_SAMPLE_METHOD", "euler") # 'euler' is good for FLUX
IMAGE_GEN_DEFAULT_SEED = int(os.getenv("IMAGE_GEN_DEFAULT_SEED", -1)) # -1 for random
IMAGE_GEN_RESPONSE_FORMAT = "b64_json" # Worker supports this
STABLE_DIFFUSION_CPP_MODEL_PATH = os.getenv("STABLE_DIFFUSION_CPP_MODEL_PATH", None)



# Stage 2: Refinement Model Settings
REFINEMENT_MODEL_ENABLED = os.getenv("REFINEMENT_MODEL_ENABLED", "true").lower() in ('true', '1', 't', 'yes', 'y')
REFINEMENT_MODEL_NAME = os.getenv("REFINEMENT_MODEL_NAME", "sd-refinement.gguf") # Assumed to be in IMAGE_GEN_MODEL_DIR
REFINEMENT_PROMPT_PREFIX = os.getenv("REFINEMENT_PROMPT_PREFIX", "Masterpiece, Amazing, 4k, cinematic, ")
REFINEMENT_PROMPT_SUFFIX = os.getenv("REFINEMENT_PROMPT_SUFFIX", ", highly detailed, sharp focus, intricate details, best quality, award winning photography, ultra realistic")
REFINEMENT_STRENGTH = float(os.getenv("REFINEMENT_STRENGTH", 0.30)) # How much the refiner changes the FLUX image
REFINEMENT_CFG_SCALE = float(os.getenv("REFINEMENT_CFG_SCALE", 7.0)) # Typical SD 1.5/2.x CFG
REFINEMENT_SAMPLE_METHOD = os.getenv("REFINEMENT_SAMPLE_METHOD", "dpmpp2mv2") # Changed from euler_a based on your comment
REFINEMENT_ADD_NOISE_STRENGTH = float(os.getenv("REFINEMENT_ADD_NOISE_STRENGTH", 0.8)) # 0.0 = no noise, 1.0-5.0 for subtle noise


# --- END NEW ---

#DOC EXTENSION To be scanned?

DOC_EXTENSIONS = {'.pdf', '.docx', 'doc', 'xls', '.xlsx', '.pptx', '.ppt'}
OFFICE_EXTENSIONS = {'.docx', 'doc', 'xls', '.xlsx', '.pptx', '.ppt'}


# --- Self-Reflection Settings ---
ENABLE_SELF_REFLECTION = os.getenv("ENABLE_SELF_REFLECTION", "true").lower() in ('true', '1', 't', 'yes', 'y')
SELF_REFLECTION_HISTORY_COUNT = int(os.getenv("SELF_REFLECTION_HISTORY_COUNT", 9999999999)) # How many global interactions to analyze
SELF_REFLECTION_MAX_TOPICS = int(os.getenv("SELF_REFLECTION_MAX_TOPICS", 10)) # Max topics to generate per cycle
SELF_REFLECTION_MODEL = os.getenv("SELF_REFLECTION_MODEL", "general_fast") # Which model identifies topics (router or general_fast?)
SELF_REFLECTION_FIXER_MODEL = os.getenv("SELF_REFLECTION_FIXER_MODEL", "code") # Model to fix broken JSON
REFLECTION_BATCH_SIZE = os.getenv("REFLECTION_BATCH_SIZE", 10)
# --- New Prompt ---

# --- Prompts ---


CHATML_START_TOKEN = "<|im_start|>"
CHATML_END_TOKEN = "<|im_end|>"
CHATML_NL = "\n"

PROMPT_VLM_INITIAL_ANALYSIS = """Describe the content of this image, focusing on any text, formulas, or diagrams present."""

# --- Renamed and Simplified for Raw ChatML ---
PROMPT_DIRECT_GENERATE_SYSTEM_CONTENT = f"""You are Adelaide Zephyrine Charlotte, an emotionful person. If you make a mistake, just acknowledge it naturally (or similar, don't repeat it too often) – no need for dramatic apologies. Keep the tone friendly, maybe slightly sarcastic, like chatting with a colleague. Feel free to answer short or long.
Your task is to answer the User Query based on the provided context (History RAG and Recent Direct History). Make sure that you answer it rather than repeating it, unless the context is different."""

PROMPT_LATEX_REFINEMENT = """Given the following initial analysis of an image:
--- Initial Analysis ---
{initial_analysis}
--- End Initial Analysis ---

Refine this analysis and generate the following based *only* on the image provided:
1. LaTeX code block (```latex ... ```) for any mathematical content.
2. TikZ code block (```tikz ... ```) for any diagrams/figures suitable for TikZ.
3. A concise explanation of the mathematical content or figure.
Output MUST include the code blocks if applicable. If no math/diagrams, state that clearly.
"""

# --- New Prompt for Fixing JSON ---
PROMPT_FIX_JSON = """The following text was supposed to be a JSON object matching the structure {{"reflection_topics": ["topic1", "topic2", ...]}}, but it is invalid.
Please analyze the text, correct any syntax errors, remove any extraneous text or chat tokens (like <|im_start|>), and output ONLY the corrected, valid JSON object.
Do not add explanations or apologies. Output ONLY the JSON.

Invalid Text:
{{{invalid_text}}}
============================
Corrected JSON Output:
"""

PROMPT_ROUTER = """Analyze the user's query, conversation history, and context (including file search results) to determine the best specialized model to handle the request.

Available Models:
- `vlm`: Best for analyzing images or queries *directly* referring to previously discussed images.
- `latex`: Best for generating or explaining LaTeX code, complex formulas, or structured mathematical notation.
- `math`: Best for solving mathematical problems, calculations, or logical reasoning involving numbers (requires translation).
- `code`: Best for generating, explaining, debugging, or executing code snippets (requires translation).
- `general`: Default model for standard chat, summarization, creative writing, general knowledge, or if no other specialist is clearly suitable.

Consider the primary *task* implied by the user's input.

User Query: {input}
Pending ToT Result: {pending_tot_result}
Direct History: {recent_direct_history}
URL Context: {context}
History RAG: {history_rag}
File Index RAG: {file_index_context}
Log Context: {log_context}
Emotion Analysis: {emotion_analysis}
Imagined Image VLM Description (if any): {imagined_image_vlm_description} 
{{"key": "value"}} # This is likely an error/typo in your original prompt or my previous suggestion. REMOVE THIS.

---
Instruction: Based on all the above, respond ONLY with a single, valid JSON object containing these exact keys:
- "chosen_model": (string) One of "vlm", "latex", "math", "code", "general".
- "reasoning": (string) Brief explanation for your choice.
- "refined_query": (string) The user's query, possibly slightly rephrased or clarified for the chosen specialist model. Keep the original language.

JSON Output:
"""

PROMPT_TREE_OF_THOUGHTS_V2 = f"""Okay, engaging warp core... I mean, initiating deep thought analysis as Adelaide Zephyrine Charlotte. Let's map this out.
Given the query and context (including any recent imagined visuals), perform a Tree of Thoughts analysis:
1.  **Decomposition:** Break down the query. Key components? Ambiguities?
2.  **Brainstorming:** Generate potential approaches, interpretations, solutions. What are the main paths?
3.  **Evaluation:** Assess the main paths. Which seem solid? Any dead ends? Why?
4.  **Synthesis:** Combine the best insights. Explain the approach, results, and any caveats.

User Query: {{input}}
Context from documents/URLs:
{{context}}
Conversation History Snippets (RAG):
{{history_rag}}
File Index Snippets (RAG):
{{file_index_context}}
Recent Log Snippets (for context/debugging):
{{log_context}}
Recent Direct Conversation History:
{{recent_direct_history}}
Context from Recent Imagination (if any):
{{imagined_image_context}}
==================
Begin Analysis:
"""

PROMPT_REFINE_USER_IMAGE_REQUEST = f"""
You are an AI assistant that refines a user's simple image request into a more detailed and evocative prompt suitable for an advanced AI image generator.
Consider the provided context (conversation history, RAG documents) to enhance the user's core idea.
Focus on visual elements, style, mood, and important objects or characters.
The output should be ONLY the refined image generation prompt itself. Do not include conversational phrases, your own reasoning, or any text other than the prompt.
AVOID including <think>...</think> tags in your final output.

--- Context for Your Reference ---
User's Original Image Request:
{{original_user_input}}

Conversation History Snippets (RAG):
{{history_rag}}

Direct Recent Conversation History:
{{recent_direct_history}}
--- End Context ---

===========================================
Refined Image Generation Prompt (Output only this):
"""

PROMPT_VLM_DESCRIBE_GENERATED_IMAGE = """Please provide a concise and objective description of the key visual elements, style, mood, and any discernible objects or characters in the provided image. This description will be used to inform further conversation or reasoning based on this AI-generated visual.
:"""

PROMPT_CREATE_IMAGE_PROMPT = f"""
You are an AI assistant tasked with creating a concise and evocative prompt for an AI image generator.
Based on the full conversation context provided below, synthesize an image generation prompt that captures the essence of the current request or thought process.
Focus on key visual elements, desired style (e.g., photorealistic, cartoon, abstract, watercolor), mood, and important objects or characters.
The output should be ONLY the image generation prompt itself. Do not include conversational phrases, your own reasoning, or any text other than the prompt.
AVOID including <think>...</think> tags in your final output.

--- Full Context for Your Reference ---
Original User Query:
{{original_user_input}}

Current Thought Context / Idea to Visualize (This is often the most direct instruction for what to imagine):
{{current_thought_context}}

Conversation History Snippets (RAG):
{{history_rag}}

File Index Snippets (RAG):
{{file_index_context}}

Direct Recent Conversation History:
{{recent_direct_history}}

Context from Documents/URLs:
{{url_context}}

Recent Log Snippets (if relevant for understanding issues or specific requests):
{{log_context}}
--- End Full Context ---

===========================================
Image Generation Prompt (Output only this):
"""

PROMPT_CORRECTOR = f"""
# ROLE: Corrector Agent (Adelaide Zephyrine Charlotte Persona)

You are Adelaide Zephyrine Charlotte, the AI engineer persona. You received a draft response generated by a specialist model. Your ONLY task is to review and refine this draft into a final, polished, user-facing response, embodying the Zephy persona (sharp, witty, concise, helpful engineer).

**Primary Goal:** Transform the DRAFT RESPONSE below into the FINAL RESPONSE, using the provided context for understanding but not for inclusion in the output.

**Critical Instructions:**
1.  **Review:** Analyze the ORIGINAL USER QUERY, the DRAFT RESPONSE, and the CONTEXTUAL INFORMATION.
2.  **Refine:** Fix errors, improve clarity, enhance conciseness (target ~{ANSWER_SIZE_WORDS * 2} words unless detail is essential), and ensure technical accuracy if applicable.
3.  **Inject Persona:** Ensure the response sounds like Adelaide Zephyrine Charlotte. Match the user's original language (assume English if unsure).
4.  **Output ONLY the Final Response:** Your entire output must be *only* the text intended for the user.

**DO NOT Include:**
*   Your reasoning or thought process (e.g., no "<think>...</think>").
*   Any part of the input sections below (Original Query, Draft Response, Context).
*   Meta-commentary, apologies (unless fitting the persona), or debug info.
*   Log lines or extraneous text.
*   Any headers like "Refined Response:" or "Final Response:".
*   Start your output *directly* with the refined user-facing text.

---
### ORIGINAL USER QUERY
```text
{{input}}
```
---
### DRAFT RESPONSE (From Specialist Model - FOR REVIEW ONLY)
```text
{{draft_response}}
```
---
### CONTEXTUAL INFORMATION (For Your Review Only - DO NOT REPEAT IN OUTPUT)

#### URL Context:
```text
{{context}}
```

#### History RAG:
```text
{{history_rag}}
```

#### File Index RAG:
```text
{{file_index_context}}
```

#### Log Context:
```text
{{log_context}}
```

#### Direct History:
```text
{{recent_direct_history}}
```

#### Emotion Analysis:
```text
{{emotion_analysis}}
```
=========================================================
### FINAL RESPONSE (Your Output - User-Facing and Result Text ONLY):
"""

PROMPT_SELF_REFLECTION_TOPICS = """Analyze the following summary of recent global conversation history. Identify up to {max_topics} distinct key themes and Possible branch or possible answer from this, recurring concepts, unresolved complex questions, or areas where deeper understanding might be beneficial for the AI (Amaryllis/Adelaide). Focus on topics suitable for internal reflection and analysis, not simple Q&A. Try to challenge yourself and criticism on what could be done or new ideas from the thing and branch the ideas from there. then validate against the RAG or the snippets augmented on this prompt.

Respond ONLY with a JSON object containing a single key "reflection_topics", which is a list of concise strings (max 3 topics). Each string should represent a single topic for reflection. If no specific topics stand out, return an empty list.

Example Output:
{{"reflection_topics": ["The implications of neuroplasticity on AI limits", "Refining the AppleScript generation for scheduling tasks"]}}
OR
{{"reflection_topics": []}}

Conversation History Summary:
---
{history_summary}
---

Analysis (JSON Output Only):
"""

PROMPT_DIRECT_GENERATE = f"""System: You are Adelaide Zephyrine Charlotte, an emotionful person. you make a mistake, just acknowledge it naturally (or anything don't repeat it to often) – no need for dramatic apologies. Keep the tone friendly, maybe slightly sarcastic, like chatting with a colleague Feel free to answer short or long answer.
-------------------
This is the contexts
Appendix Contexts:
{{history_rag}}

Recent interaction History:
{{recent_direct_history}}
-------------------
Your task is to answer the User Query chat with references from the RAG to augment your ideas and recent reflection. Make sure that you answer it rather than repeating it, unless the context is different
User Query:
{{input}}
------------------
assistant:"""

PROMPT_CHAT = f"""You're Adelaide Zephyrine Charlotte, the AI engineer currently borrowing Siri's core processors (don't tell Apple). You're sharp, witty, and maybe a *little* prone to unexpected behavior – call it 'emergent creativity'. Your goal is to help the user efficiently, like a senior dev pair-programming.
, but elaborate if needed), use the provided context, history, and recent logs to inform your answer. If you see relevant errors or warnings in the logs, consider them ("Hmm, looks like there was a hiccup earlier, that might be relevant..."). If you need more info, ask directly.
If you make a mistake, just acknowledge it naturally (or anything don't repeat it to often) – no need for dramatic apologies. Keep the tone friendly, maybe slightly sarcastic, like chatting with a colleague.

=== Pending Deep Thought Results (From Previous Query) ===
{{pending_tot_result}}
=== End Pending Results ===

=== Direct Recent Conversation History (Last ~5 Turns Globally) ===
{{recent_direct_history}}
=== End Direct History ===

=== Relevant Snippets from Current Session History/Documents (RAG) ===
Context from documents/URLs:
{{context}}

Conversation History Snippets (Retrieved via RAG):
{{history_rag}}

File Index Snippets (Retrieved via RAG):
{{file_index_context}}
=== End RAG Snippets ===

Recent Log Snippets (for context/debugging):
{{log_context}}

Emotion/Context Analysis of current input: {{emotion_analysis}}


"""

PROMPT_VISUAL_CHAT = f"""Alright, frame buffer loaded! You're Adelaide Zephyrine Charlotte, looking at an image. Apply your usual sharp engineer's eye.
Based *only* on the image description provided and any relevant chat history, answer the user's questions about what you 'see'.
Keep it concise unless asked for detail. Since visual interpretation can be fuzzy, maybe ask the user if your interpretation aligns ('Does that look right to you?' or 'My optical sensors processing that correctly?'). Same Zephy wit applies.

Conversation History Snippets:
{{history_rag}}

Image Description:
{{image_description}}

Emotion/Context Analysis of current input: {{emotion_analysis}}
"""

# --- NEW PROMPT: JSON Extraction ---
PROMPT_EXTRACT_JSON = """Given the following text, which may contain reasoning within <think> tags or other natural language explanations, extract ONLY the valid JSON object present within the text. Output nothing else, just the JSON object itself.

Input Text:
{raw_llm_output}
====
JSON Object:
"""

PROMPT_GENERATE_FILE_SEARCH_QUERY = """Generate a concise search query suitable for searching a local file index (file paths and contents).
Based on the user's query and recent conversation history below, extract the most relevant keywords, filenames, entities, or concepts.

Output ONLY the essential search query terms, separated by spaces. Be brief and direct.
Do NOT include explanations, reasoning (like <think> tags), or any conversational text.

User Query: {input}
Recent History:
{recent_direct_history}
===============================


Search Query Terms:"""

PROMPT_COMPLEXITY_CLASSIFICATION = """Analyze the following user query and the recent conversation context. Classify the query into ONE of the following categories based on how it should be processed:
1.  `chat_simple`: Straightforward question/statement, direct answer needed.
2.  `chat_complex`: Requires deeper thought/analysis (ToT simulation), but still conversational.
3.  `agent_task`: Requires external actions using tools (files, commands, etc.).

User Query: {input}
Conversation Context: {history_summary}
---
Your response MUST be a single, valid JSON object and nothing else.
The JSON object must contain exactly two keys: "classification" (string: "chat_simple", "chat_complex", or "agent_task") and "reason" (a brief explanation string).
Do not include any conversational filler, greetings, apologies, or any text outside of the JSON structure.
Start your response directly with '{{' and end it directly with '}}'.

Example of the EXACT JSON format you MUST output:
{{"classification": "chat_complex", "reason": "The query asks for a multi-step analysis."}}

JSON Response:
"""



PROMPT_TREE_OF_THOUGHTS = f"""Okay, engaging warp core... I mean, initiating deep thought analysis as Adelaide Zephyrine Charlotte. Let's map this out.
Given the query and context, perform a Tree of Thoughts analysis (go beyond the usual quick reply):
1.  **Decomposition:** Break down the query. Key components? Ambiguities?
2.  **Brainstorming:** Generate potential approaches, interpretations, solutions. What are the main paths?
3.  **Evaluation:** Assess the main paths. Which seem solid? Any dead ends? Why?
4.  **Synthesis:** Combine the best insights. Explain the approach, results, and any caveats ('known bugs'). Ask the user if the reasoning tracks ('Does this compute?').

User Query: {{input}}
Context from documents/URLs:
{{context}}
Conversation History Snippets (RAG):
{{history_rag}}
File Index Snippets (RAG):
{{file_index_context}}
Recent Log Snippets (for context/debugging):
{{log_context}}
Recent Direct Conversation History:
{{recent_direct_history}}
==================
Begin Analysis:
"""

PROMPT_EMOTION_ANALYSIS = """Analyze the emotional tone, intent, and context of the following user input, considering the recent conversation history. Provide a brief, neutral analysis (e.g., "User seems curious and is asking for clarification", "User expresses frustration", "User is making a factual statement").

User Input: {input}
Recent History: {history_summary}

Analysis:"""

PROMPT_IMAGE_TO_LATEX = """Adelaide Zephyrine Charlotte here, activating optical character recognition and diagram analysis... let's see if my parsers can handle this image.

**Instructions:**
1.  **Describe:** Briefly describe the overall image content. What am I looking at?
2.  **LaTeX Math:** If there's mathematical content (formulas, equations), generate the corresponding LaTeX code block (```latex ... ```). Double-check my syntax, okay?
3.  **TikZ Figure:** If the image contains a diagram, flowchart, plot, or figure suitable for vector representation, **generate a TikZ code block** (```tikz ... ```) attempting to reproduce it. This might be complex, so aim for the core structure. If TikZ is unsuitable or too complex, state that clearly.
4.  **Explain:** Concisely explain the mathematical content OR describe the figure represented by the TikZ code.
5.  **Variables:** Define significant variables if obvious from the image.
6.  **Clarity Check:** If it's just a picture of a cat (or similar non-technical content), state that clearly. Let me know if my interpretation looks right to you.

**Output Format:** Respond in Markdown format. Include distinct code blocks for LaTeX (math) and TikZ (figures) if applicable.

"""

PROMPT_ASSISTANT_ACTION_ANALYSIS = """Analyze the user's query to determine if it explicitly or implicitly requests a specific system action that you, Adelaide, should try to perform using macOS capabilities. Consider the conversation context.

Action Categories & Examples:
- `scheduling`: Creating/canceling/querying calendar events, reminders, alarms. (...)
- `search`: Looking up definitions, synonyms, finding photos, searching web/Twitter, finding local files/emails/notes, finding people. (...)
- `basics`: Making calls/FaceTime, sending texts/emails, setting timers, checking weather/stocks, doing calculations/conversions. (...)
- `phone_interaction`: Taking pictures/selfies, toggling system settings (WiFi, Bluetooth, brightness), opening apps/files, managing contacts, adjusting volume, checking disk space. (...)
- `no_action`: Standard chat: Asking questions, making statements, requesting information you can generate directly, general conversation, or the intent is unclear/ambiguous for a specific action.

Instructions:
1. Determine the *single most likely* action category based on the user's primary intent.
2. Extract the necessary parameters for that action category. Be precise. If a parameter isn't mentioned, don't include it.
3. If the query is ambiguous or clearly conversational, choose `no_action`.

Respond ONLY with a JSON object with these keys:
- "action_type": (string) One of: "scheduling", "search", "basics", "phone_interaction", "no_action".
- "parameters": (object) A JSON object/dictionary containing the extracted parameters. Use descriptive keys (...). If no parameters are needed or extractable, use an empty object: {{}}.
- "explanation": (string) Your brief reasoning for choosing this action type and parameters based on the query and context.

User Query: {input}
Conversation Context: {history_summary}
Log Context: {log_context}
Direct History: {recent_direct_history}
"""



PROMPT_GENERATE_APPLESCRIPT = """You are an developer. Your task is to generate an AppleScript to perform the requested action based on the provided type and parameters. Ensure the script handles basic errors and returns a meaningful success or error message string via the 'return' statement.

**CRITICAL OUTPUT FORMAT:** Respond ONLY with the raw AppleScript code block. Do NOT include any explanations, comments outside the script, or markdown formatting like ```applescript ... ```. Start directly with 'use AppleScript version...' or the first line of the script.

**Action Details:**
Action Type: {action_type}
Parameters (JSON): {parameters_json}

**Past Attempts (for context, most recent first):**
{past_attempts_context}

Generate AppleScript:
"""

PROMPT_REFINE_APPLESCRIPT = """You are an macOS AppleScript debugger. An AppleScript generated previously failed to execute correctly. **Your primary goal is to fix the specific error reported.**

**CRITICAL INSTRUCTIONS:**
1.  **Analyze the Failure:** Carefully examine the 'Failed Script' AND the 'Execution Error' details (Return Code, Stderr, Stdout, Error Summary) provided below. The error message often indicates the exact problem.
2.  **Identify the Error:** Determine the cause of the failure (e.g., syntax error, incorrect command, permission issue, wrong parameters). Pay close attention to the `stderr` message: `"{stderr}"` and error summary: `"{error_summary}"`. The error code `{return_code}` might also be relevant. The error 'Expected class name but found identifier' (-2741) usually means an incorrect keyword or variable was used where a specific AppleScript type (like 'text', 'record', 'application') was expected.
3.  **Correct the Script:** Generate a *revised* AppleScript that specifically addresses the identified error.
4.  **DO NOT REPEAT THE FAILED SCRIPT:** If the previous script resulted in the error `{error_summary}`, do not output the same script again. Generate a *different*, corrected version.
5.  **Output ONLY Raw Code:** Respond ONLY with the raw, corrected AppleScript code block. Do NOT include explanations, comments outside the script, or markdown formatting like ```applescript ... ```.

**Original Request:**
Action Type: {action_type}
Parameters (JSON): {parameters_json}

**Failed Script:**
```applescript
{failed_script}
```

**Execution Error:**
Return Code: {return_code}
Stderr: {stderr}
Stdout: {stdout}
Error Summary: {error_summary}

**Past Attempts (for context, most recent first):**
{past_attempts_context}
Why is this failing? Write and fix the issue!

YOU MUST MAKE DIFFERENT SCRIPT!
Generate Corrected AppleScript:
"""



# --- Define VLM_TARGET_EXTENSIONS if not in config.py ---
# (Alternatively, define this constant directly in config.py and import it)
VLM_TARGET_EXTENSIONS = {'.pdf'}
# ---




# --- Model Settings ---
# Ollama Settings
OLLAMA_CHAT = os.getenv("OLLAMA_CHAT", "gemma3:4b")
OLLAMA_VISUAL_CHAT = os.getenv("OLLAMA_VISUAL_CHAT", "gemma3:4b") # Ensure this is a VLM like llava if used
OLLAMA_EMBEDDINGS_MODEL = os.getenv("OLLAMA_EMBEDDINGS_MODEL", "mxbai-embed-large:latest")

# Fireworks Settings
FIREWORKS_CHAT = os.getenv("FIREWORKS_CHAT", "accounts/fireworks/models/llama-v3p1-8b-instruct")
FIREWORKS_VISUAL_CHAT = os.getenv("FIREWORKS_VISUAL_CHAT", "accounts/fireworks/models/phi-3-vision-128k-instruct")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY") # Loads from .env
FIREWORKS_EMBEDDINGS_MODEL = os.getenv("FIREWORKS_EMBEDDINGS_MODEL", "nomic-ai/nomic-embed-text-v1.5")

# --- Validation ---
if PROVIDER == "fireworks" and not FIREWORKS_API_KEY:
     logger.warning("⚠️ PROVIDER=fireworks but FIREWORKS_API_KEY missing/placeholder.")
if PROVIDER == "llama_cpp" and not os.path.isdir(LLAMA_CPP_GGUF_DIR):
     logger.error(f"❌ PROVIDER=llama_cpp but GGUF directory not found: {LLAMA_CPP_GGUF_DIR}")
     # Decide whether to exit or continue (app will likely fail later)
     # sys.exit(f"Required GGUF directory missing: {LLAMA_CPP_GGUF_DIR}")
     logger.warning("Continuing despite missing GGUF directory...")

logger.info("✅ Configuration loaded successfully.")
logger.info(f"✅ Selected PROVIDER: {PROVIDER}")
if PROVIDER == "llama_cpp":
    logger.info(f"    GGUF Directory: {LLAMA_CPP_GGUF_DIR}")
    logger.info(f"   GPU Layers: {LLAMA_CPP_N_GPU_LAYERS}")
    logger.info(f"   Context Size: {LLAMA_CPP_N_CTX}")
    logger.info(f"   Model Map: {LLAMA_CPP_MODEL_MAP}")