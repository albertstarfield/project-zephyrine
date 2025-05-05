# ai_provider.py
import os
import sys
import time
import threading
import gc # For garbage collection
from typing import Dict, Any, Optional, List, Iterator
from loguru import logger

# --- Langchain Imports ---
# Core Language Model components
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
# Core Message types (including Chunks)
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ChatMessage,
    AIMessageChunk, # <<< Correct location
    # HumanMessageChunk, # (Import if needed)
    # SystemMessageChunk, # (Import if needed)
    # FunctionMessageChunk, # (Import if needed)
    # ToolMessageChunk # (Import if needed)
)
# Core Output types (excluding Message Chunks)
from langchain_core.outputs import (
    ChatResult,
    ChatGeneration,
    GenerationChunk,
    ChatGenerationChunk
)
# Core Embeddings interface
from langchain_core.embeddings import Embeddings
# Core Callbacks
from langchain_core.callbacks import CallbackManagerForLLMRun

# --- Conditional Imports ---
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
        logger.warning("⚠️ Failed to import Ollama. Did you install 'langchain-ollama'? Ollama provider disabled.")
        ChatOllama = None
        OllamaEmbeddings = None

# Fireworks
try:
    from langchain_fireworks import ChatFireworks, FireworksEmbeddings
    logger.info("Using langchain_fireworks imports.")
except ImportError:
     logger.warning("⚠️ Failed to import Fireworks. Did you install 'langchain-fireworks'? Fireworks provider disabled.")
     ChatFireworks = None
     FireworksEmbeddings = None

# llama-cpp-python
try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
    logger.info("✅ llama-cpp-python imported.")
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("⚠️ llama-cpp-python not installed. Run 'pip install llama-cpp-python'. llama_cpp provider disabled.")
    llama_cpp = None # Placeholder

# Stable Diffusion (Placeholder)
try:
    # import stable_diffusion_cpp # Or however the python bindings are imported
    STABLE_DIFFUSION_AVAILABLE = False # Set to True if import succeeds
    logger.info("✅ stable-diffusion-cpp imported (Placeholder).")
except ImportError:
    STABLE_DIFFUSION_AVAILABLE = False
    logger.warning("⚠️ stable-diffusion-cpp bindings not found. Image generation disabled.")


# --- Local Imports ---
try:
    # Import all config variables
    from config import * # Includes PROVIDER, model names, paths, MAX_TOKENS etc.
except ImportError:
    logger.critical("❌ Failed to import config.py in ai_provider.py!")
    sys.exit("AIProvider cannot function without config.")

# === llama-cpp-python Langchain Wrappers ===

# --- Chat Model Wrapper ---
class LlamaCppChatWrapper(SimpleChatModel):
    """
    Langchain chat model wrapper for a dynamically loaded llama-cpp-python instance
    managed by AIProvider. NOW uses non-streaming _call internally.
    """
    ai_provider: 'AIProvider' # Forward reference
    model_role: str # Logical role (e.g., "router", "vlm") to request from provider
    model_kwargs: Dict[str, Any] # Kwargs for Llama.create_chat_completion

    def _format_messages_for_llama_cpp(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """
        Helper function to format Langchain messages into the llama_cpp dictionary list format.
        Handles multimodal content for the 'vlm' role.
        (Extracted formatting logic from original _stream method)
        """
        formatted_messages = []
        llm_instance = self.ai_provider._get_loaded_llama_instance(self.model_role, task_type="chat") # Needed for VLM check

        for msg in messages:
            role = "user" # Default
            if isinstance(msg, HumanMessage): role = "user"
            elif isinstance(msg, AIMessage): role = "assistant"
            elif isinstance(msg, SystemMessage): role = "system"
            elif isinstance(msg, ChatMessage): role = msg.role # Use role if specified

            # --- Handle Content Formatting ---
            if isinstance(msg.content, str):
                # Simple string content
                formatted_messages.append({"role": role, "content": msg.content})
            elif isinstance(msg.content, list):
                # List content (potentially multimodal)
                # Check if it's the VLM role AND the loaded instance appears valid
                if self.model_role == "vlm" and llm_instance:
                    logger.debug(f"Formatting multimodal input for role '{self.model_role}'...")
                    content_list = []
                    has_image = False
                    for item in msg.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            content_list.append({"type": "text", "text": item.get("text", "")})
                        elif isinstance(item, dict) and item.get("type") == "image_url":
                            img_url_data = item.get("image_url", {}).get("url", "")
                            if img_url_data.startswith("data:image"):
                                content_list.append({"type": "image_url", "image_url": {"url": img_url_data}})
                                has_image = True
                                logger.trace("Image part formatted for llama.cpp input.")
                            else:
                                logger.warning(f"Skipping unsupported image_url format: {img_url_data[:50]}...")
                                content_list.append({"type": "text", "text": "[Unsupported Image Placeholder]"})
                        else:
                            logger.warning(f"Unexpected item type in message content list: {type(item)}")
                            content_list.append({"type": "text", "text": str(item)})

                    if has_image:
                        formatted_messages.append({"role": role, "content": content_list})
                        logger.trace(f"Appended multimodal message for role '{role}'.")
                    else: # List format but no image found, combine text
                        text_content = " ".join([c.get("text","") for c in content_list if c.get("type")=="text"])
                        formatted_messages.append({"role": role, "content": text_content})
                        logger.trace(f"Appended combined text message (no image) for role '{role}'.")
                else:
                    # Not VLM role or no llm_instance, treat as text
                    logger.warning(f"Model role '{self.model_role}' received list content but is not VLM or instance missing. Combining text.")
                    text_content = " ".join([item.get("text", "") for item in msg.content if isinstance(item, dict) and item.get("type") == "text"])
                    formatted_messages.append({"role": role, "content": text_content})
            else:
                 # Fallback for other types
                 formatted_messages.append({"role": role, "content": str(msg.content)})
            # --- End Content Formatting ---

        return formatted_messages

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        INTERNAL Non-streaming generation call to llama.cpp.
        Returns the full response text at once.
        """
        logger.debug(f"LlamaCppChatWrapper: Executing non-streaming '_call' for role '{self.model_role}'.")
        llm_instance = self.ai_provider._get_loaded_llama_instance(self.model_role, task_type="chat")
        if not llm_instance:
            err_msg = f"LLAMA_CPP_ERROR (_call): Model for role '{self.model_role}' could not be loaded."
            logger.error(err_msg)
            # Propagate error clearly. Langchain might handle this, or raise it.
            # Returning error string is safer for now than raising within Langchain internal method.
            return f"[LLAMA_CPP_LOAD_ERROR: {err_msg}]"

        # Format messages using the helper
        formatted_messages = self._format_messages_for_llama_cpp(messages)
        if not formatted_messages:
            logger.warning(f"LlamaCppChatWrapper: No valid messages formatted for role '{self.model_role}'.")
            return "[LLAMA_CPP_FORMAT_ERROR: No messages to send]"

        logger.debug(f"llama_cpp invoking role '{self.model_role}' (non-streaming) with {len(formatted_messages)} messages.")
        try:
            # --- Call create_chat_completion with stream=False ---
            completion = llm_instance.create_chat_completion(
                messages=formatted_messages,
                max_tokens=self.model_kwargs.get("max_tokens", MAX_TOKENS),
                temperature=self.model_kwargs.get("temperature", 1.2),
                top_p=self.model_kwargs.get("top_p", 0.95),
                stop=stop,
                stream=False, # <<< Explicitly set stream to False
                **kwargs
            )

            # --- Extract content from the non-streaming response ---
            # Check the structure based on llama-cpp-python documentation/output
            if completion and 'choices' in completion and completion['choices']:
                first_choice = completion['choices'][0]
                if 'message' in first_choice and 'content' in first_choice['message']:
                    full_response_text = first_choice['message']['content']
                    logger.debug(f"LlamaCppChatWrapper: Received non-streaming response (len: {len(full_response_text)}).")
                    # Log a snippet of the actual response content for debugging
                    response_snippet = full_response_text.replace('\n', '\\n') # Show first 200 chars, escape newlines
                    logger.trace(f"LlamaCppChatWrapper: VLM Response Content Snippet: '{response_snippet}...'")
                    # Optional: Log token usage if available in 'completion.get("usage")'
                    return full_response_text or "" # Return empty string if content is None
                else:
                    logger.error(f"LLAMA_CPP_ERROR (_call): Response structure missing 'message' or 'content' in first choice: {completion}")
                    return "[LLAMA_CPP_RESPONSE_ERROR: Invalid structure]"
            else:
                logger.error(f"LLAMA_CPP_ERROR (_call): Unexpected non-streaming response structure: {completion}")
                return "[LLAMA_CPP_RESPONSE_ERROR: Unexpected structure]"

        except Exception as e:
            # Catch errors during the non-streaming call itself
            logger.error(f"LLAMA_CPP_ERROR (_call): Error during non-streaming call for role '{self.model_role}': {e}")
            logger.exception("Llama.cpp non-streaming call traceback:")
            # Return error message
            return f"[LLAMA_CPP_CALL_ERROR: {e}]"

    # --- _stream method remains mostly the same (but fixed previously) ---
    # It's good practice to keep it functional in case Langchain uses it elsewhere
    # or if you explicitly want streaming via .stream()/.astream().
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Streaming generation (kept functional, uses AIMessageChunk)."""
        logger.debug(f"LlamaCppChatWrapper: Executing streaming '_stream' for role '{self.model_role}'.") # Log differentiate
        llm_instance = self.ai_provider._get_loaded_llama_instance(self.model_role, task_type="chat")
        if not llm_instance:
            err_msg = f"LLAMA_CPP_ERROR (_stream): Model for role '{self.model_role}' could not be loaded."
            logger.error(err_msg)
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"[LLAMA_CPP_LOAD_ERROR: {err_msg}]")) # Yield error chunk
            return

        # Format messages using the helper
        formatted_messages = self._format_messages_for_llama_cpp(messages)
        if not formatted_messages:
             logger.warning(f"LlamaCppChatWrapper: No valid messages formatted for streaming role '{self.model_role}'.")
             yield ChatGenerationChunk(message=AIMessageChunk(content="[LLAMA_CPP_FORMAT_ERROR: No messages to send]"))
             return

        logger.debug(f"llama_cpp invoking role '{self.model_role}' (streaming) with {len(formatted_messages)} messages.")
        try:
            streamer = llm_instance.create_chat_completion(
                messages=formatted_messages,
                max_tokens=self.model_kwargs.get("max_tokens", MAX_TOKENS),
                temperature=self.model_kwargs.get("temperature", 1.2),
                top_p=self.model_kwargs.get("top_p", 0.95),
                stop=stop,
                stream=True, # <<< Use stream=True for this method
                **kwargs
            )

            for chunk in streamer:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta and delta["content"] is not None:
                    content_chunk = delta["content"]
                    # Yield CORRECT Langchain chunk format
                    yield ChatGenerationChunk(message=AIMessageChunk(content=content_chunk))
                    # Optional: Callback manager usage
                    if run_manager:
                        run_manager.on_llm_new_token(content_chunk)
                # Handle finish reason if needed

        except Exception as e:
            logger.error(f"LLAMA_CPP_ERROR (_stream): Error during streaming for role '{self.model_role}': {e}")
            logger.exception("Llama.cpp streaming traceback:")
            # Yield a final chunk indicating the error
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"[LLAMA_CPP_STREAM_ERROR: {e}]"))

    @property
    def _llm_type(self) -> str:
        # Keep this property
        return "llama_cpp_chat_wrapper"

# --- Embeddings Wrapper ---
class LlamaCppEmbeddingsWrapper(Embeddings):
    ai_provider: 'AIProvider'
    model_role: str = "embeddings" # Fixed role for embeddings

    def __init__(self, ai_provider: 'AIProvider'):
        """Initialize the wrapper with a reference to the AIProvider."""
        super().__init__() # Call parent initializer if necessary (good practice)
        self.ai_provider = ai_provider
        
        

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        llm_instance = self.ai_provider._get_loaded_llama_instance(self.model_role, task_type="embedding")
        if not llm_instance:
            logger.error(f"LLAMA_CPP_ERROR: Embedding model for role '{self.model_role}' could not be loaded.")
            # Return dummy embeddings of correct shape but indicate error?
            # Or raise error? Langchain behavior varies. Let's raise for clarity.
            raise RuntimeError(f"Llama.cpp embedding model ('{self.model_role}') failed to load.")
        try:
            logger.debug(f"llama_cpp embedding {len(texts)} documents using role '{self.model_role}'.")
            # llama.cpp embed method handles lists directly
            embeddings = llm_instance.embed(texts)
            logger.debug(f"Generated {len(embeddings)} document embeddings.")
            return embeddings
        except Exception as e:
            logger.error(f"LLAMA_CPP_ERROR: Error during document embedding: {e}")
            logger.exception("Llama.cpp document embedding traceback:")
            raise # Re-raise the exception

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        llm_instance = self.ai_provider._get_loaded_llama_instance(self.model_role, task_type="embedding")
        if not llm_instance:
            logger.error(f"LLAMA_CPP_ERROR: Embedding model for role '{self.model_role}' could not be loaded.")
            raise RuntimeError(f"Llama.cpp embedding model ('{self.model_role}') failed to load.")
        try:
            logger.debug(f"llama_cpp embedding query using role '{self.model_role}'.")
            # Pass query as a list of one item
            embedding = llm_instance.embed([text])[0]
            logger.debug("Generated query embedding.")
            return embedding
        except Exception as e:
            logger.error(f"LLAMA_CPP_ERROR: Error during query embedding: {e}")
            logger.exception("Llama.cpp query embedding traceback:")
            raise # Re-raise the exception

# === AI Provider Class ===

class AIProvider:
    """
    Handles initialization and access to different AI models (Ollama, Fireworks, llama.cpp)
    and manages dynamic loading for llama.cpp.
    """
    def __init__(self, provider_name):
        self.provider_name = provider_name.lower()
        self.models: Dict[str, Any] = {}
        self.embeddings: Optional[Embeddings] = None
        self.EMBEDDINGS_MODEL_NAME: Optional[str] = None
        self.image_generator: Any = None

        # --- llama.cpp specific state ---
        self._loaded_gguf_path: Optional[str] = None
        self._loaded_llama_instance: Optional[llama_cpp.Llama] = None
        # --->>> NEW: Track last task type <<<---
        self._last_task_type: Optional[str] = None # Stores "chat" or "embedding"
        # --->>> END NEW <<<---
        self._llama_model_access_lock = threading.Lock() if LLAMA_CPP_AVAILABLE and self.provider_name == "llama_cpp" else None
        self._llama_model_map: Dict[str, str] = {}
        self._llama_gguf_dir: Optional[str] = None


        logger.info(f"🤖 Initializing AI Provider: {self.provider_name}")
        if self.provider_name == "llama_cpp" and self._llama_model_access_lock: logger.info("   🔑 Initialized threading.Lock for llama.cpp access.")
        elif self.provider_name == "llama_cpp": logger.error("   ❌ Failed to initialize threading.Lock for llama.cpp")

        self._validate_config()
        self.setup_provider()
        self._setup_image_generator()

    def _validate_config(self):
        """Basic checks for required config based on provider."""
        if self.provider_name == "fireworks" and not FIREWORKS_API_KEY: logger.error("🔥 PROVIDER=fireworks requires FIREWORKS_API_KEY"); # Exit?
        if self.provider_name == "llama_cpp":
            if not LLAMA_CPP_AVAILABLE: logger.error("❌ PROVIDER=llama_cpp but llama-cpp-python not installed."); sys.exit("Missing llama-cpp-python dependency.")
            if not os.path.isdir(LLAMA_CPP_GGUF_DIR): logger.error(f"❌ PROVIDER=llama_cpp requires GGUF directory: {LLAMA_CPP_GGUF_DIR}"); sys.exit(f"Missing GGUF directory: {LLAMA_CPP_GGUF_DIR}")
            # --->>> NEW: Check for distinct embedding model <<<---
            embedding_file = LLAMA_CPP_MODEL_MAP.get("embeddings")
            if not embedding_file:
                logger.error("❌ PROVIDER=llama_cpp requires a distinct 'embeddings' entry in LLAMA_CPP_MODEL_MAP.")
                sys.exit("Missing embedding model configuration for llama.cpp")
            # Check if embedding file is reused for a chat role
            for role, fname in LLAMA_CPP_MODEL_MAP.items():
                if role != "embeddings" and fname == embedding_file:
                    logger.error(f"❌ PROVIDER=llama_cpp: Embedding model '{embedding_file}' CANNOT be reused for chat role '{role}'. Use different GGUF files.")
                    sys.exit("Embedding model file reused for chat role.")
            # --->>> END NEW <<<---

    def _load_llama_model(self, required_gguf_path: str, task_type: str) -> Optional[llama_cpp.Llama]:
        """
        Loads or returns the cached llama_cpp.Llama instance.
        Handles unloading previous model if path OR task_type changes. Thread-safe.
        """
        if not LLAMA_CPP_AVAILABLE or not self._llama_model_access_lock:
            logger.error("_load_llama_model: Preconditions not met (Lib available? Lock init?).")
            return None

        with self._llama_model_access_lock:
            logger.trace(f"Acquired lock for load/switch: Requesting '{os.path.basename(required_gguf_path)}' for task '{task_type}'")

            # --- Determine if Reload is Needed ---
            needs_reload = False
            if not self._loaded_llama_instance:
                needs_reload = True # First load
                logger.debug("No model loaded, proceeding to load.")
            elif self._loaded_gguf_path != required_gguf_path:
                needs_reload = True # Different model file requested
                logger.info(f"Switching model from '{os.path.basename(self._loaded_gguf_path)}' to '{os.path.basename(required_gguf_path)}'.")
            elif self._last_task_type != task_type:
                needs_reload = True # Same model file, but different task type (e.g., chat after embed)
                logger.warning(f"Forcing reload of '{os.path.basename(required_gguf_path)}' due to task type switch: '{self._last_task_type}' -> '{task_type}'.")
            else:
                # Paths and task types match, use cached instance
                logger.debug(f"Using cached llama.cpp instance: '{os.path.basename(required_gguf_path)}' for task '{task_type}'.")
                instance_to_return = self._loaded_llama_instance

            # --- Perform Reload if Needed ---
            if needs_reload:
                instance_to_return = None # Default return for this branch

                # 1. Unload if necessary
                if self._loaded_llama_instance:
                    logger.info(f"Unloading previous llama.cpp model: {os.path.basename(self._loaded_gguf_path or 'Unknown')}")
                    try:
                        del self._loaded_llama_instance # Attempt explicit deletion
                        self._loaded_llama_instance = None
                        self._loaded_gguf_path = None
                        self._last_task_type = None
                        gc.collect() # Hint garbage collector
                        logger.info("Previous llama.cpp model unloaded.")
                        # Optional: Add small delay IF experiencing driver issues after unload
                        # time.sleep(0.2)
                    except Exception as del_err:
                         logger.error(f"Error during explicit deletion of Llama instance: {del_err}")
                         # Attempt to continue with load anyway, but log the issue

                # 2. Load new model
                logger.info(f"Loading llama.cpp model: '{os.path.basename(required_gguf_path)}' for task '{task_type}'...")
                logger.info(f"  >> Path: {required_gguf_path}")
                # ... (log GPU layers, context size etc.) ...
                logger.info(f"  >> GPU Layers: {LLAMA_CPP_N_GPU_LAYERS}")
                logger.info(f"  >> Context Size: {LLAMA_CPP_N_CTX}")


                if not os.path.isfile(required_gguf_path):
                     logger.error(f"LLAMA_CPP_ERROR: Model file not found: {required_gguf_path}")
                else:
                     load_start_time = time.monotonic()
                     try:
                         # >>> MODIFIED: Determine embedding/chat_format from task_type <<<
                         is_embedding_task = (task_type == "embedding")
                         role_for_path = "unknown" # Find role for logging
                         for r, f in self._llama_model_map.items():
                             if os.path.join(self._llama_gguf_dir or "", f) == required_gguf_path:
                                 role_for_path = r; break

                         logger.info(f"Initializing Llama for role '{role_for_path}', task '{task_type}' (embedding={is_embedding_task})...")
                         chat_format_to_use = "chatml" if not is_embedding_task else None
                         if chat_format_to_use: logger.info(f"  >> Setting chat_format='{chat_format_to_use}'")

                         new_instance = llama_cpp.Llama(
                             model_path=required_gguf_path,
                             n_gpu_layers=LLAMA_CPP_N_GPU_LAYERS,
                             n_ctx=LLAMA_CPP_N_CTX,
                             embedding=is_embedding_task, # <<< MODIFIED
                             verbose=LLAMA_CPP_VERBOSE,
                             chat_format=chat_format_to_use, # <<< MODIFIED
                         )
                         # >>> END MODIFIED <<<
                         self._loaded_llama_instance = new_instance
                         self._loaded_gguf_path = required_gguf_path
                         # >>> MODIFIED: Store task type <<<
                         self._last_task_type = task_type
                         instance_to_return = new_instance
                         load_duration = time.monotonic() - load_start_time
                         logger.success(f"✅ Loaded '{os.path.basename(required_gguf_path)}' ({task_type}) in {load_duration:.2f}s.")
                     except Exception as e:
                         logger.error(f"LLAMA_CPP_ERROR: Failed to load model {required_gguf_path} for task {task_type}: {e}")
                         logger.exception("Llama.cpp loading traceback:")
                         self._loaded_llama_instance = None # Ensure state is clean
                         self._loaded_gguf_path = None
                         self._last_task_type = None

            logger.trace(f"Releasing lock after load/switch check for task '{task_type}'")

        return instance_to_return # Return the instance determined inside the lock

    def _get_loaded_llama_instance(self, model_role: str, task_type: str) -> Optional[llama_cpp.Llama]:
        """
        Ensures the correct llama.cpp model is loaded via _load_llama_model
        (passing the required task_type) and returns it.
        """
        if self.provider_name != "llama_cpp": logger.error("Attempted llama.cpp instance outside llama_cpp provider."); return None
        if not self._llama_gguf_dir: logger.error("Llama.cpp GGUF directory not set."); return None
        # --->>> NEW: Validate task_type <<<---
        if task_type not in ["chat", "embedding"]:
             logger.error(f"Invalid task_type '{task_type}' passed. Must be 'chat' or 'embedding'.")
             return None

        gguf_filename = self._llama_model_map.get(model_role)
        if not gguf_filename:
            logger.error(f"No GGUF file configured for role: '{model_role}'")
            return None

        required_path = os.path.join(self._llama_gguf_dir, gguf_filename)
        # --->>> Pass task_type to the loading function <<<---
        return self._load_llama_model(required_path, task_type)

    def setup_provider(self):
        """Sets up the AI models based on the configured PROVIDER."""
        logger.info(f"🔌 Configuring provider: {self.provider_name}")
        start_time = time.time()

        try:
            if self.provider_name == "ollama":
                if not ChatOllama or not OllamaEmbeddings: raise ImportError("Ollama components not loaded")
                logger.info(f"🔌 Configuring Ollama connection to: {OLLAMA_BASE_URL}")
                self.EMBEDDINGS_MODEL_NAME = OLLAMA_EMBEDDINGS_MODEL
                self.embeddings = OllamaEmbeddings(model=self.EMBEDDINGS_MODEL_NAME, base_url=OLLAMA_BASE_URL)

                # Map logical roles to Ollama model names
                model_name_map = {
                    "router": OLLAMA_MODEL_ROUTER, "vlm": OLLAMA_MODEL_VLM, "latex": OLLAMA_MODEL_LATEX,
                    "math": OLLAMA_MODEL_MATH, "code": OLLAMA_MODEL_CODE, "general": OLLAMA_MODEL_DEFAULT_CHAT,
                    "translator": OLLAMA_MODEL_TRANSLATOR, "general_fast": OLLAMA_MODEL_GENERAL_FAST,
                    "default": OLLAMA_MODEL_DEFAULT_CHAT # Explicit default mapping
                }
                common_params = {"temperature": 1.2} # Default temp
                if MAX_TOKENS: common_params["max_tokens"] = MAX_TOKENS

                for role, ollama_name in model_name_map.items():
                    if not ollama_name: logger.warning(f"Ollama model name for role '{role}' not configured."); continue
                    try:
                        logger.debug(f"  Loading Ollama model '{ollama_name}' for role '{role}'...")
                        self.models[role] = ChatOllama(model=ollama_name, base_url=OLLAMA_BASE_URL, **common_params)
                        logger.info(f"  ✅ Loaded Ollama model '{role}': {ollama_name}")
                    except Exception as model_load_err:
                        logger.error(f"  ❌ Failed to load Ollama model '{ollama_name}' for role '{role}': {model_load_err}")

            elif self.provider_name == "fireworks":
                if not ChatFireworks or not FireworksEmbeddings: raise ImportError("Fireworks components not loaded")
                self.EMBEDDINGS_MODEL_NAME = FIREWORKS_EMBEDDINGS_MODEL
                self.embeddings = FireworksEmbeddings(model=self.EMBEDDINGS_MODEL_NAME, fireworks_api_key=FIREWORKS_API_KEY)

                fireworks_common_params = {"temperature": 1.2} # Default temp
                if MAX_TOKENS: fireworks_common_params["max_tokens"] = MAX_TOKENS

                # Map logical roles to Fireworks model names (add mappings as needed)
                if FIREWORKS_CHAT:
                     logger.debug(f"  Loading Fireworks model '{FIREWORKS_CHAT}' for roles default/router/general...")
                     fw_chat_model = ChatFireworks(model=FIREWORKS_CHAT, fireworks_api_key=FIREWORKS_API_KEY, **fireworks_common_params)
                     self.models["default"] = fw_chat_model
                     self.models["router"] = fw_chat_model
                     self.models["general"] = fw_chat_model
                if FIREWORKS_VISUAL_CHAT:
                     logger.debug(f"  Loading Fireworks VLM model '{FIREWORKS_VISUAL_CHAT}' for role vlm...")
                     self.models["vlm"] = ChatFireworks(model=FIREWORKS_VISUAL_CHAT, fireworks_api_key=FIREWORKS_API_KEY, **fireworks_common_params)
                # Add other roles (latex, math, code, etc.) if specific Fireworks models are defined in config

            elif self.provider_name == "llama_cpp":
                if not LLAMA_CPP_AVAILABLE: raise ImportError("llama-cpp-python is not installed or failed to import.")
                self._llama_gguf_dir = LLAMA_CPP_GGUF_DIR
                self._llama_model_map = LLAMA_CPP_MODEL_MAP

                # --- Setup Embeddings ---
                self.EMBEDDINGS_MODEL_NAME = self._llama_model_map.get("embeddings", "N/A")
                if "embeddings" in self._llama_model_map:
                     logger.info(f"Setting up llama.cpp embeddings using role 'embeddings' ({self.EMBEDDINGS_MODEL_NAME})")
                     # The wrapper handles loading the actual model when used
                     self.embeddings = LlamaCppEmbeddingsWrapper(ai_provider=self)
                else:
                     logger.error("❌ No GGUF file specified for 'embeddings' role in LLAMA_CPP_MODEL_MAP.")
                     self.embeddings = None # Disable embeddings

                # --- Setup Chat Models (Wrappers only) ---
                # We don't load models here, just create wrappers for each role
                logger.info("Creating llama.cpp chat wrappers for configured roles...")
                default_temp = 1.2 # Example default temp
                common_model_kwargs = {"temperature": default_temp, "max_tokens": MAX_TOKENS}

                for role in self._llama_model_map.keys():
                    if role != "embeddings": # Don't create chat wrapper for embedding model
                        if role in self.models: # Skip if already set (e.g., by another provider setup - shouldn't happen)
                            continue
                        logger.debug(f"  Creating wrapper for role '{role}'...")
                        self.models[role] = LlamaCppChatWrapper(
                            ai_provider=self,
                            model_role=role,
                            model_kwargs=common_model_kwargs.copy() # Pass relevant defaults
                        )
                # Assign the default chat model wrapper explicitly
                default_chat_role = MODEL_DEFAULT_CHAT_LLAMA_CPP
                if default_chat_role in self.models:
                    self.models["default"] = self.models[default_chat_role]
                    logger.info(f"Assigned role '{default_chat_role}' as the default chat model.")
                else:
                    logger.error(f"Default chat role '{default_chat_role}' specified but no GGUF mapping found!")

            else:
                raise ValueError(f"❌ Invalid provider specified in config: {self.provider_name}")

            # --- Final Check ---
            if not self.embeddings:
                 logger.error("❌ Embeddings model failed to initialize for the selected provider.")
                 # Decide: raise error or allow continuation without embeddings? Raising is safer.
                 raise ValueError("Embeddings initialization failed.")
            if not self.models.get("default"):
                 logger.error("❌ Default chat model failed to initialize for the selected provider.")
                 raise ValueError("Default chat model initialization failed.")
            if not self.models.get("vlm"):
                 logger.warning("⚠️ VLM model not configured or failed to initialize for the selected provider.")

            logger.success(f"✅ AI Provider '{self.provider_name}' setup complete.")

        except Exception as e:
            logger.error(f"❌ Error setting up AI provider {self.provider_name}: {e}")
            logger.exception("AI Provider Setup Traceback:")
            # Ensure partial setup doesn't leave inconsistent state
            self.models = {}
            self.embeddings = None
            self._loaded_llama_instance = None
            raise # Re-raise the exception to prevent app startup

        finally:
            duration = (time.time() - start_time) * 1000
            logger.debug(f"⏱️ AI Provider setup took {duration:.2f} ms")

    def get_model(self, model_role: str = "default") -> Optional[Any]:
        """
        Gets the appropriate Langchain model wrapper/instance for the requested role.
        For llama.cpp, returns the wrapper which handles dynamic loading on use.
        """
        model_instance = self.models.get(model_role)
        if not model_instance:
             logger.error(f"Model/Wrapper for role '{model_role}' not available or failed to load.")
             if model_role != "default":
                  logger.warning("Falling back to default model/wrapper.")
                  return self.models.get("default")
             else:
                  return None # Return None if even default is missing
        return model_instance

    def get_embeddings(self) -> Optional[Embeddings]:
        """Returns the configured embeddings instance."""
        return self.embeddings

    def _setup_image_generator(self):
        """Placeholder for initializing Stable Diffusion."""
        if self.provider_name == "llama_cpp" and STABLE_DIFFUSION_AVAILABLE and STABLE_DIFFUSION_CPP_MODEL_PATH:
             logger.info(f"Placeholder: Would initialize Stable Diffusion from: {STABLE_DIFFUSION_CPP_MODEL_PATH}")
             # Add loading logic here when stable-diffusion-cpp bindings are integrated
             # self.image_generator = StableDiffusionCpp(...)
             self.image_generator = None # Keep as None for now
             logger.warning("Stable Diffusion integration is currently a placeholder.")
        else:
             logger.debug("Stable Diffusion not configured or dependencies missing.")
             self.image_generator = None

    def get_image_generator(self) -> Optional[Any]:
        """Returns the image generator instance (Placeholder)."""
        if self.image_generator is None:
            logger.warning("Image generator requested but not available/initialized.")
        return self.image_generator

    def unload_llama_model_if_needed(self):
        """Explicitly unload the currently loaded llama.cpp model (if any), acquiring the lock."""
        # Check if llama_cpp provider and lock exist
        if not self._llama_model_access_lock:
             logger.debug("Unload called, but provider is not llama_cpp or lock not initialized.")
             return

        logger.info("Attempting to acquire lock for explicit model unload...")
        with self._llama_model_access_lock: # Acquire lock to safely modify shared state
            logger.info("Acquired lock for explicit model unload.")
            if self._loaded_llama_instance:
                logger.warning(f"Explicitly unloading llama.cpp model: {os.path.basename(self._loaded_gguf_path or 'Unknown')}")
                try:
                    del self._loaded_llama_instance
                except Exception as del_err:
                    logger.error(f"Error during explicit deletion of Llama instance: {del_err}")
                self._loaded_llama_instance = None
                self._loaded_gguf_path = None
                self._last_task_type = None
                gc.collect()
                logger.info("llama.cpp model unloaded via explicit call.")
            else:
                logger.info("Explicit unload called, but no llama.cpp model was loaded.")
            logger.info("Releasing lock after explicit unload attempt.")
        # Lock released here

# --- Optional: Add a shutdown hook specific to AIProvider for llama.cpp ---
# This ensures the model is unloaded even if the main DB hook runs first/fails.
# Note: atexit runs hooks in reverse order of registration.
def _ai_provider_shutdown():
    # Need a way to access the global AIProvider instance created in app.py
    # This is tricky. A better pattern might be for app.py to explicitly call
    # a shutdown method on the provider instance it holds.
    # For now, we'll assume a global `ai_provider_instance` might exist (set by app.py).
    global ai_provider_instance # Assume app.py sets this global
    if 'ai_provider_instance' in globals() and ai_provider_instance:
        logger.info("Running AI Provider shutdown hook...")
        ai_provider_instance.unload_llama_model_if_needed()
    else:
        logger.debug("AI Provider shutdown hook skipped: No global instance found.")

# atexit.register(_ai_provider_shutdown)
# --- End Optional Shutdown Hook ---

# --- Global Instance Placeholder ---
# This global would be set by app.py after initialization
ai_provider_instance: Optional[AIProvider] = None