# ai_provider.py
import os
import sys
import time
import threading
import gc # For garbage collection
from typing import Dict, Any, Optional, List, Iterator
from loguru import logger

# --- Langchain Imports ---
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ChatMessage
from langchain_core.outputs import ChatResult, ChatGeneration, GenerationChunk, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings

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
    from config import * # Includes PROVIDER, model names, paths, etc.
except ImportError:
    logger.critical("❌ Failed to import config.py in ai_provider.py!")
    sys.exit("AIProvider cannot function without config.")


# === llama-cpp-python Langchain Wrappers ===

# --- Chat Model Wrapper ---
class LlamaCppChatWrapper(SimpleChatModel):
    """
    Langchain chat model wrapper for a dynamically loaded llama-cpp-python instance
    managed by AIProvider.
    """
    ai_provider: 'AIProvider' # Forward reference
    model_role: str # Logical role (e.g., "router", "vlm") to request from provider
    model_kwargs: Dict[str, Any] # Kwargs for Llama.create_chat_completion

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Sync generation (Not recommended for llama.cpp, use _stream)."""
        # Streaming is generally preferred for llama.cpp
        # Combine chunks from the streaming method for sync behavior
        full_response = ""
        for chunk in self._stream(messages, stop, run_manager, **kwargs):
            full_response += chunk.content
        return full_response

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Streaming generation."""
        llm_instance = self.ai_provider._get_loaded_llama_instance(self.model_role)
        if not llm_instance:
            err_msg = f"LLAMA_CPP_ERROR: Model for role '{self.model_role}' could not be loaded."
            logger.error(err_msg)
            yield ChatGenerationChunk(message=AIMessage(content=err_msg))
            return

        # --- Format messages for llama_cpp ---
        # Needs conversion from Langchain BaseMessage to llama_cpp dict format
        formatted_messages = []
        for msg in messages:
            role = "user" # Default
            if isinstance(msg, HumanMessage): role = "user"
            elif isinstance(msg, AIMessage): role = "assistant"
            elif isinstance(msg, SystemMessage): role = "system"
            elif isinstance(msg, ChatMessage): role = msg.role # Use role if specified

            content = ""
            # Handle complex content (text + image) for VLM models
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                # If VLM role, format for llama_cpp multimodal
                if self.model_role == "vlm" and hasattr(llm_instance, "tokenize"):
                     logger.debug(f"Formatting multimodal input for role '{self.model_role}'...")
                     # llama_cpp expects list of dicts: {"type": "text", "text": ...} or {"type": "image_url", "image_url": {"url": "data:..."}}
                     # We assume image_content_part was correctly added to the HumanMessage content list in app.py
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
                                 logger.debug("Image part added to llama.cpp input.")
                             else:
                                 logger.warning(f"Skipping unsupported image_url format: {img_url_data[:50]}...")
                                 content_list.append({"type": "text", "text": "[Unsupported Image Placeholder]"})
                         else:
                              # Fallback for unexpected content types
                              logger.warning(f"Unexpected item type in message content list: {type(item)}")
                              content_list.append({"type": "text", "text": str(item)}) # Convert to string as fallback

                     # Only pass list content if an image was actually included
                     if has_image:
                          formatted_messages.append({"role": role, "content": content_list})
                     else: # If no image found despite list format, combine text
                          text_content = " ".join([c.get("text","") for c in content_list if c.get("type")=="text"])
                          formatted_messages.append({"role": role, "content": text_content})
                     continue # Skip default content assignment below for this message
                else:
                    # Non-VLM or model doesn't support multimodal, just combine text parts
                    logger.warning(f"Model role '{self.model_role}' received list content but isn't VLM/multimodal. Combining text.")
                    content = " ".join([item.get("text", "") for item in msg.content if isinstance(item, dict) and item.get("type") == "text"])
            else:
                 content = str(msg.content) # Fallback

            formatted_messages.append({"role": role, "content": content})
        # --- End message formatting ---

        logger.debug(f"llama_cpp invoking role '{self.model_role}' with {len(formatted_messages)} messages.")
        try:
            streamer = llm_instance.create_chat_completion(
                messages=formatted_messages,
                max_tokens=self.model_kwargs.get("max_tokens", MAX_TOKENS), # Use configured max_tokens
                temperature=self.model_kwargs.get("temperature", 1.2), # Use configured temperature
                top_p=self.model_kwargs.get("top_p", 0.95),
                stop=stop,
                stream=True, # Force stream=True
                **kwargs # Pass any additional kwargs
            )

            for chunk in streamer:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta and delta["content"] is not None:
                    content_chunk = delta["content"]
                    # Yield Langchain chunk format
                    yield ChatGenerationChunk(message=AIMessage(content=content_chunk))
                    # Optional: Callback manager usage
                    if run_manager:
                        run_manager.on_llm_new_token(content_chunk)
                # Handle finish reason if present in the last chunk
                # finish_reason = chunk["choices"][0].get("finish_reason")
                # if finish_reason:
                #     # Need to check how llama-cpp signals finish in stream chunk
                #     # For now, assume Langchain handles the overall finish reason
                #     pass

        except Exception as e:
            logger.error(f"LLAMA_CPP_ERROR: Error during streaming for role '{self.model_role}': {e}")
            logger.exception("Llama.cpp streaming traceback:")
            yield ChatGenerationChunk(message=AIMessage(content=f"[LLAMA_CPP STREAM ERROR: {e}]"))

    @property
    def _llm_type(self) -> str:
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
        llm_instance = self.ai_provider._get_loaded_llama_instance(self.model_role)
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
        llm_instance = self.ai_provider._get_loaded_llama_instance(self.model_role)
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
        self.models: Dict[str, Any] = {} # Cache for model *wrappers* or instances {role: instance}
        self.embeddings: Optional[Embeddings] = None
        self.EMBEDDINGS_MODEL_NAME: Optional[str] = None # Store configured name
        self.image_generator: Any = None # Placeholder for Stable Diffusion

        # --- llama.cpp specific state ---
        self._loaded_gguf_path: Optional[str] = None
        self._loaded_llama_instance: Optional[llama_cpp.Llama] = None
        self._model_load_lock = threading.Lock() if LLAMA_CPP_AVAILABLE else None
        self._llama_model_map: Dict[str, str] = {} # {role: filename.gguf}
        self._llama_gguf_dir: Optional[str] = None

        logger.info(f"🤖 Initializing AI Provider: {self.provider_name}")
        self._validate_config()
        self.setup_provider()
        self._setup_image_generator() # Placeholder call

    def _validate_config(self):
        """Basic checks for required config based on provider."""
        if self.provider_name == "fireworks" and not FIREWORKS_API_KEY:
            logger.error("🔥 PROVIDER=fireworks requires FIREWORKS_API_KEY in config or .env")
            # Consider exiting or disabling provider
        if self.provider_name == "llama_cpp":
            if not LLAMA_CPP_AVAILABLE:
                 logger.error("❌ PROVIDER=llama_cpp selected but llama-cpp-python is not installed.")
                 sys.exit("Missing llama-cpp-python dependency.")
            if not os.path.isdir(LLAMA_CPP_GGUF_DIR):
                 logger.error(f"❌ PROVIDER=llama_cpp requires GGUF directory at: {LLAMA_CPP_GGUF_DIR}")
                 sys.exit(f"Missing GGUF directory: {LLAMA_CPP_GGUF_DIR}")
            # Check if model files in map actually exist (optional, adds startup time)
            # for role, fname in LLAMA_CPP_MODEL_MAP.items():
            #     fpath = os.path.join(LLAMA_CPP_GGUF_DIR, fname)
            #     if not os.path.isfile(fpath):
            #          logger.warning(f"⚠️ Llama.cpp model file for role '{role}' not found: {fpath}")

    def _load_llama_model(self, required_gguf_path: str) -> Optional[llama_cpp.Llama]:
        """
        Loads or returns the cached llama_cpp.Llama instance.
        Handles unloading previous model if necessary. Thread-safe.
        """
        if not LLAMA_CPP_AVAILABLE or self._model_load_lock is None:
            logger.error("Cannot load llama.cpp model: Library not available or lock not initialized.")
            return None

        with self._model_load_lock:
            # 1. Check if already loaded
            if self._loaded_llama_instance and self._loaded_gguf_path == required_gguf_path:
                logger.debug(f"Using cached llama.cpp instance: {os.path.basename(required_gguf_path)}")
                return self._loaded_llama_instance

            # 2. Unload existing model if different
            if self._loaded_llama_instance:
                logger.warning(f"Unloading previous llama.cpp model: {os.path.basename(self._loaded_gguf_path or 'Unknown')}")
                # Simply deleting the reference might be enough for llama-cpp to release VRAM
                # Explicit cleanup steps if needed:
                # if hasattr(self._loaded_llama_instance, 'close'): # Or similar method? Check llama-cpp docs
                #     self._loaded_llama_instance.close()
                del self._loaded_llama_instance
                self._loaded_llama_instance = None
                self._loaded_gguf_path = None
                gc.collect() # Hint Python's garbage collector
                logger.info("Previous llama.cpp model unloaded.")
                # Optional: Add a small delay if GPU needs time to release memory
                # time.sleep(0.5)

            # 3. Load the new model
            logger.info(f"Loading llama.cpp model: {os.path.basename(required_gguf_path)}...")
            logger.info(f"  >> Path: {required_gguf_path}")
            logger.info(f"  >> GPU Layers: {LLAMA_CPP_N_GPU_LAYERS}")
            logger.info(f"  >> Context Size: {LLAMA_CPP_N_CTX}")

            if not os.path.isfile(required_gguf_path):
                 logger.error(f"LLAMA_CPP_ERROR: Model file not found: {required_gguf_path}")
                 return None

            load_start_time = time.monotonic()
            try:
                # --- Simplified Loading ---
                # Let llama-cpp-python auto-detect multimodal capabilities from the GGUF metadata
                # Removed the explicit Llava15ChatHandler initialization based on role.
                # llama.cpp will use internal handlers if the GGUF indicates multimodal capability.
                # If issues arise with specific models, explicit handlers might be needed again.

                # Find the role for logging/embedding check
                role_for_path = None
                for r, f in self._llama_model_map.items():
                    if os.path.join(self._llama_gguf_dir or "", f) == required_gguf_path:
                        role_for_path = r
                        break

                logger.info(f"Initializing Llama for role '{role_for_path}' (auto-detecting multimodal)...")

                self._loaded_llama_instance = llama_cpp.Llama(
                    model_path=required_gguf_path,
                    n_gpu_layers=LLAMA_CPP_N_GPU_LAYERS,
                    n_ctx=LLAMA_CPP_N_CTX,
                    embedding=(role_for_path == "embeddings"), # Enable embedding mode only for the embeddings model
                    verbose=LLAMA_CPP_VERBOSE,
                    # chat_handler=None, # Let llama.cpp handle chat format internally if possible
                    # chat_format="auto", # Or explicitly set if needed, check llama-cpp-python docs
                )
                self._loaded_gguf_path = required_gguf_path
                load_duration = time.monotonic() - load_start_time
                logger.success(f"✅ Loaded '{os.path.basename(required_gguf_path)}' in {load_duration:.2f}s.")
                # Optionally log detected model type if available
                # logger.info(f"   Model Type Detected by llama.cpp: {self._loaded_llama_instance.model_type()}" ) # Check if method exists
                return self._loaded_llama_instance

            except Exception as e:
                logger.error(f"LLAMA_CPP_ERROR: Failed to load model {required_gguf_path}: {e}")
                logger.exception("Llama.cpp loading traceback:")
                self._loaded_llama_instance = None
                self._loaded_gguf_path = None
                return None

    def _get_loaded_llama_instance(self, model_role: str) -> Optional[llama_cpp.Llama]:
        """Ensures the correct llama.cpp model for the role is loaded and returns it."""
        if self.provider_name != "llama_cpp":
            logger.error("Attempted to get llama.cpp instance when provider is not llama_cpp.")
            return None
        if not self._llama_gguf_dir:
            logger.error("Llama.cpp GGUF directory not set.")
            return None

        gguf_filename = self._llama_model_map.get(model_role)
        if not gguf_filename:
            logger.error(f"No GGUF file configured for llama.cpp role: '{model_role}'")
            return None

        required_path = os.path.join(self._llama_gguf_dir, gguf_filename)
        return self._load_llama_model(required_path)

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
        """Explicitly unload the currently loaded llama.cpp model (if any)."""
        if self.provider_name == "llama_cpp" and self._model_load_lock:
            with self._model_load_lock:
                if self._loaded_llama_instance:
                    logger.warning(f"Explicitly unloading llama.cpp model: {os.path.basename(self._loaded_gguf_path or 'Unknown')}")
                    del self._loaded_llama_instance
                    self._loaded_llama_instance = None
                    self._loaded_gguf_path = None
                    gc.collect()
                    logger.info("llama.cpp model unloaded via explicit call.")
                else:
                    logger.info("Explicit unload called, but no llama.cpp model was loaded.")
        else:
            logger.debug("Unload called, but provider is not llama_cpp or not initialized.")

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