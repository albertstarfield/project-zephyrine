# ai_provider.py
import os
import sys
import time
import threading
import gc # For garbage collection
from typing import Dict, Any, Optional, List, Iterator
from loguru import logger
import json        # Added for worker communication
import subprocess  # Added for worker management
import shlex       # <<< --- ADD THIS LINE --- >>>

# --- NEW: Import the custom lock ---

from priority_lock import ELP0, ELP1 # Ensure these are imported
interruption_error_marker = "Worker task interrupted by higher priority request" # Define consistently


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


try:
    from priority_lock import PriorityQuotaLock, ELP0, ELP1
except ImportError:
    logger.critical("❌ Failed to import PriorityQuotaLock. Priority locking disabled.")
    # Fallback to standard lock to allow basic functionality? Or exit?
    # For now, define dummies to prevent crashing later code, but log error.
    PriorityQuotaLock = threading.Lock # Fallback to standard lock (no priority)
    ELP0 = 0
    ELP1 = 1
    # sys.exit("Priority Lock implementation missing") # Optionally exit

# --- Local Imports ---
try:
    # Import all config variables
    from config import * # Includes PROVIDER, model names, paths, MAX_TOKENS etc.
except ImportError:
    logger.critical("❌ Failed to import config.py in ai_provider.py!")
    sys.exit("AIProvider cannot function without config.")

class TaskInterruptedException(Exception):
    """Custom exception raised when an ELP0 task is interrupted by ELP1."""
    pass

# === llama-cpp-python Langchain Wrappers ===

# --- Chat Model Wrapper ---
class LlamaCppChatWrapper(SimpleChatModel):
    ai_provider: 'AIProvider'
    model_role: str
    model_kwargs: Dict[str, Any]

    def _call(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any,
    ) -> str:
        """
        Handles non-streaming chat completion requests by delegating to the AIProvider's
        worker execution method (_execute_in_worker), passing priority.
        """
        # Use specific logger from the instance if available, otherwise default logger
        provider_logger = getattr(self.ai_provider, 'logger', logger) # Access logger via provider instance
        wrapper_log_prefix = f"LlamaCppChatWrapper(Role:{self.model_role})"
        provider_logger.debug(f"{wrapper_log_prefix}: Received non-streaming call request.")

        # --- Extract priority, default to ELP0 if not specified ---
        # Remove 'priority' from kwargs so it's not passed directly to the underlying
        # llama.cpp model which doesn't understand it.
        priority = kwargs.pop('priority', ELP0)
        provider_logger.debug(f"{wrapper_log_prefix}: Determined priority for call: ELP{priority}")
        # --- End Extraction ---

        # Combine model-specific kwargs passed during call with defaults set on the wrapper instance
        final_model_kwargs = {**self.model_kwargs, **kwargs}
        if stop:
             final_model_kwargs["stop"] = stop # Add stop sequences if provided during the call
             provider_logger.trace(f"{wrapper_log_prefix}: Stop sequences provided: {stop}")
        provider_logger.trace(f"{wrapper_log_prefix}: Final model kwargs being sent to worker: {final_model_kwargs}")

        # Prepare the payload for the worker process
        # Format messages into the structure llama.cpp expects
        formatted_messages = self._format_messages_for_llama_cpp(messages)
        request_payload = {
            "messages": formatted_messages,
            "kwargs": final_model_kwargs # Pass the combined kwargs for the model itself inside the worker
        }

        try:
            # Delegate to the provider's worker execution method, passing the determined priority
            provider_logger.debug(f"{wrapper_log_prefix}: Delegating to _execute_in_worker with priority ELP{priority}...")
            start_worker_call = time.monotonic()
            worker_result = self.ai_provider._execute_in_worker(
                model_role=self.model_role,
                task_type="chat",
                request_data=request_payload,
                priority=priority # Pass the extracted priority here
            )
            worker_duration = time.monotonic() - start_worker_call
            provider_logger.debug(f"{wrapper_log_prefix}: Received result from worker ({worker_duration:.3f}s).")
            # provider_logger.trace(f"{wrapper_log_prefix}: Worker Result Raw: {worker_result}") # Uncomment for detailed debugging

            # --- Process result dictionary from worker ---
            if worker_result and isinstance(worker_result, dict):
                if "error" in worker_result:
                    # Handle errors reported *by* the worker execution framework or the worker itself
                    error_msg = f"LLAMA_CPP_WORKER_ERROR ({self.model_role}): {worker_result['error']}"
                    provider_logger.error(error_msg)
                    # Return the error message clearly formatted for potential downstream handling
                    return f"[{error_msg}]" # Square brackets help identify it as an error string
                elif "result" in worker_result:
                    # Worker execution seems to have succeeded, process the actual completion result
                    completion = worker_result["result"]
                    # Check if the completion structure is as expected (standard OpenAI-like dict)
                    if (completion and isinstance(completion, dict) and
                            'choices' in completion and isinstance(completion['choices'], list) and
                            completion['choices']): # Ensure 'choices' list is not empty

                        first_choice = completion['choices'][0]
                        # Further validation of the first choice structure
                        if (isinstance(first_choice, dict) and 'message' in first_choice and
                                isinstance(first_choice['message'], dict) and
                                'content' in first_choice['message']):

                            response_content = first_choice['message']['content']
                            # Handle potential None or empty content gracefully
                            if response_content is None:
                                 provider_logger.warning(f"{wrapper_log_prefix}: Worker returned None content in message. Treating as empty.")
                                 response_content = "" # Treat None as empty string
                            elif not isinstance(response_content, str):
                                 provider_logger.warning(f"{wrapper_log_prefix}: Worker returned non-string content (type: {type(response_content)}). Converting to string.")
                                 response_content = str(response_content)

                            provider_logger.debug(f"{wrapper_log_prefix}: Successfully extracted content (len:{len(response_content)}).")
                            # Return the actual string content - this is the successful path
                            return response_content
                        else:
                            # Log error if the message structure within the choice is invalid
                            provider_logger.error(f"{wrapper_log_prefix}: Worker returned unexpected 'message' structure in first choice: {first_choice}")
                            return "[LLAMA_CPP_WORKER_RESPONSE_ERROR: Invalid message structure in choice]"
                    else:
                        # Log error if the 'choices' list is missing, not a list, or empty
                        provider_logger.error(f"{wrapper_log_prefix}: Worker returned unexpected or empty 'choices' structure: {completion}")
                        return "[LLAMA_CPP_WORKER_RESPONSE_ERROR: Invalid/empty choices structure]"
                else:
                     # Worker returned a dictionary, but it lacked the expected 'error' or 'result' keys
                     provider_logger.error(f"{wrapper_log_prefix}: Worker returned unknown dictionary structure: {worker_result}")
                     return "[LLAMA_CPP_WORKER_RESPONSE_ERROR: Unknown dictionary structure]"
            else:
                # _execute_in_worker returned None or an unexpected data type (not a dict)
                provider_logger.error(f"{wrapper_log_prefix}: AIProvider._execute_in_worker failed or returned invalid data type: {type(worker_result)}")
                return "[LLAMA_CPP_PROVIDER_ERROR: Worker execution failed or returned invalid type]"

        except Exception as e:
            # Catch any unexpected exceptions during the wrapper's execution itself (e.g., errors formatting payload)
            provider_logger.error(f"{wrapper_log_prefix}: Unexpected error during _call delegation: {e}")
            provider_logger.exception(f"{wrapper_log_prefix} Chat Delegation Traceback:") # Log full traceback
            # Return a formatted error string indicating a wrapper-level issue
            return f"[LLAMA_CPP_WRAPPER_ERROR: {type(e).__name__} - {e}]"

    def _stream(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # NOTE: Streaming via a separate process is more complex.
        # For now, we'll make the stream call non-streaming via _call
        # OR yield an error indicating it's not supported in this mode.
        logger.warning(f"Streaming requested for role '{self.model_role}', but process isolation makes this complex. Falling back to non-streaming or error.")
        # Option 1: Fallback to non-streaming result (simpler)
        # full_response = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        # yield ChatGenerationChunk(message=AIMessageChunk(content=full_response))

        # Option 2: Explicitly yield error (clearer that streaming isn't happening)
        yield ChatGenerationChunk(message=AIMessageChunk(content="[LLAMA_CPP_ERROR: Streaming not supported with process isolation mode]"))

    # Keep formatter, it's still useful
    def _format_messages_for_llama_cpp(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        # (Keep the _format_messages_for_llama_cpp logic from previous version)
        # ... it correctly formats the message list which is needed by the worker ...
        formatted_messages = []
        # --- VLM Handling ---
        is_vlm_request = False
        if self.model_role == "vlm": # Check role only, instance check removed
            for msg in messages:
                 if isinstance(msg.content, list):
                    if any(isinstance(item, dict) and item.get("type") == "image_url" for item in msg.content):
                        is_vlm_request = True; break
            if is_vlm_request: logger.debug(f"VLM formatting check passed for role '{self.model_role}'.")

        # --- Main Loop ---
        for msg in messages:
            role = "user";
            if isinstance(msg, HumanMessage): role = "user"
            elif isinstance(msg, AIMessage): role = "assistant"
            elif isinstance(msg, SystemMessage): role = "system"
            elif isinstance(msg, ChatMessage): role = msg.role

            if isinstance(msg.content, str):
                formatted_messages.append({"role": role, "content": msg.content})
            elif isinstance(msg.content, list) and is_vlm_request:
                content_list = []; has_image_in_msg = False
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text": content_list.append({"type": "text", "text": item.get("text", "")})
                    elif isinstance(item, dict) and item.get("type") == "image_url":
                        img_url_data = item.get("image_url", {}).get("url", "")
                        if img_url_data.startswith("data:image"): content_list.append({"type": "image_url", "image_url": {"url": img_url_data}}); has_image_in_msg = True; logger.trace("VLM Image formatted.")
                        else: logger.warning(f"Skipping unsupported image_url: {img_url_data[:50]}..."); content_list.append({"type": "text", "text": "[Img Err]"})
                    else: logger.warning(f"Unexpected VLM item type: {type(item)}"); content_list.append({"type": "text", "text": str(item)})
                if has_image_in_msg: formatted_messages.append({"role": role, "content": content_list})
                else: text_content = " ".join([c.get("text","") for c in content_list if c.get("type")=="text"]); formatted_messages.append({"role": role, "content": text_content})
            elif isinstance(msg.content, list) and not is_vlm_request:
                 text_content = " ".join([item.get("text", "") for item in msg.content if isinstance(item, dict) and item.get("type") == "text"])
                 formatted_messages.append({"role": role, "content": text_content})
            else: formatted_messages.append({"role": role, "content": str(msg.content)})
        return formatted_messages


    @property
    def _llm_type(self) -> str:
        return "llama_cpp_chat_worker_wrapper" # New type name


# --- Embeddings Wrapper ---
# Placeholder/Modified Embeddings Wrapper
class LlamaCppEmbeddingsWrapper(Embeddings):
    ai_provider: 'AIProvider'
    model_role: str = "embeddings"

    def __init__(self, ai_provider: 'AIProvider'):
        super().__init__()
        self.ai_provider = ai_provider
    
    def embed_query(self, text: str, priority: int = ELP0) -> List[float]:
        """Embeds a single query using the specified priority."""
        # Embed single query as a list of one by calling _embed_texts with priority
        results = self._embed_texts([text], priority=priority)
        if results:
            # Return the first (and only) embedding vector
            return results[0]
        else:
            # This case should ideally be handled by exceptions in _embed_texts
            # Use specific logger from the instance if available, otherwise default logger
            provider_logger = getattr(self.ai_provider, 'logger', logger)
            provider_logger.error("Embedding query returned empty list unexpectedly (should have raised error).")

            raise RuntimeError("LLAMA_CPP_PROVIDER_ERROR: Embedding query failed unexpectedly")

    def embed_documents(self, texts: List[str], priority: int = ELP0) -> List[List[float]]:
        """Embeds a list of documents using the specified priority."""
        # Directly call _embed_texts, passing the priority along
        return self._embed_texts(texts, priority=priority)

    def _embed_texts(self, texts: List[str], priority: int = ELP0) -> List[List[float]]:
        """
        Internal helper to call worker for embedding, using specified priority.
        Handles TaskInterruptedException raised by the worker execution framework.
        Validates the structure of the returned embeddings.
        """
        # Use specific logger from the instance if available, otherwise default logger
        # Ensure self.ai_provider exists and has a logger attribute or fallback
        provider_logger = getattr(self.ai_provider, 'logger', logger)
        log_prefix = f"EmbedWrapper|ELP{priority}"
        provider_logger.debug(f"{log_prefix}: Delegating embedding for {len(texts)} texts.")

        # Prepare the request payload for the worker
        request_payload = {"texts": texts}

        try:
            # --- Pass priority to _execute_in_worker ---
            # Call the provider's method to run the task in the isolated worker process
            provider_logger.trace(f"{log_prefix}: Calling AIProvider._execute_in_worker...")
            worker_result = self.ai_provider._execute_in_worker(
                model_role=self.model_role, # Role is "embeddings" for this wrapper
                task_type="embedding",
                request_data=request_payload,
                priority=priority # Pass the received priority
            )
            provider_logger.trace(f"{log_prefix}: Received result from worker.")

            # --- Process result dictionary from worker ---
            if worker_result and isinstance(worker_result, dict):
                if "error" in worker_result:
                    # --- Handle Interruption ---
                    # Check if the error message indicates interruption
                    if interruption_error_marker in worker_result["error"]:
                         provider_logger.warning(f"🚦 {log_prefix}: Embedding INTERRUPTED: {worker_result['error']}")
                         # Raise the specific exception to signal interruption upwards
                         raise TaskInterruptedException(worker_result["error"])
                    # --- Handle Other Worker Errors ---
                    else:
                         error_msg = f"LLAMA_CPP_WORKER_ERROR (Embeddings): {worker_result['error']}"
                         provider_logger.error(error_msg)
                         # Raise a standard runtime error for embedding failures
                         raise RuntimeError(error_msg)

                elif "result" in worker_result and isinstance(worker_result["result"], list):
                    # --- Validate Embedding Structure ---
                    # Check if the result is a list of lists, where inner lists contain numbers (float or int)
                    if all(isinstance(emb, list) and all(isinstance(num, (float, int)) for num in emb) for emb in worker_result["result"]):
                         # Convert potential ints to floats for consistency before returning
                         provider_logger.trace(f"{log_prefix}: Embedding structure valid. Converting inner numbers to float...")
                         float_embeddings = [[float(num) for num in emb] for emb in worker_result["result"]]
                         provider_logger.debug(f"{log_prefix}: Embedding successful. Returning {len(float_embeddings)} vectors.")
                         return float_embeddings
                    else:
                        # Log error if the structure is invalid (e.g., not list of lists, or inner items aren't numbers)
                        provider_logger.error(f"{log_prefix}: Worker returned invalid embedding structure. Type: {type(worker_result['result'])}. Example Element Type: {type(worker_result['result'][0]) if worker_result['result'] else 'N/A'}")
                        raise RuntimeError("LLAMA_CPP_WORKER_RESPONSE_ERROR: Invalid embedding structure returned")

                else:
                    # Worker dictionary didn't contain 'error' or 'result' keys
                    provider_logger.error(f"{log_prefix}: Worker returned unknown dictionary structure: {worker_result}")
                    raise RuntimeError("LLAMA_CPP_WORKER_RESPONSE_ERROR: Unknown dictionary structure")
            else:
                # _execute_in_worker returned None or a non-dictionary type
                provider_logger.error(f"{log_prefix}: AIProvider._execute_in_worker failed or returned invalid data type: {type(worker_result)}")
                raise RuntimeError("LLAMA_CPP_PROVIDER_ERROR: Worker execution failed or returned invalid type")

        except TaskInterruptedException:
            # Re-raise interruption exception directly if caught from _execute_in_worker
            # or from the error check above
            provider_logger.warning(f"🚦 {log_prefix}: Propagating TaskInterruptedException upwards.")
            raise
        except Exception as e:
            # Catch other exceptions (RuntimeError from checks, or unexpected errors)
            provider_logger.error(f"{log_prefix}: Error during LlamaCppEmbeddingsWrapper delegation: {e}")
            # Avoid re-logging RuntimeError or TaskInterruptedException tracebacks if they were already logged/handled
            if not isinstance(e, (RuntimeError, TaskInterruptedException)):
                 provider_logger.exception(f"{log_prefix} Embedding Delegation Traceback:")
            # Re-raise the exception so the caller knows the embedding failed
            raise e

# === AI Provider Class ===
class AIProvider:
    def __init__(self, provider_name):
        # ... (Initialization as before, REMOVE _loaded_*, _last_task_type ) ...
        self.provider_name = provider_name.lower()
        self.models: Dict[str, Any] = {} # Will store WRAPPERS now
        self.embeddings: Optional[Embeddings] = None # Will store WRAPPER
        self.EMBEDDINGS_MODEL_NAME: Optional[str] = None
        self.image_generator: Any = None

        # --- llama.cpp specific state ---
        # REMOVED: self._loaded_gguf_path, self._loaded_llama_instance, self._last_task_type
        self._llama_model_access_lock = threading.Lock() if LLAMA_CPP_AVAILABLE and self.provider_name == "llama_cpp" else None
        if LLAMA_CPP_AVAILABLE and self.provider_name == "llama_cpp":
             # Ensure PriorityQuotaLock didn't fallback to threading.Lock due to import error
             if PriorityQuotaLock is not threading.Lock:
                 self._priority_quota_lock = PriorityQuotaLock()
                 logger.info("   🔑 Initialized PriorityQuotaLock for llama.cpp WORKER access.")
             else:
                  logger.error("   ❌ PriorityQuotaLock import failed, falling back to standard Lock (no priority).")
                  self._priority_quota_lock = threading.Lock() # Use standard lock as fallback
        # --- End lock replacement ---
        self._llama_model_map: Dict[str, str] = {}
        self._llama_gguf_dir: Optional[str] = None
        # Store python executable path for worker
        self._python_executable = sys.executable # Assumes provider runs in same env as worker needs

        logger.info(f"🤖 Initializing AI Provider: {self.provider_name} (Worker Process Mode)")
        # ... (rest of init: lock logging, validation, setup) ...
        if self.provider_name == "llama_cpp" and self._llama_model_access_lock: logger.info("   🔑 Initialized threading.Lock for llama.cpp WORKER access.")
        elif self.provider_name == "llama_cpp": logger.error("   ❌ Failed to initialize threading.Lock for llama.cpp WORKER access.")
        self._validate_config()
        self.setup_provider()
        self._setup_image_generator()

    def _validate_config(self):
        # (Validation remains largely the same, ensuring distinct embedding model etc.)
        # ...
        if self.provider_name == "llama_cpp":
            # ... (Check availability, GGUF dir) ...
            # --- Check for distinct embedding model (Still important) ---
            embedding_file = LLAMA_CPP_MODEL_MAP.get("embeddings")
            if not embedding_file: logger.error("❌ llama_cpp requires 'embeddings' entry in LLAMA_CPP_MODEL_MAP."); sys.exit("Missing embed config.")
            for role, fname in LLAMA_CPP_MODEL_MAP.items():
                if role != "embeddings" and fname == embedding_file:
                     logger.error(f"❌ llama_cpp: Embedding model '{embedding_file}' CANNOT be reused for chat role '{role}'."); sys.exit("Embed/Chat model reuse forbidden.")
            # --- Ensure worker script exists ---
            worker_script_path = os.path.join(os.path.dirname(__file__), "llama_worker.py")
            if not os.path.isfile(worker_script_path):
                logger.error(f"❌ Llama.cpp worker script not found at: {worker_script_path}")
                sys.exit("Missing llama_worker.py")


    # <<< --- REMOVE _load_llama_model and _get_loaded_llama_instance --- >>>

    # <<< --- NEW: Worker Execution Method --- >>>
    def _execute_in_worker(self, model_role: str, task_type: str, request_data: Dict[str, Any], priority: int = ELP0) -> Optional[Dict[str, Any]]:
        """
        Starts llama_worker.py, sends request, gets response. Handles crashes.
        Protected by the PRIORITY QUOTA lock. ELP1 can interrupt ELP0.
        """
        # Use specific logger from the instance if available, otherwise default logger
        provider_logger = getattr(self, 'logger', logger) # Use self.logger if it exists
        worker_log_prefix = f"WORKER_MGR(ELP{priority}|{model_role}/{task_type})"
        provider_logger.debug(f"{worker_log_prefix}: Attempting to execute task in worker.")

        if not self._priority_quota_lock: # Check if the lock instance exists
             provider_logger.error(f"{worker_log_prefix}: Priority Lock not initialized!")
             return {"error": "Provider lock not initialized"}
        if self.provider_name != "llama_cpp":
             provider_logger.error(f"{worker_log_prefix}: Attempted worker execution outside llama_cpp provider.")
             return {"error": "Worker execution only for llama_cpp provider"}

        # --- Get Model Path ---
        gguf_filename = self._llama_model_map.get(model_role)
        if not gguf_filename:
            provider_logger.error(f"{worker_log_prefix}: No GGUF file configured for role '{model_role}'")
            return {"error": f"No GGUF config for role {model_role}"}
        model_path = os.path.join(self._llama_gguf_dir or "", gguf_filename)
        if not os.path.isfile(model_path):
            provider_logger.error(f"{worker_log_prefix}: Model file not found: {model_path}")
            return {"error": f"Model file not found: {os.path.basename(model_path)}"}

        # --- Acquire Priority Lock ---
        lock_acquired = False
        worker_process = None # Define worker_process before the try block
        start_lock_wait = time.monotonic()
        provider_logger.debug(f"{worker_log_prefix}: Acquiring worker execution lock (Priority: ELP{priority})...")

        # Use a timeout for acquisition? For now, no timeout.
        lock_acquired = self._priority_quota_lock.acquire(priority=priority, timeout=None)
        lock_wait_duration = time.monotonic() - start_lock_wait

        if lock_acquired:
            provider_logger.info(f"{worker_log_prefix}: Lock acquired (waited {lock_wait_duration:.2f}s). Starting worker process.")
            try:
                # --- Determine Context Size Based on Role ---
                if model_role == "embeddings":
                    n_ctx_to_use = 512 # Specific override for embedding model
                    provider_logger.info(f"{worker_log_prefix}: Using specific context size {n_ctx_to_use} for embedding model role.")
                else:
                    n_ctx_to_use = LLAMA_CPP_N_CTX # Use configured default for chat models
                    provider_logger.debug(f"{worker_log_prefix}: Using configured context size {n_ctx_to_use} for role '{model_role}'.")

                # --- Prepare Worker Command ---
                worker_script_path = os.path.join(os.path.dirname(__file__), "llama_worker.py")
                command = [
                    self._python_executable, # Use Python from the provider's env
                    worker_script_path,
                    "--model-path", model_path,
                    "--task-type", task_type,
                    "--n-gpu-layers", str(LLAMA_CPP_N_GPU_LAYERS),
                    "--n-ctx", str(n_ctx_to_use), # Use determined context size
                ]
                if LLAMA_CPP_VERBOSE: command.append("--verbose")
                # Add chat format only for chat tasks
                if task_type == "chat":
                    chat_fmt = LLAMA_CPP_MODEL_MAP.get(f"{model_role}_chat_format", "chatml") # Get specific format or default
                    if chat_fmt: command.extend(["--chat-format", chat_fmt])

                provider_logger.debug(f"{worker_log_prefix}: Worker command: {' '.join(shlex.quote(c) for c in command)}")

                # --- Start Worker Process ---
                start_time = time.monotonic()
                worker_process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True, # Use text mode for JSON communication
                    encoding='utf-8',
                    errors='replace'
                )

                # --- Associate Process with Lock (if ELP0) ---
                if priority == ELP0:
                    self._priority_quota_lock.set_holder_process(worker_process)
                # --- End Association ---

                # --- Send Request Data to Worker ---
                input_json = json.dumps(request_data)
                provider_logger.debug(f"{worker_log_prefix}: Sending input JSON (len={len(input_json)}) to worker stdin...")
                stdout_data, stderr_data = "", ""
                comm_exception = None
                try:
                    # Communicate with the worker process
                    stdout_data, stderr_data = worker_process.communicate(input=input_json, timeout=300) # 5 min timeout
                    provider_logger.debug(f"{worker_log_prefix}: Worker communicate() finished.")
                except subprocess.TimeoutExpired as timeout_err:
                    provider_logger.error(f"{worker_log_prefix}: Worker process timed out after 300s.")
                    worker_process.kill() # Ensure it's killed
                    stdout_data, stderr_data = worker_process.communicate() # Try to get final output
                    provider_logger.error(f"{worker_log_prefix}: Worker timed out. Stderr: {stderr_data.strip()}")
                    comm_exception = timeout_err # Store exception
                    # Return error directly without needing finally block for this case
                    self._priority_quota_lock.release() # Need to release lock here on early exit
                    return {"error": "Worker process timed out"}
                except BrokenPipeError as bpe:
                    provider_logger.warning(f"{worker_log_prefix}: Broken pipe during communicate(). Likely interrupted by ELP1 request.")
                    # Worker was killed by the lock interrupt logic.
                    try: worker_process.wait(timeout=1) # Ensure process object state updates
                    except: pass
                    # Try to get final output, might be empty/partial
                    stdout_data, stderr_data = worker_process.communicate()
                    comm_exception = bpe # Store exception
                    # Return specific interruption error
                    self._priority_quota_lock.release() # Need to release lock here on early exit
                    return {"error": "Worker task interrupted by higher priority request."}
                except Exception as comm_err:
                     provider_logger.error(f"{worker_log_prefix}: Error communicating with worker: {comm_err}")
                     # Attempt to kill if process exists
                     if worker_process and worker_process.poll() is None:
                          try:
                              worker_process.kill(); worker_process.communicate()
                          except Exception as kill_e:
                              provider_logger.error(f"{worker_log_prefix}: Error killing worker after comm error: {kill_e}")
                     comm_exception = comm_err # Store exception
                     # Return error directly
                     self._priority_quota_lock.release() # Need to release lock here on early exit
                     return {"error": f"Communication error with worker: {comm_err}"}

                # --- Check Worker Exit Code ---
                # This section is reached only if communicate() finished without raising an exception handled above
                exit_code = worker_process.returncode
                duration = time.monotonic() - start_time
                provider_logger.info(f"{worker_log_prefix}: Worker process finished. Exit Code: {exit_code}, Duration: {duration:.2f}s")

                # Log stderr from worker (contains worker's own logs/errors)
                if stderr_data:
                    log_level = "ERROR" if exit_code != 0 else "DEBUG"
                    provider_logger.log(log_level, f"{worker_log_prefix}: Worker stderr:\n-------\n{stderr_data.strip()}\n-------")

                # --- Handle Outcome ---
                if exit_code == 0: # Successfully exited with code 0
                    provider_logger.debug(f"{worker_log_prefix}: Worker exited cleanly. Parsing stdout...")
                    if not stdout_data:
                         provider_logger.error(f"{worker_log_prefix}: Worker exited cleanly but produced no stdout.")
                         # Return error as no result was obtained
                         # Lock released in finally block
                         return {"error": "Worker produced no output."}
                    try:
                        result_json = json.loads(stdout_data)
                        provider_logger.debug(f"{worker_log_prefix}: Parsed worker result JSON successfully.")
                        # Check if the worker itself reported an error internally
                        if isinstance(result_json, dict) and "error" in result_json:
                            provider_logger.error(f"{worker_log_prefix}: Worker reported internal error: {result_json['error']}")
                        # Return the parsed JSON (could be data or worker error)
                        # Lock released in finally block
                        return result_json
                    except json.JSONDecodeError as json_err:
                        provider_logger.error(f"{worker_log_prefix}: Failed to decode worker stdout JSON: {json_err}")
                        provider_logger.error(f"{worker_log_prefix}: Raw stdout from worker:\n{stdout_data[:1000]}...")
                        # Return error indicating decode failure
                        # Lock released in finally block
                        return {"error": f"Failed to decode worker response: {json_err}"}
                else: # Non-zero exit code (crash or internal worker error)
                    provider_logger.error(f"{worker_log_prefix}: Worker process exited with error code {exit_code}.")
                    crash_reason = f"Worker process crashed or failed (exit code {exit_code}). Check worker stderr logs."
                    # Add more specific reasons based on stderr if possible
                    if stderr_data:
                        if "Assertion" in stderr_data or "failed" in stderr_data.lower(): crash_reason += " Reason likely in stderr."
                        elif "Segmentation fault" in stderr_data: crash_reason += " Likely segfault."
                        elif "terminate called after throwing" in stderr_data: crash_reason += " C++ Exception."
                    # Return crash error
                    # Lock released in finally block
                    return {"error": crash_reason}

            except Exception as e:
                # Catch unexpected errors in starting/managing the process itself
                provider_logger.error(f"{worker_log_prefix}: Unexpected error managing worker process: {e}")
                provider_logger.exception(f"{worker_log_prefix}: Worker Management Traceback")
                # Ensure process is terminated if it exists and is running
                if worker_process and worker_process.poll() is None:
                    provider_logger.warning(f"{worker_log_prefix}: Terminating worker due to manager error.")
                    try:
                        worker_process.kill()
                        worker_process.communicate() # Clean up pipes
                    except Exception as kill_e:
                         provider_logger.error(f"{worker_log_prefix}: Error killing worker after manager error: {kill_e}")
                # Return manager error
                # Lock released in finally block
                return {"error": f"Error managing worker process: {e}"}
            finally:
                # --- Release Priority Lock ---
                # This ensures the lock is released even if errors occurred within the try block
                provider_logger.info(f"{worker_log_prefix}: Releasing worker execution lock.")
                self._priority_quota_lock.release()
        else:
             # Lock acquisition failed
             provider_logger.error(f"{worker_log_prefix}: FAILED to acquire worker lock.")
             return {"error": "Failed to acquire execution lock for worker."}

    # <<< --- END NEW Worker Execution Method --- >>>

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
        """Gets the Langchain model wrapper (Chat or Embeddings)."""
        # ... (Logic remains the same - it returns the *wrapper* object) ...
        model_instance = self.models.get(model_role)
        if not model_instance:
             logger.error(f"Model/Wrapper for role '{model_role}' not available.")
             if model_role != "default": logger.warning("Falling back to default model/wrapper."); return self.models.get("default")
             else: return None
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