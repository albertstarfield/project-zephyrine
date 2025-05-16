# ai_provider.py
import os
import sys
import time
import threading
import gc # For garbage collection
from typing import Dict, Any, Optional, List, Iterator, Tuple, Union
from loguru import logger
import json        # Added for worker communication
import subprocess  # Added for worker management
import shlex       # <<< --- ADD THIS LINE --- >>>
import asyncio

# --- NEW: Import the custom lock ---

from priority_lock import ELP0, ELP1, PriorityQuotaLock #
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
            self,
            messages: Union[List[BaseMessage], str],  # Can now be a raw string
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        provider_logger = getattr(self.ai_provider, 'logger', logger)
        wrapper_log_prefix = f"LlamaCppChatWrapper(Role:{self.model_role})"

        priority = kwargs.pop('priority', ELP0)
        is_raw_chatml_prompt_mode = isinstance(messages, str)  # Detect raw mode

        provider_logger.debug(
            f"{wrapper_log_prefix}: Received call. RawMode: {is_raw_chatml_prompt_mode}, Priority: ELP{priority}")

        # Combine model-specific kwargs passed during call with defaults
        final_model_kwargs = {**self.model_kwargs, **kwargs}

        # CRITICAL: Ensure stop sequences for raw ChatML mode
        # The primary stop sequence must be <|im_end|>
        effective_stop_sequences = list(stop) if stop else []
        if CHATML_END_TOKEN not in effective_stop_sequences:  # CHATML_END_TOKEN from your utils/config
            effective_stop_sequences.append(CHATML_END_TOKEN)

        final_model_kwargs["stop"] = effective_stop_sequences
        provider_logger.trace(f"{wrapper_log_prefix}: Final stop sequences: {final_model_kwargs['stop']}")
        provider_logger.trace(f"{wrapper_log_prefix}: Final model kwargs for worker: {final_model_kwargs}")

        request_payload = {}
        task_type_for_worker = ""

        if is_raw_chatml_prompt_mode:
            raw_prompt_string = messages  # messages is the string
            request_payload = {
                "prompt": raw_prompt_string,
                "kwargs": final_model_kwargs
            }
            task_type_for_worker = "raw_text_completion"  # New task type for worker
            provider_logger.debug(
                f"{wrapper_log_prefix}: Prepared for raw_text_completion. Prompt length: {len(raw_prompt_string)}")
        else:
            # This is the old path, for compatibility or if other parts still send BaseMessage lists
            # For the "all raw ChatML" goal, this path might become unused by AIChat/Agent.
            provider_logger.warning(
                f"{wrapper_log_prefix}: Received List[BaseMessage]. Formatting for standard chat task (might be deprecated path).")
            formatted_messages_for_worker = self._format_messages_for_llama_cpp(
                messages)  # messages is List[BaseMessage]
            request_payload = {
                "messages": formatted_messages_for_worker,
                "kwargs": final_model_kwargs
            }
            task_type_for_worker = "chat"  # Existing task type for worker

        try:
            provider_logger.debug(
                f"{wrapper_log_prefix}: Delegating to _execute_in_worker (Task: {task_type_for_worker}, Priority: ELP{priority})...")
            worker_result = self.ai_provider._execute_in_worker(
                model_role=self.model_role,
                task_type=task_type_for_worker,
                request_data=request_payload,
                priority=priority
            )

            if not worker_result or not isinstance(worker_result, dict):
                provider_logger.error(
                    f"{wrapper_log_prefix}: Worker execution failed or returned invalid data type: {type(worker_result)}")
                return f"[{self._llm_type.upper()}_PROVIDER_ERROR: Worker execution failed or returned invalid type]"

            if "error" in worker_result:
                error_msg = f"{self._llm_type.upper()}_WORKER_ERROR ({self.model_role}): {worker_result['error']}"
                provider_logger.error(error_msg)
                return f"[{error_msg}]"

            if "result" not in worker_result:
                provider_logger.error(
                    f"{wrapper_log_prefix}: Worker returned unknown dictionary structure: {worker_result}")
                return f"[{self._llm_type.upper()}_WORKER_RESPONSE_ERROR: Unknown dictionary structure]"

            completion_data = worker_result["result"]

            # Parse based on task type
            response_content = ""
            if task_type_for_worker == "raw_text_completion":
                # Expecting OpenAI-like completion structure from worker: result['choices'][0]['text']
                if (completion_data and isinstance(completion_data, dict) and
                        'choices' in completion_data and isinstance(completion_data['choices'], list) and
                        completion_data['choices'] and isinstance(completion_data['choices'][0], dict) and
                        'text' in completion_data['choices'][0]):
                    response_content = completion_data['choices'][0]['text']
                else:
                    provider_logger.error(
                        f"{wrapper_log_prefix}: Worker (raw_text_completion) returned unexpected result structure: {completion_data}")
                    return f"[{self._llm_type.upper()}_WORKER_RESPONSE_ERROR: Invalid raw completion structure]"

            elif task_type_for_worker == "chat":
                # Existing parsing logic for chat completion
                if (completion_data and isinstance(completion_data, dict) and
                        'choices' in completion_data and isinstance(completion_data['choices'], list) and
                        completion_data['choices'] and isinstance(completion_data['choices'][0], dict) and
                        'message' in completion_data['choices'][0] and
                        isinstance(completion_data['choices'][0]['message'], dict) and
                        'content' in completion_data['choices'][0]['message']):
                    response_content = completion_data['choices'][0]['message']['content']
                else:
                    provider_logger.error(
                        f"{wrapper_log_prefix}: Worker (chat) returned unexpected result structure: {completion_data}")
                    return f"[{self._llm_type.upper()}_WORKER_RESPONSE_ERROR: Invalid chat completion structure]"

            else:  # Should not happen
                provider_logger.error(
                    f"{wrapper_log_prefix}: Unknown task_type_for_worker '{task_type_for_worker}' during result parsing.")
                return f"[{self._llm_type.upper()}_INTERNAL_ERROR: Unknown task type for parsing]"

            if response_content is None:
                provider_logger.warning(f"{wrapper_log_prefix}: Worker returned None content. Treating as empty.")
                response_content = ""
            elif not isinstance(response_content, str):
                provider_logger.warning(
                    f"{wrapper_log_prefix}: Worker returned non-string content (type: {type(response_content)}). Converting to string.")
                response_content = str(response_content)

            # Remove the stop token if the model included it (some models do)
            # This is important because we added <|im_end|> to stop sequences.
            if response_content.endswith(CHATML_END_TOKEN):
                response_content = response_content[:-len(CHATML_END_TOKEN)]

            response_content = response_content.strip()  # Clean leading/trailing whitespace

            provider_logger.debug(
                f"{wrapper_log_prefix}: Successfully extracted content (len:{len(response_content)}).")
            return response_content

        except Exception as e:
            provider_logger.error(f"{wrapper_log_prefix}: Unexpected error during _call: {e}")
            provider_logger.exception(f"{wrapper_log_prefix} Call Traceback:")
            return f"[{self._llm_type.upper()}_WRAPPER_ERROR: {type(e).__name__} - {e}]"

    def _stream( # This method becomes more complex if we want to stream raw ChatML
        self, messages: Union[List[BaseMessage], str], stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        logger.warning(f"LlamaCppChatWrapper: Streaming requested. Raw ChatML streaming via worker needs careful implementation.")
        # For now, fallback to non-streaming or yield error for raw ChatML mode
        if isinstance(messages, str):
             logger.warning("Streaming in raw ChatML mode is not fully supported by this wrapper's _stream method. Yielding error.")
             yield ChatGenerationChunk(message=AIMessageChunk(content=f"[{self._llm_type.upper()}_ERROR: Streaming not supported with raw ChatML mode in this wrapper]"))
             return

        # Fallback to existing stream logic if List[BaseMessage] (might be deprecated path)
        # This part would need heavy modification if streaming raw ChatML is a hard requirement.
        # It would involve sending the raw prompt to the worker and having the worker stream back chunks.
        # For now, let's keep the existing logic for List[BaseMessage] if it's still used elsewhere,
        # or simplify it to also yield an error if we are strictly moving to raw ChatML everywhere.
        logger.warning(f"LlamaCppChatWrapper: _stream called with List[BaseMessage]. This path might be deprecated.")
        # ... (original _stream logic for List[BaseMessage] or yield error) ...
        # For now, let's just make it error out for any stream attempt to be clear.
        yield ChatGenerationChunk(message=AIMessageChunk(content=f"[{self._llm_type.upper()}_ERROR: Streaming not fully implemented for LlamaCppChatWrapper]"))


    # _format_messages_for_llama_cpp might become unused if all calls are raw ChatML strings.
    # Keep it for now for potential compatibility if some parts still use List[BaseMessage].

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
        return "llama_cpp_raw_chatml_wrapper"  # New type name to reflect change


# --- Embeddings Wrapper ---
# Placeholder/Modified Embeddings Wrapper
class LlamaCppEmbeddingsWrapper(Embeddings):
    ai_provider: 'AIProvider'
    model_role: str = "embeddings"

    def __init__(self, ai_provider: 'AIProvider'):
        super().__init__()
        self.ai_provider = ai_provider

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        provider_logger = getattr(self.ai_provider, 'logger', logger)
        # This call will use the default priority (ELP0) set in _embed_texts's signature
        provider_logger.debug(f"EmbedWrapper.embed_documents: Standard call for {len(texts)} texts (delegating to _embed_texts with its default ELP0).")
        return self._embed_texts(texts) # Default priority ELP0 will be used by _embed_texts

    # Standard Langchain interface method - uses default priority ELP0
    def embed_query(self, text: str) -> List[float]:
        provider_logger = getattr(self.ai_provider, 'logger', logger)
        provider_logger.debug(f"EmbedWrapper.embed_query: Standard call for query '{text[:30]}...' (delegating to _embed_texts with its default ELP0).")
        results = self._embed_texts([text]) # Default priority ELP0 will be used by _embed_texts
        if results and isinstance(results, list) and len(results) > 0 and isinstance(results[0], list) :
            return results[0]
        provider_logger.error(f"EmbedWrapper.embed_query: _embed_texts did not return a valid vector for query. Result: {results}")
        raise RuntimeError("Embedding query failed to produce a valid vector via _embed_texts.")

    def _embed_texts(self, texts: List[str], priority: int = ELP0) -> List[List[float]]:
        provider_logger = getattr(self.ai_provider, 'logger', logger)
        log_prefix = f"EmbedWrapper._embed_texts|ELP{priority}"  # Uses passed priority

        if not texts:
            provider_logger.debug(f"{log_prefix}: Received empty list of texts. Returning empty list.")
            return []

        provider_logger.debug(f"{log_prefix}: Delegating embedding for {len(texts)} texts to worker.")
        request_payload = {"texts": texts}

        # Call the AIProvider's worker execution method with the specified priority
        worker_result = self.ai_provider._execute_in_worker(  # type: ignore
            model_role=self.model_role,  # This is "embeddings"
            task_type="embedding",
            request_data=request_payload,
            priority=priority
        )

        if worker_result and isinstance(worker_result, dict):
            if "error" in worker_result:
                error_msg_content = worker_result['error']
                if interruption_error_marker in error_msg_content:  # type: ignore
                    provider_logger.warning(f"🚦 {log_prefix}: Embedding task INTERRUPTED: {error_msg_content}")
                    raise TaskInterruptedException(error_msg_content)  # type: ignore
                else:
                    full_error_msg = f"LLAMA_CPP_EMBED_WORKER_ERROR ({self.model_role}|ELP{priority}): {error_msg_content}"
                    provider_logger.error(full_error_msg)
                    raise RuntimeError(full_error_msg)
            elif "result" in worker_result and isinstance(worker_result["result"], list):
                batch_embeddings = worker_result["result"]
                # Validate structure of returned embeddings
                if all(isinstance(emb, list) and all(isinstance(num, (float, int)) for num in emb) for emb in
                       batch_embeddings):
                    provider_logger.debug(
                        f"{log_prefix}: Embedding successful. Returning {len(batch_embeddings)} vectors.")
                    return [[float(num) for num in emb] for emb in batch_embeddings]
                else:
                    provider_logger.error(
                        f"{log_prefix}: Worker returned invalid embedding structure. Example Element Type: {type(batch_embeddings[0]) if batch_embeddings else 'N/A'}")
                    raise RuntimeError(
                        "LLAMA_CPP_EMBED_WORKER_RESPONSE_ERROR: Invalid embedding structure from worker.")
            else:
                provider_logger.error(
                    f"{log_prefix}: Worker returned unknown dictionary structure for embeddings: {str(worker_result)[:200]}...")
                raise RuntimeError(
                    "LLAMA_CPP_EMBED_WORKER_RESPONSE_ERROR: Unknown dictionary structure for embeddings.")
        else:
            provider_logger.error(
                f"{log_prefix}: AIProvider._execute_in_worker for embeddings failed or returned invalid data type: {type(worker_result)}")
            raise RuntimeError(
                "LLAMA_CPP_EMBED_PROVIDER_ERROR: Worker execution for embeddings failed or returned invalid type.")

# === AI Provider Class ===
class AIProvider:
    def __init__(self, provider_name):
        # ... (other initialization lines) ...
        self.provider_name = provider_name.lower()
        self.models: Dict[str, Any] = {}
        self.embeddings: Optional[Embeddings] = None
        self.EMBEDDINGS_MODEL_NAME: Optional[str] = None
        self.embeddings: Optional[LlamaCppEmbeddingsWrapper] = None  # NEW, if always this type
        self.setup_provider()
        # self.image_generator: Any = None # This will be implicitly handled by calling the worker

        # --- llama.cpp specific state ---
        self._llama_model_access_lock = threading.Lock() if LLAMA_CPP_AVAILABLE and self.provider_name == "llama_cpp" else None
        if LLAMA_CPP_AVAILABLE and self.provider_name == "llama_cpp":
             if PriorityQuotaLock is not threading.Lock:
                 self._priority_quota_lock = PriorityQuotaLock()
                 logger.info("   🔑 Initialized PriorityQuotaLock for worker access (Llama & Imagination).")
             else:
                  logger.error("   ❌ PriorityQuotaLock import failed, falling back to standard Lock (no priority).")
                  self._priority_quota_lock = threading.Lock()
        elif self.provider_name != "llama_cpp": # If not llama_cpp, still might need a lock for image gen
            if PriorityQuotaLock is not threading.Lock:
                self._priority_quota_lock = PriorityQuotaLock()
                logger.info("   🔑 Initialized PriorityQuotaLock for Imagination Worker access.")
            else:
                logger.error("   ❌ PriorityQuotaLock import failed, falling back to standard Lock for Imagination Worker.")
                self._priority_quota_lock = threading.Lock()

        self._llama_model_map: Dict[str, str] = {}
        self._llama_gguf_dir: Optional[str] = None
        self._python_executable = sys.executable

        # --- NEW: For Imagination Worker ---
        self._imagination_worker_script_path: Optional[str] = None
        # --- END NEW ---

        logger.info(f"🤖 Initializing AI Provider: {self.provider_name} (Worker Process Mode)")
        # Log lock status
        if self._priority_quota_lock and isinstance(self._priority_quota_lock, PriorityQuotaLock): logger.info("   🔑 PriorityQuotaLock is active.")
        elif self._priority_quota_lock: logger.warning("   🔑 Using standard threading.Lock (no priority/quota).")
        else: logger.error("   ❌ Lock for worker access NOT initialized!")

        self._validate_config()
        self.setup_provider() # Sets up LLM/Embedding models
        if self.embeddings:
            logger.info(f"AIProvider embeddings initialized. Type: {type(self.embeddings)}")
            if hasattr(self.embeddings, 'embed_query') and hasattr(self.embeddings, 'embed_documents'):
                logger.info("  AIProvider embeddings object HAS 'embed_query' and 'embed_documents' methods.")
            else:
                logger.error("  CRITICAL: AIProvider embeddings object LACKS required embedding methods!")
        else:
            logger.error("CRITICAL: AIProvider.embeddings is None after setup_provider!")
        # VVVVVVVVVV THIS IS THE LINE TO CHANGE VVVVVVVVVV
        self._setup_image_generator_config() # Renamed: Validates image gen worker config
        # ^^^^^^^^^^ THIS IS THE LINE TO CHANGE ^^^^^^^^^^

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

        global STABLE_DIFFUSION_WORKER_CONFIGURED  # To update global status
        self._imagination_worker_script_path = os.path.join(os.path.dirname(__file__), IMAGE_WORKER_SCRIPT_NAME)
        if not os.path.isfile(self._imagination_worker_script_path):
            logger.warning(
                f"⚠️ Imagination worker script '{IMAGE_WORKER_SCRIPT_NAME}' not found at: {self._imagination_worker_script_path}. Image generation will be disabled.")
            STABLE_DIFFUSION_WORKER_CONFIGURED = False
            return  # Cannot proceed with image gen validation if script is missing

        if not os.path.isdir(IMAGE_GEN_MODEL_DIR):
            logger.warning(
                f"⚠️ Imagination model directory not found: '{IMAGE_GEN_MODEL_DIR}'. Image generation will be disabled.")
            STABLE_DIFFUSION_WORKER_CONFIGURED = False
            return

        # Check for at least the main diffusion model file
        main_diffusion_path = os.path.join(IMAGE_GEN_MODEL_DIR, IMAGE_GEN_DIFFUSION_MODEL_NAME)
        if not os.path.isfile(main_diffusion_path):
            logger.warning(
                f"⚠️ Main imagination diffusion model '{IMAGE_GEN_DIFFUSION_MODEL_NAME}' not found in '{IMAGE_GEN_MODEL_DIR}'. Image generation will be disabled.")
            STABLE_DIFFUSION_WORKER_CONFIGURED = False
            return

        logger.info(f"✅ Imagination worker script and model directory appear configured.")
        STABLE_DIFFUSION_WORKER_CONFIGURED = True


    # <<< --- NEW: Worker Execution Method --- >>>
    def _execute_in_worker(self, model_role: str, task_type: str, request_data: Dict[str, Any], priority: int = ELP0) -> \
    Optional[Dict[str, Any]]:
        provider_logger = getattr(self, 'logger', logger)
        worker_log_prefix = f"WORKER_MGR(ELP{priority}|{model_role}/{task_type})"
        provider_logger.debug(f"{worker_log_prefix}: Attempting to execute task in worker.")

        if not self._priority_quota_lock:
            provider_logger.error(f"{worker_log_prefix}: Priority Lock not initialized!")
            return {"error": "Provider lock not initialized"}
        if self.provider_name != "llama_cpp":
            provider_logger.error(f"{worker_log_prefix}: Attempted worker execution outside llama_cpp provider.")
            return {"error": "Worker execution only for llama_cpp provider"}

        gguf_filename = self._llama_model_map.get(model_role)
        if not gguf_filename:
            provider_logger.error(f"{worker_log_prefix}: No GGUF file configured for role '{model_role}'")
            return {"error": f"No GGUF config for role {model_role}"}
        model_path = os.path.join(self._llama_gguf_dir or "", gguf_filename)
        if not os.path.isfile(model_path):
            provider_logger.error(f"{worker_log_prefix}: Model file not found: {model_path}")
            return {"error": f"Model file not found: {os.path.basename(model_path)}"}

        lock_acquired = False
        worker_process = None
        start_lock_wait = time.monotonic()
        provider_logger.debug(f"{worker_log_prefix}: Acquiring worker execution lock (Priority: ELP{priority})...")
        lock_acquired = self._priority_quota_lock.acquire(priority=priority, timeout=None)
        lock_wait_duration = time.monotonic() - start_lock_wait

        if lock_acquired:
            provider_logger.info(
                f"{worker_log_prefix}: Lock acquired (waited {lock_wait_duration:.2f}s). Starting worker process.")
            try:
                worker_script_path = os.path.join(os.path.dirname(__file__), "llama_worker.py")
                command = [
                    self._python_executable,
                    worker_script_path,
                    "--model-path", model_path,
                    "--task-type", task_type,
                    "--n-gpu-layers", str(LLAMA_CPP_N_GPU_LAYERS),
                ]

                # === MODIFIED n_ctx LOGIC ===
                if task_type == "embedding":
                    # Embeddings always get a fixed n_ctx (e.g., 512, or LLAMA_CPP_N_CTX if that was intended for override)
                    # Let's assume a small fixed value is best for embeddings unless an override is in LLAMA_CPP_MODEL_MAP for the embedding model specifically.
                    embedding_n_ctx = self._llama_model_map.get(f"{model_role}_n_ctx",
                                                                512)  # Check for specific override like "embeddings_n_ctx"
                    command.extend(["--n-ctx", str(embedding_n_ctx)])
                    provider_logger.info(
                        f"{worker_log_prefix}: Passing fixed --n-ctx {embedding_n_ctx} for embeddings.")
                elif LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT is not None:  # Check if config.py has an explicit override value
                    # LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT needs to be defined in config.py, e.g. can be os.getenv("LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT", None)
                    # If it's set (e.g. to 4096 from config), pass it to the worker as an override.
                    # The worker will use this if --n-ctx is provided.
                    command.extend(["--n-ctx", str(LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT)])
                    provider_logger.info(
                        f"{worker_log_prefix}: Passing --n-ctx override {LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT} for {task_type} based on config.")
                else:
                    # For chat/raw_text_completion, if no override, DO NOT pass --n-ctx.
                    # The worker will calculate it dynamically.
                    provider_logger.info(
                        f"{worker_log_prefix}: Not passing --n-ctx for {task_type}. Worker will calculate dynamically.")
                # === END MODIFIED n_ctx LOGIC ===

                if LLAMA_CPP_VERBOSE: command.append("--verbose")
                if task_type == "chat":  # Chat format is only relevant for chat tasks
                    chat_fmt = self._llama_model_map.get(f"{model_role}_chat_format", "chatml")
                    if chat_fmt: command.extend(["--chat-format", chat_fmt])

                provider_logger.debug(
                    f"{worker_log_prefix}: Worker command: {' '.join(shlex.quote(c) for c in command)}")
                start_time = time.monotonic()
                worker_process = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding='utf-8', errors='replace'
                )
                if priority == ELP0: self._priority_quota_lock.set_holder_process(worker_process)

                input_json = json.dumps(request_data)
                provider_logger.debug(
                    f"{worker_log_prefix}: Sending input JSON (len={len(input_json)}) to worker stdin...")
                stdout_data, stderr_data = "", ""

                try:
                    # Use LLAMA_WORKER_TIMEOUT from config
                    stdout_data, stderr_data = worker_process.communicate(input=input_json,
                                                                          timeout=LLAMA_WORKER_TIMEOUT)
                    provider_logger.debug(f"{worker_log_prefix}: Worker communicate() finished.")
                except subprocess.TimeoutExpired:
                    provider_logger.error(
                        f"{worker_log_prefix}: Worker process timed out after {LLAMA_WORKER_TIMEOUT}s.")
                    worker_process.kill();
                    stdout_data, stderr_data = worker_process.communicate()
                    self._priority_quota_lock.release()
                    return {"error": "Worker process timed out"}
                except BrokenPipeError:
                    provider_logger.warning(f"{worker_log_prefix}: Broken pipe. Likely interrupted.")
                    provider_logger.warning(
                        f"{worker_log_prefix}: Broken pipe during communicate(). Likely interrupted by ELP1 request.")
                    try:
                        worker_process.wait(timeout=1)
                    except:
                        pass
                    stdout_data, stderr_data = worker_process.communicate()  # Try to get any final output
                    self._priority_quota_lock.release()
                    return {"error": interruption_error_marker}  # Use the consistent marker
                except Exception as comm_err:
                    provider_logger.error(f"{worker_log_prefix}: Error communicating with worker: {comm_err}")
                    if worker_process and worker_process.poll() is None:
                        try:
                            worker_process.kill(); worker_process.communicate()
                        except:
                            pass
                    self._priority_quota_lock.release()
                    return {"error": f"Communication error with worker: {comm_err}"}

                exit_code = worker_process.returncode
                duration = time.monotonic() - start_time
                provider_logger.info(
                    f"{worker_log_prefix}: Worker process finished. Exit Code: {exit_code}, Duration: {duration:.2f}s")
                if stderr_data:
                    log_level = "ERROR" if exit_code != 0 else "DEBUG"
                    provider_logger.log(log_level,
                                        f"{worker_log_prefix}: Worker stderr:\n-------\n{stderr_data.strip()}\n-------")

                if exit_code == 0:
                    if not stdout_data:
                        provider_logger.error(f"{worker_log_prefix}: Worker exited cleanly but no stdout.")
                        return {"error": "Worker produced no output."}
                    try:
                        result_json = json.loads(stdout_data)
                        provider_logger.debug(f"{worker_log_prefix}: Parsed worker result JSON successfully.")
                        if isinstance(result_json, dict) and "error" in result_json:
                            provider_logger.error(
                                f"{worker_log_prefix}: Worker reported internal error: {result_json['error']}")
                        return result_json
                    except json.JSONDecodeError as json_err:
                        provider_logger.error(f"{worker_log_prefix}: Failed to decode worker stdout JSON: {json_err}")
                        provider_logger.error(f"{worker_log_prefix}: Raw stdout from worker:\n{stdout_data[:1000]}...")
                        return {"error": f"Failed to decode worker response: {json_err}"}
                else:
                    crash_reason = f"Worker process failed (exit code {exit_code}). Check worker stderr."
                    return {"error": crash_reason}
            except Exception as e:
                provider_logger.error(f"{worker_log_prefix}: Unexpected error managing worker: {e}")
                provider_logger.exception(f"{worker_log_prefix}: Worker Management Traceback")
                if worker_process and worker_process.poll() is None:
                    try:
                        worker_process.kill(); worker_process.communicate()
                    except:
                        pass
                return {"error": f"Error managing worker process: {e}"}
            finally:
                provider_logger.info(f"{worker_log_prefix}: Releasing worker execution lock.")
                self._priority_quota_lock.release()
        else:
            provider_logger.error(f"{worker_log_prefix}: FAILED to acquire worker lock.")
            return {"error": "Failed to acquire execution lock for worker."}

    async def _execute_imagination_worker(
            self,
            prompt: str,
            image_base64: Optional[str] = None,
            priority: int = ELP0
    ) -> Optional[Dict[str, Any]]:  # Returns the full JSON response from worker or error dict
        """
        Executes imagination_worker.py, sends request, gets response.
        Protected by the PRIORITY QUOTA lock. ELP1 can interrupt ELP0.
        Handles noisy stdout from the worker by attempting to extract the last valid JSON object.
        """
        provider_logger = getattr(self, 'logger', logger)
        task_name = "img2img" if image_base64 else "txt2img"
        worker_log_prefix = f"IMG_WORKER_MGR(ELP{priority}|{task_name})"
        provider_logger.debug(f"{worker_log_prefix}: Attempting to execute task in Imagination Worker.")

        if not STABLE_DIFFUSION_WORKER_CONFIGURED:  # Global flag set by _validate_config
            provider_logger.error(f"{worker_log_prefix}: Imagination worker not configured (script/models missing).")
            return {"error": "Imagination worker is not configured on the server."}

        if not self._priority_quota_lock:
            provider_logger.error(f"{worker_log_prefix}: Priority Lock not initialized!")
            return {"error": "Provider's shared resource lock not initialized"}

        # Variables for the blocking_lock_and_execute closure
        # These are effectively "captured" by the nested function.
        # `start_lock_wait` is defined just before calling asyncio.to_thread.
        # `lock_acquired` and `worker_process` are managed within blocking_lock_and_execute.

        def blocking_lock_and_execute() -> Optional[Dict[str, Any]]:
            # This function runs in a separate thread via asyncio.to_thread.
            # It handles lock acquisition, subprocess execution, and parsing worker output.
            # It should not use `await` and should return a dictionary (success or error).

            # Local variables for this threaded function's execution context
            current_lock_acquired = False
            current_worker_process: Optional[subprocess.Popen] = None
            # `task_name`, `prompt`, `image_base64`, `priority` are from the outer scope (closure)
            # `_python_executable`, `_imagination_worker_script_path` are instance vars (self.)
            # Config constants like IMAGE_GEN_MODEL_DIR are global from config.py

            # Acquire the shared priority lock
            # `start_lock_wait` is captured from the outer scope of _execute_imagination_worker
            current_lock_acquired = self._priority_quota_lock.acquire(priority=priority, timeout=None)
            lock_wait_duration = time.monotonic() - start_lock_wait

            if current_lock_acquired:
                provider_logger.info(
                    f"{worker_log_prefix}: Lock acquired (waited {lock_wait_duration:.2f}s). Starting worker process.")
                try:
                    # Prepare the command to execute the imagination worker script
                    command = [
                        self._python_executable,
                        self._imagination_worker_script_path,
                        "--model-dir", IMAGE_GEN_MODEL_DIR,
                        "--diffusion-model-name", IMAGE_GEN_DIFFUSION_MODEL_NAME,
                        "--clip-l-name", IMAGE_GEN_CLIP_L_NAME,
                        "--t5xxl-name", IMAGE_GEN_T5XXL_NAME,
                        "--vae-name", IMAGE_GEN_VAE_NAME,
                        "--w-device", IMAGE_GEN_DEVICE,
                        "--rng-type", IMAGE_GEN_RNG_TYPE,
                        "--n-threads", str(IMAGE_GEN_N_THREADS),
                    ]
                    provider_logger.debug(
                        f"{worker_log_prefix}: Worker command: {' '.join(shlex.quote(c) for c in command)}")

                    # Prepare the JSON request data to be sent to the worker's stdin
                    request_data_for_worker = {
                        "task_type": task_name,
                        "prompt": prompt,
                        "negative_prompt": IMAGE_GEN_DEFAULT_NEGATIVE_PROMPT,
                        "n": 1,
                        # Worker is designed to produce one image per call; looping handled by generate_image_async
                        "size": IMAGE_GEN_DEFAULT_SIZE,
                        "cfg_scale": IMAGE_GEN_DEFAULT_CFG_SCALE,
                        "sample_steps": IMAGE_GEN_DEFAULT_SAMPLE_STEPS,
                        "sample_method": IMAGE_GEN_DEFAULT_SAMPLE_METHOD,
                        "seed": IMAGE_GEN_DEFAULT_SEED,
                        "response_format": IMAGE_GEN_RESPONSE_FORMAT,
                    }
                    if image_base64:
                        request_data_for_worker["input_image_b64"] = image_base64
                        request_data_for_worker["task_type"] = "img2img"  # Ensure task_type is correct
                        provider_logger.info(f"{worker_log_prefix}: img2img task with input image.")

                    # Record start time for worker execution
                    start_time_worker_exec = time.monotonic()

                    # Start the worker subprocess
                    current_worker_process = subprocess.Popen(
                        command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,  # For text-mode communication (JSON)
                        encoding='utf-8',  # Specify encoding
                        errors='replace',  # Handle potential encoding errors in output
                        cwd=os.path.dirname(__file__)  # Run worker from AIProvider's directory
                    )

                    # If it's a background priority task, register its process with the lock
                    if priority == ELP0:
                        self._priority_quota_lock.set_holder_process(current_worker_process)

                    # Convert request data to JSON and send to worker
                    input_json_to_worker = json.dumps(request_data_for_worker)
                    provider_logger.debug(
                        f"{worker_log_prefix}: Sending input JSON (len={len(input_json_to_worker)}) to worker stdin...")

                    stdout_data_raw, stderr_data_raw = "", ""
                    communication_error_obj = None
                    # Timeout for the worker process (e.g., 5 minutes, adjust as needed)
                    worker_process_timeout = IMAGE_GEN_WORKER_TIMEOUT  # Use a config constant

                    # Communicate with the worker: send input, get output/error
                    try:
                        stdout_data_raw, stderr_data_raw = current_worker_process.communicate(
                            input=input_json_to_worker, timeout=worker_process_timeout
                        )
                        provider_logger.debug(f"{worker_log_prefix}: Worker communicate() finished.")
                    except subprocess.TimeoutExpired:
                        provider_logger.error(
                            f"{worker_log_prefix}: Worker process timed out after {worker_process_timeout}s.")
                        current_worker_process.kill()
                        try:
                            stdout_data_raw, stderr_data_raw = current_worker_process.communicate()
                        except Exception:
                            pass  # Best effort to get final output
                        communication_error_obj = TimeoutError("Imagination worker process timed out")
                    except BrokenPipeError:
                        provider_logger.warning(
                            f"{worker_log_prefix}: Broken pipe during communicate(). Likely interrupted by ELP1 request.")
                        if current_worker_process.poll() is None:
                            try:
                                current_worker_process.wait(timeout=1.0)
                            except subprocess.TimeoutExpired:
                                current_worker_process.kill()
                        try:
                            stdout_data_final_bp, stderr_data_final_bp = current_worker_process.communicate()
                            stdout_data_raw += stdout_data_final_bp  # Append any remaining output
                            stderr_data_raw += stderr_data_final_bp
                        except Exception:
                            pass
                        communication_error_obj = BrokenPipeError(interruption_error_marker)
                    except Exception as general_comm_err:
                        provider_logger.error(
                            f"{worker_log_prefix}: Error communicating with worker: {general_comm_err}")
                        if current_worker_process and current_worker_process.poll() is None:
                            try:
                                current_worker_process.kill(); current_worker_process.communicate()
                            except Exception:
                                pass
                        communication_error_obj = general_comm_err

                    # If communication failed, release lock and return error
                    if communication_error_obj:
                        self._priority_quota_lock.release()
                        if isinstance(communication_error_obj, BrokenPipeError):
                            return {"error": interruption_error_marker}
                        return {"error": f"Communication error with Imagination Worker: {communication_error_obj}"}

                    # Process finished, get exit code and log duration
                    exit_code_from_worker = current_worker_process.returncode
                    execution_duration = time.monotonic() - start_time_worker_exec
                    provider_logger.info(
                        f"{worker_log_prefix}: Worker process finished. Exit Code: {exit_code_from_worker}, Duration: {execution_duration:.2f}s")

                    # Log stderr from the worker
                    if stderr_data_raw:
                        stderr_log_level = "ERROR" if exit_code_from_worker != 0 else "DEBUG"
                        provider_logger.log(stderr_log_level,
                                            f"{worker_log_prefix}: Worker STDERR:\n-------\n{stderr_data_raw.strip()}\n-------")

                    # Handle worker outcome based on exit code
                    if exit_code_from_worker == 0:  # Worker exited cleanly
                        if not stdout_data_raw:  # Check if there's any stdout
                            provider_logger.error(f"{worker_log_prefix}: Worker exited cleanly but produced no stdout.")
                            return {"error": "Imagination worker produced no output."}

                        # Attempt to extract and parse JSON from worker's stdout
                        json_string_extracted = None
                        provider_logger.trace(
                            f"{worker_log_prefix}: Attempting to extract JSON from stdout (len {len(stdout_data_raw)}). Preview (first/last 250 chars):")
                        provider_logger.trace(f"STDOUT_START>>>\n{stdout_data_raw[:250]}\n<<<STDOUT_START_END")
                        provider_logger.trace(f"STDOUT_END>>>\n{stdout_data_raw[-250:]}\n<<<STDOUT_END_END")

                        # JSON Extraction Strategy using JSONDecoder.raw_decode
                        decoder = json.JSONDecoder()
                        search_start_index = 0
                        while search_start_index < len(stdout_data_raw):
                            # Find the next potential start of a JSON object
                            first_brace_in_remaining = stdout_data_raw.find('{', search_start_index)
                            if first_brace_in_remaining == -1:
                                break  # No more opening braces

                            substring_to_test = stdout_data_raw[first_brace_in_remaining:]
                            try:
                                obj, end_idx_relative_to_substring = decoder.raw_decode(substring_to_test)
                                # Successfully decoded a JSON object
                                current_extracted_json_str = substring_to_test[:end_idx_relative_to_substring]

                                # Check if this object has the expected top-level keys for success or a worker error
                                if isinstance(obj, dict) and (("created" in obj and "data" in obj) or "error" in obj):
                                    json_string_extracted = current_extracted_json_str  # This is likely our target
                                    provider_logger.debug(
                                        f"{worker_log_prefix}: Found potential main/error JSON payload using raw_decode: {json_string_extracted[:100]}...")
                                    # Keep this one, as it's the last valid full JSON found by iterating forward
                                # else: it's some other JSON, ignore and let loop continue to find later JSON

                                # Advance search_start_index to look for more JSON objects after this one
                                search_start_index = first_brace_in_remaining + end_idx_relative_to_substring

                            except json.JSONDecodeError:
                                # This substring is not the start of a valid JSON object.
                                # Advance past the brace that failed.
                                search_start_index = first_brace_in_remaining + 1

                        # Fallback: if iterative raw_decode didn't pinpoint the main structure, try last line
                        if not json_string_extracted:
                            provider_logger.warning(
                                f"{worker_log_prefix}: Iterative JSON search (raw_decode) did not identify a primary payload. Trying last-line fallback.")
                            stdout_lines = stdout_data_raw.strip().split('\n')
                            if stdout_lines:
                                last_line = stdout_lines[-1].strip()
                                if last_line.startswith('{') and last_line.endswith('}'):
                                    try:
                                        json.loads(last_line)  # Validate
                                        json_string_extracted = last_line
                                        provider_logger.debug(
                                            f"{worker_log_prefix}: Fallback: Used last line as JSON: {json_string_extracted[:100]}...")
                                    except json.JSONDecodeError:
                                        provider_logger.warning(
                                            f"{worker_log_prefix}: Fallback: Last line failed JSON parse: {last_line[:100]}...")

                        # Final check and parse of the extracted string
                        if not json_string_extracted:
                            provider_logger.error(
                                f"{worker_log_prefix}: All strategies failed. Could not find valid JSON payload in worker stdout.")
                            provider_logger.debug(
                                f"{worker_log_prefix}: Full stdout from worker was:\n{stdout_data_raw}")
                            return {"error": "Imagination worker stdout did not contain a valid JSON payload."}

                        try:
                            final_parsed_json = json.loads(json_string_extracted)
                            provider_logger.debug(
                                f"{worker_log_prefix}: Successfully parsed final extracted JSON string. Top-level keys: {list(final_parsed_json.keys()) if isinstance(final_parsed_json, dict) else 'Not a dict'}")

                            # Validate structure of the successfully parsed JSON
                            if isinstance(final_parsed_json, dict) and ((
                                                                                "created" in final_parsed_json and "data" in final_parsed_json) or "error" in final_parsed_json):
                                if "error" in final_parsed_json:
                                    provider_logger.warning(
                                        f"{worker_log_prefix}: Worker's JSON payload contains an error: {final_parsed_json['error']}")
                                else:
                                    provider_logger.info(
                                        f"{worker_log_prefix}: Extracted JSON appears to be the correct main payload.")
                                return final_parsed_json  # Return the parsed JSON (could be success or worker-reported error)
                            else:
                                provider_logger.error(
                                    f"{worker_log_prefix}: Extracted JSON is valid but NOT the expected main success payload or a known error structure. Content: {str(final_parsed_json)[:200]}")
                                return {
                                    "error": f"Extracted valid JSON from worker, but it's not the expected top-level structure. Got keys: {list(final_parsed_json.keys()) if isinstance(final_parsed_json, dict) else 'Not a dict'}"}

                        except json.JSONDecodeError as json_err_final_attempt:
                            provider_logger.error(
                                f"{worker_log_prefix}: CRITICAL: Final attempt to decode extracted JSON string FAILED: {json_err_final_attempt}")
                            provider_logger.error(
                                f"{worker_log_prefix}: String that failed final parse: {json_string_extracted[:1000]}...")
                            provider_logger.debug(
                                f"{worker_log_prefix}: Original full stdout from worker was:\n{stdout_data_raw}")
                            return {
                                "error": f"Failed to decode Imagination worker response (final attempt after extraction): {json_err_final_attempt}"}

                    else:  # Worker exited with a non-zero code (crashed)
                        crash_message = f"Imagination worker process crashed or failed (exit code {exit_code_from_worker})."
                        if stdout_data_raw:
                            provider_logger.error(
                                f"{worker_log_prefix}: Worker STDOUT on crash:\n-------\n{stdout_data_raw.strip()}\n-------")
                        if stderr_data_raw and (
                                "error" in stderr_data_raw.lower() or "traceback" in stderr_data_raw.lower() or "assertion" in stderr_data_raw.lower()):
                            crash_message += " Check worker stderr for details."
                        elif not stderr_data_raw and stdout_data_raw:  # If stderr empty but stdout has content with error indicators
                            if "error" in stdout_data_raw.lower() or "traceback" in stdout_data_raw.lower() or "assertion" in stdout_data_raw.lower():
                                crash_message += " Check worker stdout for potential error messages."
                        return {"error": crash_message}

                except Exception as e_outer_manage:  # Catch errors in managing the worker process
                    provider_logger.error(
                        f"{worker_log_prefix}: Unexpected error managing worker process: {e_outer_manage}")
                    provider_logger.exception(f"{worker_log_prefix}: Worker Management Traceback")
                    if current_worker_process and current_worker_process.poll() is None:
                        try:
                            current_worker_process.kill(); current_worker_process.communicate()
                        except Exception:
                            pass
                    return {"error": f"Error managing Imagination worker process: {e_outer_manage}"}
                finally:
                    provider_logger.debug(f"{worker_log_prefix}: Releasing worker execution lock.")
                    self._priority_quota_lock.release()
            else:  # Lock acquisition failed
                provider_logger.error(f"{worker_log_prefix}: FAILED to acquire worker lock.")
                return {"error": "Failed to acquire execution lock for Imagination worker."}

            # This path should not be normally reached if lock acquisition fails, due to the return above.
            # Added for completeness in case of unexpected flow.
            return {"error": "Reached unexpected point in blocking_lock_and_execute after lock check."}

        # `start_lock_wait` needs to be defined in this scope for the closure
        start_lock_wait = time.monotonic()

        worker_response = await asyncio.to_thread(blocking_lock_and_execute)
        return worker_response

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
                logger.error(f"OLLAMA IS NO LONGER SUPPORTED!")
                sys.exit(1)
            elif self.provider_name == "fireworks":
                logger.error(f"EXTERNAL ENGINE FIREWORK IS NO LONGER SUPPORTED!")
                sys.exit(1)

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
                default_temp = DEFAULT_LLM_TEMPERATURE # Example default temp
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

    def _setup_image_generator_config(self):  # Renamed from _setup_image_generator
        """Validates configuration for the imagination worker."""
        global STABLE_DIFFUSION_WORKER_CONFIGURED  # Use global status
        if STABLE_DIFFUSION_WORKER_CONFIGURED:
            logger.info("✅ Image Generation Worker appears configured (script/models validated).")
        else:
            logger.warning(
                "⚠️ Image Generation Worker is NOT configured or validation failed. Image generation unavailable.")
        # No actual instance loading here, worker script handles it.

    async def generate_image_async(
            self,
            prompt: str,
            image_base64: Optional[str] = None,
            priority: int = ELP0
    ) -> Tuple[Optional[List[Dict[str, Optional[str]]]], Optional[str]]:
        """
        Public method to generate an image using the imagination worker.
        Returns a list of image data dicts (each dict like {"b64_json": ..., "b64_avif": ...})
        or an error message in the second part of the tuple.
        """
        provider_logger = getattr(self, 'logger', logger)
        if not STABLE_DIFFUSION_WORKER_CONFIGURED:
            msg = "Image generation is not available because the worker is not configured."
            provider_logger.error(msg)
            return [], msg  # Return empty list for images and the error message

        provider_logger.info(
            f"🖼️ Requesting image generation (ELP{priority}). Prompt: '{prompt[:50]}...' ImageInput: {'Yes' if image_base64 else 'No'}")
        start_time = time.monotonic()  # Start time for the entire async operation

        try:
            worker_response = await self._execute_imagination_worker(prompt, image_base64, priority)

            duration = time.monotonic() - start_time  # Total duration
            provider_logger.debug(
                f"Imagination worker execution (via _execute_imagination_worker) completed in {duration:.2f}s")

            if not worker_response:
                provider_logger.error(
                    "Imagination worker (via _execute_imagination_worker) returned None (unexpected).")
                return [], "Imagination worker execution failed to return a response."

            # Detailed logging of what _execute_imagination_worker returned
            provider_logger.critical(
                f"generate_image_async: ENTERING CHECKS. worker_response type: {type(worker_response)}")
            if isinstance(worker_response, dict):
                provider_logger.critical(
                    f"generate_image_async: worker_response IS a dict. Top-level keys: {list(worker_response.keys())}")
                if "created" in worker_response:
                    provider_logger.critical(f"  Key 'created' FOUND. Value: {worker_response.get('created')}")
                else:
                    provider_logger.critical(f"  Key 'created' NOT FOUND.")
                if "data" in worker_response:
                    provider_logger.critical(f"  Key 'data' FOUND. Type of value: {type(worker_response.get('data'))}")
                    if isinstance(worker_response.get("data"), list):
                        provider_logger.critical(
                            f"  Value for 'data' IS a list. Length: {len(worker_response.get('data'))}")
                        if len(worker_response.get("data", [])) > 0:
                            first_item = worker_response["data"][0]
                            provider_logger.critical(f"    First item in 'data' list type: {type(first_item)}")
                            if isinstance(first_item, dict):
                                provider_logger.critical(f"    First item keys: {list(first_item.keys())}")
                                provider_logger.critical(
                                    f"    First item content (b64_json preview): {str(first_item.get('b64_json'))[:50]}...")
                            else:
                                provider_logger.critical(
                                    f"    First item in 'data' list is not a dict: {str(first_item)[:100]}")
                        else:
                            provider_logger.critical(f"  'data' list is EMPTY.")
                    else:
                        provider_logger.critical(f"  Value for 'data' IS NOT a list.")
                else:
                    provider_logger.critical(f"  Key 'data' NOT FOUND in worker_response.")
                provider_logger.critical(
                    f"generate_image_async: Full worker_response (str, first 500 chars): {str(worker_response)[:500]}")
            else:
                provider_logger.critical(
                    f"generate_image_async: worker_response is NOT a dict. Content: {str(worker_response)[:500]}")

            # Check if the worker_response itself is an error dictionary from _execute_imagination_worker
            if isinstance(worker_response, dict) and "error" in worker_response:
                error_msg = worker_response["error"]
                if interruption_error_marker in error_msg:  # Check for our specific interruption marker
                    provider_logger.warning(
                        f"🚦 Image generation task (ELP{priority}) INTERRUPTED as reported by worker execution: {error_msg}")
                    raise TaskInterruptedException(error_msg)  # Propagate as specific exception
                provider_logger.error(f"Imagination worker execution reported error at top level: {error_msg}")
                return [], f"Image generation failed: {error_msg}"  # Return empty list and error message

            # Now, process the expected success structure: {"created": ..., "data": [image_item_dict, ...]}
            if isinstance(worker_response, dict) and "data" in worker_response and isinstance(
                    worker_response.get("data"), list):
                image_data_dicts_from_data_key = []  # To store valid image items
                for item_from_worker_data in worker_response["data"]:  # item_from_worker_data should be a dict
                    if isinstance(item_from_worker_data,
                                  dict) and "b64_json" in item_from_worker_data:  # Check for essential PNG data
                        image_data_dicts_from_data_key.append(item_from_worker_data)  # Add the whole dict

                # Check if we actually extracted any valid image data dicts
                if image_data_dicts_from_data_key:
                    provider_logger.success(
                        f"✅ Successfully extracted {len(image_data_dicts_from_data_key)} image data dict(s) from worker_response['data'].")
                    return image_data_dicts_from_data_key, None  # Success: return list of dicts, no error
                else:
                    provider_logger.warning(
                        "Imagination worker's 'data' list was empty or contained no valid image items (with b64_json).")
                    return [], "Image generation worker's 'data' list was empty or invalid."
            else:
                # This means worker_response was not an error dict, but also not the expected success structure
                provider_logger.error(
                    f"Imagination worker returned unexpected JSON structure (missing 'data' list or not a dict): {str(worker_response)[:200]}...")
                return [], "Image generation worker returned an invalid response structure."

        except TaskInterruptedException as tie:
            provider_logger.warning(f"🚦 Image generation task (ELP{priority}) was interrupted: {tie}")
            return [], str(tie)  # Return empty list and the interruption message
        except Exception as e:  # Catch any other unexpected errors within generate_image_async
            provider_logger.error(f"Unhandled exception during image generation: {e}")
            provider_logger.exception("Image Generation Traceback:")  # Log full traceback
            return [], f"An unexpected error occurred during image generation: {e}"
    # --- END NEW public method ---

    def get_image_generator(self) -> Optional[Any]:
        """Returns an indicator if image generation is configured, not an instance."""
        if STABLE_DIFFUSION_WORKER_CONFIGURED:
            # Could return self or a simple status object/boolean
            # For now, let's just log and return a truthy value if app.py expects something.
            logger.debug("Image generator (worker) is configured.")
            return True # Indicate available
        logger.warning("Image generator (worker) is not configured.")
        return None

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