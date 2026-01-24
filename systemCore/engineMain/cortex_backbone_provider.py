# cortex_backbone_provider.py
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
import re
import psutil
import struct
import math
import tempfile
import base64
import signal
from loguru import logger # Logging library
from langchain_core.runnables import RunnableConfig, RunnableLambda, Runnable

try:
    import torch
except ImportError:
    torch = None

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    # The encoder itself will be loaded lazily inside the LlamaCppEmbeddingsWrapper
except ImportError:
    logger.warning("⚠️ tiktoken not installed. Embedding chunking will use less accurate character-based counting. Run 'pip install tiktoken'.")
    TIKTOKEN_AVAILABLE = False
    tiktoken = None # Placeholder

try:
    from database import add_interaction
except ImportError:
    # Fallback if database.py isn't found/circular import, prevents crash
    logger.warning("Could not import add_interaction from database. Warden flushing will be disabled.")
    add_interaction = None


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
"""try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
    logger.info("✅ llama-cpp-python imported.")
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("⚠️ llama-cpp-python not installed. Run 'pip install llama-cpp-python'. llama_cpp provider disabled.")
    llama_cpp = None # Placeholder"""
logger.warning(" llama-cpp-python is deprecated! since it is no longer updated. and now bypassed directly to binary invocation ")

# Stable Diffusion (Placeholder)
try:
    # import stable_diffusion_cpp # Or however the python bindings are imported
    STABLE_DIFFUSION_AVAILABLE = False # Set to True if import succeeds
    logger.info("✅ stable-diffusion-cpp imported")
except ImportError:
    STABLE_DIFFUSION_AVAILABLE = False
    logger.warning("⚠️ stable-diffusion-cpp bindings not found. Image generation disabled.")


try:
    from priority_lock import PriorityQuotaLock, ELP0, ELP1
    # Formally define the type for our lock variable
    LockType = Union[PriorityQuotaLock, threading.Lock]
except ImportError:
    logger.critical("❌ Failed to import PriorityQuotaLock. Priority locking disabled.")
    # Fallback to standard lock to allow basic functionality
    PriorityQuotaLock = threading.Lock # This is a class, not an instance
    LockType = threading.Lock # In the fallback case, the type is just a standard lock
    ELP0 = 0
    ELP1 = 1

# --- Local Imports ---
try:
    # Import all config variables
    from CortexConfiguration import * # Includes PROVIDER, model names, paths, TOPCAP_TOKENS etc.
except ImportError:
    logger.critical("❌ Failed to import config.py in ai_provider.py!")
    sys.exit("CortexEngine cannot function without config.")

class TaskInterruptedException(Exception):
    """Custom exception raised when an ELP0 task is interrupted by ELP1."""
    pass


#global var init

def _get_encoder():
    if TIKTOKEN_AVAILABLE and tiktoken is not None:
        try:
            # cl100k_base is the standard encoder for modern OpenAI models
            text_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.error(f"Failed to get tiktoken encoder 'cl100k_base', falling back. Error: {e}")
            try:
                # Fallback for older environments or different tiktoken versions
                text_encoder = tiktoken.encoding_for_model("gpt-4")
            except Exception as e2:
                logger.critical(f"Could not initialize any tiktoken encoder: {e2}")
    return text_encoder

def _count_tokens(texts: List[str]) -> int:
    """
    Counts the total number of tokens in a list of texts using tiktoken.
    Falls back to a character-based estimation if tiktoken is unavailable.
    """
    if not texts:
        return 0

    encoder = _get_encoder()
    if encoder:
        total_tokens = 0
        for text in texts:
            # Ensure text is a string to prevent errors
            if not isinstance(text, str):
                text = str(text)
            total_tokens += len(encoder.encode(text))
        return total_tokens
    else:
        # Fallback to character-based estimation if tiktoken failed
        total_chars = sum(len(str(text)) for text in texts)
        return total_chars // 4 # Rough but consistent fallback

def strip_initial_think_block(text: str) -> str:
    """
    Removes the first occurring <think>...</think> block if it appears at the
    beginning of the text (possibly after some leading whitespace).
    Returns the rest of the string, lstripped.
    If no such block is found at the beginning, returns the original text, lstripped.
    """
    if not isinstance(text, str):
        return ""  # Or raise TypeError, or return as is if that's preferred for non-strings

    # Regex to find <think>...</think> possibly preceded by whitespace,
    # and capture what comes AFTER it.
    # ^\s* : matches optional whitespace at the beginning of the string
    # (<think>[\s\S]*?</think>) : captures the think block (non-greedy)
    # \s* : matches optional whitespace after the think block
    # ([\s\S]*) : captures everything else that follows (the target)
    match = re.match(r"^\s*(<think>[\s\S]*?</think>)\s*([\s\S]*)", text, re.IGNORECASE)

    if match:
        think_block_content = match.group(1)  # The <think>...</think> part
        remaining_text = match.group(2)  # The part after the think block
        # logger.trace(f"Stripped initial think block. Removed: '{think_block_content[:100]}...'. Remaining: '{remaining_text[:100]}...'")
        return remaining_text.lstrip()  # Return the rest, left-stripped of any space between think and target
    else:
        # No initial <think> block found, return the original text (left-stripped)
        # logger.trace("No initial think block found to strip.")
        return text.lstrip()


class LlamaCppVisionWrapper(Runnable):
    def __init__(self, model_path, mmproj_path, provider_ref):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.provider = provider_ref

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> str:
        # 1. Adelaide sends: [HumanMessage(content=[{"type": "image_url"...}, {"type": "text"...}])]
        # We extract the content parts from the LangChain message
        messages = input if isinstance(input, list) else [input]
        prompt_text = ""
        image_b64 = None

        logger.info(f"invoked remove this later llamacppvisionwrapper debug msg metadata")

        priority = 0
        if config:
            # 1. Check metadata (The correct LangChain way)
            priority = config.get("metadata", {}).get("priority", priority)
            # 2. Check configurable (Another LangChain standard)
            if priority == 0 and "configurable" in config:
                priority = config["configurable"].get("priority", priority)
            # 3. Check top-level (Legacy/Direct way)
            if priority == 0:
                priority = config.get("priority", priority)

        n_gpu_override = None
        if config and "metadata" in config:
            n_gpu_override = config["metadata"].get("n_gpu_layers_override")

        logger.info(f"invoked remove this later llamacppvisionwrapper [Turd Code Debug] debug msg raw data caught {priority}")

        for msg in messages:
            content = getattr(msg, "content", [])
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        prompt_text = part.get("text", "")
                    elif part.get("type") == "image_url":
                        # Standard format: "data:image/png;base64,..."
                        url_val = part["image_url"].get("url", "")
                        if "base64," in url_val:
                            image_b64 = url_val.split("base64,")[1]

        # 2. Vision binaries need a physical file path
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            if image_b64:
                tmp.write(base64.b64decode(image_b64))
            temp_image_path = tmp.name

        # 3. Use the provider's execution logic to call the worker
        # We signal task_type="vision" so the worker knows to use LMMultiModal

        payload = {
            "task_type": "vision",
            "model_path": self.model_path,
            "mmproj_path": self.mmproj_path,
            "image_path": temp_image_path,
            "prompt": prompt_text,
            "kwargs": {"temperature": 0.8},
        }
        logger.info(f"llamacppvisionwrapper invoke [Turd Code Debug] payload w/ pri {priority} L payload {payload}")
        # 4. Delegate to your existing worker execution bridge
        # Assuming you have a method like _run_worker_sync in your provider
        result = self.provider._execute_in_worker(
            model_role="vlm",
            task_type="vision",
            request_data=payload,
            priority=priority,
            n_gpu_layers_override=n_gpu_override
        )

        logger.info(f"llamacppvisionwrapper invoke [Turd Code Debug] w/ pri {priority} raw result {result}")
        # 5. Cleanup
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # 6. Extract the generated text safely
        if result:
            # Handle double wrapping {'result': {'choices': ...}}
            inner = result.get("result", result)

            if isinstance(inner, dict) and "choices" in inner:
                choices = inner["choices"]
                if choices and isinstance(choices, list):
                    return choices[0]["message"].get("content", "")
        return "[Vision Bridge Error: No content returned]"

# --- Chat Model Wrapper ---
class LlamaCppChatWrapper(SimpleChatModel):
    ai_provider: 'CortexEngine'
    model_role: str
    model_kwargs: Dict[str, Any]

    def __init__(self, ai_provider: Any, model_role: str, model_kwargs: Dict[str, Any], **kwargs):
        # 2. Pass everything as KEYWORDS to the superclass (SimpleChatModel/BaseModel)
        super().__init__(
            ai_provider=ai_provider,
            model_role=model_role,
            model_kwargs=model_kwargs,
            **kwargs
        )



    def _call(
            self,
            messages: Union[List[BaseMessage], str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            # <<< THIS IS THE KEY CHANGE: Add the config parameter
            config: Optional[RunnableConfig] = None,
            **kwargs: Any,
    ) -> str:
        """
        Core method to interact with the Llama.cpp worker process.
        Handles raw string prompts (assumed to be fully formatted ChatML) and
        Langchain BaseMessage lists.
        Applies `strip_initial_think_block` to the raw LLM output.
        """
        provider_logger = getattr(self.ai_provider, 'logger', logger)
        wrapper_log_prefix = f"LlamaCppChatWrapper(Role:{self.model_role})"

        n_gpu_override = None
        if config and "metadata" in config:
            n_gpu_override = config["metadata"].get("n_gpu_layers_override")

        priority = ELP0
        # 1. Check kwargs (direct call overrides)
        if 'priority' in kwargs:
            priority = kwargs.pop('priority')
        # 2. Check config metadata (LangChain standard)
        elif config and "metadata" in config:
            priority = config["metadata"].get("priority", priority)
        # 3. Check config configurable (Another LangChain standard)
        elif config and "configurable" in config:
            priority = config["configurable"].get("priority", priority)

        is_raw_chatml_prompt_mode = isinstance(messages, str)

        provider_logger.debug(
            f"{wrapper_log_prefix}: Received _call. RawMode: {is_raw_chatml_prompt_mode}, Priority: ELP{priority}, "
            f"Incoming kwargs: {kwargs}, Stop sequences: {stop}"
        )

        final_model_kwargs_for_worker = {**self.model_kwargs, **kwargs}
        effective_stop_sequences = list(stop) if stop is not None else []
        if CHATML_END_TOKEN not in effective_stop_sequences:
            effective_stop_sequences.append(CHATML_END_TOKEN)
        final_model_kwargs_for_worker["stop"] = effective_stop_sequences

        provider_logger.trace(
            f"{wrapper_log_prefix}: Final model kwargs for worker (incl. merged stop sequences): {final_model_kwargs_for_worker}")

        request_payload: Dict[str, Any]
        task_type_for_worker: str

        if is_raw_chatml_prompt_mode:
            request_payload = {"prompt": messages, "kwargs": final_model_kwargs_for_worker}
            task_type_for_worker = "raw_text_completion"
            provider_logger.debug(
                f"{wrapper_log_prefix}: Prepared for 'raw_text_completion'. Prompt len: {len(messages)}")
        else:
            provider_logger.debug(f"{wrapper_log_prefix}: Formatting List[BaseMessage] for 'chat' task type.")
            formatted_messages_for_worker = self._format_messages_for_llama_cpp(messages)
            request_payload = {"messages": formatted_messages_for_worker, "kwargs": final_model_kwargs_for_worker}
            task_type_for_worker = "chat"

        try:
            provider_logger.debug(
                f"{wrapper_log_prefix}: Delegating to _execute_in_worker (Task: {task_type_for_worker}, Priority: ELP{priority})...")
            worker_result = self.ai_provider._execute_in_worker(
                model_role=self.model_role,
                task_type=task_type_for_worker,
                request_data=request_payload,
                priority=priority,
                n_gpu_layers_override=n_gpu_override
            )

            if not worker_result or not isinstance(worker_result, dict):
                err_msg = f"Worker execution failed or returned invalid data type: {type(worker_result)}"
                provider_logger.error(f"{wrapper_log_prefix}: {err_msg}")
                return f"[{self._llm_type.upper()}_PROVIDER_ERROR: {err_msg}]"

            if "error" in worker_result:
                error_msg_content = worker_result['error']
                if interruption_error_marker in error_msg_content:
                    provider_logger.warning(
                        f"🚦 {wrapper_log_prefix}: Task INTERRUPTED (marker from worker '{error_msg_content}'). Raising TaskInterruptedException.")
                    raise TaskInterruptedException(error_msg_content)

                error_msg_to_return = f"{self._llm_type.upper()}_WORKER_ERROR (Role:{self.model_role}): {error_msg_content}"
                provider_logger.error(f"{wrapper_log_prefix}: {error_msg_to_return}")
                return f"[{error_msg_to_return}]"

            if "result" not in worker_result:
                err_msg = f"Worker returned unknown dictionary structure (missing 'result' key): {str(worker_result)[:200]}..."
                provider_logger.error(f"{wrapper_log_prefix}: {err_msg}")
                return f"[{self._llm_type.upper()}_WORKER_RESPONSE_ERROR: {err_msg}]"

            completion_data = worker_result["result"]
            raw_response_content_from_llm_core: str = ""

            if task_type_for_worker == "raw_text_completion":
                if (completion_data and isinstance(completion_data, dict) and
                        'choices' in completion_data and isinstance(completion_data['choices'], list) and
                        completion_data['choices'] and isinstance(completion_data['choices'][0], dict) and
                        'text' in completion_data['choices'][0]):
                    raw_response_content_from_llm_core = completion_data['choices'][0]['text']
                else:
                    err_msg = f"Worker (raw_text_completion) returned unexpected result structure: {str(completion_data)[:200]}..."
                    provider_logger.error(f"{wrapper_log_prefix}: {err_msg}")
                    return f"[{self._llm_type.upper()}_WORKER_RESPONSE_ERROR: {err_msg}]"
            elif task_type_for_worker == "chat":
                if (completion_data and isinstance(completion_data, dict) and
                        'choices' in completion_data and isinstance(completion_data['choices'], list) and
                        completion_data['choices'] and isinstance(completion_data['choices'][0], dict) and
                        'message' in completion_data['choices'][0] and
                        isinstance(completion_data['choices'][0]['message'], dict) and
                        'content' in completion_data['choices'][0]['message']):
                    raw_response_content_from_llm_core = completion_data['choices'][0]['message']['content']
                else:
                    err_msg = f"Worker (chat) returned unexpected result structure: {str(completion_data)[:200]}..."
                    provider_logger.error(f"{wrapper_log_prefix}: {err_msg}")
                    return f"[{self._llm_type.upper()}_WORKER_RESPONSE_ERROR: {err_msg}]"

            if not isinstance(raw_response_content_from_llm_core, str):
                raw_response_content_from_llm_core = str(raw_response_content_from_llm_core or "")

            provider_logger.trace(
                f"{wrapper_log_prefix}: Raw content from LLM core (len={len(raw_response_content_from_llm_core)}): '{raw_response_content_from_llm_core[:150]}...'")

            final_content_after_think_strip = strip_initial_think_block(raw_response_content_from_llm_core)
            if len(final_content_after_think_strip) != len(raw_response_content_from_llm_core.lstrip()):
                provider_logger.info(
                    f"{wrapper_log_prefix}: Applied strip_initial_think_block. Result len: {len(final_content_after_think_strip)}")
            provider_logger.trace(
                f"{wrapper_log_prefix}: Content after strip_initial_think_block: '{final_content_after_think_strip[:150]}...'")

            if final_content_after_think_strip.endswith(CHATML_END_TOKEN):
                final_content_after_think_strip = final_content_after_think_strip[:-len(CHATML_END_TOKEN)]

            response_to_return = final_content_after_think_strip.strip()

            provider_logger.debug(
                f"{wrapper_log_prefix}: Successfully extracted and cleaned content (len:{len(response_to_return)}). Returning.")
            return response_to_return

        except TaskInterruptedException:
            provider_logger.warning(f"🚦 {wrapper_log_prefix}: TaskInterruptedException caught in _call. Re-raising.")
            raise
        except Exception as e_call:
            provider_logger.error(f"{wrapper_log_prefix}: Unexpected error in _call: {e_call}")
            provider_logger.exception(f"{wrapper_log_prefix} _call Traceback:")
            return f"[{self._llm_type.upper()}_WRAPPER_ERROR: {type(e_call).__name__} - {str(e_call)[:100]}]"

    #def _stream is no longer used! and removed!

    # _format_messages_for_llama_cpp might become unused if all calls are raw ChatML strings.
    # Keep it for now for potential compatibility if some parts still use List[BaseMessage].

    # Keep formatter, it's still useful
    def _format_messages_for_llama_cpp(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
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
            role = "user"
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
    ai_provider: 'CortexEngine'
    model_role: str = "embeddings"

    def __init__(self, ai_provider: 'CortexEngine'):
        super().__init__()
        self.ai_provider = ai_provider



    def embed_documents(self, texts: List[str], priority: int = ELP0) -> List[List[float]]:
        provider_logger = getattr(self.ai_provider, 'logger', logger)
        # This call will use the default priority (ELP0) set in _embed_texts's signature
        provider_logger.debug(f"EmbedWrapper.embed_documents: Standard call for {len(texts)} texts (delegating to _embed_texts with its default ELP{priority}).")
        return self._embed_texts(texts, priority=priority) # Default priority ELP0 will be used by _embed_texts

    # Standard Langchain interface method - uses default priority ELP0
    def embed_query(self, text: str, priority: int = ELP0) -> List[float]:
        provider_logger = getattr(self.ai_provider, 'logger', logger)
        log_prefix = f"EmbedWrapper.embed_query|ELP{priority}"
        
        if not isinstance(text, str):
            text = str(text) # Ensure input is a string

        # --- SAFEGUARD LOGIC ---
        # Check if this single query text exceeds the token limit.
        token_count = _count_tokens([text])
        
        text_to_embed = text
        if token_count > MAX_EMBEDDING_TOKENS_PER_BATCH:
            provider_logger.warning(
                f"⚠️ {log_prefix}: Input query text ({token_count} tokens) exceeds the single-batch limit "
                f"of {MAX_EMBEDDING_TOKENS_PER_BATCH}. The query will be truncated to prevent a worker crash."
            )
            
            # Truncate the text. We use a simple character-based truncation as a fallback.
            # A more precise method could use tiktoken to encode, slice, and decode.
            if TIKTOKEN_AVAILABLE and _get_encoder():
                encoder = _get_encoder()
                tokens = encoder.encode(text)
                truncated_tokens = tokens[:MAX_EMBEDDING_TOKENS_PER_BATCH]
                text_to_embed = encoder.decode(truncated_tokens, errors='ignore')
            else:
                # Fallback to character-based truncation if tiktoken isn't available
                safe_chars = int(MAX_EMBEDDING_TOKENS_PER_BATCH * 3.5) # Estimate
                text_to_embed = text[:safe_chars]
            
            provider_logger.warning(f"{log_prefix}: Truncated query text length: {len(text_to_embed)} characters.")

        # --- END SAFEGUARD ---

        provider_logger.debug(f"{log_prefix}: Delegating single query embedding to _embed_texts.")
        # Delegate the (potentially truncated) text to the core _embed_texts method.
        results = self._embed_texts([text_to_embed], priority=priority)
        
        if results and isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            return results[0]
            
        provider_logger.error(f"{log_prefix}: _embed_texts did not return a valid vector for the query. Result: {results}")
        raise RuntimeError("Embedding query failed to produce a valid vector via _embed_texts.")

    def _embed_large_texts_in_batches(self, texts: List[str], priority: int = ELP0) -> List[List[float]]:
        """
        Handles embedding for a list of texts that exceeds the token limit by
        splitting it into multiple smaller batches and processing them sequentially.
        Includes a safeguard to truncate individual texts that are too long.
        """
        provider_logger = getattr(self.ai_provider, 'logger', logger)
        log_prefix = f"EmbedBatcher|ELP{priority}"

        all_embeddings: List[List[float]] = []
        current_batch: List[str] = []
        current_batch_tokens = 0
        total_batches = 0

        for i, text in enumerate(texts):
            text_str = str(text) if not isinstance(text, str) else text
            text_tokens = _count_tokens([text_str])
            
            # =================== THIS IS THE FIX ===================
            # Safeguard: If a single text item (chunk) is larger than the entire
            # batch limit, we must truncate it to prevent an error.
            if text_tokens > MAX_EMBEDDING_TOKENS_PER_BATCH:
                provider_logger.warning(
                    f"⚠️ {log_prefix}: A single text item (index {i}) has {text_tokens} tokens, "
                    f"which exceeds the batch limit of {MAX_EMBEDDING_TOKENS_PER_BATCH}. "
                    f"It will be truncated."
                )
                
                # Use tiktoken for precise truncation if available
                encoder = _get_encoder()
                if TIKTOKEN_AVAILABLE and encoder:
                    tokens = encoder.encode(text_str)
                    truncated_tokens = tokens[:MAX_EMBEDDING_TOKENS_PER_BATCH]
                    text_str = encoder.decode(truncated_tokens, errors='ignore')
                else:
                    # Fallback to character-based truncation
                    safe_chars = int(MAX_EMBEDDING_TOKENS_PER_BATCH * 3.5) # Estimate
                    text_str = text_str[:safe_chars]
                
                # Recalculate the token count after truncation
                text_tokens = _count_tokens([text_str])
                provider_logger.warning(f"{log_prefix}: Truncated item now has {text_tokens} tokens.")
            # ================= END OF THE FIX ===================

            # Check if adding the new text would push the current batch over the limit.
            if current_batch_tokens + text_tokens > MAX_EMBEDDING_TOKENS_PER_BATCH and current_batch:
                # The current batch is full. Process it.
                total_batches += 1
                provider_logger.info(
                    f"{log_prefix}: Processing batch #{total_batches} with {len(current_batch)} texts "
                    f"({current_batch_tokens} tokens)."
                )
                batch_embeddings = self._embed_texts(current_batch, priority=priority)
                all_embeddings.extend(batch_embeddings)
                
                # Reset for the new batch
                current_batch = []
                current_batch_tokens = 0

            # Add the current text to the batch.
            current_batch.append(text_str)
            current_batch_tokens += text_tokens

        # Process the final batch if any texts are left over.
        if current_batch:
            total_batches += 1
            provider_logger.info(
                f"{log_prefix}: Processing final batch #{total_batches} with {len(current_batch)} texts "
                f"({current_batch_tokens} tokens)."
            )
            batch_embeddings = self._embed_texts(current_batch, priority=priority)
            all_embeddings.extend(batch_embeddings)

        provider_logger.success(f"✅ {log_prefix}: Finished processing all {total_batches} batches. "
                          f"Returning a total of {len(all_embeddings)} vectors.")
        
        return all_embeddings

    def _embed_texts(self, texts: List[str], priority: int = ELP0) -> List[List[float]]:
        provider_logger = getattr(self.ai_provider, 'logger', logger)
        log_prefix = f"EmbedWrapper._embed_texts|ELP{priority}"

        if not texts:
            provider_logger.debug(f"{log_prefix}: Received empty list of texts. Returning empty list.")
            return []

        # --- GATEKEEPER LOGIC ---
        # Count the tokens of the incoming list of texts.
        total_tokens = _count_tokens(texts)
        provider_logger.debug(f"{log_prefix}: Received {len(texts)} texts with a total of {total_tokens} tokens.")

        # Compare against the safe limit.
        if total_tokens <= MAX_EMBEDDING_TOKENS_PER_BATCH:
            # If within the limit, proceed with the original single-batch logic.
            provider_logger.debug(f"{log_prefix}: Token count is within the batch limit ({MAX_EMBEDDING_TOKENS_PER_BATCH}). Processing as a single batch.")

            request_payload = {"texts": texts}
            worker_result = self.ai_provider._execute_in_worker(
                model_role=self.model_role,
                task_type="embedding",
                request_data=request_payload,
                priority=priority
            )
            
            # (The existing error handling and result parsing logic remains the same for the single batch)
            if worker_result and isinstance(worker_result, dict):
                if "error" in worker_result:
                    error_msg_content = worker_result['error']
                    if interruption_error_marker in error_msg_content:
                        provider_logger.warning(f"🚦 {log_prefix}: Embedding task INTERRUPTED: {error_msg_content}")
                        raise TaskInterruptedException(error_msg_content)
                    else:
                        full_error_msg = f"LLAMA_CPP_EMBED_WORKER_ERROR ({self.model_role}|ELP{priority}): {error_msg_content}"
                        provider_logger.error(full_error_msg)
                        raise RuntimeError(full_error_msg)
                elif "result" in worker_result and isinstance(worker_result["result"], list):
                    batch_embeddings = worker_result["result"]
                    if all(isinstance(emb, list) for emb in batch_embeddings):
                        provider_logger.debug(f"{log_prefix}: Single-batch embedding successful. Returning {len(batch_embeddings)} vectors.")
                        return [[float(num) for num in emb] for emb in batch_embeddings]
                    else:
                        provider_logger.error(f"{log_prefix}: Worker returned invalid embedding structure.")
                        raise RuntimeError("LLAMA_CPP_EMBED_WORKER_RESPONSE_ERROR: Invalid embedding structure.")
                else:
                    provider_logger.error(f"{log_prefix}: Worker returned unknown dictionary structure: {str(worker_result)[:200]}...")
                    raise RuntimeError("LLAMA_CPP_EMBED_WORKER_RESPONSE_ERROR: Unknown dictionary structure.")
            else:
                provider_logger.error(f"{log_prefix}: Worker execution failed or returned invalid data type: {type(worker_result)}")
                raise RuntimeError("LLAMA_CPP_EMBED_PROVIDER_ERROR: Worker execution failed.")

        else:
            # If the limit is exceeded, delegate to the new batching method.
            provider_logger.warning(
                f"⚠️ {log_prefix}: Token count ({total_tokens}) exceeds batch limit ({MAX_EMBEDDING_TOKENS_PER_BATCH}). "
                f"Delegating to batch processing."
            )
            # This method will be created in the next step. Python allows us to reference it here.
            return self._embed_large_texts_in_batches(texts, priority=priority)

# === AI Provider Class ===
class CortexEngine:
    _priority_quota_lock: LockType
    def __init__(self, provider_name):
        self._call_llm_with_timing = None
        self.provider_name = provider_name.lower()
        self.models: Dict[str, Any] = {}
        self.embeddings: Optional[Embeddings] = None
        self.EMBEDDINGS_MODEL_NAME: Optional[str] = None
        self.embeddings: Optional[LlamaCppEmbeddingsWrapper] = None  # NEW, if always this type
        self.active_workers: Dict[int, subprocess.Popen] = {}
        self.active_workers_lock = threading.Lock()
        self.setup_provider()
        # self.image_generator: Any = None # This will be implicitly handled by calling the worker

        # This is connected with the deprecated llama-cpp-python binder that is outdated and can't build with the latest llama.cpp binaries
        #self._loaded_llama_instance: Optional[llama_cpp.Llama] = None
        #self._loaded_gguf_path: Optional[str] = None
        #self._last_task_type: Optional[str] = None
        self.ctx_bins = INFERCOMPLETION_CTX_BINNING #Imported from CortexConfiguration *
        self.safe_pct = SYSTEMBUFFER_SAFE_PERCENTAGE #Imported from CortexConfiguration *

        self.loop = None # For scheduler

        # --- llama.cpp specific state ---
        self._llama_model_access_lock = threading.Lock() if self.provider_name == "llama_cpp" else None
        if self.provider_name == "llama_cpp":
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
            logger.info(f"CortexEngine embeddings initialized. Type: {type(self.embeddings)}")
            if hasattr(self.embeddings, 'embed_query') and hasattr(self.embeddings, 'embed_documents'):
                logger.info("  CortexEngine embeddings object HAS 'embed_query' and 'embed_documents' methods.")
            else:
                logger.error("  CRITICAL: CortexEngine embeddings object LACKS required embedding methods!")
        else:
            logger.error("CRITICAL: CortexEngine.embeddings is None after setup_provider!")
        self._setup_image_generator_config() # Renamed: Validates image gen worker config

    def get_model_path(self, role: str) -> str:
        """
        Helper: Returns the absolute path to the GGUF file for a given role.
        Used by the Resource Warden to probe file metadata before loading.
        """
        # Ensure we are in a mode that supports GGUF paths
        if self.provider_name != "llama_cpp" or not getattr(self, '_llama_model_map', None):
            return ""

        # 1. Resolve Filename from Map
        filename = self._llama_model_map.get(role)

        # Simple Fallback: if 'general' is requested but not mapped, try 'default'
        if not filename and role == "general":
            filename = self._llama_model_map.get("default")

        if not filename:
            # logger.error(f"get_model_path: No file mapped for role '{role}'")
            return ""

        # 2. Join with Directory
        # self._llama_gguf_dir is initialized in setup_provider
        return os.path.join(self._llama_gguf_dir, filename)

    def _estimate_tokens_fast(self, text: str) -> int:
        """
        Agnostic token estimator. Uses Tiktoken if available (accurate),
        falls back to char/3.5 (approximate) to avoid heavy deps.
        """
        if not text: return 0

        # Try accurate count if Embeddings wrapper loaded tiktoken
        if self.embeddings and hasattr(self.embeddings, '_get_encoder'):
            enc = _get_encoder()
            if enc:
                try:
                    return len(enc.encode(text))
                except:
                    pass  # Fallback to math

        # Fallback: Rule of thumb for English/Code (3.5 chars per token)
        return int(len(text) / 3.5)

    def probe_gguf_metadata(self, model_path: str) -> Dict:
        """
        Robustly reads GGUF header to get 'context_length', 'file_type', etc.
        Skips large arrays safely to avoid crashing.
        """
        metadata = {"valid": False, "ctx_train": 2048, "type_id": 0, "file_size_gb": 0.0}

        if not os.path.exists(model_path): return metadata
        metadata["file_size_gb"] = os.path.getsize(model_path) / (1024 ** 3)

        try:
            with open(model_path, 'rb') as f:
                if f.read(4) != b'GGUF': return metadata
                f.read(4)  # Version
                f.read(8)  # Tensor Count
                kv_count = struct.unpack('<Q', f.read(8))[0]

                for _ in range(kv_count):
                    # Read Key
                    len_k = struct.unpack('<Q', f.read(8))[0]
                    key = f.read(len_k).decode('utf-8', errors='ignore')

                    # Read Type
                    val_type = struct.unpack('<I', f.read(4))[0]

                    # Read Value (with Array Skip logic)
                    if val_type == 8:  # String
                        l = struct.unpack('<Q', f.read(8))[0];
                        f.seek(l, 1)
                    elif val_type == 9:  # Array
                        type_a = struct.unpack('<I', f.read(4))[0]
                        len_a = struct.unpack('<Q', f.read(8))[0]
                        sz = 1
                        if type_a in [4, 5, 6]:
                            sz = 4
                        elif type_a in [10, 11, 12]:
                            sz = 8
                        if type_a == 8:  # String Array - Slow skip
                            for _ in range(len_a):
                                l = struct.unpack('<Q', f.read(8))[0];
                                f.seek(l, 1)
                        else:  # Scalar Array - Fast skip
                            f.seek(sz * len_a, 1)
                    else:  # Scalars
                        sz = 4 if val_type not in [10, 11, 12] else 8
                        if val_type in [0, 1, 7]: sz = 1
                        val_bytes = f.read(sz)

                        if "context_length" in key and "rope" not in key:
                            metadata["ctx_train"] = struct.unpack('<I' if sz == 4 else '<Q', val_bytes)[0]
                        elif key == "general.file_type":
                            metadata["type_id"] = struct.unpack('<I', val_bytes)[0]

                metadata["valid"] = True
        except Exception as e:
            logger.error(f"GGUF Probe Error: {e}")

        return metadata

    def get_memory_status(self) -> dict:
        """
        Returns memory stats.
        Fixes the '0.0GB' issue on macOS by trusting 'available' more.
        """
        vm = psutil.virtual_memory()
        
        # 1. Get raw available (OS estimate of what can be given to a new app)
        # On macOS, this correctly includes file cache that can be evicted.
        real_available_gb = vm.available / (1024 ** 3)
        
        # 2. Define Safety Buffer
        # OLD LOGIC: subtracted 4GB+ regardless, causing 0.0GB result.
        # NEW LOGIC: Dynamic buffer.
        
        # If we have lots of RAM (>32GB total), we can afford a 4GB buffer.
        # If we are tight (e.g., 16GB or 8GB laptop), a 4GB buffer is suicide for the calculation.
        total_ram_gb = vm.total / (1024 ** 3)
        
        if total_ram_gb > 32:
            safety_buffer = 4.0 
        elif total_ram_gb > 16:
            safety_buffer = 2.0
        else:
            # On 8GB/16GB machines, trust the OS 'available' metric almost fully.
            # Just keep 500MB for kernel panics.
            safety_buffer = 0.5 

        # 3. Calculate "Safe" Available
        safe_available_gb = real_available_gb - safety_buffer
        
        # 4. Floor it at 0.5GB instead of 0.0
        # If the OS says we have space, we likely do. Don't block completely unless critical.
        if safe_available_gb < 0.5:
             # If we are truly negative, it means we are swapping heavily.
             # We report a tiny amount to force the Warden to choose the smallest bin,
             # but we don't report 0.0 which looks like an error.
             safe_available_gb = 0.5

        return {
            "total_gb": total_ram_gb,
            "available_gb": real_available_gb,
            "safe_available_gb": round(safe_available_gb, 2),
            "percent_used": vm.percent
        }

    def calculate_required_memory_gb(self, model_path: str, ctx_bin: int) -> float:
        """Calculates expected RAM usage (Weights + KV Cache) for a specific bin."""
        meta = self.probe_gguf_metadata(model_path)

        # 1. Weights (Static) - Approx File Size + 5% overhead
        weights_gb = meta["file_size_gb"] * 1.05

        # 2. KV Cache (Dynamic) - 0.5 MB/token upper bound (F16)
        kv_gb = (ctx_bin * 0.5) / 1024

        # 3. Compute Overhead
        overhead_gb = 0.5

        return weights_gb + kv_gb + overhead_gb

    def get_ideal_bin_for_text(self, text_input: str) -> int:
        """Finds the smallest bin from config that fits the input text."""
        # Estimate tokens (1 token ~= 3.5 chars) + Buffer
        est_tokens = int(len(text_input) / 3.5) + 512

        sorted_bins = sorted(self.ctx_bins)
        for b in sorted_bins:
            if b >= est_tokens:
                return b
        return sorted_bins[-1]  # Cap at max

    async def save_llama_session_async(self, model_role: str, state_name: str, priority: int = ELP0) -> Dict[str, Any]:
        """
        Asynchronously requests the llama_worker to save its current KV cache state.

        Args:
            model_role: The logical role of the model whose state should be saved.
            state_name: The name to save the state file as (e.g., "my_conversation").
            priority: The execution priority for the task.

        Returns:
            A dictionary with the result from the worker.
        """
        log_prefix = f"KV_SAVE|ELP{priority}|{model_role}"
        logger.info(f"{log_prefix}: Requesting to save session state '{state_name}'.")
        
        request_data = {
            "state_name": state_name
        }
        
        # We use asyncio.to_thread because _execute_in_worker is synchronous
        result = await asyncio.to_thread(
            self._execute_in_worker,
            model_role=model_role,
            task_type="save_kv_cache",
            request_data=request_data,
            priority=priority
        )
        return result or {"error": "Failed to get response from save_kv_cache worker task."}
    
    async def load_llama_session_async(self, model_role: str, state_name: str, priority: int = ELP0) -> Dict[str, Any]:
        """
        Asynchronously requests the llama_worker to load a KV cache state.

        Args:
            model_role: The logical role of the model to load the state into.
            state_name: The name of the state file to load.
            priority: The execution priority for the task.

        Returns:
            A dictionary with the result from the worker.
        """
        log_prefix = f"KV_LOAD|ELP{priority}|{model_role}"
        logger.info(f"{log_prefix}: Requesting to load session state '{state_name}'.")

        request_data = {
            "state_name": state_name
        }

        result = await asyncio.to_thread(
            self._execute_in_worker,
            model_role=model_role,
            task_type="load_kv_cache",
            request_data=request_data,
            priority=priority
        )
        return result or {"error": "Failed to get response from load_kv_cache worker task."}

    def kill_all_workers(self, reason: str):
        """
        Forcefully terminates all tracked worker processes. This is a critical
        shutdown mechanism for severe errors like performance timeouts.
        """
        logger.critical(f"--- KILL_ALL_WORKERS TRIGGERED --- REASON: {reason} ---")
        with self.active_workers_lock:
            if not self.active_workers:
                logger.warning("kill_all_workers called, but no active workers were tracked.")
                return

            pids_to_kill = list(self.active_workers.keys())
            logger.warning(f"Attempting to forcefully terminate {len(pids_to_kill)} worker processes: {pids_to_kill}")

            for pid in pids_to_kill:
                proc = self.active_workers.get(pid)
                if proc and proc.poll() is None:  # Check if process exists and is running
                    try:
                        logger.info(f"Killing PID: {pid}...")
                        if os.name == 'nt':
                            # Forcefully terminate the process and its entire tree on Windows
                            subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], check=False, capture_output=True)
                        else:
                            # Send SIGKILL on Unix-like systems for forceful termination
                            os.kill(pid, signal.SIGKILL)
                        logger.info(f"Termination signal sent to PID: {pid}.")
                    except Exception as e:
                        logger.error(f"Error while trying to kill process PID {pid}: {e}")

            self.active_workers.clear()
            logger.critical("--- All tracked worker processes have been targeted for termination. ---")

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

    def is_resource_busy_with_high_priority(self) -> bool:
        """
        Checks if the shared resource lock is currently held by an ELP1 task
        or if ELP1 tasks are actively waiting.
        """
        if hasattr(self, '_priority_quota_lock') and self._priority_quota_lock:
            # Type hint for clarity if PriorityQuotaLock is imported properly
            lock_instance: Optional[PriorityQuotaLock] = self._priority_quota_lock  # type: ignore

            if lock_instance and isinstance(lock_instance, PriorityQuotaLock):  # Check it's the right type
                is_locked, holder_priority, _ = lock_instance.get_status()
                # Check elp1_waiting_count directly if exposed, or infer
                # For now, let's assume if it's locked by ELP1, it's busy with high priority.
                # A more advanced check could see lock_instance._elp1_waiting_count > 0
                if is_locked and holder_priority == ELP1:
                    return True
                # If you add a method to PriorityQuotaLock like `get_elp1_waiting_count()`
                # if lock_instance.get_elp1_waiting_count() > 0:
                #     return True
        return False

    # <<< --- NEW: Worker Execution Method --- >>>
    def _execute_in_worker(self, model_role: str, task_type: str, request_data: Dict[str, Any], priority: int = ELP0, n_gpu_layers_override: Optional[int] = None) -> \
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
                final_gpu_layers = str(LLAMA_CPP_N_GPU_LAYERS)
                if n_gpu_layers_override is not None:
                    final_gpu_layers = str(n_gpu_layers_override)
                    provider_logger.info(f"{worker_log_prefix}: 🛡️ WARDEN OVERRIDE: Setting n_gpu_layers={final_gpu_layers}")

                command = [
                    self._python_executable,
                    worker_script_path,
                    "--model-path", model_path,
                    "--task-type", task_type,
                    "--n-gpu-layers", final_gpu_layers,
                ]
                # === MODIFIED n_ctx LOGIC ===
                if task_type == "embedding":
                    # Embeddings always get a fixed n_ctx (e.g., 512, or LLAMA_CPP_N_CTX if that was intended for override)
                    # Let's assume a small fixed value is best for embeddings unless an override is in LLAMA_CPP_MODEL_MAP for the embedding model specifically.
                    embedding_n_ctx = self._llama_model_map.get(f"{model_role}_n_ctx",
                                                                4096)  # Check for specific override like "embeddings_n_ctx"
                    command.extend(["--n-ctx", str(embedding_n_ctx)])
                    provider_logger.info(
                        f"{worker_log_prefix}: Passing fixed --n-ctx {embedding_n_ctx} for embeddings.")
                elif LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT is not None:  # Check if config.py has an explicit override value
                    # LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT needs to be defined in config.py, e.g. can be os.getenv("LLAMA_CPP_N_CTX_OVERRIDE_FOR_CHAT", None)
                    # If it's set (e.g. to 4096 from CortexConfiguration), pass it to the worker as an override.
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
                if worker_process and worker_process.pid:
                    with self.active_workers_lock:
                        self.active_workers[worker_process.pid] = worker_process
                    logger.debug(f"Registered worker PID {worker_process.pid} for role {model_role}")
                if priority == ELP0 and isinstance(self._priority_quota_lock, PriorityQuotaLock):
                    self._priority_quota_lock.set_holder_process(worker_process)

                input_json = json.dumps(request_data)
                provider_logger.debug(
                    f"{worker_log_prefix}: Sending input JSON (len={len(input_json)}) to worker stdin...")
                stdout_data, stderr_data = "", ""

                try:
                    # Use LLAMA_WORKER_TIMEOUT from CortexConfiguration
                    stdout_data, stderr_data = worker_process.communicate(input=input_json,
                                                                          timeout=LLAMA_WORKER_TIMEOUT)
                    provider_logger.debug(f"{worker_log_prefix}: Worker communicate() finished.")
                except subprocess.TimeoutExpired:
                    provider_logger.error(
                        f"{worker_log_prefix}: Worker process timed out after {LLAMA_WORKER_TIMEOUT}s.")
                    worker_process.kill()
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
                if worker_process and worker_process.pid:
                    with self.active_workers_lock:
                        self.active_workers.pop(worker_process.pid, None)  # Remove on clean exit
                    logger.debug(f"De-registered worker PID {worker_process.pid}")
                provider_logger.info(f"{worker_log_prefix}: Releasing worker execution lock.")
                self._priority_quota_lock.release()
        else:
            provider_logger.error(f"{worker_log_prefix}: FAILED to acquire worker lock.")
            return {"error": "Failed to acquire execution lock for worker."}

    async def _execute_imagination_worker(
            self,
            prompt: str,
            image_base64: Optional[str] = None,
            priority: int = ELP0,
            apply_watermark: bool = False
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
            # Config constants like IMAGE_GEN_MODEL_DIR are global from CortexConfiguration.py

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

                    if apply_watermark and ENABLE_USER_IMAGE_WATERMARK: command.append("--apply-watermark")
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
                        cwd=os.path.dirname(__file__)  # Run worker from CortexEngine's directory
                    )
                    if current_worker_process and current_worker_process.pid:
                        with self.active_workers_lock:
                            self.active_workers[current_worker_process.pid] = current_worker_process
                        provider_logger.debug(f"Registered imagination worker PID {current_worker_process.pid}")

                    # If it's a background priority task, register its process with the lock
                    if priority == ELP0 and isinstance(self._priority_quota_lock, PriorityQuotaLock):
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
                    if current_worker_process and current_worker_process.pid:
                        with self.active_workers_lock:
                            self.active_workers.pop(current_worker_process.pid, None)
                        provider_logger.debug(f"De-registered imagination worker PID {current_worker_process.pid}")
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


    def setup_provider(self):
        """Sets up the AI models based on the configured PROVIDER."""
        logger.info(f"🔌 Configuring AI Provider: {self.provider_name}")
        start_time = time.monotonic()  # Use monotonic for duration

        try:
            if self.provider_name == "ollama":
                # This provider is marked as no longer supported in your file.
                logger.error(f"PROVIDER 'ollama' IS NO LONGER SUPPORTED in this application configuration.")
                logger.warning("Please switch to a supported provider (e.g., 'llama_cpp').")
                # Fallback or exit if Ollama was critical
                # For now, we'll let it proceed but models/embeddings will be None
                self.models = {}
                self.embeddings = None
                # sys.exit("Ollama provider selected but is no longer supported.") # Alternative: exit

            elif self.provider_name == "fireworks":
                # This provider is marked as no longer supported in your file.
                logger.error(f"PROVIDER 'fireworks' IS NO LONGER SUPPORTED in this application configuration.")
                logger.warning("Please switch to a supported provider (e.g., 'llama_cpp').")
                self.models = {}
                self.embeddings = None
                # sys.exit("Fireworks provider selected but is no longer supported.") # Alternative: exit


            elif self.provider_name == "llama_cpp":

                # We no longer check for the llama-cpp-python library (LLAMA_CPP_AVAILABLE).

                # Instead, we rely on the llama-cli binary located in the environment path.

                self._llama_gguf_dir = LLAMA_CPP_GGUF_DIR  # from CortexConfiguration.py

                self._llama_model_map = LLAMA_CPP_MODEL_MAP  # from CortexConfiguration.py

                self._python_executable = sys.executable  # Python executable for workers

                if not self._llama_gguf_dir or not os.path.isdir(self._llama_gguf_dir):
                    raise FileNotFoundError(

                        f"Llama.cpp GGUF directory not found or not a directory: {self._llama_gguf_dir}")

                # --- Verify Direct Binaries Exist ---

                # These were installed to the RuntimeVenv/bin by launcher.py

                conda_bin_dir = os.path.dirname(self._python_executable)

                cli_bin_name = "llama-cli" if os.name != "nt" else "llama-cli.exe"

                embed_bin_name = "llama-embedding" if os.name != "nt" else "llama-embedding.exe"

                if not os.path.isfile(os.path.join(conda_bin_dir, cli_bin_name)):
                    logger.warning(
                        f"⚠️ {cli_bin_name} not found in {conda_bin_dir}. Ensure launcher.py finished the direct build.")

                if not os.path.isfile(os.path.join(conda_bin_dir, embed_bin_name)):
                    logger.warning(
                        f"⚠️ {embed_bin_name} not found in {conda_bin_dir}. Ensure launcher.py finished the direct build.")

                # --- Setup Embeddings ---

                self.EMBEDDINGS_MODEL_NAME = self._llama_model_map.get("embeddings")

                if self.EMBEDDINGS_MODEL_NAME:

                    logger.info(

                        f"Setting up llama.cpp embeddings using role 'embeddings' (File: {self.EMBEDDINGS_MODEL_NAME})")

                    # The wrapper now handles delegation to llama-embedding via the worker process

                    self.embeddings = LlamaCppEmbeddingsWrapper(ai_provider=self)

                    logger.info(f"LlamaCppEmbeddingsWrapper initialized for role 'embeddings'.")

                else:

                    logger.error(

                        "❌ No GGUF file specified for 'embeddings' role in LLAMA_CPP_MODEL_MAP. Embeddings disabled.")

                    self.embeddings = None

                # --- Setup Chat Models (Wrappers only) ---

                logger.info("Creating llama.cpp chat wrappers for configured roles...")

                default_temp = DEFAULT_LLM_TEMPERATURE  # from CortexConfiguration.py

                for role, gguf_filename_in_map in self._llama_model_map.items():

                    if role == "embeddings":  # Skip embeddings role for chat models

                        continue

                    # Determine model_kwargs for this role

                    role_specific_kwargs = {"temperature": default_temp}

                    # Roles outputting structured JSON require unlimited max_tokens (-1)

                    system_task_roles = ["router", "action_analyzer", "classifier", "tot_json_formatter",

                                         "deep_translation_analyzer_role"]

                    if role in system_task_roles:

                        role_specific_kwargs["max_tokens"] = -1

                        logger.info(f"Setting max_tokens=-1 (unlimited) for system role: '{role}'")

                    else:

                        role_specific_kwargs["max_tokens"] = TOPCAP_TOKENS

                        logger.info(f"Setting max_tokens={TOPCAP_TOKENS} for role: '{role}'")

                    # Initialize the wrapper; _call will handle the subprocess execution

                    logger.debug(
                        f"  Creating LlamaCppChatWrapper for role '{role}' (model file '{gguf_filename_in_map}')")

                    self.models[role] = LlamaCppChatWrapper(

                        ai_provider=self,

                        model_role=role,

                        model_kwargs=role_specific_kwargs

                    )

                # Assign the default chat model wrapper explicitly

                default_chat_role_key = MODEL_DEFAULT_CHAT_LLAMA_CPP  # e.g., "general"

                if default_chat_role_key in self.models:

                    self.models["default"] = self.models[default_chat_role_key]

                    logger.info(f"Assigned role '{default_chat_role_key}' as the default chat model.")

                elif "general" in self.models:

                    self.models["default"] = self.models["general"]

                    logger.warning(f"Default chat role '{default_chat_role_key}' not found, falling back to 'general'.")

                else:

                    logger.error(f"Neither default chat role '{default_chat_role_key}' nor 'general' role found!")

            # --- Final Check for essential components ---
            if not self.embeddings:
                logger.error("❌ Embeddings model failed to initialize for the selected provider.")
                raise ValueError("Embeddings initialization failed.")
            if not self.models.get("default"):
                logger.error("❌ Default chat model failed to initialize for the selected provider.")
                raise ValueError("Default chat model initialization failed.")
            if not self.models.get("vlm"):  # VLM is important for image processing
                logger.warning(
                    "⚠️ VLM model role ('vlm') not configured or failed to initialize. Image understanding capabilities will be limited.")
            if not self.models.get("router"):
                logger.warning(
                    "⚠️ Router model role ('router') not configured. System routing/classification may use 'default' model.")

            logger.success(f"✅ AI Provider '{self.provider_name}' setup complete.")

        except Exception as e_setup:
            logger.error(f"❌ Error setting up AI Provider '{self.provider_name}': {e_setup}")
            logger.exception("AI Provider Setup Traceback:")
            # Ensure a clean state if setup fails catastrophically
            self.models = {}
            self.embeddings = None
            if hasattr(self, '_loaded_llama_instance'): self._loaded_llama_instance = None  # type: ignore
            raise  # Re-raise the exception to prevent app startup with a broken provider

        finally:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.debug(f"⏱️ AI Provider setup method took {duration_ms:.2f} ms.")

    def get_hardware_memory_gb(self) -> float:
        """
        Hardware Agnostic check of AVAILABLE System RAM in GB.
        Crucial for Mac (Unified), AMD, and CPU inference.
        """
        vm = psutil.virtual_memory()
        # .available is better than .free because it includes reclaimable buffers/cache
        available_gb = vm.available / (1024 ** 3)
        return available_gb

    def get_model(self, role: str):
        # Fetch filenames from your updated model map
        model_filename = LLAMA_CPP_MODEL_MAP.get(role)
        if not model_filename:
            return None

        model_path = os.path.join(LLAMA_CPP_GGUF_DIR, model_filename)

        # NEW: Seamless Vision Routing
        if role == "vlm":
            mmproj_filename = LLAMA_CPP_MODEL_MAP.get("vlm_mmproj")
            mmproj_path = os.path.join(LLAMA_CPP_GGUF_DIR, mmproj_filename) if mmproj_filename else None

            # Return the wrapper that pretends to be a LangChain model
            return LlamaCppVisionWrapper(model_path, mmproj_path, self)

        # Standard text models return your existing LlamaCppChatWrapper
        return LlamaCppChatWrapper(
            ai_provider=self,
            model_role=role,
            model_kwargs={"temperature": 0.8}  # or your default kwargs
        )

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
            priority: int = ELP0,
            apply_watermark: bool = False
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
            worker_response = await self._execute_imagination_worker(prompt, image_base64, priority, apply_watermark)

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
            # For now, let's just log and return a truthy value if AdelaideAlbertCortex.py expects something.
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

        #logger.info("Attempting to acquire lock for explicit model unload...")
        pass
        # Lock released here

# --- Optional: Add a shutdown hook specific to CortexEngine for llama.cpp ---
# This ensures the model is unloaded even if the main DB hook runs first/fails.
# Note: atexit runs hooks in reverse order of registration.
def _ai_provider_shutdown():
    # Need a way to access the global CortexEngine instance created in app.py
    # This is tricky. A better pattern might be for AdelaideAlbertCortex.py to explicitly call
    # a shutdown method on the provider instance it holds.
    # For now, we'll assume a global `ai_provider_instance` might exist (set by app.py).
    global ai_provider_instance # Assume AdelaideAlbertCortex.py sets this global
    if 'ai_provider_instance' in globals() and ai_provider_instance:
        logger.info("Running AI Provider shutdown hook...")
        ai_provider_instance.unload_llama_model_if_needed()
    else:
        logger.debug("AI Provider shutdown hook skipped: No global instance found.")

# atexit.register(_ai_provider_shutdown)
# --- End Optional Shutdown Hook ---

# --- Global Instance Placeholder ---
# This global would be set by AdelaideAlbertCortex.py after initialization
ai_provider_instance: Optional[CortexEngine] = None