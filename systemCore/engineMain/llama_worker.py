# llama_worker.py

import sys
import os
import json
import time
import traceback
import argparse
import subprocess
import shlex
import re
import pickle
from typing import Union, List, Dict, Any, Optional  # Added for type hints
from cortex_backbone_provider import INFERCOMPLETION_CTX_BINNING

# --- Try importing llama_cpp ---
"""try:
    import llama_cpp

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    # This print will go to stderr, which cortex_backbone_provider.py reads for worker errors
    print(json.dumps({"error": "llama-cpp-python not found in worker environment."}))
    sys.exit(1)
"""
#llama_cpp python binding is now deprecated. and replaced with direct call to c++ lib

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True, help="Path to GGUF model")
parser.add_argument("--task-type", type=str, required=True, help="Task: chat, embedding, etc.")
parser.add_argument("--n-gpu-layers", type=int, default=-1)
parser.add_argument("--n-ctx", type=int, default=2048)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--chat-format", type=str, default="chatml", help="Chat template format (e.g., chatml, llama-3)")
args = parser.parse_args()

base_cmd = [
    "LMExec",
    "--model", args.model_path,
    "--n-gpu-layers", str(args.n_gpu_layers),
    "--simple-io",
    "--offline",
    "--no-warmup",
    "--no-host",
    "-no-cnv",               # Use single dash as per your terminal test
    "--mmap",
    "--cpu-strict", "1",
    "-ot", ".ffn_.*_exps.=CPU",
    "-fa", "off"
]

# Run the command and capture the output
process = subprocess.run(
                    base_cmd,
                    stdout=subprocess.PIPE,  # Capture the JSON result
                    stderr=subprocess.DEVNULL,  # Kill the ggml_metal chatter
                    text=True
                )
generated_text = process.stdout.strip()


# --- Try importing tiktoken ---
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
    # Attempt to load the encoder once globally to catch issues early
    try:
        cl100k_base_encoder = tiktoken.get_encoding("cl100k_base")
    except Exception as e_enc:
        # Fallback if cl100k_base specifically fails (e.g., network issue during first download)
        try:
            cl100k_base_encoder = tiktoken.encoding_for_model("gpt-4")  # gpt-4 uses cl100k_base
        except Exception as e_enc_fallback:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} WORKER|ERROR] Failed to load tiktoken cl100k_base encoder: {e_enc}, Fallback error: {e_enc_fallback}",
                file=sys.stderr, flush=True)
            TIKTOKEN_AVAILABLE = False  # Disable if encoder can't be loaded
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} WORKER|WARNING] tiktoken library not found. Token counting/dynamic n_ctx limited.",
        file=sys.stderr, flush=True)
    cl100k_base_encoder = None  # Ensure it's defined for type hints


# --- Basic Logging to stderr ---
def log_worker(level, message):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{now} WORKER|{level}] {message}", file=sys.stderr, flush=True)

# In llama_worker.py, near the top with other imports
import hashlib

# --- Configuration Import for State Directory ---
# This ensures the worker knows where to save/load state files.
try:
    # Assuming the config file is in the parent directory of this worker
    # Adjust the path if your structure is different.
    from CortexConfiguration import MODULE_DIR, LLAMA_CPP_N_CTX
    STATE_SAVE_DIR = os.path.join(MODULE_DIR, "staticModelState")
    EMBEDDING_CTX_CONFIG = LLAMA_CPP_N_CTX
    log_worker("INFO", f"Loaded EMBEDDING_CTX_CONFIG from CortexConfiguration: {EMBEDDING_CTX_CONFIG}")
except ImportError:
    # Fallback if CortexConfiguration is not found (e.g., during standalone testing)
    STATE_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "staticModelState")
    log_worker("WARNING", f"Could not import STATE_SAVE_DIR from config. Using default: {STATE_SAVE_DIR}")
    EMBEDDING_CTX_CONFIG = 512 # Fallback to old hardcoded value
    log_worker("WARNING", f"Could not import LLAMA_CPP_N_CTX. Using fallback for embedding n_ctx: {EMBEDDING_CTX_CONFIG}")

os.makedirs(STATE_SAVE_DIR, exist_ok=True)
# --- End Configuration Import ---


#For Max Dynamic N CTX
# MAX_DYNAMIC_N_CTX define it using available memory


# --- Constants for Dynamic Context ---
N_CTX_BINS = INFERCOMPLETION_CTX_BINNING
DEFAULT_N_CTX_FOR_FALLBACK = INFERCOMPLETION_CTX_BINNING[2]  # If token counting fails or not applicable
MIN_DYNAMIC_N_CTX = INFERCOMPLETION_CTX_BINNING[1]  # Minimum context size for dynamic calculation (can be tuned)

MAX_DYNAMIC_N_CTX = INFERCOMPLETION_CTX_BINNING[-1]  # Maximum context size for dynamic calculation
EMBEDDING_N_CTX = 4096  # Fixed context for embedding tasks

# --- Think Tag Cleanup Helper ---
def cleanup_initial_think_tag(text: str) -> str:
    if not text: return ""
    match = re.match(r"<think>(.*?)</think>(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        remaining_text = match.group(2).lstrip()
        # log_worker("TRACE", f"ThinkCleanup: Found initial think tag. Remainder len: {len(remaining_text))}") # Optional: very verbose
        return remaining_text
    return text


# --- Token Counting Helper ---
def count_tokens_cl100k(text_or_messages: Union[str, List[Dict[str, str]]]) -> int:
    # Handle empty or None input first: 0 tokens
    if not text_or_messages:
        log_worker("TRACE", "count_tokens_cl100k: Input is None or empty. Returning 0 tokens.")
        return 0

    if not TIKTOKEN_AVAILABLE or not cl100k_base_encoder:
        log_worker("WARNING", "count_tokens_cl100k: Tiktoken not available or encoder not loaded. Using character-based estimate.")
        # Fallback to character-based estimation if tiktoken is unavailable
        if isinstance(text_or_messages, str):
            return len(text_or_messages) // 4  # Rough estimate, can be > 0
        elif isinstance(text_or_messages, list):
            char_count = 0
            for msg in text_or_messages:
                if isinstance(msg, dict) and "content" in msg:
                    content_item = msg["content"]
                    if isinstance(content_item, str):
                        char_count += len(content_item)
                    elif isinstance(content_item, list): # VLM content
                        for part in content_item:
                             if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                                 char_count += len(part.get("text", ""))
            return char_count // 4 # Rough estimate
        return 0 # Default for unknown type with no tiktoken

    total_tokens = 0
    try:
        if isinstance(text_or_messages, str):
            # Ensure empty string after strip results in 0 tokens if tiktoken would count it as 1
            if not text_or_messages.strip():
                # log_worker("TRACE", "count_tokens_cl100k: Input string is whitespace only. Returning 0 tokens.")
                return 0
            tokens = cl100k_base_encoder.encode(text_or_messages)
            total_tokens = len(tokens)
        elif isinstance(text_or_messages, list):
            if not any(msg.get("content") for msg in text_or_messages if isinstance(msg,dict)): # Check if all content fields are empty
                # log_worker("TRACE", "count_tokens_cl100k: All messages in list have no/empty content. Returning 0 tokens.")
                return 0
            for msg in text_or_messages:
                if isinstance(msg, dict) and "content" in msg:
                    content_item = msg["content"]
                    current_msg_tokens = 0
                    if isinstance(content_item, str):
                        if content_item.strip(): # Only count if non-whitespace
                            tokens = cl100k_base_encoder.encode(content_item)
                            current_msg_tokens = len(tokens)
                    elif isinstance(content_item, list): # VLM content
                        for part_idx, part in enumerate(content_item):
                            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                                if part.get("text", "").strip():
                                    tokens = cl100k_base_encoder.encode(part.get("text", ""))
                                    current_msg_tokens += len(tokens)
                    if current_msg_tokens > 0:
                        total_tokens += current_msg_tokens
                        total_tokens += 4 # Approx overhead per message with content
        # log_worker("TRACE", f"count_tokens_cl100k: Counted {total_tokens} tokens using tiktoken.")
        return total_tokens
    except Exception as e:
        log_worker("ERROR", f"Tiktoken counting error in worker: {e}. Returning -2 to indicate error.")
        return -2 # Indicate a counting error, distinct from 0 or character estimate

# --- Adaptive Middle Truncate Function ---
def adaptive_middle_truncate(text: str, target_max_tokens: int, model_actual_n_ctx: int) -> str:
    """
    Iteratively truncates text from the middle to fit within target_max_tokens.
    Uses tiktoken for counting. model_actual_n_ctx is the Llama instance's context.
    """
    if not TIKTOKEN_AVAILABLE or not cl100k_base_encoder:
        log_worker("WARNING", "adaptive_middle_truncate: Tiktoken not available. Using basic char limit as fallback.")
        char_limit = target_max_tokens * 3  # Very rough estimate
        if len(text) > char_limit:
            # Simple end truncation for fallback
            return text[:char_limit] + "\n[...TEXT TRUNCATED (NO TIKTOKEN)...]\n"
        return text

    if not text: return ""

    try:
        current_tokens = count_tokens_cl100k(text)
    except Exception as e_count:
        log_worker("ERROR", f"adaptive_middle_truncate: Error counting initial tokens: {e_count}. Using char count.")
        current_tokens = len(text) // 4

    if current_tokens <= target_max_tokens:
        log_worker("TRACE",
                   f"adaptive_middle_truncate: Text ({current_tokens} tokens) already within target ({target_max_tokens}). No truncation needed.")
        return text

    log_worker("DEBUG",
               f"adaptive_middle_truncate: Initial tokens {current_tokens} > target {target_max_tokens}. Starting middle crunch.")

    chars_to_remove_at_once = 2
    iteration_count = 0
    max_iterations = 200  # Safety break
    original_length_chars = len(text)
    truncated_text = text
    initial_overshoot_tokens = current_tokens - target_max_tokens

    while current_tokens > target_max_tokens and iteration_count < max_iterations and len(
            truncated_text) > 10:  # Ensure some text remains
        iteration_count += 1
        if len(truncated_text) <= chars_to_remove_at_once:
            log_worker("WARNING", "adaptive_middle_truncate: Text too short for further middle removal. Stopping.")
            break

        middle_index = len(truncated_text) // 2
        remove_half = chars_to_remove_at_once // 2
        start_remove_index = max(0, middle_index - remove_half)
        end_remove_index = min(len(truncated_text), middle_index + (chars_to_remove_at_once - remove_half))

        if start_remove_index >= end_remove_index:
            log_worker("WARNING",
                       f"adaptive_middle_truncate: Invalid removal indices (start {start_remove_index}, end {end_remove_index}). Stopping.")
            break

        truncated_text = truncated_text[:start_remove_index] + truncated_text[end_remove_index:]

        try:
            current_tokens = count_tokens_cl100k(truncated_text)
        except Exception:
            current_tokens = len(truncated_text) // 4  # Fallback

        # Adjust removal strategy
        if current_tokens > target_max_tokens:
            # Estimate how many more tokens to remove
            tokens_still_over = current_tokens - target_max_tokens
            if tokens_still_over > initial_overshoot_tokens * 0.5:  # Still far
                chars_to_remove_at_once = min(chars_to_remove_at_once * 2, len(truncated_text) // 8,
                                              2000)  # Aggressive, capped
            elif tokens_still_over > initial_overshoot_tokens * 0.1:  # Getting closer
                chars_to_remove_at_once = min(max(10, chars_to_remove_at_once), len(truncated_text) // 10,
                                              500)  # Moderate
            else:  # Very close
                chars_to_remove_at_once = max(2, chars_to_remove_at_once // 2)  # Fine-tune

        chars_to_remove_at_once = max(2, chars_to_remove_at_once)  # Ensure at least 2 chars are attempted

        # log_worker("TRACE", f"adaptive_middle_truncate: Iter {iteration_count}, Tokens: {current_tokens}, Next remove batch: {chars_to_remove_at_once}")

    if iteration_count >= max_iterations:
        log_worker("WARNING", "adaptive_middle_truncate: Reached max iterations. Truncation might be inexact.")

    if len(truncated_text) < original_length_chars:
        marker = "\n[...CONTENT CRUNCHED FROM MIDDLE TO FIT CONTEXT...]\n"
        # Decide where to put marker: prepend, append, or try to find the cut point (hard)
        # Prepending is often less disruptive than trying to insert in the exact middle.
        final_text_with_marker = marker + truncated_text

        # Final check: if marker pushes it over, return without marker
        # (This can happen if target_max_tokens was very close to original token count)
        if count_tokens_cl100k(final_text_with_marker) > target_max_tokens + (
                count_tokens_cl100k(marker) // 2):  # Allow some leeway for marker
            log_worker("WARNING",
                       "adaptive_middle_truncate: Marker addition pushed tokens over limit. Returning text without marker.")
            final_text_with_marker = truncated_text

        log_worker("INFO",
                   f"adaptive_middle_truncate: Truncated text. Original tokens: {count_tokens_cl100k(text)}, Final tokens: {current_tokens} (target: {target_max_tokens}).")
        return final_text_with_marker
    else:
        log_worker("TRACE", "adaptive_middle_truncate: No truncation was performed.")
        return truncated_text


def format_messages_as_string(messages: list) -> str:
    """
    Converts OpenAI-style message list to a raw string for LMExec.
    Standard ChatML format: <|im_start|>role\ncontent<|im_end|>\n
    """
    formatted_prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Format based on the specific role
        formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    # We add the final assistant header to trigger the AI to start typing
    formatted_prompt += "<|im_start|>assistant\n"

    return formatted_prompt

def get_state_path_for_model(model_path: str, n_ctx: int) -> str:
    """Generates a unique state cache (native KV cache) file path based on model path and context size."""
    # e.g. /path/to/models/llava-v1.5-7b.Q4_K_M.gguf
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    # e.g. llava-v1.5-7b.Q4_K_M.gguf
    state_filename = f"{model_name}.nctx_{n_ctx}.cache"
    # e.g. /path/to/models/llava-v1.5-7b.Q4_K_M.gguf.nctx_4096.cache
    return os.path.join(model_dir, state_filename)

def extract_final_answer(raw_text):
    # 1. Handle Thinking Tags first
    if "[End thinking]" in raw_text:
        raw_text = raw_text.split("[End thinking]")[-1]
    elif "</think>" in raw_text:
        raw_text = raw_text.split("</think>")[-1]
    
    # 2. Specifically remove the "[end of text]" token
    # We use .replace() in case it appears unexpectedly mid-text
    # or .removesuffix() if it is always at the very end.
    cleaned_text = raw_text.replace("[end of text]", "").strip()
    
    return cleaned_text


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Llama.cpp Worker Process with Automatic KV Caching and Dynamic Context")
    parser.add_argument("--model-path", required=True, help="Path to GGUF model file")
    parser.add_argument("--task-type", required=True, choices=["chat", "embedding", "raw_text_completion", "vision"],
                        help="Task type")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="Number of GPU layers")
    parser.add_argument("--n-ctx", type=int, default=None,
                        help="Context size (fixed for embeddings, OPTIONAL override for chat/raw_text)")
    parser.add_argument("--verbose", action="store_true", help="Enable llama.cpp verbose logging")
    parser.add_argument("--chat-format", type=str, default=None,
                        help="Chat format for 'chat' task (e.g., chatml, llama-2)")
    args = parser.parse_args()

    # --- Initializations ---
    request_data_str = ""
    request_data_dict = None
    input_tokens_for_dynamic_ctx = 0
    calculated_n_ctx_for_model_load = DEFAULT_N_CTX_FOR_FALLBACK
    #llm: Optional[llama_cpp.Llama] = None
    result_payload: Dict[str, Any] = {"error": "Worker did not process a valid task."}

    #original_kwargs_from_provider = request_data_dict.get("kwargs", {})
    #original_kwargs_from_provider = request_data_dict.get("kwargs", {}).copy()
    
    try:
        log_worker("INFO", "Reading request data from stdin...")
        request_data_str = sys.stdin.read()
        if not request_data_str.strip():
            raise ValueError("Received empty input string from stdin.")
        
        # This is the critical change: assign here and check immediately.
        request_data_dict = json.loads(request_data_str)
        if not isinstance(request_data_dict, dict):
            raise TypeError("Parsed stdin is not a dictionary.")
            
        log_worker("DEBUG", f"Request data JSON parsed. Task: {args.task_type}")

        # Now that we know request_data_dict is valid, we can define this.
        original_kwargs_from_provider = request_data_dict.get("kwargs", {})

        # Token counting for dynamic context (if applicable)
        input_tokens_for_dynamic_ctx = 0
        if args.task_type == "chat":
            messages_for_count = request_data_dict.get("messages")
            if messages_for_count:
                input_tokens_for_dynamic_ctx = count_tokens_cl100k(messages_for_count)
        elif args.task_type == "raw_text_completion":
            prompt_for_count = request_data_dict.get("prompt")
            if prompt_for_count:
                input_tokens_for_dynamic_ctx = count_tokens_cl100k(prompt_for_count)
        
        log_worker("INFO", f"Initial estimated input tokens (for n_ctx calc): {input_tokens_for_dynamic_ctx}")

    except (json.JSONDecodeError, ValueError, TypeError) as e_stdin:
        log_worker("CRITICAL", f"Failed to read/parse stdin: {e_stdin}\n{traceback.format_exc()}")
        # This ensures the script stops and reports the error correctly.
        print(json.dumps({"error": f"Worker critical error: Failed to read/parse request: {e_stdin}"}))
        sys.exit(1)


    # --- Determine n_ctx (CTX binning, NOT main execution of the LMExec or LMMultiModal) for Llama model loading ---
    if args.task_type == "chat":
        # For chat, the provider sends 'messages'
        messages_from_provider = request_data_dict.get("messages", [])
        prompt_for_llm = format_messages_as_string(messages_from_provider)  # Ensure you have this helper
    else:
        # For raw_text_completion, it sends 'prompt'
        prompt_for_llm = request_data_dict.get("prompt", "")

    if args.task_type == "embedding":
        calculated_n_ctx_for_model_load = EMBEDDING_CTX_CONFIG
        log_worker("INFO", f"Forcing n_ctx to EMBEDDING_CTX_CONFIG ({EMBEDDING_CTX_CONFIG}) for embedding task, ignoring CLI args.")
    elif args.n_ctx is not None:
        calculated_n_ctx_for_model_load = args.n_ctx
        log_worker("INFO", f"Using n_ctx={calculated_n_ctx_for_model_load} from cortex_backbone_provider override.")

    # ------------------- MODIFIED BLOCK FOR CONTEXT BINNING -------------------
    elif input_tokens_for_dynamic_ctx > 0:
        target_n_ctx = input_tokens_for_dynamic_ctx * 2

        # Find the smallest bin that can fit the target context size
        locked_n_ctx = N_CTX_BINS[-1]  # Default to the largest available bin
        for bin_size in N_CTX_BINS:
            if target_n_ctx <= bin_size:
                locked_n_ctx = bin_size
                break  # Found the smallest sufficient bin, so we stop

        calculated_n_ctx_for_model_load = locked_n_ctx
        log_worker("INFO", f"Input tokens ({input_tokens_for_dynamic_ctx}) need ~{target_n_ctx} context. Locking to bin: {calculated_n_ctx_for_model_load}")
    # ------------------------------------------------------------------------

    else:
        calculated_n_ctx_for_model_load = DEFAULT_N_CTX_FOR_FALLBACK
        log_worker("INFO", f"Falling back to default n_ctx={calculated_n_ctx_for_model_load} for model load.")

    log_worker("INFO", f"--- Worker Config (effective for model load) ---")
    log_worker("INFO", f"  Model: '{os.path.basename(args.model_path)}', Task: '{args.task_type}'")
    log_worker("INFO", f"  Effective n_ctx for Llama load: {calculated_n_ctx_for_model_load}")

    # --- Main Execution Block ---
    try:
        # 1. Determine the dedicated state path for this model AND n_ctx
        state_file_path = get_state_path_for_model(args.model_path, calculated_n_ctx_for_model_load)
        state_file_exists = os.path.exists(state_file_path)

        # 2. Prepare Execution (Loading is now handled by the binary)
        load_start_time = time.monotonic()

        # Resolve binary and format references
        binary_to_use = "LMExec" if args.task_type != "embedding" else "LMText2Vector"
        effective_chat_template = args.chat_format if args.task_type == "chat" else None

        # Check for pre-existing native cache (No longer using pickle)
        should_attempt_cache_load = (args.task_type != "embedding" and state_file_exists)

        if should_attempt_cache_load:
            log_worker("INFO", f"KV cache found. LMExec will attempt to load: {state_file_path}")
            # Note: If LMExec finds the cache incompatible with n_ctx,
            # it will automatically recompute. We don't need manual flush logic here.
        else:
            log_worker("INFO", f"Starting {binary_to_use} fresh (No cache or embedding task).")

        # We no longer create an 'llm' object.
        # We just keep the configuration ready for the subprocess call below.
        llm = None
        log_worker("INFO", f"Model loaded in {(time.monotonic() - load_start_time) * 1000:.2f} ms.")

        # 3. Process the Task
        task_processing_start_time = time.monotonic()
        completion_result_dict: Optional[Dict[str, Any]] = None
        
        GENERATION_OUTPUT_BUFFER_TOKENS = max(256, calculated_n_ctx_for_model_load // 4)
        if args.task_type == "chat":
            messages_for_llm = request_data_dict.get("messages")
            if not messages_for_llm or not isinstance(messages_for_llm, list): raise ValueError("Missing or invalid 'messages' for 'chat' task.")
            current_input_token_count_chat = count_tokens_cl100k(messages_for_llm)
            if current_input_token_count_chat == -1: current_input_token_count_chat = len(json.dumps(messages_for_llm)) // 3
            max_gen_tokens_chat = min(original_kwargs_from_provider.get("max_tokens", calculated_n_ctx_for_model_load // 2), calculated_n_ctx_for_model_load - current_input_token_count_chat - 64)
            if max_gen_tokens_chat <= 0: max_gen_tokens_chat = GENERATION_OUTPUT_BUFFER_TOKENS // 2
            original_kwargs_from_provider["max_tokens"] = max_gen_tokens_chat
            chat_cmd = base_cmd + [
                "-c", str(calculated_n_ctx_for_model_load),
                "--no-display-prompt",
                "--chat-template", args.chat_format or "chatml",
                "--prompt", format_messages_as_string(messages_for_llm),
                "--n-predict", str(max_gen_tokens_chat)
            ]

            # Execute and capture stdout
            result = subprocess.run(chat_cmd, capture_output=True, text=True)
            raw_generated_text = result.stdout
            raw_generated_text = extract_final_answer(raw_generated_text)
            completion_result_dict = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": os.path.basename(args.model_path),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": cleanup_initial_think_tag(raw_generated_text)  # Reuse your existing helper
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": input_tokens_for_dynamic_ctx,
                    "completion_tokens": count_tokens_cl100k(raw_generated_text),  # Reuse your token counter
                    "total_tokens": input_tokens_for_dynamic_ctx + count_tokens_cl100k(raw_generated_text)
                }
            }
            result_payload = {"result": completion_result_dict}
        elif args.task_type == "raw_text_completion":
            prompt_from_provider = request_data_dict.get("prompt")
            if not prompt_from_provider or not isinstance(prompt_from_provider, str): raise ValueError("Missing or invalid 'prompt' for 'raw_text_completion' task.")
            target_input_prompt_max_tokens = calculated_n_ctx_for_model_load - GENERATION_OUTPUT_BUFFER_TOKENS
            if target_input_prompt_max_tokens <= 0: target_input_prompt_max_tokens = calculated_n_ctx_for_model_load // 2
            prompt_for_llm = adaptive_middle_truncate(prompt_from_provider, target_input_prompt_max_tokens, calculated_n_ctx_for_model_load)
            current_input_token_count_raw = count_tokens_cl100k(prompt_for_llm)
            if current_input_token_count_raw == -1: current_input_token_count_raw = len(prompt_for_llm) // 3
            original_kwargs_from_provider.setdefault("stop", ["<|im_end|>"])
            max_gen_tokens_raw = min(original_kwargs_from_provider.get("max_tokens", calculated_n_ctx_for_model_load // 2), calculated_n_ctx_for_model_load - current_input_token_count_raw - 64)
            if max_gen_tokens_raw <= 0:
                max_gen_tokens_raw = GENERATION_OUTPUT_BUFFER_TOKENS // 2

                # 1. Build the CLI command with core parameters
            cli_cmd = [
                "LMExec",
                "--model", args.model_path,
                "--n-gpu-layers", str(args.n_gpu_layers),
                "-c", str(calculated_n_ctx_for_model_load),
                "--prompt", prompt_for_llm,
                "--n-predict", str(max_gen_tokens_raw),
                "--prompt-cache", state_file_path,  # Native binary KV cache
                "--prompt-cache-all",  # Cache the prompt for future turns
                "--simple-io",  # Use basic IO for subprocesses
                "--offline",
                "--no-host",
                "--no-warmup",
                "-no-cnv",
                "--log-disable"  # Prevent logs from polluting stdout
            ]

            # 2. Map sampling parameters from original_kwargs_from_provider to CLI flags
            if "temperature" in original_kwargs_from_provider:
                cli_cmd.extend(["--temp", str(original_kwargs_from_provider["temperature"])])
            if "top_p" in original_kwargs_from_provider:
                cli_cmd.extend(["--top-p", str(original_kwargs_from_provider["top_p"])])
            if "top_k" in original_kwargs_from_provider:
                cli_cmd.extend(["--top-k", str(original_kwargs_from_provider["top_k"])])

            # Map stop sequences to reverse prompts
            stop_seqs = original_kwargs_from_provider.get("stop", ["<|im_end|>"])
            if isinstance(stop_seqs, list):
                for s in stop_seqs:
                    cli_cmd.extend(["--reverse-prompt", s])

            # 3. Execute and capture the result
            #process = subprocess.run(cli_cmd, capture_output=True, text=True)
            process = subprocess.run(
                cli_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            if process.returncode != 0:
                raise RuntimeError(f"LMExec execution failed: {process.stderr}")

            # 4. Standardize output to keep the rest of llama_worker.py logic working
            completion_result_dict = {
                "choices": [
                    {
                        "text": process.stdout.strip(),
                        "finish_reason": "stop"
                    }
                ]
            }
        
        elif args.task_type == "embedding":
            texts_to_embed = request_data_dict.get("texts", [])
            embedding_results = []

            for text in texts_to_embed:
                if not text.strip(): continue

                # Construct the subprocess command
                embed_cmd = [
                    "LMText2Vector",
                    "--model", args.model_path,
                    "--n-gpu-layers", str(args.n_gpu_layers),
                    "--pooling", "mean",
                    "--embd-output-format", "json", # Returns OpenAI-compatible JSON
                    "--prompt", text
                ]

                # Execute
                process = subprocess.run(
                    embed_cmd,
                    stdout=subprocess.PIPE,  # Capture the JSON result
                    stderr=subprocess.DEVNULL,  # Kill the ggml_metal chatter
                    text=True
                )
                output_data = "Placeholder... Before process.stdout fails?"
                if process.returncode == 0:
                    try:
                        # 3. Parse the captured stdout
                        # The binary returns: {"embedding": [0.123, -0.456, ...]}
                        output_data = json.loads(process.stdout) # recieve the stdio (THIS IS RAW DATA So it is debug target!)

                        vector = []
                        
                        # CASE A: OpenAI Style (What your binary is returning)
                        # {"object": "list", "data": [{"embedding": [...]}]}
                        if isinstance(output_data, dict) and "data" in output_data and isinstance(output_data["data"], list):
                            if len(output_data["data"]) > 0:
                                vector = output_data["data"][0].get("embedding", [])
                        
                        # CASE B: Flat Dict (Some llama.cpp versions)
                        # {"embedding": [...]}
                        elif isinstance(output_data, dict) and "embedding" in output_data:
                            vector = output_data.get("embedding", [])
                            
                        # CASE C: List of Dicts (Legacy/Other)
                        # [{"embedding": [...]}]
                        elif isinstance(output_data, list) and len(output_data) > 0:
                            vector = output_data[0].get("embedding", [])
                            
                        if vector and isinstance(vector, list):
                            log_worker("INFO", "Parsed Successfully")
                            embedding_results.append(vector)
                        else:
                            log_worker("ERROR", f"Parsed JSON did not contain 'embedding' field. Debug Dump: {output_data} with Key Dump {output_data.keys() if isinstance(output_data, dict) else 'List'}")
                            # Append empty list to maintain index alignment if needed, or fail
                            # embedding_results.append([])
                            
                    except json.JSONDecodeError as e:
                        # Log the error if the binary output wasn't valid JSON
                        log_worker("ERROR", f"Failed to parse LMText2Vector JSON: {e} Debug Dump: {output_data}")
                        completion_result_dict = {"error": "Invalid JSON from embedding binary"}
                else:
                    log_worker("ERROR", f"LMText2Vector failed with code {process.returncode}")
                    completion_result_dict = {"error": "Embedding binary execution failed"}

            result_payload = {"result": embedding_results}
        
        elif args.task_type == "vision":
            # 1. Vision models REQUIRE the projector file
            # We expect the provider to send this in the JSON request data
            mmproj_path = request_data_dict.get("mmproj_path")
            image_path = request_data_dict.get("image_path")

           # prompt_for_llm = request_data_dict.get("messages")
            prompt_for_llm = request_data_dict.get("prompt", "Describe this image. with all the atributes environment and activity")
            
            # 2. Construct the LMMultiModal command
            vision_cmd = [
                "LMMultiModal", # Renamed from llama-mtmd-cli
                "--model", args.model_path,
                "--mmproj", mmproj_path,
                "--image", image_path,
                "--mmap",
                "--cpu-strict", "1",
                "-fa", "off",
                "-ot", ".ffn_.*_exps.=CPU", #Unsloth recommendation on optimizaiton of memory
                "--no-warmup",
                "--offline",
                "--no-host",
                "--n-gpu-layers", "-1",
                "-c", "4096", #maintain context window limitation for limiting memory consumption issue (Was sabotaged by Gemini)
                "--temp", str(original_kwargs_from_provider.get("temperature", 0.8)),
                "-n", "2048", # Limit generation for descriptions
                "-p", prompt_for_llm
            ]

            # 3. Execute with Stdout bridge and Stderr suppression
            process = subprocess.run(
                vision_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, # Silence hardware logs
                text=True
            )

            # 4. Extract only the generated text
            raw_response = process.stdout.strip()
            # Clean up any leftover prompt text or thinking tags
            cleaned_response = extract_final_answer(raw_response)
            
            completion_result_dict = {
                "choices": [{"message": {"content": cleaned_response}, "finish_reason": "stop"}]
            }
            
        else:
            raise ValueError(f"Unknown task type in processing block: {args.task_type}")
        if args.task_type != "embedding":
            # --- 1. Ensure completion_result_dict is a dictionary ---
            if isinstance(completion_result_dict, str):
                completion_result_dict = {"choices": [{"text": completion_result_dict}]}
            if not completion_result_dict:
                raise RuntimeError("LLM call did not return a result dictionary.")

            # --- 2. Extract Text and Initialize max_gen_tokens_raw ---
            raw_generated_text = ""
            # <<< FIX: Initialize the variable here to guarantee it exists >>>
            max_gen_tokens_raw = 0

            if args.task_type == "chat" or args.task_type == "vision":
                choices = completion_result_dict.get('choices', [])
                if choices and isinstance(choices[0].get('message'), dict):
                    raw_generated_text = choices[0]['message'].get('content', "")
            elif args.task_type == "raw_text_completion":
                choices = completion_result_dict.get('choices', [])
                if choices:
                    raw_generated_text = choices[0].get('text', "")
                
                # We still need to calculate this for the rerequest logic if this path is taken
                current_input_token_count_raw = count_tokens_cl100k(prompt_for_llm)
                if current_input_token_count_raw == -1: current_input_token_count_raw = len(prompt_for_llm) // 3
                max_gen_tokens_raw = min(original_kwargs_from_provider.get("max_tokens", calculated_n_ctx_for_model_load // 2), calculated_n_ctx_for_model_load - current_input_token_count_raw - 64)
                if max_gen_tokens_raw <= 0:
                    max_gen_tokens_raw = GENERATION_OUTPUT_BUFFER_TOKENS // 2

            cleaned_text = cleanup_initial_think_tag(str(raw_generated_text or ""))
            token_count_after_cleanup = count_tokens_cl100k(cleaned_text)

            # --- 3. Empty Response Handling (Rerequest Logic) ---
            if TIKTOKEN_AVAILABLE and token_count_after_cleanup == 0:
                log_worker("WARNING", "Empty response detected. Attempting re-request with system nudge.")

                complaint_prefix = "[System Note: Previous response was empty. Please provide a substantive answer.]\n"
                rereq_base = [
                    "LMExec", "--model", args.model_path,
                    "--n-gpu-layers", str(args.n_gpu_layers),
                    "-c", str(calculated_n_ctx_for_model_load),
                    "--simple-io", "-no-cnv", "--no-display-prompt", "--no-warmup",
                    "--no-show-timings", "--log-disable"
                ]

                if args.task_type == "chat" or args.task_type == "vision":
                    modified_prompt = complaint_prefix + format_messages_as_string(request_data_dict.get("messages", []))
                    rereq_cmd = rereq_base + [
                        "--n-predict", str(calculated_n_ctx_for_model_load // 2),
                        "--prompt", modified_prompt
                    ]
                else:  # raw_text_completion
                    modified_prompt = complaint_prefix + request_data_dict.get("prompt", "")
                    if max_gen_tokens_raw <= 0:
                        log_worker("WARNING", "max_gen_tokens_raw was 0 for re-request, defaulting to 256.")
                        max_gen_tokens_raw = 256
                    rereq_cmd = rereq_base + [
                        "--n-predict", str(max_gen_tokens_raw),
                        "--prompt", modified_prompt
                    ]

                process = subprocess.run(rereq_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                raw_generated_text_attempt2 = extract_final_answer(process.stdout.strip())
                cleaned_text_attempt2 = cleanup_initial_think_tag(raw_generated_text_attempt2)

                if args.task_type == "chat" or args.task_type == "vision":
                    completion_result_dict['choices'][0]['message']['content'] = cleaned_text_attempt2
                else:
                    completion_result_dict['choices'][0]['text'] = cleaned_text_attempt2

            # --- 4. Final Payload Packaging ---
            result_payload = {"result": completion_result_dict}
        task_processing_duration_ms = (time.monotonic() - task_processing_start_time) * 1000
        log_worker("INFO", f"Task '{args.task_type}' core processing completed in {task_processing_duration_ms:.2f} ms.")

        # 4. Save the New State Automatically After Successful Task
        if args.task_type != "embedding":
            # If the subprocess finished with returncode 0, the cache file
            # at 'state_file_path' has already been updated by the binary.
            log_worker("SUCCESS", f"KV cache automatically updated via llama C++ binary at: {state_file_path}")
        else:
            log_worker("INFO", "Embedding task finished. Native state caching skipped as per logic.")

    except Exception as e:
        log_worker("ERROR", f"Error during worker execution: {e}\n{traceback.format_exc()}")
        result_payload = {"error": f"Worker execution failed: {str(e)}"}

    # --- Send Result to Parent and Exit ---
    finally:
        try:
            output_json_to_parent = json.dumps(result_payload)
            print(output_json_to_parent, flush=True)
            log_worker("INFO", "Result/Error JSON sent to stdout successfully.")
        except Exception as e_serialize:
            log_worker("CRITICAL", f"Failed to serialize/write final result_payload to stdout: {e_serialize}")
            fallback_error_msg = {"error": f"Worker critical error: Failed to serialize/write result to stdout: {e_serialize}"}
            print(json.dumps(fallback_error_msg), flush=True)
            sys.exit(1)

    log_worker("INFO", f"Llama Worker (PID: {os.getpid()}) process finished gracefully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
