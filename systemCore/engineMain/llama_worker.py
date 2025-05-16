# llama_worker.py

import sys
import os
import json
import time
import traceback
import argparse
import re
from typing import Union, List, Dict, Any, Optional  # Added for type hints

# --- Try importing llama_cpp ---
try:
    import llama_cpp

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    # This print will go to stderr, which ai_provider.py reads for worker errors
    print(json.dumps({"error": "llama-cpp-python not found in worker environment."}), flush=True)
    sys.exit(1)

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

# --- Constants for Dynamic Context ---
MIN_DYNAMIC_N_CTX = 2048  # Minimum context size for dynamic calculation (can be tuned)
MAX_DYNAMIC_N_CTX = 32768  # Maximum context size for dynamic calculation
DEFAULT_N_CTX_FOR_FALLBACK = 4096  # If token counting fails or not applicable
EMBEDDING_N_CTX = 512  # Fixed context for embedding tasks


# --- Basic Logging to stderr ---
def log_worker(level, message):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{now} WORKER|{level}] {message}", file=sys.stderr, flush=True)


# --- Think Tag Cleanup Helper ---
def cleanup_initial_think_tag(text: str) -> str:
    if not text: return ""
    match = re.match(r"<think>(.*?)</think>(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        remaining_text = match.group(2).lstrip()
        # log_worker("TRACE", f"ThinkCleanup: Found initial think tag. Remainder len: {len(remaining_text)}") # Optional: very verbose
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
                                 char_count += len(part.get("text",""))
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
                                if part.get("text","").strip():
                                    tokens = cl100k_base_encoder.encode(part.get("text",""))
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


def main():
    parser = argparse.ArgumentParser(description="Llama.cpp Worker Process")
    parser.add_argument("--model-path", required=True, help="Path to GGUF model file")
    parser.add_argument("--task-type", required=True, choices=["chat", "embedding", "raw_text_completion"],
                        help="Task type")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="Number of GPU layers")
    parser.add_argument("--n-ctx", type=int, default=None,
                        help="Context size (fixed for embeddings, OPTIONAL override for chat/raw_text)")
    parser.add_argument("--verbose", action="store_true", help="Enable llama.cpp verbose logging")
    parser.add_argument("--chat-format", type=str, default=None,
                        help="Chat format for 'chat' task (e.g., chatml, llama-2)")
    args = parser.parse_args()

    request_data_str = ""
    request_data_dict = None
    input_tokens_for_dynamic_ctx = 0  # Estimated from input before model load
    calculated_n_ctx_for_model_load = DEFAULT_N_CTX_FOR_FALLBACK  # Actual n_ctx Llama model will be loaded with

    try:
        log_worker("INFO", "Reading request data from stdin for dynamic n_ctx calculation...")
        request_data_str = sys.stdin.read()
        if not request_data_str.strip(): raise ValueError("Received empty input string from stdin.")
        request_data_dict = json.loads(request_data_str)
        log_worker("DEBUG", f"Request data JSON parsed for n_ctx calc. Task: {args.task_type}")

        if args.task_type == "chat":
            messages_for_count = request_data_dict.get("messages")
            if messages_for_count: input_tokens_for_dynamic_ctx = count_tokens_cl100k(messages_for_count)
        elif args.task_type == "raw_text_completion":
            prompt_for_count = request_data_dict.get("prompt")
            if prompt_for_count: input_tokens_for_dynamic_ctx = count_tokens_cl100k(prompt_for_count)

        log_worker("INFO", f"Initial estimated input tokens (for n_ctx calc): {input_tokens_for_dynamic_ctx}")

    except Exception as e_stdin:
        log_worker("CRITICAL", f"Failed to read/parse stdin: {e_stdin}\n{traceback.format_exc()}")
        print(json.dumps({"error": f"Worker critical error: Failed to read/parse request: {e_stdin}"}), flush=True)
        sys.exit(1)

    # --- Determine n_ctx for Llama model loading ---
    if args.task_type == "embedding":
        calculated_n_ctx_for_model_load = args.n_ctx if args.n_ctx is not None else EMBEDDING_N_CTX
        if args.n_ctx is None: log_worker("INFO", f"Using fixed n_ctx={EMBEDDING_N_CTX} for embeddings task.")
    elif args.n_ctx is not None:  # Explicit override from ai_provider.py
        calculated_n_ctx_for_model_load = args.n_ctx
        log_worker("INFO", f"Using n_ctx={calculated_n_ctx_for_model_load} from ai_provider override.")
    elif input_tokens_for_dynamic_ctx > 0:
        dynamic_ctx_attempt = input_tokens_for_dynamic_ctx * 2  # Target double the input
        # Ensure it's at least MIN_DYNAMIC_N_CTX and at most MAX_DYNAMIC_N_CTX
        calculated_n_ctx_for_model_load = max(MIN_DYNAMIC_N_CTX, min(dynamic_ctx_attempt, MAX_DYNAMIC_N_CTX))
        # Also ensure it's not smaller than the input itself plus some buffer, if dynamic_ctx_attempt was too small
        # (e.g. if input_tokens_for_dynamic_ctx was already > MAX_DYNAMIC_N_CTX / 2)
        # This means if input is 20k tokens, and MAX_DYNAMIC_N_CTX is 32k, dynamic_ctx_attempt is 40k, so it caps at 32k.
        # If input is 2k tokens, MIN_DYNAMIC_N_CTX is 4k, dynamic_ctx_attempt is 4k, so it uses 4k.
        log_worker("INFO",
                   f"Dynamically calculated n_ctx for model load: {calculated_n_ctx_for_model_load} (InputTokens: {input_tokens_for_dynamic_ctx}, Attempt: {dynamic_ctx_attempt})")
    else:  # Fallback (e.g. tiktoken error, no input)
        calculated_n_ctx_for_model_load = DEFAULT_N_CTX_FOR_FALLBACK
        log_worker("INFO", f"Falling back to default n_ctx={calculated_n_ctx_for_model_load} for model load.")

    log_worker("INFO", f"--- Worker Config (effective for model load) ---")
    log_worker("INFO", f"  Model: '{os.path.basename(args.model_path)}', Task: '{args.task_type}'")
    log_worker("INFO", f"  Effective n_ctx for Llama load: {calculated_n_ctx_for_model_load}")
    # ... (other logs as before)

    llm: Optional[llama_cpp.Llama] = None
    try:
        load_start_time = time.monotonic()
        effective_chat_format = args.chat_format if args.task_type == "chat" else None
        if args.task_type == "raw_text_completion": effective_chat_format = None

        llm = llama_cpp.Llama(
            model_path=args.model_path, n_gpu_layers=args.n_gpu_layers,
            n_ctx=calculated_n_ctx_for_model_load,  # Use the determined n_ctx
            embedding=(args.task_type == "embedding"),
            verbose=args.verbose, chat_format=effective_chat_format,
        )
        log_worker("INFO",
                   f"Model loaded in {(time.monotonic() - load_start_time) * 1000:.2f} ms using n_ctx={calculated_n_ctx_for_model_load}.")
    except Exception as e_load:
        log_worker("CRITICAL", f"Model loading failed: {e_load}\n{traceback.format_exc()}")
        print(json.dumps({
                             "error": f"Worker critical error: Failed to load model with n_ctx={calculated_n_ctx_for_model_load}: {e_load}"}),
              flush=True)
        sys.exit(1)

    # --- Request Processing (using request_data_dict from earlier) ---
    result_payload: Dict[str, Any] = {"error": "Worker did not process a valid task."}
    try:
        task_processing_start_time = time.monotonic()
        completion_result_dict: Optional[Dict[str, Any]] = None
        original_kwargs_from_provider = request_data_dict.get("kwargs", {}).copy()

        # Define a buffer for LLM's own generation output
        # This is how many tokens we reserve for the *response*, so input prompt must be smaller.
        # It can be a fixed value, or a percentage of the effective context window.
        GENERATION_OUTPUT_BUFFER_TOKENS = max(256,
                                              calculated_n_ctx_for_model_load // 4)  # e.g., 25% of context or 256, whichever is larger
        log_worker("DEBUG", f"Generation output buffer set to: {GENERATION_OUTPUT_BUFFER_TOKENS} tokens.")

        if args.task_type == "chat":
            messages_for_llm = request_data_dict.get("messages")
            if not messages_for_llm or not isinstance(messages_for_llm, list):
                raise ValueError("Missing or invalid 'messages' for 'chat' task.")

            # For 'chat' (list of messages), the primary context management is the dynamically loaded n_ctx.
            # Adaptive middle truncation is complex for structured messages.
            # We just need to ensure `max_tokens` for generation is sane.
            current_input_token_count_chat = count_tokens_cl100k(messages_for_llm)
            if current_input_token_count_chat == -1:  # Tiktoken error
                current_input_token_count_chat = len(json.dumps(messages_for_llm)) // 3  # Rough fallback
                log_worker("WARNING",
                           f"Tiktoken error for chat messages, using char estimate: {current_input_token_count_chat} tokens.")

            # `max_tokens` for generation output
            max_gen_tokens_chat = min(
                original_kwargs_from_provider.get("max_tokens", calculated_n_ctx_for_model_load // 2),
                # Provider's hint or half context
                calculated_n_ctx_for_model_load - current_input_token_count_chat - 64  # Safety buffer for prompt, roles
            )
            if max_gen_tokens_chat <= 0: max_gen_tokens_chat = GENERATION_OUTPUT_BUFFER_TOKENS // 2  # Min sensible value if calc is bad
            original_kwargs_from_provider["max_tokens"] = max_gen_tokens_chat

            log_worker("INFO",
                       f"Attempt 1 (Chat): Msgs={len(messages_for_llm)}, InputTokensEst={current_input_token_count_chat}, Effective GenMaxTokens={max_gen_tokens_chat}, Kwargs={original_kwargs_from_provider}")
            if current_input_token_count_chat + max_gen_tokens_chat > calculated_n_ctx_for_model_load:
                log_worker("ERROR",
                           f"Potential token overflow for CHAT: Input ({current_input_token_count_chat}) + MaxGen ({max_gen_tokens_chat}) > Model n_ctx ({calculated_n_ctx_for_model_load}). Model might truncate input.")

            completion_result_dict = llm.create_chat_completion(
                messages=messages_for_llm, stream=False, **original_kwargs_from_provider
            )

        elif args.task_type == "raw_text_completion":
            prompt_from_provider = request_data_dict.get("prompt")  # This is the full assembled ChatML string
            if not prompt_from_provider or not isinstance(prompt_from_provider, str):
                raise ValueError("Missing or invalid 'prompt' for 'raw_text_completion' task.")

            # Max tokens the INPUT prompt can occupy *before* truncation
            target_input_prompt_max_tokens = calculated_n_ctx_for_model_load - GENERATION_OUTPUT_BUFFER_TOKENS
            if target_input_prompt_max_tokens <= 0:
                log_worker("ERROR",
                           f"target_input_prompt_max_tokens ({target_input_prompt_max_tokens}) is too low. Model n_ctx: {calculated_n_ctx_for_model_load}, GenBuffer: {GENERATION_OUTPUT_BUFFER_TOKENS}")
                target_input_prompt_max_tokens = calculated_n_ctx_for_model_load // 2  # Fallback to half context for prompt
                log_worker("WARNING", f"Adjusted target_input_prompt_max_tokens to {target_input_prompt_max_tokens}")

            prompt_for_llm = adaptive_middle_truncate(prompt_from_provider, target_input_prompt_max_tokens,
                                                      calculated_n_ctx_for_model_load)

            # Recalculate input tokens after potential truncation
            current_input_token_count_raw = count_tokens_cl100k(prompt_for_llm)
            if current_input_token_count_raw == -1:  # Tiktoken error
                current_input_token_count_raw = len(prompt_for_llm) // 3
                log_worker("WARNING",
                           f"Tiktoken error for raw prompt, using char estimate: {current_input_token_count_raw} tokens.")

            original_kwargs_from_provider.setdefault("stop", ["<|im_end|>"])  # Ensure stop token
            # `max_tokens` for generation output, ensuring it fits remaining context
            max_gen_tokens_raw = min(
                original_kwargs_from_provider.get("max_tokens", calculated_n_ctx_for_model_load // 2),
                calculated_n_ctx_for_model_load - current_input_token_count_raw - 64  # Safety buffer
            )
            if max_gen_tokens_raw <= 0: max_gen_tokens_raw = GENERATION_OUTPUT_BUFFER_TOKENS // 2
            original_kwargs_from_provider["max_tokens"] = max_gen_tokens_raw

            log_worker("INFO",
                       f"Attempt 1 (Raw): PromptLen={len(prompt_for_llm)}, InputTokensEst={current_input_token_count_raw}, Effective GenMaxTokens={max_gen_tokens_raw}, Kwargs={original_kwargs_from_provider}")
            if current_input_token_count_raw + max_gen_tokens_raw > calculated_n_ctx_for_model_load:
                log_worker("ERROR",
                           f"Potential token overflow for RAW: Input ({current_input_token_count_raw}) + MaxGen ({max_gen_tokens_raw}) > Model n_ctx ({calculated_n_ctx_for_model_load}). Model might truncate input.")

            completion_result_dict = llm.create_completion(
                prompt=prompt_for_llm, stream=False, **original_kwargs_from_provider
            )

        elif args.task_type == "embedding":
            texts_to_embed = request_data_dict.get("texts")
            if not texts_to_embed or not isinstance(texts_to_embed, list) or not all(
                    isinstance(t, str) for t in texts_to_embed):
                raise ValueError("Invalid 'texts' for 'embedding' task.")
            log_worker("INFO", f"Performing embedding for {len(texts_to_embed)} texts...")
            embedding_vectors = llm.embed(texts_to_embed)
            result_payload = {"result": embedding_vectors}
        else:
            raise ValueError(f"Unknown task type in processing block: {args.task_type}")

        # --- Process and Cleanup/Re-request (if not embedding) ---
        if args.task_type != "embedding":
            if not completion_result_dict:
                raise RuntimeError("LLM call did not return a result dictionary on first attempt.")
            log_worker("DEBUG", f"Raw LLM Output (Attempt 1):\n{json.dumps(completion_result_dict, indent=2)}")

            raw_generated_text = ""
            if args.task_type == "chat":
                if (isinstance(completion_result_dict.get('choices'), list) and completion_result_dict['choices'] and
                        isinstance(completion_result_dict['choices'][0].get('message'), dict)):
                    raw_generated_text = completion_result_dict['choices'][0]['message'].get('content', "")
            elif args.task_type == "raw_text_completion":
                if (isinstance(completion_result_dict.get('choices'), list) and completion_result_dict['choices']):
                    raw_generated_text = completion_result_dict['choices'][0].get('text', "")

            if not isinstance(raw_generated_text, str): raw_generated_text = str(raw_generated_text or "")
            cleaned_text = cleanup_initial_think_tag(raw_generated_text)
            if cleaned_text != raw_generated_text: log_worker("INFO",
                                                              f"ThinkCleanup applied (Attempt 1). Original len: {len(raw_generated_text)}, Cleaned len: {len(cleaned_text)}")
            # log_worker("DEBUG", f"Cleaned Output (Attempt 1):\n{cleaned_text}") # Can be verbose

            token_count_after_cleanup = count_tokens_cl100k(cleaned_text)
            log_worker("INFO", f"Token count of cleaned output (Attempt 1): {token_count_after_cleanup}")

            # --- Re-request Logic for Zero Tokens ---
            if TIKTOKEN_AVAILABLE and token_count_after_cleanup == 0 and args.task_type != "embedding":  # Ensure not embedding
                log_worker("WARNING", "Cleaned output has zero tokens. Attempting re-request...")
                rerequest_kwargs = original_kwargs_from_provider.copy()  # Use kwargs from provider for 1st attempt
                complaint_prefix = "[System Note: Previous response was empty. Please provide a substantive answer.]\n"

                if args.task_type == "chat":
                    modified_messages_for_rerequest = request_data_dict.get("messages", []).copy()  # Original messages
                    if isinstance(modified_messages_for_rerequest, list):
                        modified_messages_for_rerequest.append({"role": "system", "content": complaint_prefix.strip()})

                    input_tokens_rerequest_chat = count_tokens_cl100k(modified_messages_for_rerequest)
                    if input_tokens_rerequest_chat == -1: input_tokens_rerequest_chat = len(
                        json.dumps(modified_messages_for_rerequest)) // 3

                    max_gen_tokens_rereq_chat = min(calculated_n_ctx_for_model_load // 2,
                                                    calculated_n_ctx_for_model_load - input_tokens_rerequest_chat - 128)
                    if max_gen_tokens_rereq_chat <= 0: max_gen_tokens_rereq_chat = 128
                    rerequest_kwargs["max_tokens"] = max_gen_tokens_rereq_chat

                    log_worker("INFO",
                               f"Attempt 2 (Chat): Re-requesting. InputTokensEst={input_tokens_rerequest_chat}, New GenMaxTokens={rerequest_kwargs['max_tokens']}.")
                    completion_result_dict = llm.create_chat_completion(messages=modified_messages_for_rerequest,
                                                                        stream=False, **rerequest_kwargs)

                elif args.task_type == "raw_text_completion":
                    # For raw, re-use the already (potentially) truncated prompt_for_llm from the first attempt.
                    # Prepend the complaint to it.
                    prompt_for_rerequest = complaint_prefix + prompt_for_llm  # prompt_for_llm is from adaptive_middle_truncate
                    input_tokens_rerequest_raw = count_tokens_cl100k(prompt_for_rerequest)
                    if input_tokens_rerequest_raw == -1: input_tokens_rerequest_raw = len(prompt_for_rerequest) // 3

                    max_gen_tokens_rereq_raw = min(calculated_n_ctx_for_model_load // 2,
                                                   calculated_n_ctx_for_model_load - input_tokens_rerequest_raw - 128)
                    if max_gen_tokens_rereq_raw <= 0: max_gen_tokens_rereq_raw = 128
                    rerequest_kwargs["max_tokens"] = max_gen_tokens_rereq_raw

                    log_worker("INFO",
                               f"Attempt 2 (Raw): Re-requesting. InputTokensEst={input_tokens_rerequest_raw}, New GenMaxTokens={rerequest_kwargs['max_tokens']}.")
                    completion_result_dict = llm.create_completion(prompt=prompt_for_rerequest, stream=False,
                                                                   **rerequest_kwargs)

                log_worker("INFO", "Re-request (Attempt 2) finished.")
                log_worker("DEBUG", f"Raw LLM Output (Attempt 2):\n{json.dumps(completion_result_dict, indent=2)}")
                raw_generated_text_attempt2 = ""
                if args.task_type == "chat":
                    if (isinstance(completion_result_dict.get('choices'), list) and completion_result_dict[
                        'choices'] and isinstance(completion_result_dict['choices'][0].get('message'), dict)):
                        raw_generated_text_attempt2 = completion_result_dict['choices'][0]['message'].get('content', "")
                elif args.task_type == "raw_text_completion":
                    if (isinstance(completion_result_dict.get('choices'), list) and completion_result_dict['choices']):
                        raw_generated_text_attempt2 = completion_result_dict['choices'][0].get('text', "")
                if not isinstance(raw_generated_text_attempt2, str): raw_generated_text_attempt2 = str(
                    raw_generated_text_attempt2 or "")
                cleaned_text_attempt2 = cleanup_initial_think_tag(raw_generated_text_attempt2)
                if cleaned_text_attempt2 != raw_generated_text_attempt2: log_worker("INFO",
                                                                                    f"ThinkCleanup applied (Attempt 2). Original len: {len(raw_generated_text_attempt2)}, Cleaned len: {len(cleaned_text_attempt2)}")
                # log_worker("DEBUG", f"Cleaned Output (Attempt 2):\n{cleaned_text_attempt2}")
                if args.task_type == "chat":
                    completion_result_dict['choices'][0]['message']['content'] = cleaned_text_attempt2
                elif args.task_type == "raw_text_completion":
                    completion_result_dict['choices'][0]['text'] = cleaned_text_attempt2

            result_payload = {"result": completion_result_dict}

        task_processing_duration_ms = (time.monotonic() - task_processing_start_time) * 1000
        log_worker("INFO",
                   f"Task '{args.task_type}' core processing completed in {task_processing_duration_ms:.2f} ms.")

    except json.JSONDecodeError as e_json_proc:  # Should be caught by initial stdin read
        log_worker("ERROR", f"Internal JSON error during processing (should have been caught earlier): {e_json_proc}")
        result_payload = {"error": f"Worker Internal JSON Error: {e_json_proc}"}
    except ValueError as e_val_proc:
        log_worker("ERROR", f"Input validation error during processing: {e_val_proc}")
        result_payload = {"error": f"Worker Input Error: {e_val_proc}"}
    except Exception as e_proc:
        log_worker("ERROR", f"Unexpected exception during task execution: {e_proc}")
        log_worker("ERROR", traceback.format_exc())
        result_payload = {"error": f"Worker Execution Error: {type(e_proc).__name__} - {e_proc}"}

    # --- Send Result to Parent ---
    try:
        output_json_to_parent = json.dumps(result_payload)
        # log_worker("DEBUG", f"Final result payload being sent to parent (len={len(output_json_to_parent)}):\n{output_json_to_parent[:200]}...")
        print(output_json_to_parent, flush=True)
        log_worker("INFO", "Result/Error JSON sent to stdout successfully.")
    except Exception as e_serialize:
        log_worker("CRITICAL", f"Failed to serialize/write final result_payload to stdout: {e_serialize}")
        fallback_error_msg = {
            "error": f"Worker critical error: Failed to serialize/write result to stdout: {e_serialize}"}
        try:
            print(json.dumps(fallback_error_msg), flush=True)
        except Exception:
            pass
        sys.exit(1)

    log_worker("INFO", f"Llama Worker (PID: {os.getpid()}) process finished gracefully.")
    sys.exit(0)


if __name__ == "__main__":
    if not LLAMA_CPP_AVAILABLE:
        log_worker("CRITICAL", "Llama.cpp library not available. Worker cannot run.")
        sys.exit(1)  # Exit if core lib missing
    main()