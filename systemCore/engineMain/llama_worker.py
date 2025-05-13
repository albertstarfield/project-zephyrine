# llama_worker.py

import sys
import os
import json
import time
import traceback
import argparse
import re
from typing import Union, List, Dict

# --- Try importing llama_cpp ---
try:
    import llama_cpp

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print(json.dumps({"error": "llama-cpp-python not found in worker environment."}), flush=True)
    sys.exit(1)

# --- Try importing tiktoken ---
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
    try:
        cl100k_base_encoder = tiktoken.get_encoding("cl100k_base")
    except Exception as e_enc:
        try:
            cl100k_base_encoder = tiktoken.encoding_for_model("gpt-4")
        except Exception as e_enc_fallback:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} WORKER|ERROR] Failed to load tiktoken cl100k_base encoder: {e_enc}, Fallback: {e_enc_fallback}",
                file=sys.stderr, flush=True)
            TIKTOKEN_AVAILABLE = False
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} WORKER|WARNING] tiktoken library not found. Token counting/dynamic n_ctx limited.",
        file=sys.stderr, flush=True)

# --- Constants for Dynamic Context ---
MIN_DYNAMIC_N_CTX = 4096  # Minimum context size for dynamic calculation
MAX_DYNAMIC_N_CTX = 32768  # Maximum context size for dynamic calculation
DEFAULT_N_CTX_FOR_FALLBACK = 4096  # If token counting fails or not applicable


# --- Basic Logging to stderr ---
def log_worker(level, message):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{now} WORKER|{level}] {message}", file=sys.stderr, flush=True)


# --- Think Tag Cleanup Helper ---
def cleanup_initial_think_tag(text: str) -> str:
    # ... (same as before) ...
    if not text: return ""
    match = re.match(r"<think>(.*?)</think>(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        remaining_text = match.group(2).lstrip()
        log_worker("TRACE", f"ThinkCleanup: Found initial think tag. Remainder len: {len(remaining_text)}")
        return remaining_text
    return text


# --- Token Counting Helper ---
def count_tokens_cl100k(text_or_messages: Union[str, List[Dict[str, str]]]) -> int:
    if not TIKTOKEN_AVAILABLE or not text_or_messages:
        return -1

    total_tokens = 0
    try:
        if isinstance(text_or_messages, str):
            tokens = cl100k_base_encoder.encode(text_or_messages)
            total_tokens = len(tokens)
        elif isinstance(text_or_messages, list):
            # For chat messages, count tokens for content of each message
            # A more precise count would include role tokens if the model uses them internally
            for msg in text_or_messages:
                if isinstance(msg, dict) and "content" in msg and isinstance(msg["content"], str):
                    tokens = cl100k_base_encoder.encode(msg["content"])
                    total_tokens += len(tokens)
                    # Add a few tokens per message for role/structure overhead (e.g., <|im_start|>, role, <|im_end|>)
                    total_tokens += 4  # Approximation
        return total_tokens
    except Exception as e:
        log_worker("ERROR", f"Tiktoken counting error: {e}")
        return -2


def main():
    parser = argparse.ArgumentParser(description="Llama.cpp Worker Process")
    parser.add_argument("--model-path", required=True, help="Path to GGUF model file")
    parser.add_argument("--task-type", required=True, choices=["chat", "embedding", "raw_text_completion"],
                        help="Task type")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="Number of GPU layers")
    # n_ctx is now optional for chat/raw_text_completion, but required for embeddings or as an override
    parser.add_argument("--n-ctx", type=int, default=None,
                        help="Context size (fixed for embeddings, optional override for others)")
    parser.add_argument("--verbose", action="store_true", help="Enable llama.cpp verbose logging")
    parser.add_argument("--chat-format", type=str, default=None, help="Chat format for 'chat' task")
    args = parser.parse_args()

    # --- Read Request Data from Stdin FIRST to determine dynamic n_ctx ---
    request_data_str = ""
    request_data_dict = None
    input_tokens_for_dynamic_ctx = 0
    calculated_n_ctx = DEFAULT_N_CTX_FOR_FALLBACK  # Default

    try:
        log_worker("INFO", "Reading request data from stdin for dynamic n_ctx calculation...")
        request_data_str = sys.stdin.read()
        if not request_data_str.strip():
            raise ValueError("Received empty input string from stdin.")
        request_data_dict = json.loads(request_data_str)
        log_worker("DEBUG", f"Request data JSON parsed for dynamic n_ctx calc. Task: {args.task_type}")

        if args.task_type == "chat":
            messages_for_count = request_data_dict.get("messages")
            if messages_for_count:
                input_tokens_for_dynamic_ctx = count_tokens_cl100k(messages_for_count)
        elif args.task_type == "raw_text_completion":
            prompt_for_count = request_data_dict.get("prompt")
            if prompt_for_count:
                input_tokens_for_dynamic_ctx = count_tokens_cl100k(prompt_for_count)

        log_worker("INFO", f"Estimated input tokens for dynamic n_ctx: {input_tokens_for_dynamic_ctx}")

    except Exception as e_stdin:
        log_worker("CRITICAL", f"Failed to read/parse stdin for dynamic n_ctx: {e_stdin}\n{traceback.format_exc()}")
        # Fallback or exit if essential data missing
        # If we can't read stdin, we can't process the request anyway.
        print(json.dumps({"error": f"Worker critical error: Failed to read/parse request for n_ctx: {e_stdin}"}),
              flush=True)
        sys.exit(1)

    # --- Determine n_ctx ---
    if args.task_type == "embedding":
        calculated_n_ctx = args.n_ctx if args.n_ctx is not None else 512  # Embeddings specific context
        if args.n_ctx is None: log_worker("INFO", "Using fixed n_ctx=512 for embeddings task.")
    elif args.n_ctx is not None:  # User explicitly provided n_ctx as override
        calculated_n_ctx = args.n_ctx
        log_worker("INFO", f"Using explicitly provided n_ctx={calculated_n_ctx} (override).")
    elif input_tokens_for_dynamic_ctx > 0:  # Valid token count for dynamic calculation
        # Dynamic n_ctx: input tokens * 2, with min and max caps
        dynamic_ctx = input_tokens_for_dynamic_ctx * 2
        calculated_n_ctx = max(MIN_DYNAMIC_N_CTX, min(dynamic_ctx, MAX_DYNAMIC_N_CTX))
        log_worker("INFO",
                   f"Dynamically calculated n_ctx: {calculated_n_ctx} (InputTokens: {input_tokens_for_dynamic_ctx}, Dynamic Attempt: {dynamic_ctx}, Min: {MIN_DYNAMIC_N_CTX}, Max: {MAX_DYNAMIC_N_CTX})")
    else:  # Fallback if token counting failed or not applicable and no override
        calculated_n_ctx = DEFAULT_N_CTX_FOR_FALLBACK
        log_worker("INFO", f"Falling back to default n_ctx={calculated_n_ctx} (token count invalid or task type).")

    log_worker("INFO", f"--- Worker Configuration ---")
    log_worker("INFO", f"  PID: {os.getpid()}")
    log_worker("INFO", f"  Task Type: '{args.task_type}'")
    log_worker("INFO", f"  Model: '{os.path.basename(args.model_path)}'")
    log_worker("INFO", f"  GPU Layers: {args.n_gpu_layers}")
    log_worker("INFO", f"  Effective n_ctx: {calculated_n_ctx}")  # Log the n_ctx that will be used
    log_worker("INFO", f"  Verbose: {args.verbose}")
    log_worker("INFO", f"  Chat Format: {args.chat_format if args.task_type == 'chat' else 'N/A'}")
    if TIKTOKEN_AVAILABLE:
        log_worker("INFO", "  Tiktoken: Available")
    else:
        log_worker("WARNING",
                   "  Tiktoken: NOT available (dynamic n_ctx may be less accurate / zero-token check disabled)")
    log_worker("INFO", f"----------------------------")

    llm = None
    try:
        load_start_time = time.monotonic()
        effective_chat_format = args.chat_format if args.task_type == "chat" else None
        # Ensure chat_format is None if it's a raw text completion, even if passed
        if args.task_type == "raw_text_completion" and args.chat_format:
            effective_chat_format = None  # Explicitly nullify

        llm = llama_cpp.Llama(
            model_path=args.model_path,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=calculated_n_ctx,  # Use the determined context size
            embedding=(args.task_type == "embedding"),
            verbose=args.verbose,
            chat_format=effective_chat_format,
        )
        log_worker("INFO",
                   f"Model loaded in {(time.monotonic() - load_start_time) * 1000:.2f} ms using n_ctx={calculated_n_ctx}.")
    except Exception as e:
        log_worker("CRITICAL", f"Model loading failed: {e}\n{traceback.format_exc()}")
        # Send error back to parent (ai_provider)
        print(json.dumps({"error": f"Worker critical error: Failed to load model with n_ctx={calculated_n_ctx}: {e}"}),
              flush=True)
        sys.exit(1)

    # --- Request Processing (using request_data_dict from earlier) ---
    result_payload = {"error": "Worker did not process a valid task."}  # Default
    try:
        task_processing_start_time = time.monotonic()
        completion_result_dict = None
        original_prompt_for_rerequest = None
        original_messages_for_rerequest = None
        # Use .get("kwargs", {}) on request_data_dict now
        original_kwargs_for_rerequest = request_data_dict.get("kwargs", {}).copy()

        if args.task_type == "chat":
            original_messages_for_rerequest = request_data_dict.get("messages")
            if not original_messages_for_rerequest or not isinstance(original_messages_for_rerequest, list):
                raise ValueError("Missing or invalid 'messages' for 'chat' task.")
            # Ensure max_tokens is not greater than effective context - safety buffer
            original_kwargs_for_rerequest["max_tokens"] = min(
                original_kwargs_for_rerequest.get("max_tokens", calculated_n_ctx // 2),
                calculated_n_ctx - input_tokens_for_dynamic_ctx - 128  # Buffer for prompt + response
            )
            if original_kwargs_for_rerequest["max_tokens"] <= 0: original_kwargs_for_rerequest[
                "max_tokens"] = 128  # Min sensible value

            log_worker("INFO",
                       f"Attempt 1: Chat completion with {len(original_messages_for_rerequest)} messages. Effective MaxTokens: {original_kwargs_for_rerequest['max_tokens']}. Kwargs: {original_kwargs_for_rerequest}")
            completion_result_dict = llm.create_chat_completion(
                messages=original_messages_for_rerequest, stream=False, **original_kwargs_for_rerequest
            )
        elif args.task_type == "raw_text_completion":
            original_prompt_for_rerequest = request_data_dict.get("prompt")
            if not original_prompt_for_rerequest or not isinstance(original_prompt_for_rerequest, str):
                raise ValueError("Missing or invalid 'prompt' for 'raw_text_completion' task.")
            original_kwargs_for_rerequest.setdefault("stop", ["<|im_end|>"])
            # Ensure max_tokens is not greater than effective context - safety buffer
            original_kwargs_for_rerequest["max_tokens"] = min(
                original_kwargs_for_rerequest.get("max_tokens", calculated_n_ctx // 2),
                calculated_n_ctx - input_tokens_for_dynamic_ctx - 128  # Buffer
            )
            if original_kwargs_for_rerequest["max_tokens"] <= 0: original_kwargs_for_rerequest["max_tokens"] = 128

            log_worker("INFO",
                       f"Attempt 1: Raw text completion. Prompt len: {len(original_prompt_for_rerequest)}. Effective MaxTokens: {original_kwargs_for_rerequest['max_tokens']}. Kwargs: {original_kwargs_for_rerequest}")
            completion_result_dict = llm.create_completion(
                prompt=original_prompt_for_rerequest, stream=False, **original_kwargs_for_rerequest
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

        # --- Process and Cleanup First Attempt (if not embedding) ---
        if args.task_type != "embedding":
            # ... (Re-request logic for zero tokens as before, using calculated_n_ctx for limits) ...
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
            log_worker("DEBUG", f"Cleaned Output (Attempt 1):\n{cleaned_text}")
            token_count_after_cleanup = count_tokens_cl100k(cleaned_text)
            log_worker("INFO", f"Token count of cleaned output (Attempt 1): {token_count_after_cleanup} (cl100k_base)")

            if TIKTOKEN_AVAILABLE and token_count_after_cleanup == 0:
                log_worker("WARNING", "Cleaned output has zero tokens. Attempting re-request...")
                rerequest_kwargs = original_kwargs_for_rerequest.copy()
                current_max_tokens = rerequest_kwargs.get("max_tokens", calculated_n_ctx // 2)
                new_max_tokens = min(calculated_n_ctx - input_tokens_for_dynamic_ctx - 128,
                                     int(current_max_tokens * 1.5) + 128)  # Cap by new calculated_n_ctx
                if new_max_tokens <= 0: new_max_tokens = 128  # ensure positive
                rerequest_kwargs["max_tokens"] = new_max_tokens
                complaint_prefix = "[System Note: Previous response was empty after processing. Please provide a direct and substantive answer to the following:]\n"
                if args.task_type == "chat":
                    modified_messages_for_rerequest = original_messages_for_rerequest.copy()
                    if isinstance(modified_messages_for_rerequest, list):
                        modified_messages_for_rerequest.append({"role": "system", "content": complaint_prefix.strip()})
                    log_worker("INFO",
                               f"Attempt 2 (Chat): Re-requesting. New MaxTokens: {rerequest_kwargs['max_tokens']}. Kwargs: {rerequest_kwargs}")
                    completion_result_dict = llm.create_chat_completion(messages=modified_messages_for_rerequest,
                                                                        stream=False, **rerequest_kwargs)
                elif args.task_type == "raw_text_completion":
                    prompt_for_rerequest = complaint_prefix + original_prompt_for_rerequest
                    log_worker("INFO",
                               f"Attempt 2 (Raw): Re-requesting. New MaxTokens: {rerequest_kwargs['max_tokens']}. Prompt len: {len(prompt_for_rerequest)}. Kwargs: {rerequest_kwargs}")
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
                log_worker("DEBUG", f"Cleaned Output (Attempt 2):\n{cleaned_text_attempt2}")
                if args.task_type == "chat":
                    completion_result_dict['choices'][0]['message']['content'] = cleaned_text_attempt2
                elif args.task_type == "raw_text_completion":
                    completion_result_dict['choices'][0]['text'] = cleaned_text_attempt2

            result_payload = {"result": completion_result_dict}

        task_processing_duration_ms = (time.monotonic() - task_processing_start_time) * 1000
        log_worker("INFO",
                   f"Task '{args.task_type}' core processing completed in {task_processing_duration_ms:.2f} ms.")

    except json.JSONDecodeError as e:  # This shouldn't happen here anymore if stdin read first
        log_worker("ERROR", f"Internal JSON error (should not occur here): {e}")
        result_payload = {"error": f"Worker Internal JSON Error: {e}"}
    except ValueError as e:
        log_worker("ERROR", f"Input validation error: {e}")
        result_payload = {"error": f"Worker Input Error: {e}"}
    except Exception as e:
        log_worker("ERROR", f"Unexpected exception during task execution: {e}")
        log_worker("ERROR", traceback.format_exc())
        result_payload = {"error": f"Worker Execution Error: {type(e).__name__} - {e}"}

    # --- Send Result to Parent ---
    try:
        output_json_to_parent = json.dumps(result_payload)
        log_worker("DEBUG",
                   f"Final result payload being sent to parent (len={len(output_json_to_parent)}):\n{output_json_to_parent[:200]}...")
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
        sys.exit(1)
    main()