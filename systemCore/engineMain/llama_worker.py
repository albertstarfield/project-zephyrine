# llama_worker.py
import sys
import os
import json
import time
import traceback
import argparse
import re

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
        f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} WORKER|WARNING] tiktoken library not found. Zero-token check and re-request will be disabled.",
        file=sys.stderr, flush=True)


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
        log_worker("TRACE", f"ThinkCleanup: Found initial think tag. Remainder len: {len(remaining_text)}")
        return remaining_text
    return text


# --- Token Counting Helper ---
def count_tokens_cl100k(text: str) -> int:
    if not TIKTOKEN_AVAILABLE or text is None:
        return -1  # Indicate not countable or no text
    try:
        # Use the globally loaded encoder
        tokens = cl100k_base_encoder.encode(text)
        return len(tokens)
    except Exception as e:
        log_worker("ERROR", f"Tiktoken counting error: {e}")
        return -2  # Indicate counting error


def main():
    parser = argparse.ArgumentParser(description="Llama.cpp Worker Process")
    parser.add_argument("--model-path", required=True, help="Path to GGUF model file")
    parser.add_argument("--task-type", required=True, choices=["chat", "embedding", "raw_text_completion"],
                        help="Task type")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="Number of GPU layers")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context size")
    parser.add_argument("--verbose", action="store_true", help="Enable llama.cpp verbose logging")
    parser.add_argument("--chat-format", type=str, default=None, help="Chat format for 'chat' task")
    args = parser.parse_args()

    # ... (initial logging and model loading as in V20) ...
    log_worker("INFO",
               f"Starting worker. PID: {os.getpid()}. Task: '{args.task_type}'. Model: '{os.path.basename(args.model_path)}'. GPU Layers: {args.n_gpu_layers}. Context: {args.n_ctx}. Verbose: {args.verbose}. ChatFmt: {args.chat_format if args.task_type == 'chat' else 'N/A'}")
    if TIKTOKEN_AVAILABLE:
        log_worker("INFO", "Tiktoken is available for token counting.")
    else:
        log_worker("WARNING", "Tiktoken is NOT available. Zero-token re-request feature will be disabled.")

    llm = None
    try:
        load_start_time = time.monotonic()
        effective_chat_format = args.chat_format if args.task_type == "chat" else None
        if args.task_type == "raw_text_completion" and args.chat_format:
            effective_chat_format = None
        llm = llama_cpp.Llama(
            model_path=args.model_path, n_gpu_layers=args.n_gpu_layers, n_ctx=args.n_ctx,
            embedding=(args.task_type == "embedding"), verbose=args.verbose, chat_format=effective_chat_format,
        )
        log_worker("INFO", f"Model loaded in {(time.monotonic() - load_start_time) * 1000:.2f} ms.")
    except Exception as e:
        log_worker("CRITICAL", f"Model loading failed: {e}\n{traceback.format_exc()}")
        print(json.dumps({"error": f"Worker critical error: Failed to load model: {e}"}), flush=True)
        sys.exit(1)

    request_data = None
    result_payload = {"error": "Worker did not receive or process valid input."}
    input_json_str_for_logging = ""

    try:
        log_worker("INFO", "Waiting for request data on stdin...")
        input_json_str_for_logging = sys.stdin.read()
        if not input_json_str_for_logging.strip(): raise ValueError("Received empty input string from stdin.")
        log_worker("DEBUG",
                   f"Received raw input string (len={len(input_json_str_for_logging)}):\n----------\n{input_json_str_for_logging}\n----------")
        request_data = json.loads(input_json_str_for_logging)
        log_worker("INFO", f"Request data JSON parsed for task: {args.task_type}")

        task_processing_start_time = time.monotonic()
        completion_result_dict = None

        # --- Store original request parts for potential re-request ---
        original_prompt_for_rerequest = None
        original_messages_for_rerequest = None
        original_kwargs_for_rerequest = request_data.get("kwargs", {}).copy()  # Get a copy

        # --- First Attempt ---
        if args.task_type == "chat":
            original_messages_for_rerequest = request_data.get("messages")
            if not original_messages_for_rerequest or not isinstance(original_messages_for_rerequest, list):
                raise ValueError("Missing or invalid 'messages' for 'chat' task.")
            log_worker("INFO",
                       f"Attempt 1: Chat completion with {len(original_messages_for_rerequest)} messages. Kwargs: {original_kwargs_for_rerequest}")
            completion_result_dict = llm.create_chat_completion(
                messages=original_messages_for_rerequest, stream=False, **original_kwargs_for_rerequest
            )
        elif args.task_type == "raw_text_completion":
            original_prompt_for_rerequest = request_data.get("prompt")
            if not original_prompt_for_rerequest or not isinstance(original_prompt_for_rerequest, str):
                raise ValueError("Missing or invalid 'prompt' for 'raw_text_completion' task.")
            original_kwargs_for_rerequest.setdefault("stop", ["<|im_end|>"])  # Ensure stop
            log_worker("INFO",
                       f"Attempt 1: Raw text completion. Prompt len: {len(original_prompt_for_rerequest)}. Kwargs: {original_kwargs_for_rerequest}")
            completion_result_dict = llm.create_completion(
                prompt=original_prompt_for_rerequest, stream=False, **original_kwargs_for_rerequest
            )
        elif args.task_type == "embedding":  # Embeddings don't need re-request logic
            texts_to_embed = request_data.get("texts")
            if not texts_to_embed or not isinstance(texts_to_embed, list) or not all(
                    isinstance(t, str) for t in texts_to_embed):
                raise ValueError("Invalid 'texts' for 'embedding' task.")
            log_worker("INFO", f"Performing embedding for {len(texts_to_embed)} texts...")
            embedding_vectors = llm.embed(texts_to_embed)
            result_payload = {"result": embedding_vectors}
            # Skip cleanup and re-request for embeddings
        else:  # Should not happen due to arg choices
            raise ValueError(f"Unknown task type: {args.task_type}")

        # --- Process and Cleanup First Attempt (if not embedding) ---
        if args.task_type != "embedding":
            if not completion_result_dict:  # Should not happen if no exception above
                raise RuntimeError("LLM call did not return a result dictionary on first attempt.")

            log_worker("DEBUG",
                       f"Raw LLM Output (Attempt 1):\n----------\n{json.dumps(completion_result_dict, indent=2)}\n----------")

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
            if cleaned_text != raw_generated_text:
                log_worker("INFO",
                           f"ThinkCleanup applied (Attempt 1). Original len: {len(raw_generated_text)}, Cleaned len: {len(cleaned_text)}")
            log_worker("DEBUG", f"Cleaned Output (Attempt 1):\n----------\n{cleaned_text}\n----------")

            token_count_after_cleanup = count_tokens_cl100k(cleaned_text)
            log_worker("INFO", f"Token count of cleaned output (Attempt 1): {token_count_after_cleanup} (cl100k_base)")

            # --- Re-request Logic ---
            if TIKTOKEN_AVAILABLE and token_count_after_cleanup == 0:
                log_worker("WARNING", "Cleaned output has zero tokens. Attempting re-request with modified parameters.")

                rerequest_kwargs = original_kwargs_for_rerequest.copy()
                # Modify kwargs for re-request: e.g., increase max_tokens
                current_max_tokens = rerequest_kwargs.get("max_tokens", args.n_ctx // 2)  # Default if not set
                new_max_tokens = min(args.n_ctx - 256, int(current_max_tokens * 1.5) + 128)  # Increase, but cap
                rerequest_kwargs["max_tokens"] = new_max_tokens
                # Optionally, slightly increase temperature or change other params for diversity
                # rerequest_kwargs["temperature"] = min(1.5, rerequest_kwargs.get("temperature", 0.8) + 0.2)

                complaint_prefix = "[System Note: Previous response was empty after processing. Please provide a direct and substantive answer to the following:]\n"

                if args.task_type == "chat":
                    # Prepend complaint to the last user message or add as a new system message
                    modified_messages_for_rerequest = original_messages_for_rerequest.copy()
                    # Simplest: add a new system message at the end before assistant is cued
                    # This requires the ChatML constructor to handle a system message before assistant.
                    # Or, modify the last user message.
                    # Let's try adding a system message *before* the last user message if possible,
                    # or just prepend to the whole message list if it's simpler for ChatML structure.
                    # For now, let's add it as a new system turn just before the point where assistant would respond.

                    # This is tricky with raw ChatML if the original was already a full string.
                    # If original_messages_for_rerequest is List[Dict], we can add a system turn.
                    if isinstance(modified_messages_for_rerequest, list):
                        # Find last user message to insert system complaint before it, then re-add user message
                        # This is complex if we are not sure about the structure of original_messages_for_rerequest
                        # A simpler approach for Chat (if it's a list of dicts):
                        # Add a new system turn at the beginning of history for the re-request
                        # This assumes the `llm.create_chat_completion` can handle this.
                        # For true raw ChatML, we'd rebuild the string.

                        # Since the worker's 'chat' task uses the model's chat_format,
                        # adding a system message to the list is the Langchain way.
                        modified_messages_for_rerequest.append({"role": "system", "content": complaint_prefix.strip()})
                        # Then the original user query will be re-processed by the model with this new system guidance.

                    log_worker("INFO",
                               f"Attempt 2 (Chat): Re-requesting with modified messages. New kwargs: {rerequest_kwargs}")
                    completion_result_dict = llm.create_chat_completion(
                        messages=modified_messages_for_rerequest, stream=False, **rerequest_kwargs
                    )

                elif args.task_type == "raw_text_completion":
                    prompt_for_rerequest = complaint_prefix + original_prompt_for_rerequest
                    log_worker("INFO",
                               f"Attempt 2 (Raw): Re-requesting with prepended complaint. New kwargs: {rerequest_kwargs}. New prompt len: {len(prompt_for_rerequest)}")
                    completion_result_dict = llm.create_completion(
                        prompt=prompt_for_rerequest, stream=False, **rerequest_kwargs
                    )

                log_worker("INFO", "Re-request (Attempt 2) finished.")
                log_worker("DEBUG",
                           f"Raw LLM Output (Attempt 2):\n----------\n{json.dumps(completion_result_dict, indent=2)}\n----------")

                # Re-process and re-cleanup
                raw_generated_text_attempt2 = ""
                if args.task_type == "chat":
                    if (isinstance(completion_result_dict.get('choices'), list) and completion_result_dict[
                        'choices'] and
                            isinstance(completion_result_dict['choices'][0].get('message'), dict)):
                        raw_generated_text_attempt2 = completion_result_dict['choices'][0]['message'].get('content', "")
                elif args.task_type == "raw_text_completion":
                    if (isinstance(completion_result_dict.get('choices'), list) and completion_result_dict['choices']):
                        raw_generated_text_attempt2 = completion_result_dict['choices'][0].get('text', "")

                if not isinstance(raw_generated_text_attempt2, str): raw_generated_text_attempt2 = str(
                    raw_generated_text_attempt2 or "")

                cleaned_text_attempt2 = cleanup_initial_think_tag(raw_generated_text_attempt2)
                if cleaned_text_attempt2 != raw_generated_text_attempt2:
                    log_worker("INFO",
                               f"ThinkCleanup applied (Attempt 2). Original len: {len(raw_generated_text_attempt2)}, Cleaned len: {len(cleaned_text_attempt2)}")
                log_worker("DEBUG", f"Cleaned Output (Attempt 2):\n----------\n{cleaned_text_attempt2}\n----------")

                # Update the completion_result_dict with the twice-cleaned text
                if args.task_type == "chat":
                    completion_result_dict['choices'][0]['message']['content'] = cleaned_text_attempt2
                elif args.task_type == "raw_text_completion":
                    completion_result_dict['choices'][0]['text'] = cleaned_text_attempt2

            # After potential re-request, set the final payload
            result_payload = {"result": completion_result_dict}

        # If it was an embedding task, result_payload is already set
        task_processing_duration_ms = (time.monotonic() - task_processing_start_time) * 1000
        log_worker("INFO",
                   f"Task '{args.task_type}' core processing (incl. potential re-request) completed in {task_processing_duration_ms:.2f} ms.")

    except json.JSONDecodeError as e:
        log_worker("ERROR", f"Failed to decode JSON input: {e}")
        log_worker("ERROR",
                   f"Invalid input string received by worker (first 500 chars): '{input_json_str_for_logging[:500]}'")
        result_payload = {
            "error": f"Worker JSON Decode Error: {e}. Input started with: {input_json_str_for_logging[:100]}"}
    except ValueError as e:
        log_worker("ERROR", f"Input validation error: {e}")
        result_payload = {"error": f"Worker Input Error: {e}"}
    except Exception as e:
        log_worker("ERROR", f"Unexpected exception during task execution: {e}")
        log_worker("ERROR", traceback.format_exc())
        result_payload = {"error": f"Worker Execution Error: {e}"}

    try:
        output_json_to_parent = json.dumps(result_payload)
        log_worker("DEBUG",
                   f"Final result payload being sent to parent (len={len(output_json_to_parent)}):\n----------\n{output_json_to_parent}\n----------")
        print(output_json_to_parent, flush=True)
        log_worker("INFO", "Result/Error JSON sent to stdout successfully.")
    except Exception as e_serialize:
        log_worker("CRITICAL", f"Failed to serialize/write final result_payload to stdout: {e_serialize}")
        log_worker("CRITICAL",
                   f"Payload that failed serialization (type: {type(result_payload)}): {str(result_payload)[:500]}")
        fallback_error_msg = {
            "error": f"Worker critical error: Failed to serialize or write result to stdout: {e_serialize}"}
        try:
            print(json.dumps(fallback_error_msg), flush=True)
        except Exception:
            pass
        sys.exit(1)

    log_worker("INFO", f"Llama Worker (PID: {os.getpid()}) process finished gracefully.")
    sys.exit(0)


if __name__ == "__main__":
    if not LLAMA_CPP_AVAILABLE:
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} WORKER|CRITICAL] Llama.cpp library not available. Worker cannot run.",
            file=sys.stderr, flush=True)
        sys.exit(1)
    main()