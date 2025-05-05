# llama_worker.py
import sys
import os
import json
import time
import traceback
import argparse

# --- Try importing llama_cpp ---
# Assume it's installed in the same environment launcher.py uses
try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    # Write error to stderr so the parent process can see it
    print(json.dumps({"error": "llama-cpp-python not found in worker environment."}), file=sys.stderr)
    sys.exit(1)

# --- Basic Logging to stderr (for debugging the worker itself) ---
def log_worker(level, message):
    print(f"[WORKER|{level}] {message}", file=sys.stderr, flush=True)

def main():
    parser = argparse.ArgumentParser(description="Llama.cpp Worker Process")
    parser.add_argument("--model-path", required=True, help="Path to GGUF model file")
    parser.add_argument("--task-type", required=True, choices=["chat", "embedding"], help="Task type: chat or embedding")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="Number of GPU layers")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context size")
    parser.add_argument("--verbose", action="store_true", help="Enable llama.cpp verbose logging")
    # Add other necessary params like chat_format if needed for chat
    parser.add_argument("--chat-format", type=str, default=None, help="Chat format for chat task")

    args = parser.parse_args()
    log_worker("INFO", f"Starting worker for task '{args.task_type}' with model '{os.path.basename(args.model_path)}'")

    # --- Load Model ---
    llm = None
    try:
        load_start = time.monotonic()
        llm = llama_cpp.Llama(
            model_path=args.model_path,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.n_ctx,
            embedding=(args.task_type == "embedding"),
            verbose=args.verbose,
            chat_format=args.chat_format if args.task_type == "chat" else None,
             # Hardcode seed or make it configurable if needed
            # seed=1337,
        )
        load_duration = time.monotonic() - load_start
        log_worker("INFO", f"Model loaded successfully in {load_duration:.2f}s.")
    except Exception as e:
        log_worker("ERROR", f"Model loading failed: {e}")
        log_worker("ERROR", traceback.format_exc())
        # Send error back to parent via stdout (as JSON)
        print(json.dumps({"error": f"Worker failed to load model: {e}"}), flush=True)
        sys.exit(1) # Exit indicates failure

    # --- Read Request from stdin ---
    request_data = None
    result_payload = {"error": "Worker did not receive valid input"} # Default error
    try:
        log_worker("INFO", "Waiting for request data on stdin...")
        input_json_str = sys.stdin.read()
        log_worker("DEBUG", f"Received raw input string (len={len(input_json_str)}): {input_json_str[:200]}...")
        if not input_json_str:
            raise ValueError("Received empty input from stdin.")
        request_data = json.loads(input_json_str)
        log_worker("INFO", f"Request data JSON parsed successfully. Task: {args.task_type}")

        # --- Execute Task ---
        task_start = time.monotonic()
        if args.task_type == "chat":
            messages = request_data.get("messages")
            completion_kwargs = request_data.get("kwargs", {})
            if not messages: raise ValueError("Missing 'messages' for chat task.")

            log_worker("INFO", f"Performing chat completion with {len(messages)} messages...")
            # Use stream=False for simplicity in worker communication
            completion = llm.create_chat_completion(messages=messages, stream=False, **completion_kwargs)
            log_worker("INFO", "Chat completion finished.")
            result_payload = {"result": completion} # Send back the full completion dict

        elif args.task_type == "embedding":
            texts = request_data.get("texts")
            if not texts: raise ValueError("Missing 'texts' for embedding task.")

            log_worker("INFO", f"Performing embedding for {len(texts)} text(s)...")
            embeddings = llm.embed(texts)
            log_worker("INFO", "Embedding finished.")
            result_payload = {"result": embeddings} # Send back the list of embeddings

        task_duration = time.monotonic() - task_start
        log_worker("INFO", f"Task '{args.task_type}' completed in {task_duration:.2f}s")

    except json.JSONDecodeError as e:
        log_worker("ERROR", f"Failed to decode JSON input: {e}")
        log_worker("ERROR", f"Invalid input received: {input_json_str[:500]}") # Log problematic input
        result_payload = {"error": f"Worker failed to decode JSON input: {e}"}
    except ValueError as e:
        log_worker("ERROR", f"Input validation error: {e}")
        result_payload = {"error": f"Worker input validation error: {e}"}
    except Exception as e:
        log_worker("ERROR", f"Exception during task execution: {e}")
        log_worker("ERROR", traceback.format_exc())
        result_payload = {"error": f"Worker execution error: {e}"}
        # Note: GGML Asserts/Segfaults might kill the process *before* this point

    # --- Write Result/Error to stdout ---
    try:
        output_json = json.dumps(result_payload)
        log_worker("DEBUG", f"Sending output JSON (len={len(output_json)}): {output_json[:200]}...")
        print(output_json, flush=True)
        log_worker("INFO", "Result sent to stdout.")
    except Exception as e:
        # If writing the result fails, log it to stderr
        log_worker("ERROR", f"Failed to serialize/write result to stdout: {e}")
        # Attempt to write a simple error message to stdout if possible
        try: print(json.dumps({"error": f"Worker failed to write result: {e}"}), flush=True)
        except: pass
        sys.exit(1) # Indicate failure

    log_worker("INFO", "Worker process finished successfully.")
    sys.exit(0) # Explicitly exit with success code

if __name__ == "__main__":
    # Ensure llama_cpp is available before trying to run
    if not LLAMA_CPP_AVAILABLE:
        log_worker("ERROR", "Llama.cpp library not available. Worker cannot run.")
        sys.exit(1)
    main()