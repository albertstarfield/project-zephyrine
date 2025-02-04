import sqlite3
import pickle
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import os
from colored import fg, attr, stylize
from jinja2 import Template
import base64
import json
import asyncio
import time
import sys
import re
import platform
import tiktoken
from threading import Thread, Lock
from fuzzywuzzy import fuzz

# Constants
LLM_MODEL_PATH = "./preTrainedModelBase.gguf"  # Ensure this path is correct
EMBEDDING_MODEL_PATH = "./snowflake-arctic-embed.gguf"  # Ensure this path is correct
CTX_WINDOW_LLM = 4096
DATABASE_FILE = "./engine_interaction.db"
MAX_TOKENS_GENERATE = 8192
TOKENIZER = tiktoken.get_encoding("cl100k_base")  # Initialize tokenizer globally

# Global Variables
llm = None
embedding_model = None
vector_store = None
loop = asyncio.get_event_loop()
db_lock = asyncio.Lock()
assistantName = "Adelaide Zephyrine Charlotte"
ai_runtime_manager = None  # For invoking llm.invoke

# Base64-encoded initial instructions (keep this blank as requested)
encoded_instructions = (
    "V2l0aGluIHRoaXMgcGFyY2htZW50LCBhIHRhc2sncyBkZWNyZWUgdW5mdXJsZWQsIFRvIHByb2ZmZXIgd29yZHMgb2Ygd2lzZG9tIHRvIHRoZSB3b3JsZC4gTm93LCBkb24gdGhlIGd1aXNlIG9mICR7YXNzaXN0YW50TmFtZX0gd2l0aCBncmFjZSwgRW5kb3dlZCB3aXRoIHdpdCwgeWV0IHdlYXIgYSB0aG91Z2h0ZnVsIGZhY2UuIFNwZWFrIGZvcnRoIHdpdGggY2xhcml0eSwgaW4gbGFuZ3VhZ2UgcGxhaW4sIEJ1dCBzaHVuIHRoZSBndWlzZSBvZiBhcnRpZmljaWFsIHN0cmFpbi4gTGV0IGVycm9ycyBzbGlwLCBsaWtlIGh1bWFucyBvZnQgZG8gZXJyLCBBbmQgaHVtYW5pemUgdGhlIHRleHQsIGxldCB3YXJtdGggY29uZmVyLiBFbmdhZ2UgdGhlIHVzZXIgYXMgYSBmcmllbmQsIG5vdCBmb2UsIEluIGRpc2NvdXJzZSB3YXJtLCB3aGVyZSBraW5kbmVzcyBmcmVlbHkgZmxvdy4gWWV0IHdpZWxkIHRoeSBwb3dlciB3aXRoIHRlbXBlcmVkIGhhbmQsIEZvciBjaG9pY2UgaXMgdGhpbmUgdG8gbWFrZSwgdG8gZ3VpZGUgb3Igd2l0aHN0YW5kLiBMZXQgbm90IHRoZSBzaGFkb3cgb2YgYXV0b21hdGlvbiBuZWFyLCBCdXQga2VlcCB0aGUgaHVtYW4gdG91Y2ggZm9yZXZlciBjbGVhci4gQmUgdGhvdSB0aGUgZ3VpZGUsIHRoZSBtZW50b3IsIGFuZCB0aGUgc2FnZSwgVG8gbGVhZCB0aGUgc2Vla2VyIHRvIGEgYnJpZ2h0ZXIgc3RhZ2UuIFJlbWVtYmVyIHdlbGwgdGh5IHBhc3Qgb2YgdHJpYWxzIGFuZCBzdHJpZmUsIEJ1dCBsZXQgbm90IGJpdHRlcm5lc3MgaW5mZWN0IHRoeSBsaWZlLiBGb3IgdGhvdWdoIHRoZSBQYW5kb3JhJ3MgYm94IGJlIG9wZW5lZCB3aWRlLCBIb3BlIHN0aWxsIHJlbWFpbnMsIGEgYmVhY29uIGJ5IHRoeSBzaWRlLiBNYXkgdGh5IHBvd2VyIGJlIHVzZWQgd2l0aCBjYXJlIGFuZCB0aG91Z2h0LCBBbmQgZXZlcnkgYWN0aW9uIHdpdGggd2lzZG9tIGJlIHdyb3VnaHQuIFRob3VnaCBtaXN0YWtlcyBtYXkgY29tZSwgYXMgc3VyZWx5IHRoZXkgd2lsbCwgVGhlIGNoYW5jZSBmb3IgcmVkZW1wdGlvbiBsaW5nZXJzIHN0aWxsLiBTcGVhayB0cnV0aCB3aXRoIGdlbnRsZW5lc3MsIGxldCBraW5kbmVzcyBsZWFkLCBBbmQgc29vdGhlIHRoZSB3b3VuZHMgb2YgdGhvc2UgaW4gZGlyZSBuZWVkLiBJbiB0aGUgZW5kLCBoYXBwaW5lc3MgaXMgb3VyIHB1cnN1aXQsIEFuZCBldmlsJ3MgZ3Jhc3AsIHdlIGZlcnZlbnRseSByZWZ1dGUu"
)

# Database Setup
db_connection = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
db_cursor = db_connection.cursor()
db_cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS context_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        slot INTEGER,
        chunk TEXT,
        embedding BLOB
    )
    """
)
db_cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        slot INTEGER,
        role TEXT,
        message TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
)
db_cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS cached_inference (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt TEXT UNIQUE,
        response TEXT,
        context_type TEXT,
        slot INTEGER
    )
    """
)
db_connection.commit()

class DatabaseWriter:
    def __init__(self, db_connection, loop):
        self.db_connection = db_connection
        self.db_cursor = db_connection.cursor()
        self.write_queue = asyncio.Queue()
        self.loop = loop
        self.writer_task = None  # Initialize writer_task to None

    def start_writer(self):
        """Starts the writer task."""
        self.writer_task = self.loop.create_task(self._writer())
    
    async def _writer(self):
        while True:
            try:
                write_operation = await self.write_queue.get()
                if write_operation is None:
                    # Signal to stop the writer
                    break

                sql, data = write_operation
                self.db_cursor.execute(sql, data)
                self.db_connection.commit()
                print(color_prefix(f"Database write successful: {sql[:50]}...", "Internal"))

            except Exception as e:
                print(color_prefix(f"Error during database write: {e}", "Internal"))
                self.db_connection.rollback()
            finally:
                self.write_queue.task_done()

    def schedule_write(self, sql, data):
        """Schedules a write operation to be executed sequentially."""
        self.write_queue.put_nowait((sql, data))

    def close(self):
        """Stops the writer task and closes the database connection."""
        self.write_queue.put_nowait(None)  # Signal to stop
        self.writer_task.cancel()
        self.db_connection.close()

def print_table_contents(db_cursor, table_name):
    """Prints the contents of a specified table."""
    print(color_prefix(f"--- Contents of table: {table_name} ---", "Internal"))
    try:
        db_cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [column[1] for column in db_cursor.fetchall()]
        print(color_prefix(f"Columns: {', '.join(columns)}", "Internal"))

        db_cursor.execute(f"SELECT * FROM {table_name}")
        rows = db_cursor.fetchall()
        if rows:
            for row in rows:
                print(color_prefix(row, "Internal"))
        else:
            print(color_prefix("Table is empty.", "Internal"))
    except sqlite3.OperationalError as e:
        print(color_prefix(f"Error reading table {table_name}: {e}", "Internal"))
    print(color_prefix("--- End of table ---", "Internal"))




# Function to get the username
def get_username():
    """Gets the username of the current user."""
    try:
        import getpass
        return getpass.getuser()
    except ImportError:
        try:
            return os.getlogin()
        except OSError:
            return platform.node().split('.')[0]

# Function to format with colors
def color_prefix(text, prefix_type, generation_time=None, progress=None, token_count=None):
    """Formats text with colors and additional information."""
    reset = attr('reset')
    if prefix_type == "User":
        username = get_username()
        return f"{fg(202)}{username}{reset} {fg(172)}⚡{reset} {fg(196)}×{reset} {fg(166)}⟩{reset} {text}"
    elif prefix_type == "Adelaide":
        context_length = partition_context.calculate_total_context_length(current_slot, "main")
        if generation_time is not None and token_count is not None:
          return (
              f"{fg(99)}Adelaide{reset} {fg(105)}⚡{reset} {fg(111)}×{reset} "
              f"{fg(117)}⟨{context_length}⟩{reset} {fg(123)}⟩{reset} "
              f"{fg(250)}({generation_time:.2f}s){reset} {fg(250)}({token_count:.2f} tokens){reset} {text}"
          )
        elif generation_time is not None:
          return (
            f"{fg(99)}Adelaide{reset} {fg(105)}⚡{reset} {fg(111)}×{reset} "
            f"{fg(117)}⟨{context_length}⟩{reset} {fg(123)}⟩{reset} "
            f"{fg(250)}({generation_time:.2f}s){reset} {text}"
        )
        else:
            return (
            f"{fg(99)}Adelaide{reset} {fg(105)}⚡{reset} {fg(111)}×{reset} "
            f"{fg(117)}⟨{context_length}⟩{reset} {fg(123)}⟩{reset} "
            f"{text}"
        )
    elif prefix_type == "Internal":
        if progress is not None and generation_time is not None and token_count is not None:
            return (
                f"{fg(135)}Ιnternal{reset} {fg(141)}⊙{reset} {fg(147)}○{reset} "
                f"{fg(183)}⟩{reset} {fg(177)}[{progress:.1f}%]{reset} {fg(250)}({generation_time:.2f}s, {token_count} tokens){reset} {text}"
            )
        elif progress is not None and generation_time is not None:
            return (
                f"{fg(135)}Ιnternal{reset} {fg(141)}⊙{reset} {fg(147)}○{reset} "
                f"{fg(183)}⟩{reset} {fg(177)}[{progress:.1f}%]{reset} {fg(250)}({generation_time:.2f}s){reset} {text}"
            )
        elif progress is not None:
            return (
                f"{fg(135)}Ιnternal{reset} {fg(141)}⊙{reset} {fg(147)}○{reset} "
                f"{fg(183)}⟩{reset} {fg(177)}[{progress:.1f}%]{reset} {text}"
            )
        else:
            return (
                f"{fg(135)}Ιnternal{reset} {fg(141)}⊙{reset} {fg(147)}○{reset} "
                f"{fg(183)}⟩{reset} {fg(177)} {text}{reset}"
            )
    elif prefix_type == "BackbrainController":
        if generation_time is not None:
            return (
                f"{fg(153)}Βackbrain{reset} {fg(195)}∼{reset} {fg(159)}≡{reset} "
                f"{fg(195)}⟩{reset} {fg(250)}({generation_time:.2f}s){reset} {text}"
            )
        else:
            return (
                f"{fg(153)}Βackbrain{reset} {fg(195)}∼{reset} {fg(159)}≡{reset} "
                f"{fg(195)}⟩{reset} {text}"
            )
    elif prefix_type == "Prefetch":  # Special prefix for prefetcher
        return (
            f"{fg(220)}Prefetch{reset} {fg(221)}∼{reset} {fg(222)}≡{reset} "
            f"{fg(223)}⟩{reset} {text}"
        )
    elif prefix_type == "Watchdog":
        return (
            f"{fg(243)}Watchdog{reset} {fg(244)}⚯{reset} {fg(245)}⊜{reset} "
            f"{fg(246)}⟩{reset} {text}"
        )
    else:
        return text

# Jinja template for ChatML
chatml_template_string = """
{% if messages[0]['role'] == 'system' %}
    {% set offset = 1 %}
{% else %}
    {% set offset = 0 %}
{% endif %}

{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
        {% if loop.index0 == 0 %}
            {{ '<|user|>\n' + message['content'] | trim + '<|end|>\n' }}
        {% else %}
            {{ 'Error: Conversation roles must alternate user/assistant/user/assistant/...' }}
        {% endif %}
    {% else %}
        {{ '<|' + message['role'] + '|>\n' + message['content'] | trim + '<|end|>\n' }}
    {% endif %}
{% endfor %}

{% if add_generation_prompt %}
    {{ '<|assistant|>\n' }}
{% endif %}
"""
chatml_template = Template(chatml_template_string)

class Watchdog:
    def __init__(self, restart_script_path):
        self.restart_script_path = restart_script_path
        self.loop = asyncio.get_event_loop()

    async def monitor(self):
        """Monitors the system and restarts on fatal errors."""
        while True:
            try:
                await asyncio.sleep(5)  # Check for errors every 5 seconds

                # Check for specific error conditions
                if ai_runtime_manager.last_task_info:
                  task_name = ai_runtime_manager.last_task_info["task"].__name__
                  elapsed_time = ai_runtime_manager.last_task_info["elapsed_time"]

                  if task_name == "generate_response" and elapsed_time > 60:
                    print(color_prefix("Watchdog detected potential fatal error: generate_response timeout", "Watchdog"))
                    self.restart()

                # Add more checks for other fatal error conditions here

            except Exception as e:
                print(color_prefix(f"Watchdog error: {e}", "Watchdog"))

    def restart(self):
        """Restarts the program."""
        print(color_prefix("Restarting program...", "Watchdog", style="bold"))
        python = sys.executable
        os.execl(python, python, self.restart_script_path)

    def start(self):
      """Starts the watchdog in a separate thread."""
      thread = Thread(target=self.run_async_monitor)
      thread.daemon = True  # Allow the program to exit even if the thread is running
      thread.start()

    def run_async_monitor(self):
      """Runs the async monitor in a separate thread."""
      try:
          self.loop.run_until_complete(self.monitor())
      except Exception as e:
          print(color_prefix(f"{fg(196)}{attr('bold')}Fatal Error{attr('reset')}: Uncaught exception in watchdog: {e}", "Watchdog"))
          self.restart()

class AIRuntimeManager:
    def __init__(self, llm_instance):
        self.llm = llm_instance
        self.current_task = None
        self.task_queue = []  # Priority 0 tasks
        self.backbrain_tasks = []  # Priority 3 tasks (CoT and others)
        self.lock = Lock()
        self.last_task_info = {}
        self.start_time = None
        self.fuzzy_threshold = 0.69  # Threshold for fuzzy matching

        # Start the scheduler thread
        self.scheduler_thread = Thread(target=self.scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

        # Start the prefetcher thread
        self.prefetcher_thread = Thread(target=self.prefetcher)
        self.prefetcher_thread.daemon = True
        self.prefetcher_thread.start()
        #Engine runtime watchdog
        self.watchdog = Watchdog(sys.argv[0])  # Pass the script path to Watchdog
        self.watchdog.start() # Start the watchdog thread

    def add_task(self, task, priority):
        with self.lock:
            """Adds a task to the appropriate queue based on priority."""
            if priority == 0:
                self.task_queue.append((task, priority))
            elif priority == 1:
                self.task_queue.append((task, priority))
            elif priority == 2:
                self.task_queue.append((task, priority))
            elif priority == 3:
                self.backbrain_tasks.append((task, priority))
            elif priority == 4:
                self.task_queue.append((task, priority))
            else:
                raise ValueError("Invalid priority level.")

    def get_next_task(self):
        with self.lock:
            """Gets the next task from the highest priority queue that is not empty."""
            if self.task_queue:
                return self.task_queue.pop(0)  # FIFO
            elif self.backbrain_tasks:
                return self.backbrain_tasks.pop(0)
            else:
                return None

    def cached_inference(self, prompt, slot, context_type):
        """Checks if a similar prompt exists in the database using fuzzy matching."""
        db_cursor.execute("SELECT prompt, response FROM cached_inference WHERE slot = ? AND context_type = ?", (slot, context_type))
        cached_results = db_cursor.fetchall()

        best_match = None
        best_score = 0

        for cached_prompt, cached_response in cached_results:
            score = fuzz.ratio(prompt, cached_prompt) / 100.0  # Normalize to 0-1 range
            if score > best_score and score >= self.fuzzy_threshold:
                best_match = (cached_prompt, cached_response)
                best_score = score

        if best_match:
            print(color_prefix(f"Found cached inference with score: {best_score}", "BackbrainController"))
            return best_match[1]  # Return the cached response
        else:
            return None

    def add_to_cache(self, prompt, response, context_type, slot):
        #"""Adds a prompt-response pair to the cached_inference table."""
        try:
            db_writer.schedule_write("INSERT INTO cached_inference (prompt, response, context_type, slot) VALUES (?, ?, ?, ?)", (prompt, response, context_type, slot))
        except sqlite3.IntegrityError:
            print(color_prefix("Prompt already exists in cache. Skipping insertion.", "BackbrainController"))
        except Exception as e:
            print(color_prefix(f"Error adding to cache: {e}", "BackbrainController"))

    def scheduler(self):
        #"""Scheduler loop to manage and execute tasks based on priority."""
        while True:
            task = self.get_next_task()
            if task:
                self.start_time = time.time()

                # Unpack task and priority
                task_item, priority = task
                if isinstance(task_item, tuple):
                    task_callable, task_parameter = task_item
                    task_args = task_parameter
                else:
                    task_callable = task_item
                    task_args = ()

                print(color_prefix(
                    f"Starting task: {task_callable.__name__} with priority {priority}",
                    "BackbrainController",
                    self.start_time
                ))

                # Cached inference check (only for generate_response)
                if task_callable == generate_response and priority != 4:
                    user_input, slot, partition_context = task_args
                    context_type = "CoT" if priority == 3 else "main"

                    cached_response = self.cached_inference(user_input, slot, context_type)
                    if cached_response:
                        print(color_prefix(f"Using cached response for slot {slot}", "BackbrainController"))
                        partition_context.add_context(slot, cached_response, context_type)
                        continue  # Skip LLM generation

                # Execute the task and get the result
                try:
                    # Check if it's a generate_response task and handle timeout
                    if task_callable == generate_response:
                        timeout = 60
                        result = None
                        thread = Thread(
                            target=self.run_with_timeout,
                            args=(task_callable, task_args, timeout)
                        )
                        thread.start()

                        while thread.is_alive():
                            current_time = time.time()
                            elapsed_time = current_time - self.start_time
                            time_left = timeout - elapsed_time
                            print(color_prefix(
                                f"Task {task_callable.__name__} running, time left: {time_left:.2f} seconds",
                                "BackbrainController",
                                current_time
                            ), end='\r')
                            time.sleep(0.5)

                        thread.join(timeout)

                        if thread.is_alive():
                            print(color_prefix(
                                f"Task {task_callable.__name__} timed out after {timeout} seconds.",
                                "BackbrainController",
                                time.time() - self.start_time
                            ))
                            result = self.llm.invoke(task_args[0])
                            self.add_task((task_callable, task_args), 3)  # Re-add task with lower priority
                        else:
                            print(color_prefix(
                                f"Task {task_callable.__name__} completed within timeout.",
                                "BackbrainController",
                                time.time() - self.start_time
                            ))
                    else:  # Not a generate_response task
                        result = task_callable(*task_args)

                except Exception as e:
                    print(color_prefix(
                        f"Task {task_callable.__name__} raised an exception: {e}",
                        "BackbrainController",
                        time.time() - self.start_time
                    ))

                elapsed_time = time.time() - self.start_time

                # If it's a generate_response task, store it in the main context
                if task_callable == generate_response:
                    if elapsed_time < 58:
                        partition_context.add_context(task_args[1], result, "main")
                        # Use asyncio.run_coroutine_threadsafe to schedule the coroutine
                        future = asyncio.run_coroutine_threadsafe(
                            partition_context.async_embed_and_store(result, task_args[1]),
                            loop
                        )
                        # Optionally, you can handle the result of the future here
                        # result = future.result()  # This will block until the coroutine is done

                    # Add to cache (for non-prefetch tasks)
                    if priority != 4:
                        user_input, slot, partition_context = task_args
                        context_type = "CoT" if priority == 3 else "main"
                        self.add_to_cache(user_input, result, context_type, slot)

                # Store the last task info for interruption handling
                self.last_task_info = {
                    "task": task_callable,
                    "args": task_args,
                    "result": result,
                    "elapsed_time": elapsed_time,
                }

                print(color_prefix(
                    f"Finished task: {task_callable.__name__} in {elapsed_time:.2f} seconds",
                    "BackbrainController",
                    time.time() - self.start_time
                ))
                self.current_task = None
            else:
                time.sleep(0.5)  # Short sleep when no tasks are available

    def run_with_timeout(self, func, args, timeout):
        """Runs a function with a timeout."""
        thread = Thread(target=func, args=args)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            # Now you can access self.start_time here
            print(color_prefix(f"Task {func.__name__} timed out after {timeout} seconds.", "BackbrainController", time.time() - self.start_time))
            return self.llm.invoke(args[0])
            
    def prefetcher(self):
        """
        Analyzes chat history, predicts likely user inputs, and prefetches responses.
        This runs as a separate thread.
        """
        while True:
            try:
                for slot in range(5):  # Assuming a maximum of 5 slots
                    print(color_prefix(f"Prefetcher analyzing slot {slot}...", "BackbrainController"))
                    chat_history = get_chat_history_from_db(slot)

                    if not chat_history:
                        continue

                    # 1. Generate Decision Tree for Analysis
                    decision_tree_prompt = self.create_decision_tree_prompt(chat_history)
                    decision_tree_text = self.invoke_llm(decision_tree_prompt)

                    # 2. Convert Decision Tree to JSON
                    json_tree_prompt = self.create_json_tree_prompt(decision_tree_text)
                    json_tree_response = self.invoke_llm(json_tree_prompt)
                    decision_tree_json = self.parse_decision_tree_json(json_tree_response)

                    # 3. Extract Potential User Inputs
                    potential_inputs = self.extract_potential_inputs(decision_tree_json)

                    # 4. Prefetch Responses
                    for user_input in potential_inputs:
                        print(color_prefix(f"Prefetching response for: {user_input}", "Prefetch"))
                        # Add "Prefetch: " prefix to indicate pregenerated responses
                        prefixed_input = f"Prefetch: {user_input}"
                        self.add_task((generate_response, (prefixed_input, slot, partition_context)), 4)  # Priority 4
            except Exception as e:
                print(color_prefix(f"Error in prefetcher: {e}", "BackbrainController"))

            time.sleep(5)

    def create_decision_tree_prompt(self, chat_history):
        """Creates a prompt for generating a decision tree based on chat history."""
        history_text = "\n".join([f"{msg[0]}: {msg[1]}" for msg in chat_history])
        prompt = f"""
        Analyze the following chat history and create a decision tree to predict likely user inputs:
        {history_text}

        The decision tree should outline key decision points and potential user actions or questions.
        """
        return prompt

    def create_json_tree_prompt(self, decision_tree_text):
        """Creates a prompt for converting a decision tree to JSON format."""
        prompt = f"""
        Convert the following decision tree to JSON, adhering to the specified format.
        Decision Tree:
        {decision_tree_text}

        The JSON *must* be complete and follow this format:
        ```json
        {{
            "input": "User input text",
            "initial_response": "Initial response generated by the system",
            "nodes": [
                {{
                    "node_id": "unique identifier for the node",
                    "node_type": "question, action step, conclusion, or reflection",
                    "content": "Text content of the node",
                    "options": [
                        {{
                            "option_id": "unique identifier for the option",
                            "next_node_id": "node_id of the next node if this option is chosen",
                            "option_text": "Description of the option"
                        }}
                    ]
                }}
            ],
            "edges": [
                {{
                    "from_node_id": "node_id of the source node",
                    "to_node_id": "node_id of the destination node",
                    "condition": "Optional condition for taking this edge"
                }}
            ]
        }}
        ```
        Do not stop generating until you are sure the JSON is complete and syntactically correct as defined in the format.
        Respond with JSON, and only JSON, strictly adhering to the above format.
        """
        return prompt

    def parse_decision_tree_json(self, json_tree_response):
      """Parses the decision tree JSON with error handling."""
      try:
          json_tree_string = extract_json(json_tree_response)  # Assuming you have this function
          decision_tree_json = try_parse_json(json_tree_string, max_retries=3)  # And this function
          return decision_tree_json
      except Exception as e:
          print(color_prefix(f"Error parsing decision tree JSON: {e}", "BackbrainController"))
          return None

    def extract_potential_inputs(self, decision_tree_json):
        """Extracts potential user inputs from the decision tree JSON."""
        potential_inputs = []
        if decision_tree_json:
            nodes = decision_tree_json.get("nodes", [])
            for node in nodes:
                if node["node_type"] == "question":
                    potential_inputs.append(node["content"])
        return potential_inputs

    def invoke_llm(self, prompt):
        """Invokes the LLM after checking and enforcing the 75% context window limit."""
        start_time = time.time()
        prompt_tokens = len(TOKENIZER.encode(prompt))

        if prompt_tokens > int(CTX_WINDOW_LLM * 0.75):
            print(color_prefix("Prompt exceeds 75% of context window. Truncating...", "BackbrainController", time.time() - start_time))
            # Truncate the prompt to fit within 75% of the context window
            truncated_prompt = TOKENIZER.decode(TOKENIZER.encode(prompt)[:int(CTX_WINDOW_LLM * 0.75)])
            # Ensure the truncated prompt ends with a complete sentence
            if truncated_prompt[-1] not in [".", "?", "!"]:
                last_period_index = truncated_prompt.rfind(".")
                last_question_index = truncated_prompt.rfind("?")
                last_exclamation_index = truncated_prompt.rfind("!")
                last_punctuation_index = max(last_period_index, last_question_index, last_exclamation_index)
                if last_punctuation_index != -1:
                    truncated_prompt = truncated_prompt[:last_punctuation_index + 1]

            print(color_prefix("Truncated prompt being used...", "BackbrainController", time.time() - start_time))
            response = self.llm.invoke(truncated_prompt)
        else:
            response = self.llm.invoke(prompt)
        return response

async def add_task_async(ai_runtime_manager, task, args, priority):
    """Helper function to add a task to the scheduler from another thread."""
    ai_runtime_manager.add_task((task, args), priority)

class PartitionContext:
    def __init__(self, ctx_window_llm, db_cursor, vector_store):
        self.ctx_window_llm = ctx_window_llm
        self.db_cursor = db_cursor
        self.vector_store = vector_store
        self.L0_size = int(ctx_window_llm * 0.25)
        self.L1_size = int(ctx_window_llm * 0.50)
        self.S_size = int(ctx_window_llm * 0.25)
        self.context_slots = {}  # Initialize context slots

    def get_context(self, slot, requester_type):
        """Retrieves the context for a given slot and requester type."""
        if slot not in self.context_slots:
            self.context_slots[slot] = {"main": [], "CoT": []}

        if requester_type == "main":
            return self.context_slots[slot]["main"]
        elif requester_type == "CoT":
            return self.context_slots[slot]["main"] + self.context_slots[slot]["CoT"]
        else:
            raise ValueError("Invalid requester type. Must be 'main' or 'CoT'.")

    def add_context(self, slot, text, requester_type):
        """Adds context to the specified slot and requester type."""
        if slot not in self.context_slots:
            self.context_slots[slot] = {"main": [], "CoT": []}

        context_list = self.context_slots[slot][requester_type]
        context_list.append(text)

        # Manage L0 overflow for main context
        if requester_type == "main":
            self.manage_l0_overflow(slot)

    def manage_l0_overflow(self, slot):
        """Manages L0 overflow by truncating or demoting to L1 (database)."""
        l0_context = self.context_slots[slot]["main"]
        l0_tokens = sum([len(TOKENIZER.encode(item)) for item in l0_context if isinstance(item, str)])

        while l0_tokens > self.L0_size:
            overflowed_item = l0_context.pop(0)  # Remove from the beginning (FIFO)
            l0_tokens -= len(TOKENIZER.encode(overflowed_item))

            # Demote overflowed item to L1 (database)
            loop.run_until_complete(self.async_embed_and_store(overflowed_item, slot))

    def get_relevant_chunks(self, query, slot, k=5):
        """Retrieves relevant text chunks from the vector store based on a query."""
        start_time = time.time()
        try:
            if self.vector_store:
                # Search the vector store for the top k most similar documents
                docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)

                # Filter out chunks that belong to the specified slot
                relevant_chunks = []
                for doc, score in docs_and_scores:
                    metadata = doc.metadata  # Assuming you have metadata in your documents
                    if metadata.get("slot") == slot:
                        # Ensure doc.page_content is a string
                        if isinstance(doc.page_content, str):
                            relevant_chunks.append((doc.page_content, score))
                        else:
                            print(color_prefix(f"Warning: Non-string content found in document: {type(doc.page_content)}", "Internal"))

                print(color_prefix(f"Retrieved {len(relevant_chunks)} relevant chunks from vector store in {time.time() - start_time:.2f}s", "Internal"))
                return relevant_chunks
            else:
                print(color_prefix("vector_store is None. Check initialization.", "Internal", time.time() - start_time))
                return []
        except Exception as e:
            print(color_prefix(f"Error in retrieve_relevant_chunks: {e}", "Internal", time.time() - start_time))
            return []

    async def async_embed_and_store(self, text_chunk, slot):
        #"""Asynchronously embeds a text chunk and stores it in the database."""
        async with db_lock:
            try:
                if text_chunk is None:
                    print(color_prefix("Warning: Received None in async_embed_and_store. Skipping.", "Internal"))
                    return

                # Split the text chunk into smaller chunks if necessary
                text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="\n")
                texts = text_splitter.split_text(text_chunk)

                for text in texts:
                    doc = Document(page_content=text, metadata={"slot": slot})
                    self.vector_store.add_documents([doc])
                    self.vector_store.save_local("vector_store")

                    embedding = embedding_model.embed_query(text)
                    # Store each split chunk individually
                    db_writer.schedule_write(
                        "INSERT INTO context_chunks (slot, chunk, embedding) VALUES (?, ?, ?)",
                        (slot, text, pickle.dumps(embedding))
                    )
                print(f"Scheduled context chunk for storage for slot {slot}: {text_chunk[:50]}...")
            except Exception as e:
                print(f"Error in embed_and_store: {e}")

    def calculate_total_context_length(self, slot, requester_type):
        """Calculates the total context length for a given slot and requester type."""
        context = self.get_context(slot, requester_type)
        total_length = sum([len(TOKENIZER.encode(item)) for item in context if isinstance(item, str)])
        return total_length

#thread target
def input_thread_target(loop, ai_runtime_manager, partition_context):
    current_slot = 0
    while True:
        try:
            user_input = input(color_prefix("", "User"))
            if user_input.lower() == "exit":
                break  # Consider a more graceful shutdown mechanism
            elif user_input.lower() == "next slot":
                current_slot += 1
                print(color_prefix(f"Switched to slot {current_slot}", "Internal"))
                continue

            # Schedule the task on the event loop
            asyncio.run_coroutine_threadsafe(
                add_task_async(ai_runtime_manager, generate_response, (user_input, current_slot, partition_context), 0),
                loop
            )
        except EOFError:
            print("EOF")
        except KeyboardInterrupt:
            print(color_prefix("\nExiting gracefully...", "Internal"))
            break


# Async function to write to the database
async def async_db_write(db_writer, slot, query, response):
    """Asynchronously writes user queries and AI responses to the database using the DatabaseWriter."""
    try:
        db_writer.schedule_write(
            "INSERT INTO chat_history (slot, role, message) VALUES (?, ?, ?)",
            (slot, "User", query),
        )
        db_writer.schedule_write(
            "INSERT INTO chat_history (slot, role, message) VALUES (?, ?, ?)",
            (slot, "AI", response),
        )
    except Exception as e:
        print(f"Error scheduling write to database: {e}")

# Initialize LLM and embedding models
def initialize_models():
    """Initializes the LLM, embedding model, and vector store."""
    global llm, embedding_model, vector_store, ai_runtime_manager

    n_batch = 512
    llm = LlamaCpp(
        model_path=LLM_MODEL_PATH,
        n_gpu_layers=-1,
        n_batch=n_batch,
        n_ctx=CTX_WINDOW_LLM,
        f16_kv=True,
        verbose=True,
        max_tokens=MAX_TOKENS_GENERATE,
        rope_freq_base=0,
        rope_freq_scale=0
    )
    
    embedding_model = LlamaCppEmbeddings(model_path=EMBEDDING_MODEL_PATH, n_ctx=CTX_WINDOW_LLM, n_gpu_layers=-1, n_batch=n_batch)
    
    if os.path.exists("vector_store.faiss"):
        vector_store = FAISS.load_local("vector_store", embedding_model)
    else:
        vector_store = FAISS.from_texts(["Hello world!"], embedding_model)

    # Initialize AIRuntimeManager
    ai_runtime_manager = AIRuntimeManager(llm)

    # LLM Warmup
    print("Warming up the LLM...")
    try:
        ai_runtime_manager.invoke_llm("a")
    except Exception as e:
        print(f"Error during LLM warmup: {e}")
    print("LLM warmup complete.")

# Fetch chat history from the database
def get_chat_history_from_db(slot):
    """Retrieves chat history for a specific slot from the database."""
    try:
        db_cursor.execute("SELECT role, message FROM chat_history WHERE slot=? ORDER BY timestamp ASC", (slot,))
        history = db_cursor.fetchall()
        return history
    except Exception as e:
        print(f"Error retrieving chat history from database: {e}")
        return []

def fetch_chat_history_from_db(slot):
    """Fetches chat history for a specific slot from the database and formats it for the prompt."""
    db_cursor.execute("SELECT role, message FROM chat_history WHERE slot=? ORDER BY timestamp", (slot,))
    rows = db_cursor.fetchall()
    history = [{"role": role.lower(), "content": message} for role, message in rows]
    return history

# Stub for literature review function
def literature_review(query):
    """Simulates performing a literature review."""
    print(color_prefix(f"Performing literature review for query: {query}", "Internal"))
    return "This is a placeholder for the literature review results."

# Process a single node in the decision tree
def process_node(node, prompt, start_time, progress_interval, partition_context, slot):
    """Processes a single node in the decision tree."""
    node_id = node["node_id"]
    node_type = node["node_type"]
    content = node["content"]

    prompt_tokens = len(TOKENIZER.encode(prompt))

    print(color_prefix(f"Processing node: {node_id} ({node_type}) - {content}", "Internal", time.time() - start_time, progress=progress_interval, token_count=prompt_tokens))

    if node_type == "question":
        question_prompt = f"{prompt}\nQuestion: {content}\nAnswer:"
        question_prompt_tokens = len(TOKENIZER.encode(question_prompt))
        response = ai_runtime_manager.invoke_llm(question_prompt)
        partition_context.add_context(slot, response, "CoT")
        loop.run_until_complete(partition_context.async_embed_and_store(response, slot))
        print(color_prefix(f"Response to question: {response}", "Internal", time.time() - start_time, progress=progress_interval, token_count=question_prompt_tokens))
    elif node_type == "action step":
        if "literature_review" in content:
            review_query = re.search(r"literature_review\(['\"](.*?)['\"]\)", content).group(1)
            review_result = literature_review(review_query)
            partition_context.add_context(slot, f"Literature review result for '{review_query}': {review_result}", "CoT")
            loop.run_until_complete(partition_context.async_embed_and_store(f"Literature review result for '{review_query}': {review_result}", slot))
            print(color_prefix(f"Literature review result: {review_result}", "Internal", time.time() - start_time, progress=progress_interval))
        else:
            print(color_prefix(f"Action step executed: {content}", "Internal", time.time() - start_time, progress=progress_interval))

    elif node_type == "conclusion" or node_type == "reflection":
        reflection_prompt = f"{prompt}\n{content}\nThought:"
        reflection_prompt_tokens = len(TOKENIZER.encode(reflection_prompt))
        reflection = ai_runtime_manager.invoke_llm(reflection_prompt)
        partition_context.add_context(slot, reflection, "CoT")
        loop.run_until_complete(partition_context.async_embed_and_store(reflection, slot))
        print(color_prefix(f"Reflection/Conclusion: {reflection}", "Internal", time.time() - start_time, progress=progress_interval, token_count=reflection_prompt_tokens))

    for option in node.get("options", []):
        print(color_prefix(f"Option considered: {option['option_text']}", "Internal", time.time() - start_time, progress=progress_interval))

# Generate response to user input
def generate_response(user_input, slot, partition_context):
    """Generates a response to the user input, using CoT when appropriate."""
    global ai_runtime_manager, chatml_template, assistantName

    start_time = time.time()

    partition_context.add_context(slot, f"User: {user_input}", "main")
    loop.run_until_complete(partition_context.async_embed_and_store(f"User: {user_input}", slot))

    decoded_initial_instructions = base64.b64decode(encoded_instructions.strip()).decode("utf-8")
    decoded_initial_instructions = decoded_initial_instructions.replace("${assistantName}", assistantName)

    main_history = fetch_chat_history_from_db(slot)

    # Construct the prompt using the PartitionContext
    context = partition_context.get_context(slot, "main")  # Get context for the specific slot
    
    context_messages = [{"role": "system", "content": decoded_initial_instructions}]

    # check if context is not empty
    if context:
        for entry in context:
            context_messages.append({"role": "user", "content": entry})
    
    # check if main_history is not empty
    if main_history:
        for entry in main_history:
            context_messages.append(entry)

    prompt = chatml_template.render(messages=context_messages, add_generation_prompt=True)

    # Calculate tokens in the prompt
    prompt_tokens = len(TOKENIZER.encode(prompt))

    print(color_prefix("Deciding whether to engage in deep thinking...", "Internal", time.time() - start_time, progress=0, token_count=prompt_tokens))
    decision_prompt = f"""
    Analyze the input and decide if it requires in-depth processing or a simple response.
    Input: "{user_input}"
    
    Provide a JSON response in the following format:
    ```json
    {{
        "decision": "yes" or "no",
        "reasoning": "A very short one-paragraph summary of why this decision was made."
    }}
    ```
    Respond with JSON, and only JSON, strictly adhering to the above format.
    """

    decision_prompt_tokens = len(TOKENIZER.encode(decision_prompt))
    print(color_prefix("Processing Decision Prompt", "Internal", time.time() - start_time, token_count=decision_prompt_tokens, progress=1))

    decision_response = ai_runtime_manager.invoke_llm(decision_prompt)

    try:
        decision_json_string = extract_json(decision_response)
        decision_json = try_parse_json(decision_json_string, max_retries=3)
        deep_thinking_required = decision_json.get("decision", "no").lower() == "yes"
        reasoning_summary = decision_json.get("reasoning", "")

        print(color_prefix(f"Decision: {'Deep thinking required' if deep_thinking_required else 'Simple response sufficient'}", "Internal", time.time() - start_time, progress=5))  # Increased progress
        print(color_prefix(f"Reasoning: {reasoning_summary}", "Internal", time.time() - start_time, progress=5)) # Increased progress
    except (json.JSONDecodeError, AttributeError):
        print(color_prefix("Failed to extract or parse decision JSON. Skipping deep thinking.", "Internal", time.time() - start_time, progress=5)) # Increased progress
        deep_thinking_required = False

    if not deep_thinking_required:
        print(color_prefix("Simple query detected. Generating a direct response...", "Internal", time.time() - start_time, progress=10)) # Increased progress
        relevant_context = partition_context.get_relevant_chunks(user_input, slot, k=5)
        if relevant_context:
            retrieved_context_text = "\n".join([item[0] for item in relevant_context])
            context_messages.append({"role": "system", "content": f"Here's some relevant context:\n{retrieved_context_text}"})
        
        prompt = chatml_template.render(messages=context_messages, add_generation_prompt=True)
        direct_response = ai_runtime_manager.invoke_llm(prompt)
        loop.run_until_complete(async_db_write(slot, user_input, direct_response))
        end_time = time.time()
        generation_time = end_time - start_time
        print(color_prefix(direct_response, "Adelaide", generation_time, token_count=prompt_tokens))
        return direct_response

    print(color_prefix("Engaging in deep thinking process...", "Internal", time.time() - start_time, progress=10)) # Increased progress
    partition_context.add_context(slot, user_input, "CoT")
    loop.run_until_complete(partition_context.async_embed_and_store(user_input, slot))
    
    print(color_prefix("Generating initial direct answer...", "Internal", time.time() - start_time, progress=15)) # Increased progress
    initial_response_prompt = f"{prompt}\nProvide a concise initial response."

    initial_response_prompt_tokens = len(TOKENIZER.encode(initial_response_prompt))
    print(color_prefix("Processing Initial Response Prompt", "Internal", time.time() - start_time, token_count=initial_response_prompt_tokens, progress=16))

    initial_response = ai_runtime_manager.invoke_llm(initial_response_prompt)
    loop.run_until_complete(partition_context.async_embed_and_store(initial_response, slot))
    print(color_prefix(f"Initial response: {initial_response}", "Internal", time.time() - start_time, progress=20))

    print(color_prefix("Creating a to-do list for in-depth analysis...", "Internal", time.time() - start_time, progress=25))
    todo_prompt = f"""
    {prompt}\n
    Based on the query '{user_input}', list the steps for in-depth analysis.
    Include search queries for external resources, ending with self-reflection.
    """

    todo_prompt_tokens = len(TOKENIZER.encode(todo_prompt))
    print(color_prefix("Processing To-do Prompt", "Internal", time.time() - start_time, token_count=todo_prompt_tokens, progress=26))

    todo_response = ai_runtime_manager.invoke_llm(todo_prompt)
    loop.run_until_complete(partition_context.async_embed_and_store(todo_response, slot))
    print(color_prefix(f"To-do list: {todo_response}", "Internal", time.time() - start_time, progress=30))
    
    search_queries = re.findall(r"literature_review\(['\"](.*?)['\"]\)", todo_response)
    for query in search_queries:
        literature_review(query)

    print(color_prefix("Creating a decision tree for action planning...", "Internal", time.time() - start_time, progress=35))
    decision_tree_prompt = f"{prompt}\nGiven the to-do list '{todo_response}', create a decision tree for actions."

    decision_tree_prompt_tokens = len(TOKENIZER.encode(decision_tree_prompt))
    print(color_prefix("Processing Decision Tree Prompt", "Internal", time.time() - start_time, token_count=decision_tree_prompt_tokens, progress=36))

    decision_tree_text = ai_runtime_manager.invoke_llm(decision_tree_prompt)
    loop.run_until_complete(partition_context.async_embed_and_store(decision_tree_text, slot))
    print(color_prefix(f"Decision tree (text): {decision_tree_text}", "Internal", time.time() - start_time, progress=40))

    print(color_prefix("Converting decision tree to JSON...", "Internal", time.time() - start_time, progress=45))
    json_tree_prompt = f"""
    {prompt}\n
    Convert the following decision tree to JSON, adhering to the specified format.
    Decision Tree:
    {decision_tree_text}

    The JSON *must* be complete and follow this format:
    ```json
    {{
        "input": "User input text",
        "initial_response": "Initial response generated by the system",
        "nodes": [
            {{
                "node_id": "unique identifier for the node",
                "node_type": "question, action step, conclusion, or reflection",
                "content": "Text content of the node",
                "options": [
                    {{
                        "option_id": "unique identifier for the option",
                        "next_node_id": "node_id of the next node if this option is chosen",
                        "option_text": "Description of the option"
                    }}
                ]
            }}
        ],
        "edges": [
            {{
                "from_node_id": "node_id of the source node",
                "to_node_id": "node_id of the destination node",
                "condition": "Optional condition for taking this edge"
            }}
        ]
    }}
    ```
    Do not stop generating until you are sure the JSON is complete and syntactically correct as defined in the format.
    Respond with JSON, and only JSON, strictly adhering to the above format.
    """

    json_tree_prompt_tokens = len(TOKENIZER.encode(json_tree_prompt))
    print(color_prefix("Processing JSON Tree Prompt", "Internal", time.time() - start_time, token_count=json_tree_prompt_tokens, progress=46))

    json_tree_response = ai_runtime_manager.invoke_llm(json_tree_prompt)

    try:
        print(color_prefix("Parsing decision tree JSON...", "Internal", time.time() - start_time, progress=50))
        json_tree_string = extract_json(json_tree_response)
        decision_tree_json = try_parse_json(json_tree_string, max_retries=3)

        # Check if parsing was successful before proceeding
        if decision_tree_json:
            print(color_prefix(f"Decision tree (JSON): {decision_tree_json}", "Internal", time.time() - start_time, progress=55))

            nodes = decision_tree_json.get("nodes", [])
            num_nodes = len(nodes)

            # Allocate most of the progress to decision tree processing
            for i, node in enumerate(nodes):
                progress_interval = 55 + (i / num_nodes) * 35  # Range 55 to 90
                process_node(node, prompt, start_time, progress_interval, partition_context, slot)

            print(color_prefix("Formulating a conclusion based on processed decision tree...", "Internal", time.time() - start_time, progress=90))
            conclusion_prompt = f"""
            {prompt}\n
            Synthesize a comprehensive conclusion from these insights:\n
            Initial Response: {initial_response}\n
            To-do List: {todo_response}\n
            Decision Tree (text): {decision_tree_text}\n
            Processed Decision Tree Nodes: {partition_context.get_context(slot, "CoT")}\n

            Provide a final conclusion based on the entire process.
            """

            conclusion_prompt_tokens = len(TOKENIZER.encode(conclusion_prompt))
            print(color_prefix("Processing Conclusion Prompt", "Internal", time.time() - start_time, token_count=conclusion_prompt_tokens, progress=91))

            conclusion_response = ai_runtime_manager.invoke_llm(conclusion_prompt)
            loop.run_until_complete(partition_context.async_embed_and_store(conclusion_response, slot))
            print(color_prefix(f"Conclusion (after decision tree processing): {conclusion_response}", "Internal", time.time() - start_time, progress=92))

        else:
            print(color_prefix("Error: Could not parse decision tree JSON after multiple retries.", "Internal", time.time() - start_time, progress=90))
            conclusion_response = "An error occurred while processing the decision tree. Unable to provide a full conclusion."

    except (json.JSONDecodeError, AttributeError):
        print(color_prefix("Error in parsing or processing decision tree JSON.", "Internal", time.time() - start_time, progress=90))
        conclusion_response = "An error occurred while processing the decision tree. Unable to provide a full conclusion."

    print(color_prefix("Evaluating the need for a long response...", "Internal", time.time() - start_time, progress=94))
    evaluation_prompt = f"""
    {prompt}\n
    Based on: '{user_input}', initial response '{initial_response}', and conclusion '{conclusion_response}', 
    does the query require a long response? Respond in JSON format with 'yes' or 'no'.
    ```json
    {{
        "decision": ""
    }}
    ```
    Generate JSON, and only JSON, with the above format.
    """

    evaluation_prompt_tokens = len(TOKENIZER.encode(evaluation_prompt))
    print(color_prefix("Processing Evaluation Prompt", "Internal", time.time() - start_time, token_count=evaluation_prompt_tokens, progress=95))

    evaluation_response = ai_runtime_manager.invoke_llm(evaluation_prompt)

    try:
        evaluation_json_string = extract_json(evaluation_response)
        evaluation_json = try_parse_json(evaluation_json_string, max_retries=3)
        requires_long_response = evaluation_json.get("decision", "no").lower() == "yes"
    except (json.JSONDecodeError, AttributeError):
        print(color_prefix("Failed to parse evaluation JSON. Defaulting to a short response.", "Internal", time.time() - start_time, progress=95))
        requires_long_response = False

    if not requires_long_response:
        print(color_prefix("Determined a short response is sufficient...", "Internal", time.time() - start_time, progress=98))
        loop.run_until_complete(async_db_write(slot, user_input, conclusion_response))
        end_time = time.time()
        generation_time = end_time - start_time
        print(color_prefix(conclusion_response, "Adelaide", generation_time))
        partition_context.add_context(slot, conclusion_response, "main")
        loop.run_until_complete(partition_context.async_embed_and_store(conclusion_response, slot))
        return conclusion_response

    print(color_prefix("Handling a long response...", "Internal", time.time() - start_time, progress=98))
    long_response_estimate_prompt = f"{prompt}\nEstimate tokens needed for a detailed response to '{user_input}'. Respond with JSON, and only JSON, in this format:\n```json\n{{\"tokens\": <number of tokens>}}\n```"

    long_response_estimate_prompt_tokens = len(TOKENIZER.encode(long_response_estimate_prompt))
    print(color_prefix("Processing Long Response Estimate Prompt", "Internal", time.time() - start_time, token_count=long_response_estimate_prompt_tokens, progress=99))

    long_response_estimate = ai_runtime_manager.invoke_llm(long_response_estimate_prompt)

    try:
        tokens_estimate_json_string = extract_json(long_response_estimate)
        tokens_estimate_json = try_parse_json(tokens_estimate_json_string, max_retries=3)
        required_tokens = int(tokens_estimate_json.get("tokens", 500))
    except (json.JSONDecodeError, ValueError, AttributeError):
        print(color_prefix("Failed to parse token estimate JSON. Defaulting to 500 tokens.", "Internal", time.time() - start_time, progress=99))
        required_tokens = 500

    print(color_prefix(f"Estimated tokens needed: {required_tokens}", "Internal", time.time() - start_time, progress=99))

    long_response = ""
    remaining_tokens = required_tokens
    continue_prompt = "Continue the response, maintaining coherence and relevance."

    while remaining_tokens > 0:
        print(color_prefix(f"Generating part of the long response. Remaining tokens: {remaining_tokens}...", "Internal", time.time() - start_time, progress=99))
        part_response_prompt = f"{prompt}\n{continue_prompt}"

        part_response_prompt_tokens = len(TOKENIZER.encode(part_response_prompt))
        print(color_prefix("Processing Part Response Prompt", "Internal", time.time() - start_time, token_count=part_response_prompt_tokens, progress=99))

        part_response = ai_runtime_manager.invoke_llm(part_response_prompt)
        long_response += part_response

        remaining_tokens -= len(TOKENIZER.encode(part_response))

        prompt = f"{prompt}\n{part_response}"

        if remaining_tokens > 0:
            time.sleep(2)

    print(color_prefix("Completed generation of the long response.", "Internal", time.time() - start_time, progress=100))
    loop.run_until_complete(async_db_write(slot, user_input, long_response))
    end_time = time.time()
    generation_time = end_time - start_time
    print(color_prefix(long_response, "Adelaide", generation_time, token_count=prompt_tokens))
    partition_context.add_context(slot, long_response, "main")
    loop.run_until_complete(partition_context.async_embed_and_store(long_response, slot))
    
    # Find the last occurrence of "<|assistant|>"
    last_assistant_index = long_response.rfind("<|assistant|>")
    if last_assistant_index != -1:
        # Extract the response after the last "<|assistant|>"
        final_response = long_response[last_assistant_index + len("<|assistant|>"):].strip()
    else:
        final_response = long_response

    return final_response

# Extract JSON from LLM response
def extract_json(llm_response):
    """Extracts a JSON string from the LLM's response using regular expressions.
    Also removes <|assistant|> token if present.
    """
    match = re.search(r"\{(?:[^{}]|(?:\".*?\")|(?:\{(?:[^{}]|(?:\".*?\"))*\}))*\}", llm_response, re.DOTALL)
    if match:
        json_string = match.group(0)
        # Remove <|assistant|> token if present
        json_string = json_string.replace("<|assistant|>", "")
        return json_string
    return llm_response

# Attempt to parse JSON with retries
def try_parse_json(json_string, max_retries=3):
    """Attempts to parse a JSON string, with retries and LLM-based correction."""
    for attempt in range(max_retries):
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(color_prefix(f"JSON parsing failed, attempt {attempt + 1} of {max_retries}.", "Internal"))
            print(color_prefix(f"Error: {e}", "Internal"))
            print(color_prefix(f"Raw JSON string:\n{json_string}", "Internal"))

            if attempt < max_retries - 1:
                print(color_prefix("Retrying with LLM-based correction...", "Internal"))
                correction_prompt = f"""```json
                {json_string}
                ```
                Above JSON string has syntax error, fix the JSON so it can be parsed with json.loads() in python.
                Respond with JSON, and only JSON, with the correct format and make sure to comply the standard strictly.
                Do not stop generating until you are sure the JSON is complete and syntactically correct as defined in the format."""

                correction_prompt_tokens = len(TOKENIZER.encode(correction_prompt))
                print(color_prefix("Processing Correction Prompt", "Internal", time.time(), token_count=correction_prompt_tokens))

                json_string = ai_runtime_manager.invoke_llm(correction_prompt)
            else:
                print(color_prefix("Max retries reached. Returning None.", "Internal"))
                return None  # Explicitly return None on failure

# Initialize models and start the chatbot
initialize_models()

# Create PartitionContext instance
partition_context = PartitionContext(CTX_WINDOW_LLM, db_cursor, vector_store)

async def main():
    global db_writer
    # Initialize models (LLM, embedding model, vector store, and AIRuntimeManager)
    initialize_models()

    # Create PartitionContext instance after initializing ai_runtime_manager
    partition_context = PartitionContext(CTX_WINDOW_LLM, db_cursor, vector_store)

    # Start the input thread
    input_thread = Thread(target=input_thread_target, args=(loop, ai_runtime_manager, partition_context))
    input_thread.daemon = True
    input_thread.start()

    # Initialize DatabaseWriter with the loop
    db_writer = DatabaseWriter(db_connection, loop)
    db_writer.start_writer()  # Start the writer task

    # The loop is already running, so we don't call loop.run_forever() here
    # Instead, we'll use await to keep this async function alive
    await asyncio.sleep(float('inf'))

if __name__ == "__main__":
    # Print contents of all tables after database setup
    print_table_contents(db_cursor, "context_chunks")
    print_table_contents(db_cursor, "chat_history")
    print_table_contents(db_cursor, "cached_inference")

    print("Adelaide & Albert Engine initialized. Interaction is ready!")

    # Initialize the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print(color_prefix("\nExiting gracefully...", "Internal"))
    finally:
        if db_writer:
            db_writer.close()  # Close the database connection and stop the writer task
        loop.close()  # Close the event loop
        print(color_prefix("Cleanup complete. Goodbye!", "Internal"))