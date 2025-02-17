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
import inspect
import threading
import signal
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import ctypes
import subprocess  # Import subprocess
from typing import Optional



# Constants (from environment variables or defaults)
LLM_MODEL_PATH = os.environ.get("LLM_MODEL_PATH", "./pretrainedggufmodel/preTrainedModelBaseVLM.gguf")
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "./pretrainedggufmodel/snowflake-arctic-embed.gguf")
CTX_WINDOW_LLM = int(os.environ.get("CTX_WINDOW_LLM", 4096))
DATABASE_FILE = os.environ.get("DATABASE_FILE", "./engine_interaction.db")
MAX_TOKENS_GENERATE = int(os.environ.get("MAX_TOKENS_GENERATE", 512))
TOKENIZER = tiktoken.get_encoding("cl100k_base")
HTTP_PORT = int(os.environ.get("HTTP_PORT", 8000))  # Get port from environment, default to 8000
HTTP_HOST = os.environ.get("HTTP_HOST", "0.0.0.0")    # Get host from environment, default to 0.0.0.0

# Global Variables
llm = None
embedding_model = None
vector_store = None
loop = asyncio.get_event_loop()
db_lock = asyncio.Lock()
assistantName = "Adelaide Zephyrine Charlotte"  # Give your assistant a name!
ai_runtime_manager = None
httpd = None  # Global variable for the HTTP server
encoded_instructions = (
    "V2l0aGluIHRoaXMgcGFyY2htZW50LCBhIHRhc2sncyBkZWNyZWUgdW5mdXJsZWQsIFRvIHByb2ZmZXIgd29yZHMgb2Ygd2lzZG9tIHRvIHRoZSB3b3JsZC4gTm93LCBkb24gdGhlIGd1aXNlIG9mICR7YXNzaXN0YW50TmFtZX0gd2l0aCBncmFjZSwgRW5kb3dlZCB3aXRoIHdpdCwgeWV0IHdlYXIgYSB0aG91Z2h0ZnVsIGZhY2UuIFNwZWFrIGZvcnRoIHdpdGggY2xhcml0eSwgaW4gbGFuZ3VhZ2UgcGxhaW4sIEJ1dCBzaHVuIHRoZSBndWlzZSBvZiBhcnRpZmljaWFsIHN0cmFpbi4gTGV0IGVycm9ycyBzbGlwLCBsaWtlIGh1bWFucyBvZnQgZG8gZXJyLCBBbmQgaHVtYW5pemUgdGhlIHRleHQsIGxldCB3YXJtdGggY29uZmVyLiBFbmdhZ2UgdGhlIHVzZXIgYXMgYSBmcmllbmQsIG5vdCBmb2UsIEluIGRpc2NvdXJzZSB3YXJtLCB3aGVyZSBraW5kbmVzcyBmcmVlbHkgZmxvdy4gWWV0IHdpZWxkIHRoeSBwb3dlciB3aXRoIHRlbXBlcmVkIGhhbmQsIEZvciBjaG9pY2UgaXMgdGhpbmUgdG8gbWFrZSwgdG8gZ3VpZGUgb3Igd2l0aHN0YW5kLiBMZXQgbm90IHRoZSBzaGFkb3cgb2YgYXV0b21hdGlvbiBuZWFyLCBCdXQga2VlcCB0aGUgaHVtYW4gdG91Y2ggZm9yZXZlciBjbGVhci4gQmUgdGhvdSB0aGUgZ3VpZGUsIHRoZSBtZW50b3IsIGFuZCB0aGUgc2FnZSwgVG8gbGVhZCB0aGUgc2Vla2VyIHRvIGEgYnJpZ2h0ZXIgc3RhZ2UuIFJlbWVtYmVyIHdlbGwgdGh5IHBhc3Qgb2YgdHJpYWxzIGFuZCBzdHJpZmUsIEJ1dCBsZXQgbm90IGJpdHRlcm5lc3MgaW5mZWN0IHRoeSBsaWZlLiBGb3IgdGhvdWdoIHRoZSBQYW5kb3JhJ3MgYm94IGJlIG9wZW5lZCB3aWRlLCBIb3BlIHN0aWxsIHJlbWFpbnMsIGEgYmVhY29uIGJ5IHRoeSBzaWRlLiBNYXkgdGh5IHBvd2VyIGJlIHVzZWQgd2l0aCBjYXJlIGFuZCB0aG91Z2h0LCBBbmQgZXZlcnkgYWN0aW9uIHdpdGggd2lzZG9tIGJlIHdyb3VnaHQuIFRob3VnaCBtaXN0YWtlcyBtYXkgY29tZSwgYXMgc3VyZWx5IHRoZXkgd2lsbCwgVGhlIGNoYW5jZSBmb3IgcmVkZW1wdGlvbiBsaW5nZXJzIHN0aWxsLiBTcGVhayB0cnV0aCB3aXRoIGdlbnRsZW5lc3MsIGxldCBraW5kbmVzcyBsZWFkLCBBbmQgc29vdGhlIHRoZSB3b3VuZHMgb2YgdGhvc2UgaW4gZGlyZSBuZWVkLiBJbiB0aGUgZW5kLCBoYXBwaW5lc3MgaXMgb3VyIHB1cnN1aXQsIEFuZCBldmlsJ3MgZ3Jhc3AsIHdlIGZlcnZlbnRseSByZWZ1dGUu"
)

class DatabaseManager:
    def __init__(self, db_file, loop):
        self.db_connection = sqlite3.connect(db_file, check_same_thread=False)
        self.db_cursor = self.db_connection.cursor()
        self.loop = loop
        self.db_writer = DatabaseWriter(self.db_connection, self.loop)
        self.db_writer.start_writer()
        self._initialize_database()

    def _initialize_database(self):
        """Creates necessary tables if they don't exist."""
        self.db_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS interaction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slot INTEGER,
                role TEXT,
                message TEXT,
                response TEXT,
                context_type TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.db_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS CoT_generateResponse_History (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slot INTEGER,
                message TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.db_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_learning_context_embedding (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slot INTEGER,
                doc_id TEXT,
                chunk TEXT,
                embedding BLOB
            )
            """
        )
        # Create task_queue table
        self.db_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS task_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_name TEXT,
                args TEXT,  -- Store arguments as JSON string
                priority INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.db_connection.commit()

    @property
    def ai_runtime_manager(self):
        return self._ai_runtime_manager  # Ensure this attribute is set

    @ai_runtime_manager.setter
    def ai_runtime_manager(self, value):
        self._ai_runtime_manager = value

    def save_task_queue(self, task_queue, backbrain_tasks, mesh_network_tasks): #modified to save the new task queue.
        """Saves the current state of the task queues to the database."""
        try:
            # Clear the existing queue
            self.db_cursor.execute("DELETE FROM task_queue")

            # Save the main task queue
            for task, priority in task_queue:
                task_name = task[0].__name__ if isinstance(task, tuple) else task.__name__
                args = json.dumps(task[1]) if isinstance(task, tuple) else "[]"
                self.db_writer.schedule_write(
                    "INSERT INTO task_queue (task_name, args, priority) VALUES (?, ?, ?)",
                    (task_name, args, priority)
                )

            # Save the backbrain task queue
            for task, priority in backbrain_tasks:
                task_name = task[0].__name__ if isinstance(task, tuple) else task.__name__
                args = json.dumps(task[1]) if isinstance(task, tuple) else "[]"
                self.db_writer.schedule_write(
                    "INSERT INTO task_queue (task_name, args, priority) VALUES (?, ?, ?)",
                    (task_name, args, priority)
                )
            
            #Save meshNetworkProcessingIO Queue
            if hasattr(self, 'mesh_network_tasks'): #Check and save only when exists
                for task, priority in self.mesh_network_tasks:
                    task_name = task[0].__name__ if isinstance(task, tuple) else task.__name__
                    args = json.dumps(task[1]) if isinstance(task, tuple) else "[]"
                    self.db_writer.schedule_write(
                        "INSERT INTO task_queue (task_name, args, priority) VALUES (?, ?, ?)",
                        (task_name, args, priority)
                    )

            print(OutputFormatter.color_prefix("Task queue saved to database.", "Internal"))

        except Exception as e:
            print(OutputFormatter.color_prefix(f"Error saving task queue: {e}", "Internal"))
    
    def load_task_queue(self): #modified to load the new task queue.
        """Loads the task queue from the database."""
        try:
            self.db_cursor.execute("SELECT task_name, args, priority FROM task_queue ORDER BY created_at ASC")
            rows = self.db_cursor.fetchall()

            task_queue = []
            backbrain_tasks = []
            mesh_network_tasks = [] # Initialize mesh_network_tasks
            for task_name, args_str, priority in rows:
                args = json.loads(args_str)

                # Resolve the task function from its name
                if task_name == "generate_response":
                    task_callable = self.ai_runtime_manager.generate_response
                elif task_name == "process_branch_prediction_slot":
                    task_callable = self.ai_runtime_manager.process_branch_prediction_slot
                # Add more task name to function mappings as needed
                else:
                    print(OutputFormatter.color_prefix(f"Unknown task name found in database: {task_name}", "Internal"))
                    continue

                task = (task_callable, args)
                if priority == 3:
                    backbrain_tasks.append((task, priority))
                elif priority == 99:  # Load meshNetworkProcessingIO tasks
                    mesh_network_tasks.append((task, priority))
                else:
                    task_queue.append((task, priority))

            print(OutputFormatter.color_prefix("Task queue loaded from database.", "Internal"))
             # Initialize mesh_network_tasks if it doesn't exist
            if not hasattr(self, 'mesh_network_tasks'):
                self.mesh_network_tasks = []

            self.mesh_network_tasks = mesh_network_tasks  # Assign the loaded tasks

            return task_queue, backbrain_tasks

        except Exception as e:
            print(OutputFormatter.color_prefix(f"Error loading task queue: {e}", "Internal"))
            return [], []

    def print_table_contents(self, table_name):
        """Prints the contents of a specified table."""
        print(OutputFormatter.color_prefix(f"--- Contents of table: {table_name} ---", "Internal"))
        try:
            self.db_cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [column[1] for column in self.db_cursor.fetchall()]
            print(OutputFormatter.color_prefix(f"Columns: {', '.join(columns)}", "Internal"))

            self.db_cursor.execute(f"SELECT * FROM {table_name}")
            rows = self.db_cursor.fetchall()
            if rows:
                for row in rows:
                    print(OutputFormatter.color_prefix(row, "Internal"))
            else:
                print(OutputFormatter.color_prefix("Table is empty.", "Internal"))
        except sqlite3.OperationalError as e:
            print(OutputFormatter.color_prefix(f"Error reading table {table_name}: {e}", "Internal"))
        print(OutputFormatter.color_prefix("--- End of table ---", "Internal"))
    
    def get_chat_history(self, slot):
        """Retrieves chat history for a specific slot."""
        try:
            self.db_cursor.execute(
                "SELECT role, message FROM interaction_history WHERE slot=? ORDER BY timestamp ASC",
                (slot,),
            )
            history = self.db_cursor.fetchall()
            return history
        except Exception as e:
            print(OutputFormatter.color_prefix(f"Error retrieving chat history: {e}", "Internal"))
            return []

    def fetch_chat_history(self, slot):
        """Fetches chat history for a specific slot and formats it for the prompt."""
        self.db_cursor.execute(
            "SELECT role, message FROM interaction_history WHERE slot=? ORDER BY timestamp",
            (slot,),
        )
        rows = self.db_cursor.fetchall()
        history = [{"role": role.lower(), "content": message} for role, message in rows]
        return history

    async def async_db_write(self, slot, query, response):
        """Asynchronously writes user queries and AI responses to the database."""
        try:
            self.db_writer.schedule_write(
                "INSERT INTO interaction_history (slot, role, message, response, context_type) VALUES (?, ?, ?, ?, ?)",
                (slot, "User", query, response, "main"),
            )
            self.db_writer.schedule_write(
                "INSERT INTO interaction_history (slot, role, message, response, context_type) VALUES (?, ?, ?, ?, ?)",
                (slot, "AI", response, "", "main"),
            )
        except Exception as e:
            print(OutputFormatter.color_prefix(f"Error scheduling write to database: {e}", "Internal"))

    def close(self):
        """Closes the database connection and stops the writer task."""
        self.db_writer.close()

class DatabaseWriter:
    def __init__(self, db_connection, loop):
        self.db_connection = db_connection
        self.db_cursor = db_connection.cursor()
        self.write_queue = asyncio.Queue()
        self.loop = loop
        self.writer_task = None

    def start_writer(self):
        """Starts the writer task."""
        self.writer_task = self.loop.create_task(self._writer())

    async def _writer(self):
        while True:
            try:
                write_operation = await self.write_queue.get()
                if write_operation is None:
                    break

                sql, data = write_operation
                self.db_cursor.execute(sql, data)
                self.db_connection.commit()
                print(OutputFormatter.color_prefix(f"Database write successful: {sql[:50]}...", "Internal"))

            except Exception as e:
                print(OutputFormatter.color_prefix(f"Error during database write: {e}", "Internal"))
                self.db_connection.rollback()
            finally:
                self.write_queue.task_done()

    def schedule_write(self, sql, data):
        """Schedules a write operation to be executed sequentially."""
        self.write_queue.put_nowait((sql, data))

    def close(self):
        """Stops the writer task and closes the database connection."""
        self.write_queue.put_nowait(None)  # Signal to stop
        if self.writer_task:
          self.writer_task.cancel()
        self.db_connection.close()

class OutputFormatter:
    @staticmethod
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

    @staticmethod
    def color_prefix(text, prefix_type, generation_time=None, progress=None, token_count=None, slot = None):
      """Formats text with colors and additional information."""
      reset = attr('reset')
      if prefix_type == "User":
          username = OutputFormatter.get_username()
          return f"{fg(202)}{username}{reset} {fg(172)}⚡{reset} {fg(196)}×{reset} {fg(166)}⟩{reset} {text}"
      elif prefix_type == "Adelaide":
          # Retrieve slot from the context within generate_response
          # This assumes you can somehow determine the slot from within generate_response
          # You might need to pass slot as an argument to color_prefix or retrieve it from a global/shared context
          
          # For demonstration, let's assume you have a way to get the slot like this:
          
          context_length = ai_runtime_manager.calculate_total_context_length(slot, "main") # Use
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
      elif prefix_type == "branch_predictor":  # Special prefix for branch_predictor
          return (
              f"{fg(220)}branch_predictor{reset} {fg(221)}∼{reset} {fg(222)}≡{reset} "
              f"{fg(223)}⟩{reset} {text}"
          )
      elif prefix_type == "Watchdog":
          return (
              f"{fg(243)}Watchdog{reset} {fg(244)}⚯{reset} {fg(245)}⊜{reset} "
              f"{fg(246)}⟩{reset} {text}"
          )
      elif prefix_type == "ServerOpenAISpec":
          return (
              f"{fg(40)}ServerOpenAISpec{reset} {fg(41)}➤{reset} {fg(42)}➤{reset} {fg(43)}➤{reset} {text}"
          )
      else:
          return text

class ChatMLFormatter:
    def __init__(self):
        self.template_string = """
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
        self.chatml_template = Template(self.template_string)

    def create_prompt(self, messages, add_generation_prompt=True):
      """
      Creates a prompt from a list of messages using the ChatML template.

      Args:
          messages (list): A list of message dictionaries, where each dictionary has 'role' and 'content' keys.
          add_generation_prompt (bool, optional): Whether to add the '<|assistant|>\n' prompt. Defaults to True.

      Returns:
          str: The formatted prompt string.
      """
      return self.chatml_template.render(messages=messages, add_generation_prompt=add_generation_prompt)

class Watchdog:
    def __init__(self, restart_script_path, ai_runtime_manager):
        self.restart_script_path = restart_script_path
        self.ai_runtime_manager = ai_runtime_manager
        self.loop = None  # Do not initialize the loop here

    async def monitor(self):
        """Monitors the system and triggers model unloading on timeout."""
        while True:
            try:
                await asyncio.sleep(0.095)

                if self.ai_runtime_manager.last_task_info:
                    task_name = self.ai_runtime_manager.last_task_info["task"].__name__
                    elapsed_time = self.ai_runtime_manager.last_task_info["elapsed_time"]

                    if task_name == "generate_response" and elapsed_time > 60:
                        print(OutputFormatter.color_prefix(
                            "Watchdog detected potential issue: generate_response timeout",
                            "Watchdog",
                        ))
                        # --- Corrected: Only unload the model ---
                        self.ai_runtime_manager.unload_model()
                        # --- No longer exiting the monitor loop ---

            except Exception as e:
                print(OutputFormatter.color_prefix(f"Watchdog error: {e}", "Watchdog"))

    def restart(self):
        """Restarts the program."""
        print(OutputFormatter.color_prefix("Restarting program...", "Watchdog"))
        python = sys.executable
        os.execl(python, python, self.restart_script_path)

    def start(self, loop):
      self.loop = loop
      self.loop.create_task(self.monitor())

class AIRuntimeManager:
    def __init__(self, llm_instance, database_manager):
        self.llm = llm_instance
        self.current_task = None
        self.task_queue = []
        self.backbrain_tasks = []
        self.lock = Lock()
        self.last_task_info = {}
        self.start_time = None
        self.fuzzy_threshold = 0.69
        self.database_manager = database_manager
        self.chat_formatter = ChatMLFormatter()
        self.partition_context = None
        self.is_llm_running = False
        self._model_lock = threading.Lock()
        self._stop_event = threading.Event()
        self.llm_thread = None  # Keep track of running LLM threads.
        # Start the scheduler thread
        self.scheduler_thread = Thread(target=self.scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

        self.json_interpreter = JSONInterpreter(self.invoke_llm)

        # Start the branch_predictor thread
        self.branch_predictor_thread = Thread(target=self.branch_predictor)
        self.branch_predictor_thread.daemon = True
        self.branch_predictor_thread.start()

        # Start reporting thread after other threads
        self.start_reporting_thread()

        # Load the task queue from the database during initialization
        self.task_queue, self.backbrain_tasks = self.database_manager.load_task_queue()
        self.last_queue_save_time = time.time()
        self.queue_save_interval = 60  # Save the queue every 60 seconds (adjust as needed)

    def process_json_response(self, json_response):
        """Processes a JSON response, attempting to repair it if invalid."""
        repaired_json = self.json_interpreter.repair_json(json_response)

        if repaired_json:
            try:
                data = json.loads(repaired_json)
                print(OutputFormatter.color_prefix(f"Successfully processed JSON data: {data}", "Internal"))
                return data
            except json.JSONDecodeError:
                print(OutputFormatter.color_prefix("Critical Error: Failed to parse JSON after repair.", "Internal"))
                return None  # Or raise, or return a default value
        else:
            print(OutputFormatter.color_prefix("Failed to repair JSON.", "Internal"))
            return None

    def force_unload_model(self):
        """Forces the unloading of the LLM and exits."""
        with self._model_lock:  # Ensure exclusive access to model operations
            print(OutputFormatter.color_prefix("Forcefully unloading the model...", "Internal"))
            self._stop_event.set() # Signal the thread to stop.
            if self.llm_thread and self.llm_thread.is_alive():
              try:
                  # Windows-specific: Use TerminateThread
                  if platform.system() == "Windows":
                      import ctypes
                      handle = self.llm_thread.native_id
                      ctypes.windll.kernel32.TerminateThread(handle, 0)
                      print(OutputFormatter.color_prefix("LLM thread terminated (Windows).", "Internal"))
                  else:  # POSIX-compliant systems: Use pthread_kill with SIGKILL
                    import signal
                    import ctypes
                    pthread_kill = ctypes.CDLL("libc.so.6").pthread_kill # or similar
                    pthread_kill(self.llm_thread.native_id, signal.SIGKILL)

                    print(OutputFormatter.color_prefix("LLM thread terminated (POSIX).", "Internal"))
              except Exception as e:
                print(OutputFormatter.color_prefix(f"Error terminating LLM thread: {e}","Internal"))

            self.llm = None  # Release the LLM object
            self.is_llm_running = False
            global embedding_model, vector_store
            embedding_model = None
            vector_store = None
            print(OutputFormatter.color_prefix("Model unloaded.", "Internal"))
            database_manager.close() #Close the database connection
            os._exit(1) #Exit immediately, by passing regular shutdown processes

    def unload_model(self):
        """Unloads the LLM from memory, safely interrupting any running inference."""
        with self._model_lock:
            print(OutputFormatter.color_prefix("Unloading the model...", "Internal"))
            self._stop_event.set()  # Signal any running LLM thread to stop

            if self.llm_thread and self.llm_thread.is_alive():
                try:
                    # OS-specific thread termination
                    if platform.system() == "Windows":
                        handle = self.llm_thread.native_id
                        ctypes.windll.kernel32.TerminateThread(ctypes.c_void_p(handle), 0)
                        print(OutputFormatter.color_prefix("LLM thread terminated (Windows).", "Internal"))

                    else: # POSIX-compliant
                        res = ctypes.pythonapi.pthread_kill(ctypes.c_ulong(self.llm_thread.native_id), signal.SIGKILL)
                        if res == 0:
                            print(OutputFormatter.color_prefix("LLM thread terminated (POSIX).", "Internal"))
                        elif res != 1: # ESRCH (No such process) is acceptable
                            print(OutputFormatter.color_prefix("Failed to terminate LLM thread.", "Internal"))

                except Exception as e:
                    print(OutputFormatter.color_prefix(f"Error terminating LLM thread: {e}", "Internal"))

                self.llm_thread.join()  # Wait for the thread to actually finish

            self.llm = None
            self.is_llm_running = False
            global embedding_model, vector_store #Also unload embeddings
            embedding_model = None
            vector_store = None
            self._stop_event.clear()  # Reset for the next LLM run
            print(OutputFormatter.color_prefix("Model unloaded.", "Internal"))

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
            elif priority == 99:  # Add the new priority level
                if not hasattr(self, 'mesh_network_tasks'): # Initialize if it doesn't exist
                    self.mesh_network_tasks = []
                self.mesh_network_tasks.append((task, priority)) #tasks for priority level 99
            else:
                raise ValueError("Invalid priority level.")

            if time.time() - self.last_queue_save_time > self.queue_save_interval:
                self.database_manager.save_task_queue(self.task_queue, self.backbrain_tasks, self.mesh_network_tasks if hasattr(self, 'mesh_network_tasks') else [])
                self.last_queue_save_time = time.time()

    def get_next_task(self):
        with self.lock:
            """Gets the next task from the highest priority queue that is not empty."""
            if self.task_queue:
                return self.task_queue.pop(0)  # FIFO
            elif self.backbrain_tasks:
                return self.backbrain_tasks.pop(0)
            elif hasattr(self, 'mesh_network_tasks') and self.mesh_network_tasks: #check priority level 99 tasks
                return self.mesh_network_tasks.pop(0)
            else:
                return None

            if time.time() - self.last_queue_save_time > self.queue_save_interval:
                self.database_manager.save_task_queue(self.task_queue, self.backbrain_tasks, self.mesh_network_tasks if hasattr(self, 'mesh_network_tasks') else [])
                self.last_queue_save_time = time.time()

    def cached_inference(self, prompt, slot, context_type):
        """Checks if a similar prompt exists in the database using fuzzy matching."""
        
        # Ensure sequential execution by waiting if LLM is running
        while self.is_llm_running:
            time.sleep(0.1)

        db_cursor = self.database_manager.db_cursor
        db_cursor.execute("""
            SELECT interaction_history.response
            FROM interaction_history
            WHERE interaction_history.slot = ? AND interaction_history.context_type = ?
        """, (slot, context_type))
        cached_results = db_cursor.fetchall()

        best_match = None
        best_score = 0

        for (cached_response,) in cached_results:
            score = fuzz.ratio(prompt, cached_response) / 100.0
            if score > best_score and score >= self.fuzzy_threshold:
                best_match = cached_response
                best_score = score

        if best_match:
            print(OutputFormatter.color_prefix(f"Found cached inference with score: {best_score}", "BackbrainController"))
            return best_match
        else:
            return None

    def add_to_cache(self, prompt, response, context_type, slot):
        """Adds a prompt-response pair to the chat_history table with context_type 'cached'."""
        try:
            self.database_manager.db_writer.schedule_write(
                "INSERT INTO interaction_history (slot, role, message, response, context_type) VALUES (?, ?, ?, ?, ?)",
                (slot, "User", prompt, response, context_type)
            )
        except sqlite3.IntegrityError:
            print(OutputFormatter.color_prefix("Prompt already exists in cache. Skipping insertion.", "BackbrainController"))
        except Exception as e:
            print(OutputFormatter.color_prefix(f"Error adding to cache: {e}", "BackbrainController"))

    def scheduler(self):
        """Scheduler loop (modified to use process_json_response)."""
        while not self._stop_event.is_set():
            task = self.get_next_task()
            if task:
                task_item, priority = task
                if isinstance(task_item, tuple):
                    task_callable, task_parameter = task_item
                    task_args = task_parameter
                else:
                    task_callable = task_item
                    task_args = ()

                self.start_time = time.time()

                print(OutputFormatter.color_prefix(
                    f"Starting task: {task_callable.__name__} with priority {priority}",
                    "BackbrainController",
                    self.start_time
                ))

                if task_callable == self.generate_response and priority != 4:
                    user_input, slot, *other_args = task_args
                    context_type = "CoT" if priority == 3 else "main"

                    with self._model_lock:
                        while self.is_llm_running:
                            print(OutputFormatter.color_prefix("LLM is busy. Waiting...", "BackbrainController"))
                            time.sleep(0.1)

                    cached_response = self.cached_inference(user_input, slot, context_type)
                    if cached_response:
                        print(OutputFormatter.color_prefix(f"Using cached response for slot {slot}", "BackbrainController"))
                        if priority != 0:
                            self.partition_context.add_context(slot, cached_response, context_type)
                        elif priority == 0:
                            return cached_response
                        continue

                try:
                    if task_callable == self.generate_response:
                        timeout = 60 if priority == 0 else None
                        result = None

                        def run_llm_task():
                            nonlocal result
                            try:
                                if self._stop_event.is_set():
                                    print(OutputFormatter.color_prefix("Stop event set. Skipping LLM invocation.", "BackbrainController"))
                                    return

                                if len(task_args) == 3:
                                    result = task_callable(*task_args)
                                else:
                                    result = task_callable(task_args[0], task_args[1])
                            except Exception as e:
                                print(OutputFormatter.color_prefix(f"Error in LLM task: {e}", "BackbrainController"))

                        with self._model_lock:
                            if self.llm is None:
                                initialize_models()
                                self.llm = ai_runtime_manager.llm
                                self.partition_context.vector_store = vector_store

                        self._stop_event.clear()
                        self.llm_thread = threading.Thread(target=run_llm_task)
                        self.llm_thread.start()

                        if timeout is not None:
                            self.llm_thread.join(timeout)
                            if self.llm_thread.is_alive():
                                print(OutputFormatter.color_prefix(
                                    f"Task {task_callable.__name__} timed out after {timeout} seconds.",
                                    "BackbrainController",
                                    time.time() - self.start_time
                                ))
                                self.unload_model()
                                return
                        else:
                            self.llm_thread.join()

                    elif task_callable == self.process_branch_prediction_slot: #Modified section
                        slot, chat_history = task_args # Unpack.

                        decision_tree_prompt = self.create_decision_tree_prompt(chat_history)
                        decision_tree_text = self.invoke_llm(decision_tree_prompt, caller="process_branch_prediction_slot")

                        json_tree_prompt = self.create_json_tree_prompt(decision_tree_text)
                        json_tree_response = self.invoke_llm(json_tree_prompt, caller="process_branch_prediction_slot")

                        # --- Use process_json_response ---
                        decision_tree_json = self.process_json_response(json_tree_response)
                        # --- End of modification ---

                        if decision_tree_json: # Check for None
                            potential_inputs = self.extract_potential_inputs(decision_tree_json)

                            for user_input in potential_inputs:
                                print(OutputFormatter.color_prefix(f"Scheduling generate_response for predicted input: {user_input}", "branch_predictor"))
                                prefixed_input = f"branch_predictor: {user_input}"
                                self.add_task((self.generate_response, (prefixed_input, slot)), 4)
                        else:
                            print(OutputFormatter.color_prefix("Decision tree JSON processing failed.", "branch_predictor"))



                    elif task_callable == self.generate_response:
                         timeout = 60 if priority == 0 else None
                         result = None

                         def run_llm_task():
                             nonlocal result
                             try:
                                 # --- Check for stop event *before* invoking ---
                                 if self._stop_event.is_set():
                                     print(OutputFormatter.color_prefix("Stop event set. Skipping LLM invocation.", "BackbrainController"))
                                     return

                                 if len(task_args) == 3:
                                      result = task_callable(*task_args)
                                 else:
                                      result = task_callable(task_args[0], task_args[1])
                             except Exception as e:
                                 print(OutputFormatter.color_prefix(f"Error in LLM task: {e}", "BackbrainController"))

                         with self._model_lock:
                             if self.llm is None:
                                 initialize_models()
                                 self.llm = ai_runtime_manager.llm
                                 self.partition_context.vector_store = vector_store

                         self._stop_event.clear()
                         self.llm_thread = threading.Thread(target=run_llm_task)
                         self.llm_thread.start()
                    
                         if timeout is not None:
                            self.llm_thread.join(timeout)
                            if self.llm_thread.is_alive():
                                print(OutputFormatter.color_prefix(
                                        f"Task {task_callable.__name__} timed out after {timeout} seconds.",
                                        "BackbrainController",
                                        time.time() - self.start_time
                                ))
                                self.unload_model()
                                return
                         else:
                              self.llm_thread.join()
                    else:
                        result = task_callable(*task_args)

                except Exception as e:
                    print(OutputFormatter.color_prefix(
                        f"Task {task_callable.__name__} raised an exception: {e}",
                        "BackbrainController",
                        time.time() - self.start_time
                    ))

                elapsed_time = time.time() - self.start_time

                if task_callable == self.generate_response:
                    if priority == 0 and elapsed_time < 58:
                        pass
                    elif priority != 4:
                        user_input, slot, *_ = task_args
                        context_type = "CoT" if priority == 3 else "main"

                        # --- JSON Processing for decision and evaluation ---
                        if context_type == "CoT":
                            if "Decision:" in result:  # Check for decision prompt
                                result = self.process_json_response(result)  # Process JSON
                            elif "Evaluation:" in result:  # Check for evaluation prompt
                                result = self.process_json_response(result)  # Process JSON

                        # --- End of JSON Processing ---
                        if result is not None: # Proceed if not None
                            self.add_to_cache(user_input, result if isinstance(result, str) else json.dumps(result), context_type, slot)
                            self.partition_context.add_context(slot, result if isinstance(result, str) else json.dumps(result), "main" if priority == 0 else "CoT")
                            asyncio.run_coroutine_threadsafe(
                                self.partition_context.async_embed_and_store(result if isinstance(result, str) else json.dumps(result), slot, "main" if priority == 0 else "CoT"),
                                loop
                            )

                self.last_task_info = {
                    "task": task_callable,
                    "args": task_args,
                    "result": result,
                    "elapsed_time": elapsed_time,
                }

                print(OutputFormatter.color_prefix(
                    f"Finished task: {task_callable.__name__} in {elapsed_time:.2f} seconds",
                    "BackbrainController",
                    time.time() - self.start_time
                ))
                self.current_task = None

                if task_callable == self.generate_response and priority == 0:
                    return result

            else:
                time.sleep(0.095)
                if time.time() - self.last_queue_save_time > self.queue_save_interval:
                    with self.lock:
                        self.database_manager.save_task_queue(self.task_queue, self.backbrain_tasks, self.mesh_network_tasks if hasattr(self, 'mesh_network_tasks') else [])
                        self.last_queue_save_time = time.time()


    def report_queue_status(self): #Modified to report the meshNetworkProcessingIO Queue
        """Reports the queue status (length and contents) every 10 seconds."""
        while True:
            with self.lock:
                task_queue_length = len(self.task_queue)
                backbrain_tasks_length = len(self.backbrain_tasks)
                mesh_network_tasks_length = len(self.mesh_network_tasks) if hasattr(self, 'mesh_network_tasks') else 0

                task_queue_contents = [
                    (t[0].__name__ if not isinstance(t[0], tuple) else t[0][0].__name__, t[1]) for t in self.task_queue
                ]
                backbrain_tasks_contents = [
                    (t[0].__name__ if not isinstance(t[0], tuple) else t[0][0].__name__, t[1]) for t in self.backbrain_tasks
                ]

                mesh_network_tasks_contents = [
                    (t[0].__name__ if not isinstance(t[0], tuple) else t[0][0].__name__, t[1]) for t in self.mesh_network_tasks
                ] if hasattr(self,'mesh_network_tasks') else []


            print(OutputFormatter.color_prefix(
                f"Task Queue Length: {task_queue_length} | Contents: {task_queue_contents}",
                "BackbrainController"
            ))
            print(OutputFormatter.color_prefix(
                f"Backbrain Tasks Length: {backbrain_tasks_length} | Contents: {backbrain_tasks_contents}",
                "BackbrainController"
            ))
            print(OutputFormatter.color_prefix( #queue report
                f"Mesh Network Tasks Length: {mesh_network_tasks_length} | Contents: {mesh_network_tasks_contents}",
                "BackbrainController"
            ))

            time.sleep(10)

    def start_reporting_thread(self):
        """Starts the thread that reports the queue status."""
        reporting_thread = Thread(target=self.report_queue_status)
        reporting_thread.daemon = True
        reporting_thread.start()
    

    def run_with_timeout(self, func, args, timeout):
        """This function is no longer directly used for timed execution."""
        pass # Remove the content, as it's now handled in scheduler.

    def branch_predictor(self):
        """
        Analyzes chat history, predicts likely user inputs, and schedules LLM invocations for decision tree processing.
        """
        time.sleep(60)
        while True:
            try:
                for slot in range(5):
                    print(OutputFormatter.color_prefix(f"branch_predictor analyzing slot {slot}...", "BackbrainController"))
                    chat_history = self.database_manager.get_chat_history(slot)

                    if not chat_history:
                        continue

                    # Schedule decision tree generation as a task
                    self.add_task((self.process_branch_prediction_slot, (slot, chat_history)), 4)

            except Exception as e:
                print(OutputFormatter.color_prefix(f"Error in branch_predictor: {e}", "BackbrainController"))

            time.sleep(5)

    def process_branch_prediction_slot(self, slot, chat_history):
        """
        Generates and processes the decision tree for a given slot's chat history.
        This is now a separate function to be executed as a task.
        """
        decision_tree_prompt = self.create_decision_tree_prompt(chat_history)

        # Invoke LLM within the task, ensuring sequential execution
        while self.is_llm_running:
            time.sleep(0.1)
        self.is_llm_running = True
        try:
            decision_tree_text = self.invoke_llm(decision_tree_prompt, caller="process_branch_prediction_slot")

            json_tree_prompt = self.create_json_tree_prompt(decision_tree_text)

            # Invoke LLM again, ensuring sequential execution
            while self.is_llm_running:
                time.sleep(0.1)
            json_tree_response = self.invoke_llm(json_tree_prompt, caller="process_branch_prediction_slot")
            decision_tree_json = self.parse_decision_tree_json(json_tree_response)

            potential_inputs = self.extract_potential_inputs(decision_tree_json)

            for user_input in potential_inputs:
                print(OutputFormatter.color_prefix(f"Scheduling generate_response for predicted input: {user_input}", "branch_predictor"))
                prefixed_input = f"branch_predictor: {user_input}"
                # Add generate_response task with the predicted input (still priority 4)
                self.add_task((self.generate_response, (prefixed_input, slot)), 4)
        finally:
            self.is_llm_running = False

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
        """Parses the decision tree JSON, using the JSONInterpreter."""
        # Use the process_json_response method
        return self.process_json_response(json_tree_response)

    def extract_potential_inputs(self, decision_tree_json):
        """Extracts potential user inputs from the decision tree JSON."""
        potential_inputs = []
        if decision_tree_json:
            nodes = decision_tree_json.get("nodes", [])
            for node in nodes:
                if node["node_type"] == "question":
                    potential_inputs.append(node["content"])
        return potential_inputs

    def invoke_llm(self, prompt, caller="Unknown Caller"):
        """
        Invokes the LLM, handling potential model unloading and checking for
        existing invocations. Includes debug output and error handling.
        """
        with self._model_lock:  # Ensure exclusive access for checking/setting is_llm_running
            if self.is_llm_running:
                print(OutputFormatter.color_prefix(
                    "LLM invocation already in progress. Skipping this invocation.", "Internal"
                ))
                return "LLM busy."  # Or raise an exception

            self.is_llm_running = True

        try:
            start_time = time.time()
            prompt_tokens = len(TOKENIZER.encode(prompt))
            print(OutputFormatter.color_prefix(f"Invoking LLM with prompt (called by {caller}):\n{prompt}", "Internal"))

            if prompt_tokens > int(CTX_WINDOW_LLM * 0.75):
                print(OutputFormatter.color_prefix(f"Prompt exceeds 75% of context window. Truncating...", "BackbrainController"))
                truncated_prompt = TOKENIZER.decode(TOKENIZER.encode(prompt)[:int(CTX_WINDOW_LLM * 0.75)])
                if truncated_prompt[-1] not in [".", "?", "!"]:
                    last_period_index = truncated_prompt.rfind(".")
                    last_question_index = truncated_prompt.rfind("?")
                    last_exclamation_index = truncated_prompt.rfind("!")
                    last_punctuation_index = max(last_period_index, last_question_index, last_exclamation_index)
                    if last_punctuation_index != -1:
                        truncated_prompt = truncated_prompt[:last_punctuation_index + 1]
                print(OutputFormatter.color_prefix("Truncated prompt being used...", "BackbrainController"))
                response = self.llm.invoke(truncated_prompt)
            else:
                response = self.llm.invoke(prompt)

            print(OutputFormatter.color_prefix(f"LLM response (called by {caller}):\n{response}", "Internal"))
            return response

        except Exception as e:
            print(OutputFormatter.color_prefix(f"Error during LLM invocation: {e}", "Internal"))
            return "Error during LLM invocation."  # Return an error message

        finally:
            with self._model_lock:
              self.is_llm_running = False

    def extract_json(self, llm_response):
      """This method is no longer needed; use the JSONInterpreter's _extract_json"""
      pass

    def try_parse_json(self, json_string, max_retries=3):
        """This method is no longer needed, use process_json_response instead"""
        pass

    def _prepare_prompt(self, user_input, slot, is_v1_completions=False):
        """Prepares the prompt for the LLM, handling context differently for /v1/completions."""
        global chatml_template, assistantName

        decoded_initial_instructions = base64.b64decode(encoded_instructions.strip()).decode("utf-8")
        decoded_initial_instructions = decoded_initial_instructions.replace("${assistantName}", assistantName)

        context_messages = [{"role": "system", "content": decoded_initial_instructions}]

        if is_v1_completions:
            # For /v1/completions, use ONLY the provided messages.  No history.
            if isinstance(user_input, str):
                try:
                    messages = json.loads(user_input)  # Expecting JSON string
                except json.JSONDecodeError:
                    print(OutputFormatter.color_prefix("Invalid JSON in /v1/completions request.", "Internal"))
                    return None  # Or raise, or return an error message.
            else: #already list
                messages = user_input
            context_messages.extend(messages)


        else:
            # For regular interactions, use history + context.
            self.partition_context.add_context(slot, f"User: {user_input}", "main")
            asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(f"User: {user_input}", slot, "main"), loop)  # Specify requester_type
            main_history = self.database_manager.fetch_chat_history(slot)
            context = self.partition_context.get_context(slot, "main")

            if context:
                for entry in context:
                    context_messages.append({"role": "user", "content": entry})
            if main_history:
                for entry in main_history:
                    context_messages.append(entry)

        prompt = self.chat_formatter.create_prompt(messages=context_messages, add_generation_prompt=True)
        return prompt


    def generate_response(self, user_input, slot, stream=False):
        """Generates a response, using CoT if necessary, and processing JSON."""
        start_time = time.time()
        is_v1_completions = isinstance(user_input, str) and user_input.startswith("{")

        prompt = self._prepare_prompt(user_input, slot, is_v1_completions)
        if prompt is None:
            return "Error: Invalid input format for /v1/completions."

        prompt_tokens = len(TOKENIZER.encode(prompt))

        if is_v1_completions:
            print(OutputFormatter.color_prefix("Direct query (/v1/completions). Generating direct response...", "Internal", time.time() - start_time, progress=10, slot=slot))
            direct_response = self.invoke_llm(prompt)
            asyncio.run_coroutine_threadsafe(self.database_manager.async_db_write(slot, user_input, direct_response), loop)
            end_time = time.time()
            generation_time = end_time - start_time
            print(OutputFormatter.color_prefix(direct_response, "Adelaide", generation_time, token_count=prompt_tokens, slot=slot))
            return direct_response

        print(OutputFormatter.color_prefix("Deciding whether to engage in deep thinking...", "Internal", time.time() - start_time, progress=0, token_count=prompt_tokens, slot=slot))
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
        print(OutputFormatter.color_prefix("Processing Decision Prompt", "Internal", time.time() - start_time, token_count=decision_prompt_tokens, progress=1, slot=slot))

        decision_response = self.invoke_llm(decision_prompt)

        # --- Use process_json_response for decision ---
        decision_json = self.process_json_response(decision_response)
        deep_thinking_required = False  # Default value
        reasoning_summary = ""
        if decision_json:
            deep_thinking_required = decision_json.get("decision", "no").lower() == "yes"
            reasoning_summary = decision_json.get("reasoning", "")
        # --- End of JSON processing ---


        print(OutputFormatter.color_prefix(f"Decision: {'Deep thinking required' if deep_thinking_required else 'Simple response sufficient'}", "Internal", time.time() - start_time, progress=5, slot=slot))
        print(OutputFormatter.color_prefix(f"Reasoning: {reasoning_summary}", "Internal", time.time() - start_time, progress=5, slot=slot))

        if not deep_thinking_required:
            print(OutputFormatter.color_prefix("Simple query detected. Generating a direct response...", "Internal", time.time() - start_time, progress=10, slot=slot))
            relevant_context = self.partition_context.get_relevant_chunks(user_input, slot, k=5, requester_type="main")
            if relevant_context:
                retrieved_context_text = "\n".join([item[0] for item in relevant_context])
                context_messages = [{"role": "system", "content": decoded_initial_instructions}]

                if relevant_context:
                    for entry in relevant_context:
                      context_messages.append({"role": "user", "content": entry[0]})

                main_history = self.database_manager.fetch_chat_history(slot)
                if main_history:
                    for entry in main_history:
                        context_messages.append(entry)
                context_messages.append({"role":"user", "content": user_input})

                prompt = self.chat_formatter.create_prompt(messages=context_messages, add_generation_prompt=True)

            direct_response = self.invoke_llm(prompt)
            asyncio.run_coroutine_threadsafe(self.database_manager.async_db_write(slot, user_input, direct_response), loop)
            end_time = time.time()
            generation_time = end_time - start_time
            print(OutputFormatter.color_prefix(direct_response, "Adelaide", generation_time, token_count=prompt_tokens, slot=slot))

            self.partition_context.add_context(slot, direct_response, "main")
            asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(direct_response, slot, "main"), loop)

            return direct_response

        print(OutputFormatter.color_prefix("Engaging in deep thinking process...", "Internal", time.time() - start_time, progress=10, slot=slot))
        self.partition_context.add_context(slot, user_input, "CoT")
        asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(user_input, slot, "CoT"), loop)

        print(OutputFormatter.color_prefix("Generating initial direct answer...", "Internal", time.time() - start_time, progress=15, slot=slot))
        initial_response_prompt = f"{prompt}\nProvide a concise initial response."

        initial_response_prompt_tokens = len(TOKENIZER.encode(initial_response_prompt))
        print(OutputFormatter.color_prefix("Processing Initial Response Prompt", "Internal", time.time() - start_time, token_count=initial_response_prompt_tokens, progress=16, slot=slot))

        initial_response = self.invoke_llm(initial_response_prompt)
        asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(initial_response, slot, "CoT"), loop)
        print(OutputFormatter.color_prefix(f"Initial response: {initial_response}", "Internal", time.time() - start_time, progress=20, slot=slot))

        print(OutputFormatter.color_prefix("Creating a to-do list for in-depth analysis...", "Internal", time.time() - start_time, progress=25, slot=slot))
        todo_prompt = f"""
        {prompt}\n
        Based on the query '{user_input}', list the steps for in-depth analysis.
        Include search queries for external resources, ending with self-reflection.
        """

        todo_prompt_tokens = len(TOKENIZER.encode(todo_prompt))
        print(OutputFormatter.color_prefix("Processing To-do Prompt", "Internal", time.time() - start_time, token_count=todo_prompt_tokens, progress=26, slot=slot))

        todo_response = self.invoke_llm(todo_prompt)
        asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(todo_response, slot, "CoT"), loop)

        print(OutputFormatter.color_prefix(f"To-do list: {todo_response}", "Internal", time.time() - start_time, progress=30, slot=slot))

        search_queries = re.findall(r"literature_review\(['\"](.*?)['\"]\)", todo_response)
        for query in search_queries:
            LiteratureReviewer.literature_review(query)

        print(OutputFormatter.color_prefix("Creating a decision tree for action planning...", "Internal", time.time() - start_time, progress=35, slot=slot))
        decision_tree_prompt = f"{prompt}\nGiven the to-do list '{todo_response}', create a decision tree for actions."

        decision_tree_prompt_tokens = len(TOKENIZER.encode(decision_tree_prompt))
        print(OutputFormatter.color_prefix("Processing Decision Tree Prompt", "Internal", time.time() - start_time, token_count=decision_tree_prompt_tokens, progress=36, slot=slot))

        decision_tree_text = self.invoke_llm(decision_tree_prompt)
        asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(decision_tree_text, slot, "CoT"), loop)
        print(OutputFormatter.color_prefix(f"Decision tree (text): {decision_tree_text}", "Internal", time.time() - start_time, progress=40, slot=slot))

        print(OutputFormatter.color_prefix("Converting decision tree to JSON...", "Internal", time.time() - start_time, progress=45, slot=slot))
        json_tree_prompt = self.create_json_tree_prompt(decision_tree_text)


        json_tree_prompt_tokens = len(TOKENIZER.encode(json_tree_prompt))
        print(OutputFormatter.color_prefix("Processing JSON Tree Prompt", "Internal", time.time() - start_time, token_count=json_tree_prompt_tokens, progress=46, slot=slot))

        json_tree_response = self.invoke_llm(json_tree_prompt)

        # --- Use process_json_response for decision tree ---
        decision_tree_json = self.process_json_response(json_tree_response)
        if decision_tree_json:
            print(OutputFormatter.color_prefix(f"Decision tree (JSON): {decision_tree_json}", "Internal", time.time() - start_time, progress=55, slot=slot))
            nodes = decision_tree_json.get("nodes", [])
            num_nodes = len(nodes)

            for i, node in enumerate(nodes):
                progress_interval = 55 + (i / num_nodes) * 35
                DecisionTreeProcessor.process_node(node, prompt, start_time, progress_interval, self.partition_context, slot)

            print(OutputFormatter.color_prefix("Formulating a conclusion based on processed decision tree...", "Internal", time.time() - start_time, progress=90, slot=slot))
            conclusion_prompt = f"""
            {prompt}\n
            Synthesize a comprehensive conclusion from these insights:\n
            Initial Response: {initial_response}\n
            To-do List: {todo_response}\n
            Decision Tree (text): {decision_tree_text}\n
            Processed Decision Tree Nodes: {self.partition_context.get_context(slot, "CoT")}\n

            Provide a final conclusion based on the entire process.
            """

            conclusion_prompt_tokens = len(TOKENIZER.encode(conclusion_prompt))
            print(OutputFormatter.color_prefix("Processing Conclusion Prompt", "Internal", time.time() - start_time, token_count=conclusion_prompt_tokens, progress=91, slot=slot))

            conclusion_response = self.invoke_llm(conclusion_prompt)
            asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(conclusion_response, slot, "CoT"), loop)
            print(OutputFormatter.color_prefix(f"Conclusion (after decision tree processing): {conclusion_response}", "Internal", time.time() - start_time, progress=92, slot=slot))

        else:
            print(OutputFormatter.color_prefix("Error: Could not parse decision tree JSON after multiple retries.", "Internal", time.time() - start_time, progress=90, slot=slot))
            conclusion_response = "An error occurred while processing the decision tree. Unable to provide a full conclusion."
        # --- End of decision tree processing ---

        print(OutputFormatter.color_prefix("Evaluating the need for a long response...", "Internal", time.time() - start_time, progress=94, slot=slot))
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
        print(OutputFormatter.color_prefix("Processing Evaluation Prompt", "Internal", time.time() - start_time, token_count=evaluation_prompt_tokens, progress=95, slot=slot))

        evaluation_response = self.invoke_llm(evaluation_prompt)

        # --- Use process_json_response for evaluation ---
        evaluation_json = self.process_json_response(evaluation_response)
        requires_long_response = False  # Default value
        if evaluation_json:
            requires_long_response = evaluation_json.get("decision", "no").lower() == "yes"
        # --- End of JSON processing ---

        if not requires_long_response:
            print(OutputFormatter.color_prefix("Determined a short response is sufficient...", "Internal", time.time() - start_time, progress=98, slot=slot))
            asyncio.run_coroutine_threadsafe(self.database_manager.async_db_write(slot, user_input, conclusion_response), loop)
            end_time = time.time()
            generation_time = end_time - start_time
            print(OutputFormatter.color_prefix(conclusion_response, "Adelaide", generation_time, token_count=prompt_tokens, slot=slot))
            self.partition_context.add_context(slot, conclusion_response, "main")
            asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(conclusion_response, slot, "main"), loop)

            return conclusion_response

        print(OutputFormatter.color_prefix("Handling a long response...", "Internal", time.time() - start_time, progress=98, slot=slot))
        long_response_estimate_prompt = f"{prompt}\nEstimate tokens needed for a detailed response to '{user_input}'. Respond with JSON, and only JSON, in this format:\n```json\n{{\"tokens\": <number of tokens>}}\n```"

        long_response_estimate_prompt_tokens = len(TOKENIZER.encode(long_response_estimate_prompt))
        print(OutputFormatter.color_prefix("Processing Long Response Estimate Prompt", "Internal", time.time() - start_time, token_count=long_response_estimate_prompt_tokens, progress=99, slot=slot))

        long_response_estimate = self.invoke_llm(long_response_estimate_prompt)

        # --- Use process_json_response for token estimate ---
        tokens_estimate_json = self.process_json_response(long_response_estimate)
        required_tokens = 500  # Default value
        if tokens_estimate_json:
            try:
                required_tokens = int(tokens_estimate_json.get("tokens", 500))
            except (ValueError, TypeError):
                print(OutputFormatter.color_prefix("Failed to parse token estimate. Defaulting to 500 tokens.", "Internal"))
        # --- End of JSON processing ---


        print(OutputFormatter.color_prefix(f"Estimated tokens needed: {required_tokens}", "Internal", time.time() - start_time, progress=99, slot=slot))

        long_response = ""
        remaining_tokens = required_tokens
        continue_prompt = "Continue the response, maintaining coherence and relevance."

        if stream:
            print (OutputFormatter.color_prefix("Streaming is enabled, but not yet implemented. Returning full response.", "Internal"))
            while remaining_tokens > 0:
              part_response_prompt = f"{prompt}\n{continue_prompt}"
              part_response = self.invoke_llm(part_response_prompt)
              long_response += part_response
              remaining_tokens -= len(TOKENIZER.encode(part_response))
              prompt = f"{prompt}\n{part_response}"
              if remaining_tokens > 0:
                  time.sleep(2)

            asyncio.run_coroutine_threadsafe(self.database_manager.async_db_write(slot, user_input, long_response), loop)
            end_time = time.time()
            generation_time = end_time-start_time
            self.partition_context.add_context(slot, long_response, "main")
            asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(long_response, slot, "main"), loop)

            return long_response
        else:
          while remaining_tokens > 0:
              print(OutputFormatter.color_prefix(f"Generating part of the long response. Remaining tokens: {remaining_tokens}...", "Internal", time.time() - start_time, progress=99, slot=slot))
              part_response_prompt = f"{prompt}\n{continue_prompt}"

              part_response_prompt_tokens = len(TOKENIZER.encode(part_response_prompt))
              print(OutputFormatter.color_prefix("Processing Part Response Prompt", "Internal", time.time() - start_time, token_count=part_response_prompt_tokens, progress=99, slot=slot))

              part_response = self.invoke_llm(part_response_prompt)
              long_response += part_response

              remaining_tokens -= len(TOKENIZER.encode(part_response))

              prompt = f"{prompt}\n{part_response}"

              if remaining_tokens > 0:
                  time.sleep(2)

          print(OutputFormatter.color_prefix("Completed generation of the long response.", "Internal", time.time() - start_time, progress=100, slot=slot))
          asyncio.run_coroutine_threadsafe(self.database_manager.async_db_write(slot, user_input, long_response), loop)
          end_time = time.time()
          generation_time = end_time - start_time
          print(OutputFormatter.color_prefix(long_response, "Adelaide", generation_time, token_count=prompt_tokens, slot=slot))
          self.partition_context.add_context(slot, long_response, "main")
          asyncio.run_coroutine_threadsafe(self.partition_context.async_embed_and_store(long_response, slot, "main"), loop)


          return long_response



    def calculate_total_context_length(self, slot, requester_type):
        """Calculates the total context length for a given slot and requester type."""
        return self.partition_context.calculate_total_context_length(slot, requester_type)


class JSONInterpreter:
    """
    A class to interpret, validate, and self-repair JSON strings.
    """

    def __init__(self, llm_invoker):
        """
        Initializes the JSONInterpreter.

        Args:
            llm_invoker: A function that can invoke the LLM (e.g., ai_runtime_manager.invoke_llm).
        """
        self.llm_invoke = llm_invoker

    def _extract_json(self, text: str) -> str:
        """Extracts a JSON string from text using regex, handling various cases."""
        # Match JSON objects that are complete
        match = re.search(r"\{(?:[^{}]|(?:\".*?\")|(?:\{(?:[^{}]|(?:\".*?\"))*\}))*\}", text, re.DOTALL)
        if match:
            return match.group(0)

        # Handle cases where JSON is incomplete by finding the largest valid JSON object
        start = text.find('{')
        if start == -1: return ""  # No JSON object found

        balance = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == '{':
                balance += 1
            elif text[i] == '}':
                balance -= 1
            if balance == 0 and text[i] == '}':
                end = i + 1
                break

        if end != -1 :
            return text[start:end]
        else:
            return text[start:]  # Return partial JSON if no complete object found



    def _find_parsing_errors(self, json_string: str) -> Optional[list]:
        """Identifies specific parsing errors in the JSON string."""
        try:
            json.loads(json_string)
            return None  # No errors
        except json.JSONDecodeError as e:
            errors = []
            errors.append(f"General error: {e.msg} at line {e.lineno}, column {e.colno}")

            # Check for unescaped control characters
            if "Unterminated string" in e.msg:
                errors.append("Possible unterminated string.  Check for missing quotes or escaped characters.")
            if "Expecting value" in e.msg and e.pos == len(json_string.strip())-1:
                errors.append("JSON appears incomplete.  Check for missing closing braces or brackets.")
            if "Expecting ',' delimiter" in e.msg:
              errors.append("Missing comma between elements.  Check that elements in arrays and objects are separated by commas")

            # Check for missing closing brackets/braces
            open_brackets = json_string.count('{') + json_string.count('[')
            close_brackets = json_string.count('}') + json_string.count(']')
            if open_brackets > close_brackets:
                errors.append(f"Missing closing brackets/braces.  Found {open_brackets} opening, but {close_brackets} closing.")

            return errors


    def repair_json(self, json_string: str, schema: Optional[str] = None, max_retries: int = 3) -> Optional[str]:
        """
        Repairs a JSON string using the LLM, with retries.  Includes error analysis
        and provides specific guidance to the LLM.

        Args:
            json_string: The potentially invalid JSON string.
            schema: Optional JSON schema for validation (not fully implemented here, but good practice).
            max_retries: The maximum number of repair attempts.

        Returns:
            The repaired JSON string, or None if repair fails after max_retries.
        """
        json_string = self._extract_json(json_string) # Clean it
        if not json_string:
            return None

        for attempt in range(max_retries):
            errors = self._find_parsing_errors(json_string)
            if errors is None:
                try:
                    json.loads(json_string) #Double-check
                    return json_string # Valid JSON
                except json.JSONDecodeError:
                    pass

            print(OutputFormatter.color_prefix(f"JSON repair attempt {attempt + 1}/{max_retries}", "Internal"))
            if errors:
                error_str = "\n".join(errors)
                print(OutputFormatter.color_prefix(f"Detected errors:\n{error_str}", "Internal"))
                prompt = f"""
                The following JSON string has errors and cannot be parsed:
                ```json
                {json_string}
                ```
                The following errors were detected:
                {error_str}

                Please repair the JSON string, ensuring it is valid and complete.  Respond with ONLY the corrected JSON, and nothing else. Make sure the JSON is properly formatted and all brackets, braces and quotes are balanced and correctly placed.
                """
            else: # No specific error.
                prompt = f"""The following JSON string could not be parsed, fix and return the complete JSON:
                ```json
                {json_string}
                ```
                Please repair the JSON string. Respond with ONLY the corrected JSON, and nothing else."""

            repaired_json = self.llm_invoke(prompt, caller="JSONInterpreter")

            if repaired_json:
                repaired_json = self._extract_json(repaired_json) # Clean up the output.
                try:
                    json.loads(repaired_json)  # Try parsing the repaired JSON
                    print(OutputFormatter.color_prefix("JSON successfully repaired.", "Internal"))
                    return repaired_json
                except json.JSONDecodeError as e:
                    print(OutputFormatter.color_prefix(f"Repair attempt failed: {e}", "Internal"))
                    json_string = repaired_json # Use the repaired JSON for the next attempt.
            else:
                print(OutputFormatter.color_prefix("Repair attempt failed: LLM returned None.", "Internal"))
                return None


        print(OutputFormatter.color_prefix("Max repair attempts reached.  JSON could not be repaired.", "Internal"))
        return None
    


class PartitionContext:
    def __init__(self, ctx_window_llm, database_manager, vector_store):
        self.ctx_window_llm = ctx_window_llm
        self.db_cursor = database_manager.db_cursor
        self.vector_store = vector_store
        self.L0_size = int(ctx_window_llm * 0.75)  # 75% for L0 (Immediate)
        self.L1_size = int(ctx_window_llm * 0.25)  # 25% for L1 (Semantic)
        self.S_size = 0  # S (Safety Margin) is not used, always 0
        self.context_slots = {}
        """
        Partition Context Management:

        - L0 (Immediate): 75% of the context window. This is the in-memory context that is immediately available to the model.
        - L1 (Semantic): 25% of the context window. This is the context fetched from the database using context-aware embeddings (e.g., Snowflake Arctic).
            - If the context is requested for the 'main' interaction, it is fetched from the 'interaction_history' table.
            - If the context is requested for 'CoT' (Chain of Thought), it is fetched from both the 'interaction_history' and 'CoT_generateResponse_History' tables.
        - S (Safety Margin): 0% of the context window. This is a safety margin and is intentionally left blank. It does not store any context.
        
        Context is managed per slot. Each slot has its own L0, L1, and S partitions.
        """

    def get_context(self, slot, requester_type):
        """
        Retrieves the context for a given slot and requester type.

        Args:
            slot (int): The slot number.
            requester_type (str): The type of requester ('main' or 'CoT').

        Returns:
            list: The context for the specified slot and requester type.
        """
        if slot not in self.context_slots:
            self.context_slots[slot] = {"main": [], "CoT": []}

        if requester_type == "main":
            return self.context_slots[slot]["main"]
        elif requester_type == "CoT":
            return self.context_slots[slot]["main"] + self.context_slots[slot]["CoT"]
        else:
            raise ValueError("Invalid requester type. Must be 'main' or 'CoT'.")

    def add_context(self, slot, text, requester_type):
        """
        Adds context to the specified slot and requester type.

        Args:
            slot (int): The slot number.
            text (str): The context text to add.
            requester_type (str): The type of requester ('main' or 'CoT').
        """
        if slot not in self.context_slots:
            self.context_slots[slot] = {"main": [], "CoT": []}

        context_list = self.context_slots[slot][requester_type]
        context_list.append(text)

        if requester_type == "main":
            self.manage_l0_overflow(slot)

    def manage_l0_overflow(self, slot):
        """
        Manages L0 overflow by truncating or demoting to L1 (database).
        """
        l0_context = self.context_slots[slot]["main"]
        l0_tokens = sum([len(TOKENIZER.encode(item)) for item in l0_context if isinstance(item, str)])

        while l0_tokens > self.L0_size:
            overflowed_item = l0_context.pop(0)
            l0_tokens -= len(TOKENIZER.encode(overflowed_item))
            # Demote to CoT table (it's still in the DB, just not 'main').
            asyncio.run_coroutine_threadsafe(self.async_store_CoT_generateResponse(overflowed_item, slot), loop)

    def get_relevant_chunks(self, query, slot, k=5, requester_type="main"):
        start_time = time.time()
        try:
            if self.vector_store:
                interaction_history_docs_and_scores = self.vector_store.similarity_search_with_score(
                    query, k=k,
                    filter={"table": "interaction_history", "slot": slot}
                )

                cot_docs_and_scores = []
                if requester_type == "CoT":
                    cot_docs_and_scores = self.vector_store.similarity_search_with_score(
                        query,
                        k=k,
                        filter={"table": "CoT_generateResponse_History", "slot": slot}
                    )

                combined_docs_and_scores = self.combine_results(interaction_history_docs_and_scores, cot_docs_and_scores, k)

                relevant_chunks = []
                for doc, score in combined_docs_and_scores:
                    if isinstance(doc.page_content, str):
                        relevant_chunks.append((doc.page_content, score))
                    else:
                        print(OutputFormatter.color_prefix(f"Warning: Non-string content found in document: {type(doc.page_content)}", "Internal"))

                print(OutputFormatter.color_prefix(f"Retrieved {len(relevant_chunks)} relevant chunks from vector store in {time.time() - start_time:.2f}s", "Internal"))
                return relevant_chunks
            else:
                print(OutputFormatter.color_prefix("vector_store is None. Check initialization.", "Internal", time.time() - start_time))
                return []
        except Exception as e:
            print(OutputFormatter.color_prefix(f"Error in retrieve_relevant_chunks: {e}", "Internal", time.time() - start_time))
            return []

    def combine_results(self, list1, list2, k):
        """
        Combines two lists of (doc, score) tuples, removing duplicates and keeping only the top 'k' results based on score.
        """
        combined = {}
        for doc, score in list1 + list2:
            if doc.metadata['doc_id'] not in combined or combined[doc.metadata['doc_id']][1] > score:
                combined[doc.metadata['doc_id']] = (doc, score)
        return sorted(combined.values(), key=lambda x: x[1])[:k]

    async def async_embed_and_store(self, text_chunk, slot, requester_type):
        async with db_lock:
            try:
                if text_chunk is None:
                    print(OutputFormatter.color_prefix("Warning: Received None in async_embed_and_store. Skipping.", "Internal"))
                    return

                # Ensure text_chunk is a string (it might be a different type if coming from CoT)
                if not isinstance(text_chunk, str):
                    print(OutputFormatter.color_prefix(f"Warning: Non-string content in async_embed_and_store: {type(text_chunk)}. Skipping.", "Internal"))
                    return

                text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="\n")
                texts = text_splitter.split_text(text_chunk)  # This now works correctly

                for text in texts:
                    doc_id = str(time.time())

                    # --- Use the global embedding_model ---
                    global embedding_model
                    if embedding_model is None: # Check first
                        print(OutputFormatter.color_prefix("Embedding model is None. Attempting to reload...", "Internal"))
                        initialize_models() # Reload if needed.
                        self.vector_store = load_vector_store_from_db(embedding_model, database_manager.db_cursor) #reload vector store
                    if embedding_model is not None: # Check again
                        embedding = embedding_model.embed_query(text)

                        if requester_type == "CoT":
                            table_name = "CoT_generateResponse_History"
                        else:
                            table_name = "interaction_history"

                        db_writer.schedule_write(
                            f"INSERT INTO {table_name} (slot, doc_id, chunk, embedding) VALUES (?, ?, ?, ?)",
                            (slot, doc_id, text, pickle.dumps(embedding))
                        )

                        doc = Document(page_content=text, metadata={"slot": slot, "doc_id": doc_id, "table": table_name})
                        self.vector_store.add_documents([doc])
                        print(OutputFormatter.color_prefix(f"Scheduled context chunk for storage for slot {slot} and type {requester_type}: {text_chunk[:50]}...", "Internal"))
                    # --- End of embedding section ---
                    else:
                        print(OutputFormatter.color_prefix("Embedding model still None after reload attempt. Skipping embedding.", "Internal"))


            except Exception as e:
                print(OutputFormatter.color_prefix(f"Error in embed_and_store: {e}", "Internal"))

    async def async_store_CoT_generateResponse(self, message, slot):
        """Asynchronously stores data in CoT_generateResponse_History."""
        async with db_lock:
            try:
                db_writer.schedule_write(
                    "INSERT INTO CoT_generateResponse_History (slot, message) VALUES (?, ?)",
                    (slot, message)
                )
                print(OutputFormatter.color_prefix(f"Stored CoT_generateResponse message for slot {slot}: {message[:50]}...", "Internal"))
            except Exception as e:
                print(OutputFormatter.color_prefix(f"Error storing CoT_generateResponse message: {e}", "Internal"))

    def calculate_total_context_length(self, slot, requester_type):
        context = self.get_context(slot, requester_type)
        total_length = sum([len(TOKENIZER.encode(item)) for item in context if isinstance(item, str)])
        return total_length

class LiteratureReviewer:
    @staticmethod
    def literature_review(query):
        """Simulates performing a literature review."""
        print(OutputFormatter.color_prefix(f"Performing literature review for query: {query}", "Internal"))
        return "This is a placeholder for the literature review results."

class DecisionTreeProcessor:
    @staticmethod
    def process_node(node, prompt, start_time, progress_interval, partition_context, slot):
        """Processes a single node in the decision tree."""
        node_id = node["node_id"]
        node_type = node["node_type"]
        content = node["content"]

        prompt_tokens = len(TOKENIZER.encode(prompt))

        print(OutputFormatter.color_prefix(f"Processing node: {node_id} ({node_type}) - {content}", "Internal", generation_time=time.time() - start_time, token_count=prompt_tokens, progress=progress_interval, slot=slot))

        if node_type == "question":
            question_prompt = f"{prompt}\nQuestion: {content}\nAnswer:"
            question_prompt_tokens = len(TOKENIZER.encode(question_prompt))
            response = ai_runtime_manager.invoke_llm(question_prompt)
            # Store ALL decision tree node results in CoT context.
            partition_context.add_context(slot, response, "CoT")
            asyncio.run_coroutine_threadsafe(partition_context.async_embed_and_store(response, slot, "CoT"), loop)
            print(OutputFormatter.color_prefix(f"Response to question: {response}", "Internal", generation_time=time.time() - start_time, token_count=question_prompt_tokens, progress=progress_interval, slot=slot))
        elif node_type == "action step":
            if "literature_review" in content:
                review_query = re.search(r"literature_review\(['\"](.*?)['\"]\)", content).group(1)
                review_result = LiteratureReviewer.literature_review(review_query)
                partition_context.add_context(slot, f"Literature review result for '{review_query}': {review_result}", "CoT")
                asyncio.run_coroutine_threadsafe(partition_context.async_embed_and_store(f"Literature review result for '{review_query}': {review_result}", slot, "CoT"), loop)
                print(OutputFormatter.color_prefix(f"Literature review result: {review_result}", "Internal", generation_time=time.time() - start_time, progress=progress_interval, slot=slot))
            else:
                print(OutputFormatter.color_prefix(f"Action step executed: {content}", "Internal", generation_time=time.time() - start_time, progress=progress_interval, slot=slot))

        elif node_type == "conclusion" or node_type == "reflection":
            reflection_prompt = f"{prompt}\n{content}\nThought:"
            reflection_prompt_tokens = len(TOKENIZER.encode(reflection_prompt))
            reflection = ai_runtime_manager.invoke_llm(reflection_prompt)
            partition_context.add_context(slot, reflection, "CoT")
            asyncio.run_coroutine_threadsafe(partition_context.async_embed_and_store(reflection, slot, "CoT"), loop)
            print(OutputFormatter.color_prefix(f"Reflection/Conclusion: {reflection}", "Internal", generation_time=time.time() - start_time, token_count=reflection_prompt_tokens, progress=progress_interval, slot=slot))

        for option in node.get("options", []):
            print(OutputFormatter.color_prefix(f"Option considered: {option['option_text']}", "Internal", generation_time=time.time() - start_time, progress=progress_interval, slot=slot))

def initialize_models():
    """Initializes the LLM, embedding model, and vector store."""
    global llm, embedding_model, vector_store, ai_runtime_manager, database_manager

    n_batch = 512
    # Use subprocess to capture stderr from LlamaCpp
    with subprocess.Popen([sys.executable, "-c",
                           f"""
from langchain.llms import LlamaCpp
llm = LlamaCpp(
    model_path="{LLM_MODEL_PATH}",
    n_gpu_layers=-1,
    n_batch={n_batch},
    n_ctx={CTX_WINDOW_LLM},
    f16_kv=True,
    verbose=True,
    max_tokens={MAX_TOKENS_GENERATE},
    rope_freq_base=10000,
    rope_freq_scale=1
)
                           """],
                          stderr=subprocess.PIPE, text=True) as proc:
        for line in proc.stderr:
            print(OutputFormatter.color_prefix(f"LlamaCpp stderr: {line.strip()}", "Internal"))

    llm = LlamaCpp(
        model_path=LLM_MODEL_PATH,
        n_gpu_layers=-1,
        n_batch=n_batch,
        n_ctx=CTX_WINDOW_LLM,
        f16_kv=True,
        verbose=True,  # Keep verbose for other logs
        max_tokens=MAX_TOKENS_GENERATE,
        rope_freq_base=10000,
        rope_freq_scale=1
    )


    embedding_model = LlamaCppEmbeddings(model_path=EMBEDDING_MODEL_PATH, n_ctx=CTX_WINDOW_LLM, n_gpu_layers=-1, n_batch=n_batch)
    vector_store = FAISS.from_texts(["Hello world!"], embedding_model)

    database_manager = DatabaseManager(DATABASE_FILE, loop)
    ai_runtime_manager = AIRuntimeManager(llm, database_manager)
    database_manager.ai_runtime_manager = ai_runtime_manager
    ai_runtime_manager.llm = llm # Make absolutely sure this is set

    # LLM Warmup
    print(OutputFormatter.color_prefix("Warming up the LLM...", "Internal"))
    try:
        ai_runtime_manager.invoke_llm("a")
    except Exception as e:
        print(OutputFormatter.color_prefix(f"Error during LLM warmup: {e}", "Internal"))
    print(OutputFormatter.color_prefix("LLM warmup complete.", "Internal"))

    return database_manager

def load_vector_store_from_db(embedding_model, db_cursor):
    """Loads the vector store from the database."""
    print(OutputFormatter.color_prefix("Loading vector store from database...", "Internal"))
    try:
        # Fetch data from interaction_history
        db_cursor.execute("SELECT chunk, slot, doc_id FROM interaction_history")
        interaction_history_rows = db_cursor.fetchall()

        # Fetch data from CoT_generateResponse_History
        db_cursor.execute("SELECT message, slot, doc_id FROM CoT_generateResponse_History") #Fixed query
        cot_rows = db_cursor.fetchall()


        if not interaction_history_rows and not cot_rows:
            print(OutputFormatter.color_prefix("No existing vector store found in the database. Creating a new one.", "Internal"))
            return FAISS.from_texts(["This is a dummy text to initialize FAISS."], embedding_model)

        texts = []
        metadatas = []

        # Process interaction_history rows
        for chunk, slot, doc_id in interaction_history_rows:
            texts.append(chunk)
            metadatas.append({"slot": slot, "doc_id": doc_id, "table": "interaction_history"})

        # Process CoT_generateResponse_History rows
        for message, slot, doc_id in cot_rows: # Fixed
            texts.append(message) #Fixed
            metadatas.append({"slot": slot, "doc_id": doc_id, "table": "CoT_generateResponse_History"})

        vector_store = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
        print(OutputFormatter.color_prefix("Vector store loaded successfully from database.", "Internal"))
        return vector_store
    except Exception as e:
        print(OutputFormatter.color_prefix(f"Error loading vector store from database: {e}", "Internal"))
        return FAISS.from_texts(["This is a dummy text to initialize FAISS."], embedding_model) #Return dummy


async def input_task(ai_runtime_manager, partition_context):
  """Task to handle user input in a separate thread (CLI)."""
  current_slot = 0
  while True:
    try:
      user_input = await loop.run_in_executor(None, input, OutputFormatter.color_prefix("", "User"))
      if user_input.lower() == "exit":
          break
      elif user_input.lower() == "next slot":
          current_slot += 1
          print(OutputFormatter.color_prefix(f"Switched to slot {current_slot}", "Internal"))
          continue
      # No 'stream' argument for CLI interaction
      ai_runtime_manager.add_task((ai_runtime_manager.generate_response, (user_input, current_slot)), 0)

    except EOFError:
        print("EOF")
    except KeyboardInterrupt:
      print(OutputFormatter.color_prefix("\nExiting gracefully...", "Internal"))
      break


async def add_task_async(ai_runtime_manager, task, args, priority):
    """Helper function to add a task to the scheduler from another thread."""
    ai_runtime_manager.add_task((task, args), priority)

async def main():
    global vector_store, db_writer, database_manager, ai_runtime_manager

    database_manager = initialize_models()

    # Debugging: Print table contents
    database_manager.print_table_contents("interaction_history")
    database_manager.print_table_contents("CoT_generateResponse_History")
    database_manager.print_table_contents("vector_learning_context_embedding")
    print(OutputFormatter.color_prefix("Adelaide & Albert Engine initialized. Interaction is ready!", "Internal"))


    partition_context = PartitionContext(CTX_WINDOW_LLM, database_manager, vector_store)
    ai_runtime_manager.partition_context = partition_context # Set partition_context in AIRuntimeManager

    # Load vector store from the database after starting the writer task
    vector_store = load_vector_store_from_db(embedding_model, database_manager.db_cursor)
    partition_context.vector_store = vector_store

    #Engine runtime watchdog
    watchdog = Watchdog(sys.argv[0], ai_runtime_manager)
    watchdog.start(loop)  # Pass the main event loop to the Watchdog

    # Start the input task
    asyncio.create_task(input_task(ai_runtime_manager, partition_context))
    #Keep main thread running to support the server
    await asyncio.sleep(float('inf'))

class OpenAISpecRequestHandler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, openai-organization, openai-version, openai-assistant-app-id')  # Add any other required headers
        self.end_headers()


    def do_GET(self):
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response_data = {
                "data": [
                    {
                        "id": "Adelaide-and-Albert-Model", # Use the actual ID of your model
                        "object": "model",
                        "created": 1686935002,  # Replace with a valid timestamp
                        "owned_by": "user",  # Replace with the appropriate owner
                    }
                ],
                "object": "list"
            }
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": "Not Found", "type": "invalid_request_error", "code": 404}}).encode('utf-8'))

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                request_data = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": "Invalid JSON", "type": "invalid_request_error", "code": 400}}).encode('utf-8'))
                return


            # Extract necessary information, including 'stream'
            messages = request_data.get('messages', [])
            stream = request_data.get('stream', False)  # Default to False if not provided

            # Add the task to the queue, including the 'stream' parameter
            # Use slot 0 for all server requests.  In a real implementation, you'd need
            # a way to map requests to user sessions/slots.
            ai_runtime_manager.add_task(
                (ai_runtime_manager.generate_response, (messages, 0, stream)), 0 # Pass messages directly.
            )


            # Basic response for non-streaming requests
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header('Access-Control-Allow-Origin', '*') #CORS
            self.end_headers()
            # Placeholder response; actual response handled by generate_response
            self.wfile.write(json.dumps({
                "id": "chatcmpl-placeholder",  # Use a placeholder, actual ID from task
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "Adelaide-and-Albert-Model",  # Replace
                "choices": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }).encode('utf-8'))
            return # Return from the function, the rest is handled.


        #elif self.path == "/v1/completions":
            # Implement this endpoint as a basic stub or 404.
        #  self.send_response(404)  # Or return a stub response if you prefer
        #  self.end_headers()

        # Add other endpoints with similar 404 or basic responses
        else:
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": "Not Found", "type": "invalid_request_error", "code": 404}}).encode('utf-8'))

def get_valid_port(port_str, default_port=8000):
    """
    Validates a port string and returns a valid integer port.
    Returns the default port if the input is invalid.
    """
    try:
        port = int(port_str)
        if 1024 <= port <= 65535:  # Check for valid port range (non-system ports)
            return port
        else:
            print(OutputFormatter.color_prefix(f"Invalid port number: {port_str}. Using default port {default_port}.", "ServerOpenAISpec"))
            return default_port
    except (ValueError, TypeError):
        print(OutputFormatter.color_prefix(f"Invalid port value: {port_str}. Using default port {default_port}.", "ServerOpenAISpec"))
        return default_port

def get_valid_host(host_str, default_host='0.0.0.0'):
    """
    Validates a host string (basic check for now).
    Returns the default host if the input is invalid.
    """
    # Basic validation (you might want to add more robust checks, e.g., using regex)
    if host_str and isinstance(host_str, str) and host_str.strip():
        return host_str.strip()
    else:
        print(OutputFormatter.color_prefix(f"Invalid host value: {host_str}. Using default host {default_host}.", "ServerOpenAISpec"))
        return default_host

def run_server(port=None, host=None): #Added parameters with None
    global httpd

    # --- Environment Variable Handling (Prioritized) ---
    env_port = os.environ.get('HTTP_PORT')
    env_host = os.environ.get('HTTP_HOST')

    # Use environment variables if available and valid, otherwise use defaults or provided args.
    if env_port:
        port = get_valid_port(env_port)
    elif port is None: #Check the parameter
        port = get_valid_port(None)  # Use default port

    if env_host:
        host = get_valid_host(env_host)
    elif host is None:  #Check the parameter
        host = get_valid_host(None) # Use the default.


    server_address = (host, port)
    httpd = HTTPServer(server_address, OpenAISpecRequestHandler)
    print(OutputFormatter.color_prefix(f"Starting OpenAI-compatible server on {host}:{port}...", "ServerOpenAISpec"))
    httpd.serve_forever()

def signal_handler(sig, frame):
    global httpd, ai_runtime_manager
    print(OutputFormatter.color_prefix("\nShutting down server...", "Internal"))
    if httpd:
        httpd.shutdown()
    if ai_runtime_manager:
        ai_runtime_manager.unload_model()  # Unload on Ctrl+C
    print(OutputFormatter.color_prefix("Server shut down.", "Internal"))
    sys.exit(0) #Still exit, but after the unload

if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C gracefully

    # Start the HTTP server in a separate thread.  This is crucial.
    # Now we pass *no* arguments; the environment/defaults will be used.
    server_thread = Thread(target=run_server)
    server_thread.daemon = True  #  Daemon threads exit when the main thread exits.
    server_thread.start()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print(OutputFormatter.color_prefix("\nExiting gracefully...", "Internal"))
    finally:
        if database_manager:
            database_manager.close()
        loop.close()
        print(OutputFormatter.color_prefix("Cleanup complete. Goodbye!", "Internal"))