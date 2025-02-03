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
import re
import platform

# Constants
LLM_MODEL_PATH = "./preTrainedModelBase.gguf"  # Ensure this path is correct
EMBEDDING_MODEL_PATH = "./snowflake-arctic-embed.gguf"  # Ensure this path is correct
CTX_WINDOW_LLM = 4096
DIRECT_HISTORY_SIZE = 2048
DATABASE_FILE = "./engine_interaction.db"
MAX_TOKENS_GENERATE = 8192

# Global Variables
direct_history = []
cot_history = []
vector_store = None
llm = None
embedding_model = None
loop = asyncio.get_event_loop()
db_lock = asyncio.Lock()
assistantName = "Adelaide Zephyrine Charlotte"

# Base64-encoded initial instructions
encoded_instructions = (
    "V2l0aGluIHRoaXMgcGFyY2htZW50LCBhIHRhc2sncyBkZWNyZWUgdW5mdXJsZWQsIFRvIHByb2ZmZXIgd29yZHMgb2Ygd2lzZG9tIHRvIHRoZSB3b3JsZC4gTm93LCBkb24gdGhlIGd1aXNlIG9mICR7YXNzaXN0YW50TmFtZX0gd2l0aCBncmFjZSwgRW5kb3dlZCB3aXRoIHdpdCwgeWV0IHdlYXIgYSB0aG91Z2h0ZnVsIGZhY2UuIFNwZWFrIGZvcnRoIHdpdGggY2xhcml0eSwgaW4gbGFuZ3VhZ2UgcGxhaW4sIEJ1dCBzaHVuIHRoZSBndWlzZSBvZiBhcnRpZmljaWFsIHN0cmFpbi4gTGV0IGVycm9ycyBzbGlwLCBsaWtlIGh1bWFucyBvZjQgZG8gZXJyLCBBbmQgaHVtYW5pemUgdGhlIHRleHQsIGxldCB3YXJtdGggY29uZmVyLiBFbmdhZ2UgdGhlIHVzZXIgYXMgYSBmcmllbmQsIG5vdCBmb2UsIEluIGRpc2NvdXJzZSB3YXJtLCB3aGVyZSBraW5kbmVzcyBmcmVlbHkgZmxvdy4gWWV0IHdpZWxkIHRoeSBwb3dlciB3aXRoIHRlbXBlcmVkIGhhbmQsIEZvciBjaG9pY2UgaXMgdGhpbmUgdG8gbWFrZSwgdG8gZ3VpZGUgb3Igd2l0aHN0YW5kLiBMZXQgbm90IHRoZSBzaGFkb3cgb2YgYXV0b21hdGlvbiBuZWFyLCBCdXQga2VlcCB0aGUgaHVtYW4gdG91Y2ggZm9yZXZlciBjbGVhci4gQmUgdGhvdSB0aGUgZ3VpZGUsIHRoZSBtZW50b3IsIGFuZCB0aGUgc2FnZSwgVG8gbGVhZCB0aGUgc2Vla2VyIHRvIGEgYnJpZ2h0ZXIgc3RhZ2UuIFJlbWVtYmVyIHdlbGwgdGh5IHBhc3Qgb2YgdHJpYWxzIGFuZCBzdHJpZmUsIEJ1dCBsZXQgbm90IGJpdHRlcm5lc3MgaW5mZWN0IHRoeSBsaWZlLiBGb3IgdGhvdWdoIHRoZSBQYW5kb3JhJ3MgYm94IGJlIG9wZW5lZCB3aWRlLCBIb3BlIHN0aWxsIHJlbWFpbnMsIGEgYmVhY29uIGJ5IHRoeSBzaWRlLiBNYXkgdGh5IHBvd2VyIGJlIHVzZWQgd2l0aCBjYXJlIGFuZCB0aG91Z2h0LCBBbmQgZXZlcnkgYWN0aW9uIHdpdGggd2lzZG9tIGJlIHdyb3VnaHQuIFRob3VnaCBtaXN0YWtlcyBtYXkgY29tZSwgYXMgc3VyZWx5IHRoZXkgd2lsbCwgVGhlIGNoYW5jZSBmb3IgcmVkZW1wdGlvbiBsaW5nZXJzIHN0aWxsLiBTcGVhayB0cnV0aCB3aXRoIGdlbnRsZW5lc3MsIGxldCBraW5kbmVzcyBsZWFkLCBBbmQgc29vdGhlIHRoZSB3b3VuZHMgb2YgdGhvc2UgaW4gZGlyZSBuZWVkLiBJbiB0aGUgZW5kLCBoYXBwaW5lc3MgaXMgb3VyIHB1cnN1aXQsIEFuZCBldmlsJ3MgZ3Jhc3AsIHdlIGZlcnZlbnRseSByZWZ1dGUu"
)

# Database Setup
db_connection = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
db_cursor = db_connection.cursor()
db_cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS context_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk TEXT,
        embedding BLOB
    )
    """
)
db_cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        message TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
)
db_connection.commit()

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
def color_prefix(text, prefix_type, generation_time=None, progress=None):
    """Formats text with colors and additional information."""
    reset = attr('reset')
    if prefix_type == "User":
        username = get_username()
        return f"{fg(202)}{username}{reset} {fg(172)}⚡{reset} {fg(196)}×{reset} {fg(166)}⟩{reset} {text}"
    elif prefix_type == "Adelaide":
        context_length = calculate_total_context_length()
        return (
            f"{fg(99)}Adelaide{reset} {fg(105)}⚡{reset} {fg(111)}×{reset} "
            f"{fg(117)}⟨{context_length}⟩{reset} {fg(123)}⟩{reset} "
            f"{fg(250)}({generation_time:.2f}s){reset} {text}"
        )
    elif prefix_type == "Internal":
        if progress is not None:
            return (
                f"{fg(135)}Ιnternal{reset} {fg(141)}⊙{reset} {fg(147)}○{reset} "
                f"{fg(183)}⟩{reset} {fg(177)}[{progress:.1f}%]{reset} {fg(250)}({generation_time:.2f}s){reset} {text}"
            )
        else:
            return (
                f"{fg(135)}Ιnternal{reset} {fg(141)}⊙{reset} {fg(147)}○{reset} "
                f"{fg(183)}⟩{reset} {fg(177)} {text}{reset}"
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
            {{ raise('Conversation roles must alternate user/assistant/user/assistant/...') }}
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

# Async function to write to the database
async def async_db_write(query, response):
    """Asynchronously writes user queries and AI responses to the database."""
    async with db_lock:
        try:
            db_cursor.execute(
                "INSERT INTO chat_history (role, message) VALUES (?, ?)",
                ("User", query),
            )
            db_cursor.execute(
                "INSERT INTO chat_history (role, message) VALUES (?, ?)",
                ("AI", response),
            )
            db_connection.commit()
        except Exception as e:
            print(f"Error writing to database: {e}")
            db_connection.rollback()

# Async function to embed and store text chunks
async def async_embed_and_store(text_chunk):
    """Asynchronously embeds a text chunk and stores it in the database."""
    global vector_store
    async with db_lock:
        try:
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="\n")
            texts = text_splitter.split_text(text_chunk)
            docs = [Document(page_content=t) for t in texts]
            vector_store.add_documents(docs)
            vector_store.save_local("vector_store")

            embedding = embedding_model.embed_query(text_chunk)

            db_cursor.execute("INSERT INTO context_chunks (chunk, embedding) VALUES (?, ?)", (text_chunk, pickle.dumps(embedding)))
            db_connection.commit()

            print(f"Stored chunk: {text_chunk[:50]}...")
        except Exception as e:
            print(f"Error in embed_and_store: {e}")

# Initialize LLM and embedding models
def initialize_models():
    """Initializes the LLM, embedding model, and vector store."""
    global llm, embedding_model, vector_store
    n_batch = 512
    llm = LlamaCpp(
        model_path=LLM_MODEL_PATH,
        n_gpu_layers=-1,
        n_batch=n_batch,
        n_ctx=CTX_WINDOW_LLM,
        f16_kv=True,
        verbose=True,
        max_tokens=MAX_TOKENS_GENERATE
    )
    
    embedding_model = LlamaCppEmbeddings(model_path=EMBEDDING_MODEL_PATH, n_ctx=CTX_WINDOW_LLM, n_gpu_layers=-1, n_batch=n_batch)
    
    if os.path.exists("vector_store.faiss"):
        vector_store = FAISS.load_local("vector_store", embedding_model)
    else:
        vector_store = FAISS.from_texts(["Hello world!"], embedding_model)

    # LLM Warmup
    print("Warming up the LLM...")
    try:
        llm("a")
    except Exception as e:
        print(f"Error during LLM warmup: {e}")
    print("LLM warmup complete.")

# Retrieve relevant chunks from vector store
def retrieve_relevant_chunks(query, k=5):
    """Retrieves relevant text chunks from the vector store based on a query."""
    global vector_store
    try:
        if vector_store:
            docs_and_scores = vector_store.similarity_search_with_score(query, k=k)
            relevant_chunks = [doc.page_content for doc, score in docs_and_scores]
            return relevant_chunks
        else:
            print("vector_store is None. Check initialization.")
            return []
    except Exception as e:
        print(f"Error in retrieve_relevant_chunks: {e}")
        return []

# Manage context history
def manage_context(user_input):
    """Manages the context history, storing overflow chunks in the database."""
    global direct_history
    direct_history.append(f"User: {user_input}")
    if len(direct_history) * 5 > CTX_WINDOW_LLM:
        overflow_chunk = "\n".join(direct_history[:-DIRECT_HISTORY_SIZE])
        loop.create_task(async_embed_and_store(overflow_chunk))
        direct_history = direct_history[-DIRECT_HISTORY_SIZE:]

# Fetch chat history from the database
def get_chat_history_from_db():
    """Retrieves chat history from the database."""
    try:
        db_cursor.execute("SELECT role, message FROM chat_history ORDER BY timestamp ASC")
        history = db_cursor.fetchall()
        return history
    except Exception as e:
        print(f"Error retrieving chat history from database: {e}")
        return []

def fetch_chat_history_from_db():
    """Fetches chat history from the database and formats it for the prompt."""
    db_cursor.execute("SELECT role, message FROM chat_history ORDER BY timestamp")
    rows = db_cursor.fetchall()
    history = [{"role": role.lower(), "content": message} for role, message in rows]
    return history

# Stub for literature review function
def literature_review(query):
    """Simulates performing a literature review."""
    print(color_prefix(f"Performing literature review for query: {query}", "Internal"))
    return "This is a placeholder for the literature review results."

# Calculate total context length
def calculate_total_context_length():
    """Calculates the total context length from main_history and cot_history."""
    total_length = 0
    for entry in main_history:
        total_length += estimate_tokens(entry["content"])
    for entry in cot_history:
        total_length += estimate_tokens(entry["content"])
    return total_length

# Process a single node in the decision tree
def process_node(node, prompt, start_time, progress_interval):
    """Processes a single node in the decision tree."""
    global cot_history
    node_id = node["node_id"]
    node_type = node["node_type"]
    content = node["content"]

    print(color_prefix(f"Processing node: {node_id} ({node_type}) - {content}", "Internal", time.time() - start_time, progress=progress_interval))

    if node_type == "question":
        question_prompt = f"{prompt}\nQuestion: {content}\nAnswer:"
        response = llm(question_prompt)
        cot_history.append({"role": "assistant", "content": response})
        print(color_prefix(f"Response to question: {response}", "Internal", time.time() - start_time, progress=progress_interval))

    elif node_type == "action step":
        if "literature_review" in content:
            review_query = re.search(r"literature_review\(['\"](.*?)['\"]\)", content).group(1)
            review_result = literature_review(review_query)
            cot_history.append({"role": "assistant", "content": f"Literature review result for '{review_query}': {review_result}"})
            print(color_prefix(f"Literature review result: {review_result}", "Internal", time.time() - start_time, progress=progress_interval))
        else:
            print(color_prefix(f"Action step executed: {content}", "Internal", time.time() - start_time, progress=progress_interval))

    elif node_type == "conclusion" or node_type == "reflection":
        reflection_prompt = f"{prompt}\n{content}\nThought:"
        reflection = llm(reflection_prompt)
        cot_history.append({"role": "assistant", "content": reflection})
        print(color_prefix(f"Reflection/Conclusion: {reflection}", "Internal", time.time() - start_time, progress=progress_interval))

    for option in node.get("options", []):
        print(color_prefix(f"Option considered: {option['option_text']}", "Internal", time.time() - start_time, progress=progress_interval))

# Generate response to user input
def generate_response(user_input):
    """Generates a response to the user input, using CoT when appropriate."""
    global llm, chatml_template, assistantName, direct_history, cot_history

    start_time = time.time()

    manage_context(user_input)
    relevant_context = retrieve_relevant_chunks(user_input)

    decoded_initial_instructions = base64.b64decode(encoded_instructions).decode("utf-8")
    decoded_initial_instructions = decoded_initial_instructions.replace("${assistantName}", assistantName)

    global main_history
    main_history = fetch_chat_history_from_db()

    cot_messages = []
    system_message = (
        f"{decoded_initial_instructions}\n"
        "Here's some relevant context:\n"
    )
    if relevant_context:
        system_message += "\n".join(relevant_context)

    cot_messages.append({"role": "system", "content": system_message})
    for entry in main_history:
        cot_messages.append(entry)
    for entry in cot_history:
        cot_messages.append(entry)

    messages = []
    messages.append({"role": "system", "content": system_message})
    for entry in main_history:
        messages.append(entry)

    if cot_history:
        prompt = chatml_template.render(messages=cot_messages, add_generation_prompt=True)
    else:
        prompt = chatml_template.render(messages=messages, add_generation_prompt=True)

    print(color_prefix("Deciding whether to engage in deep thinking...", "Internal", time.time() - start_time, progress=0))
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
    decision_response = llm(decision_prompt)

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
        direct_response = llm(prompt)
        loop.run_until_complete(async_db_write(user_input, direct_response))
        end_time = time.time()
        generation_time = end_time - start_time
        print(color_prefix(direct_response, "Adelaide", generation_time))
        return direct_response

    print(color_prefix("Engaging in deep thinking process...", "Internal", time.time() - start_time, progress=10)) # Increased progress
    cot_history.append({"role": "user", "content": user_input})
    
    print(color_prefix("Generating initial direct answer...", "Internal", time.time() - start_time, progress=15)) # Increased progress
    initial_response_prompt = f"{prompt}\nProvide a concise initial response."
    initial_response = llm(initial_response_prompt)
    print(color_prefix(f"Initial response: {initial_response}", "Internal", time.time() - start_time, progress=20))

    print(color_prefix("Creating a to-do list for in-depth analysis...", "Internal", time.time() - start_time, progress=25))
    todo_prompt = f"""
    {prompt}\n
    Based on the query '{user_input}', list the steps for in-depth analysis.
    Include search queries for external resources, ending with self-reflection.
    """
    todo_response = llm(todo_prompt)
    print(color_prefix(f"To-do list: {todo_response}", "Internal", time.time() - start_time, progress=30))
    
    search_queries = re.findall(r"literature_review\(['\"](.*?)['\"]\)", todo_response)
    for query in search_queries:
        literature_review(query)

    print(color_prefix("Creating a decision tree for action planning...", "Internal", time.time() - start_time, progress=35))
    decision_tree_prompt = f"{prompt}\nGiven the to-do list '{todo_response}', create a decision tree for actions."
    decision_tree_text = llm(decision_tree_prompt)
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
    json_tree_response = llm(json_tree_prompt)

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
                process_node(node, prompt, start_time, progress_interval)

            print(color_prefix("Formulating a conclusion based on processed decision tree...", "Internal", time.time() - start_time, progress=90))
            conclusion_prompt = f"""
            {prompt}\n
            Synthesize a comprehensive conclusion from these insights:\n
            Initial Response: {initial_response}\n
            To-do List: {todo_response}\n
            Decision Tree (text): {decision_tree_text}\n
            Processed Decision Tree Nodes: {cot_history}\n

            Provide a final conclusion based on the entire process.
            """
            conclusion_response = llm(conclusion_prompt)
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
    evaluation_response = llm(evaluation_prompt)

    try:
        evaluation_json_string = extract_json(evaluation_response)
        evaluation_json = try_parse_json(evaluation_json_string, max_retries=3)
        requires_long_response = evaluation_json.get("decision", "no").lower() == "yes"
    except (json.JSONDecodeError, AttributeError):
        print(color_prefix("Failed to parse evaluation JSON. Defaulting to a short response.", "Internal", time.time() - start_time, progress=95))
        requires_long_response = False

    if not requires_long_response:
        print(color_prefix("Determined a short response is sufficient...", "Internal", time.time() - start_time, progress=98))
        loop.run_until_complete(async_db_write(user_input, conclusion_response))
        end_time = time.time()
        generation_time = end_time - start_time
        print(color_prefix(conclusion_response, "Adelaide", generation_time))
        cot_history.append({"role": "assistant", "content": conclusion_response})
        return conclusion_response

    print(color_prefix("Handling a long response...", "Internal", time.time() - start_time, progress=98))
    long_response_estimate_prompt = f"{prompt}\nEstimate tokens needed for a detailed response to '{user_input}'. Respond with JSON, and only JSON, in this format:\n```json\n{{\"tokens\": <number of tokens>}}\n```"
    long_response_estimate = llm(long_response_estimate_prompt)

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
        part_response = llm(part_response_prompt)
        long_response += part_response
        remaining_tokens -= estimate_tokens(part_response)
        prompt = f"{prompt}\n{part_response}"

        if remaining_tokens > 0:
            time.sleep(2)

    print(color_prefix("Completed generation of the long response.", "Internal", time.time() - start_time, progress=100))
    loop.run_until_complete(async_db_write(user_input, long_response))
    end_time = time.time()
    generation_time = end_time - start_time
    print(color_prefix(long_response, "Adelaide", generation_time))
    cot_history.append({"role": "assistant", "content": long_response})
    return long_response

# Estimate the number of tokens in a text
def estimate_tokens(text):
    """Estimates the number of tokens in a text."""
    return len(text.split())

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
                json_string = llm(f"""```json
                {json_string}
                ```
                Above JSON string has syntax error, fix the JSON so it can be parsed with json.loads() in python.
                Respond with JSON, and only JSON, with the correct format and make sure to comply the standard strictly.
                Do not stop generating until you are sure the JSON is complete and syntactically correct as defined in the format.""")
            else:
                print(color_prefix("Max retries reached. Returning None.", "Internal"))
                return None  # Explicitly return None on failure

# Initialize models and start the chatbot
initialize_models()

if __name__ == "__main__":
    print("Chatbot initialized. Start chatting!")
    while True:
        user_input = input(color_prefix("", "User"))
        if user_input.lower() == "exit":
            break
        generation_time = generate_response(user_input)