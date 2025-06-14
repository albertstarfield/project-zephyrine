# agent.py

import asyncio
import time
import json
import xml.etree.ElementTree as ET
import re
import os
import sys
import subprocess # For executing commands
import shlex # For safely splitting command strings
from loguru import logger
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional, Tuple, Union

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Assuming these are in your config.py and accessible
# Import necessary constants from CortexConfiguration (ensure they are defined there)
try:
    from CortexConfiguration import PROVIDER, TOPCAP_TOKENS, RAG_HISTORY_COUNT
except ImportError:
    # Fallbacks if config import fails during agent initialization (less ideal)
    logger.error("Failed to import config constants in agent.py")
    RAG_HISTORY_COUNT = 5

# Import necessary DB functions and session factory
try:
    from database import add_interaction, get_recent_interactions, SessionLocal, Interaction # Import Interaction model
except ImportError:
    logger.error("Failed to import database components in agent.py")
    # Define dummy functions/classes if needed for basic loading, but app will likely fail
    def add_interaction(*args, **kwargs): 
        pass
    def get_recent_interactions(*args, **kwargs): 
        return []
    class SessionLocal: 
        def __call__(self): 
            return None # type: ignore
    class Interaction: 
        pass


# --- NEW: Import the custom lock ---

from priority_lock import ELP0, ELP1 # Ensure these are imported
interruption_error_marker = "Worker task interrupted by higher priority request" # Define consistently

# --- Define the path to the system prompt file ---
AGENT_PROMPT_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_system_prompt.txt")


# Implementation for Agent tools using basic os/subprocess
class AgentTools:
    """Agent tool execution implementation using basic system calls."""
    def __init__(self, cwd: str, provider_embeddings: Any):
        self.cwd = cwd
        self.provider_embeddings = provider_embeddings
        logger.info(f"🛠️ Initializing AgentTools with CWD: {self.cwd}")
        # NOTE: These implementations are basic. Real-world tools need more robust error handling,
        # security checks (especially for execute_command), and platform compatibility checks.

    async def execute_command(self, command: str, requires_approval: bool) -> str:
        """Executes a shell command."""
        logger.info(f"Executing command: '{command}' (Approval: {requires_approval}) in CWD: {self.cwd}")
        # Security Note: Avoid shell=True if possible. Split command string safely.
        # This basic split might not handle complex shell syntax (pipes, redirection, etc.)
        # If complex commands are needed, consider a more robust parser or use shell=True with extreme caution.
        try:
            # Use shlex to split command safely for different OS
            if sys.platform == 'win32':
                cmd_list = command # Windows might need the command string directly for some builtins
                use_shell = True # Often needed for builtins like 'help' or 'dir' on Windows
            else:
                cmd_list = shlex.split(command)
                use_shell = False # Safer on Unix-like systems

            loop = asyncio.get_running_loop()
            process = await loop.run_in_executor(
                None, # Use default ThreadPoolExecutor
                subprocess.run,
                cmd_list, # Pass list or string depending on OS/shell
                shell=use_shell, # Set based on platform need
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=120 # Timeout for command execution
            )

            stdout = process.stdout.strip()
            stderr = process.stderr.strip()
            returncode = process.returncode

            logger.debug(f"Command '{command}' finished RC: {returncode}")
            output = f"Command: {command}\nReturn Code: {returncode}\n"
            if stdout:
                output += f"STDOUT:\n{stdout}\n"
            if stderr:
                output += f"STDERR:\n{stderr}\n"

            if returncode != 0:
                 logger.error(f"Command failed with return code {returncode}: {command}")
                 return f"Error executing command (Return Code {returncode}):\n{output}" # Indicate failure to the LLM

            return output # Return combined output

        except FileNotFoundError:
            logger.error(f"Command not found: {command}")
            return f"Error executing command: Command or executable not found."
        except subprocess.TimeoutExpired:
             logger.error(f"Command timed out after 120 seconds: {command}")
             return f"Error executing command: Command timed out after 120 seconds."
        except Exception as e:
            logger.error(f"Unexpected error during command execution: {e}")
            logger.exception("Command Execution Traceback:")
            return f"Error executing command: {e}"


    async def read_file(self, path: str) -> str:
        """Reads the content of a file."""
        full_path = os.path.join(self.cwd, path)
        logger.info(f"Reading file: '{full_path}'")
        loop = asyncio.get_running_loop()
        try:
            # Use to_thread for blocking file read
            content = await loop.run_in_executor(
                None,
                lambda: open(full_path, 'r', encoding='utf-8', errors='ignore').read() # Added errors='ignore'
            )
            return f"File content of '{path}':\n```\n{content}\n```"
        except FileNotFoundError:
            logger.warning(f"File not found: {full_path}")
            return f"Error: File not found at '{path}'."
        except IsADirectoryError:
             logger.warning(f"Path is directory: {full_path}")
             return f"Error: Path '{path}' is a directory, not a file."
        except Exception as e:
            logger.error(f"Error reading file '{full_path}': {e}")
            return f"Error reading file '{path}': {e}"

    async def write_to_file(self, path: str, content: str) -> str:
        """Writes content to a file, overwriting or creating it."""
        full_path = os.path.join(self.cwd, path)
        logger.info(f"Writing to file: '{full_path}' ({len(content)} bytes)")
        loop = asyncio.get_running_loop()
        try:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(full_path)
            if parent_dir: # Only create if not writing to root CWD
                await loop.run_in_executor(None, os.makedirs, parent_dir, exist_ok=True)

            # Write file content in a thread
            await loop.run_in_executor(
                None,
                lambda: open(full_path, 'w', encoding='utf-8').write(content)
            )
            return f"Successfully wrote content to '{path}'."
        except Exception as e:
            logger.error(f"Error writing to file '{full_path}': {e}")
            return f"Error writing to file '{path}': {e}"

    async def list_files(self, path: str, recursive: bool = False) -> str:
        """Lists files and directories."""
        # Ensure path is relative to cwd for safety, or handle absolute paths carefully
        if os.path.isabs(path):
            logger.warning(f"Attempted to list absolute path: {path}. Restricting to CWD.")
            try:
                path = os.path.relpath(path, self.cwd)
                if path.startswith(".."): # Basic check to prevent going above CWD
                     return "Error: Cannot list files outside the working directory."
            except ValueError:
                 return "Error: Cannot list files outside the working directory."

        full_path = os.path.join(self.cwd, path)
        logger.info(f"Listing files in: '{full_path}' (recursive: {recursive})")
        loop = asyncio.get_running_loop()
        try:
             items = []
             if recursive:
                 # os.walk is blocking, run in thread
                 for root, dirs, files in await loop.run_in_executor(None, os.walk, full_path):
                      # Make paths relative to the original requested path for clarity
                      relative_root = os.path.relpath(root, full_path)
                      if relative_root == '.':
                          relative_root = "" # Avoid './' prefix
                      items.extend(os.path.join(relative_root, d) + '/' for d in dirs)
                      items.extend(os.path.join(relative_root, f) for f in files)
             else:
                 # os.listdir is blocking, run in thread
                 raw_items = await loop.run_in_executor(None, os.listdir, full_path)
                 items_with_type = []
                 for item in raw_items:
                     item_full_path = os.path.join(full_path, item)
                     # Check if it's a directory within the thread to avoid blocking loop
                     is_dir = await loop.run_in_executor(None, os.path.isdir, item_full_path)
                     items_with_type.append(item + '/' if is_dir else item)
                 items = items_with_type # Assign back to items

             formatted_list = "\n".join(sorted(items)) # Sort alphabetically for consistency
             if not formatted_list:
                 return f"Directory '{path}' is empty."
             return f"Contents of '{path}':\n{formatted_list}"
        except FileNotFoundError:
             logger.warning(f"Directory not found for listing: {full_path}")
             return f"Error: Directory not found at '{path}'."
        except NotADirectoryError:
             logger.warning(f"Path is not a directory for listing: {full_path}")
             return f"Error: Path '{path}' is not a directory."
        except Exception as e:
             logger.error(f"Error listing files in '{full_path}': {e}")
             return f"Error listing files in '{path}': {e}"

    # --- Placeholder Tools ---
    async def replace_in_file(self, path: str, diff: str) -> str:
        logger.warning(f"Replacing in file (placeholder): '{path}'")
        await asyncio.sleep(0.1)
        return f"Placeholder: Would replace content in '{path}' based on diff."

    async def search_files(self, path: str, regex: str, file_pattern: Optional[str] = None) -> str:
        logger.warning(f"Searching files (placeholder): '{path}' regex '{regex}' pattern '{file_pattern}'")
        await asyncio.sleep(0.1)
        return f"Placeholder: Would search files in '{path}' for regex '{regex}'."

    async def list_code_definition_names(self, path: str) -> str:
         logger.warning(f"Listing code definitions (placeholder) in '{path}'")
         await asyncio.sleep(0.1)
         return f"Placeholder: Would list definitions in '{path}'."

    async def browser_action(self, action: str, url: Optional[str] = None, coordinate: Optional[str] = None, text: Optional[str] = None) -> str:
        logger.warning(f"Browser action (placeholder): {action}")
        await asyncio.sleep(0.1)
        return f"Placeholder: Would perform browser action '{action}'."

    async def use_mcp_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
         logger.warning(f"MCP tool (placeholder): {server_name}/{tool_name}")
         await asyncio.sleep(0.1)
         return f"Placeholder: Would execute MCP tool '{tool_name}'."

    async def access_mcp_resource(self, server_name: str, uri: str) -> str:
         logger.warning(f"MCP resource (placeholder): {server_name}/{uri}")
         await asyncio.sleep(0.1)
         return f"Placeholder: Would access MCP resource '{uri}'."


class AmaryllisAgent:
    """Manages the state and execution loop for the Agent persona (Adelaide)."""
    def __init__(self, provider: Any, cwd: str, supports_computer_use: bool = True):
        self.provider = provider
        self.cwd = cwd
        self.supports_computer_use = supports_computer_use
        class MockMcpHub:
            def getServers(self):
                return []
        self.mcp_hub = MockMcpHub()
        class MockBrowserSettings:
            class Viewport:
                width = 1024
                height = 768
            viewport = Viewport()
        self.browser_settings = MockBrowserSettings()
        self.agent_tools = AgentTools(self.cwd, self.provider.embeddings)
        self.setup_prompts()
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_session_id: Optional[str] = None

    def setup_prompts(self):
        """Reads and formats the agent system prompt."""
        try:
            with open(AGENT_PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
                raw_agent_system_prompt = f.read()
            logger.info(f"📖 Read agent prompt from: {AGENT_PROMPT_FILE_PATH}")
        except Exception as e:
            logger.critical(f"❌ Failed reading agent prompt: {e}")
            raw_agent_system_prompt = "You are Adelaide, a helpful AI agent. Respond to the user's requests."

        def osName(): return sys.platform
        def getShell(): return os.environ.get("SHELL", "/bin/sh")
        class PosixPath:
            def __init__(self, path): self._path = path
            def toPosix(self): return self._path.replace("\\", "/")
        cwd_posix = PosixPath(self.cwd)

        def format_user_custom_instructions(**kwargs) -> str:
            settingsCustomInstructions = kwargs.get("settingsCustomInstructions", "Be helpful and professional.")
            customInstructions = f"{settingsCustomInstructions}\n"
            # Add logic here to load other instructions if needed
            return f"====\nUSER'S CUSTOM INSTRUCTIONS\n\n{customInstructions.strip()}\n====" if customInstructions.strip() else ""

        user_custom_instructions_content = format_user_custom_instructions(settingsCustomInstructions="Focus on Python code.")

        # Format the base prompt, replacing placeholders carefully
        formatted_base_prompt_content = raw_agent_system_prompt
        replacements = {
            '${osName()}': osName(),
            '${getShell()}': getShell(),
            '${os.homedir().toPosix()}': PosixPath(os.path.expanduser("~")).toPosix(),
            '${cwd.toPosix()}': cwd_posix.toPosix(),
            '${browserSettings.viewport.width}': str(self.browser_settings.viewport.width),
            '${browserSettings.viewport.height}': str(self.browser_settings.viewport.height),
            # Simplify complex JS conditionals during replacement
            '${\n\tmcpHub.getServers().length > 0\n\t\t? `${mcpHub...` : "(No MCP servers currently connected)"\n}': "(No MCP servers currently connected)",
            '${\n\tsupportsComputerUse\n\t\t? `\n\n## browser_action...`\n\t\t: ""\n}': "\n\n## browser_action\nDescription: (Browser Interaction Tool)\n..." if self.supports_computer_use else "",
            '${\n\tsupportsComputerUse\n\t\t? "\\n- You can use the browser_action tool..."\n\t\t: ""\n}': "\n- Browser interaction is enabled." if self.supports_computer_use else "",
            '${\n\tsupportsComputerUse\n\t\t? `\\n- The user may ask generic non-development tasks...`\n\t\t: ""\n}': "\n- Generic web tasks via browser are possible." if self.supports_computer_use else "",
            '${\n\tsupportsComputerUse\n\t\t? " Then if you want to test your work..."\n\t\t: ""\n}': " Browser testing is available." if self.supports_computer_use else ""
        }
        for placeholder, value in replacements.items():
            try:
                 # Use regex for safer replacement of complex multi-line placeholders
                 # This requires careful escaping of special characters in the placeholder key
                 escaped_placeholder = re.escape(placeholder)
                 formatted_base_prompt_content = re.sub(escaped_placeholder, value, formatted_base_prompt_content, flags=re.DOTALL | re.MULTILINE)
            except Exception as replace_err:
                 logger.error(f"Error replacing placeholder '{placeholder[:50]}...': {replace_err}")


        final_system_prompt_template_string = f"{formatted_base_prompt_content.strip()}\n{user_custom_instructions_content.strip()}\n====\nCURRENT ENVIRONMENT DETAILS\n# Mode: {{mode}}\nFile list:\n{{file_list}}\nActively Running Terminals:\n{{running_terminals}}\n====\nCONVERSATION HISTORY SNIPPETS (RAG)\n{{agent_history_rag}}\n====\nRAG CONTEXT FROM DOCUMENTS/URLS\n{{url_rag_context}}\n====\n"

        self.agent_prompt_template = ChatPromptTemplate.from_messages([ SystemMessage(content=final_system_prompt_template_string), MessagesPlaceholder(variable_name="agent_history_turns"), HumanMessage(content="{current_input}"),])

        agent_model_role = "default" # Or "router" if you want it to use the same as the corrector
        agent_llm = self.provider.get_model(agent_model_role)
        if not agent_llm:
             # Handle error: The required model wasn't initialized in CortexEngine
             logger.error(f"Agent setup failed: Could not get model for role '{agent_model_role}' from CortexEngine.")
             # You might want to raise an exception here to stop initialization
             raise ValueError(f"Agent model role '{agent_model_role}' not found in CortexEngine.")

        self.agent_chain = self.agent_prompt_template | agent_llm | StrOutputParser()


    def _parse_tool_request(self, llm_response: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parses the LLM's response to find tool requests formatted as XML."""
        # Use regex to find the outermost tool tag first
        outer_match = re.search(r'<(\w+?)>(.*?)</\1>', llm_response, re.DOTALL | re.IGNORECASE)
        if not outer_match:
            logger.debug("No valid XML tool request found in LLM response.")
            return None, None

        tool_name = outer_match.group(1)
        tool_content = outer_match.group(2)
        parameters: Dict[str, Any] = {}

        logger.info(f"Detected tool request: <{tool_name}>")

        # Parse parameters within the tool content using XML parsing
        try:
            # Wrap content in a root tag to handle potential multiple params
            xml_content = f"<root>{tool_content}</root>"
            root = ET.fromstring(xml_content)
            for child in root:
                # Store parameter value, handling empty tags
                parameters[child.tag] = child.text.strip() if child.text else ""
                # Log each parameter found
                logger.trace(f"  Parsed param: <{child.tag}>{parameters[child.tag]}</{child.tag}>")

            # Special handling for 'arguments' which should be JSON string
            if 'arguments' in parameters and isinstance(parameters['arguments'], str):
                 try:
                     parameters['arguments'] = json.loads(parameters['arguments'])
                     logger.trace("  Parsed 'arguments' parameter as JSON.")
                 except json.JSONDecodeError:
                     logger.error(f"Failed JSON parse for 'arguments'. Value: {parameters['arguments']}")
                     parameters['arguments'] = f"Invalid JSON: {parameters['arguments']}" # Indicate failure

        except ET.ParseError as e:
             logger.error(f"Failed XML parse for tool params '{tool_name}': {e}. Content: {tool_content}")
             # Fallback: try simple regex for parameters if XML parsing fails
             param_matches = re.findall(r'<(\w+?)>(.*?)</\1>', tool_content, re.DOTALL)
             parameters = {name: value.strip() for name, value in param_matches}
             logger.warning(f"Attempted regex fallback for parameters: {parameters}")

        logger.debug(f"Final parsed parameters for tool '{tool_name}': {parameters}")
        return tool_name, parameters


    async def _execute_agent_tool(self, tool_name: str, parameters: Dict[str, Any], db: Session) -> str:
        """Executes the requested agent tool using the AgentTools instance."""
        logger.warning(f"🛠️ Executing tool: {tool_name}")
        # Log the request before execution
        tool_request_output_log = f"<{tool_name}>\n"
        for key, value in parameters.items():
            tool_request_output_log += f"  <{key}>{value}</{key}>\n"
        tool_request_output_log += f"</{tool_name}>"
        add_interaction(
            db, session_id=self.current_session_id, mode="agent", input_type="tool_request",
            user_input=f"[Agent requested tool '{tool_name}']",
            llm_response=tool_request_output_log, # Log the formatted request
            tool_name=tool_name, tool_parameters=json.dumps(parameters) # Store raw params as JSON
        )

        result = f"Tool '{tool_name}' execution failed or encountered an issue." # Default result
        start_time = time.monotonic()
        try:
            tool_method = getattr(self.agent_tools, tool_name, None)
            if tool_method and callable(tool_method):
                 logger.debug(f"Calling tool method: agent_tools.{tool_name}(**{parameters})")
                 result = await tool_method(**parameters) # Call the tool method
            elif tool_name in ["attempt_completion", "plan_mode_respond", "ask_followup_question", "new_task", "load_mcp_documentation"]:
                 result = f"Error: LLM attempted to 'execute' output type tool '{tool_name}'. This indicates the LLM should have stopped."
                 logger.warning(f"LLM requested output tool '{tool_name}'.")
            else:
                result = f"Error: Unknown tool requested: '{tool_name}'."
                logger.error(result)

        except TypeError as te:
            # Log specific error about mismatched arguments
            result = f"Error executing tool '{tool_name}': Invalid parameters provided. {te}. Check tool usage format in prompt."
            logger.error(result)
            logger.exception(f"TypeError during tool execution ({tool_name}):")
        except Exception as e:
            result = f"Error executing tool '{tool_name}': {e}"
            logger.error(result)
            logger.exception(f"Traceback for tool execution error ({tool_name}):")

        execution_time_ms = (time.monotonic() - start_time) * 1000
        logger.info(f"Tool '{tool_name}' execution finished ({execution_time_ms:.2f} ms).")
        logger.debug(f"Tool Result Snippet:\n{result[:500]}...") # Log result snippet

        # Log the tool result interaction *after* execution
        add_interaction(
            db, session_id=self.current_session_id, mode="agent", input_type="tool_result",
            user_input=f"[Result for '{tool_name}']",
            llm_response=result, # Store the actual tool output
            tool_name=tool_name, execution_time_ms=execution_time_ms
        )

        # Format the result for the LLM to consume in the next turn
        formatted_result_for_llm = f"\n<tool_result>\n<{tool_name}>\n{result}\n</{tool_name}>\n</tool_result>\n"
        return formatted_result_for_llm


    def _get_environment_details(self) -> Dict[str, str]:
        """Gets environment details as a dictionary (blocking calls)."""
        file_list_str = "Error: Could not list files."
        running_terminals_str = "Error: Could not retrieve running terminals."
        current_mode = "ACT MODE" # Assume ACT mode for background tasks

        try:
             files = os.listdir(self.cwd)
             file_list_str = "\n".join(f"- {f}" for f in sorted(files)[:50]) # Limit list size
             if len(files) > 50:
                 file_list_str += "\n- ..."
             if not files:
                 file_list_str = "(Directory is empty)"
        except Exception as e:
             logger.warning(f"Could not get file list for env details: {e}")
             file_list_str = f"(Error getting file list: {e})"

        # Placeholder for running terminals - requires OS-specific logic
        running_terminals_str = "(Not implemented)"

        return {
            "file_list": file_list_str,
            "running_terminals": running_terminals_str,
            "mode": current_mode,
        }


    def _get_agent_history_rag_string(self, db: Session) -> str:
        """Retrieves and formats recent agent interactions as a string for prompt."""
        recent_interactions = get_recent_interactions(db, limit=RAG_HISTORY_COUNT * 2, session_id=self.current_session_id, mode="agent", include_logs=True) # Include logs for agent context
        if not recent_interactions:
            return "No recent agent history."

        history_str_parts = []
        recent_interactions.reverse() # Oldest first for prompt context flow
        for interaction in recent_interactions:
             prefix, text = None, None
             formatted_result = None
             input_type = interaction.input_type or "log" # Default to log if None

             if input_type == 'text' and interaction.user_input:
                 prefix, text = "User Input:", interaction.user_input
             elif input_type == 'llm_response' and interaction.llm_response:
                 prefix, text = "Adelaide Response:", interaction.llm_response
             elif input_type == 'tool_request' and interaction.llm_response:
                 # Show the raw request XML the LLM generated
                 prefix, text = "Adelaide Tool Request:", interaction.llm_response
             elif input_type == 'tool_result' and interaction.llm_response:
                 # Format the tool result as the environment's response
                 tool_name = interaction.tool_name or "unknown_tool"
                 result_content = interaction.llm_response or "[Empty Result]"
                 # Use the specific XML format expected by the prompt
                 formatted_result = f"<tool_result>\n<{tool_name}>\n{result_content}\n</{tool_name}>\n</tool_result>"
             elif input_type == 'system':
                 prefix, text = "System Message:", interaction.user_input
             elif input_type.startswith('log_') or input_type == 'error':
                 prefix = f"Log ({input_type.upper()}):"
                 text = interaction.llm_response or interaction.user_input # Log message stored here

             if formatted_result:
                 history_str_parts.append(formatted_result)
             elif prefix and text:
                 # Truncate long history items
                 text_snippet = (text[:300] + '...') if len(text) > 300 else text
                 history_str_parts.append(f"{prefix}\n{text_snippet}")

        return "\n---\n".join(history_str_parts) if history_str_parts else "No recent agent history."


    def _get_agent_url_rag_string(self, db: Session, user_input: str) -> str:
        """Agent relies on tools for URL data, returns placeholder."""
        # If URL vector store is available globally (e.g., shared instance or loaded by agent),
        # could implement RAG here. For now, stick to tool-based access.
        return "URL context not actively used by Agent. Use 'read_file' or other tools if needed."


    async def _run_task_in_background(self, initial_interaction_id: int, user_input: str, session_id: str):
        """
        Runs the Agent's task execution loop in the background with ELP0 priority.
        Handles interruptions signaled by the CortexEngine.
        """
        logger.warning(f"🧑‍💻🧵 Starting Agent background task ID: {initial_interaction_id} (Priority: ELP0)")
        db: Optional[Session] = None # Initialize db session variable
        initial_interaction: Optional[Interaction] = None # Initialize interaction variable
        # --- Define the marker string for interruption ---
        interruption_error_marker = "Worker task interrupted by higher priority request"
        # ---

        try:
            db = SessionLocal() # Get a new DB session for this background task
            if not db:
                 logger.critical(f"❌ Agent BG {initial_interaction_id}: Failed to get DB session.")
                 return # Cannot proceed without DB

            self.current_session_id = session_id

            # Load the initial interaction record associated with this task
            initial_interaction = db.query(Interaction).filter(Interaction.id == initial_interaction_id).first()
            if not initial_interaction:
                logger.error(f"❌ Agent BG: Cannot find initial Interaction ID {initial_interaction_id}.")
                return # Cannot proceed without the initial record

            max_turns = 10 # Limit agent loops to prevent runaways
            turn_count = 0
            current_input_for_llm = user_input # Start with the user's initial input

            while turn_count < max_turns:
                turn_count += 1
                logger.info(f"Agent Task {initial_interaction_id}: Turn {turn_count}")
                logger.debug(f"Agent Turn {turn_count} Input Snippet:\n{current_input_for_llm[:500]}...")

                # --- Check for stop event before proceeding (optional but good practice) ---
                # if stop_event_is_set(): logger.info(...); break

                # --- Get current environment details and RAG context (sync DB calls) ---
                # These are relatively quick and less likely to be interrupted targets
                env_details_dict = self._get_environment_details()
                agent_history_rag_string = self._get_agent_history_rag_string(db)
                url_rag_context_string = self._get_agent_url_rag_string(db, user_input)

                # --- Build the message list for the prompt's MessagesPlaceholder ---
                agent_history_turns_messages: List[Union[HumanMessage, AIMessage]] = []
                # Fetch the relevant history for this specific task turn from DB
                recent_interactions_for_turns = db.query(Interaction).filter(
                    Interaction.session_id == self.current_session_id,
                    Interaction.mode == "agent",
                    Interaction.id >= initial_interaction_id # Only turns related to this task
                ).order_by(Interaction.timestamp.asc()).all()

                for interaction in recent_interactions_for_turns:
                    # Format turns for the LLM history placeholder
                    if interaction.input_type == 'text' and interaction.user_input and interaction.id == initial_interaction_id:
                        agent_history_turns_messages.append(HumanMessage(content=interaction.user_input))
                    elif interaction.input_type == 'llm_response' and interaction.llm_response:
                        agent_history_turns_messages.append(AIMessage(content=interaction.llm_response))
                    elif interaction.input_type == 'tool_request' and interaction.llm_response:
                         agent_history_turns_messages.append(AIMessage(content=interaction.llm_response))
                    elif interaction.input_type == 'tool_result' and interaction.llm_response:
                         tool_name = interaction.tool_name or "unknown"
                         result_content = interaction.llm_response or "[Empty Result]"
                         formatted_result = f"<tool_result>\n<{tool_name}>\n{result_content}\n</{tool_name}>\n</tool_result>\n"
                         agent_history_turns_messages.append(HumanMessage(content=formatted_result)) # Tool result comes back as Human

                prompt_history_turns = agent_history_turns_messages

                # --- Prepare all inputs for the agent_prompt_template ---
                prompt_inputs = {
                    "mode": env_details_dict.get("mode", "ACT MODE"),
                    "file_list": env_details_dict.get("file_list", "N/A"),
                    "running_terminals": env_details_dict.get("running_terminals", "N/A"),
                    "agent_history_rag": agent_history_rag_string,
                    "url_rag_context": url_rag_context_string,
                    "agent_history_turns": prompt_history_turns, # Pass constructed history list
                    "current_input": current_input_for_llm, # Pass current input (user query or tool result)
                }

                # --- Call the Agent LLM (Run sync Langchain call in executor thread with ELP0) ---
                llm_start_time = time.monotonic()
                logger.debug(f"Agent Turn {turn_count}: Calling LLM with Priority ELP0...")
                agent_llm_response = "" # Initialize response variable
                try:
                    # Pass ELP0 priority via the config dictionary
                    agent_llm_response = await asyncio.to_thread(
                        self.agent_chain.invoke, prompt_inputs, config={'priority': ELP0}
                    )
                    llm_duration = (time.monotonic() - llm_start_time) * 1000
                    logger.info(f"🧠 Agent LLM Turn {turn_count} finished ({llm_duration:.2f} ms).")
                    logger.trace(f"Agent LLM Turn {turn_count} Raw Response:\n{agent_llm_response}")

                except Exception as llm_err:
                     # Handle errors during the LLM invocation itself
                     logger.error(f"❌ Agent Task {initial_interaction_id}: LLM invocation failed on turn {turn_count}: {llm_err}")
                     logger.exception("LLM Invocation Traceback:")
                     # Mark task as failed and exit loop
                     if db and initial_interaction:
                         initial_interaction.llm_response = f"[Agent task {initial_interaction_id} failed on turn {turn_count} due to LLM error: {llm_err}]"
                         initial_interaction.classification = "task_failed_llm_error"
                         initial_interaction.execution_time_ms = (time.monotonic() - initial_interaction.timestamp.timestamp()) * 1000
                         db.commit()
                     break # Exit the while loop

                # --- *** Interruption Handling *** ---
                if isinstance(agent_llm_response, str) and interruption_error_marker in agent_llm_response:
                    logger.warning(f"🚦 Agent Task {initial_interaction_id}: Turn {turn_count} INTERRUPTED by higher priority task.")
                    # Log interruption to DB (associate with the original interaction)
                    if db and initial_interaction and initial_interaction.timestamp: # Check timestamp exists
                        initial_interaction.llm_response = f"[Agent task {initial_interaction_id} interrupted by ELP1 on turn {turn_count}]"
                        initial_interaction.classification = "task_failed_interrupted"
                        initial_interaction.execution_time_ms = (time.monotonic() - initial_interaction.timestamp.timestamp()) * 1000
                        try:
                            db.commit()
                            logger.info(f"Marked interaction {initial_interaction_id} as interrupted in DB.")
                        except Exception as db_err:
                            logger.error(f"Failed to update interaction {initial_interaction_id} after interruption: {db_err}")
                            db.rollback()
                    else:
                         logger.error(f"Could not mark interruption for task {initial_interaction_id}: DB or initial_interaction missing.")
                    # Exit the agent's task loop cleanly after interruption
                    break # Exit the while loop
                # --- *** End Interruption Handling *** ---

                # --- Log LLM response to DB (if not interrupted) ---
                # Note: The interaction ID here is the one returned by add_interaction,
                # NOT necessarily the initial_interaction_id for the whole task.
                add_interaction(
                    db, session_id=session_id, mode="agent", input_type="llm_response",
                    user_input=f"[Agent response turn {turn_count}]", # Contextual input description
                    llm_response=agent_llm_response, execution_time_ms=llm_duration
                )

                # --- Parse LLM response for tool request or final output types ---
                tool_name, parameters = self._parse_tool_request(agent_llm_response)

                if tool_name and tool_name not in ["attempt_completion", "plan_mode_respond", "ask_followup_question", "new_task", "load_mcp_documentation"]:
                    # Execute tool (async call)
                    logger.info(f"Agent Task {initial_interaction_id}: Executing tool '{tool_name}'.")
                    # _execute_agent_tool handles its own DB logging for request/result
                    tool_result_formatted = await self._execute_agent_tool(tool_name, parameters, db)

                    # Set the tool result as the input for the *next* LLM turn
                    current_input_for_llm = tool_result_formatted
                    # Continue the loop for the next turn
                    continue
                else:
                    # No executable tool requested - check for final outputs or just end
                    final_output_handled = False
                    final_agent_response = agent_llm_response # Default to the last LLM output

                    if tool_name == "attempt_completion":
                        logger.info(f"🏁 Agent Task {initial_interaction_id}: Used <attempt_completion>.")
                        final_output_handled = True
                        result_match = re.search(r'<result>(.*?)</result>', agent_llm_response, re.DOTALL)
                        if result_match: final_agent_response = result_match.group(1).strip()
                        if initial_interaction: initial_interaction.classification = "task_completed"

                    elif tool_name == "plan_mode_respond":
                        logger.info(f"🗣️ Agent Task {initial_interaction_id}: Used <plan_mode_respond>.")
                        final_output_handled = True
                        match = re.search(r'<response>(.*?)</response>', agent_llm_response, re.DOTALL)
                        if match: final_agent_response = match.group(1).strip()
                        if initial_interaction: initial_interaction.classification = "plan"

                    elif tool_name == "ask_followup_question":
                        logger.warning(f"❓ Agent Task {initial_interaction_id}: Used <ask_followup_question>.")
                        final_output_handled = True
                        if initial_interaction: initial_interaction.classification = "waiting_for_user"

                    elif tool_name == "new_task":
                        logger.info(f"✨ Agent Task {initial_interaction_id}: Used <new_task>.")
                        final_output_handled = True
                        if initial_interaction: initial_interaction.classification = "new_task_context"

                    elif tool_name == "load_mcp_documentation":
                         logger.info(f"📚 Agent Task {initial_interaction_id}: Used <load_mcp_documentation>.")
                         current_input_for_llm = "<tool_result><load_mcp_documentation>Placeholder: MCP Documentation Loaded.</load_mcp_documentation></tool_result>"
                         add_interaction(db, session_id=session_id, mode="agent", input_type="tool_result", user_input="[Result for load_mcp_documentation]", llm_response="Placeholder: MCP Documentation Loaded.", tool_name="load_mcp_documentation")
                         continue # Go to next LLM turn with this result

                    # --- Update initial interaction record with the final state ---
                    if initial_interaction and initial_interaction.timestamp: # Check timestamp exists
                        initial_interaction.llm_response = final_agent_response # Store meaningful final output
                        initial_interaction.execution_time_ms = (time.monotonic() - initial_interaction.timestamp.timestamp()) * 1000
                        db.commit() # Commit final state of original interaction
                    else:
                         logger.error(f"Could not update final state for task {initial_interaction_id}: initial_interaction record missing or timestamp invalid.")

                    if final_output_handled:
                        logger.success(f"✅ Agent background task finished (final output type '{tool_name or 'text'}') for ID: {initial_interaction_id}.")
                    else:
                        logger.warning(f"🧐 Agent Task {initial_interaction_id}: No tool or recognized final output. Ending task.")
                        logger.success(f"✅ Agent background task finished (unexpected output) for ID: {initial_interaction_id}.")

                    break # Exit the while loop

            # --- End of while loop ---
            if turn_count >= max_turns:
                 # Check if initial_interaction was loaded before accessing timestamp
                 if initial_interaction and initial_interaction.timestamp:
                     is_already_marked = (initial_interaction.llm_response or "").startswith("[Agent task reached max turns")
                     if not is_already_marked:
                         logger.warning(f"Agent Task {initial_interaction_id} reached max turns ({max_turns}).")
                         initial_interaction.llm_response = "[Agent task reached max turns without completing]"
                         initial_interaction.execution_time_ms = (time.monotonic() - initial_interaction.timestamp.timestamp()) * 1000
                         initial_interaction.classification = "task_failed_max_turns"
                         db.commit()
                 else:
                      logger.error(f"Agent Task {initial_interaction_id} reached max turns, but initial interaction record is missing.")

        except Exception as e:
            # Catch any unexpected errors during the agent's execution loop
            logger.error(f"❌❌ Error in Agent background task ID {initial_interaction_id}: {e}")
            logger.exception("Traceback:")
            # Attempt to log the error to the database associated with the initial interaction
            if db:
                try:
                    error_msg = f"Agent task {initial_interaction_id} failed: {e}"
                    # Use the already loaded initial_interaction if available
                    interaction_to_update = initial_interaction if initial_interaction else db.query(Interaction).filter(Interaction.id == initial_interaction_id).first()
                    if interaction_to_update:
                         existing_response = interaction_to_update.llm_response or ""
                         interaction_to_update.llm_response = (existing_response + f"\n---\nERROR: {e}")[:4000] # Append error, limit length
                         if interaction_to_update.timestamp: # Check timestamp exists
                             interaction_to_update.execution_time_ms = (time.monotonic() - interaction_to_update.timestamp.timestamp()) * 1000
                         interaction_to_update.classification = "task_failed" # Generic failure classification
                         db.commit()
                    else:
                         # If we can't even find the initial record, add a new error log
                         add_interaction(db, session_id=session_id, mode="agent", input_type='error',
                                         user_input=f"[Error Task {initial_interaction_id}]",
                                         llm_response=error_msg[:4000])
                except Exception as db_err:
                    logger.error(f"Failed to log Agent BG error to DB: {db_err}")
                    if db: db.rollback() # Rollback if logging the error failed

        finally:
            # Ensure the database session is closed for this background task
            if db:
                try:
                    db.close()
                except Exception as close_err:
                     logger.error(f"Error closing DB session for Agent Task {initial_interaction_id}: {close_err}")
            logger.warning(f"🧵 Agent background task thread finished for ID: {initial_interaction_id}")


    def reset(self, db: Session, session_id: str = None):
         """Resets the agent's internal state."""
         logger.warning(f"🔄 Resetting Agent state. (Session: {session_id})")
         self.conversation_history = []
         self.current_session_id = None
         logger.info("🧹 Agent internal history cleared.")
         try:
            add_interaction(db, session_id=session_id, mode="agent", input_type='system', user_input='Agent Session Reset Requested', llm_response='Agent state cleared.')
         except Exception as db_err:
             logger.error(f"Failed log agent reset: {db_err}")
         return "Agent state cleared."


# Global function to start agent task in background (async)
async def _start_agent_task(agent_instance: AmaryllisAgent, initial_interaction_id: int, user_input: str, session_id: str):
    """Schedules the Agent's background task."""
    logger.warning(f"▶️ Spawning Agent background task for Interaction ID: {initial_interaction_id}")
    loop = asyncio.get_event_loop()
    # Schedule the agent's main task function to run
    loop.create_task(agent_instance._run_task_in_background(initial_interaction_id, user_input, session_id))
    logger.warning(f"▶️ Agent background task scheduled.")

# Global agent instance placeholder - initialized in app.py
ai_agent: Optional[AmaryllisAgent] = None